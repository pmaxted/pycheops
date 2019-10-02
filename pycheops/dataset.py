# -*- coding: utf-8 -*-
#
#   pycheops - Tools for the analysis of data from the ESA CHEOPS mission
#
#   Copyright (C) 2018  Dr Pierre Maxted, Keele University
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
"""
Dataset
=======
 Object class for data access, data caching and data inspection tools

"""

from __future__ import (absolute_import, division, print_function,
                                unicode_literals)
import numpy as np
import tarfile
from zipfile import ZipFile
import re
from pathlib import Path
from .core import load_config
from astropy.io import fits
from astropy.table import Table, MaskedColumn
import matplotlib.pyplot as plt
from .instrument import transit_noise
from ftplib import FTP
from .models import TransitModel, FactorModel
from uncertainties import UFloat
from lmfit import Parameter, Parameters, minimize, Minimizer,fit_report
from scipy.interpolate import InterpolatedUnivariateSpline
from warnings import simplefilter,  catch_warnings
import matplotlib.pyplot as plt
from emcee import EnsembleSampler
import corner
import copy
from celerite import terms, GP
from sys import stdout 
from astropy.coordinates import SkyCoord
from lmfit.printfuncs import gformat

_dataset_re = re.compile(r'PR(\d{2})(\d{4})_TG(\d{4})(\d{2})')

# Utility function for model fitting
def _kwarg_to_Parameter(name, kwarg, min=min, max=max):
    if isinstance(kwarg, float):
        return Parameter(name=name, value=kwarg, vary=False)
    if isinstance(kwarg, int):
        return Parameter(name=name, value=float(kwarg), vary=False)
    if isinstance(kwarg, tuple):
        return Parameter(name=name, value=np.median(kwarg), 
                min=min(kwarg), max=max(kwarg))
    if isinstance(kwarg, UFloat):
        return Parameter(name=name, value=kwarg.n, user_data=kwarg)
    if isinstance(kwarg, Parameter):
        return kwarg
    raise ValueError('Unrecognised type for keyword argument {}'.
        format(name))

# Target functions for emcee
def _log_posterior_jitter(pos, model, time, flux, flux_err,  params, vn,
        return_fit):


    # Check for pos[i] within valid range has to be done here
    # because it gets set to the limiting value if out of range by the
    # assignment to a parameter with min/max defined.
    parcopy = params.copy()
    for i, p in enumerate(vn):
        v = pos[i]
        if (v < parcopy[p].min) or (v > parcopy[p].max):
            return -np.inf
        parcopy[p].value = v
    fit = model.eval(parcopy, t=time)
    if return_fit:
        return fit

    if False in np.isfinite(fit):
        return -np.inf

    # Also check parameter range here so we catch "derived" parameters
    # that are out of range.
    lnprior = 0
    for p in parcopy:
        v = parcopy[p].value
        if (v < parcopy[p].min) or (v > parcopy[p].max):
            return -np.inf
        if np.isnan(v):
            return -np.inf
        u = parcopy[p].user_data
        if isinstance(u, UFloat):
            lnprior += -0.5*((u.n - v)/u.s)**2
    if not np.isfinite(lnprior):
        return -np.inf

    jitter = np.exp(parcopy['log_sigma'].value)
    s2 =flux_err**2 + jitter**2
    lnlike = -0.5*(np.sum((flux-fit)**2/s2 + np.log(2*np.pi*s2)))
    return lnlike + lnprior

#----

def _log_posterior_SHOTerm(pos, model, time, flux, flux_err,  params, vn, gp, 
        return_fit):

    # Check for pos[i] within valid range has to be done here
    # because it gets set to the limiting value if out of range by the
    # assignment to a parameter with min/max defined.
    parcopy = params.copy()
    for i, p in enumerate(vn):
        v = pos[i]
        if (v < parcopy[p].min) or (v > parcopy[p].max):
            return -np.inf
        parcopy[p].value = v
    fit = model.eval(parcopy, t=time)
    if return_fit:
        return fit

    if False in np.isfinite(fit):
        return -np.inf
    
    # Also check parameter range here so we catch "derived" parameters
    # that are out of range.
    lnprior = 0
    for p in parcopy:
        v = parcopy[p].value
        if (v < parcopy[p].min) or (v > parcopy[p].max):
            return -np.inf
        if np.isnan(v):
            return -np.inf
        u = parcopy[p].user_data
        if isinstance(u, UFloat):
            lnprior += -0.5*((u.n - v)/u.s)**2
    if not np.isfinite(lnprior):
        return -np.inf

    resid = flux-fit
    gp.set_parameter('kernel:terms[0]:log_S0',
            parcopy['log_S0'].value)
    gp.set_parameter('kernel:terms[0]:log_Q',
            parcopy['log_Q'].value)
    gp.set_parameter('kernel:terms[0]:log_omega0',
            parcopy['log_omega0'].value)
    gp.set_parameter('kernel:terms[1]:log_sigma',
            parcopy['log_sigma'].value)
    return gp.log_likelihood(resid) + lnprior
    
#---------------

class Dataset(object):
    """
    CHEOPS Dataset object

    """

    def __init__(self, dataset_id=None, force_download=False, 
            configFile=None, target=None, verbose=True):

        if dataset_id is None:
            return None
        self.dataset_id = dataset_id

        m = _dataset_re.search(dataset_id)
        if m is None:
            raise ValueError('Invalid dataset_id')
        l = [int(i) for i in m.groups()]
        self.progtype, self.prog_id, self.req_id, self.visitctr = l

        config = load_config(configFile)
        _cache_path = config['DEFAULT']['data_cache_path']
        tgzPath = Path(_cache_path,dataset_id).with_suffix('.tgz')
        self.tgzfile = str(tgzPath)

        if tgzPath.is_file():
            if verbose: print('Found archive tgzfile',self.tgzfile)
        else:
            ftp=FTP(config['DEFAULT']['archive_url'])
            _ = ftp.login(user=config['DEFAULT']['archive_username'],
                    passwd=config['DEFAULT']['archive_password'])
            ftp.cwd(wd)
            if verbose: print('Downloading dataset from',
                    config['DEFAULT']['archive_url'])
            cmd = 'RETR {}.tgz'.format(dataset_id)
            ftp.retrbinary(cmd, open(self.tgzfile, 'wb').write)
            ftp.quit()

        lisPath = Path(_cache_path,dataset_id).with_suffix('.lis')
        if lisPath.is_file():
            self.list = [line.rstrip('\n') for line in open(lisPath)]
        else:
            if verbose: print('Creating dataset file list')
            tar = tarfile.open(self.tgzfile)
            self.list = tar.getnames()
            tar.close()
            with open(str(lisPath), 'w') as fh:  
                fh.writelines("%s\n" % l for l in self.list)

        # Extract OPTIMAL light curve data file from .tgz file so we can
        # access the FITS file header information
        aperture='OPTIMAL'
        lcFile = "{}-{}.fits".format(self.dataset_id,aperture)
        lcPath = Path(self.tgzfile).parent / lcFile
        if lcPath.is_file():
            with fits.open(lcPath) as hdul:
                hdr = hdul[1].header
        else:
            tar = tarfile.open(self.tgzfile)
            r=re.compile('(.*_SCI_COR_Lightcurve-{}_.*.fits)'.format(aperture))
            datafile = list(filter(r.match, self.list))
            if len(datafile) == 0:
                raise Exception('Dataset does not contain light curve data.')
            if len(datafile) > 1:
                raise Exception('Multiple light curve files in datset')
            with tar.extractfile(datafile[0]) as fd:
                hdul = fits.open(fd)
                table = Table.read(hdul[1])
                hdr = hdul[1].header
                hdul.writeto(lcPath)
            tar.close()
        self.pi_name = hdr['PI_NAME']
        self.obsid = hdr['OBSID']
        if target is None:
            targetPath = Path(_cache_path,dataset_id).with_suffix('.target')
            if targetPath.is_file():
                with open(str(targetPath), 'r') as fh:
                    self.target = fh.readline().rstrip('\n')
            else:
               self.target = hdr['TARGNAME']
        else:
            self.target = target
        coords = SkyCoord(hdr['RA_TARG'],hdr['DEC_TARG'],unit='degree,degree')
        self.ra = coords.ra.to_string(precision=2,unit='hour',sep=':',pad=True)
        self.dec = coords.dec.to_string(precision=1,sep=':',unit='degree',
                alwayssign=True,pad=True)
        if verbose:
            print(' PI name     : {}'.format(self.pi_name))
            print(' OBS ID      : {}'.format(self.obsid))
            print(' Target      : {}'.format(self.target))
            print(' Coordinates : {} {}'.format(self.ra, self.dec))

    @classmethod
    def from_simulation(self, job,  target=None, configFile=None, 
            verbose=True):
        ftp=FTP('obsftp.unige.ch')
        _ = ftp.login()
        wd = "pub/cheops/simulated_data/CHEOPSim_job{}".format(job)
        ftp.cwd(wd)
        filelist = [fl[0] for fl in ftp.mlsd()]
        _re = re.compile(r'CH_(PR\d{6}_TG\d{6}).zip')
        zipfiles = list(filter(_re.match, filelist))
        if len(zipfiles) > 1:
            raise ValueError('More than one dataset in ftp directory')
        if len(zipfiles) == 0:
            raise ValueError('No zip files for datasets in ftp directory')
        zipfile = zipfiles[0]
        config = load_config(configFile)
        _cache_path = config['DEFAULT']['data_cache_path']
        zipPath = Path(_cache_path,zipfile)
        if zipPath.is_file():
            if verbose: print('{} already downloaded'.format(str(zipPath)))
        else:
            cmd = 'RETR {}'.format(zipfile)
            if verbose: print('Downloading {} ...',format(zipfile))
            ftp.retrbinary(cmd, open(str(zipPath), 'wb').write)
            ftp.quit()
        
        dataset_id = zipfile[3:-4]
        m = _dataset_re.search(dataset_id)
        l = [int(i) for i in m.groups()]

        tgzPath = Path(_cache_path,dataset_id).with_suffix('.tgz')
        tgzfile = str(tgzPath)

        zpf = ZipFile(str(zipPath), mode='r')
        ziplist = zpf.namelist()

        _re = re.compile('(CH_.*SCI_RAW_Imagette_.*.fits)')
        imgfiles = list(filter(_re.match, ziplist))
        if len(imgfiles) > 1:
            raise ValueError('More than one imagette file in zip file')
        if len(imgfiles) == 0:
            raise ValueError('No imagette file in zip file')
        imgfile = imgfiles[0]

        _re = re.compile('(CH_.*_SCI_COR_Lightcurve-.*fits)')
        with tarfile.open(tgzfile, mode='w:gz') as tgz:
            tarPath = Path('visit')/Path(dataset_id)/Path(imgfile).name 
            tarinfo = tarfile.TarInfo(name=str(tarPath))
            zipinfo = zpf.getinfo(imgfile)
            tarinfo.size = zipinfo.file_size
            zf = zpf.open(imgfile)
            if verbose: print("Writing Imagette data to .tgz file...")
            tgz.addfile(tarinfo=tarinfo, fileobj=zf)
            zf.close()
            if verbose: print("Writing Lightcurve data to .tgz file...")
            for lcfile in list(filter(_re.match, ziplist)):
                tarPath = Path('visit')/Path(dataset_id)/Path(lcfile).name
                tarinfo = tarfile.TarInfo(name=str(tarPath))
                zipinfo = zpf.getinfo(lcfile)
                tarinfo.size = zipinfo.file_size
                zf = zpf.open(lcfile)
                tgz.addfile(tarinfo=tarinfo, fileobj=zf)
                zf.close()
                if verbose: print ('.. {} - done'.format(Path(lcfile).name))
        zpf.close()

        if target is not None:
            targetPath = Path(_cache_path,dataset_id).with_suffix('.target')
            with open(str(targetPath), 'w') as fh:  
                fh.writelines("{}\n".format(target))

        return self(dataset_id=dataset_id, target=target, verbose=verbose)
        
    def get_imagettes(self, verbose=True):
        imFile = "{}-Imagette.fits".format(self.dataset_id)
        imPath = Path(self.tgzfile).parent / imFile
        if imPath.is_file():
            with fits.open(imPath) as hdul:
                cube = hdul[1].data
                hdr = hdul[1].header
                meta = Table.read(hdul[2])
            if verbose: print ('Imagette data loaded from ',imPath)
        else:
            if verbose: print ('Extracting imagette data from ',self.tgzfile)
            r=re.compile('(.*SCI_RAW_Imagette.*.fits)' )
            datafile = list(filter(r.match, self.list))
            if len(datafile) == 0:
                raise Exception('Dataset does not contains imagette data.')
            if len(datafile) > 1:
                raise Exception('Multiple imagette data files in dataset')
            tar = tarfile.open(self.tgzfile)
            with tar.extractfile(datafile[0]) as fd:
                hdul = fits.open(fd)
                cube = hdul[1].data
                hdr = hdul[1].header
                meta = Table.read(hdul[2])
                hdul.writeto(imPath)
            tar.close()
            if verbose: print('Saved imagette data to ',imPath)

        self.imagettes = (cube, hdr, meta)
        self.imagettes = {'data':cube, 'header':hdr, 'meta':meta}

        return cube

    def get_lightcurve(self, aperture=None,
            returnTable=False, reject_highpoints=True, verbose=True):

        if aperture not in ('OPTIMAL','RSUP','RINF','DEFAULT'):
            raise ValueError('Invalid/missing aperture name')

        lcFile = "{}-{}.fits".format(self.dataset_id,aperture)
        lcPath = Path(self.tgzfile).parent / lcFile
        if lcPath.is_file(): 
            with fits.open(lcPath) as hdul:
                table = Table.read(hdul[1])
                hdr = hdul[1].header
            if verbose: print ('Light curve data loaded from ',lcPath)
        else:
            if verbose: print ('Extracting light curve from ',self.tgzfile)
            tar = tarfile.open(self.tgzfile)
            r=re.compile('(.*_SCI_COR_Lightcurve-{}_.*.fits)'.format(aperture))
            datafile = list(filter(r.match, self.list))
            if len(datafile) == 0:
                raise Exception('Dataset does not contain light curve data.')
            if len(datafile) > 1:
                raise Exception('Multiple light curve files in datset')
            with tar.extractfile(datafile[0]) as fd:
                hdul = fits.open(fd)
                table = Table.read(hdul[1])
                hdr = hdul[1].header
                hdul.writeto(lcPath)
            if verbose: print('Saved lc data to ',lcPath)

        ok = (table['EVENT'] == 0) | (table['EVENT'] == 100)
        bjd = np.array(table['BJD_TIME'][ok])
        bjd_ref = np.int(bjd[0])
        time = bjd-bjd_ref
        flux = np.array(table['FLUX'][ok])
        flux_err = np.array(table['FLUXERR'][ok])
        fluxmed = np.nanmedian(flux)
        xoff = np.array(table['CENTROID_X'][ok]- table['LOCATION_X'][ok])
        yoff = np.array(table['CENTROID_Y'][ok]- table['LOCATION_Y'][ok])
        roll_angle = np.array(table['ROLL_ANGLE'][ok])
        if reject_highpoints:
            C_cut = (2*np.nanmedian(flux)-np.min(flux))
            ok  = (flux < C_cut).nonzero()
            time = time[ok]
            flux = flux[ok]
            flux_err = flux_err[ok]
            N_cut = len(bjd) - len(time)
        if verbose:
            if reject_highpoints:
                print('C_cut = {:0.0f}'.format(C_cut))
                print('N(C > C_cut) = {}'.format(N_cut))
            print('Mean counts = {:0.1f}'.format(flux.mean()))
            print('Median counts = {:0.1f}'.format(fluxmed))
            print('RMS counts = {:0.1f} [{:0.0f} ppm]'.format(np.std(flux), 
                1e6*np.std(flux)/fluxmed))
            print('Median standard error = {:0.1f} [{:0.0f} ppm]'.format(
                fluxmed, 1e6*np.nanmedian(flux_err)/fluxmed))

        flux = flux/fluxmed
        flux_err = flux_err/fluxmed
        self.lc = {'time':time, 'flux':flux, 'flux_err':flux_err,
                'bjd_ref':bjd_ref, 'table':table, 'header':hdr,
                'xoff':xoff, 'yoff':yoff, 'roll_angle':roll_angle, 
                'aperture':aperture}

        if returnTable:
            return table
        else:
            return time, flux, flux_err

 #----------------------------------------------------------------------------
 # Eclipse and transit fitting

    def lmfit_transit(self, 
            T_0=None, P=None, D=None, W=None, S=None, f_c=None, f_s=None,
            h_1=None, h_2=None,
            c=None, dfdx=None, dfdy=None, d2fdx2=None, d2fdy2=None,
            dfdsinphi=None, dfdcosphi=None, dfdt=None, d2fdt2=None, 
            logrhoprior=None):

        def _chisq_prior(params, *args):
            r =  (flux - model.eval(params, t=time))/flux_err
            for p in params:
                u = params[p].user_data
                if isinstance(u, UFloat):
                    r = np.append(r, (u.n - params[p].value)/u.s)
            return r

        try:
            time = self.lc['time']
            flux = self.lc['flux']
            flux_err = self.lc['flux_err']
            xoff = self.lc['xoff']
            yoff = self.lc['xoff']
            phi = self.lc['roll_angle']*np.pi/180
        except AttributeError:
            raise AttributeError("Use get_lightcurve() to load data first.")

        params = Parameters()
        if T_0 is None:
            params.add(name='T_0', value=np.nanmedian(time),
                    min=min(time),max=max(time))
        else:
            params['T_0'] = _kwarg_to_Parameter('T_0', T_0)
        if P is None:
            params.add(name='P', value=1, vary=False)
        else:
            params['P'] = _kwarg_to_Parameter('P', P)
        _P = params['P'].value
        if D is None:
            params.add(name='D', value=1-min(flux), min=0,max=0.5)
        else:
            params['D'] = _kwarg_to_Parameter('D', D)
        k = np.sqrt(params['D'].value)
        if W is None:
            params.add(name='W', value=np.ptp(time)/2/_P,
                    min=np.ptp(time)/len(time)/_P, max=np.ptp(time)/_P) 
        else:
            params['W'] = _kwarg_to_Parameter('W', W)
        if S is None:
            params.add(name='S', value=((1-k)**2-0.25)/((1+k)**2-0.25),
                    min=0, max=1)
        else:
            params['S'] = _kwarg_to_Parameter('S', S)
        if f_c is None:
            params.add(name='f_c', value=0, vary=False)
        else:
            params['f_c'] = _kwarg_to_Parameter('f_c', f_c)
        if f_s is None:
            params.add(name='f_s', value=0, vary=False)
        else:
            params['f_s'] = _kwarg_to_Parameter('f_s', f_s)
        if h_1 is None:
            params.add(name='h_1', value=0.7224, vary=False)
        else:
            params['h_1'] = _kwarg_to_Parameter('h_1', h_1, min=0, max=1)
        if h_2 is None:
            params.add(name='h_2', value=0.6713, vary=False)
        else:
            params['h_2'] = _kwarg_to_Parameter('h_2', h_2, min=0, max=1)
        if c is None:
            params.add(name='c', value=1, min=min(flux)/2,max=2*max(flux))
        else:
            params['c'] = _kwarg_to_Parameter('c', c)
        if dfdx is not None:
            params['dfdx'] = _kwarg_to_Parameter('dfdx', dfdx)
        if dfdy is not None:
            params['dfdy'] = _kwarg_to_Parameter('dfdy', dfdy)
        if d2fdx2 is not None:
            params['d2fdx2'] = _kwarg_to_Parameter('d2fdx2', dfdx)
        if d2fdy2 is not None:
            params['d2fdy2'] = _kwarg_to_Parameter('d2fdy2', dfdy)
        if dfdt is not None:
            params['dfdt'] = _kwarg_to_Parameter('dfdt', dfdt)
        if d2fdt2 is not None:
            params['d2fdt2'] = _kwarg_to_Parameter('d2fdt2', dfdt)
        if dfdsinphi is not None:
            params['dfdsinphi'] = _kwarg_to_Parameter('dfdsinphi', dfdsinphi)
        if dfdcosphi is not None:
            params['dfdcosphi'] = _kwarg_to_Parameter('dfdcosphi', dfdcosphi)


        params.add('k',expr='sqrt(D)',min=0,max=1)
        params.add('bsq', expr='((1-k)**2-S*(1+k)**2)/(1-S)', min=0, max=1)
        params.add('b', expr='sqrt(((1-k)**2-S*(1+k)**2)/(1-S))', min=0, max=1)
        params.add('aR',expr='sqrt((1+k)**2-b**2)/W/pi',min=1)
        # Avoid use of aR in this expr for logrho - breaks error propogation.
        expr = 'log10(4.3275e-4*((1+k)**2-b**2)**1.5/W**3/P**2)'
        params.add('logrho',expr=expr,min=-6,max=3)
        params['logrho'].user_data=logrhoprior
        params.add('e',min=0,max=1,expr='f_c**2 + f_s**2')
        params.add('q_1',min=0,max=1,expr='(1-h_2)**2')
        params.add('q_2',min=0,max=1,expr='(h_1-h_2)/(1-h_2)')

        model = TransitModel()*FactorModel(
            dx = InterpolatedUnivariateSpline(time, xoff),
            dy = InterpolatedUnivariateSpline(time, yoff),
            sinphi = InterpolatedUnivariateSpline(time,np.sin(phi)),
            cosphi = InterpolatedUnivariateSpline(time,np.cos(phi)) )

        result = minimize(_chisq_prior, params,nan_policy='propagate',
                args=(model, time, flux, flux_err))
        self.model = model
        self.lmfit = result
        return result

    # ----------------------------------------------------------------

    def lmfit_report(self):
        report = fit_report(self.lmfit)
        noPriors = True
        params = self.lmfit.params
        parnames = list(params.keys())
        namelen = max([len(n) for n in parnames])
        for p in params:
            u = params[p].user_data
            if isinstance(u, UFloat):
                if noPriors:
                    report+="\n[[Priors]]\n"
                    noPriors = False
                report += "    %s:%s" % (p, ' '*(namelen-len(p)))
                report += '%s +/-%s\n' % (gformat(u.n), gformat(u.s))
        return(report)

    # ----------------------------------------------------------------

    def emcee_transit(self, params=None,
            steps=64, nwalkers=64, burn=64, thin=4, 
            add_shoterm=False, init_scale=1e-3, progress=True):

        try:
            time = np.array(self.lc['time'])
            flux = np.array(self.lc['flux'])
            flux_err = np.array(self.lc['flux_err'])
        except AttributeError:
            raise AttributeError("Use get_lightcurve() to load data first.")

        try:
            model = self.model
        except AttributeError:
            raise AttributeError("Use lmfit_transit() to get model first.")

        # Make a copy of the lmfit Minimizer result as a template for the
        # output of this method
        result = copy.copy(self.lmfit)
        result.method ='emcee'
        # Remove components on result not relevant for emcee
        result.status = None
        result.success = None
        result.message = None
        result.ier = None
        result.lmdif_message = None

        if params is None:
            params = copy.copy(self.lmfit.params)
            if add_shoterm:
                # Minimum here is about 0.1ppm 
                params.add('log_S0', value=-11,  min=-16, max=-1)
                # For time in days, and the default value of Q=1/sqrt(2),
                # log_omega0=12  is a correlation length 
                # of about 0.5s and -2.3 is about 10 days.
                params.add('log_omega0', value=6, min=-2.3, max=12)
                params.add('log_Q', value=np.log(1/np.sqrt(2)), vary=False)

        if not 'log_sigma' in params:
            params.add('log_sigma', value=-10, min=-15,max=0)
            params.add('sigma_w',expr='exp(log_sigma)*1e6')
            params['log_sigma'].stderr = 1

        vv = []
        vs = []
        vn = []
        for p in params:
            if params[p].vary:
                vn.append(p)
                vv.append(params[p].value)
                if params[p].stderr is None:
                    vs.append(0.1*(params[p].max-params[p].min))
                else:
                    vs.append(params[p].stderr)

        result.var_names = vn
        result.init_vals = vv
        result.init_values = dict()
        for n,v in zip(vn, vv):
            result.init_values[n] = v

        vv = np.array(vv)
        vs = np.array(vs)

        args=(model, time, flux, flux_err,  params, vn)
        p = list(params.keys())
        if 'log_S0' in p and 'log_omega0' in p and 'log_Q' in p :
            kernel = terms.SHOTerm(log_S0=params['log_S0'].value,
                    log_Q=params['log_Q'].value,
                    log_omega0=params['log_omega0'].value)
            kernel += terms.JitterTerm(log_sigma=params['log_sigma'].value)

            gp = GP(kernel, mean=0, fit_mean=False)
            gp.compute(time, flux_err)
            log_posterior_func = _log_posterior_SHOTerm
            args += (gp,)
        else:
            log_posterior_func = _log_posterior_jitter
            gp = None
        return_fit = False
        args += (return_fit, )
    
        # Initialize sampler positions ensuring all walkers produce valid
        # function values.
        pos = []
        n_varys = len(vv)
        for i in range(nwalkers):
            params_tmp = params.copy()
            lnlike_i = -np.inf
            while lnlike_i == -np.inf:
                pos_i = vv + vs*np.random.randn(n_varys)*init_scale
                lnlike_i = log_posterior_func(pos_i, *args)

            pos.append(pos_i)

        sampler = EnsembleSampler(nwalkers, n_varys, log_posterior_func,
                args=args)
        if progress:
            print('Running burn-in ..')
            stdout.flush()
        pos, _, _ = sampler.run_mcmc(pos, burn, store=False, 
                skip_initial_state_check=True, progress=progress)
        sampler.reset()
        if progress:
            print('Running sampler ..')
            stdout.flush()
        state = sampler.run_mcmc(pos, steps, thin_by=thin,
                skip_initial_state_check=True, progress=progress)

        flatchain = sampler.get_chain(flat=True).reshape((-1, len(vn)))
        pos_i = flatchain[np.argmax(sampler.get_log_prob()),:]
        return_fit = True
        if gp is None:
            fit = _log_posterior_jitter(pos_i, model, time, flux, flux_err,
                    params, vn, return_fit)
        else:
            fit = _log_posterior_SHOTerm(pos_i, model, time, flux, flux_err,
                    params, vn, gp, return_fit)

        result.bestfit = fit
        result.chain = flatchain
        parbest = params.copy()
        quantiles = np.percentile(flatchain, [15.87, 50, 84.13], axis=0)
        for i, n in enumerate(vn):
            std_l, median, std_u = quantiles[:, i]
            params[n].value = median
            params[n].stderr = 0.5 * (std_u - std_l)
            params[n].correl = {}
            parbest[n].value = pos_i[i]
            parbest[n].stderr = None
            parbest[n].correl = None
        result.params = params
        result.params_best = parbest
        corrcoefs = np.corrcoef(flatchain.T)
        for i, n in enumerate(vn):
            for j, n2 in enumerate(vn):
                if i != j:
                    result.params[n].correl[n2] = corrcoefs[i, j]
        result.lnprob = np.copy(sampler.get_log_prob())
        result.errorbars = True
        result.nvarys = n_varys
        result.nfev = nwalkers*steps*thin
        result.ndata = len(time)
        result.nfree = len(time) - n_varys
        result.chisqr = np.sum((flux-fit)**2/flux_err**2)
        result.redchi = result.chisqr/(len(time) - n_varys)
        loglmax = np.max(sampler.get_log_prob())
        result.aic = 2*n_varys - 2*loglmax
        result.bic = np.log(len(time))*n_varys - 2*loglmax
        result.covar = np.cov(flatchain.T)
        self.emcee = result
        self.sampler = sampler
        self.gp = gp
        return result

    # ----------------------------------------------------------------

    def emcee_report(self):
        report = fit_report(self.emcee)
        noPriors = True
        params = self.emcee.params
        parnames = list(params.keys())
        namelen = max([len(n) for n in parnames])
        for p in params:
            u = params[p].user_data
            if isinstance(u, UFloat):
                if noPriors:
                    report+="\n[[Priors]]\n"
                    noPriors = False
                report += "    %s:%s" % (p, ' '*(namelen-len(p)))
                report += '%s +/-%s\n' % (gformat(u.n), gformat(u.s))
        return(report)

    # ----------------------------------------------------------------

    def corner_plot(self, plotkeys=['T_0', 'D', 'W', 'S'], 
            show_priors=True):

        params = self.emcee.params
        chain = self.sampler.get_chain(flat=True)
        labels = []
        xs = []

        varkeys = []
        for key in params:
            if params[key].vary:
                varkeys.append(key)

        for key in plotkeys:
            if key in varkeys:
                xs.append(chain[:,varkeys.index(key)])
                if key == 'T_0':
                    labels.append(r'T$_0-{}$'.format(self.lc['bjd_ref']))
                elif key == 'dfdx':
                    labels.append(r'$df/dx$')
                elif key == 'd2fdx2':
                    labels.append(r'$d^2f/dx^2$')
                elif key == 'dfdy':
                    labels.append(r'$df/dy$')
                elif key == 'd2fdy2':
                    labels.append(r'$d^2f/dy^2$')
                elif key == 'dfdt':
                    labels.append(r'$df/dt$')
                elif key == 'd2fdt2':
                    labels.append(r'$d^2f/dt^2$')
                elif key == 'dfdsinphi':
                    labels.append(r'$df/d\sin\phi$')
                elif key == 'dfdcosphi':
                    labels.append(r'$df/d\cos\phi$')
                elif key == 'log_sigma':
                    labels.append(r'$\log\sigma$')
                elif key == 'log_omega0':
                    labels.append(r'$\log\omega_0$')
                elif key == 'log_S0':
                    labels.append(r'$\log{\rm S}_0$')
                elif key == 'log_Q':
                    labels.append(r'$\log{\rm Q}$')
                else:
                    labels.append(key)

            if key == 'sigma_w' and params['log_sigma'].vary:
                xs.append(np.exp(self.result.chain[:,-1])*1e6)
                labels.append(r'$\sigma_w$ [ppm]')

            if 'D' in varkeys:
                k = np.sqrt(chain[:,varkeys.index('D')])
            else:
                k = np.sqrt(params['D'].value)

            if key == 'k' and 'D' in varkeys:
                xs.append(k)
                labels.append(r'k')

            if 'S' in varkeys:
                S = chain[:,varkeys.index('S')]
            else:
                S = params['S'].value

            b = np.sqrt(((1-k)**2-S*(1+k)**2)/(1-S))

            if key == 'b':
                xs.append(b)
                labels.append(r'b')

            if 'W' in varkeys:
                W = chain[:,varkeys.index('W')]
            else:
                W = params['W'].value

            aR = np.sqrt((1+k)**2-b**2)/W/np.pi

            if key == 'aR':
                xs.append(aR)
                labels.append(r'aR')

            if 'P' in varkeys:
                P = chain[:,varkeys.index('P')]
            else:
                P = params['P'].value

            if key == 'logrho':
                logrho = np.log10(4.3275e-4*((1+k)**2-b**2)**1.5/W**3/P**2)
                xs.append(logrho)
                labels.append(r'$\log\rho_{\star}$')


        xs = np.array(xs).T
        figure = corner.corner(xs, labels=labels)

        nax = len(labels)
        axes = np.array(figure.axes).reshape((nax, nax))
        for i, key in enumerate(plotkeys):
            u = params[key].user_data
            if isinstance(u, UFloat):
                ax = axes[i, i]
                ax.axvline(u.n - u.s, color="g", linestyle='--')
                ax.axvline(u.n + u.s, color="g", linestyle='--')

    # ------------------------------------------------------------
    
    def plot_lmfit(self, figsize=(6,4), fontsize=11, title=None):
        try:
            time = np.array(self.lc['time'])
            flux = np.array(self.lc['flux'])
            flux_err = np.array(self.lc['flux_err'])
        except AttributeError:
            raise AttributeError("Use get_lightcurve() to load data first.")
        try:
            model = self.model
        except AttributeError:
            raise AttributeError("Use lmfit_transit() to generate model first.")
        try:
            params = self.lmfit.params
        except AttributeError:
            raise AttributeError(
                    "Use lmfit_transit() to get best-fit parameters first.")

        fit = self.model.eval(params, t=time)
        res = flux - fit
        plt.rc('font', size=fontsize)    
        fig,ax=plt.subplots(nrows=2,sharex=True, figsize=figsize,
                gridspec_kw={'height_ratios':[2,1]})

        ax[0].errorbar(time,flux,yerr=flux_err,fmt='bo',ms=3,zorder=0)
        tmin = np.round(np.min(time)-0.05*np.ptp(time),2)
        tmax = np.round(np.max(time)+0.05*np.ptp(time),2)
        tp = np.linspace(tmin, tmax, 10*len(time))
        ax[0].plot(tp,self.model.eval(params,t=tp),c='orange',zorder=1)
        ax[0].set_xlim(tmin, tmax)
        ymin = np.min(flux-flux_err)-0.05*np.ptp(flux)
        ymax = np.max(flux+flux_err)+0.05*np.ptp(flux)
        ax[0].set_ylim(ymin,ymax)
        ax[0].set_title(title)
        ax[0].set_ylabel('Flux')
        ax[1].errorbar(time,res,yerr=flux_err,fmt='bo',ms=3,zorder=0)
        ax[1].plot([tmin,tmax],[0,0],ls=':',c='orange',zorder=1)
        ax[1].set_xlabel('BJD-{}'.format(self.lc['bjd_ref']))
        ax[1].set_ylabel('Residual')
        ylim = np.max(np.abs(res-flux_err)+0.05*np.ptp(res))
        ax[1].set_ylim(-ylim,ylim)
        fig.tight_layout()
        return fig
        
    # ------------------------------------------------------------
    
    def plot_emcee(self, title=None, nsamples=32, figsize=(6,4), fontsize=11):
        try:
            time = np.array(self.lc['time'])
            flux = np.array(self.lc['flux'])
            flux_err = np.array(self.lc['flux_err'])
        except AttributeError:
            raise AttributeError("Use get_lightcurve() to load data first.")
        try:
            model = self.model
        except AttributeError:
            raise AttributeError("Use lmfit_transit() to get a model first.")
        try:
            sampler = self.sampler
        except AttributeError:
            raise AttributeError(
                    "Use emcee_transit() or emcee_eclipse() first.")

        res = flux - self.emcee.bestfit
        plt.rc('font', size=fontsize)    
        fig,ax=plt.subplots(nrows=2,sharex=True, figsize=figsize,
                gridspec_kw={'height_ratios':[2,1]})

        tmin = np.round(np.min(time)-0.05*np.ptp(time),2)
        tmax = np.round(np.max(time)+0.05*np.ptp(time),2)
        tp = np.linspace(tmin, tmax, 10*len(time))
        parbest = self.emcee.params_best
        nchain = self.emcee.chain.shape[0]
        partmp = parbest.copy()
        ax[0].errorbar(time,flux,yerr=flux_err,fmt='bo',ms=3,zorder=0)
        if self.gp is None:
            ax[0].plot(tp,self.model.eval(parbest,t=tp), c='orange',zorder=1)
            for i in np.linspace(0,nchain,nsamples,endpoint=False,dtype=np.int):
                for j, n in enumerate(self.emcee.var_names):
                    partmp[n].value = self.emcee.chain[i,j]
                ax[0].plot(tp,self.model.eval(partmp,t=tp), 
                        c='orange',zorder=1,alpha=0.1)
        else:
            self.gp.set_parameter('kernel:terms[0]:log_S0',
                    parbest['log_S0'].value)
            self.gp.set_parameter('kernel:terms[0]:log_Q',
                    parbest['log_Q'].value)
            self.gp.set_parameter('kernel:terms[0]:log_omega0',
                    parbest['log_omega0'].value)
            self.gp.set_parameter('kernel:terms[1]:log_sigma',
                    parbest['log_sigma'].value)
            mu0, var = self.gp.predict(res, tp, return_var=True)
            pp = mu0+self.model.eval(parbest,t=tp)
            ax[0].plot(tp,pp,c='orange',zorder=1)
            for i in np.linspace(0,nchain,nsamples,endpoint=False,dtype=np.int):
                for j, n in enumerate(self.emcee.var_names):
                    partmp[n].value = self.emcee.chain[i,j]
                ff = self.model.eval(partmp, t=time)
                rr = flux - ff
                self.gp.set_parameter('kernel:terms[0]:log_S0',
                        partmp['log_S0'].value)
                self.gp.set_parameter('kernel:terms[0]:log_Q',
                        partmp['log_Q'].value)
                self.gp.set_parameter('kernel:terms[0]:log_omega0',
                        partmp['log_omega0'].value)
                self.gp.set_parameter('kernel:terms[1]:log_sigma',
                        partmp['log_sigma'].value)
                mu = self.gp.predict(rr, tp, return_var=False, return_cov=False)
                pp = mu + self.model.eval(partmp, t=tp)
                ax[0].plot(tp, pp, c='orange',zorder=1,alpha=0.1)
                
        ymin = np.min(flux-flux_err)-0.05*np.ptp(flux)
        ymax = np.max(flux+flux_err)+0.05*np.ptp(flux)
        ax[0].set_xlim(tmin, tmax)
        ax[0].set_ylim(ymin,ymax)
        ax[0].set_title(title)
        ax[0].set_ylabel('Flux')
        ax[1].errorbar(time,res,yerr=flux_err,fmt='bo',ms=3,zorder=0)
        if self.gp is not None:
            ax[1].plot(tp,mu0,c='orange', zorder=1)
        ax[1].plot([tmin,tmax],[0,0],ls=':',c='orange', zorder=1)
        ax[1].set_xlabel('BJD-{}'.format(self.lc['bjd_ref']))
        ax[1].set_ylabel('Residual')
        ylim = np.max(np.abs(res-flux_err)+0.05*np.ptp(res))
        ax[1].set_ylim(-ylim,ylim)
        fig.tight_layout()
        return fig
        
 #----------------------------------------------------------------------------
 # Data display and diagnostics

 # Eclipse and transit fitting
    def transit_noise_plot(self, width=3, steps=500,
            fname=None, figsize=(6,4), fontsize=11, verbose=True):

        try:
            time = np.array(self.lc['time'])
            flux = np.array(self.lc['flux'])
            flux_err = np.array(self.lc['flux_err'])
        except AttributeError:
            raise AttributeError("Use get_lightcurve() to load data first.")
        T = np.linspace(np.min(time)+width/48,np.max(time)-width/48 , steps)

        Nsc = np.zeros_like(T)
        Fsc = np.zeros_like(T)
        Nmn = np.zeros_like(T)

        for i,_t in enumerate(T):
            _n,_f = transit_noise(time, flux, flux_err, T_0=_t,
                              width=width, method='scaled')
            if np.isfinite(_n):
                Nsc[i] = _n
                Fsc[i] = _f
            _n = transit_noise(time, flux, flux_err, T_0=_t,
                           width=width, method='minerr')
            if np.isfinite(_n):
                Nmn[i] = _n

        msk = (Nsc > 0) 
        Tsc = T[msk]
        Nsc = Nsc[msk]
        Fsc = Fsc[msk]
        msk = (Nmn > 0) 
        Tmn = T[msk]
        Nmn = Nmn[msk]

        if verbose:
            print('Scaled noise method')
            print('Mean noise = {:0.1f} ppm'.format(Nsc.mean()))
            print('Min. noise = {:0.1f} ppm'.format(Nsc.min()))
            print('Max. noise = {:0.1f} ppm'.format(Nsc.max()))
            print('Mean noise scaling factor = {:0.3f} '.format(Fsc.mean()))
            print('Min. noise scaling factor = {:0.3f} '.format(Fsc.min()))
            print('Max. noise scaling factor = {:0.3f} '.format(Fsc.max()))

            print('\nMinimum error noise method')
            print('Mean noise = {:0.1f} ppm'.format(Nmn.mean()))
            print('Min. noise = {:0.1f} ppm'.format(Nmn.min()))
            print('Max. noise = {:0.1f} ppm'.format(Nmn.max()))

        plt.rc('font', size=fontsize)    
        fig,ax=plt.subplots(2,1,figsize=figsize,sharex=True)

        ax[0].set_xlim(np.min(time),np.max(time))
        ax[0].plot(time, flux,'b.',ms=1)
        ax[0].set_ylabel("Flux ")
        ylo = np.min(flux) - 0.2*np.ptp(flux)
        ypl = np.max(flux) + 0.2*np.ptp(flux)
        yhi = np.max(flux) + 0.4*np.ptp(flux)
        ax[0].set_ylim(ylo, yhi)
        ax[0].errorbar(np.median(T),ypl,xerr=width/48,
               capsize=5,color='b',ecolor='b')
        ax[1].plot(Tsc,Nsc,'b.',ms=1)
        ax[1].plot(Tmn,Nmn,'g.',ms=1)
        ax[1].set_ylabel("Transit noise [ppm] ")
        ax[1].set_xlabel("Time");
        fig.tight_layout()
        if fname is None:
            plt.show()
        else:
            plt.savefig(fname)
        

    def diagnostic_plot(self, fname=None,
            figsize=(8,8), fontsize=10):
        try:
            D = Table(self.lc['table'], masked=True)
        except AttributeError:
            raise AttributeError("Use get_lightcurve() to load data first.")

        EventMask = (D['EVENT'] > 0) & (D['EVENT'] != 100)
        D['FLUX'].mask = EventMask
        D['FLUX_BAD'] = MaskedColumn(self.lc['table']['FLUX'], 
                mask = (EventMask == False))
        D['BACKGROUND'].mask = EventMask
        D['BACKGROUND_BAD'] = MaskedColumn(self.lc['table']['BACKGROUND'],
                mask = (EventMask == False))

        tjdb = D['BJD_TIME']
        flux = D['FLUX']
        flux_bad = D['FLUX_BAD']
        flux_err = D['FLUXERR']
        back = D['BACKGROUND']
        back_bad = D['BACKGROUND_BAD']
        dark = D['DARK']
        contam = D['CONTA_LC']
        contam_err = D['CONTA_LC_ERR']
        rollangle = D['ROLL_ANGLE']
        xloc = D['LOCATION_X']
        yloc = D['LOCATION_Y']
        xcen = D['CENTROID_X']
        ycen = D['CENTROID_Y']

        plt.rc('font', size=fontsize)    
        fig, ax = plt.subplots(4,2,figsize=figsize)
        cgood = 'c'
        cbad = 'r'
        
        ax[0,0].scatter(tjdb,flux,s=2,c=cgood)
        ax[0,0].scatter(tjdb,flux_bad,s=2,c=cbad)
        ax[0,0].set_xlabel('BJD')
        ax[0,0].set_ylabel('Flux in ADU')
        
        ax[0,1].scatter(rollangle,flux,s=2,c=cgood)
        ax[0,1].scatter(rollangle,flux_bad,s=2,c=cbad)
        ax[0,1].set_xlabel('Roll angle in degrees')
        ax[0,1].set_ylabel('Flux in ADU')
        
        ax[1,0].scatter(tjdb,back,s=2,c=cgood)
        ax[1,0].scatter(tjdb,back_bad,s=2,c=cbad)
        ax[1,0].set_xlabel('BJD')
        ax[1,0].set_ylabel('Background in ADU')
        ax[1,0].set_ylim(0.9*np.quantile(back,0.005),
                         1.1*np.quantile(back,0.995))
        
        ax[1,1].scatter(rollangle,back,s=2,c=cgood)
        ax[1,1].scatter(rollangle,back_bad,s=2,c=cbad)
        ax[1,1].set_xlabel('Roll angle in degrees')
        ax[1,1].set_ylabel('Background in ADU')
        ax[1,1].set_ylim(0.9*np.quantile(back,0.005),
                         1.1*np.quantile(back,0.995))
        ax[2,0].scatter(xcen,flux,s=2,c=cgood)
        ax[2,0].scatter(xcen,flux_bad,s=2,c=cbad)
        ax[2,0].set_xlabel('Centroid x')
        ax[2,0].set_ylabel('Flux in ADU')
        
        ax[2,1].scatter(ycen,flux,s=2,c=cgood)
        ax[2,1].scatter(ycen,flux_bad,s=2,c=cbad)
        ax[2,1].set_xlabel('Centroid y')
        ax[2,1].set_ylabel('Flux in ADU')
        
        ax[3,0].scatter(contam,flux,s=2,c=cgood)
        ax[3,0].scatter(contam,flux_bad,s=2,c=cbad)
        
        ax[3,0].set_xlabel('Contamination estimate')
        ax[3,0].set_ylabel('Flux in ADU')
        ax[3,0].set_xlim(np.min(contam),np.max(contam))
        
        ax[3,1].scatter(rollangle,xcen,s=2,c=cgood)
        ax[3,1].scatter(rollangle,ycen,s=2,c=cbad)
        ax[3,1].set_xlabel('Roll angle in degrees')
        ax[3,1].set_ylabel('Centroid x (cyan), y (red)')

        fig.tight_layout()
        if fname is None:
            plt.show()
        else:
            plt.savefig(fname)

