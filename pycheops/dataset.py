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
from .models import TransitModel, FactorModel, EclipseModel
from uncertainties import UFloat
from lmfit import Parameter, Parameters, minimize, Minimizer,fit_report
from scipy.interpolate import InterpolatedUnivariateSpline
import matplotlib.pyplot as plt
from emcee import EnsembleSampler
import corner
import copy
from celerite import terms, GP
from sys import stdout 
from astropy.coordinates import SkyCoord
from lmfit.printfuncs import gformat
from scipy.signal import medfilt
from . import __version__
try:
    from dace.cheops import Cheops
except ModuleNotFoundError: 
    pass

_file_key_re = re.compile(r'CH_PR(\d{2})(\d{4})_TG(\d{4})(\d{2})_V(\d{4})')

# Utility function for model fitting
def _kw_to_Parameter(name, kwarg, min=min, max=max):
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

# Prior on (D, W, b) for transit/eclipse fitting.
# This prior assumes uniform priors on cos(i), log(k) and log(aR). The
# factor 2kW is the absolute value of the determinant of the Jacobian, 
# J = d(D, W, b)/d(cosi, k, aR)
def _log_prior(D, W, b):
    if (D < 2e-6) or (D > 0.2): return -np.inf
    if (b < 0) or (b > 1): return -np.inf
    if (W < 1e-4): return -np.inf
    k = np.sqrt(D)
    aR = np.sqrt((1+k)**2 - b**2)/(np.pi*W)
    if (aR < 2): return -np.inf
    return -np.log(2*k*W) - np.log(k) - np.log(aR)

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
    lnprior = _log_prior(parcopy['D'], parcopy['W'], parcopy['b'])
    if not np.isfinite(lnprior):
        return -np.inf

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
    lnprior = _log_prior(parcopy['D'], parcopy['W'], parcopy['b'])
    if not np.isfinite(lnprior):
        return -np.inf
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

    :param file_key:
    :param force_download:
    :param download_all: If False, download light curves only
    :param configFile:
    :param target:
    :param verbose:

    """

    def __init__(self, file_key, force_download=False, download_all=True,
            configFile=None, target=None, verbose=True):

        self.file_key = file_key
        m = _file_key_re.search(file_key)
        if m is None:
            raise ValueError('Invalid file_key {}'.format(file_key))
        l = [int(i) for i in m.groups()]
        self.progtype,self.prog_id,self.req_id,self.visitctr,self.ver = l

        config = load_config(configFile)
        _cache_path = config['DEFAULT']['data_cache_path']
        tgzPath = Path(_cache_path,file_key).with_suffix('.tgz')
        self.tgzfile = str(tgzPath)

        if tgzPath.is_file() and not force_download:
            if verbose: print('Found archive tgzfile',self.tgzfile)
        else:
            if download_all:
                file_type='all'
            else:
                file_type='lightcurves'
            Cheops.download(file_type, 
                filters={'file_key':{'contains':file_key}},
                output_full_file_path=str(tgzPath)
                )

        lisPath = Path(_cache_path,file_key).with_suffix('.lis')
        # The file list can be out-of-date is force_download is used
        if lisPath.is_file() and not force_download:
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
        lcFile = "{}-{}.fits".format(self.file_key,aperture)
        lcPath = Path(self.tgzfile).parent/lcFile
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
            self.target = hdr['TARGNAME']
        else:
            self.target = target
        coords = SkyCoord(hdr['RA_TARG'],hdr['DEC_TARG'],unit='degree,degree')
        self.ra = coords.ra.to_string(precision=2,unit='hour',sep=':',pad=True)
        self.dec = coords.dec.to_string(precision=1,sep=':',unit='degree',
                alwayssign=True,pad=True)
        self.vmag = hdr['MAG_V']
        self.e_vmag = hdr['MAG_VERR']
        self.spectype = hdr['SPECTYPE']
        self.exptime = hdr['EXPTIME']
        self.texptime = hdr['TEXPTIME']
        self.pipe_ver = hdr['PIPE_VER']
        if verbose:
            print(' PI name     : {}'.format(self.pi_name))
            print(' OBS ID      : {}'.format(self.obsid))
            print(' Target      : {}'.format(self.target))
            print(' Coordinates : {} {}'.format(self.ra, self.dec))
            print(' Spec. type  : {}'.format(self.spectype))
            print(' V magnitude : {:0.2f} +- {:0.2f}'.
                    format(self.vmag, self.e_vmag))
        
#----

    @classmethod
    def from_test_data(self, subdir,  target=None, configFile=None, 
            verbose=True):
        ftp=FTP('obsftp.unige.ch')
        _ = ftp.login()
        wd = "pub/cheops/test_data/{}".format(subdir)
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
            if verbose: print('Downloading {} ...'.format(zipfile))
            ftp.retrbinary(cmd, open(str(zipPath), 'wb').write)
            ftp.quit()
        
        file_key = zipfile[:-4]+'_V0000'
        m = _file_key_re.search(file_key)
        l = [int(i) for i in m.groups()]
        self.progtype,self.prog_id,self.req_id,self.visitctr,self.ver = l

        tgzPath = Path(_cache_path,file_key).with_suffix('.tgz')
        tgzfile = str(tgzPath)

        zpf = ZipFile(str(zipPath), mode='r')
        ziplist = zpf.namelist()

        _re_im = re.compile('(CH_.*SCI_RAW_Imagette_.*.fits)')
        _re_lc = re.compile('(CH_.*_SCI_COR_Lightcurve-.*fits)')
        with tarfile.open(tgzfile, mode='w:gz') as tgz:
            imgfiles = list(filter(_re_im.match, ziplist))
            if len(imgfiles) > 1:
                raise ValueError('More than one imagette file in zip file')
            if len(imgfiles) == 1:
                if verbose: print("Writing Imagette data to .tgz file...")
                imgfile=imgfiles[0]
                tarPath = Path('visit')/Path(file_key)/Path(imgfile).name 
                tarinfo = tarfile.TarInfo(name=str(tarPath))
                zipinfo = zpf.getinfo(imgfile)
                tarinfo.size = zipinfo.file_size
                zf = zpf.open(imgfile)
                tgz.addfile(tarinfo=tarinfo, fileobj=zf)
                zf.close()
            if verbose: print("Writing Lightcurve data to .tgz file...")
            for lcfile in list(filter(_re_lc.match, ziplist)):
                tarPath = Path('visit')/Path(file_key)/Path(lcfile).name
                tarinfo = tarfile.TarInfo(name=str(tarPath))
                zipinfo = zpf.getinfo(lcfile)
                tarinfo.size = zipinfo.file_size
                zf = zpf.open(lcfile)
                tgz.addfile(tarinfo=tarinfo, fileobj=zf)
                zf.close()
                if verbose: print ('.. {} - done'.format(Path(lcfile).name))
        zpf.close()

        return self(file_key=file_key, target=target, verbose=verbose)
        
#----

    @classmethod
    def from_simulation(self, job,  target=None, configFile=None, 
            version=0, verbose=True):
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
            if verbose: print('Downloading {} ...'.format(zipfile))
            ftp.retrbinary(cmd, open(str(zipPath), 'wb').write)
            ftp.quit()
        
        file_key = "{}_V{:04d}".format(zipfile[:-4],version)
        m = _file_key_re.search(file_key)
        l = [int(i) for i in m.groups()]

        tgzPath = Path(_cache_path,file_key).with_suffix('.tgz')
        tgzfile = str(tgzPath)

        zpf = ZipFile(str(zipPath), mode='r')
        ziplist = zpf.namelist()

        _re_im = re.compile('(CH_.*SCI_RAW_Imagette_.*.fits)')
        _re_lc = re.compile('(CH_.*_SCI_COR_Lightcurve-.*fits)')
        with tarfile.open(tgzfile, mode='w:gz') as tgz:
            imgfiles = list(filter(_re_im.match, ziplist))
            if len(imgfiles) > 1:
                raise ValueError('More than one imagette file in zip file')
            if len(imgfiles) == 1:
                if verbose: print("Writing Imagette data to .tgz file...")
                imgfile=imgfiles[0]
                tarPath = Path('visit')/Path(file_key)/Path(imgfile).name 
                tarinfo = tarfile.TarInfo(name=str(tarPath))
                zipinfo = zpf.getinfo(imgfile)
                tarinfo.size = zipinfo.file_size
                zf = zpf.open(imgfile)
                tgz.addfile(tarinfo=tarinfo, fileobj=zf)
                zf.close()
            if verbose: print("Writing Lightcurve data to .tgz file...")
            for lcfile in list(filter(_re_lc.match, ziplist)):
                tarPath = Path('visit')/Path(file_key)/Path(lcfile).name
                tarinfo = tarfile.TarInfo(name=str(tarPath))
                zipinfo = zpf.getinfo(lcfile)
                tarinfo.size = zipinfo.file_size
                zf = zpf.open(lcfile)
                tgz.addfile(tarinfo=tarinfo, fileobj=zf)
                zf.close()
                if verbose: print ('.. {} - done'.format(Path(lcfile).name))
        zpf.close()

        return self(file_key=file_key, target=target, verbose=verbose)

#----
        
    def get_imagettes(self, verbose=True):
        imFile = "{}-Imagette.fits".format(self.file_key)
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
            returnTable=False, reject_highpoints=False, verbose=True):

        if aperture not in ('OPTIMAL','RSUP','RINF','DEFAULT'):
            raise ValueError('Invalid/missing aperture name')

        lcFile = "{}-{}.fits".format(self.file_key,aperture)
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
        self.bjd_ref = bjd_ref
        time = bjd-bjd_ref
        flux = np.array(table['FLUX'][ok])
        flux_err = np.array(table['FLUXERR'][ok])
        fluxmed = np.nanmedian(flux)
        xoff = np.array(table['CENTROID_X'][ok]- table['LOCATION_X'][ok])
        yoff = np.array(table['CENTROID_Y'][ok]- table['LOCATION_Y'][ok])
        roll_angle = np.array(table['ROLL_ANGLE'][ok])
        bg = np.array(table['BACKGROUND'][ok])
        contam = np.array(table['CONTA_LC'][ok])
        ap_rad = hdr['AP_RADI']
        self.bjd_ref = bjd_ref
        self.ap_rad = ap_rad
        if verbose:
            print('Time stored relative to BJD = {:0.0f}'.format(bjd_ref))
            print('Aperture radius used = {:0.0f} arcsec'.format(ap_rad))

        if reject_highpoints:
            C_cut = (2*np.nanmedian(flux)-np.min(flux))
            ok  = (flux < C_cut).nonzero()
            time = time[ok]
            flux = flux[ok]
            flux_err = flux_err[ok]
            xoff = xoff[ok]
            yoff = yoff[ok]
            roll_angle = roll_angle[ok]
            bg = bg[ok]
            contam = contam[ok]
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
                np.nanmedian(flux_err), 1e6*np.nanmedian(flux_err)/fluxmed))

        self.flux_mean = flux.mean()
        self.flux_median = fluxmed
        self.flux_rms = np.std(flux)
        self.flux_mse = np.nanmedian(flux_err)
        flux = flux/fluxmed
        flux_err = flux_err/fluxmed
        self.lc = {'time':time, 'flux':flux, 'flux_err':flux_err,
                'bjd_ref':bjd_ref, 'table':table, 'header':hdr,
                'xoff':xoff, 'yoff':yoff, 'bg':bg, 'contam':contam,
                'centroid_x':np.array(table['CENTROID_X'][ok]),
                'centroid_y':np.array(table['CENTROID_Y'][ok]),
                'roll_angle':roll_angle, 'aperture':aperture}

        if returnTable:
            return table
        else:
            return time, flux, flux_err

 #----------------------------------------------------------------------------
 # Eclipse and transit fitting

    def lmfit_transit(self, 
            T_0=None, P=None, D=None, W=None, b=None, f_c=None, f_s=None,
            h_1=None, h_2=None,
            c=None, dfdbg=None, dfdcontam=None, 
            dfdx=None, dfdy=None, d2fdx2=None, d2fdy2=None,
            dfdsinphi=None, dfdcosphi=None, dfdsin2phi=None, dfdcos2phi=None,
            dfdsin3phi=None, dfdcos3phi=None, dfdt=None, d2fdt2=None, 
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
            params['T_0'] = _kw_to_Parameter('T_0', T_0)
        if P is None:
            params.add(name='P', value=1, vary=False)
        else:
            params['P'] = _kw_to_Parameter('P', P)
        _P = params['P'].value
        if D is None:
            params.add(name='D', value=1-min(flux), min=0,max=0.5)
        else:
            params['D'] = _kw_to_Parameter('D', D)
        k = np.sqrt(params['D'].value)
        if W is None:
            params.add(name='W', value=np.ptp(time)/2/_P,
                    min=np.ptp(time)/len(time)/_P, max=np.ptp(time)/_P) 
        else:
            params['W'] = _kw_to_Parameter('W', W)
        if b is None:
            params.add(name='b', value=0.5, min=0, max=1)
        else:
            params['b'] = _kw_to_Parameter('b', b)
        if f_c is None:
            params.add(name='f_c', value=0, vary=False)
        else:
            params['f_c'] = _kw_to_Parameter('f_c', f_c)
        if f_s is None:
            params.add(name='f_s', value=0, vary=False)
        else:
            params['f_s'] = _kw_to_Parameter('f_s', f_s)
        if h_1 is None:
            params.add(name='h_1', value=0.7224, vary=False)
        else:
            params['h_1'] = _kw_to_Parameter('h_1', h_1, min=0, max=1)
        if h_2 is None:
            params.add(name='h_2', value=0.6713, vary=False)
        else:
            params['h_2'] = _kw_to_Parameter('h_2', h_2, min=0, max=1)
        if c is None:
            params.add(name='c', value=1, min=min(flux)/2,max=2*max(flux))
        else:
            params['c'] = _kw_to_Parameter('c', c)
        if dfdbg is not None:
            params['dfdbg'] = _kw_to_Parameter('dfdbg', dfdbg)
        if dfdcontam is not None:
            params['dfdcontam'] = _kw_to_Parameter('dfdcontam', dfdcontam)
        if dfdx is not None:
            params['dfdx'] = _kw_to_Parameter('dfdx', dfdx)
        if dfdy is not None:
            params['dfdy'] = _kw_to_Parameter('dfdy', dfdy)
        if d2fdx2 is not None:
            params['d2fdx2'] = _kw_to_Parameter('d2fdx2', d2fdx2)
        if d2fdy2 is not None:
            params['d2fdy2'] = _kw_to_Parameter('d2fdy2', d2fdy2)
        if dfdt is not None:
            params['dfdt'] = _kw_to_Parameter('dfdt', dfdt)
        if d2fdt2 is not None:
            params['d2fdt2'] = _kw_to_Parameter('d2fdt2', d2fdt2)
        if dfdsinphi is not None:
            params['dfdsinphi'] = _kw_to_Parameter('dfdsinphi', dfdsinphi)
        if dfdcosphi is not None:
            params['dfdcosphi'] = _kw_to_Parameter('dfdcosphi', dfdcosphi)
        if dfdsin2phi is not None:
            params['dfdsin2phi'] = _kw_to_Parameter('dfdsin2phi', dfdsin2phi)
        if dfdcos2phi is not None:
            params['dfdcos2phi'] = _kw_to_Parameter('dfdcos2phi', dfdcos2phi)
        if dfdsin3phi is not None:
            params['dfdsin3phi'] = _kw_to_Parameter('dfdsin3phi', dfdsin3phi)
        if dfdcos3phi is not None:
            params['dfdcos3phi'] = _kw_to_Parameter('dfdcos3phi', dfdcos3phi)

        params.add('k',expr='sqrt(D)',min=0,max=1)
        params.add('aR',expr='sqrt((1+k)**2-b**2)/W/pi',min=1)
        # Avoid use of aR in this expr for logrho - breaks error propogation.
        expr = 'log10(4.3275e-4*((1+k)**2-b**2)**1.5/W**3/P**2)'
        params.add('logrho',expr=expr,min=-9,max=6)
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
    def lmfit_eclipse(self, 
            T_0=None, P=None, D=None, W=None, b=None, L=None, 
            f_c=None, f_s=None, a_c=None, dfdbg=None, dfdcontam=None, 
            c=None, dfdx=None, dfdy=None, d2fdx2=None, d2fdy2=None,
            dfdsinphi=None, dfdcosphi=None, dfdsin2phi=None, dfdcos2phi=None,
            dfdsin3phi=None, dfdcos3phi=None, dfdt=None, d2fdt2=None):

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
            params['T_0'] = _kw_to_Parameter('T_0', T_0)
        if P is None:
            params.add(name='P', value=1, vary=False)
        else:
            params['P'] = _kw_to_Parameter('P', P)
        _P = params['P'].value
        if D is None:
            params.add(name='D', value=1-min(flux), min=0,max=0.5)
        else:
            params['D'] = _kw_to_Parameter('D', D)
        k = np.sqrt(params['D'].value)
        if W is None:
            params.add(name='W', value=np.ptp(time)/2/_P,
                    min=np.ptp(time)/len(time)/_P, max=np.ptp(time)/_P) 
        else:
            params['W'] = _kw_to_Parameter('W', W)
        if b is None:
            params.add(name='b', value=0.5, min=0, max=1)
        else:
            params['b'] = _kw_to_Parameter('b', b)
        if L is None:
            params.add(name='L', value=0.001, min=0, max=1)
        else:
            params['L'] = _kw_to_Parameter('L', L)
        if f_c is None:
            params.add(name='f_c', value=0, vary=False)
        else:
            params['f_c'] = _kw_to_Parameter('f_c', f_c)
        if f_s is None:
            params.add(name='f_s', value=0, vary=False)
        else:
            params['f_s'] = _kw_to_Parameter('f_s', f_s)
        if c is None:
            params.add(name='c', value=1, min=min(flux)/2,max=2*max(flux))
        else:
            params['c'] = _kw_to_Parameter('c', c)
        if a_c is None:
            params.add(name='a_c', value=0, vary=False)
        else:
            params['a_c'] = _kw_to_Parameter('a_c', a_c)
        if dfdbg is not None:
            params['dfdbg'] = _kw_to_Parameter('dfdbg', dfdbg)
        if dfdcontam is not None:
            params['dfdcontam'] = _kw_to_Parameter('dfdcontam', dfdcontam)
        if dfdx is not None:
            params['dfdx'] = _kw_to_Parameter('dfdx', dfdx)
        if dfdy is not None:
            params['dfdy'] = _kw_to_Parameter('dfdy', dfdy)
        if d2fdx2 is not None:
            params['d2fdx2'] = _kw_to_Parameter('d2fdx2', df2dx2)
        if d2fdy2 is not None:
            params['d2fdy2'] = _kw_to_Parameter('d2fdy2', df2dy2)
        if dfdt is not None:
            params['dfdt'] = _kw_to_Parameter('dfdt', dfdt)
        if d2fdt2 is not None:
            params['d2fdt2'] = _kw_to_Parameter('d2fdt2', df2dt2)
        if dfdsinphi is not None:
            params['dfdsinphi'] = _kw_to_Parameter('dfdsinphi', dfdsinphi)
        if dfdcosphi is not None:
            params['dfdcosphi'] = _kw_to_Parameter('dfdcosphi', dfdcosphi)
        if dfdsin2phi is not None:
            params['dfdsin2phi'] = _kw_to_Parameter('dfdsin2phi', dfdsin2phi)
        if dfdcos2phi is not None:
            params['dfdcos2phi'] = _kw_to_Parameter('dfdcos2phi', dfdcos2phi)
        if dfdsin3phi is not None:
            params['dfdsin3phi'] = _kw_to_Parameter('dfdsin3phi', dfdsin3phi)
        if dfdcos3phi is not None:
            params['dfdcos3phi'] = _kw_to_Parameter('dfdcos3phi', dfdcos3phi)

        params.add('k',expr='sqrt(D)',min=0,max=1)
        params.add('aR',expr='sqrt((1+k)**2-b**2)/W/pi',min=1)
        params.add('e',min=0,max=1,expr='f_c**2 + f_s**2')

        model = EclipseModel()*FactorModel(
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
        report += 'pycheops version %s\n' % __version__
        report += 'CHEOPS DRP version %s\n' % self.pipe_ver
        return(report)

    # ----------------------------------------------------------------

    def emcee_sampler(self, params=None,
            steps=128, nwalkers=64, burn=256, thin=4, log_sigma=None, 
            add_shoterm=False, log_omega0=None, log_S0=None, log_Q=None,
            init_scale=1e-3, progress=True):

        try:
            time = np.array(self.lc['time'])
            flux = np.array(self.lc['flux'])
            flux_err = np.array(self.lc['flux_err'])
        except AttributeError:
            raise AttributeError("Use get_lightcurve() to load data first.")

        try:
            model = self.model
        except AttributeError:
            raise AttributeError(
                    "Use lmfit_transit() or lmfit_eclipse() first.")

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
                if log_S0 is None:
                    params.add('log_S0', value=-11,  min=-16, max=-1)
                else:
                    params['log_S0'] = _kw_to_Parameter('log_S0', log_S0)
                # For time in days, and the default value of Q=1/sqrt(2),
                # log_omega0=12  is a correlation length 
                # of about 0.5s and -2.3 is about 10 days.
                if log_omega0 is None:
                    params.add('log_omega0', value=6, min=-2.3, max=12)
                else:
                    lw0 =  _kw_to_Parameter('log_omega0', log_omega0)
                    params['log_omega0'] = lw0
                if log_Q is None:
                    params.add('log_Q', value=np.log(1/np.sqrt(2)), vary=False)
                else:
                    params['log_Q'] = _kw_to_Parameter('log_Q', log_Q)

        if log_sigma is None:
            if not 'log_sigma' in params:
                params.add('log_sigma', value=-10, min=-15,max=0)
                params['log_sigma'].stderr = 1
        else:
            params['log_sigma'] = _kw_to_Parameter('log_sigma', log_sigma)
        params.add('sigma_w',expr='exp(log_sigma)*1e6')

        vv = []
        vs = []
        vn = []
        for p in params:
            if params[p].vary:
                vn.append(p)
                vv.append(params[p].value)
                if params[p].stderr is None:
                    if params[p].user_data is None:
                        vs.append(0.1*(params[p].max-params[p].min))
                    else:
                        vs.append(params[p].user_data.s)
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
        report += 'pycheops version %s\n' % __version__
        report += 'CHEOPS DRP version %s\n' % self.pipe_ver
        return(report)

    # ----------------------------------------------------------------

    def corner_plot(self, plotkeys=['T_0', 'D', 'W', 'b'], 
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
                elif key == 'dfdbg':
                    labels.append(r'$df/d{\rm (bg)}$')
                elif key == 'dfdcontam':
                    labels.append(r'$df/d{\rm (contam)}$')
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
                elif key == 'dfdsin2phi':
                    labels.append(r'$df/d\sin(2\phi)$')
                elif key == 'dfdcos2phi':
                    labels.append(r'$df/d\cos(2\phi)$')
                elif key == 'dfdsin3phi':
                    labels.append(r'$df/d\sin(3\phi)$')
                elif key == 'dfdcos3phi':
                    labels.append(r'$df/d\cos(3\phi)$')
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

            if 'b' in varkeys:
                b = chain[:,varkeys.index('b')]
            else:
                b = params['b'].value

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
    
    def plot_lmfit(self, figsize=(6,4), fontsize=11, title=None, detrend=False):
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

        res = flux - self.model.eval(params, t=time)
        tmin = np.round(np.min(time)-0.05*np.ptp(time),2)
        tmax = np.round(np.max(time)+0.05*np.ptp(time),2)
        tp = np.linspace(tmin, tmax, 10*len(time))
        fp = self.model.eval(params,t=tp)
        if detrend:
            fp = fp / model.right.eval(params, t=tp) 
            flux = flux / model.right.eval(params, t=time) 

        plt.rc('font', size=fontsize)    
        fig,ax=plt.subplots(nrows=2,sharex=True, figsize=figsize,
                gridspec_kw={'height_ratios':[2,1]})
        ax[0].errorbar(time,flux,yerr=flux_err,fmt='bo',ms=3,zorder=0)
        ax[0].plot(tp,fp,c='orange',zorder=1)
        ax[0].set_xlim(tmin, tmax)
        ymin = np.min(flux-flux_err)-0.05*np.ptp(flux)
        ymax = np.max(flux+flux_err)+0.05*np.ptp(flux)
        ax[0].set_ylim(ymin,ymax)
        ax[0].set_title(title)
        if detrend:
            ax[0].set_ylabel('Flux/trend')
        else:
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
    
    def plot_emcee(self, title=None, nsamples=32, detrend=False,
            figsize=(6,4), fontsize=11):
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
        if detrend:
            flux = flux / self.model.right.eval(parbest, t=time)
        nchain = self.emcee.chain.shape[0]
        partmp = parbest.copy()
        ax[0].errorbar(time,flux,yerr=flux_err,fmt='bo',ms=3,zorder=0)
        if self.gp is None:
            fp = self.model.eval(parbest,t=tp)
            if detrend:
                fp = fp / self.model.right.eval(parbest, t=tp)
            ax[0].plot(tp,fp, c='orange',zorder=1)
            for i in np.linspace(0,nchain,nsamples,endpoint=False,dtype=np.int):
                for j, n in enumerate(self.emcee.var_names):
                    partmp[n].value = self.emcee.chain[i,j]
                    fp = self.model.eval(partmp,t=tp)
                    if detrend:
                        fp = fp / self.model.right.eval(partmp, t=tp)
                ax[0].plot(tp,fp, c='orange',zorder=1,alpha=0.1)
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
            pp = mu0+self.model.eval(parbest, t=tp)
            if detrend:
                pp = pp / self.model.right.eval(parbest, t=tp)
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
                if detrend:
                    pp = pp / self.model.right.eval(parbest, t=tp)
                ax[0].plot(tp, pp, c='orange',zorder=1,alpha=0.1)
                
        ymin = np.min(flux-flux_err)-0.05*np.ptp(flux)
        ymax = np.max(flux+flux_err)+0.05*np.ptp(flux)
        ax[0].set_xlim(tmin, tmax)
        ax[0].set_ylim(ymin,ymax)
        ax[0].set_title(title)
        if detrend:
            ax[0].set_ylabel('Flux/trend')
        else:
            ax[0].set_ylabel('Flux')
        # SHOTerm sometimes offset from 0 - fix that here
        off = res.mean()
        ax[1].errorbar(time,res-off,yerr=flux_err,fmt='bo',ms=3,zorder=0)
        if self.gp is not None:
            ax[1].plot(tp,mu0-off,c='orange', zorder=1)
        ax[1].plot([tmin,tmax],[0,0],ls=':',c='orange', zorder=1)
        ax[1].set_xlabel('BJD-{}'.format(self.lc['bjd_ref']))
        ax[1].set_ylabel('Residual')
        ylim = np.max(np.abs(res-flux_err)+0.05*np.ptp(res))
        ax[1].set_ylim(-ylim,ylim)
        fig.tight_layout()
        return fig
        
 #----------------------------------------------------------------------------
 # Data display and diagnostics

    def transit_noise_plot(self, width=3, steps=500,
            fname=None, figsize=(6,4), fontsize=11,
            requirement=None, local=False, verbose=True):

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
            if local:
                j = (np.abs(time-_t) < (width/48)).nonzero()[0]
                _n,_f = transit_noise(time[j], flux[j], flux_err[j], T_0=_t,
                              width=width, method='scaled')
                _m = transit_noise(time[j], flux[j], flux_err[j], T_0=_t,
                           width=width, method='minerr')
            else:
                _n,_f = transit_noise(time, flux, flux_err, T_0=_t,
                              width=width, method='scaled')
                _m = transit_noise(time, flux, flux_err, T_0=_t,
                           width=width, method='minerr')
            if np.isfinite(_n):
                Nsc[i] = _n
                Fsc[i] = _f
            if np.isfinite(_m):
                Nmn[i] = _m

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
        if requirement is not None:
            xr = [np.min(time),np.max(time)]
            yr = [requirement, requirement]
            ax[1].plot(xr, yr, color='darkcyan',ls=':')
        fig.tight_layout()
        if fname is None:
            plt.show()
        else:
            plt.savefig(fname)
        

    #------

    def flatten(self, mask_centre, mask_width, npoly=2):
        """
        Renormalize using a polynomial fit excluding a section of the data
     
        The position and width of the mask to exclude the transit/eclipse is
        specified on the same time scale as the light curve data.

        :param mask_centre: time at the centre of the mask
        :param mask_width: full width of the mask
        :param npoly: number of terms in the normalizing polynomial

        :returns: time, flux, flux_err

        """
        time = self.lc['time']
        flux = self.lc['flux']
        flux_err = self.lc['flux_err']
        mask = abs(time-mask_centre) > mask_width/2
        n = np.polyval(np.polyfit(time[mask],flux[mask],npoly-1),time)
        self.lc['flux'] /= n
        self.lc['flux_err'] /= n

        return self.lc['time'], self.lc['flux'], self.lc['flux_err']

    #------

    def clip_outliers(self, clip=5, width=11, verbose=True):
        """
        Remove outliers from the light curve.

        Data more than clip*mad from a smoothed version of the light curve are
        removed where mad is the mean absolute deviation from the
        median-smoothed light curve.

        :param clip: tolerance on clipping
        :param width: width of window for median-smoothing filter

        :returns: time, flux, flux_err

        """
        flux = self.lc['flux']
        d = abs(flux - medfilt(flux, width))
        mad = d.mean()
        ok = d < clip*mad
        self.lc = {'time':self.lc['time'][ok], 'flux':flux[ok],
                'flux_err':self.lc['flux_err'][ok], 'xoff':self.lc['xoff'][ok],
                'yoff':self.lc['yoff'][ok], 'bjd_ref':self.lc['bjd_ref'],
                'table':self.lc['table'], 'header':self.lc['header'],
                'centroid_x':self.lc['centroid_x'],
                'centroid_y':self.lc['centroid_y'],
                'roll_angle':self.lc['roll_angle'][ok]}
        if verbose:
            print('\nRejected {} points more than {:0.1f} x MAD = {:0.0f} ppm '
                    'from the median'.format(sum(~ok),clip,1e6*mad*clip))
        return self.lc['time'], self.lc['flux'], self.lc['flux_err']

    #------

    def diagnostic_plot(self, fname=None,
            figsize=(8,8), fontsize=10, compare=None):
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
        
        if compare:
            time_detrend = np.array(self.lc['time'])+self.lc['bjd_ref']
            flux_detrend = np.array(self.lc['flux'])*np.nanmean(flux)
            flux_err_detrend = np.array(self.lc['flux_err'])*np.nanmean(flux)
            rollangle_detrend = np.array(self.lc['roll_angle'])
            xcen_detrend = np.array(self.lc['centroid_x'])
            ycen_detrend = np.array(self.lc['centroid_y'])
        
        plt.rc('font', size=fontsize)    
        fig, ax = plt.subplots(4,2,figsize=figsize)
        cgood = 'c'
        cbad = 'r'
        cdetrend = 'b'
        
        ylim_min, ylim_max = 0.995*np.nanmean(flux), 1.005*np.nanmean(flux)
        ax[0,0].scatter(tjdb,flux,s=2,c=cgood)
        #ax[0,0].scatter(tjdb,flux_bad,s=2,c=cbad)
        if compare:
            ax[0,0].scatter(time_detrend,flux_detrend,s=2,c=cdetrend)   
            ax[0,0].set_ylim(ylim_min,ylim_max)
        ax[0,0].set_xlabel('BJD')
        ax[0,0].set_ylabel('Flux in ADU')
        
        ax[0,1].scatter(rollangle,flux,s=2,c=cgood)
        #ax[0,1].scatter(rollangle,flux_bad,s=2,c=cbad)
        if compare:
            ax[0,1].scatter(rollangle_detrend,flux_detrend,s=2,c=cdetrend)
            ax[0,1].set_ylim(ylim_min,ylim_max)
        ax[0,1].set_xlabel('Roll angle in degrees')
        ax[0,1].set_ylabel('Flux in ADU')
        
        ax[1,0].scatter(tjdb,back,s=2,c=cgood)
        #ax[1,0].scatter(tjdb,back_bad,s=2,c=cbad)
        ax[1,0].set_xlabel('BJD')
        ax[1,0].set_ylabel('Background in ADU')
        ax[1,0].set_ylim(0.9*np.quantile(back,0.005),
                         1.1*np.quantile(back,0.995))
        
        ax[1,1].scatter(rollangle,back,s=2,c=cgood)
        #ax[1,1].scatter(rollangle,back_bad,s=2,c=cbad)
        ax[1,1].set_xlabel('Roll angle in degrees')
        ax[1,1].set_ylabel('Background in ADU')
        ax[1,1].set_ylim(0.9*np.quantile(back,0.005),
                         1.1*np.quantile(back,0.995))
        
        ax[2,0].scatter(xcen,flux,s=2,c=cgood)
        #ax[2,0].scatter(xcen,flux_bad,s=2,c=cbad)
        if compare:
            ax[2,0].scatter(xcen_detrend,flux_detrend,s=2,c=cdetrend)
            ax[2,0].set_ylim(ylim_min,ylim_max)
        ax[2,0].set_xlabel('Centroid x')
        ax[2,0].set_ylabel('Flux in ADU')
        
        ax[2,1].scatter(ycen,flux,s=2,c=cgood)
        #ax[2,1].scatter(ycen,flux_bad,s=2,c=cbad)
        if compare:
            ax[2,1].scatter(ycen_detrend,flux_detrend,s=2,c=cdetrend)
            ax[2,1].set_ylim(ylim_min,ylim_max)
        ax[2,1].set_xlabel('Centroid y')
        ax[2,1].set_ylabel('Flux in ADU')
        
        ax[3,0].scatter(contam,flux,s=2,c=cgood)
        #ax[3,0].scatter(contam,flux_bad,s=2,c=cbad)
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

    #------

    def decorr(self, dfdt=False, d2fdt2=False, dfdx=False, d2fdx2=False, 
                dfdy=False, d2fdy2=False, d2fdxdy=False, dfdsinphi=False, 
                dfdcosphi=False, dfdsin2phi=False, dfdcos2phi=False):

        time = np.array(self.lc['time'])
        flux = np.array(self.lc['flux'])
        flux_err = np.array(self.lc['flux_err'])
        phi = self.lc['roll_angle']*np.pi/180
        sinphi = InterpolatedUnivariateSpline(time,np.sin(phi))
        cosphi = InterpolatedUnivariateSpline(time,np.cos(phi))

        dx = InterpolatedUnivariateSpline(time,self.lc['xoff'])
        dy = InterpolatedUnivariateSpline(time,self.lc['yoff'])

        model = FactorModel(sinphi=sinphi, cosphi=cosphi, dx=dx, dy=dy)
        params = model.make_params()
        params.add('dfdt', value=0, vary=dfdt)
        params.add('d2fdt2', value=0, vary=d2fdt2)
        params.add('dfdx', value=0, vary=dfdx)
        params.add('d2fdx2', value=0, vary=d2fdx2)
        params.add('dfdy', value=0, vary=dfdy)
        params.add('d2fdy2', value=0, vary=d2fdy2)
        params.add('d2fdxdy', value=0, vary=d2fdxdy)
        params.add('dfdsinphi', value=0, vary=dfdsinphi)
        params.add('dfdcosphi', value=0, vary=dfdcosphi)
        params.add('dfdsin2phi', value=0, vary=dfdsin2phi)
        params.add('dfdcos2phi', value=0, vary=dfdcos2phi)

        result = model.fit(flux, params, t=time)
        print("Fit Report")
        print(result.fit_report())
        result.plot()

        print("\nCompare the lightcurve RMS before and after decorrelation")
        print('RMS before = {:0.1f} ppm'.format(1e6*self.lc['flux'].std()))
        self.lc['flux'] =  flux/result.best_fit
        self.lc['flux_err'] =  flux_err/result.best_fit
        print('RMS after = {:0.1f} ppm'.format(1e6*self.lc['flux'].std()))

        flux = flux/result.best_fit
        fig,ax=plt.subplots(1,2,figsize=(8,4))
        y = 1e6*(flux-1)
        ax[0].plot(time, y,'b.',ms=1)
        ax[0].set_xlabel("BJD-{}".format((self.lc['bjd_ref'])),fontsize=12)
        ax[0].set_ylabel("Flux-1 [ppm]",fontsize=12)
        fig.suptitle('Detrended fluxes')
        n, bins, patches = ax[1].hist(y, 50, density=True, stacked=True)
        ax[1].set_xlabel("Flux-1 [ppm]",fontsize=12)
        v  = np.var(y)
        ax[1].plot(bins,np.exp(-0.5*bins**2/v)/np.sqrt(2*np.pi*v))
        fig.tight_layout()
        fig.subplots_adjust(top=0.88)
        

    def should_I_decorr(self,cut=20,compare=False):
        
        cut_val = cut
        time = np.array(self.lc['time'])
        flux = np.array(self.lc['flux'])
        flux_err = np.array(self.lc['flux_err'])
        phi = self.lc['roll_angle']*np.pi/180
        sinphi = InterpolatedUnivariateSpline(time,np.sin(phi))
        cosphi = InterpolatedUnivariateSpline(time,np.cos(phi))
        dx = InterpolatedUnivariateSpline(time,self.lc['xoff'])
        dy = InterpolatedUnivariateSpline(time,self.lc['yoff'])

        dfdx_bad, dfdy_bad, dfdsinphi_bad, dfdcosphi_bad = np.array([]), np.array([]), np.array([]), np.array([])
        for dfdx in [False, True]:
            for dfdy in [False, True]:
                for dfdsinphi in [False, True]:
                    for dfdcosphi in [False, True]:
                
                        model = FactorModel(sinphi=sinphi, cosphi=cosphi, dx=dx, dy=dy)
                        params = model.make_params()
                        params.add('dfdt', value=0, vary=False)
                        params.add('d2fdt2', value=0, vary=False)
                        params.add('dfdx', value=0, vary=dfdx)
                        params.add('d2fdx2', value=0, vary=False)
                        params.add('dfdy', value=0, vary=dfdy)
                        params.add('d2fdy2', value=0, vary=False)
                        params.add('d2fdxdy', value=0, vary=False)
                        params.add('dfdsinphi', value=0, vary=dfdsinphi)
                        params.add('dfdcosphi', value=0, vary=dfdcosphi)
                        params.add('dfdsin2phi', value=0, vary=False)
                        params.add('dfdcos2phi', value=0, vary=False)

                        result = model.fit(flux, params, t=time)        

                        if result.params['dfdx'].vary == True:
                            if abs(100*result.params['dfdx'].stderr/result.params['dfdx'].value) < cut_val:
                                dfdx_bad = np.append(dfdx_bad, abs(100*result.params['dfdx'].stderr/result.params['dfdx'].value))
                        if result.params['dfdy'].vary == True:
                            if abs(100*result.params['dfdy'].stderr/result.params['dfdy'].value) < cut_val:
                                dfdy_bad = np.append(dfdy_bad, abs(100*result.params['dfdy'].stderr/result.params['dfdy'].value))
                        if result.params['dfdsinphi'].vary == True:
                            if abs(100*result.params['dfdsinphi'].stderr/result.params['dfdsinphi'].value) < cut_val:
                                dfdsinphi_bad = np.append(dfdsinphi_bad, abs(100*result.params['dfdsinphi'].stderr/result.params['dfdsinphi'].value))
                        if result.params['dfdcosphi'].vary == True:
                            if abs(100*result.params['dfdcosphi'].stderr/result.params['dfdcosphi'].value) < cut_val:
                                dfdcosphi_bad = np.append(dfdcosphi_bad,
                                        abs(100*result.params['dfdcosphi'].stderr/result.params['dfdcosphi'].value))
            
        if len(dfdx_bad) == 0 and len(dfdy_bad) == 0 and len(dfdsinphi_bad) == 0 and len(dfdcosphi_bad) == 0:
            print("No! You don't need to decorrelate.")
        else:
            if len(dfdx_bad) > 0:
                print("Yes! Check flux against centroid x.")
                
            if len(dfdy_bad) > 0:
                print("Yes! Check flux against centroid y.")
                
            if len(dfdsinphi_bad) > 0 or len(dfdcosphi_bad) > 0:
                print("Yes! Check flux against roll angle.")
            
            self.diagnostic_plot(fontsize=9,compare=compare)
            
            decorr_check = input('Do you want to decorrelate? ')
            if decorr_check.lower()[0] == "y":
                which_decorr = input('Which to you wish to decorrelate? Please enter from the follow: centroid_x, centroid_y, and/or roll_angle. Multiple entries should be comma separated. ')
                dfdx_arg, dfdy_arg, dfdsinphi_arg, dfdcosphi_arg = False, False, False, False
                which_decorr = which_decorr.split(",")
                
                for index, i in enumerate(which_decorr):
                    which_decorr[index] = i.lower().replace(' ', '')
                if "centroid_x" in which_decorr:
                    dfdx_arg = True
                if "centroid_y" in which_decorr:
                    dfdy_arg = True
                if "roll_angle" in which_decorr:
                    dfdsinphi_arg, dfdcosphi_arg = True, True
                self.decorr(dfdx=dfdx_arg, dfdy=dfdy_arg,
                        dfdsinphi=dfdsinphi_arg, dfdcosphi=dfdcosphi_arg)
                
            elif "centroid_x" in decorr_check or "centroid_y" in decorr_check or "roll_angle" in decorr_check:
                dfdx_arg, dfdy_arg, dfdsinphi_arg, dfdcosphi_arg = False, False, False, False
                decorr_check = decorr_check.split(",")
                
                for index, i in enumerate(decorr_check):
                    decorr_check[index] = i.lower().replace(' ', '')
                if "centroid_x" in decorr_check:
                    dfdx_arg = True
                if "centroid_y" in decorr_check:
                    dfdy_arg = True
                if "roll_angle" in decorr_check:
                    dfdsinphi_arg, dfdcosphi_arg = True, True
                self.decorr(dfdx=dfdx_arg, dfdy=dfdy_arg,
                        dfdsinphi=dfdsinphi_arg, dfdcosphi=dfdcosphi_arg)
                
            else:
                print("Ok then")
        print('\n')    
            
            
            

