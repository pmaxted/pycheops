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
import logging
from pathlib import Path
from .core import load_config
from astropy.io import fits
from astropy.table import Table, MaskedColumn
import matplotlib.pyplot as plt
from .instrument import transit_noise
from ftplib import FTP
from .models import TransitModel, FactorModel, EclipseModel
from lmfit import Parameter, Parameters, minimize, Minimizer, fit_report
from lmfit import __version__ as _lmfit_version_
from lmfit import Model
from scipy.interpolate import interp1d, LSQUnivariateSpline
import matplotlib.pyplot as plt
from emcee import EnsembleSampler
import corner
from copy import copy, deepcopy
from celerite2 import terms, GaussianProcess
from celerite2.terms import SHOTerm
from sys import stdout 
from astropy.coordinates import SkyCoord, get_body, Angle
from lmfit.printfuncs import gformat
from scipy.signal import medfilt
from .utils import lcbin, mode
import astropy.units as u
from uncertainties import ufloat, UFloat
from uncertainties.umath import sqrt as usqrt
from astropy.timeseries import LombScargle
from astropy.time import Time
from astropy.convolution import convolve, Gaussian1DKernel
from .instrument import CHEOPS_ORBIT_MINUTES
from scipy.stats import skewnorm
from scipy.optimize import minimize as scipy_minimize
from . import __version__
from .funcs import rhostar, massradius
from tqdm import tqdm_notebook as tqdm
import matplotlib.animation as animation
import matplotlib.colors as colors
from IPython.display import Image
import subprocess
import pickle
import warnings
from astropy.units import UnitsWarning
from photutils.aperture import aperture_photometry, CircularAperture
import cdspyreadme
from textwrap import fill, indent
import os
from contextlib import redirect_stderr
from dace_query.cheops import Cheops

_file_key_re = re.compile(r'CH_PR(\d{2})(\d{4})_TG(\d{4})(\d{2})_V(\d{4})')
_file_key_reT = re.compile(r'TIC_(\d{10})_SEC(\d{4})_V(\d{4})')
_file_key_reP = re.compile(r'PIPE_CH_PR(\d{2})(\d{4})_TG(\d{4})(\d{2})_V(\d{4})')
_file_key_reK = re.compile(r'KIC_(\d{10})_SEC_(\d{4})')

#---

# Utility function for model fitting
def _kw_to_Parameter(name, kwarg):
    if isinstance(kwarg, float):
        return Parameter(name=name, value=kwarg, vary=False)
    if isinstance(kwarg, int):
        return Parameter(name=name, value=float(kwarg), vary=False)
    if isinstance(kwarg, list):
        return Parameter(name=name, value=np.median(kwarg), 
                min=min(kwarg), max=max(kwarg))
    if isinstance(kwarg, tuple):
        if len(kwarg) == 2:
            if (min(kwarg) != kwarg[0]) or (max(kwarg) != kwarg[1]):
                raise ValueError('Invalid initial tuple values (max < min')
            return Parameter(name=name, value=np.median(kwarg), 
                             min=kwarg[0], max=kwarg[1])
        elif len(kwarg) == 3:
            if (min(kwarg) != kwarg[0]) or (max(kwarg) != kwarg[2]):
                raise ValueError('Invalid initial tuple values')
            return Parameter(name=name, value=kwarg[1],
                             min=kwarg[0], max=kwarg[2])
        else:
            raise ValueError('Invalid initial tuple length')

    if isinstance(kwarg, UFloat):
        return Parameter(name=name, value=kwarg.n, user_data=kwarg)
    if isinstance(kwarg, Parameter):
        return kwarg
    raise ValueError('Unrecognised type for keyword argument {}'.
        format(name))

#----

def _make_interp(t,x,scale=None):
    if scale == None:
        z = x
    elif np.ptp(x) == 0:
        z = np.zeros_like(x)
    elif scale == 'max':
        z = (x-min(x))/np.ptp(x) 
    elif scale == 'range':
        z = (x-np.median(x))/np.ptp(x)
    else:
        raise ValueError('scale must be None, max or range')
    # Use copy=False to store time and value arrays by reference rather than
    # as copies. 
    return interp1d(t, z, bounds_error=False, fill_value=(z[0],z[-1]),
                    copy=False)
#---

def _glint_func(t, glint_scale, f_theta=None, f_glint=None ):
    return glint_scale * f_glint(f_theta(t))

#---

def _make_trial_params(pos, params, vn):
    # Create a copy of the params object with the parameter values give in
    # list vn replaced with trial values from array pos.
    # Also returns the contribution to the log-likelihood of the parameter
    # values. 
    # Return value is parcopy, lnprior
    # If any of the parameters are out of range, returns None, -inf
    parcopy = params.copy()
    lnprior = 0 
    for i, p in enumerate(vn):
        v = pos[i]
        if (v < parcopy[p].min) or (v > parcopy[p].max):
            return None, -np.inf
        parcopy[p].value = v

    lnprior = _log_prior(parcopy['D'], parcopy['W'], parcopy['b'])
    if not np.isfinite(lnprior):
        return None, -np.inf

    # Also check parameter range here so we catch "derived" parameters
    # that are out of range.
    for p in parcopy:
        v = parcopy[p].value
        if (v < parcopy[p].min) or (v > parcopy[p].max):
            return None, -np.inf
        if np.isnan(v):
            return None, -np.inf
        u = parcopy[p].user_data
        if isinstance(u, UFloat):
            lnprior += -0.5*((u.n - v)/u.s)**2
    if not np.isfinite(lnprior):
        return None, -np.inf

    return parcopy, lnprior

#---

# Prior on (D, W, b) for transit/eclipse fitting.
# This prior assumes uniform priors on cos(i), log(k) and log(aR). The
# factor 2kW is the absolute value of the determinant of the Jacobian, 
# J = d(D, W, b)/d(cosi, k, aR)
def _log_prior(D, W, b):
    if (D < 2e-6) or (D > 0.25): return -np.inf
    k = np.sqrt(D)
    if (b < 0) : return -np.inf
    if (W < 1e-4): return -np.inf
    q = (1+k)**2 - b**2
    if (q < 0): return -np.inf
    aR = np.sqrt(q)/(np.pi*W)
    if (aR < 1): return -np.inf
    return -np.log(2*k*W) - np.log(k) - np.log(aR)

#---

# Target functions for emcee
def _log_posterior_jitter(pos, model, time, flux, flux_err,  params, vn,
        return_fit):

    parcopy, lnprior = _make_trial_params(pos, params, vn)
    if parcopy == None: return -np.inf, -np.inf

    fit = model.eval(parcopy, t=time)
    if return_fit:
        return fit

    if False in np.isfinite(fit):
        return -np.inf, -np.inf

    jitter = np.exp(parcopy['log_sigma'].value)
    s2 =flux_err**2 + jitter**2
    lnlike = -0.5*(np.sum((flux-fit)**2/s2 + np.log(2*np.pi*s2)))
    return lnlike + lnprior, lnlike

#----

def _log_posterior_SHOTerm(pos, model, time, flux, flux_err,  params, vn, 
        return_fit):

    parcopy, lnprior = _make_trial_params(pos, params, vn)
    if parcopy == None: return -np.inf, -np.inf

    fit = model.eval(parcopy, t=time)
    if return_fit:
        return fit

    if False in np.isfinite(fit):
        return -np.inf, -np.inf
    
    resid = flux-fit
    kernel = SHOTerm(
                    S0=np.exp(parcopy['log_S0'].value),
                    Q=np.exp(parcopy['log_Q'].value),
                    w0=np.exp(parcopy['log_omega0'].value))
    gp = GaussianProcess(kernel, mean=0)
    yvar = flux_err**2+np.exp(2*parcopy['log_sigma'].value)
    gp.compute(time, diag=yvar, quiet=True)
    lnlike = gp.log_likelihood(resid)
    return lnlike + lnprior, lnlike
    
#---------------

def _make_labels(plotkeys, bjd_ref, extra_decorr_vectors=None):
    labels = []
    xbf = {} if extra_decorr_vectors==None else extra_decorr_vectors
    for key in plotkeys:
        if key == 'T_0':
            labels.append(r'T$_0-{}$'.format(bjd_ref))
        elif key == 'h_1':
            labels.append(r'$h_1$')
        elif key == 'h_2':
            labels.append(r'$h_2$')
        elif key == 'f_c':
            labels.append(r'$f_c$')
        elif key == 'f_s':
            labels.append(r'$f_s$')
        elif key == 'l_3':
            labels.append(r'$\ell_3$')
        elif key == 'dfdbg':
            labels.append(r'$df\,/\,d{\rm (bg)}$')
        elif key == 'dfdsmear':
            labels.append(r'$df\,/\,d{\rm (smear)}$')
        elif key == 'dfdcontam':
            labels.append(r'$df\,/\,d{\rm (contam)}$')
        elif key == 'ramp':
            labels.append(r'$df\,/\,d\Delta T$')
        elif key == 'dfdx':
            labels.append(r'$df\,/\,dx$')
        elif key == 'd2fdx2':
            labels.append(r'$d^2f\,/\,dx^2$')
        elif key == 'dfdy':
            labels.append(r'$df\,/\,dy$')
        elif key == 'd2fdy2':
            labels.append(r'$d^2f\,/\,dy^2$')
        elif key == 'dfdt':
            labels.append(r'$df\,/\,dt$')
        elif key == 'd2fdt2':
            labels.append(r'$d^2f\,/\,dt^2$')
        elif key == 'dfdsinphi':
            labels.append(r'$df\,/\,d\sin(\phi)$')
        elif key == 'dfdcosphi':
            labels.append(r'$df\,/\,d\cos(\phi)$')
        elif key == 'dfdsin2phi':
            labels.append(r'$df\,/\,d\sin(2\phi)$')
        elif key == 'dfdcos2phi':
            labels.append(r'$df\,/\,d\cos(2\phi)$')
        elif key == 'dfdsin3phi':
            labels.append(r'$df\,/\,d\sin(3\phi)$')
        elif key == 'dfdcos3phi':
            labels.append(r'$df\,/\,d\cos(3\phi)$')
        elif key == 'log_sigma':
            labels.append(r'$\log\sigma$')
        elif key == 'log_omega0':
            labels.append(r'$\log\omega_0$')
        elif key == 'log_S0':
            labels.append(r'$\log{\rm S}_0$')
        elif key == 'log_Q':
            labels.append(r'$\log{\rm Q}$')
        elif key == 'sigma_w':
            labels.append(r'$\sigma_w$ [ppm]')
        elif key == 'logrho':
            labels.append(r'$\log\rho_{\star}$')
        elif key == 'aR':
            labels.append(r'${\rm a}\,/\,{\rm R}_{\star}$')
        elif key == 'sini':
            labels.append(r'\sin i')
        # for an extra basis function 'extra', key will be 'dfdextra'
        elif key[3:] in xbf:
            k = key[3:]
            if 'label' in xbf[k]:
                labels.append(xbf[k]['label'])
            else:
                labels.append(key)
        else:
            labels.append(key)
    return labels

#---------------

class Dataset(object):
    """
    CHEOPS Dataset object

    :param file_key:
    :param force_download:
    :param download_all: If False, download light curves only
    :param configFile:
    :param target:
    :param view_report_on_download: 
    :param metadata: True to load meta data
    :param verbose:

    """

    def __init__(self, file_key, source=None, force_download=False,
                 download_all=True, configFile=None, target=None,
                 verbose=True, metadata=True, view_report_on_download=True):

        if source == None:
            if _file_key_re.search(file_key):
                if file_key[21:] == 'V9193':
                    source = 'PIPE'
                else:
                    source = 'CHEOPS'
            elif _file_key_reP.search(file_key):
                source = 'PIPE'
            elif _file_key_reK.search(file_key):
                source = 'Kepler'
            elif _file_key_reT.search(file_key):
                source = 'TESS'

        if source == 'TESS':
            m = _file_key_reT.search(file_key)
        elif source == 'PIPE':
            m = _file_key_reP.search(file_key)
            if m == None:  # file_key for files from_pipe_file same as CHEOPS
                m = _file_key_re.search(file_key)
        elif source == 'Kepler' or source == 'K2':
            m = _file_key_reK.search(file_key)
        else:
            m = _file_key_re.search(file_key)
        if m == None:
            raise ValueError('Invalid file_key {}'.format(file_key))

        self.source = source
        self.file_key = file_key
        
        l = [int(i) for i in m.groups()]
        try:
            self.progtype,self.prog_id,self.req_id,self.visitctr,self.ver = l
        except:
            try:
                self.ticid, self.sector, self.ver = l
            except:
                self.kicid, self.num = l

        config = load_config(configFile)
        _cache_path = config['DEFAULT']['data_cache_path']
        tgzPath = Path(_cache_path,file_key).with_suffix('.tgz')
        self.tgzfile = str(tgzPath)

        view_report = view_report_on_download
        if tgzPath.is_file() and not force_download:
            if verbose:
                print('Found archive tgzfile',self.tgzfile)
            view_report = False
        else:
            if download_all:
                file_type='all'
            else:
                file_type='lightcurves'
                view_report = False
            # Bodge to avoid logging errors in jupyter notebooks
            with open(os.devnull,'w+') as devnull:
                with redirect_stderr(devnull):
                    Cheops.download(file_type,
                        filters={'file_key':{'contains':[file_key]}},
                        output_directory=str(tgzPath.parent),
                        output_filename=str(tgzPath.name) )

        lisPath = Path(_cache_path,file_key).with_suffix('.lis')
        # The file list can be out-of-date if force_download is used
        if lisPath.is_file() and not force_download:
            self.list = [line.rstrip('\n') for line in open(lisPath)]
        else:
            if verbose: print('Creating dataset file list')
            tar = tarfile.open(self.tgzfile)
            self.list = tar.getnames()
            tar.close()
            with open(str(lisPath), 'w') as fh:  
                fh.writelines("%s\n" % l for l in self.list)

        # Extract light curve data file from .tgz file so we can access the
        # FITS file header information. 
        # V9193 files are generated from PIPE output files and have only one
        # aperture called 'PSF'
        if self.file_key[-5:] == 'V9193':
            aperture = 'PSF'
        else:
            aperture='DEFAULT'
        lcFile = "{}-{}.fits".format(self.file_key,aperture)
        lcPath = Path(self.tgzfile).parent/lcFile
        if lcPath.is_file():
            with fits.open(lcPath) as hdul:
                hdr = hdul[1].header
        else:
            tar = tarfile.open(self.tgzfile)
            s = '(?!\.)(.*_SCI_COR_Lightcurve-{}_V[0-9]{{4}}.fits)'
            r=re.compile(s.format(aperture))
            datafile = list(filter(r.match, self.list))
            if len(datafile) == 0:
                raise Exception('Requested light curve not in this Dataset.')
            if len(datafile) > 1:
                raise Exception('Multiple light curve files in datset')
            with tar.extractfile(datafile[0]) as fd:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UnitsWarning)
                    hdul = fits.open(fd)
                    hdr = hdul[1].header
                    table = Table.read(hdul[1])
                hdul.writeto(lcPath)
            tar.close()
        self.pi_name = hdr['PI_NAME']
        self.obsid = hdr['OBSID']
        if target == None:
            self.target = hdr['TARGNAME']
        else:
            self.target = target
        coords = SkyCoord(hdr['RA_TARG'],hdr['DEC_TARG'],unit='degree,degree')
        self.ra = coords.ra.to_string(precision=2,unit='hour',sep=':',pad=True)
        self.dec = coords.dec.to_string(precision=1,sep=':',unit='degree',
                alwayssign=True,pad=True)
        if  'MAG_V' in hdr:
            self.vmag = hdr['MAG_V'] 
            self.e_vmag = hdr['MAG_VERR'] 
        else:
            self.vmag =None
        if 'MAG_G' in hdr:
            self.gmag = hdr['MAG_G'] 
            self.e_gmag = hdr['MAG_GERR'] 
        else:
            self.gmag =None
        if 'T_EFF' in hdr:
            self.teff = self.teff = hdr['T_EFF']
        self.spectype = hdr['SPECTYPE']
        self.nexp = hdr['NEXP']
        self.exptime = hdr['EXPTIME']
        self.texptime = hdr['TEXPTIME']
        self.pipe_ver = hdr['PIPE_VER']
        if verbose:
            print(' PI name     : {}'.format(self.pi_name))
            print(' OBS ID      : {}'.format(self.obsid))
            print(' Target      : {}'.format(self.target))
            print(' Coordinates : {} {}'.format(self.ra, self.dec))
            print(' Spec. type  : {}'.format(self.spectype))
            if self.vmag is not None:
                print(' V magnitude : {:0.2f} +- {:0.2f}'.
                    format(self.vmag, self.e_vmag))
            if self.gmag is not None:
                print(' G magnitude : {:0.2f} +- {:0.2f}'.
                    format(self.gmag, self.e_gmag))

        if metadata:
            metaFile = "{}-meta.fits".format(self.file_key)
            metaPath = Path(self.tgzfile).parent/metaFile
            if metaPath.is_file():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UnitsWarning)
                    self.metadata = Table.read(metaPath)
            else:
                tar = tarfile.open(self.tgzfile)
                r=re.compile('(?!\.)(.*SCI_RAW_SubArray.*.fits)')
                metafile = list(filter(r.match, self.list))
                if len(metafile) > 1:
                    raise Exception('Multiple metadata files in datset')
                # Load metadata
                if source in ['Kepler','TESS']:
                    pass
                elif len(metafile) == 0:
                    msg = "No metadata in file {}".format(self.tgzfile)
                    warnings.warn(msg)
                else:
                    with tar.extractfile(metafile[0]) as fd:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", UnitsWarning)
                            hdul = fits.open(fd)
                            table = Table.read(hdul,hdu='SCI_RAW_ImageMetadata')
                            table.write(metaPath)
                    tar.close()
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UnitsWarning)
                        self.metadata = Table.read(metaPath)

        if view_report:
            self.view_report(configFile=configFile)
#----

    @classmethod
    def from_pipe_file(self, pipe_file, file_key=None, configFile=None,
                       metadata=True, verbose=True):
        """
        Create a Dataset object from a PIPE output file.

        PIPE is a PSF photomety extraction package for CHEOPS.

        https://pipe-cheops.readthedocs.io/

        If file_key=None (default) then the DACE archive is queried to find
        the file_key value for the observation identification number OBSID
        obtained from the header of the PIPE output file. The version number
        in the file_key is set to "V9193", e.g. CH_PR100001_TG000101_V9193.

        The output is saved in the directory data_cache_path specified in the
        pycheops configuration file. It can subsequently be loaded as a normal
        Dataset object. The aperture name for dataset_get_lightcurve is 'PSF'.
        This is detected automatically by get_lightcurve(), e.g. 

        >>> dataset = Dataset('CH_PR100001_TG000101_V9193').
        >>> time, flux, flux_err = dataset.get_lightcurve()

        :param pipe_file: PIPE output FITS file
        :param file_key: (optional) file_key to use for saving data
        :param configFile: pycheops configuration file
        :param metadata: download metadata
        :param verbose: (optional, default=True) verbose output, none if False

        """

        config = load_config(configFile)
        _cache_path = config['DEFAULT']['data_cache_path']

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UnitsWarning)
            pipedata = Table.read(pipe_file)
        target = pipedata.meta['TARGNAME']

        if file_key == None:
            obs_id =  pipedata.meta['OBSID']
            db = Cheops.query_database(filters={'obs_id':{'equal':[obs_id]}})
            file_key = db['file_key'][0][:-5]+'V9193'

        tgzPath = Path(_cache_path,file_key).with_suffix('.tgz')
        tgzfile = str(tgzPath)
        file_stats = os.stat(pipe_file)
        if metadata:
            dblist = list(Cheops.list_data_product(
                visit_filepath=str(db.get('file_rootpath', [])[0]))['file'])
            _re_meta = re.compile('(.*CH_.*SCI_RAW_SubArray.*.fits)')
            dbmetapath = list(filter(_re_meta.match, dblist))
            Cheops.download_files(files=dbmetapath, file_type='files',
                                  output_filename=tgzfile)
            metaFile = "{}-meta.fits".format(file_key)
            metaPath = Path(_cache_path, metaFile)
            tar = tarfile.open(tgzfile)
            tarmetafile = tar.getnames()[0]
            with tar.extractfile(tarmetafile) as fd:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UnitsWarning)
                    hdul = fits.open(fd)
                    table = Table.read(hdul,hdu='SCI_RAW_ImageMetadata')
                    table.write(metaPath, overwrite=True)
            tar.close()

        lcFile = (file_key[3:20] + '/' + file_key[:-5] +
                  'TU'+pipedata.meta['DATE'].replace(':','-') +
                  '_SCI_COR_Lightcurve-PSF_V9193.fits')
        with tarfile.open(tgzfile, mode='w:gz') as tgz:
            tarinfo = tarfile.TarInfo(name=lcFile)
            tarinfo.size = file_stats.st_size
            with open(pipe_file,'rb') as fp:
                tgz.addfile(tarinfo=tarinfo, fileobj=fp)
            if metadata:
                tarinfo = tarfile.TarInfo(name=tarmetafile)
                tarinfo.size = os.stat(metaPath).st_size
                with open(metaPath,'rb') as fp:
                    tgz.addfile(tarinfo=tarinfo, fileobj=fp)

        return self(file_key=file_key, target=target, metadata=metadata,
                    configFile=configFile, source='PIPE', verbose=verbose)

#----

    @classmethod
    def from_test_data(self, subdir,  target=None, configFile=None, 
            verbose=True):

        config = load_config(configFile)
        _cache_path = config['DEFAULT']['data_cache_path']

        ftp=FTP('obsftp.unige.ch')
        _ = ftp.login()
        wd = "pub/cheops/test_data/{}".format(subdir)
        ftp.cwd(wd)
        filelist = [fl[0] for fl in ftp.mlsd()]
        _re = re.compile(r'(CH_PR\d{6}_TG\d{6}).zip')
        zipfiles = list(filter(_re.match, filelist))
        if len(zipfiles) > 1:
            raise ValueError('More than one dataset in ftp directory')
        if len(zipfiles) == 0:
            raise ValueError('No zip files for datasets in ftp directory')
        zipfile = zipfiles[0]

        file_key = zipfile[:-4]+'_V0000'
        m = _file_key_re.search(file_key)
        l = [int(i) for i in m.groups()]
        self.progtype,self.prog_id,self.req_id,self.visitctr,self.ver = l

        zipPath = Path(_cache_path,zipfile)
        if zipPath.is_file():
            if verbose: print('{} already downloaded'.format(str(zipPath)))
        else:
            cmd = 'RETR {}'.format(zipfile)
            if verbose: print('Downloading {} ...'.format(zipfile))
            ftp.retrbinary(cmd, open(str(zipPath), 'wb').write)

        pdfFile = "{}_DataReduction.pdf".format(file_key)
        pdfPath = Path(_cache_path,pdfFile)
        if pdfPath.is_file():
            if verbose: print('{} already downloaded'.format(pdfFile))
        else:
            _re = re.compile(r'CH_.*RPT_COR_DataReduction.*pdf')
            pdffiles = list(filter(_re.match, filelist))
            if len(pdffiles) > 0: 
                cmd = 'RETR {}'.format(pdffiles[0])
                if verbose: print('Downloading {} ...'.format(pdfFile))
                ftp.retrbinary(cmd, open(str(pdfPath), 'wb').write)
        ftp.quit()
        
        tgzPath = Path(_cache_path,file_key).with_suffix('.tgz')
        tgzfile = str(tgzPath)

        zpf = ZipFile(str(zipPath), mode='r')
        ziplist = zpf.namelist()

        _re_sa = re.compile('(CH_.*SCI_RAW_SubArray_.*.fits)')
        _re_im = re.compile('(CH_.*SCI_RAW_Imagette_.*.fits)')
        _re_lc = re.compile('(CH_.*_SCI_COR_Lightcurve-.*fits)')
        with tarfile.open(tgzfile, mode='w:gz') as tgz:
            subfiles = list(filter(_re_sa.match, ziplist))
            if len(subfiles) > 1:
                raise ValueError('More than one sub-array file in zip file')
            if len(subfiles) == 1:
                if verbose: print("Writing sub-array data to .tgz file...")
                subfile=subfiles[0]
                tarPath = Path('visit')/Path(file_key)/Path(subfile).name 
                tarinfo = tarfile.TarInfo(name=str(tarPath))
                zipinfo = zpf.getinfo(subfile)
                tarinfo.size = zipinfo.file_size
                zf = zpf.open(subfile)
                tgz.addfile(tarinfo=tarinfo, fileobj=zf)
                zf.close()
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

        pdfFile = "{}_DataReduction.pdf".format(file_key)
        pdfPath = Path(_cache_path,pdfFile)
        if pdfPath.is_file():
            if verbose: print('{} already downloaded'.format(pdfFile))
        else:
            _re = re.compile(r'CH_.*RPT_COR_DataReduction.*pdf')
            pdffiles = list(filter(_re.match, filelist))
            if len(pdffiles) > 0:

                cmd = 'RETR {}'.format(pdffiles[0])
                if verbose: print('Downloading {} ...'.format(pdfFile))
                ftp.retrbinary(cmd, open(str(pdfPath), 'wb').write)
        ftp.quit()

        tgzPath = Path(_cache_path,file_key).with_suffix('.tgz')
        tgzfile = str(tgzPath)

        zpf = ZipFile(str(zipPath), mode='r')
        ziplist = zpf.namelist()

        _re_im = re.compile('(CH_.*SCI_RAW_Imagette_.*.fits)')
        _re_lc = re.compile('(CH_.*_SCI_COR_Lightcurve-.*fits)')
        _re_meta = re.compile('(CH_.*SCI_RAW_HkCe-SubArray_.*.fits)')
        with tarfile.open(tgzfile, mode='w:gz') as tgz:
            metafiles = list(filter(_re_meta.match, ziplist))
            if len(metafiles) > 1:
                raise ValueError('More than one metadata file in zip file')
            if len(metafiles) == 1:
                if verbose: print("Writing metadata to .tgz file...")
                metafile=metafiles[0]
                tarPath = Path('visit')/Path(file_key)/Path(metafile).name 
                tarinfo = tarfile.TarInfo(name=str(tarPath))
                zipinfo = zpf.getinfo(metafile)
                tarinfo.size = zipinfo.file_size
                zf = zpf.open(metafile)
                tgz.addfile(tarinfo=tarinfo, fileobj=zf)
                zf.close()
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
    #------

    def save(self, tag="", overwrite=False):
        """
        Save the current Dataset instance as a pickle file

        :param tag: string to tag different versions of the same Dataset

        :param overwrite: set True to overwrite existing version of file

        :returns: pickle file name
        """
        fl = self.target.replace(" ","_")+'_'+tag+'_'+self.file_key+'.dataset'
        if os.path.isfile(fl) and not overwrite:
            msg = f'File {fl} exists. If you mean to replace it then '
            msg += 'use the argument "overwrite=True".'
            raise OSError(msg)
        with open(fl, 'wb') as fp:
            pickle.dump(self, fp, pickle.HIGHEST_PROTOCOL)
        return fl

    #------

    @classmethod
    def load(self, filename):
        """
        Load a dataset from a pickle file

        :param filename: pickle file name

        :returns: dataset object
        
        """
        with open(filename, 'rb') as fp:
            self = pickle.load(fp)
        return self

#----
        
    def get_imagettes(self, verbose=True):
        imFile = "{}-Imagette.fits".format(self.file_key)
        imPath = Path(self.tgzfile).parent / imFile
        if imPath.is_file():
            with fits.open(imPath) as hdul:
                cube = hdul['SCI_RAW_Imagette'].data
                hdr = hdul['SCI_RAW_Imagette'].header
                with warnings.catch_warnings():
                   warnings.simplefilter("ignore", UnitsWarning)
                   meta = Table.read(hdul['SCI_RAW_ImagetteMetadata'])
            if verbose: print ('Imagette data loaded from ',imPath)
        else:
            if verbose: print ('Extracting imagette data from ',self.tgzfile)
            r=re.compile('(?!\.)(.*SCI_RAW_Imagette.*.fits)' )
            datafile = list(filter(r.match, self.list))
            if len(datafile) == 0:
                raise Exception('Dataset does not contains imagette data.')
            if len(datafile) > 1:
                raise Exception('Multiple imagette data files in dataset')
            tar = tarfile.open(self.tgzfile)
            with tar.extractfile(datafile[0]) as fd:
                hdul = fits.open(fd)
                cube = hdul['SCI_RAW_Imagette'].data
                hdr = hdul['SCI_RAW_Imagette'].header
                with warnings.catch_warnings():
                   warnings.simplefilter("ignore", UnitsWarning)
                   meta = Table.read(hdul['SCI_RAW_ImagetteMetadata'])
                hdul.writeto(imPath)
            tar.close()
            if verbose: print('Saved imagette data to ',imPath)

        self.imagettes = (cube, hdr, meta)
        self.imagettes = {'data':cube, 'header':hdr, 'meta':meta}

        return cube

#----

    def get_subarrays(self, verbose=True):
        subFile = "{}-SubArray.fits".format(self.file_key)
        subPath = Path(self.tgzfile).parent / subFile
        if subPath.is_file():
            with fits.open(subPath) as hdul:
                cube = hdul['SCI_COR_SubArray'].data
                with warnings.catch_warnings():
                   warnings.simplefilter("ignore", UnitsWarning)
                   hdr = hdul['SCI_COR_SubArray'].header
                   meta = Table.read(hdul['SCI_COR_ImageMetadata'])
            if verbose: print ('Subarray data loaded from ',subPath)
        else:
            if verbose: print ('Extracting subarray data from ',self.tgzfile)
            r=re.compile('(?!\.)(.*SCI_COR_SubArray.*.fits)' )
            datafile = list(filter(r.match, self.list))
            if len(datafile) == 0:
                r=re.compile('(?!\.)(.*SCI_RAW_SubArray.*.fits)' )
                datafile = list(filter(r.match, self.list))
            if len(datafile) == 0:
                raise Exception('Dataset does not contains subarray data.')
            if len(datafile) > 1:
                raise Exception('Multiple subarray data files in dataset')
            tar = tarfile.open(self.tgzfile)
            with tar.extractfile(datafile[0]) as fd:
                hdul = fits.open(fd)
                if 'SCI_COR_SubArray' in hdul:
                    ext = 'SCI_COR_SubArray'
                    mext = 'SCI_COR_ImageMetadata'
                elif 'SCI_RAW_SubArray' in hdul:
                    ext = 'SCI_RAW_SubArray'
                    mext = 'SCI_RAW_ImageMetadata'
                else:
                    raise KeyError('No SubArray extension in file')
                cube = hdul[ext].data
                hdr = hdul[ext].header
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UnitsWarning)
                    meta = Table.read(hdul[mext])
                    hdul.writeto(subPath)
            tar.close()
            if verbose: print('Saved subarray data to ',subPath)

        self.subarrays = (cube, hdr, meta)
        self.subarrays = {'data':cube, 'header':hdr, 'meta':meta}

        return cube

#----

    def list_apertures(self):
        r=re.compile('.*_SCI_COR_Lightcurve-(.*)_V.*.fits')
        apertures = [r.match(f).group(1) for f in filter(r.match, self.list)]
        apertures.sort()
        return apertures 
#----

    def _get_table_(self, aperture, verbose):
        lcFile = "{}-{}.fits".format(self.file_key, aperture)
        lcPath = Path(self.tgzfile).parent / lcFile
        if lcPath.is_file(): 
            with fits.open(lcPath) as hdul:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', UnitsWarning)
                    table = Table.read(hdul[1])
                    hdr = hdul[1].header
            if verbose: print ('Light curve data loaded from ',lcPath)
        else:
            if verbose: print ('Extracting light curve from ',self.tgzfile)
            tar = tarfile.open(self.tgzfile)
            s = '(?!\.)(.*_SCI_COR_Lightcurve-{}_V[0-9]{{4}}.fits)'
            r=re.compile(s.format(aperture))
            datafile = list(filter(r.match, self.list))
            if len(datafile) == 0:
                raise Exception('Dataset does not contain light curve data.')
            if len(datafile) > 1:
                raise Exception('Multiple light curve files in dataset')
            with tar.extractfile(datafile[0]) as fd:
                hdul = fits.open(fd)
                with warnings.catch_warnings():
                   warnings.simplefilter("ignore", UnitsWarning)
                   table = Table.read(hdul[1])
                hdr = hdul[1].header
                hdul.writeto(lcPath)
            if verbose: print('Saved lc data to ',lcPath)

        return table, hdr

#----

    def get_lightcurve(self, aperture=None, decontaminate=None,
            returnTable=False, reject_highpoints=False, verbose=True):
        """
        Read light curve data for current data set for selected aperture.

        By default, return time, flux and flux_error. Use returnTable=True to
        return the full table of light curve data and metadata.

        Use reject_highpoints=True to remove points to remove positive
        outliers automatically. 

        :param aperture: use dataset.list_apertures() to list options
        :param decontaminate: if True, subtract flux from background stars 
        :param returnTable: 
        :param reject_highpoints: 
        :param verbose:

        :returns: time, flux, flux_err

        The offset of the telescope tube temperature from its nominal value 
        (thermFront_2 + 12) is stored in dataset.lc['deltaT']

        N.B. for PIPE data (aperture='PSF'), only data with FLAG=0 are used.

        """

        if self.source == 'CHEOPS':
            if aperture not in self.list_apertures():
                raise ValueError('Invalid/missing aperture name')
            if decontaminate not in (True, False):
                raise ValueError('Set decontaminate =True or =False')

        elif self.source == 'PIPE':
            aperture = 'PSF'

        elif self.source in ['Kepler', 'TESS']:
            aperture = 'DEFAULT'

        else:
            raise ValueError('Dataset has unknown source attribute value')


        table, hdr = self._get_table_(aperture, verbose)

        if aperture == 'PSF':
            ok = table['FLAG'] == 0
        else:
            ok = (table['EVENT'] == 0) | (table['EVENT'] == 100)
        ok &= table['FLUX'] > 0
        m = np.isnan(table['FLUX'])
        if sum(m) > 0:
            msg = "Light curve contains {} NaN values".format(sum(m))
            warnings.warn(msg)
            ok = ok & ~m
        bjd = np.array(table['BJD_TIME'][ok])
        bjd_ref = int(bjd[0])
        self.bjd_ref = bjd_ref
        time = bjd-bjd_ref
        flux = np.array(table['FLUX'][ok])
        flux_err = np.array(table['FLUXERR'][ok])
        if aperture == 'PSF':
            xc = table['XC'][ok]
            xoff = np.array(xc - np.mean(xc))
            yc = table['YC'][ok]
            yoff = np.array(yc - np.mean(yc))
            roll_angle = np.array(table['ROLL'][ok])
            bg = np.array(table['BG'][ok])
            contam = np.zeros_like(bjd)
            ap_rad = np.nan
        else:
            xc = table['CENTROID_X'][ok]
            yc = table['CENTROID_Y'][ok]
            xoff = np.array(xc - table['LOCATION_X'][ok])
            yoff = np.array(yc - table['LOCATION_Y'][ok])
            roll_angle = np.array(table['ROLL_ANGLE'][ok])
            bg = np.array(table['BACKGROUND'][ok])
            contam = np.array(table['CONTA_LC'][ok])
            ap_rad = hdr['AP_RADI']
        try:
            smear = np.array(table['SMEARING_LC'][ok])
        except:
            smear = np.zeros_like(bjd)
        try:
            deltaT = np.array(self.metadata['thermFront_2'][ok]) + 12
        except:
            deltaT = np.zeros_like(bjd)
        self.bjd_ref = bjd_ref
        self.ap_rad = ap_rad
        self.aperture = aperture
        if verbose:
            print('Time stored relative to BJD = {:0.0f}'.format(bjd_ref))
            if self.aperture == 'PSF':
                print('Photometry from PSF fitting')
            else:
                print('Aperture radius used = {:0.1f} arcsec'.format(ap_rad))
            if 'V_STRT_U' in table.meta:
                print('UTC start: ',table.meta['V_STRT_U'][0:19])
            if 'V_STOP_U' in table.meta:
                print('UTC end:   ',table.meta['V_STOP_U'][0:19])
            duration = (table['MJD_TIME'][-1] - table['MJD_TIME'][0])*86400
            print('Visit duration: {:0.0f} s'.format(duration))
            print('Exposure time: {} x {:0.1f} s'.format(self.nexp,
                self.exptime))
            xloc = np.median(xc)
            yloc = np.median(yc)
            print(f'Target location on CCD: ({xloc:0.1f}, {yloc:0.1f})')
            eff = 100*len(ok)/(1+duration/self.texptime)
            print('Number of non-flagged data points: {}'.format(len(ok)))
            print('Efficiency (non-flagged data): {:0.1f} %'.format(eff))

        if self.source in ['PIPE', 'Kepler','TESS']:
            self.decontaminated = False
            if verbose and decontaminate:
                print('Ignored decontaminate=True for PSF photometry.')
        elif decontaminate:
            flux = flux/(1 + contam) 
            if verbose:
                print('Light curve corrected for flux from background stars')
            self.decontaminated = True
        else:
            if verbose:
                print('Correction for flux from background stars not applied')
            self.decontaminated = False

        if reject_highpoints:
            C_cut = (2*np.nanmedian(flux)-np.nanmin(flux))
            ok  = (flux < C_cut).nonzero()
            time = time[ok]
            flux = flux[ok]
            flux_err = flux_err[ok]
            xoff = xoff[ok]
            yoff = yoff[ok]
            xc = xc[ok]
            yc = yc[ok]
            roll_angle = roll_angle[ok]
            bg = bg[ok]
            contam = contam[ok]
            smear = smear[ok]
            deltaT = deltaT[ok]
            N_cut = len(bjd) - len(time)

        fluxmed = np.nanmedian(flux)
        self.flux_mean = flux.mean()
        self.flux_median = fluxmed
        self.flux_rms = np.std(flux)
        self.flux_mse = np.nanmedian(flux_err)

        if verbose:
            if reject_highpoints:
                print('C_cut = {:0.0f}'.format(C_cut))
                print('N(C > C_cut) = {}'.format(N_cut))
            print('Mean counts = {:0.1f}'.format(self.flux_mean))
            print('Median counts = {:0.1f}'.format(fluxmed))
            print('RMS counts = {:0.1f} [{:0.0f} ppm]'.format(np.nanstd(flux), 
                1e6*np.nanstd(flux)/fluxmed))
            print('Median standard error = {:0.1f} [{:0.0f} ppm]'.format(
                np.nanmedian(flux_err), 1e6*np.nanmedian(flux_err)/fluxmed))
            print('Median background = {:0.0f} e-'.format(np.median(bg)))
            print('Mean contamination = {:0.1f} ppm'.format(1e6*contam.mean()))
            print('Mean smearing correction = {:0.1f} ppm'.
                    format(1e6*smear.mean()/fluxmed))
            if np.max(np.abs(deltaT)) > 0:
                f = interp1d([22.5, 25, 30, 40], [140,200,330,400],
                        bounds_error=False, fill_value='extrapolate')
                ramp =  np.ptp(f(ap_rad)*deltaT)
                print('Predicted amplitude of ramp = {:0.0f} ppm'.format(ramp))

        flux = flux/fluxmed
        flux_err = flux_err/fluxmed
        smear = smear/fluxmed
        bg = bg/fluxmed
        self.lc = {'time':time, 'flux':flux, 'flux_err':flux_err,
                'bjd_ref':bjd_ref, 'table':table, 'header':hdr,
                'xoff':xoff, 'yoff':yoff, 'bg':bg,
                'contam':contam, 'smear':smear, 'deltaT':deltaT,
                'centroid_x':xc, 'centroid_y':yc,
                'roll_angle':roll_angle, 'aperture':aperture}

        if returnTable:
            return table
        else:
            return time, flux, flux_err
        
    def view_report(self, pdf_cmd=None, configFile=None):
        '''
        View the PDF DRP report.

        :param pdf_cmd: command to launch PDF viewer with {} as placeholder for
                        file name.
        '''
        if pdf_cmd == None:
            config = load_config(configFile)
            try:
                pdf_cmd = config['DEFAULT']['pdf_cmd']
            except KeyError:
                raise KeyError("Run pycheops.core.setup_config to set your"
                        " default PDF viewer")

        pdfFile = "{}_DataReduction.pdf".format(self.file_key)
        pdfPath = Path(self.tgzfile).parent/pdfFile
        if not pdfPath.is_file():
            tar = tarfile.open(self.tgzfile)
            r = re.compile('(?!\.)(.*_RPT_COR_DataReduction_.*.pdf)')
            report = list(filter(r.match, self.list))
            if len(report) == 0:
                raise Exception('Dataset does not contain DRP report.')
            if len(report) > 1:
                raise Exception('Multiple reports in datset')
            print('Extracting report from .tgz file ...')
            with tar.extractfile(report[0]) as fin:
                with open(pdfPath,'wb') as fout:
                    for line in fin:
                        fout.write(line)
            tar.close()

        subprocess.run(pdf_cmd.format(pdfPath),shell=True)

#----

    def animate_frames(self, nframes=10, vmin=1., vmax=1., subarray=True,
             imagette=False, grid=False, aperture=None, writer='pillow',
             figsize=(10,10), fontsize=12, linewidth=3):

        if aperture == None:
            aperture = self.ap_rad

        sub_anim, imag_anim = [], []
        for hindex, h in enumerate([subarray, imagette]):
            if h == True:
                if hindex == 0:
                    if type(aperture) == str:
                        title = str(self.target) + " - subarray - " + aperture
                    else:
                        title = str(self.target) + " - subarray - R = " + str(aperture) + " pix"
                    try:
                        frame_cube = self.get_subarrays()[::nframes,:,:]
                        pltlims = np.shape(frame_cube)[2]
                        cen_x = _make_interp(self.lc['table']['MJD_TIME'], self.lc['table']['CENTROID_X']-self.lc['table']['LOCATION_X']+(pltlims/2))(self.subarrays['meta']['MJD_TIME'])[::nframes]
                        cen_y = _make_interp(self.lc['table']['MJD_TIME'], self.lc['table']['CENTROID_Y']-self.lc['table']['LOCATION_Y']+(pltlims/2))(self.subarrays['meta']['MJD_TIME'])[::nframes]
                    except:
                        print("\nNo subarray data.")
                        continue
                if hindex == 1:
                    if type(aperture) == str:
                        title = str(self.target) + " - imagette - " + aperture
                    else:
                        title = str(self.target) + " - imagette - R = " + str(aperture) + " pix"
                    try:
                        frame_cube = self.get_imagettes()[::nframes,:,:]
                        pltlims = np.shape(frame_cube)[2]
                        cen_x = _make_interp(self.lc['table']['MJD_TIME'], self.lc['table']['CENTROID_X']-self.lc['table']['LOCATION_X']+(pltlims/2))(self.imagettes['meta']['MJD_TIME'])[::nframes]
                        cen_y = _make_interp(self.lc['table']['MJD_TIME'], self.lc['table']['CENTROID_Y']-self.lc['table']['LOCATION_Y']+(pltlims/2))(self.imagettes['meta']['MJD_TIME'])[::nframes]
                    except:
                        print("\nNo imagette data.")
                        continue
            else:
                continue

            fig = plt.figure(figsize=figsize)
            plt.rc('font', size=fontsize)
            plt.xlabel("Row (pixel)")
            plt.ylabel("Column (pixel)")
            plt.xlim(-1,pltlims-1)
            plt.ylim(-1,pltlims-1)
            plt.title(title)
            if grid:
                ax = plt.gca()
                ax.grid(color='w', linestyle='-', linewidth=1)


            frames = []
            for i in tqdm(range(len(frame_cube))):
                ax = plt.gca()

                if str(np.amin(frame_cube[i,:,:])) == "nan":
                    img_min = 0
                else:
                    img_min = np.amin(frame_cube[i,:,:])
                if str(np.amax(frame_cube[i,:,:])) == "nan":
                    img_max = 200000
                else:
                    img_max = np.amax(frame_cube[i,:,:])

                image = ax.imshow(frame_cube[i,:,:],
                        norm=colors.Normalize(vmin=vmin*img_min,
                            vmax=vmax*img_max),
                        origin="lower")

                if aperture:
                    xpos,ypos = cen_x[i],cen_y[i]
                    if type(aperture) == int or type(aperture) == float:
                        circle1 = plt.Circle((xpos,ypos), aperture, color='r', lw=linewidth, fill=False, clip_on=True)
                        ax.add_patch(circle1)
                    else:
                        if aperture == "DEFAULT":
                            aprad = 25
                        elif aperture == "RINF":
                            aprad = 22.5
                        elif aperture == "RSUP":
                            aprad = 30
                        elif aperture[:1] == "R":
                            aprad = float(aperture[1:])
                        elif aperture == "OPTIMAL":
                            lcFile = "{}-{}.fits".format(self.file_key,aperture)
                            lcPath = Path(self.tgzfile).parent / lcFile
                            if lcPath.is_file():
                                with fits.open(lcPath) as hdul:
                                    hdr = hdul[1].header
                            else:
                                tar = tarfile.open(self.tgzfile)
                                s = '(?!\.)(.*_SCI_COR_Lightcurve-{}_.*.fits)'
                                r=re.compile(s.format(aperture))
                                datafile = list(filter(r.match, self.list))
                                with tar.extractfile(datafile[0]) as fd:
                                    hdul = fits.open(fd)
                                    hdr = hdul[1].header
                            aprad = hdr['AP_RADI']
                        circle1 = plt.Circle((xpos,ypos), aprad, color='r', lw=linewidth, fill=False, clip_on=True)
                        ax.add_patch(circle1)
                frames.append([image,circle1])

            # Suppress annoying logger warnings from animation module
            logging.getLogger('matplotlib.animation').setLevel(logging.ERROR)
            if hindex == 0:
                sub_anim = animation.ArtistAnimation(fig, frames, blit=True)
                sub_anim.save(title.replace(" ","")+'.gif', writer=writer)
                with open(title.replace(" ","")+'.gif','rb') as file:
                    display(Image(file.read()))
                print("Subarray is saved in the current directory as " +
                        title.replace(" ","")+'.gif')

            elif hindex == 1:
                imag_anim = animation.ArtistAnimation(fig, frames, blit=True)
                imag_anim.save(title.replace(" ","")+'.gif', writer=writer)
                with open(title.replace(" ","")+'.gif','rb') as file:
                    display(Image(file.read()))
                print("Imagette is saved in the current directory as " +
                        title.replace(" ","")+'.gif')

            plt.close()

        if subarray and not imagette:
            return sub_anim
        elif imagette and not subarray:
            return imag_anim
        elif subarray and imagette:
            return sub_anim, imag_anim
         
 #----------------------------------------------------------------------------
 
 # Eclipse and transit fitting

    def __make_extra_basis_funcs__(self, extra_decorr_vectors, time, params):
        # Also adds parameters 'dfd'+(vector key) to params

        if extra_decorr_vectors == None:
            return {}
        
        print('Adding extra decorrelation basis vector functions.')
        extra_basis_funcs = {}
        vectors = extra_decorr_vectors.copy()

        if 't' in vectors:
            # pop 't' so it gets skipped when we loop over parameters
            t = vectors.pop('t') 
            if (min(t) > max(time)) or (max(t) < min(time)):
                raise ValueError('time array for extra basis vectors does'
                                 ' not overlap times in light curve')
        else:
            t = time

        for v in vectors:
            if not 'init' in vectors[v]:
                p_init = (-1, 1)
            else:
                p_init = vectors[v]['init']
            params['dfd'+v] = _kw_to_Parameter('dfd'+v, p_init)

            x = vectors[v]['x']

            if 'fill_value' in vectors[v]:
                fill_value = vectors[v]['fill_value']
            else:
                fill_value = (x[0], x[-1])

            extra_basis_funcs[v] = interp1d(t, x, bounds_error=False,
                            fill_value=fill_value, copy=False)

            print(f'{v:12s}: mean = {x.mean():8.3f}'
                  f', std. dev. = {x.std():8.3f}'
                  f', min = {x.min():8.3f}'
                  f', max = {x.max():8.3f}')

        return extra_basis_funcs

 #----

    def __factor_model__(self, scale, extra_basis_funcs=None):

        time = np.array(self.lc['time'])
        phi = self.lc['roll_angle']*np.pi/180
        # For backwards compatibility
        try:
            smear = self.lc['smear']
        except KeyError:
            smear = np.zeros_like(time)
        try:
            deltaT = self.lc['deltaT']
        except KeyError:
            deltaT = np.zeros_like(time)

        if scale:
            F = FactorModel(
            dx = _make_interp(time, self.lc['xoff'], scale='range'),
            dy = _make_interp(time, self.lc['yoff'], scale='range'),
            sinphi = _make_interp(time,np.sin(phi)),
            cosphi = _make_interp(time,np.cos(phi)),
            bg = _make_interp(time,self.lc['bg'], scale='range'),
            contam = _make_interp(time,self.lc['contam'], scale='range'),
            smear = _make_interp(time,smear, scale='range'),
            deltaT = _make_interp(time,deltaT),
            extra_basis_funcs=extra_basis_funcs)
        else:
            F = FactorModel(
            dx = _make_interp(time, self.lc['xoff']),
            dy = _make_interp(time, self.lc['yoff']),
            sinphi = _make_interp(time,np.sin(phi)),
            cosphi = _make_interp(time,np.cos(phi)),
            bg = _make_interp(time,self.lc['bg']),
            contam = _make_interp(time,self.lc['contam']),
            smear = _make_interp(time,smear),
            deltaT = _make_interp(time,deltaT),
            extra_basis_funcs=extra_basis_funcs)
        return F

    #---

    def lmfit_transit(self, 
            T_0=None, P=None, D=None, W=None, b=None, f_c=None, f_s=None,
            h_1=None, h_2=None, l_3=None, scale=True, 
            c=None, dfdbg=None, dfdcontam=None, dfdsmear=None, ramp=None,
            dfdx=None, dfdy=None, d2fdx2=None, d2fdy2=None,
            dfdsinphi=None, dfdcosphi=None, dfdsin2phi=None, dfdcos2phi=None,
            dfdsin3phi=None, dfdcos3phi=None, dfdt=None, d2fdt2=None, 
            glint_scale=None, logrhoprior=None, extra_decorr_vectors=None,
            t1=None, a1=None, w1=None, f1=None, s1=None,
            t2=None, a2=None, w2=None, f2=None, s2=None, 
            log_sigma=None):
        """
        Fit a transit to the light curve in the current dataset.

        Parameter values can be specified in one of the following ways:

        * fixed value, e.g., P=1.234
        * free parameter with uniform prior interval specified as a 2-tuple,
          e.g., dfdx=(-1,1). The initial value is taken as the the mid-point of
          the allowed interval;
        * free parameter with uniform prior interval and initial value
          specified as a 3-tuple, e.g., (0.1, 0.2, 1);
        * free parameter with a Gaussian prior specified as a ufloat, e.g.,
          ufloat(0,1);
        * as an lmfit Parameter object.

        To enable decorrelation against a parameter, specifiy it as a free
        parameter, e.g., dfdbg=(-1,1).

        If scale=True (default), decorrelation is done against a scaled
        version of the quantities  xoff, yoff, bg, contam and smear with a
        peak-to-peak range of 1. This means the coefficients dfdx, dfdy,
        dfdbg, etc. correspond to the amplitude of the flux variation due to
        the correlation with the relevant parameter.

        Decorrelation against the telescope tube temperature can be included
        using the parameter "ramp" which has units of ppm/degree_C. If
        correct_ramp has been applied then this parameter should have a value
        close to zero (within a few ppm/degree_C). 

        The AIC and BIC values report in the MinimizerResult object returned
        by this method are defined by 
        - AIC = 2*k - 2*lnlike
        - BIC = k*ln(n) - 2*ln(Lmax)
        where 
        - k = number of free parameter
        - n = number of data points
        - Lmax - maximum likelihood

        A fixed value for the logarithm of the additional Gaussian white noise
        in can be added to the flux measurements using the keyword log_sigma.

        Arbitrary basis vectors for decorrelation specified by the user, each
        with its own linear coefficient, can be included in the model using
        the extra_decorr_vectors keyword. Use the keyword extra_decorr_vectors
        to specify these detending basis vectors in the following format ...
     
          extra_decorr_vectors = {'t':t, 'a':{'x':a}, 'b',{'x':b}}

        The times at which the basis vectors are sampled can be specified
        using the key 't'. Times are specfied using the same time scale as 
        dataset.lc['time'], i.e. BJD_TT-dataset.bjd_ref. Each basis vector
        is then provided by the user using a dict with the value of the basis
        function at these times specified as an array-like object provided
        using the key 'x'. If 't' is not provided then the basis functions are
        assumed to be sampled at the same times as dataset.lc['time']. An
        exception is raised if the times specified do not overlap the same
        time range as dataset.lc['time'].

        The array of values provided for each basis vector are used to create
        a linear interpolating function that can be used to evaluate the
        basis function at arbitrary times. By default, the first/last value in
        the array is used to extrapolate to times before/after the input array
        of times. To specify different extrapolated values, use the
        'fill_value' key to specify a value of the fill_value keyword to be
        used in scipy.interpolate.interp1d, e.g. 

          extra_decorr_vectors = { 'a':{'x':a, 'fill_value':0},
                                'b':{'x':b, 'fill_value':np.mean(b)},
                                'c':{'x':c, 'fill_value':'extrapolate'} }

        Summary statistics for each basis vector are printed on initialisation
        of FactorModel. It is advisable to use basis vectors with a mean
        value of 0 and with a range or standard deviation of about 1. It is
        also advisable to avoid basis vectors that are strongly correlated
        with one another or other parameters being using for decorrelation.

        By default, the coefficients for each basis vector are labeled in
        plots using the key prefixed by 'dfd'. Alternative labels can be
        specified using the 'label' key, e.g. 

          extra_decorr_vectors={'x2':{'x':dx**2,
                                'label':'$d^2f/d(\Delta x)^2$'} }

        Initial values and priors for each linear coefficient can be specified
        in the same way as other parameters used in dataset.lmfit_transit() or
        dataset.lmfit_eclipse() using the 'init' key,  e.g. 

          extra_decorr_vectors = { 'a':{'x':a, 'init':(-2,2)},
                                   'b':{'x':b, 'init':ufloat(0,1),
                                   'c':{'x':c, 'init':0} }

        If not specified, the parameter is initialised using (-1, 1), i.e. 
        initial value = 0, min=-1, max=1.

        Up to two spot crossing events during the transit can be modelled
        using a simple polynomial model with the following parameters:

           * t1 - mid-point of spot crossing event 1
           * c1 - contrast factor for spot crossing event 1 (0 <= c1 <= 1)
           * w1 - half-width of spot crossing event 1 (> 0) 
           * f1 - flattening parameter for spot crossing event 1 (0 <= f1 <= 1)
           * s1 - skew parameter for spot crossing event 1 (-1 <= s1 <= 1) 
           * t2 - mid-point of spot crossing event 2
           * c2 - contrast factor for spot crossing event 2 (0 <= c2 <= 1)
           * w2 - half-width of spot crossing event 2 (> 0) 
           * f2 - flattening parameter for spot crossing event 2 (0 <= f2 <= 1)
           * s2 - skew parameter for spot crossing event 2 (-2 <= s2 <= 1) 

        The amplitude of the bump in the light curve for a spot crossing event 
        See pycheops.models.SpotCrossingModel() for more details of this
        model..

        The times of the spot crossing events (t1, t2) are specified using the
        same time scale as dataset.lc['time'], i.e. BJD_TT-dataset.bjd_ref.

        The half-widths of the spot crossing events (w1, w2) are specified in
        units of days.

        If a1 or a2 are not specified then the value 0.001 is used and the
        range of the free parameter is set to (1e-6, 1e-2). 

        N.B. /a1


        If f1 or f2 are not specified then the fixed default value 0.5 is used.

        If s1 or s2 are not specified then the fixed default value 0 is used.


        """

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
            yoff = self.lc['yoff']
            phi = self.lc['roll_angle']*np.pi/180
            bg = self.lc['bg']
            contam = self.lc['contam']
            smear = self.lc['smear']
            deltaT = self.lc['deltaT']
        except AttributeError:
            raise AttributeError("Use get_lightcurve() to load data first.")

        params = Parameters()
        if T_0 == None:
            params.add(name='T_0', value=np.nanmedian(time),
                    min=min(time),max=max(time))
        else:
            params['T_0'] = _kw_to_Parameter('T_0', T_0)
        if P == None:
            params.add(name='P', value=1, vary=False)
        else:
            params['P'] = _kw_to_Parameter('P', P)
        _P = params['P'].value
        if D == None:
            params.add(name='D', value=1-min(flux), min=0,max=0.5)
        else:
            params['D'] = _kw_to_Parameter('D', D)
        k = np.sqrt(params['D'].value)
        if W == None:
            params.add(name='W', value=np.ptp(time)/2/_P,
                    min=np.ptp(time)/len(time)/_P, max=np.ptp(time)/_P) 
        else:
            params['W'] = _kw_to_Parameter('W', W)
        if b == None:
            params.add(name='b', value=0.5, min=0, max=1)
        else:
            params['b'] = _kw_to_Parameter('b', b)
        if f_c == None:
            params.add(name='f_c', value=0, vary=False)
        else:
            params['f_c'] = _kw_to_Parameter('f_c', f_c)
        if f_s == None:
            params.add(name='f_s', value=0, vary=False)
        else:
            params['f_s'] = _kw_to_Parameter('f_s', f_s)
        if l_3 == None:
            params.add(name='l_3', value=0, vary=False)
        else:
            params['l_3'] = _kw_to_Parameter('l_3', l_3)
        if h_1 == None:
            params.add(name='h_1', value=0.7224, vary=False)
        else:
            params['h_1'] = _kw_to_Parameter('h_1', h_1)
        if h_2 == None:
            params.add(name='h_2', value=0.6713, vary=False)
        else:
            params['h_2'] = _kw_to_Parameter('h_2', h_2)
        if c == None:
            params.add(name='c', value=1, min=min(flux)/2,max=2*max(flux))
        else:
            params['c'] = _kw_to_Parameter('c', c)
        # Error message for decorrelation against parameters with 0 range
        zero_range_err = "Decorrelation against parameter with zero range - "
        if dfdbg is not None:
            if np.ptp(bg) == 0:
                raise ValueError(zero_range_err+'bg')
            params['dfdbg'] = _kw_to_Parameter('dfdbg', dfdbg)
        if dfdcontam is not None:
            if np.ptp(contam) == 0:
                raise ValueError(zero_range_err+'contam')
            params['dfdcontam'] = _kw_to_Parameter('dfdcontam', dfdcontam)
        if dfdsmear is not None:
            if np.ptp(smear) == 0:
                raise ValueError(zero_range_err+'smear')
            params['dfdsmear'] = _kw_to_Parameter('dfdsmear', dfdsmear)
        if ramp is not None:
            if np.ptp(deltaT) == 0:
                raise ValueError(zero_range_err+'ramp')
            params['ramp'] = _kw_to_Parameter('ramp', ramp)
        if dfdx is not None:
            if np.ptp(xoff) == 0:
                raise ValueError(zero_range_err+'x')
            params['dfdx'] = _kw_to_Parameter('dfdx', dfdx)
        if dfdy is not None:
            if np.ptp(yoff) == 0:
                raise ValueError(zero_range_err+'y')
            params['dfdy'] = _kw_to_Parameter('dfdy', dfdy)
        if d2fdx2 is not None:
            if np.ptp(xoff) == 0:
                raise ValueError(zero_range_err+'x')
            params['d2fdx2'] = _kw_to_Parameter('d2fdx2', d2fdx2)
        if d2fdy2 is not None:
            if np.ptp(yoff) == 0:
                raise ValueError(zero_range_err+'y')
            params['d2fdy2'] = _kw_to_Parameter('d2fdy2', d2fdy2)
        if dfdt is not None:
            params['dfdt'] = _kw_to_Parameter('dfdt', dfdt)
        if d2fdt2 is not None:
            params['d2fdt2'] = _kw_to_Parameter('d2fdt2', d2fdt2)
        l = [dfdsinphi, dfdcosphi,dfdsin2phi,dfdcos2phi,dfdsin3phi,dfdcos3phi]
        if (l.count(None) < 6) and (np.ptp(phi) == 0):
            raise ValueError(zero_range_err+'phi')
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
        if glint_scale is not None:
            params['glint_scale']=_kw_to_Parameter('glint_scale', glint_scale)

        # Derived parameters
        params.add('k',expr='sqrt(D)',min=0,max=1)
        params.add('aR',expr='sqrt((1+k)**2-b**2)/W/pi',min=1)
        params.add('sini',expr='sqrt(1 - (b/aR)**2)')
        # Avoid use of aR in this expr for logrho - breaks error propogation.
        expr = 'log10(4.3275e-4*((1+k)**2-b**2)**1.5/W**3/P**2)'
        params.add('logrho',expr=expr,min=-9,max=6)
        params['logrho'].user_data=logrhoprior
        params.add('e',min=0,max=1,expr='f_c**2 + f_s**2')
        params.add('q_1',min=0,max=1,expr='(1-h_2)**2')
        params.add('q_2',min=0,max=1,expr='(h_1-h_2)/(1-h_2)')
        # For eccentric orbits only from Winn, arXiv:1001.2010
        if (params['e'].value>0) or params['f_c'].vary or params['f_s'].vary:
            params.add('esinw',expr='sqrt(e)*f_s')
            params.add('ecosw',expr='sqrt(e)*f_c')
            params.add('b_tra',expr='b*(1-e**2)/(1+esinw)')
            params.add('b_occ',expr='b*(1-e**2)/(1-esinw)')
            params.add('T_tot',expr='P*W*sqrt(1-e**2)/(1+esinw)')

        l = ['dfdbg','dfdcontam','dfdsmear','dfdx','dfdy','d2fdx2','d2fdy2']
        if True in [p in l for p in params]:
            self.__scale__ = scale
        else:
            self.__scale__ = None

        self.extra_decorr_vectors = extra_decorr_vectors
        extra_basis_funcs = self.__make_extra_basis_funcs__(
                    extra_decorr_vectors, time, params)
        self.__extra_basis_funcs__ = extra_basis_funcs

        model = TransitModel()*self.__factor_model__(scale, extra_basis_funcs)

        if 'glint_scale' in params.valuesdict().keys():
            try:
                f_theta = self.f_theta
                f_glint = self.f_glint
            except AttributeError:
                raise AttributeError("Use add_glint() to first.")
            model += Model(_glint_func, independent_vars=['t'],
                           f_theta=f_theta, f_glint=f_glint)

        # Additional white noise
        if log_sigma is not None:
            flux_err = np.hypot(flux_err, np.exp(log_sigma))
            params.add(name='log_sigma', value=log_sigma, vary=False)

        result = minimize(_chisq_prior, params, nan_policy='propagate',
                args=(model, time, flux, flux_err))
        self.model = model
        fit = model.eval(result.params,t=time)
        result.bestfit = fit
        result.rms = (flux-fit).std()
        # Move priors out of result.residual into their own object and update
        # result.ndata, result.chisqr, etc.
        npriors = len(result.residual) - len(time)
        if npriors > 0:
            result.prior_residual = result.residual[-npriors:]
            result.residual = result.residual[:-npriors]
            result.npriors = npriors
            result.ndata = len(time)
            result.nfree = result.ndata - result.nvarys
            result.chisqr = np.sum(result.residual**2)
            result.redchi = result.chisqr/(result.ndata-result.nvarys)
        # Renormalize AIC and BIC so they are consistent with emcee values
        lnlike = -0.5*np.sum(result.residual**2 + np.log(2*np.pi*flux_err**2))
        result.lnlike = lnlike
        result.aic = 2*result.nvarys-2*lnlike
        result.bic = result.nvarys*np.log(result.ndata) - 2*lnlike

        self.lmfit = result
        self.__lastfit__ = 'lmfit'
        return result

    # ----------------------------------------------------------------

    def correct_ramp(self, beta=None, plot=False, force=False, 
            figsize=(6,3), fontsize=12):
        """
        Linear correction for ramp effect based on telescope tube temperature.

        A flux ramp is often observed in the beginning of a visit with an
        amplitude of a few hundred ppm (either positive or negative) and
        decaying over a time scale of several hours. This ramp is due to a
        small scale change in the shape of the PSF. This in turn can be
        understood as a slight focus change as a result of a thermal
        adaptation of the telescope tube to the new heat load by the thermal
        radiation from the Earth. This thermal adaptation (*breathing*) is
        monitored by thermal sensors in the tube.

        At the time of writing (Dec 2020) several algorithms are being
        investigated to correct for this ramp effect. One algorithm that is
        simple to implement and seems to work quite well is to correct the
        measured flux using the equation 
           Flux_corrected = Flux_measured (1+beta*deltaT) 
        where deltaT = T_thermFront_2 + 12 
        
        The following values of the coefficient beta have been determined by
        Goran Olofsson.

        | Aperture | beta    |
        |:---------|:--------|
        | 22.5     | 0.00014 |
        | 25.0     | 0.00020 |
        | 30.0     | 0.00033 |
        | 40.0     | 0.00040 |

        This routine uses linear interpolation in this table to predict the
        slope beta for the aperture radius of the light curve.

        :param beta: user-defined value of beta (None to use value from table)
        :param plot: plot flux values before/after correction v. deltaT
        :param force: apply ramp correction even if already corrected

        :returns: time, flux, flux_err

        """
        if hasattr(self,'ramp_correction'):
            if force:
                warnings.warn('Ramp correction already applied')
            else:
                raise Exception('Ramp correction already applied')
        T = self.lc['deltaT']
        flux = self.lc['flux']
        if beta == None:
            f = interp1d([22.5, 25, 30, 40], [0.00014,0.00020,0.00033,0.00040],
                        bounds_error=False, fill_value='extrapolate')
            beta = f(self.ap_rad)
            if (self.ap_rad < 22.5) or (self.ap_rad > 40):
                warnings.warn("Ramp correction extrapolated") 
        fcor = flux * (1+beta*T)
        self.ramp_correction = True
        if plot:
            plt.rc('font', size=fontsize)
            fig,ax=plt.subplots(figsize=figsize)
            ax.plot(T, flux, 'o',c='skyblue',ms=2, label='Measured')
            ax.plot(T, 1+beta*T, c='skyblue')
            ax.plot(T, fcor, 'o',c='midnightblue',ms=2,label='Corrected')
            ax.set_xlabel(r'T$_{\rm thermFront\_2} +12^{\circ}$ C')
            ax.set_ylabel(r'Flux')
            ax.legend()
        self.lc['flux'] = fcor
        return self.lc['time'], self.lc['flux'], self.lc['flux_err']
    

    # ----------------------------------------------------------------
    
    def add_glint(self, nspline=8, mask=None, fit_flux=False,
            moon=False, angle0=None, gapmax=30, 
            show_plot=True, binwidth=15,  figsize=(6,3), fontsize=11):
        """
        Adds a glint model to the current dataset.

        The glint model is a smooth function v. roll angle that can be scaled
        to account for artefacts in the data caused by internal reflections.

        If moon=True the roll angle is measured relative to the apparent
        direction of the Moon, i.e. assume that the glint is due to
        moonlight.

        To use this model, include the the parameter glint_scale in the
        lmfit least-squares fit.

        * nspline - number of splines in the fit
        * mask - fit only data for which mask array is False
        * fit_flux - fit flux rather than residuals from previous fit
        * moon - use roll-angle relative to apparent Moon direction
        * angle0 = dependent variable is (roll angle - angle0)
        * gapmax = parameter to identify large gaps in data - used to
          calculate angle0 of not specified by the user.
        * show_plot - default is to show a plot of the fit
        * binwidth - in degrees for binned points on plot (or None to ignore)
        * figsize  -
        * fontsize -

        Returns the glint function as a function of roll angle/moon angle.

        """
        try:
            time = np.array(self.lc['time'])
            flux = np.array(self.lc['flux'])
            angle = np.array(self.lc['roll_angle'])
        except AttributeError:
            raise AttributeError("Use get_lightcurve() to load data first.")

        if moon:
            bjd = Time(self.bjd_ref+self.lc['time'],format='jd',scale='tdb')
            moon_coo = get_body('moon', bjd)
            target_coo = SkyCoord(self.ra,self.dec,unit=('hour','degree'))
            ra_m = moon_coo.ra.radian
            ra_s = target_coo.ra.radian
            dec_m = moon_coo.dec.radian
            dec_s = target_coo.dec.radian
            v_moon = np. arccos(
                    np.cos(ra_m)*np.cos(dec_m)*np.cos(ra_s)*np.cos(dec_s) +
                    np.sin(ra_m)*np.cos(dec_m)*np.sin(ra_s)*np.cos(dec_s) +
                    np.sin(dec_m)*np.sin(dec_s))
            dv_rot = np.degrees(np.arcsin(np.sin(ra_m-ra_s)*np.cos(dec_m)/
                np.sin(v_moon)))
            angle -= dv_rot
        if fit_flux:
            y = flux - 1
        else:
            l = self.__lastfit__
            fit = self.emcee.bestfit if l == 'emcee' else self.lmfit.bestfit
            y = flux - fit

        if angle0 == None:
            x = np.sort(angle)
            gap = np.hstack((x[0], x[1:]-x[:-1]))
            if max(gap) > gapmax:
                angle0 = x[np.argmax(gap)]
            else:
                angle0 = 0 
        if abs(angle0) < 0.01:
            if moon:
                xlab = r'Moon angle [$^{\circ}$]'
            else:
                xlab = r'Roll angle [$^{\circ}$]'
            xlim = (0,360)
            theta = angle % 360
        else:
            if moon:
                xlab = r'Moon angle - {:0.0f}$^{{\circ}}$'.format(angle0)
            else:
                xlab = r'Roll angle - {:0.0f}$^{{\circ}}$'.format(angle0)
            theta = (360 + angle - angle0) % 360
            xlim = (min(theta),max(theta))

        f_theta = _make_interp(time, theta)

        if mask is not None:
            time = time[~mask]
            theta = theta[~mask]
            y = y[~mask]

        # Copies of data for theta-360 and theta+360 used to make
        # interpolating function periodic
        y = y - np.nanmedian(y)
        y = y[np.argsort(theta)]
        x = np.sort(theta)
        t = np.linspace(min(x),max(x),1+nspline,endpoint=False)[1:]
        x = np.hstack([x-360,x,x+360])
        y = np.hstack([y,y,y])
        t = np.hstack([t-360,t,t+360])
        f_glint = LSQUnivariateSpline(x,y,t,ext='const')

        self.glint_moon = moon
        self.glint_angle0 = angle0
        self.f_theta = f_theta
        self.f_glint = f_glint

        if show_plot:
            plt.rc('font', size=fontsize)
            fig,ax=plt.subplots(nrows=1, figsize=figsize, sharex=True)
            ax.plot(x, y, 'o',c='skyblue',ms=2)
            if binwidth:
                r_, f_, e_, n_ = lcbin(x, y, binwidth=binwidth)
                ax.errorbar(r_,f_,yerr=e_,fmt='o',c='midnightblue',ms=5,
                    capsize=2)
            ax.set_xlim(xlim)
            ylim = np.max(np.abs(y))+0.05*np.ptp(y)
            ax.set_ylim(-ylim,ylim)
            xt = np.linspace(xlim[0],xlim[1],10001)
            yt = f_glint(xt)
            ax.plot(xt, yt, color='saddlebrown')
            ax.set_xlabel(xlab)
            ax.set_ylabel('Glint')

        return f_glint(f_theta(time))

    # ----------------------------------------------------------------

    def lmfit_eclipse(self, 
            T_0=None, P=None, D=None, W=None, b=None, L=None,
            f_c=None, f_s=None, l_3=None, a_c=None, dfdbg=None,
            dfdcontam=None, dfdsmear=None, ramp=None, scale=True, 
            c=None, dfdx=None, dfdy=None, d2fdx2=None, d2fdy2=None,
            dfdsinphi=None, dfdcosphi=None, dfdsin2phi=None, dfdcos2phi=None,
            dfdsin3phi=None, dfdcos3phi=None, dfdt=None, d2fdt2=None,
            glint_scale=None, extra_decorr_vectors=None, log_sigma=None):
        """
        See lmfit_transit for options
        """

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
            yoff = self.lc['yoff']
            phi = self.lc['roll_angle']*np.pi/180
            bg = self.lc['bg']
            contam = self.lc['contam']
            smear = self.lc['smear']
            deltaT = self.lc['deltaT']
        except AttributeError:
            raise AttributeError("Use get_lightcurve() to load data first.")

        params = Parameters()
        if T_0 == None:
            params.add(name='T_0', value=np.nanmedian(time),
                    min=min(time),max=max(time))
        else:
            params['T_0'] = _kw_to_Parameter('T_0', T_0)
        if P == None:
            params.add(name='P', value=1, vary=False)
        else:
            params['P'] = _kw_to_Parameter('P', P)
        _P = params['P'].value
        if D == None:
            params.add(name='D', value=1-min(flux), min=0,max=0.5)
        else:
            params['D'] = _kw_to_Parameter('D', D)
        k = np.sqrt(params['D'].value)
        if W == None:
            params.add(name='W', value=np.ptp(time)/2/_P,
                    min=np.ptp(time)/len(time)/_P, max=np.ptp(time)/_P) 
        else:
            params['W'] = _kw_to_Parameter('W', W)
        if b == None:
            params.add(name='b', value=0.5, min=0, max=1)
        else:
            params['b'] = _kw_to_Parameter('b', b)
        if L == None:
            params.add(name='L', value=0.001, min=0, max=1)
        else:
            params['L'] = _kw_to_Parameter('L', L)
        if f_c == None:
            params.add(name='f_c', value=0, vary=False)
        else:
            params['f_c'] = _kw_to_Parameter('f_c', f_c)
        if f_s == None:
            params.add(name='f_s', value=0, vary=False)
        else:
            params['f_s'] = _kw_to_Parameter('f_s', f_s)
        if l_3 == None:
            params.add(name='l_3', value=0, vary=False)
        else:
            params['l_3'] = _kw_to_Parameter('l_3', l_3)
        if c == None:
            params.add(name='c', value=1, min=min(flux)/2,max=2*max(flux))
        else:
            params['c'] = _kw_to_Parameter('c', c)
        if a_c == None:
            params.add(name='a_c', value=0, vary=False)
        else:
            params['a_c'] = _kw_to_Parameter('a_c', a_c)
        # Error message for decorrelation against parameters with 0 range
        zero_range_err = "Decorrelation against parameter with zero range - "
        if dfdbg is not None:
            if np.ptp(bg) == 0:
                raise ValueError(zero_range_err+'bg')
            params['dfdbg'] = _kw_to_Parameter('dfdbg', dfdbg)
        if dfdcontam is not None:
            if np.ptp(contam) == 0:
                raise ValueError(zero_range_err+'contam')
            params['dfdcontam'] = _kw_to_Parameter('dfdcontam', dfdcontam)
        if dfdsmear is not None:
            if np.ptp(smear) == 0:
                raise ValueError(zero_range_err+'smear')
            params['dfdsmear'] = _kw_to_Parameter('dfdsmear', dfdsmear)
        if ramp is not None:
            if np.ptp(deltaT) == 0:
                raise ValueError(zero_range_err+'ramp')
            params['ramp'] = _kw_to_Parameter('ramp', ramp)
        if dfdx is not None:
            if np.ptp(xoff) == 0:
                raise ValueError(zero_range_err+'x')
            params['dfdx'] = _kw_to_Parameter('dfdx', dfdx)
        if dfdy is not None:
            if np.ptp(yoff) == 0:
                raise ValueError(zero_range_err+'y')
            params['dfdy'] = _kw_to_Parameter('dfdy', dfdy)
        if d2fdx2 is not None:
            if np.ptp(xoff) == 0:
                raise ValueError(zero_range_err+'x')
            params['d2fdx2'] = _kw_to_Parameter('d2fdx2', d2fdx2)
        if d2fdy2 is not None:
            if np.ptp(yoff) == 0:
                raise ValueError(zero_range_err+'y')
            params['d2fdy2'] = _kw_to_Parameter('d2fdy2', d2fdy2)
        if dfdt is not None:
            params['dfdt'] = _kw_to_Parameter('dfdt', dfdt)
        if d2fdt2 is not None:
            params['d2fdt2'] = _kw_to_Parameter('d2fdt2', d2fdt2)
        l = [dfdsinphi, dfdcosphi,dfdsin2phi,dfdcos2phi,dfdsin3phi,dfdcos3phi]
        if (l.count(None) < 6) and (np.ptp(phi) == 0):
            raise ValueError(zero_range_err+'phi')
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
        if glint_scale is not None:
            params['glint_scale']=_kw_to_Parameter('glint_scale', glint_scale)

        # Derived parameters
        params.add('k',expr='sqrt(D)',min=0,max=1)
        params.add('aR',expr='sqrt((1+k)**2-b**2)/W/pi',min=1)
        params.add('sini',expr='sqrt(1 - (b/aR)**2)')
        params.add('e',min=0,max=1,expr='f_c**2 + f_s**2')
        # For eccentric orbits only from Winn, arXiv:1001.2010
        if (params['e'].value>0) or params['f_c'].vary or params['f_s'].vary:
            params.add('esinw',expr='sqrt(e)*f_s')
            params.add('ecosw',expr='sqrt(e)*f_c')
            params.add('b_tra',expr='b*(1-e**2)/(1+esinw)')
            params.add('b_occ',expr='b*(1-e**2)/(1-esinw)')
            params.add('T_tot',expr='P*W*sqrt(1-e**2)/(1-esinw)')

        l = ['dfdbg','dfdcontam','dfdsmear','dfdx','dfdy']
        if True in [p in l for p in params]:
            self.__scale__ = scale
        else:
            self.__scale__ = None

        self.extra_decorr_vectors = extra_decorr_vectors
        extra_basis_funcs = self.__make_extra_basis_funcs__(
                    extra_decorr_vectors, time, params)
        self.__extra_basis_funcs__ = extra_basis_funcs

        model = EclipseModel()*self.__factor_model__(scale, extra_basis_funcs)

        if 'glint_scale' in params.valuesdict().keys():
            try:
                f_theta = self.f_theta
                f_glint = self.f_glint
            except AttributeError:
                raise AttributeError("Use add_glint() to first.")
            model += Model(_glint_func, independent_vars=['t'],
                           f_theta=f_theta, f_glint=f_glint)
        
        # Additional white noise
        if log_sigma is not None:
            flux_err = np.hypot(flux_err, np.exp(log_sigma))
            params.add(name='log_sigma', value=log_sigma, vary=False)

        result = minimize(_chisq_prior, params, nan_policy='propagate',
                args=(model, time, flux, flux_err))
        self.model = model
        fit = model.eval(result.params,t=time)
        result.bestfit = fit
        result.rms = (flux-fit).std()
        # Move priors out of result.residual into their own object and update
        # result.ndata, result.chisqr, etc.
        npriors = len(result.residual) - len(time)
        if npriors > 0:
            result.prior_residual = result.residual[-npriors:]
            result.residual = result.residual[:-npriors]
            result.npriors = npriors
            result.ndata = len(time)
            result.nfree = result.ndata - result.nvarys
            result.chisqr = np.sum(result.residual**2)
            result.redchi = result.chisqr/(result.ndata-result.nvarys)
        # Renormalize AIC and BIC so they are consistent with emcee values
        lnlike = -0.5*np.sum(result.residual**2 + np.log(2*np.pi*flux_err**2))
        result.lnlike = lnlike
        result.aic = 2*result.nvarys-2*lnlike
        result.bic = result.nvarys*np.log(result.ndata) - 2*lnlike

        self.lmfit = result
        self.__lastfit__ = 'lmfit'
        return result

    # ----------------------------------------------------------------

    def lmfit_report(self, **kwargs):
        report = fit_report(self.lmfit, **kwargs)
        rms = self.lmfit.rms*1e6
        s = "    RMS residual       = {:0.1f} ppm\n".format(rms)
        j = report.index('[[Variables]]')
        report = report[:j] + s + report[j:]
        noPriors = True
        params = self.lmfit.params
        parnames = list(params.keys())
        namelen = max([len(n) for n in parnames])
        for p in params:
            u = params[p].user_data
            if isinstance(u, UFloat):
                if noPriors:
                    report+="\n[[Priors]]"
                    noPriors = False
                report += "\n    %s:%s" % (p, ' '*(namelen-len(p)))
                report += '%s +/-%s' % (gformat(u.n), gformat(u.s))
        # Bayes factors
        noBayes  = True
        for p in params:
            u = params[p].user_data
            if (isinstance(u, UFloat) and 
                    (p.startswith('dfd') or p.startswith('d2f') or
                     (p == 'ramp') or (p == 'glint_scale') ) ):
                if noBayes:
                    report+="\n[[Bayes Factors]]  "
                    report+="(values >~1 => free parameter may not be useful)"
                    noBayes = False
                v = params[p].value
                s = params[p].stderr
                if s is not None:
                    B = np.exp(-0.5*((v-u.n)/s)**2) * u.s/s
                    report += "\n    %s:%s" % (p, ' '*(namelen-len(p)))
                    report += ' %12.3f' % (B)

        # Decorrelation parameter scaling
        has_notes = False
        if self.__scale__ is not None:
            has_notes = True
            report += '\n[[Notes]]'
            if self.__scale__:
                report +='\n    Decorrelation parameters were scaled'
            else:
                report +='\n    Decorrelation parameters were not scaled'
        if params['e'].value > 0:
            if not has_notes:
                report += '\n[[Notes]]'
                has_notes = True
            report +='\n    T_tot from Winn, arXiv:1001.2010 is approximate'

        report += '\n[[Software versions]]'
        report += '\n    CHEOPS DRP : %s' % self.pipe_ver
        report += '\n    pycheops   : %s' % __version__
        report += '\n    lmfit      : %s' % _lmfit_version_
        return(report)

    # ----------------------------------------------------------------
    def select_detrend(self, max_bayes_factor=1, exclude=None,
                       keep_original=False, dprior=None, tprior=None,
                       t2prior=None, verbose=True):
        """
        Select choice of detrending model coefficients using Bayes factors

        See Maxted et al. 2022MNRAS.514...77M section 2.7.2 for an explanation
        of how the Bayes factor is calculated for models with/without a given
        decorrelation parameter. As suggested, decorrelation parameters are
        added one-by-one, selecting  the parameter that has the highest Bayes
        factor at each step until no parameters have a Bayes factor >
        max_bayes_factor. To avoid overfitting, if any parameters then have a
        Bayes factor < max_bayes_factor, they are removed one-by-one. 

        A least-squares fit to the light curve using lmfit_transit() or
        lmfit_eclipse() must be run succesfully prior to calling
        select_detrend(). Any detrending parameters included in this prior
        least-squares fit will be included in the dictionary of detrending
        parameters returned by this method, irrespective of their Bayes
        factor.

        Use exclude=[] to specify a list of decorrelation parameters that
        should never be included in the decorrelation model, irrespective of
        their Bayes factors.

        If dprior=None (default) then the priors on all decorrelation 
        parameters apart from dfdt and d2fdt2 are Gaussians with mean of 0 and
        standard deviation equal to the root mean square residual (rms) of the
        prior least-squares fit. Otherwise, the priors on these decorrelation
        parameters are Gaussians with mean of 0 and standard deviation
        specified by the user using this keyword.

        If tprior=None (default) then the prior on dfdt is a Gaussian with
        mean of 0 and standard deviation dprior/ptp(time), where ptp(time) is
        the length of time (in days) covered by the light curve.  Otherwise,
        the prior on this decorrelation parameter is a Gaussian with mean of 0
        and standard deviation specified by the user using this keyword.

        If t2prior=None (default) then the prior on d2fdt2 is a Gaussian with
        mean of 0 and standard deviation dprior/ptp(time)**2. Otherwise, the
        prior on this decorrelation parameter is a Gaussian with mean of 0 and
        standard deviation specified by the user using this keyword.

        If keep_original=False (default), detrending parameters from the last
        least-squares fit will be removed based on the Bayes factor calculated
        with the Gaussian prior specified in that fit for each parameter, if
        present. If no Gaussian prior was specified, dprior, tprior or t2prior
        is used to calculate the Bayes factor, as appropriate.

        N.B. the prior least-squares fit is not affected by running
        dataset.select_detrend(). To overwrite the prior least-squares fit,
        call lmfit_transit() or lmfit_eclipse() including the
        argument "**detrend" in the argument list, where "detrend" in the
        python dict returned by dataset.select_detrend().

        :param max_bayes_factor: Bayes factor limit for selection
        :param exclude: list of coefficients to exclude
        :param dprior: default Gaussian prior (ufloat)
        :param tprior: Gaussian prior for dfdt (ufloat)
        :param t2prior: Gaussian prior for d2fdt2 (ufloat)
        :param keep_original: Do not reject parameters from original lmfit
        :param verbose: set False to suppress printed output

        :returns: python dict of selected detrending coefficients

        Example
        -------

        >>> lmfit0 = dataset.lmfit_transit(P=0.123, T_0=0.654)
        >>> x = ['d2fdt2','d2fdx2','d2fdy2']
        >>> detrend = dataset.select_detrend(exclude=x, max_bayes_factor=0.5)
        >>> lmfit = dataset.lmfit_transit(iP=0.123, T_0=0.654, **detrend)

        """

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
        except AttributeError:
            raise AttributeError("Use get_lightcurve() to load data first.")

        try:
            params = self.lmfit.params.copy()
        except AttributeError:
            raise AttributeError('no valid lmfit result in dataset.')

        if verbose:
            print('Parameter     BF     Delta_BIC RMS(ppm)')

        allpar = ['dfdsinphi', 'dfdsin2phi', 'dfdsin3phi',
                  'dfdcos3phi', 'dfdcosphi', 'dfdcos2phi',
                  'dfdx', 'd2fdx2', 'dfdy', 'd2fdy2',
                  'dfdsmear', 'dfdbg', 'dfdcontam',
                  'dfdt', 'd2fdt2']
        
        if keep_original:
            keep = [p for p in params]
        else:
            keep = []

        if dprior == None:
            dprior = ufloat(0, self.lmfit.rms)
        if tprior == None:
            tprior = dprior/np.ptp(time)
        if t2prior == None:
            t2prior = dprior/np.ptp(time)**2

        user_priors = {}
        for p in params:
            if (p in allpar):
                u = params[p].user_data
                if isinstance(u, UFloat):
                    user_priors[p] = u

        def pprior(p):
            if p in user_priors:
                return user_priors[p]
            elif p == 'dfdt':
                return tprior
            elif p == 'd2fdt2':
                return t2prior
            else:
                return dprior

        detrend = {}
        for p in params:
            if (p in allpar):
                allpar.remove(p)
                detrend[p] = pprior(p)

        if exclude != None:
            for p in exclude:
                if (p in allpar):
                    allpar.remove(p)

        model = self.model
        result0 = minimize(_chisq_prior, params, nan_policy='propagate',
                           args=(model, time, flux, flux_err))

        bestbf = 0
        lastbic = result0.bic
        while bestbf < max_bayes_factor:
            bestbf = np.inf
            for p in allpar:
                partmp = params.copy()
                partmp[p] = Parameter(p,value=0,user_data=pprior(p))
                result = minimize(_chisq_prior, partmp, nan_policy='propagate',
                                  args=(model, time, flux, flux_err))
                v = result.params[p].value
                s = result.params[p].stderr
                if s != None:
                    bf = np.exp(-0.5*((v-pprior(p).n)/s)**2) * pprior(p).s/s
                    if bf < bestbf:
                        bestbf = bf
                        newpar = p

            if bestbf < max_bayes_factor:
                p = newpar
                detrend[p] = pprior(p)
                params[p] = Parameter(p,value=0,user_data=pprior(p))
                result = minimize(_chisq_prior, params, nan_policy='propagate',
                                  args=(model, time, flux, flux_err))
                if verbose:
                    dbic = result.bic - lastbic
                    lastbic = result.bic
                    rms = (flux-model.eval(result.params,t=time)).std()
                    print(f'+{newpar:<12s} {bestbf:6.2f}  {dbic:8.1f}'
                          f' {rms*1e6:8.1f}')
                allpar.remove(newpar)

        worstbf = max_bayes_factor + 1
        while worstbf > max_bayes_factor:
            worstbf = 0
            for p in [p for p in detrend if p not in keep]:
                v = result.params[p].value
                s = result.params[p].stderr
                if s != None:
                    bf = np.exp(-0.5*((v-pprior(p).n)/s)**2) * pprior(p).s/s
                    if bf > worstbf:
                        worstbf = bf
                        delpar = p

            if worstbf > max_bayes_factor:
                del params[delpar]
                del detrend[delpar]
                result = minimize(_chisq_prior, params, nan_policy='propagate',
                                  args=(model, time, flux, flux_err))
                if verbose:
                    dbic = result.bic - lastbic
                    lastbic = result.bic
                    rms = (flux-model.eval(result.params,t=time)).std()
                    print(f'-{delpar:<12s} {bestbf:6.2f}  {dbic:8.1f}'
                          f' {rms*1e6:8.1f}')


        return detrend

    # ----------------------------------------------------------------

    def aperture_scan(self, xy_detrend_fixed=True, data_match=True,
                        verbose=True, return_full=False, ramp=None,
                        extra_decorr_vectors=None, copy_initial=False):
        """
        Repeat lmfit fit to light curve for all available apertures 

        If data_match=True (default), all data that have been removed from the
        light curve in the current dataset are excluded from the fits.

        If ramp=None (default), ramp correction is applied if and only if ramp
        correction has been applied to the light curve in the current dataset.
        Set ramp=False or ramp=True to force ramp correction off or on,
        respectively.

        If xy_detrend_fixed=True (default) then dfdx and dfdy are included in
        the fit to the "FIXED" aperture(s), whether or not they were included
        in the previous fit to the light curve.

        If verbose=True (default), a summary of the results is printed to the
        terminal.

        If copy_initial=False (default) then the initial parameter values are
        taken from the last best-fit values using lmfit_transit() or
        lmfit_eclipse(). If copy_initial=True, the initial parameter values
        will be the same as the initial values from the last call to
        lmfit_transi() or lmfit_eclipse().

        If return_full=True, return a dict that includes the MinimizerResult
        objects for each aperture. Default is False, in which case an astropy
        Table is returned containing a summary of the fits to each aperture.
        N.B. the MinimizerResult object includes any Gaussian priors on
        parameter as part of the data, i.e. n_data = n_obs + n_priors

        The signal-to-noise ratio (SNR) given in the output from this method
        is (depth)/(standard error on depth) for the depth of the eclipse or
        transit, depending on whether the prior least-squares fit the current
        light curve was done using lmfit_transit() or lmfit_eclipse().

        N.B. the fits to the light curves for each aperture will do the
        equivalent of "scale=True", even if the previous least-squares fit to
        the light curve used scale=False. 

        N.B. the existing light curve in the current dataset is not affected
        by running aperture_scan(). Use get_lightcurve() to change the choice
        aperture for the light curve in the current dataset based on the
        results from aperture_scan().

        """

        try:
            bjd0 = self.lc['time'] + self.bjd_ref
        except AttributeError:
            raise AttributeError("Use get_lightcurve() to load data first.")

        if self.source != 'CHEOPS': 
            raise TypeError('aperture_scan only available for CHEOPS data')

        try:
            params = self.lmfit.params.copy()
        except AttributeError:
            raise AttributeError('no valid lmfit result in dataset.')

        aplist = self.list_apertures()
        # Re-order apertures so that they are in radius order with
        # DEFAULT ahead place of R25
        if ('DEFAULT' in aplist) and ('R25' in aplist):
            aplist.remove('DEFAULT')
            aplist.insert(aplist.index('R25'),'DEFAULT')
        if ('RINF' in aplist) and ('R23' in aplist) :
            aplist.remove('RINF')
            aplist.insert(aplist.index('R23'),'RINF')

        # For data matching, interpolate BJD to array index
        i = np.arange(len(bjd0))
        I=interp1d(np.round(bjd0,6),i,bounds_error=False,fill_value=0.5)
            
        def _chisq_prior(params, *args):
            r =  (flux - model.eval(params, t=time))/flux_err
            for p in params:
                u = params[p].user_data
                if isinstance(u, UFloat):
                    r = np.append(r, (u.n - params[p].value)/u.s)
            return r

        results = {}
        rad_var = set([])
        rad_fix = set([])

        if ramp == None:
            do_ramp = hasattr(self,'ramp_correction')
        else:
            do_ramp = ramp
        if do_ramp:
            beta = interp1d([22.5, 25, 30, 40],
                            [0.00014,0.00020,0.00033,0.00040],
                            bounds_error=False, fill_value='extrapolate')

        if verbose:
            hdr = 'Aperture  Type    R[pxl]  rms[ppm]  mad[ppm] chisq/ndf SNR'
            hdr += "      N_data"
            print(hdr)
        for ap in aplist:
            params = self.lmfit.params.copy()
            table, hdr = self._get_table_(ap, False)
            rad = hdr['AP_RADI']
            ap_type = table.meta['AP_TYPE']
            if ap_type == 'Fixed':
                if rad in rad_fix:
                    continue
                rad_fix.add(rad)
                if xy_detrend_fixed:
                    if not 'dfdx' in params:
                        params['dfdx'] = Parameter('dfdx', value=0, vary=True)
                    if not 'dfdy' in params:
                        params['dfdy'] = Parameter('dfdy', value=0, vary=True)
            else:
                if rad in rad_var:
                    continue
                rad_var.add(rad)

            ok = (((table['EVENT'] == 0) | (table['EVENT'] == 100))
                & (table['FLUX']>0) & np.isfinite(table['FLUX']))
            bjd = np.array(table['BJD_TIME'])[ok]
            time = bjd-self.bjd_ref
            flux = np.array(table['FLUX'][ok])
            flux_err = np.array(table['FLUXERR'][ok])
            bg = np.array(table['BACKGROUND'][ok])
            smear = np.array(table['SMEARING_LC'][ok])
            xoff = np.array(table['CENTROID_X'][ok]- table['LOCATION_X'][ok])
            yoff = np.array(table['CENTROID_Y'][ok]- table['LOCATION_Y'][ok])
            phi = np.array(table['ROLL_ANGLE'][ok])*np.pi/180
            contam = np.array(table['CONTA_LC'][ok])
            deltaT = np.array(self.metadata['thermFront_2'][ok]) + 12
            if self.decontaminated:
                flux /= (1 + contam) 

            if data_match:
                j = I(np.round(bjd,6)) % 1 == 0
                time =time[j] 
                flux =flux[j] 
                flux_err =flux_err[j] 
                bg =bg[j] 
                smear =smear[j] 
                xoff =xoff[j] 
                yoff =yoff[j] 
                phi =phi[j] 
                contam =contam[j] 
                deltaT =deltaT[j] 

            fluxmed = np.nanmedian(flux)
            flux = flux/fluxmed
            flux_err = flux_err/fluxmed
            smear = smear/fluxmed
            bg = bg/fluxmed

            if do_ramp:
                flux *= (1+beta(rad)*deltaT)

            if '_transit_func' in self.model.__repr__():
                model = TransitModel()
            else:
                model = EclipseModel()
            model *= FactorModel(
                    dx=_make_interp(time, self.lc['xoff'], scale='range'),
                    dy=_make_interp(time, self.lc['yoff'], scale='range'),
                    sinphi=_make_interp(time,np.sin(phi)),
                    cosphi=_make_interp(time,np.cos(phi)),
                    bg=_make_interp(time,self.lc['bg'], scale='range'),
                    contam=_make_interp(time,self.lc['contam'], scale='range'),
                    smear=_make_interp(time,smear, scale='range'),
                    deltaT=_make_interp(time,deltaT),
                    extra_decorr_vectors=extra_decorr_vectors)

            if hasattr(self,'f_theta'):
                model += Model(_glint_func, independent_vars=['t'],
                               f_theta=self.f_theta, f_glint=self.f_glint)

            if copy_initial:
                for p in params:
                    if params[p].vary:
                        params[p].value = params[p].init_value

            result = minimize(_chisq_prior, params, nan_policy='propagate',
                args=(model, time, flux, flux_err))
            fit = model.eval(result.params,t=time)
            rad = hdr['AP_RADI']
            rms = 1e6*(flux-fit).std()
            mad = 1e6*abs(flux-fit).mean()
            chisq = np.sum((flux-fit)**2/flux_err**2)
            ndf = len(flux)-sum([params[p].vary for p in params])
            chisqr = np.sum((flux-fit)**2/flux_err**2)/ndf
            try:
                if '_transit_func' in self.model.__repr__():
                    snr = result.params['D']/result.params['D'].stderr
                else:
                    snr = result.params['L']/result.params['L'].stderr
            except TypeError:
                snr = np.nan

            if verbose:
                txt = f'{ap:9s} {ap_type:9s} {rad:4.1f} {rms:9.1f} {mad:9.1f}'
                txt += f' {chisqr:9.4f} {snr:8.2f} {len(flux):6d}'
                print(txt)
            results[ap] = {'aperture_radius':rad, 'ap_type':ap_type,
                           'rms':rms, 'mad':mad, 'ndf':ndf, 'chisq':chisq,
                           'snr':snr,'ndata':len(flux)}
            if return_full:
                results[ap]['result'] = result
                results[ap]['time'] = time
                results[ap]['flux'] = flux
                results[ap]['flux_err'] = flux_err

        if return_full:
            return results
        else:
            T = Table()
            keys = list(results.keys())
            T['aperture'] = keys
            for f in ['aperture_radius','rms','mad','ndf','chisq','snr']:
                T[f] = [round(results[k][f],3) for k in keys]
            T['rms'].unit = 'ppm'
            T['mad'].unit = 'ppm'
            return T

    # ----------------------------------------------------------------

    def emcee_sampler(self, params=None,
            steps=128, nwalkers=64, burn=256, thin=1, log_sigma=None, 
            add_shoterm=False, log_omega0=None, log_S0=None, log_Q=None,
            init_scale=1e-2, progress=True, backend=None):
        """
        If you only want to store and yield 1-in-thin samples in the chain, set
        thin to an integer greater than 1. When this is set, thin*steps will be
        made and the chains returned with have "steps" values per walker.

        See https://emcee.readthedocs.io/en/stable/tutorials/monitor/ for use 
        of the backend keyword.

        """

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

        # Make a copy of the lmfit MinimizerResult as a template for the
        # output of this method
        result = deepcopy(self.lmfit)
        result.method ='emcee'
        # Remove components on result not relevant for emcee
        result.status = None
        result.success = None
        result.message = None
        result.ier = None
        result.lmdif_message = None

        if params == None:
            params = self.lmfit.params.copy()
        k = params.valuesdict().keys()
        if add_shoterm:
            if 'log_S0' in k:
                pass
            elif log_S0 == None:
                params.add('log_S0', value=-12,  min=-30, max=0)
            else:
                params['log_S0'] = _kw_to_Parameter('log_S0', log_S0)
            # For time in days, and the default value of Q=1/sqrt(2),
            # log_omega0=8  is a correlation length of about 30s and 
            # -2.3 is about 10 days.
            if 'log_omega0' in k:
                pass
            elif log_omega0 == None:
                params.add('log_omega0', value=3, min=-2.3, max=8)
            else:
                lw0 =  _kw_to_Parameter('log_omega0', log_omega0)
                params['log_omega0'] = lw0
            if 'log_Q' in params:
                pass
            elif log_Q == None:
                params.add('log_Q', value=np.log(1/np.sqrt(2)), vary=False)
            else:
                params['log_Q'] = _kw_to_Parameter('log_Q', log_Q)
            params.add('rho_SHO', expr='2*pi/exp(log_omega0)')
            params.add('tau_SHO', expr='2*exp(log_Q)/exp(log_omega0)')
            params.add('sigma_SHO', expr='sqrt(exp(log_Q+log_S0+log_omega0))')

        if 'log_sigma' in k:
            pass
        elif log_sigma == None:
            if not 'log_sigma' in params:
                params.add('log_sigma', value=-10, min=-16,max=-1)
                params['log_sigma'].stderr = 1
        else:
            params['log_sigma'] = _kw_to_Parameter('log_sigma', log_sigma)
        params.add('sigma_w',expr='exp(log_sigma)*1e6')

        vv, vs, vn = [], [], []
        for p in params:
            if params[p].vary:
                vn.append(p)
                vv.append(params[p].value)
                if params[p].stderr == None:
                    if params[p].user_data == None:
                        vs.append(0.01*(params[p].max-params[p].min))
                    else:
                        vs.append(params[p].user_data.s)
                else:
                    if np.isfinite(params[p].stderr):
                        vs.append(params[p].stderr)
                    else:
                        vs.append(0.01*(params[p].max-params[p].min))

        result.var_names = vn
        result.init_vals = vv
        result.init_values = {}
        for n,v in zip(vn, vv):
            result.init_values[n] = v

        vv = np.array(vv)
        vs = np.array(vs)

        args=(model, time, flux, flux_err,  params, vn)
        p = list(params.keys())
        if 'log_S0' in p and 'log_omega0' in p and 'log_Q' in p :
            log_posterior_func = _log_posterior_SHOTerm
            self.gp = True
        else:
            log_posterior_func = _log_posterior_jitter
            self.gp = False
        return_fit = False
        args += (return_fit, )
    
        # Initialize sampler positions ensuring all walkers produce valid
        # function values (or pos=None if restarting from a backend)
        n_varys = len(vv)
        if backend == None:
            iteration = 0
        else:
            try:
                iteration = backend.iteration
            except OSError:
                iteration = 0
        if iteration > 0:
            pos = None
        else:
            pos = []
            for i in range(nwalkers):
                params_tmp = params.copy()
                lnpost_i = -np.inf
                while lnpost_i == -np.inf:
                    pos_i = vv + vs*np.random.randn(n_varys)*init_scale
                    lnpost_i, lnlike_i = log_posterior_func(pos_i, *args)
                pos.append(pos_i)

        sampler = EnsembleSampler(nwalkers, n_varys, log_posterior_func,
            args=args, backend=backend)
        if progress:
            print('Running burn-in ..')
            stdout.flush()
        pos,_,_,_ = sampler.run_mcmc(pos, burn, store=False, 
            skip_initial_state_check=True, progress=progress)
        sampler.reset()
        if progress:
            print('Running sampler ..')
            stdout.flush()
        state = sampler.run_mcmc(pos, steps, thin_by=thin,
            skip_initial_state_check=True, progress=progress)

        flatchain = sampler.get_chain(flat=True).reshape((-1, len(vn)))
        pos_i = flatchain[np.argmax(sampler.get_log_prob()),:]
        fit = log_posterior_func(pos_i, model, time, flux, flux_err,
                params, vn, return_fit=True)

        # Use scaled resiudals for consistency with lmfit
        result.residual = (flux - fit)/flux_err
        result.bestfit =  fit
        result.chain = flatchain
        # Store median and stanadrd error of PPD in result.params
        # Store best fit in result.parbest
        parbest = params.copy()
        quantiles = np.percentile(flatchain, [15.87, 50, 84.13], axis=0)
        for i, n in enumerate(vn):
            std_l, median, std_u = quantiles[:, i]
            params[n].value = median
            params[n].stderr = 0.5 * (std_u - std_l)
            params[n].correl = {}
            parbest[n].value = pos_i[i]
            parbest[n].stderr = 0.5 * (std_u - std_l)
            parbest[n].correl = {}
        result.params = params
        result.params_best = parbest
        corrcoefs = np.corrcoef(flatchain.T)
        for i, n in enumerate(vn):
            for j, n2 in enumerate(vn):
                if i != j:
                    result.params[n].correl[n2] = corrcoefs[i, j]
                    result.params_best[n].correl[n2] = corrcoefs[i, j]
        result.lnprob = np.copy(sampler.get_log_prob())
        result.errorbars = True
        result.nvarys = n_varys
        af = sampler.acceptance_fraction.mean()
        result.acceptance_fraction = af
        result.nfev = int(thin*nwalkers*steps/af)
        result.thin = thin
        result.ndata = len(time)
        result.nfree = len(time) - n_varys
        result.chisqr = np.sum((flux-fit)**2/flux_err**2)
        result.redchi = result.chisqr/(len(time) - n_varys)
        loglmax = np.max(sampler.get_blobs())
        result.lnlike = loglmax
        result.aic = 2*n_varys - 2*loglmax
        result.bic = np.log(len(time))*n_varys - 2*loglmax
        result.covar = np.cov(flatchain.T)
        result.rms = (flux - fit).std()
        self.emcee = result
        self.sampler = sampler
        self.__lastfit__ = 'emcee'
        return result

    # ----------------------------------------------------------------

    def emcee_report(self, **kwargs):
        report = fit_report(self.emcee, **kwargs)
        rms = self.emcee.rms*1e6
        s = "    RMS residual       = {:0.1f} ppm\n".format(rms)
        j = report.index('[[Variables]]')
        report = report[:j] + s + report[j:]
        noPriors = True
        params = self.emcee.params
        parnames = list(params.keys())
        namelen = max([len(n) for n in parnames])
        for p in params:
            u = params[p].user_data
            if isinstance(u, UFloat):
                if noPriors:
                    report+="\n[[Priors]]"
                    noPriors = False
                report += "\n    %s:%s" % (p, ' '*(namelen-len(p)))
                report += '%s +/-%s' % (gformat(u.n), gformat(u.s))
        
        # Decorrelation parameter scaling
        has_notes = False
        if self.__scale__ is not None:
            has_notes = True
            report += '\n[[Notes]]'
            if self.__scale__:
                report +='\n    Decorrelation parameters were scaled'
            else:
                report +='\n    Decorrelation parameters were not scaled'
        if params['e'].value > 0:
            if not has_notes:
                report += '\n[[Notes]]'
                has_notes = True
            report +='\n    T_tot from Winn, arXiv:1001.2010 is approximate'

        report += '\n[[Software versions]]'
        report += '\n    CHEOPS DRP : %s' % self.pipe_ver
        report += '\n    pycheops   : %s' % __version__
        report += '\n    lmfit      : %s' % _lmfit_version_
        return(report)

    # ----------------------------------------------------------------

    def trail_plot(self, plotkeys=['T_0', 'D', 'W', 'b'],
            width=8, height=1.5):
        """
        Plot parameter values v. step number for each walker.

        These plots are useful for checking the convergence of the sampler.

        The parameters width and height specifiy the size of the subplot for
        each parameter.

        The parameters to be plotted at specified by the keyword plotkeys, or
        plotkeys='all' to plot every jump parameter.

        """

        params = self.emcee.params
        samples = self.sampler.get_chain()

        varkeys = []
        for key in params:
            if params[key].vary:
                varkeys.append(key)

        if plotkeys == 'all':
            plotkeys = varkeys

        n = len(plotkeys)
        fig,ax = plt.subplots(nrows=n, figsize=(width,n*height), sharex=True)
        if n == 1: ax = [ax,]
        labels = _make_labels(plotkeys, self.bjd_ref, self.extra_decorr_vectors)
        for i,key in enumerate(plotkeys):
            ax[i].plot(samples[:,:,varkeys.index(key)],'k',alpha=0.1)
            ax[i].set_ylabel(labels[i])
            ax[i].yaxis.set_label_coords(-0.1, 0.5)
        ax[-1].set_xlim(0, len(samples)-1)
        ax[-1].set_xlabel("step number");

        fig.tight_layout()
        return fig

    # ----------------------------------------------------------------

    def corner_plot(self, plotkeys=['T_0', 'D', 'W', 'b'], 
            show_priors=True, show_ticklabels=False,  kwargs=None):

        params = self.emcee.params

        varkeys = []
        for key in params:
            if params[key].vary:
                varkeys.append(key)

        if plotkeys == 'all':
            plotkeys = varkeys

        chain = self.sampler.get_chain(flat=True)
        xs = []
        for key in plotkeys:
            if key in varkeys:
                xs.append(chain[:,varkeys.index(key)])

            if key == 'sigma_w' and params['log_sigma'].vary:
                xs.append(np.exp(self.emcee.chain[:,-1])*1e6)

            if 'D' in varkeys:
                k = np.sqrt(chain[:,varkeys.index('D')])
            else:
                k = np.sqrt(params['D'].value) # Needed for later calculations

            if key == 'k' and 'D' in varkeys:
                xs.append(k)

            if 'b' in varkeys:
                b = chain[:,varkeys.index('b')]
            else:
                b = params['b'].value  # Needed for later calculations

            if 'W' in varkeys:
                W = chain[:,varkeys.index('W')]
            else:
                W = params['W'].value

            aR = np.sqrt((1+k)**2-b**2)/W/np.pi
            if key == 'aR':
                xs.append(aR)

            sini = np.sqrt(1 - (b/aR)**2)
            if key == 'sini':
                xs.append(sini)

            if 'P' in varkeys:
                P = chain[:,varkeys.index('P')]
            else:
                P = params['P'].value   # Needed for later calculations

            if key == 'logrho':
                logrho = np.log10(4.3275e-4*((1+k)**2-b**2)**1.5/W**3/P**2)
                xs.append(logrho)

        kws = {} if kwargs == None else kwargs

        xs = np.array(xs).T
        labels = _make_labels(plotkeys, self.bjd_ref, self.extra_decorr_vectors)
        figure = corner.corner(xs, labels=labels, **kws)

        nax = len(labels)
        axes = np.array(figure.axes).reshape((nax, nax))
        if not show_ticklabels:
            for i in range(nax):
                ax = axes[-1, i]
                ax.set_xticklabels([])
                ax.set_xlabel(labels[i])
                ax.xaxis.set_label_coords(0.5, -0.1)
            for i in range(1,nax):
                ax = axes[i,0]
                ax.set_yticklabels([])
                ax.set_ylabel(labels[i])
                ax.yaxis.set_label_coords(-0.1, 0.5)

        if show_priors:
            for i, key in enumerate(plotkeys):
                u = params[key].user_data
                if isinstance(u, UFloat):
                    ax = axes[i, i]
                    ax.axvline(u.n - u.s, color="g", linestyle='--')
                    ax.axvline(u.n + u.s, color="g", linestyle='--')
        return figure

    # ------------------------------------------------------------

    def plot_fft(self, star=None, gsmooth=5, logxlim = (1.5,4.5),
            title=None, fontsize=12, figsize=(8,5)):
        """ 
        
        Lomb-Scargle power-spectrum of the residuals. 

        If the previous fit included a GP then this is _not_ included in the
        calculation of the residuals, i.e. the power spectrum includes the
        power "fitted-out" using the GP. The assumption here is that the GP
        has been used to model stellar variability that we wish to
        characterize using the power spectrum. 

        The red vertical dotted lines show the CHEOPS  orbital frequency and
        its first two harmonics.

        If star is a pycheops starproperties object and
        5000 K < star.teff < 7000 K, then the likely range of nu_max is shown
        using green dashed lines.

        The expected power due to white noise is shown as a horizontal dashed 
        gray line. 

        """
        try:
            time = np.array(self.lc['time'])
            flux = np.array(self.lc['flux'])
            flux_err = np.array(self.lc['flux_err'])
        except AttributeError:
            raise AttributeError("Use get_lightcurve() to load data first.")

        try:
            l = self.__lastfit__
        except AttributeError:
            raise AttributeError(
                    "Use lmfit_transit() to get best-fit parameters first.")

        model = self.model

        params = self.emcee.params_best if l == 'emcee' else self.lmfit.params
        res = flux - self.model.eval(params, t=time)

        # print('nu_max = {:0.0f} muHz'.format(nu_max))
        t_s = time*86400*u.second
        y = (1e6*res)*u.dimensionless_unscaled
        ls = LombScargle(t_s, y, normalization='psd')
        frequency, power = ls.autopower()
        p_smooth = convolve(power, Gaussian1DKernel(gsmooth))

        plt.rc('font', size=fontsize)
        fig,ax=plt.subplots(figsize=figsize)
        # Expected white-noise level based on median error bar
        sigma_w = 1e6 * np.nanmedian(flux_err/flux)   # Median error in ppm
        power_w = 1e-6 * sigma_w**2         # ppm^2/micro-Hz
        ax.axhline(power_w, ls='--', c='dimgray')
        ax.loglog(frequency*1e6,power/1e6,c='gray',alpha=0.5)
        ax.loglog(frequency*1e6,p_smooth/1e6,c='darkcyan')
        # nu_max from Campante et al. (2016) eq (20)
        if star is not None:
            if abs(star.teff-6000) < 1000:
                nu_max = 3090 * 10**(star.logg-4.438)*usqrt(star.teff/5777)
                ax.axvline(nu_max.n-nu_max.s,ls='--',c='g')
                ax.axvline(nu_max.n+nu_max.s,ls='--',c='g')
        f_cheops = 1e6/(CHEOPS_ORBIT_MINUTES*60)
        for h in range(1,4):
            ax.axvline(h*f_cheops,ls=':',c='darkred')
        ax.set_xlim(10**logxlim[0],10**logxlim[1])
        ax.set_xlabel(r'Frequency [$\mu$Hz]')
        ax.set_ylabel('Power [ppm$^2$ $\mu$Hz$^{-1}$]');
        ax.set_title(title)
        return fig

    # ------------------------------------------------------------
    
    def plot_lmfit(self, figsize=(6,4), fontsize=11, title=None, 
                   show_model=True, binwidth=0.005, detrend=False,
                   xlim=None):
        """
        Plot the best fit from lmfit_transit / lmfit_eclipse

        """
        try:
            time = np.array(self.lc['time'])
            flux = np.array(self.lc['flux'])
            flux_err = np.array(self.lc['flux_err'])
        except AttributeError:
            raise AttributeError("Use get_lightcurve() to load data first.")
        try:
            model = self.model
        except AttributeError:
            raise AttributeError("Use lmfit_transit() to fit a model first.")
        try:
            params = self.lmfit.params
        except AttributeError:
            raise AttributeError(
                    "Use lmfit_transit() to get best-fit parameters first.")

        res = flux - self.model.eval(params, t=time)
        if xlim is None:
            tmin = np.round(np.min(time)-0.05*np.ptp(time),2)
            tmax = np.round(np.max(time)+0.05*np.ptp(time),2)
        else:
            tmin, tmax = xlim
        tp = np.linspace(tmin, tmax, 10*len(time))
        fp = self.model.eval(params,t=tp)
        glint = '_glint_func' in model.right.name
        if detrend:
            if glint:
                flux -= model.right.eval(params, t=time)  # de-glint
                fp -= model.right.eval(params, t=tp)  # de-glint
                flux /= model.left.right.eval(params, t=time) # de-trend
                fp /= model.left.right.eval(params, t=tp) # de-trend
            else: 
                flux /= model.right.eval(params, t=time) 
                fp /= model.right.eval(params, t=tp) 

        # Transit model only 
        if glint:
            ft = model.left.left.eval(params, t=tp)
        else:
            ft = model.left.eval(params, t=tp)
        if not detrend:
            ft *= params['c'].value

        plt.rc('font', size=fontsize)    
        fig,ax=plt.subplots(nrows=2,sharex=True, figsize=figsize,
                gridspec_kw={'height_ratios':[2,1]})
        ax[0].plot(time,flux,'o',c='skyblue',ms=2,zorder=0)
        ax[0].plot(tp,fp,c='saddlebrown',zorder=2)
        if binwidth:
            t_, f_, e_, n_ = lcbin(time, flux, binwidth=binwidth)
            ax[0].errorbar(t_,f_,yerr=e_,fmt='o',c='midnightblue',ms=5,zorder=2,
                    capsize=2)
        if show_model:
            ax[0].plot(tp,ft,c='forestgreen',zorder=1, lw=2)
        ax[0].set_xlim(tmin, tmax)
        ymin = np.min(flux-flux_err)-0.05*np.ptp(flux)
        ymax = np.max(flux+flux_err)+0.05*np.ptp(flux)
        ax[0].set_ylim(ymin,ymax)
        ax[0].set_title(title)
        if detrend:
            if glint:
                ax[0].set_ylabel('(Flux-glint)/trend')
            else:
                ax[0].set_ylabel('Flux/trend')
        else:
            ax[0].set_ylabel('Flux')
        ax[1].plot(time,res,'o',c='skyblue',ms=2,zorder=0)
        ax[1].plot([tmin,tmax],[0,0],ls=':',c='saddlebrown',zorder=1)
        if binwidth:
            t_, f_, e_, n_ = lcbin(time, res, binwidth=binwidth)
            ax[1].errorbar(t_,f_,yerr=e_,fmt='o',c='midnightblue',ms=5,zorder=2,
                    capsize=2)
        ax[1].set_xlabel('BJD-{}'.format(self.lc['bjd_ref']))
        ax[1].set_ylabel('Residual')
        ylim = np.max(np.abs(res-flux_err)+0.05*np.ptp(res))
        ax[1].set_ylim(-ylim,ylim)
        fig.tight_layout()
        return fig
        
    # ------------------------------------------------------------
    
    def plot_emcee(self, title=None, nsamples=32, detrend=False, 
            binwidth=0.005, show_model=True,  xlim=None, 
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
            parbest = self.emcee.params_best
        except AttributeError:
            raise AttributeError(
                    "Use emcee_transit() or emcee_eclipse() first.")

        res = flux - model.eval(parbest, t=time)
        if xlim is None:
            tmin = np.round(np.min(time)-0.05*np.ptp(time),2)
            tmax = np.round(np.max(time)+0.05*np.ptp(time),2)
        else:
            tmin, tmax = xlim
        tp = np.linspace(tmin, tmax, 10*len(time))
        fp = model.eval(parbest,t=tp)
        glint = '_glint_func' in model.right.name
        flux0 = copy(flux)
        if detrend:
            if glint:
                flux -= model.right.eval(parbest, t=time)  # de-glint
                fp -= model.right.eval(parbest, t=tp)  # de-glint
                flux /= model.left.right.eval(parbest, t=time) # de-trend
                fp /= model.left.right.eval(parbest, t=tp) # de-trend
            else: 
                flux /=  model.right.eval(parbest, t=time) 
                fp /= model.right.eval(parbest, t=tp) 

        # Transit model only 
        if glint:
            ft = model.left.left.eval(parbest, t=tp)
        else:
            ft = model.left.eval(parbest, t=tp)
        if not detrend:
            ft *= parbest['c'].value

        plt.rc('font', size=fontsize)    
        fig,ax=plt.subplots(nrows=2,sharex=True, figsize=figsize,
                gridspec_kw={'height_ratios':[2,1]})

        ax[0].plot(time,flux,'o',c='skyblue',ms=2,zorder=0)
        ax[0].plot(tp,fp,c='saddlebrown',zorder=1)
        if binwidth:
            t_, f_, e_, n_ = lcbin(time, flux, binwidth=binwidth)
            ax[0].errorbar(t_,f_,yerr=e_,fmt='o',c='midnightblue',ms=5,
                    zorder=2, capsize=2)
        if show_model:
            ax[0].plot(tp,ft,c='forestgreen',zorder=1, lw=2)

        nchain = self.emcee.chain.shape[0]
        partmp = parbest.copy()
        if self.gp:
            kernel = SHOTerm(
                    S0=np.exp(parbest['log_S0'].value),
                    Q=np.exp(parbest['log_Q'].value),
                    w0=np.exp(parbest['log_omega0'].value))
            gp = GaussianProcess(kernel, mean=0)
            yvar = flux_err**2+np.exp(2*parbest['log_sigma'].value)
            gp.compute(time, diag=yvar, quiet=True)
            mu0 = gp.predict(res,tp,return_cov=False,return_var=False)
            pp = mu0 + model.eval(parbest,t=tp)
            if detrend:
                if glint:
                    pp -= model.right.eval(parbest, t=tp)  # de-glint
                    pp /= model.left.right.eval(parbest, t=tp) # de-trend
                else: 
                    pp /= model.right.eval(parbest, t=tp) 
                ax[0].plot(tp,pp,c='saddlebrown',zorder=1)
            for i in np.linspace(0,nchain,nsamples,endpoint=False,
                    dtype=int):
                for j, n in enumerate(self.emcee.var_names):
                    partmp[n].value = self.emcee.chain[i,j]
                rr = flux0 - model.eval(partmp, t=time)
                kernel = SHOTerm(
                    S0=np.exp(partmp['log_S0'].value),
                    Q=np.exp(partmp['log_Q'].value),
                    w0=np.exp(partmp['log_omega0'].value))
                gp = GaussianProcess(kernel, mean=0)
                yvar = flux_err**2+np.exp(2*partmp['log_sigma'].value)
                gp.compute(time, diag=yvar, quiet=True)
                mu = gp.predict(rr,tp,return_var=False,return_cov=False)
                pp = mu + model.eval(partmp, t=tp)
                if detrend:
                    if glint:
                        pp -= model.right.eval(partmp, t=tp)  # de-glint
                        pp /= model.left.right.eval(partmp, t=tp) # de-trend
                    else: 
                        pp /= model.right.eval(partmp, t=tp) 
                ax[0].plot(tp,pp,c='saddlebrown',zorder=1,alpha=0.1)
                
        else:
            for i in np.linspace(0,nchain,nsamples,endpoint=False,
                    dtype=int):
                for j, n in enumerate(self.emcee.var_names):
                    partmp[n].value = self.emcee.chain[i,j]
                    fp = model.eval(partmp,t=tp)
                    if detrend:
                        if glint:
                            fp -= model.right.eval(partmp, t=tp)
                            fp /= model.left.right.eval(partmp, t=tp) 
                        else: 
                            fp /= model.right.eval(partmp, t=tp)
                ax[0].plot(tp,fp,c='saddlebrown',zorder=1,alpha=0.1)

        ymin = np.min(flux-flux_err)-0.05*np.ptp(flux)
        ymax = np.max(flux+flux_err)+0.05*np.ptp(flux)
        ax[0].set_xlim(tmin, tmax)
        ax[0].set_ylim(ymin,ymax)
        ax[0].set_title(title)
        if detrend:
            if glint:
                ax[0].set_ylabel('(Flux-glint)/trend')
            else:
                ax[0].set_ylabel('Flux/trend')
        else:
            ax[0].set_ylabel('Flux')
        ax[1].plot(time,res,'o',c='skyblue',ms=2,zorder=0)
        if self.gp:
            ax[1].plot(tp,mu0,c='saddlebrown', zorder=1)
        ax[1].plot([tmin,tmax],[0,0],ls=':',c='saddlebrown', zorder=1)
        if binwidth:
            t_, f_, e_, n_ = lcbin(time, res, binwidth=binwidth)
            ax[1].errorbar(t_,f_,yerr=e_,fmt='o',c='midnightblue',ms=5,zorder=2,
                    capsize=2)
        ax[1].set_xlabel('BJD-{}'.format(self.lc['bjd_ref']))
        ax[1].set_ylabel('Residual')
        ylim = np.max(np.abs(res-flux_err)+0.05*np.ptp(res))
        ax[1].set_ylim(-ylim,ylim)
        fig.tight_layout()
        return fig
        
    # ------------------------------------------------------------

    def massradius(self, m_star=None, r_star=None, K=None, q=0, 
            jovian=True, plot_kws=None, return_samples=False,
            verbose=True):
        '''
        Use the results from the previous emcee/lmfit transit light curve fit
        to estimate the mass and/or radius of the planet.

        Requires that stellar properties are supplied using the keywords
        m_star and/or r_star. If only one parameter is supplied then the other
        is estimated using the stellar density derived from the transit light
        curve analysis. The planet mass can only be estimated if the the
        semi-amplitude of its orbit (in m/s) is supplied using the keyword
        argument K. See pycheops.funcs.massradius for valid formats to specify
        these parameters.

        N.B. by default, the mean stellar density calculated from the light
        curve fit is an uses the approximation q->0, where  q=m_p/m_star is
        the mass ratio. If this approximation is not valid then supply an
        estimate of the mass ratio using the keyword argment q.
        
        Output units are selected using the keyword argument jovian=True
        (Jupiter mass/radius) or jovian=False (Earth mass/radius).

        See pycheops.funcs.massradius for options available using the plot_kws
        keyword argument.
        '''

        # Generate value(s) from previous emcee sampler run
        def _v(p):
            vn = self.emcee.var_names
            chain = self.emcee.chain
            pars = self.emcee.params
            if (p in vn):
                v = chain[:,vn.index(p)]
            elif p in pars.valuesdict().keys():
                v = pars[p].value
            else:
                raise AttributeError(
                        'Parameter {} missing from dataset'.format(p))
            return v
    
        # Generate ufloat  from previous lmfit run 
        def _u(p):
            vn = self.lmfit.var_names
            pars = self.lmfit.params
            if (p in vn):
                u = ufloat(pars[p].value, pars[p].stderr)
            elif p in pars.valuesdict().keys():
                u = pars[p].value
            else:
                raise AttributeError(
                        'Parameter {} missing from dataset'.format(p))
            return u
    
        # Generate a sample of values for a parameter
        def _s(x, nm=100_000):
            if isinstance(x,float) or isinstance(x,int):
                return np.full(nm, x, dtype=float)
            elif isinstance(x, UFloat):
                return np.random.normal(x.n, x.s, nm)
            elif isinstance(x, np.ndarray):
                if len(x) == nm:
                    return x
                elif len(x) > nm:
                    return x[random_sample(range(len(x)), nm)]
                else:
                    return x[(np.random.random(nm)*len(x+1)).astype(int)]
            elif isinstance(x, tuple):
                if len(x) == 2:
                    return np.random.normal(x[0], x[1], nm)
                elif len(x) == 3:
                    raise NotImplementedError
            raise ValueError("Unrecognised type for parameter values")

    
        # If last fit was emcee then generate samples for derived parameters
        # not specified by the user from the chain rather than the summary
        # statistics 
        if self.__lastfit__ == 'emcee':
            k = np.sqrt(_v('D'))
            b = _v('b')
            W = _v('W')
            P = _v('P')
            aR = np.sqrt((1+k)**2-b**2)/W/np.pi
            sini = np.sqrt(1 - (b/aR)**2)
            f_c = _v('f_c')
            f_s = _v('f_s')
            ecc = f_c**2 + f_s**2
            _q = _s(q, len(self.emcee.chain))
            rho_star = rhostar(1/aR,P,_q)
            # N.B. use of np.abs to cope with values with large errors
            if r_star == None and m_star is not None:
                _m = np.abs(_s(m_star, len(self.emcee.chain)))
                r_star = (_m/rho_star)**(1/3)
            if m_star == None and r_star is not None:
                _r = np.abs(_s(r_star, len(self.emcee.chain)))
                m_star = rho_star*_r**3
    
        # If last fit was lmfit then extract parameter values as ufloats or, for
        # fixed parameters, as floats 
        if self.__lastfit__ == 'lmfit':
            k = usqrt(_u('D'))
            b = _u('b')
            W = _u('W')
            P = _u('P')
            aR = usqrt((1+k)**2-b**2)/W/np.pi
            sini = usqrt(1 - (b/aR)**2)
            ecc = _u('e')
            _q = ufloat(q[0], q[1]) if isinstance(q, tuple) else q
            rho_star = rhostar(1/aR, P, _q)
            if r_star == None and m_star is not None:
                if isinstance(m_star, tuple):
                    _m = ufloat(m_star[0], m_star[1])
                else:
                    _m = m_star
                r_star = (_m/rho_star)**(1/3)

        if m_star == None and r_star is not None:
            if isinstance(r_star, tuple):
                _r = ufloat(r_star[0], r_star[1])
            else:
                _r = r_star
            m_star = rho_star*_r**3
        if verbose:
            print('[[Mass/radius]]')
       
        if plot_kws == None:
            plot_kws = {}
       
        return massradius(P=P, k=k, sini=sini, ecc=ecc,
                m_star=m_star, r_star=r_star, K=K, aR=aR,
                jovian=jovian, verbose=verbose,
                return_samples=return_samples, **plot_kws)
    
    # ------------------------------------------------------------

    def bright_star_check(self, vmax=3, sepmax=6):
        """
        Check for bright stars near target
        Only stars from the Bright Star Catalogue, 5th Revised Ed. 
        (Hoffleit+, 1991) are checked.

        vmax   - maximum V magnitude to check
        sepmax - maximum separation in degrees to check

        Return an astropy table with stars from the bright star catalog
        brighter than V magnitude vmax within sepmax degrees from the target

        """


        if vmax > 6.5:
            warnings.warn('Bright star catalogue only complete to V=6.5')
        if sepmax > 24: 
            warnings.warn('No internal reflections for stars > 24 deg away')

        target_coo = SkyCoord(self.ra,self.dec,unit=('hour','degree'))
        catpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                'data','BrightStarCat','bscat.fits')
        T = Table.read(catpath)
        T.remove_column('recno')
        T = T[np.isfinite(T['RAJ2000'])]
        bscat = SkyCoord(T['RAJ2000'], T['DEJ2000'],unit='degree, degree')
        sep = target_coo.separation(bscat)
        T.add_column(sep, name='Separation', index=0)
        T['Separation'].info.format = '7.3f'
        T['SpType'].info.format = '<18s'
        i = (sep.degree < sepmax) & (T['Vmag'] < vmax)
        T = T[i]
        T.sort('Separation')
        return T

    # ------------------------------------------------------------

    def planet_check(self):
        """
        Show target separation from solar system objects at time of observation

        """
        visit_mid_time = Time(np.median(self.lc['table']['MJD_TIME']),
                              format='mjd', scale='utc')
        target_coo = SkyCoord(self.ra,self.dec,unit=('hour','degree'))
        print(f'UTC = {visit_mid_time.isot}')
        print(f'Target coordinates = {target_coo.to_string("hmsdms")}')
        print('Body     R.A.         Declination  Sep(deg)')
        print('-------------------------------------------')
        for p in ('moon','mars','jupiter','saturn','uranus','neptune'):
            c = get_body(p, visit_mid_time)
            ra = c.ra.to_string(precision=2,unit='hour',sep=':',pad=True)
            dec = c.dec.to_string(precision=1,sep=':',unit='degree',
                    alwayssign=True,pad=True)
            sep = c.separation(target_coo).degree
            print(f'{p.capitalize():8s} {ra:12s} {dec:12s} {sep:8.1f}')
        
    
    # ------------------------------------------------------------

    def cds_data_export(self, lcfile="lc.dat",title=None, author=None, 
            authors=None, abstract=None, keywords=None, bibcode=None,
            acknowledgements=None):
        '''
        Save light curve, best fit, etc. to files suitable for CDS upload

        Generates ReadMe file and a data file with the following columns..
        Format Units  Label    Explanations
        F11.6 d       time     Time of mid-exposure (BJD_TT)
        F8.6  ---     flux     Normalized flux 
        F8.6  ---     e_flux   Normalized flux error
        F8.6  ---     flux_d   Normalized flux corrected for instrumental trends
        F8.4  pix     xoff     Target position offset in x-direction
        F8.4  pix     yoff     Target position offset in y-direction
        F8.4  deg     roll     Spacecraft roll angle
        F9.7  ---     contam   Fraction of flux in aperture from nearby stars
        F9.7  ---     smear    Fraction of flux in aperture from readout trails
        F9.7  ---     bg       Fraction of flux in aperture from background
        F6.3  ---     temp_2   thermFront_2 temperature sensor reading

        :param lcfile: output file for upload to CDS
        :param title: title
        :param author: First author
        :param authors: Full author list of the paper
        :param abstract: Abstract of the paper
        :param keywords: list of keywords as in the printed publication
        :param bibcode: Bibliography code for the printed publication
        :param acknowledgements: list of acknowledgements

        See http://cdsarc.u-strasbg.fr/submit/catstd/catstd-3.1.htx for the
        correct formatting of title, keywords, etc.

        The acknowledgements are normally used to give the name and e-mail
        address of the person who generated the table, e.g. 
        "Pierre Maxted, p.maxted(at)keele.ac.uk"

        '''
        try:
            time = np.array(self.lc['time'])
            flux = np.array(self.lc['flux'])
        except AttributeError:
            raise AttributeError("Use get_lightcurve() to load data first.")

        try:
            l = self.__lastfit__
        except AttributeError:
            raise AttributeError(
                    "Use lmfit_transit() to get best-fit parameters first.")

        model = self.model
        params = self.lmfit.params if l == 'lmfit' else self.emcee.params_best
        if  model.right.name == 'Model(_glint_func)':
            flux_d = flux - model.right.eval(params, t=time)  # de-glint
            flux_d /= model.left.right.eval(params, t=time)   # de-trend
        else: 
            flux_d = flux/model.right.eval(params, t=time) 

        tmk = cdspyreadme.CDSTablesMaker()
        tmk.title = title if title is not None else ""
        tmk.author = author if author is not None else ""
        tmk.authors = authors if author is not None else ""
        tmk.abstract = abstract if abstract is not None else ""
        tmk.keywords = keywords if keywords is not None else ""
        tmk.bibcode = bibcode if bibcode is not None else ""
        tmk.date = Time.now().value.year

        T=Table()
        T['time'] = time + self.lc['bjd_ref']
        T['time'].info.format = '16.6f'
        T['time'].description = 'Time of mid-exposure'
        T['time'].units = u.day
        T['flux'] = flux
        T['flux'].info.format = '8.6f'
        T['flux'].description = 'Normalized flux'
        T['e_flux'] = self.lc['flux_err']
        T['e_flux'].info.format = '8.6f'
        T['e_flux'].description = 'Normalized flux error'
        T['flux_d'] = flux_d
        T['flux_d'].info.format = '8.6f'
        T['flux_d'].description = 'Normalized flux corrected for instrumental trends'
        T['xoff'] = self.lc['xoff']
        T['xoff'].info.format = '8.4f'
        T['xoff'].description = "Target position offset in x-direction"
        T['yoff'] = self.lc['yoff']
        T['yoff'].info.format = '8.4f'
        T['yoff'].description = "Target position offset in y-direction"
        T['roll'] = self.lc['roll_angle']
        T['roll'].info.format = '8.4f'
        T['roll'].description = "Spacecraft roll angle"
        T['roll'].units = u.degree
        T['contam'] = self.lc['contam']
        T['contam'].info.format = '9.7f'
        T['contam'].description = "Fraction of flux in aperture from nearby stars"
        if np.ptp(self.lc['smear']) > 0:
            T['smear'] = self.lc['smear']
            T['smear'].info.format = '9.7f'
            T['smear'].description = "Fraction of flux in aperture from readout trails"
        T['bg'] = self.lc['bg']
        T['bg'].info.format = '9.7f'
        T['bg'].description = "Fraction of flux in aperture from background"
        if np.ptp(self.lc['deltaT']) > 0:
            T['temp_2'] = self.lc['deltaT'] - 12
            T['temp_2'].info.format = '6.3f'
            T['temp_2'].description = "thermFront_2 temperature sensor reading"
            T['temp_2'].units = u.Celsius

        table = tmk.addTable(T, lcfile,
                description=f"CHEOPS photometry of {self.target}")
        # Set output format
        for p in T.colnames:
            c=table.get_column(p)
            c.set_format(f'F{T[p].format[:-1]}')
        # Units
        c=table.get_column('time'); c.unit = 'd'
        c=table.get_column('xoff'); c.unit = 'pix'
        c=table.get_column('yoff'); c.unit = 'pix'
        c=table.get_column('roll'); c.unit = 'deg'
        tmk.writeCDSTables()

        templatename = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                'data','cdspyreadme','ReadMe.template')
        coo = SkyCoord(
                self.lc['header']['RA_TARG'],
                self.lc['header']['DEC_TARG'],unit='deg')
        rastr = coo.ra.to_string(unit='hour',sep=' ',precision=1, pad=True)
        destr = coo.dec.to_string(unit='deg',sep=' ',precision=0, 
                alwayssign=True, pad=True)
        desc = (indent(fill(
            f'Photometry of {self.target} generated from CHEOPS archive '+
            f'files with file key {self.file_key} using pycheops version '+
            f'{__version__}.', width=78),'  ') + 
            f'\n  Aperture radius = {self.ap_rad} pixels.'+
            f'\n  Exposure time: {self.nexp} x {self.exptime:0.1f} s')
        templateValue = {
                'object':f'{rastr} {destr}   {self.target}',
                'description':desc,
                'acknowledgements':acknowledgements
                }
        tmk.setReadmeTemplate(templatename, templateValue)
        with open("ReadMe", "w") as fd:
            tmk.makeReadMe(out=fd)
    
    # ------------------------------------------------------------

    def rollangle_plot(self, binwidth=15, figsize=None, fontsize=11,
            title=None):
        '''
        Plot of residuals from last fit v. roll angle

        The upper panel shows the fit to the glint and/or trends v. roll angle

        The lower panel shows the residuals from the best fit.

        If a glint correction v. moon angle has been applied, this is shown in
        the middle panel.
        
        '''

        try:
            flux = np.array(self.lc['flux'])
            time = np.array(self.lc['time'])
            angle = np.array(self.lc['roll_angle'])
        except AttributeError:
            raise AttributeError("Use get_lightcurve() to load data first.")

        try:
            l = self.__lastfit__
        except AttributeError:
            raise AttributeError(
                    "Use lmfit_transit() to get best-fit parameters first.")

        # Residuals from last fit and trends due to glint and roll angle
        fit = self.emcee.bestfit if l == 'emcee' else self.lmfit.bestfit
        res = flux - fit
        params = self.emcee.params_best if l == 'emcee' else self.lmfit.params
        rolltrend = np.zeros_like(angle)
        glint = np.zeros_like(angle)
        phi = angle*np.pi/180           # radians for calculation

        # Grid of angle values for plotting smooth version of trends
        tang = np.linspace(0,360,3600)  # degrees
        tphi = tang*np.pi/180           # radians for calculation
        tr = np.zeros_like(tang)        # roll angle trend
        tg = np.zeros_like(tang)        # glint

        vd = params.valuesdict()
        vk = vd.keys()
        notrend = True
        noglint = True
        # Roll angle trend
        for n in range(1,4):
            p = "dfdsinphi" if n==1 else "dfdsin{}phi".format(n)
            if p in vk:
                notrend = False
                rolltrend += vd[p] * np.sin(n*phi)
                tr += vd[p] * np.sin(n*tphi)
            p = "dfdcosphi" if n==1 else "dfdcos{}phi".format(n)
            if p in vk:
                notrend = False
                rolltrend += vd[p] * np.cos(n*phi)
                tr += vd[p] * np.cos(n*tphi)

        if 'glint_scale' in vk:
            notrend = False
            if self.glint_moon:
                glint_theta = self.f_theta(time)
                glint = vd['glint_scale']*self.f_glint(glint_theta)
                tg = vd['glint_scale']*self.f_glint(tang)
                noglint = False
            else:
                glint_theta = (360 + angle - self.glint_angle0) % 360
                glint = vd['glint_scale']*self.f_glint(glint_theta)
                gt = (360 + tang - self.glint_angle0) % 360
                tg = vd['glint_scale']*self.f_glint(gt)

        plt.rc('font', size=fontsize)
        if notrend:
            figsize = (9,4) if figsize == None else figsize
            fig,ax=plt.subplots(nrows=1, figsize=figsize, sharex=True)
            ax.plot(angle, res, 'o',c='skyblue',ms=2)
            if binwidth:
                r_, f_, e_, n_ = lcbin(angle, res, binwidth=binwidth)
                ax.errorbar(r_,f_,yerr=e_,fmt='o',c='midnightblue',ms=5,
                    capsize=2)
            ax.set_xlim(0, 360)
            ylim = np.max(np.abs(res))+0.05*np.ptp(res)
            ax.set_ylim(-ylim,ylim)
            ax.axhline(0, color='saddlebrown',ls=':')
            ax.set_xlabel(r'Roll angle [$^{\circ}$]')
            ax.set_ylabel('Residual')
            ax.set_title(title)

        elif 'glint_scale' in vk and self.glint_moon:
            figsize = (9,8) if figsize == None else figsize
            fig,ax=plt.subplots(nrows=3, figsize=figsize)
            y = res + rolltrend 
            ax[0].plot(angle, y, 'o',c='skyblue',ms=2)
            ax[0].plot(tang, tr, c='saddlebrown')
            if binwidth:
                r_, f_, e_, n_ = lcbin(angle, y, binwidth=binwidth)
                ax[0].errorbar(r_,f_,yerr=e_,fmt='o',c='midnightblue',ms=5,
                    capsize=2)
            ax[0].set_xlabel(r'Roll angle [$^{\circ}$] (Sky)')
            ax[0].set_ylabel('Roll angle trend')
            ylim = np.max(np.abs(y))+0.05*np.ptp(y)
            ax[0].set_xlim(0, 360)
            ax[0].set_ylim(-ylim,ylim)
            ax[0].set_title(title)

            y = res + glint
            ax[1].plot(glint_theta, y, 'o',c='skyblue',ms=2)
            ax[1].plot(tang, tg, c='saddlebrown')
            if binwidth:
                r_, f_, e_, n_ = lcbin(glint_theta, y, binwidth=binwidth)
                ax[1].errorbar(r_,f_,yerr=e_,fmt='o',c='midnightblue',ms=5,
                    capsize=2)
            ylim = np.max(np.abs(y))+0.05*np.ptp(y)
            ax[1].set_xlim(0, 360)
            ax[1].set_ylim(-ylim,ylim)
            ax[1].set_xlabel(r'Roll angle [$^{\circ}$] (Moon)')
            ax[1].set_ylabel('Moon glint')

            ax[2].plot(angle, res, 'o',c='skyblue',ms=2)
            if binwidth:
                r_, f_, e_, n_ = lcbin(angle, res, binwidth=binwidth)
                ax[2].errorbar(r_,f_,yerr=e_,fmt='o',c='midnightblue',ms=5,
                    capsize=2)
            ax[2].axhline(0, color='saddlebrown',ls=':')
            ax[2].set_xlim(0, 360)
            ylim = np.max(np.abs(res))+0.05*np.ptp(res)
            ax[2].set_ylim(-ylim,ylim)
            ax[2].set_xlabel(r'Roll angle [$^{\circ}$] (Sky)')
            ax[2].set_ylabel('Residuals')

        else:

            figsize = (8,6) if figsize == None else figsize
            fig,ax=plt.subplots(nrows=2, figsize=figsize, sharex=True)
            y = res + rolltrend + glint 
            ax[0].plot(angle, y, 'o',c='skyblue',ms=2)
            ax[0].plot(tang, tr+tg, c='saddlebrown')
            if binwidth:
                r_, f_, e_, n_ = lcbin(angle, y, binwidth=binwidth)
                ax[0].errorbar(r_,f_,yerr=e_,fmt='o',c='midnightblue',ms=5,
                    capsize=2)
            if noglint:
                ax[0].set_ylabel('Roll angle trend')
            else:
                ax[0].set_ylabel('Roll angle trend + glint')
            ylim = np.max(np.abs(y))+0.05*np.ptp(y)
            ax[0].set_ylim(-ylim,ylim)
            ax[0].set_title(title)

            ax[1].plot(angle, res, 'o',c='skyblue',ms=2)
            if binwidth:
                r_, f_, e_, n_ = lcbin(angle, res, binwidth=binwidth)
                ax[1].errorbar(r_,f_,yerr=e_,fmt='o',c='midnightblue',ms=5,
                    capsize=2)
            ax[1].axhline(0, color='saddlebrown',ls=':')
            ax[1].set_xlim(0, 360)
            ylim = np.max(np.abs(res))+0.05*np.ptp(res)
            ax[1].set_ylim(-ylim,ylim)
            ax[1].set_xlabel(r'Roll angle [$^{\circ}$]')
            ax[1].set_ylabel('Residuals')
        fig.tight_layout()
        return fig
    
# ------------------------------------------------------------
    
# Data display and diagnostics

    def transit_noise_plot(self, width=3, steps=500,
            fname=None, figsize=(6,4), fontsize=11, return_values=False,
            requirement=None, local=False, verbose=True):
        """
        Transit noise plot

        fname: to specify an output file for the plot
        return_values: return a dictionary of statistics - noise in ppm

        """

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
            ax[1].axhline(requirement, color='darkcyan',ls=':')
        fig.tight_layout()
        if fname == None:
            plt.show()
        else:
            plt.savefig(fname)

        if return_values:
            d = {}
            d['Scaled noise, mean noise'] = Nsc.mean()
            d['Scaled noise, min. noise'] = Nsc.min()
            d['Scaled noise, max. noise'] = Nsc.max()
            d['Scaled noise, mean scaling factor'] = Fsc.mean()
            d['Scaled noise, min. scaling factor'] = Fsc.min()
            d['Scaled noise, max. scaling factor'] = Fsc.max()
            d['Minimum error, mean noise'] = Nmn.mean()
            d['Minimum error, min. noise'] = Nmn.min()
            d['Minimum error, max. noise'] = Nmn.max()
            return d
        
    #------

    def decontaminate(self, Gmag=None, count_rate=None, verbose=True,
            configFile=None):
        """
        Correction to light curve for contamination by nearby stars.

        The parameter count_rate is used to pass the assumed values of the
        target counts per exposure relative to 10**(-0.4*Gmag), i.e. assuming
        that a star with G-band magnitude Gmag has a count_rate value of 1.
        Must have the same number of elements as the observed lightcurve
        currently stored in dataset.lc. Set elements of count_rate to np.nan
        to exclude observations from the calculation of the zero point
        calculation.

        :param Gmag: default is to use value from FITS keyword MAG_G
        :param count_rate: Normalised count rate values for light curve
        :param verbose: 
        :param configFile:

        :returns: time, flux, flux_err

        """

        if self.decontaminated:
            raise Exception('Decontamination correction already applied.')

        time = self.lc['time']
        flux = self.lc['flux']
        flux_err = self.lc['flux_err']
        contam = self.lc['contam']

        config = load_config(configFile)
        psf_file = config['psf_file']['psf_file']
        psf_x0 =  config['psf_file']['x0']
        psf_y0 =  config['psf_file']['y0']
        here = os.path.abspath(os.path.dirname(__file__))
        data_path = os.path.join(here,'data','instrument')
        try:
            psf_path = os.path.join(data_path, psf_file)
        except KeyError:
            raise KeyError("Run pycheops.core.setup_config(overwrite=True) to"
                           " update your config file.")
        with open(psf_path) as fp:
            psf = [[float(digit) for digit in line.split()] for line in fp]
        position0 = [psf_x0, psf_y0]
        aperture0 = CircularAperture(position0, r=self.ap_rad)
        photTable0 = aperture_photometry(psf, aperture0)
        target_flux = photTable0['aperture_sum'][0]
        flx_frac = target_flux/np.sum(psf)

        if Gmag == None:
            Gmag = self.lc['table'].meta['MAG_G']

        if count_rate == None:
            count_rate = np.ones_like(time)

        k = np.isfinite(count_rate)
        nk = sum(k)
        G0 = -2.5*np.log10( (contam[k]+count_rate[k])*
                flx_frac*10**(-0.4*Gmag)/ flux[k])
        G0mean = np.nanmean(G0)
        contam_flux = contam*10**(-0.4*(Gmag-G0mean)) 
        flux = (flux - contam_flux)/(1-np.nanmean(contam_flux))
        flux_err = flux_err/(1-np.nanmean(contam_flux))

        if verbose:
            print(f'Fraction of target flux in aperture = {flx_frac:0.4f}')
            print(f'Target G magnitude = {Gmag:0.3f}')
            mncr = np.nanmedian(count_rate)
            print(f'Median normalized count rate = {mncr:0.3f}')
            print(f'No. of valid count rate values = {sum(k)}')
            if nk>1:
                G0err  = np.nanstd(G0)/np.sqrt(sum(k))
                print(f'G-band zero point = {G0mean:0.4f} +/- {G0err:0.4f}')
            else:
                print(f'G-band zero point = {G0mean:0.4f}')

        self.lc['flux'] = flux
        self.lc['flux_err'] = flux_err
        self.decontaminated = True
        return time, flux, flux_err
        

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

    def mask_data(self, mask, verbose=True):
        """
        Mask light curve data

        Replace the light curve in the dataset with a subset of the data for
        which the input mask is False.

        The orignal data are saved in lc_unmask

        """
        self.lc_unmask = copy(self.lc)
        for k in self.lc:
            if isinstance(self.lc[k],np.ndarray):
                self.lc[k] = self.lc[k][~mask]
        if verbose:
            print('\nMasked {} points'.format(sum(mask)))
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
        # medfilt pads the array to be filtered with zeros, so edge behaviour
        # is better if we filter flux-1 rather than flux.
        d = abs(medfilt(flux-1, width)+1-flux)
        mad = d.mean()
        ok = d < clip*mad
        for k in self.lc:
            if isinstance(self.lc[k],np.ndarray):
                self.lc[k] = self.lc[k][ok]
        if verbose:
            print('\nRejected {} points more than {:0.1f} x MAD = {:0.0f} '
                    'ppm from the median'.format(sum(~ok),clip,1e6*mad*clip))
        return self.lc['time'], self.lc['flux'], self.lc['flux_err']

#----------------------------------

    def diagnostic_plot(self, fname=None,
            figsize=(8,8), fontsize=10, flagged=None):
        
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

        tjdb_table = D['BJD_TIME']
        flux_table = D['FLUX']
        flux_err_table = D['FLUXERR']
        back_table = D['BACKGROUND']
        rollangle_table = D['ROLL_ANGLE']
        xcen_table = D['CENTROID_X']
        ycen_table = D['CENTROID_Y']
        contam_table = D['CONTA_LC']
        contam_err_table = D['CONTA_LC_ERR']
        try:
            smear_table = D['SMEARING_LC']
        except:
            smear_table = np.zeros_like(tjdb_table)

        flux_bad_table = D['FLUX_BAD']
        back_bad_table = D['BACKGROUND_BAD']

        xloc_table = D['LOCATION_X']
        yloc_table = D['LOCATION_Y']

        time = np.array(self.lc['time'])+self.lc['bjd_ref']
        flux = np.array(self.lc['flux'])*np.nanmean(flux_table)
        flux_err = np.array(self.lc['flux_err'])*np.nanmean(flux_table)
        rollangle = np.array(self.lc['roll_angle'])
        xcen = np.array(self.lc['centroid_x'])
        ycen = np.array(self.lc['centroid_y'])
        xoff = np.array(self.lc['xoff'])
        yoff = np.array(self.lc['yoff'])
        bg = np.array(self.lc['bg'])
        contam = np.array(self.lc['contam'])
        try:
            smear = np.array(self.lc['smear'])
        except KeyError:
            smear = np.zeros_like(time)

        plt.rc('font', size=fontsize)
        fig, ax = plt.subplots(5,2,figsize=figsize)
        cgood = 'midnightblue'
        cbad = 'xkcd:red'

        if flagged:
            flux_measure = copy(flux_table)
        else:
            flux_measure = copy(flux)
        ax[0,0].scatter(time,flux,s=2,c=cgood)
        if flagged:
            ax[0,0].scatter(tjdb_table,flux_bad_table,s=2,c=cbad)
        ax[0,0].set_ylim(0.998*np.quantile(flux_measure,0.16),
                         1.002*np.quantile(flux_measure,0.84))
        ax[0,0].set_xlabel('BJD')
        ax[0,0].set_ylabel('Flux [e-]')
        
        ax[0,1].scatter(rollangle,flux,s=2,c=cgood)
        if flagged:
            ax[0,1].scatter(rollangle_table,flux_bad_table,s=2,c=cbad)
        ax[0,1].set_ylim(0.998*np.quantile(flux_measure,0.16),
                         1.002*np.quantile(flux_measure,0.84))
        ax[0,1].set_xlabel('Roll angle in degrees')
        ax[0,1].set_ylabel('Flux [e-]')
        
        ax[1,0].scatter(time,bg,s=2,c=cgood)
        if flagged:
            ax[1,0].scatter(tjdb_table,back_bad_table,s=2,c=cbad)
        ax[1,0].set_xlabel('BJD')
        ax[1,0].set_ylabel('Background [e-]')
        ax[1,0].set_ylim(0.9*np.quantile(bg,0.005),
                         1.1*np.quantile(bg,0.995))
        
        ax[1,1].scatter(rollangle,bg,s=2,c=cgood)
        if flagged:
            ax[1,1].scatter(rollangle_table,back_bad_table,s=2,c=cbad)
        ax[1,1].set_xlabel('Roll angle in degrees')
        ax[1,1].set_ylabel('Background [e-]')
        ax[1,1].set_ylim(0.9*np.quantile(bg,0.005),
                         1.1*np.quantile(bg,0.995))
        
        ax[2,0].scatter(xcen,flux,s=2,c=cgood)
        if flagged:
            ax[2,0].scatter(xcen_table,flux_bad_table,s=2,c=cbad)
        ax[2,0].set_ylim(0.998*np.quantile(flux_measure,0.16),
                         1.002*np.quantile(flux_measure,0.84))
        ax[2,0].set_xlabel('Centroid x')
        ax[2,0].set_ylabel('Flux [e-]')
        
        ax[2,1].scatter(ycen,flux,s=2,c=cgood)
        if flagged:
            ax[2,1].scatter(ycen_table,flux_bad_table,s=2,c=cbad)
        ax[2,1].set_ylim(0.998*np.quantile(flux_measure,0.16),
                         1.002*np.quantile(flux_measure,0.84))
        ax[2,1].set_xlabel('Centroid y')
        ax[2,1].set_ylabel('Flux [e-]')
        
        ax[3,0].scatter(contam,flux,s=2,c=cgood)
        if flagged:
            ax[3,0].scatter(contam_table,flux_bad_table,s=2,c=cbad)
        ax[3,0].set_xlabel('Contamination estimate')
        ax[3,0].set_ylabel('Flux [e-]')
        ax[3,0].set_xlim(np.min(contam),np.max(contam))
        ax[3,0].set_ylim(0.998*np.quantile(flux_measure,0.16),
                         1.002*np.quantile(flux_measure,0.84))     
        
        ax[3,1].scatter(smear,flux,s=2,c=cgood)
        if flagged:
            ax[3,1].scatter(smear_table,flux_bad_table,s=2,c=cbad)
        ax[3,1].set_xlabel('Smear estimate')
        ax[3,1].set_ylabel('Flux [e-]')
        if np.ptp(smear) > 0:
            ax[3,1].set_xlim(np.min(smear),np.max(smear))
        else:
            ax[3,1].set_xlim(-1,1)
        ax[3,1].set_ylim(0.998*np.quantile(flux_measure,0.16),
                         1.002*np.quantile(flux_measure,0.84))

        ax[4,0].scatter(rollangle,xoff,s=2,c=cgood)
        #ax[4,0].scatter(rollangle,yoff,s=2,c=cbad)
        ax[4,0].set_xlabel('Roll angle in degrees')
        ax[4,0].set_ylabel('X centroid offset')

        #ax[4,1].scatter(rollangle,xoff,s=2,c=cgood)
        ax[4,1].scatter(rollangle,yoff,s=2,c=cbad)
        ax[4,1].set_xlabel('Roll angle in degrees')
        ax[4,1].set_ylabel('Y centroid offset')

        fig.tight_layout()
        if fname == None:
            plt.show()
        else:
            plt.savefig(fname)

    #------

    def decorr(self, dfdt=False, d2fdt2=False, dfdx=False, d2fdx2=False, 
                dfdy=False, d2fdy2=False, d2fdxdy=False, dfdsinphi=False, 
                dfdcosphi=False, dfdsin2phi=False, dfdcos2phi=False,
                dfdsin3phi=False, dfdcos3phi=False, dfdbg=False,
                dfdcontam=False, dfdsmear=False, scale=True):

        time = np.array(self.lc['time'])
        flux = np.array(self.lc['flux'])
        flux_err = np.array(self.lc['flux_err'])
        phi = self.lc['roll_angle']*np.pi/180
        sinphi = interp1d(time,np.sin(phi), fill_value=0, bounds_error=False)
        cosphi = interp1d(time,np.cos(phi), fill_value=0, bounds_error=False)

        dx = interp1d(time,self.lc['xoff'], fill_value=0, bounds_error=False)
        dy = interp1d(time,self.lc['yoff'], fill_value=0, bounds_error=False)

        model = self.__factor_model__(scale)
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
        params.add('dfdsin3phi', value=0, vary=dfdsin3phi)
        params.add('dfdcos3phi', value=0, vary=dfdcos3phi)
        params.add('dfdbg', value=0, vary=dfdbg)
        params.add('dfdcontam', value=0, vary=dfdcontam)
        params.add('dfdsmear', value=0, vary=dfdsmear)
        
        result = model.fit(flux, params, t=time)
        print("Fit Report")
        print(result.fit_report())
        result.plot()

        print("\nCompare the lightcurve RMS before and after decorrelation")
        print('RMS before = {:0.1f} ppm'.format(1e6*self.lc['flux'].std()))
        self.lc['flux'] =  flux/result.best_fit
        self.lc['flux_err'] =  flux_err/result.best_fit
        print('RMS after = {:0.1f} ppm'.format(1e6*self.lc['flux'].std()))

        flux_d = flux/result.best_fit
        flux_err_d = flux_err/result.best_fit
        fig,ax=plt.subplots(1,2,figsize=(8,4))
        y = 1e6*(flux_d-1)
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
        
        return flux_d, flux_err_d
        
#-----------------------------------
    def should_I_decorr(self,mask_centre=0,mask_width=0,scale=True):
        
        flux = np.array(self.lc['flux'])
        flux_err = np.array(self.lc['flux_err'])
        phi = self.lc['roll_angle']*np.pi/180
        sinphi = interp1d(np.array(self.lc['time']),np.sin(phi), fill_value=0,
                          bounds_error=False)
        cosphi = interp1d(np.array(self.lc['time']),np.cos(phi), fill_value=0,
                          bounds_error=False)
        bg = interp1d(np.array(self.lc['time']),self.lc['bg'], fill_value=0,
                      bounds_error=False)
        contam = interp1d(np.array(self.lc['time']),self.lc['contam'], fill_value=0,
                          bounds_error=False)
        smear = interp1d(np.array(self.lc['time']),self.lc['smear'], fill_value=0,
                          bounds_error=False)
        dx = interp1d(np.array(self.lc['time']),self.lc['xoff'], fill_value=0,
                      bounds_error=False)
        dy = interp1d(np.array(self.lc['time']),self.lc['yoff'], fill_value=0,
                      bounds_error=False)
        time = np.array(self.lc['time'])
        
        if mask_centre != 0:    
            flux = flux[(self.lc['time'] < (mask_centre-mask_width/2)) | 
                        (self.lc['time'] > (mask_centre+mask_width/2))]
            flux_err = flux_err[(self.lc['time'] < (mask_centre-mask_width/2)) |
                                (self.lc['time'] > (mask_centre+mask_width/2))]
            
            time_cut = time[(self.lc['time'] < (mask_centre-mask_width/2)) |
                            (self.lc['time'] > (mask_centre+mask_width/2))]
            phi_cut = self.lc['roll_angle'][(self.lc['time'] < (mask_centre-mask_width/2)) |
                                            (self.lc['time'] > (mask_centre+mask_width/2))] *np.pi/180
            sinphi = interp1d(time_cut,np.sin(phi_cut), fill_value=0, bounds_error=False)        
            cosphi = interp1d(time_cut,np.cos(phi_cut), fill_value=0, bounds_error=False)
            
            bg_cut = self.lc['bg'][(self.lc['time'] < (mask_centre-mask_width/2)) |
                                   (self.lc['time'] > (mask_centre+mask_width/2))]
            bg = interp1d(time_cut,bg_cut, fill_value=0, bounds_error=False)
            contam_cut = self.lc['contam'][(self.lc['time'] < (mask_centre-mask_width/2)) |
                                           (self.lc['time'] > (mask_centre+mask_width/2))]
            contam = interp1d(time_cut,contam_cut, fill_value=0, bounds_error=False)
            dx_cut = self.lc['xoff'][(self.lc['time'] < (mask_centre-mask_width/2)) |
                                     (self.lc['time'] > (mask_centre+mask_width/2))]
            dx = interp1d(time_cut,dx_cut, fill_value=0, bounds_error=False)
            dy_cut = self.lc['yoff'][(self.lc['time'] < (mask_centre-mask_width/2)) |
                                     (self.lc['time'] > (mask_centre+mask_width/2))]
            dy = interp1d(time_cut,dy_cut, fill_value=0, bounds_error=False)

            time = time[(self.lc['time'] < (mask_centre-mask_width/2)) |
                        (self.lc['time'] > (mask_centre+mask_width/2))]


        params_d = ['dfdt', 'dfdx', 'dfdy', 'dfdsinphi', 'dfdcosphi', 'dfdbg', 'dfdcontam',
                    'dfdsmear', 'd2fdt2', 'd2fdx2', 'd2fdy2', 'dfdsin2phi', 'dfdcos2phi']
        boolean = [[False, True]]*len(params_d)
        decorr_arr = [[]]*len(params_d)

        for kindex, k in enumerate(range(len(params_d))):
            temp = []
            for jindex, j in enumerate([False, True]):
                for index, i in enumerate(range(len(params_d)-kindex)):
                    temp.append(boolean[index][jindex])
            decorr_arr[kindex] = temp
            for hindex, h in enumerate(range(kindex)):
                decorr_arr[kindex].append(False)
                decorr_arr[kindex].append(True)

        for index, i in enumerate(decorr_arr[0]):
            dfdt=decorr_arr[0][index]
            dfdx=decorr_arr[1][index]
            dfdy=decorr_arr[2][index]
            dfdsinphi=decorr_arr[3][index]
            dfdcosphi=decorr_arr[4][index]
            dfdbg=decorr_arr[5][index]
            dfdcontam=decorr_arr[6][index]
            dfdsmear=decorr_arr[7][index]
            d2fdt2=decorr_arr[8][index]
            d2fdx2=decorr_arr[9][index]
            d2fdy2=decorr_arr[10][index]
            dfdsin2phi=decorr_arr[11][index]
            dfdcos2phi=decorr_arr[12][index]
            
            model = self.__factor_model__(scale)
            params = model.make_params()
            params.add('dfdt', value=0, vary=dfdt)
            params.add('dfdx', value=0, vary=dfdx)
            params.add('dfdy', value=0, vary=dfdy)
            params.add('dfdsinphi', value=0, vary=dfdsinphi)
            params.add('dfdcosphi', value=0, vary=dfdcosphi)
            params.add('dfdbg', value=0, vary=dfdbg)
            params.add('dfdcontam', value=0, vary=dfdcontam)
            params.add('dfdsmear', value=0, vary=dfdsmear)
            params.add('d2fdt2', value=0, vary=d2fdt2)
            params.add('d2fdx2', value=0, vary=d2fdx2)
            params.add('d2fdy2', value=0, vary=d2fdy2)
            params.add('dfdsin2phi', value=0, vary=dfdsin2phi)
            params.add('dfdcos2phi', value=0, vary=dfdcos2phi)
            
            result = model.fit(flux, params, t=time)

            if index == 0:
                min_BIC = copy(result.bic)
                decorr_params = []
            else:
                if result.bic < min_BIC:
                    min_BIC = copy(result.bic)
                    decorr_params = []
                    for xindex, x in enumerate([dfdt, dfdx, dfdy, dfdsinphi, dfdcosphi, dfdbg, dfdcontam,
                                                dfdsmear, d2fdt2, d2fdx2, d2fdy2, dfdsin2phi, dfdcos2phi]):
                        if x == True:
                            if params_d[xindex] == "dfdsinphi":
                                decorr_params.append("dfdsinphi")
                                decorr_params.append("dfdcosphi")
                            elif params_d[xindex] == "dfdcosphi" and "dfdsinphi" not in decorr_params:
                                decorr_params.append("dfdsinphi") 
                                decorr_params.append("dfdcosphi")
                            elif params_d[xindex] == "dfdcosphi" and "dfdcosphi" in decorr_params:
                                continue
                            elif params_d[xindex] == "dfdsin2phi":
                                decorr_params.append("dfdsin2phi")
                                decorr_params.append("dfdcos2phi")
                            elif params_d[xindex] == "dfdcos2phi" and "dfdsin2phi" not in decorr_params:
                                decorr_params.append("dfdsin2phi")  
                                decorr_params.append("dfdcos2phi")
                            elif params_d[xindex] == "dfdcos2phi" and "dfdcos2phi" in decorr_params:
                                continue
                            else:
                                decorr_params.append(params_d[xindex])
            
        if len(decorr_params) == 0:
            print("No decorrelation is needed.")
        else:
            print("Decorrelate in", *decorr_params, "using decorr, lmfit_transt, or lmfit_eclipse functions.")
        return(min_BIC, decorr_params)  

#---------------------------------

# Pickling

    def __getstate__(self):
        state = self.__dict__.copy()

        # Replace lmfit model with its string representation
        if 'model' in state.keys():
            model_repr = state['model'].__repr__()
            state['model'] = model_repr
        else:
            state['model'] = ''

        # There may also be an instance of an lmfit model buried in 
        # sampler.log_prob_fn.args - replace with its string representation
        if 'sampler' in state.keys():
            args = state['sampler'].log_prob_fn.args
            model_repr = args[0].__repr__()
            state['sampler'].log_prob_fn.args = (model_repr, *args[1:])

        return state

    #------

    def __setstate__(self, state):

        # Fix for old saved datasets with no __scale__ attribute
        if not hasattr(self, '__scale__'):
            self.__scale__ = True
        
        # Fix for old saved datasets with no __extra_basis_funcs__ attribute
        if not hasattr(self, '__extra_basis_funcs__'):
            self.__extra_basis_funcs__ = {}

        def reconstruct_model(model_repr,state):
            F = self.__factor_model__(self.__scale__,
                                      self.__extra_basis_funcs__)
            if '_transit_func' in model_repr:
                model = TransitModel()
                model *= self.__factor_model__(self.__scale__,
                                               self.__extra_basis_funcs__)
            elif '_eclipse_func' in model_repr:
                model = EclipseModel()
                model *= self.__factor_model__(self.__scale__,
                                               self.__extra_basis_funcs__)
            else:
                model = None
            if 'glint_func' in model_repr:
                model += Model(_glint_func, independent_vars=['t'],
                    f_theta=state['f_theta'], f_glint=state['f_glint'])
            return model

        self.__dict__.update(state)

        if 'model' in state.keys():
            self.model = reconstruct_model(state['model'],state)

        if 'sampler' in state.keys():
            args = state['sampler'].log_prob_fn.args
            model = reconstruct_model(args[0],state)
            state['sampler'].log_prob_fn.args = (model, *args[1:])

