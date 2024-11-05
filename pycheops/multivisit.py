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
MultiVisit
==========
 Object class for analysis of multiple data sets

"""

from __future__ import (absolute_import, division, print_function,
                                unicode_literals)
import numpy as np
from glob import glob
from .dataset import Dataset
from .starproperties import StarProperties
import re
import pickle
from warnings import warn
from .dataset import _kw_to_Parameter,  _log_prior
from .dataset import _make_interp
from lmfit import Parameters, Parameter
from lmfit import fit_report as lmfit_report
from lmfit import __version__ as _lmfit_version_
from . import __version__
from lmfit.models import ExpressionModel, Model
from lmfit.minimizer import MinimizerResult
from .models import TransitModel, FactorModel, EclipseModel, EBLMModel
from .models import PlanetModel, HotPlanetModel
from celerite2.terms import Term, SHOTerm
from celerite2 import GaussianProcess
from .funcs import rhostar, massradius, eclipse_phase
from uncertainties import UFloat, ufloat
from emcee import EnsembleSampler
import corner
from sys import stdout
import matplotlib.pyplot as plt
from collections import OrderedDict
from lmfit.printfuncs import gformat
from copy import copy, deepcopy
from .utils import phaser, lcbin
from scipy.stats import iqr
from astropy.time import Time
from astropy.table import Table
from astropy import units as u
from astropy.coordinates import SkyCoord
import cdspyreadme
import os

# Iteration limit for initialisation of walkers
_ITMAX_ = 999

#--------

class CosineTerm(Term):
    def __init__(self, omega_j, sigma_j):
        self.omega_j = omega_j
        self.sigma_j = sigma_j
    def get_coefficients(self):
        ar = np.empty(0)
        cr = np.empty(0)
        ac = np.array([self.sigma_j])
        bc = np.zeros(1)
        cc = np.zeros(1)
        dc = np.array([self.omega_j])
        return (ar, cr, ac, bc, cc, dc)

#--------

SineModel = ExpressionModel('sin(2*pi*(x-x0)/P)')

#--------

# Parameter delta_t needed here to cope with change of time system between
# Dataset and Multivisit
def _glint_func(t, glint_scale, f_theta=None, f_glint=None, delta_t=None):
    glint = f_glint(f_theta(t-delta_t))
    return glint_scale * glint

#--------

def _make_labels(plotkeys, d0, extra_labels):
    labels = []
    r = re.compile('dfd(.*)_([0-9][0-9])')
    r2 = re.compile('d2fd(.*)2_([0-9][0-9])')
    rt = re.compile('ttv_([0-9][0-9])')
    rr = re.compile('ramp_([0-9][0-9])')
    rl = re.compile('L_([0-9][0-9])')
    rc = re.compile('c_([0-9][0-9])')
    for key in plotkeys:
        if key in extra_labels.keys():
            labels.append(extra_labels[key])
        elif key == 'T_0':
            labels.append(r'T$_0-{:.0f}$'.format(d0))
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
        elif r.match(key):
            p,n = r.match(key).group(1,2)
            p = p.replace('_','\_')
            labels.append(r'$df\,/\,d{}_{{{}}}$'.format(p,n))
        elif r2.match(key):
            p,n = r2.match(key).group(1,2)
            p = p.replace('_','\_')
            labels.append(r'$d^2f\,/\,d{}^2_{{{}}}$'.format(p,n))
        elif rt.match(key):
            n = rt.match(key).group(1)
            labels.append(r'$\Delta\,T_{{{}}}$'.format(n))
        elif rr.match(key):
            n = rr.match(key).group(1)
            labels.append(r'$df\,/\,d\Delta\,T_{{{}}}$'.format(n))
        elif rl.match(key):
            n = rl.match(key).group(1)
            labels.append(r'$L_{{{}}}$'.format(n))
        elif rc.match(key):
            n = rc.match(key).group(1)
            labels.append(r'$c_{{{}}}$'.format(n))
        elif key == 'log_sigma_w':
            labels.append(r'$\log\sigma_w$')
        elif key == 'log_omega0':
            labels.append(r'$\log\omega_0$')
        elif key == 'log_S0':
            labels.append(r'$\log{\rm S}_0$')
        elif key == 'log_Q':
            labels.append(r'$\log{\rm Q}$')
        elif key == 'logrho':
            labels.append(r'$\log\rho_{\star}$')
        elif key == 'aR':
            labels.append(r'${\rm a}\,/\,{\rm R}_{\star}$')
        elif key == 'sini':
            labels.append(r'\sin i')
        else:
            labels.append(key)
    return labels

class MultiVisit(object):
    """
    CHEOPS MultiVisit object

    Specify a target name to initialize from pickled datasets in the current
    working directory (or in datadir if datadir is not None).

    The target name can include blanks - these are replaced by "_"
    automatically before searching for matching file names. 

    The parameter ident is used to collect star and planet properties from the
    relevant tables at DACE. If ident is None (default) then the target name
    is used in place of ident. Set ident='none' to disable this feature.  See
    also StarProperties for other options that can be set using id_kws, e.g.,
    id_kws={'dace':False} to use SWEET-Cat instead of DACE.

    All dates and times in each of the dataset are stored as BJD-2457000 (same
    as TESS).

    :param target: target name to identify pickled datasets

    :param datadir: directory containing pickled datasets

    :param tag: tag used when desired datasets were saved 

    :param ident: identifier in star properties table. If None use target. If
    'none' 

    :param id_kws: keywords for call to StarProperties.

    :param verbose: print dataset names, etc. if True

    Notes on fitting routines
    ~~~~~~~~~~~~~~~~~~~~~~~~~

    Transit parameters
    ~~~~~~~~~~~~~~~~~~

    The same values of the transit parameters T_0, P, D, W, b, f_c and f_s are
    used for all the datasets in the combined fit. This also applies to h_1
    and h_2 when fitting transits.

    User-defined parameters can be specified in one of the following ways:

    * fixed value, e.g., P=1.234
    * free parameter with uniform prior interval specified as a 2-tuple,
      e.g., f_c=(-0.5,0.5). The initial value is taken as the the mid-point of
      the allowed interval;
    * free parameter with uniform prior interval and initial value
      specified as a 3-tuple, e.g., (0.1, 0.2, 1);
    * free parameter with a Gaussian prior specified as a ufloat, e.g.,
      ufloat(0,1);
    * as an lmfit Parameter object.

    A transit parameter will be fixed in the fit to the combined datasets 
    only if the same parameter was fixed in the last fit to all datasets
    and the same parameter is not specified as a free parameter in the
    call to this method. 

    If no user-defined value is provided then the initial value for each
    transit parameter is set using the mean value across the individual
    datasets. For T_0 an integer number of periods are added or subtracted
    from the individual T_0 values so that the mean T_0 value corresponds
    to a time of mid-transit near the centre of the datasets.

    N.B. The timescale for T_0 in BJD_TT - 2457000.

    Priors on transit parameters are only set if they are specified in the
    call to the fitting method using either a ufloat, or as an lmfit Parameter
    object that includes a ufloat in its user_data.

    Priors on the derived parameters e, q_1, q_2, logrho, etc. can be
    specified as a dictionary of ufloat values using the extra_priors
    keyword, e.g., extra_priors={'e':ufloat(0.2,0.01)}. Priors on parameters
    that apply to individual datasets can also be specified in extra_priors,
    e.g., extra_priors = {'dfdt_01':ufloat(0.0,0.001)}. Priors listed in
    extra_priors will supercede priors on parameters saved with the individual
    datasets.
    
    Noise model
    ~~~~~~~~~~~

    The noise model assumes that the error bars on each data point have
    addition white noise with standard deviation log_sigma_w. Optionally,
    correlated noise can be included using celerite2 with kernel
    SHOTerm(log_omega0, log_S0, log_Q). The same values of log_sigma_w,
    log_omega0, log_S0 and log_Q are used for all the datasets in the combined
    fit.
    
    The fit to the combined datasets will only include a GP if log_omega0 and
    log_S0 are both specified as arguments in the call to the fitting method.
    If log_Q is not specified as an argument in the call to the fitting method
    then it is fixed at the value log_Q=1/sqrt(2).

    Gaussian priors on the values of log_omega0, log_S0 and log_Q will
    only be applied if the user-specified value includes a Gaussian prior,
    e.g., log_omega0=ufloat(6,1), log_S0=ufloat(-24,2). 

    N.B. Gaussian priors on log_omega0, log_S0 and log_Q specified in the
    individual datasets are ignored. 

    Parameter decorrelation
    ~~~~~~~~~~~~~~~~~~~~~~~

    Decorrelation against roll angle (phi) is handled differently in
    Multivisit to Dataset. The decorrelation against cos(phi), sin(phi),
    cos(2.phi), sin(2.phi), etc. is done using a combination of the trick
    from Rodrigo et al. (2017RNAAS...1....7L) and the celerite model by
    Foremann-Mackey et al. (2017AJ....154..220F). This enables the
    coefficients of this "linear harmonic instrumental noise model" to be
    treated as nuisance parameters that are automatically marginalised
    away by adding a suitable term (CosineTerm) to the covariance matrix. This
    is all done transparently by setting "unroll=True". The number of harmonic
    terms is set by nroll, e.g., setting nroll=3 (default) includes terms
    up to sin(3.phi) and cos(3.phi). This requires that phi is a linear
    function of time for each dataset, which is a good approximation for
    individual CHEOPS visits. 
    
    Other decorrelation parameters not derived from the roll angle, e.g. dfdx,
    dfdy, etc. are included in the fit to individual datasets only if they
    were free parameters in the last fit to that dataset. The decorrelation is
    done independently for each dataset. The free parameters are labelled
    dfdx_ii, dfdy_ii where ii is the number of the dataset to which each
    decorrelation parameter applies, i.e. ii=01, 02, 03, etc. 

    Glint correction is done independently for each dataset if the glint
    correction was included in the last fit to that dataset. The glint
    scale factor for dataset ii is labelled glint_scale_ii. The glint
    scaling factor for each dataset can either be a fixed or a free
    parameter, depending on whether it was a fixed or a free parameter in
    the last fit to that dataset.

    Note that the "unroll" method implicitly assumes that the rate of change
    of roll angle, Omega = d(phi)/dt, is constant. This is a reasonable
    approximation but can introduce some extra noise in cases where
    instrumental noise correlated with roll angle is large, e.g., observations
    of faint stars in crowded fields. In this case it may be better to
    include the best-fit trends against roll angle from the last fit stored in
    the .dataset file in the fit to each dataset. This case be done using the
    keyword argument "unwrap=True". This option can be combined with the
    "unroll=True" option, i.e. to use "unroll"  as a small correction to the
    "unwrap" roll-angle decorrelation from the last fit to each data set.

     If you only want to store and yield 1-in-thin samples in the chain, set
    thin to an integer greater than 1. When this is set, thin*steps will be
    made and the chains returned with have "steps" values per walker.

    Fits, models, trends and correlated noise
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    The best fit to the light curve in each data set is

      f_fit = f_sys x f_fac + f_glint + f_celerite + f_unwrap 
    
    - "f_sys" includes all the photometric effects intrinsic to the
       star/planet system, i.e. transits and eclipses
    - "f_fac" includes all the trends correlated with parameters apart
       from spacecraft roll angle
    - "f_glint" is an optional function of roll angle scaled by the parameter
      glint_scale used to model internal reflections or other features
      correlated with roll angle (otherwise f_glint=0). 
    - "f_celerite" is the maximum-likelihood Gaussian process generated for a
      kernel SHOTerm() + CosineTerm(Omega) + CosineTerm(2*Omega) + ..., where
      the number of CosineTerm() kernels is specified by nroll and SHOTerm()
      is only included if correlated noise is included in the model. 
    - "f_unwrap" are the trends correlated with spacecraft roll angle removed
      if the unwrap=True option is specified (otherwise f_unwrap = 0)

    For plotting and data output we require the "detrended flux", i.e.
    
      flux_d = f_sys + f_sho + f_fit - f_obs

    where f_obs is the observed flux and f_sho is the maximum-likelihood
    Gaussian process generated using only the SHOTerm() kernel, i.e. the
    detrended fluxes include the correlated noise modelled by f_sho. The
    detrended fluxes for the best fits to each dataset are included in the
    output lmfit ModelResult object in the attribute fluxes_det.

    Return value
    ~~~~~~~~~~~~
     The fitting routines return lmfit MinimizerResult objects with a few
     extra attributes. Samples generated by emcee are returned as a python
     array in the attribute flat_chain instead of a pandas.DataFrame object in
     the attribute flatchain.

    Backends
    --------
     See https://emcee.readthedocs.io/en/stable/tutorials/monitor/ for use of
     the backend keyword.
      

    """

    def __init__(self, target=None, datadir=None, tag="", 
            ident=None, id_kws={'dace':True},
            verbose=True):

        self.target = target
        self.datadir = datadir
        self.datasets = []

        if target is None: return

        ptn = target.replace(" ","_")+'_'+tag+'_*.dataset'
        if datadir is not None:
            ptn = os.path.join(datadir,ptn)

        datatimes = [Dataset.load(i).bjd_ref for i in glob(ptn)]
        g = [x for _,x in sorted(zip(datatimes,glob(ptn)))] 
        if len(g) == 0:
            warn(f'No matching dataset names for target {target}', UserWarning)
            return

        if ident != 'none':
            if ident is None: ident = target 
            self.star = StarProperties(ident, **id_kws)

        if verbose:
            print(self.star)
            print('''
 N  file_key                   Aperture last_ GP Glint Scale pipe_ver extra
 --------------------------------------------------------------------------''')

        for n,fl in enumerate(g):
            d = Dataset.load(fl)

            # Make time scales consistent
            dBJD = d.bjd_ref - 2457000
            d._old_bjd_ref = d.bjd_ref
            d.bjd_ref = 2457000
            d.lc['time'] += dBJD
            d.lc['bjd_ref'] = dBJD
            for xbf in d.__extra_basis_funcs__:
                d.__extra_basis_funcs__[xbf].x += dBJD
            if 'lmfit' in d.__dict__:
                p = deepcopy(d.lmfit.params['T_0'])
                p._val += dBJD
                p.init_value += dBJD
                p.min += dBJD
                p.max += dBJD
                d.lmfit.params['T_0'] = p
                if 'T_0' in d.lmfit.var_names: 
                    d.lmfit.init_vals[d.lmfit.var_names.index('T_0')] += dBJD
                if 'T_0' in d.lmfit.init_values: 
                    d.lmfit.init_values['T_0'] += dBJD

            if 'emcee' in d.__dict__:
                p = deepcopy(d.emcee.params['T_0'])
                p._val += dBJD
                p.init_value += dBJD
                p.min += dBJD
                p.max += dBJD
                d.emcee.params['T_0'] = p
                p = deepcopy(d.emcee.params_best['T_0'])
                p._val += dBJD
                p.init_value += dBJD
                p.min += dBJD
                p.max += dBJD
                d.emcee.params_best['T_0'] = p
                if 'T_0' in d.emcee.var_names: 
                    j = d.emcee.var_names.index('T_0')
                    d.emcee.init_vals[j] += dBJD
                    d.emcee.chain[:,j] += dBJD
                if 'T_0' in d.emcee.init_values: 
                    d.emcee.init_values['T_0'] += dBJD

            self.datasets.append(d)
            if verbose:
                dd = d.__dict__
                ap = d.lc['aperture'] if 'lc' in dd else '---'
                lf = d.__lastfit__ if '__lastfit__' in dd else '---'
                try: 
                    gp = 'Yes' if d.gp else 'No'
                except AttributeError:
                    gp = 'No'
                gl = 'Yes' if 'f_glint' in dd else 'No'
                if d.__scale__ == None:
                    sc = 'n/a'
                elif d.__scale__:
                    sc = 'True'
                else:
                    sc = 'False'
                pv = d.pipe_ver
                nx = len(d.__extra_basis_funcs__)
                print(f' {n+1:2} {d.file_key} {ap:8} {lf:5} {gp:3}'
                      f' {gl:5} {sc:5}  {pv}     {nx}')

#--------------------------------------------------------------------------
#
# Big slab of code here to run the emcee sampler because almost everything is
# common to all fitting routines. Mostly this is parameter handling and model
# creation.
# 
# "params" is an lmfit Parameters object that is used for storing the results,
#  initial values, etc. Not passed to the target log-posterior function.
#
# "self.__models__" is a list of lmfit models that get evaluated in the target
# log-posterior function. 
#
# "self.__modpars__" is a list of Parameters objects, one for each dataset.
# These parameters used to evaluate the models in "self.__models__". These all
# have the same transit model parameters, but different decorrelation
# parameters sent to FactorModel for each dataset. The values in these
# parameter objects are updated in every call to "self._lnpost_".
#
# "self.__rolls__" is a list of celerite kernels for implicit roll-angle
# decorrelation if unroll=True, else a list of "None" values. A separate
# kernel is needed for each dataset because the average roll angle rate is
# different for each visit. 
#
# "self.__noisemodel__" is an lmfit Parameters object used for passing the
#  noise model parameters log_sigma_w, log_omega0, etc. to the target
#  log-posterior function. The user_data may be a ufloat with the "prior".
#
# "self.__fluxes_unwrap__" is the list of roll-angle corrections computed if
# unwrap=True, or a list of arrays contaning 0 if unwrap=False
#
# "self.__priors__" is a list of priors stored as ufloat values. 
#
# "self.__var_names__" is a list of the free parameters in the combined fit.
#
    def __run_emcee__(self, **kwargs):

        # Dict of initial parameter values for creation of models
        # Calculation of mean needs P and W so T_0 is not first in the list
        vals = OrderedDict()
        fittype = self.__fittype__
        klist = ['D', 'W', 'b', 'P', 'T_0', 'f_c', 'f_s', 'l_3']
        if fittype in ['transit', 'eblm', 'planet', 'hotplanet']:
            klist.append('h_1')
            klist.append('h_2')
        if fittype in ['eclipse', 'eblm']:
            for k in ['L', 'a_c']:
                klist.append(k)
        if fittype in ['planet']:
            for k in ['A_g', 'a_c']:
                klist.append(k)
        if fittype in ['hotplanet']:
            for k in ['F_max', 'F_min', 'ph_off', 'a_c']:
                klist.append(k)
        for k in klist:
            vals[k] = kwargs[k]

        # dicts of parameter limits and step sizes for initialisation
        pmin = {'P':0, 'D':0, 'W':0, 'b':0, 'f_c':-1, 'f_s':-1,
                'h_1':0, 'h_2':0, 'L':0, 'F_max':0, 'l_3':-0.99}
        pmax = {'D':0.3, 'W':0.3, 'b':2.0, 'f_c':1, 'f_s':1,
                'h_1':1, 'h_2':1, 'L':1.0, 'F_max':1.0, 'l_3':1e6}
        step = {'D':1e-4, 'W':1e-4, 'b':1e-2, 'P':1e-6, 'T_0':1e-4,
                'f_c':1e-4, 'f_s':1e-3, 'h_1':1e-3, 'h_2':1e-2,
                'L':1e-5, 'F_max':1e-5, 'l_3':1e-3, 'A_g':1e-3}

        # Initial stderr value for list of values that may be np.nan or None
        def robust_stderr(vals, stds, default):
            varr = np.array([v if v is not None else np.nan for v in vals])
            sarr = np.array([s if s is not None else np.nan for s in stds])
            vok = np.isfinite(varr)
            nv = sum(vok)
            sok = vok & np.isfinite(sarr)
            ns = sum(sok)
            if nv == 0: return default
            if nv == 1:
                if ns == 1: return sarr[sok][0]
                return default
            if ns == nv:
                t = np.nanmean(sarr)/np.sqrt(nv)
                if t > 0: return t
                return default
            t = np.nanstd(varr)
            if t > 0: return t
            return default

        # Create a Parameters() object with initial values and priors on model
        # parameters (including fixed parameters)
        extra_priors = kwargs['extra_priors']
        priors = {} if extra_priors is None else extra_priors
        params = Parameters()  
        plist = [d.emcee.params if d.__lastfit__ == 'emcee' else 
                 d.lmfit.params for d in self.datasets]
        vv,vs,vn  = [],[],[]     # Free params for emcee, name value, err

        for k in vals:

            # For fit_hotplanet, 'L'='F_max', so ...
            if (fittype == 'hotplanet') and (k == 'F_max'):
                kp,kv = 'L','F_max'
            # For fit_planet we use 'L' to compute 'A_g', so store it
            # temporarily in 'A_g' ready for computation below
            elif (fittype == 'planet') and (k == 'A_g'):
                kp,kv = 'L','A_g'
            else:
                kp,kv = k,k

            if vals[k] is None:    # No user-defined value 
                vary = True in [p[kp].vary if kp in p else False for p in plist]

                # Use mean of best-fit values from datasets
                if kp == 'T_0':  
                    t = np.array([p[kp].value for p in plist])
                    c = np.round((t-t[0])/params['P'])
                    c -= c.max()//2
                    t -= c*params['P']
                    val = t.mean()
                    vmin = val - params['W']*params['P']/2
                    vmax = val + params['W']*params['P']/2
                    if vary:
                        stds = [p[kp].stderr for p in plist]
                        stderr = robust_stderr(t, stds, step['T_0'])
                else:
                    # Not all datasets have all parameters so ...
                    v = [p[kp].value if kp in p else np.nan for p in plist]
                    val = np.nanmean(v)
                    if vary:
                        stds=[p[kp].stderr if kp in p else None for p in plist]
                        stderr = robust_stderr(v, stds, step[kv])
                    v = [p[kp].min if kp in p else np.nan for p in plist]
                    vmin = np.nanmin(v)
                    if (kv in pmin) and not np.isfinite(vmin):
                        vmin = pmin[kv]
                    v = [p[kp].max if kp in p else np.nan for p in plist]
                    vmax = np.nanmax(v)
                    if (kv in pmax) and not np.isfinite(vmax):
                        vmax = pmax[kv]
                    # Limits of 'A_g' inherited from 'L' will not be right
                    if (kv == 'A_g'):
                        vmin = 0
                        vmax = 1

                params.add(kv, val, vary=vary, min=vmin, max=vmax)
                if vary:
                    params[kv].stderr = stderr
                vals[kv] = val

            else:    # Value for parameter from kwargs

                params[kv] = _kw_to_Parameter(kv, vals[kv])
                vals[kv] = params[kv].value
                if (kv in pmin) and not np.isfinite(params[kv].min):
                    params[kv].min = pmin[kv]
                if (kv in pmax) and not np.isfinite(params[kv].max):
                    params[kv].max = pmax[kv]

            if params[kv].vary:
                vn.append(kv)
                vv.append(params[kv].value)
                if isinstance(params[kv].user_data, UFloat):
                    priors[kv] = params[kv].user_data
                # Step size for setting up initial walker positions
                if params[kv].stderr is None:
                    if params[kv].user_data is None:
                        vs.append(step[kv])
                    else:
                        vs.append(params[kv].user_data.s)
                else:
                    if np.isfinite(params[kv].stderr):
                        vs.append(params[kv].stderr)
                    else:
                        vs.append(step[kv])
            else:
                # Needed to avoid errors when printing parameters
                params[kv].stderr = None


        # Derived parameters
        params.add('k',expr='sqrt(D)',min=0,max=1)
        params.add('aR',expr='sqrt((1+k)**2-b**2)/W/pi',min=1)
        params.add('sini',expr='sqrt(1 - (b/aR)**2)')
        # Avoid use of aR in this expr for logrho - breaks error propogation.
        expr = 'log10(4.3275e-4*((1+k)**2-b**2)**1.5/W**3/P**2)'
        params.add('logrho',expr=expr,min=-9,max=6)
        params.add('e',min=0,max=1,expr='f_c**2 + f_s**2')
        # For eccentric orbits only from Winn, arXiv:1001.2010
        if (params['e'].value>0) or params['f_c'].vary or params['f_s'].vary:
            params.add('esinw',expr='sqrt(e)*f_s')
            params.add('ecosw',expr='sqrt(e)*f_c')
            params.add('b_tra',expr='b*(1-e**2)/(1+esinw)')
            params.add('b_occ',expr='b*(1-e**2)/(1-esinw)')
            params.add('T_tra',expr='P*W*sqrt(1-e**2)/(1+esinw)')
            params.add('T_occ',expr='P*W*sqrt(1-e**2)/(1-esinw)')

        if 'F_min' in params:
            params.add('A',min=0,max=1,expr='F_max-F_min')

        if 'h_1' in params:
            params.add('q_1',min=0,max=1,expr='(1-h_2)**2')
            params.add('q_2',min=0,max=1,expr='(h_1-h_2)/(1-h_2)')
        # Priors given in extra_priors overwrite existing priors
        if extra_priors is not None:
            for k in extra_priors:
                if k in params:
                    params[k].user_data = extra_priors[k]

        # Compute A_g for fit_planet
        # So far, we have stored the value of L and its standard error in
        # place of this variable. Needs to be done here so that we can use
        # consistent values of D and R*/a for the calculation.
        if (fittype == 'planet'):
            L = params['A_g'].value   # this is actually the value of L 
            e_L = params['A_g'].stderr
            params['A_g'].value = params['aR']**2 * L/params['D'].value
            params['A_g'].stderr = params['A_g'].value * e_L/L
            params.add('L_0',expr='D*A_g/aR**2')
            vv[vn.index('A_g')] = params['A_g'].value
            vs[vn.index('A_g')] = params['A_g'].stderr

        if fittype == 'transit':
            ttv = kwargs['ttv']
            ttv_prior = kwargs['ttv_prior']
            if ttv and (params['T_0'].vary or params['P'].vary):
                raise ValueError('TTV not allowed if P or T_0 are variables')
            edv, edv_prior = False, None

        if fittype == 'eclipse':
            edv = kwargs['edv']
            edv_prior = kwargs['edv_prior']
            if edv and params['L'].vary:
                raise ValueError('L must be a fixed parameter of edv=True.')
            ttv, ttv_prior = False, None

        if fittype in ['eblm', 'planet', 'hotplanet']:
            ttv = kwargs['ttv']
            ttv_prior = kwargs['ttv_prior']
            if ttv and (params['T_0'].vary or params['P'].vary):
                raise ValueError('TTV not allowed if P or T_0 are variables')
            edv = kwargs['edv']
            edv_prior = kwargs['edv_prior']
            if edv and params['L'].vary:
                raise ValueError('L must be a fixed parameter of edv=True.')
            if edv and params['F_max'].vary:
                raise ValueError('F_max must be a fixed parameter of edv=True.')
            if edv and params['A_g'].vary:
                raise ValueError('A_g must be a fixed parameter of edv=True.')

        # Make an lmfit Parameters() object that defines the noise model
        noisemodel = Parameters()  
        k = 'log_sigma_w'
        log_sigma_w = kwargs['log_sigma_w']
        if log_sigma_w is None:
            noisemodel.add(k, -6, min=-12, max=-2)
        else:
            noisemodel[k] = _kw_to_Parameter(k, log_sigma_w)
            # Avoid crazy-low values that are consistent with sigma_w = 0
            if not np.isfinite(noisemodel[k].min):
                noisemodel[k].min = np.min([noisemodel[k].value-10, -30])
        params[k] = copy(noisemodel[k])
        if isinstance(noisemodel[k].user_data, UFloat):
            priors[k] = noisemodel[k].user_data
        if noisemodel[k].vary:
            vn.append(k)
            vv.append(noisemodel[k].value)
            vs.append(1)

        log_S0 = kwargs['log_S0']
        log_omega0 = kwargs['log_omega0']
        log_Q = kwargs['log_Q']
        if log_S0 is not None and log_omega0 is not None:
            if log_Q is None: log_Q = np.log(1/np.sqrt(2))
            nvals = {'log_S0':log_S0, 'log_omega0':log_omega0, 'log_Q':log_Q}
            for k in nvals:
                noisemodel[k] = _kw_to_Parameter(k, nvals[k])
                params[k] = copy(noisemodel[k])
                if isinstance(noisemodel[k].user_data, UFloat):
                    priors[k] = noisemodel[k].user_data
                if noisemodel[k].vary:
                    vn.append(k)
                    vv.append(noisemodel[k].value)
                    vs.append(1)
            params.add('rho_SHO',expr='2*pi/exp(log_omega0)')
            params.add('tau_SHO',expr='2*exp(log_Q)/exp(log_omega0)')
            params.add('sigma_SHO',expr='sqrt(exp(log_Q+log_S0+log_omega0))')
            noisemodel.add('rho_SHO',expr='2*pi/exp(log_omega0)')
            noisemodel.add('tau_SHO',expr='2*exp(log_Q)/exp(log_omega0)')
            noisemodel.add('sigma_SHO',
                            expr='sqrt(exp(log_Q+log_S0+log_omega0))')

        # Lists of model parameters and data for individual datasets
        fluxes_unwrap = []
        n_unwrap = []
        rolls = []
        models = []
        modpars = []
        scales = []

        # Cycle over datasets, each with its own set of parameters
        for i,(d,p) in enumerate(zip(self.datasets, plist)):

            f_unwrap = np.zeros_like(d.lc['time'])
            n = 0
            if kwargs['unwrap']:
                phi = d.lc['roll_angle']*np.pi/180
                for j in range(1,4):
                    k = 'dfdsinphi' if j < 2 else f'dfdsin{j}phi'
                    if k in p:
                        f_unwrap += p[k]*np.sin(j*phi)
                        n = j
                    k = 'dfdcosphi' if j < 2 else f'dfdcos{j}phi'
                    if k in p:
                        f_unwrap += p[k]*np.cos(j*phi)
                        n = j
            n_unwrap.append(n)
            fluxes_unwrap.append(f_unwrap)

            t = d.lc['time']
            try:
                smear = d.lc['smear']
            except KeyError:
                smear = np.zeros_like(t)
            try:
                deltaT = d.lc['deltaT']
            except KeyError:
                deltaT = np.zeros_like(t)
            if d.__scale__:
                factor_model = FactorModel(
                    dx = _make_interp(t,d.lc['xoff'], scale='range'),
                    dy = _make_interp(t,d.lc['yoff'], scale='range'),
                    bg = _make_interp(t,d.lc['bg'], scale='range'),
                    contam = _make_interp(t,d.lc['contam'], scale='range'),
                    smear = _make_interp(t,smear, scale='range'),
                    deltaT = _make_interp(t,deltaT),
                    extra_basis_funcs=d.__extra_basis_funcs__)
            else:
                factor_model = FactorModel(
                    dx = _make_interp(t,d.lc['xoff']),
                    dy = _make_interp(t,d.lc['yoff']),
                    bg = _make_interp(t,d.lc['bg']),
                    contam = _make_interp(t,d.lc['contam']),
                    smear = _make_interp(t,smear),
                    deltaT = _make_interp(t,deltaT),
                    extra_basis_funcs=d.__extra_basis_funcs__)

            if fittype == 'transit':
                model = TransitModel()*factor_model
            elif fittype == 'eclipse':
                model = EclipseModel()*factor_model
            elif fittype == 'eblm':
                model = EBLMModel()*factor_model
            elif fittype == 'planet':
                model = PlanetModel()*factor_model
            elif fittype == 'hotplanet':
                model = HotPlanetModel()*factor_model
            l = ['dfdbg','dfdcontam','dfdsmear','dfdx','dfdy']
            if True in [p_ in l for p_ in p]:
                scales.append(d.__scale__)
            else:
                scales.append(None)

            if 'glint_scale' in p:
                delta_t = d._old_bjd_ref - d.bjd_ref
                model += Model(_glint_func, independent_vars=['t'],
                    f_theta=d.f_theta, f_glint=d.f_glint, delta_t=delta_t)
            models.append(model)

            modpar = model.make_params(verbose=False, **vals)
            # Copy min/max values from params to modpar
            for pm in modpar:
                if pm in params:
                    modpar[pm].min = params[pm].min
                    modpar[pm].max = params[pm].max

            if ttv: 
                modpar['T_0'].init_value = modpar['T_0'].value
            modpars.append(modpar)

            if ttv:
                t = f'ttv_{i+1:02d}'
                params.add(t, 0)
                params[t].user_data = ufloat(0,ttv_prior)
                vn.append(t)
                vv.append(0)
                vs.append(30)
                priors[t] = params[t].user_data
                
            if edv:
                t = f'L_{i+1:02d}'
                params.add(t, vals['L'])
                params[t].user_data = ufloat(vals['L'], edv_prior)
                vn.append(t)
                vv.append(vals['L'])
                vs.append(edv_prior)
                priors[t] = params[t].user_data
                
            # Now the decorrelation parameters, incliding arbitary
            # basis functions, if present
            for dfdp in [k for k in p if (k[:3]=='dfd' or k[:4]=='d2fd' or
                         k=='c' or k=='ramp' or k=='glint_scale') and
                         k[:6]!='dfdsin' and k[:6]!='dfdcos']:

                if  p[dfdp].vary:
                    pj = f'{dfdp}_{i+1:02d}'
                    params.add(pj, p[dfdp].value,
                            min=p[dfdp].min, max=p[dfdp].max)
                    if pj in priors:
                        params[pj].user_data = priors[pj]
                    vn.append(pj)
                    vv.append(p[dfdp].value)
                    try:
                        vs.append(p[dfdp].stderr)
                    except KeyError:
                        if dfdp == 'glint_scale':
                            vs.append(0.01)
                        elif dfdp == 'ramp':
                            vs.append(50)
                        else:
                            vs.append(1e-6)

            if kwargs['unroll']:
                sinphi = np.sin(np.radians(d.lc['roll_angle']))
                s = SineModel.fit(sinphi, P=99/1440, x0=0, x=d.lc['time'])
                Omega= 2*np.pi/s.params['P']
                fluxrms = np.nanstd(d.lc['flux'])
                roll = CosineTerm(omega_j=Omega, sigma_j=fluxrms)
                for j in range(2,kwargs['nroll']+1):
                    roll = roll + CosineTerm(omega_j=j*Omega, sigma_j=fluxrms)
                rolls.append(roll)
            else:
                rolls.append(None)
        # END of for dataset in self.datasets:

        # Copy parameters, models, priors, etc. to self.
        self.__unwrap__ = kwargs["unwrap"]
        self.__unroll__ = kwargs["unroll"]
        self.__nroll__ = kwargs["nroll"]
        self.__rolls__ = rolls
        self.__models__ = models
        self.__modpars__ = modpars
        self.__noisemodel__ = noisemodel
        self.__priors__ = priors
        self.__var_names__ = vn # Change of name for consistency with result
        self.__fluxes_unwrap__ = fluxes_unwrap
        self.__n_unwrap__ = n_unwrap
        self.__scales__ = scales

        backend = kwargs['backend']
        if backend is None:
            iteration = 0
        else:
            try:
                iteration = backend.iteration
            except OSError:
                iteration = 0
        # Setup sampler
        vv = np.array(vv)
        vs = np.array(vs)
        n_varys = len(vv)
        nwalkers = kwargs['nwalkers']
        if iteration > 0:
            pos = None
        else:
            pos = []
            for i in range(nwalkers):
                lnpost_i = -np.inf
                it = 0
                while lnpost_i == -np.inf:
                    pos_i=vv+vs*np.random.randn(n_varys)*kwargs['init_scale']
                    lnpost_i, lnlike_i = self._lnpost_(pos_i)
                    it += 1
                    if it > _ITMAX_:  
                        for n,v,s, in zip(vn, vv, vs):
                            print(n,v,s)
                        raise Exception('Failed to initialize walkers')
                pos.append(pos_i)
    
        sampler = EnsembleSampler(nwalkers, n_varys, self._lnpost_,
                                  backend=backend)

        progress = kwargs['progress']
        if progress:
            print('Running burn-in ..')
            stdout.flush()
        if iteration == 0:
            pos,_,_,_ = sampler.run_mcmc(pos, kwargs['burn'], store=False, 
                skip_initial_state_check=True, progress=progress)
            sampler.reset()
        if progress:
            print('Running sampler ..')
            stdout.flush()
        state = sampler.run_mcmc(pos, kwargs['steps'], thin_by=kwargs['thin'], 
            skip_initial_state_check=True, progress=progress)

        # Run self._lnpost_ with best-fit parameters to obtain
        # best-fit light curves, detrended fluxes, etc.
        flatchain = sampler.get_chain(flat=True)
        pos = flatchain[np.argmax(sampler.get_log_prob()),:]
        f_fit,f_sys,f_det,f_sho,f_phi = self._lnpost_(pos,return_fit=True)
        self.__fluxes_fit__ = f_fit
        self.__fluxes_sys__ = f_sys
        self.__fluxes_det__ = f_det
        self.__fluxes_sho__ = f_sho
        self.__fluxes_phi__ = f_phi

        # lmfit MinimizerResult object summary of results for printing and
        # plotting. Data/objects required to re-run the analysis go directly
        # into self.

        result = MinimizerResult()
        result.status = 0
        result.var_names = vn
        result.covar = np.cov(flatchain.T)
        result.init_vals = vv
        result.init_values = copy(params.valuesdict())
        af = sampler.acceptance_fraction.mean()
        result.acceptance_fraction = af
        steps, nwalkers, ndim = sampler.get_chain().shape
        result.thin = kwargs['thin']
        result.nfev = int(kwargs['thin']*nwalkers*steps/af)
        result.nwalkers = nwalkers
        result.nvarys = ndim
        result.ndata = sum([len(d.lc['time']) for d in self.datasets])
        result.nfree = result.ndata - ndim
        result.method = 'emcee'
        result.errorbars = True
        result.bestfit = f_fit
        result.fluxes_det = f_det
        z = zip(self.datasets, f_fit)
        result.residual = [(d.lc['flux']-ft) for d,ft in z]
        z = zip(self.datasets, result.residual)
        result.chisqr = np.sum([((r/d.lc['flux_err'])**2).sum() for d,r in z])
        result.redchi = result.chisqr/result.nfree
        lnlike = np.max(sampler.get_blobs())
        result.lnlike = lnlike
        result.aic = 2*result.nvarys - 2*lnlike
        result.bic = result.nvarys*np.log(result.ndata) - 2*lnlike
        result.rms = np.array([r.std() for r in result.residual])
        result.npriors = len(self.__priors__)
        result.priors = self.__priors__
        
        quantiles = np.percentile(flatchain, [15.87, 50, 84.13], axis=0)
        corrcoefs = np.corrcoef(flatchain.T)
        parbest = params.copy()
        for i, n in enumerate(vn):
            std_l, median, std_u = quantiles[:, i]
            params[n].value = median
            params[n].stderr = 0.5 * (std_u - std_l)
            parbest[n].value = pos[i]
            parbest[n].stderr = 0.5 * (std_u - std_l)
            if n in self.__noisemodel__:
                self.__noisemodel__[n].value = median
                self.__noisemodel__[n].stderr = 0.5 * (std_u - std_l)
            correl = {}
            for j, n2 in enumerate(vn):
                if i != j:
                    correl[n2] = corrcoefs[i, j]
            params[n].correl = correl
            parbest[n].correl = correl
        result.params  = params
        result.parbest = parbest
        result.flat_chain = flatchain
        self.__parbest__ = parbest
        self.__result__ = result
        self.__sampler__ = sampler
        
        return result

#--------------------------------------------------------------------------

    def _lnpost_(self, pos, return_fit=False):
    
        lnlike = 0 
        if return_fit:
            fluxes_sys = []   # transits and eclipses only
            fluxes_fit = []   # lc fit per dataset
            fluxes_sho = []   # GP process from SHOTerm() kernel only
            fluxes_det = []   # detrended fluxes
            fluxes_phi = []   # Roll-angle trends if unroll=True

        # Update self.__noisemodel__ parameters
        vn = self.__var_names__
        noisemodel = self.__noisemodel__
        for p in ('log_sigma_w', 'log_omega0', 'log_S0', 'log_Q'):
            if p in vn:
                v = pos[vn.index(p)] 
                if (v < noisemodel[p].min) or (v > noisemodel[p].max):
                    return -np.inf, -np.inf
                noisemodel[p].set(value=v)
        if 'log_Q' in noisemodel:
            sho = SHOTerm(
                    S0=np.exp(noisemodel['log_S0'].value),
                    Q=np.exp(noisemodel['log_Q'].value),
                    w0=np.exp(noisemodel['log_omega0'].value))
        else:
            sho = False
    
        for i, dataset in enumerate(self.datasets):
            lc = dataset.lc 
            model = self.__models__[i]
            modpar = self.__modpars__[i]
            roll = self.__rolls__[i]
            f_unwrap = self.__fluxes_unwrap__[i]
    
            for p in ('T_0', 'P', 'D', 'W', 'b', 'f_c', 'f_s', 'l_3', 
                    'h_1', 'h_2', 'L', 'A_g', 'F_max', 'F_min', 'ph_off'):
                if p in vn:
                    v = pos[vn.index(p)]
                    if not np.isfinite(v): return -np.inf, -np.inf
                    if (v < modpar[p].min) or (v > modpar[p].max):
                        return -np.inf, -np.inf
                    modpar[p].value = v
    
            # Check that none of the derived parameters are out of range
            for p in ('e', 'q_1', 'q_2', 'k', 'aR',  'rho', 'L_0'):
                if p in modpar:
                    v = modpar[p].value
                    if not np.isfinite(v): return -np.inf, -np.inf
                    if (v < modpar[p].min) or (v > modpar[p].max):
                        return -np.inf, -np.inf
    
            for d in [k for k in modpar if k[:3]=='dfd' or k[:4]=='d2fd' or
                      k=='c' or k=='ramp' or k=='glint_scale']:
                p = f'{d}_{i+1:02d}' 
                if p in vn:
                    v = pos[vn.index(p)]
                    if (v < modpar[d].min) or (v > modpar[d].max):
                        return -np.inf, -np.inf
                    modpar[d].value = v
    
            p = f'ttv_{i+1:02d}'
            if p in vn:
                v = pos[vn.index(p)]
                modpar['T_0'].value = modpar['T_0'].init_value + v/86400
    
            p = f'L_{i+1:02d}'
            if p in vn:
                # Exclude negative eclipse depths
                if pos[vn.index(p)] < 0: 
                    return -np.inf, -np.inf
                modpar['L'].value = pos[vn.index(p)]
    
            # Evalate components of the model so that we can extract them
            f_model = model.eval(modpar, t=lc['time'])
            resid = lc['flux'] - f_unwrap - f_model
            yvar = np.exp(2*noisemodel['log_sigma_w']) + lc['flux_err']**2
    
            if roll or sho:
                if roll and sho:
                    kernel = sho + roll
                elif sho:
                    kernel = sho
                else:
                    kernel = roll
                gp = GaussianProcess(kernel)
                gp.compute(lc['time'], diag=yvar, quiet=True)
                if return_fit:
                    k = f'_{self.__fittype__}_func'
                    f_sys = model.eval_components(params=modpar,
                                                  t=lc['time'])[k]
                    fluxes_sys.append(f_sys)
                    f_celerite = gp.predict(resid, include_mean=False)
                    f_fit = f_model + f_celerite  + f_unwrap
                    fluxes_fit.append(f_fit)
                    if roll and sho:
                        f_sho = gp.predict(resid, include_mean=False,
                                kernel=gp.kernel.terms[0])
                        f_phi = gp.predict(resid, include_mean=False,
                                kernel=gp.kernel.terms[1])
                    elif sho:
                        f_sho = f_celerite
                        f_phi = np.zeros_like(resid)
                    else:
                        f_sho = np.zeros_like(resid)
                        f_phi = f_celerite

                    f_det = f_sys + f_sho + f_fit - lc['flux']
                    fluxes_det.append(f_det)
                    fluxes_sho.append(f_sho)
                    fluxes_phi.append(f_phi)
                else:
                    lnlike += gp.log_likelihood(resid)
            else:
                if return_fit:
                    k = f'_{self.__fittype__}_func'
                    f_sys = model.eval_components(params=modpar,
                                                  t=lc['time'])[k]
                    fluxes_sys.append(f_sys)
                    f_fit = f_model + f_unwrap
                    fluxes_fit.append(f_fit)
                    f_det = f_sys + f_fit - lc['flux']
                    fluxes_det.append(f_det)
                    fluxes_sho.append(np.zeros_like(f_sys))
                else:
                    lnlike += -0.5*np.sum(resid**2/yvar+np.log(2*np.pi*yvar))
    
        if return_fit:
            return fluxes_fit, fluxes_sys, fluxes_det, fluxes_sho, fluxes_phi
    
        args=[modpar[p] for p in ('D','W','b')]
        lnprior = _log_prior(*args)  # Priors on D, W and b
        if not np.isfinite(lnprior): return -np.inf, -np.inf
        for p in self.__priors__:
            pn = self.__priors__[p].n
            ps = self.__priors__[p].s
            if p in vn:
                z = (pos[vn.index(p)] - pn)/ps
            elif p in ('e', 'q_1', 'q_2', 'k', 'aR',  'rho',):
                z = (modpar[p] - pn)/ps
            elif p == 'logrho':
                z = (np.log10(modpar['rho']) - pn)/ps
            else:
                z = None
            if z is not None:
                lnprior += -0.5*(z**2 + np.log(2*np.pi*ps**2))
    
        if np.isnan(lnprior) or np.isnan(lnlike):
            return -np.inf, -np.inf

        return lnlike + lnprior, lnlike

#--------------------------------------------------------------------------

    def fit_transit(self, 
            steps=128, nwalkers=64, burn=256, 
            T_0=None, P=None, D=None, W=None, b=None, f_c=None, f_s=None,
            h_1=None, h_2=None, l_3=None,
            ttv=False, ttv_prior=3600, extra_priors=None, 
            log_sigma_w=None, log_omega0=None, log_S0=None, log_Q=None,
            unroll=True, nroll=3, unwrap=False, thin=1, 
            init_scale=0.5, progress=True, backend=None):
        """
        Use emcee to fit the transits in the current datasets 

        If T_0 and P are both fixed parameters then ttv=True can be used to
        include the free parameters ttv_i, the offset in seconds from the
        predicted time of mid-transit for each dataset i = 1, ..., N. The
        prior on the values of ttv_i is a Gaussian with a width ttv_prior in
        seconds.

        """
        # Get a dictionary of all keyword arguments excluding 'self'
        kwargs = dict(locals())
        del kwargs['self']
        self.__fittype__ = 'transit'

        return self.__run_emcee__(**kwargs)


#--------------------------------------------------------------------------

    def fit_eclipse(self, 
            steps=128, nwalkers=64, burn=256, 
            T_0=None, P=None, D=None, W=None, b=None, f_c=None, f_s=None,
            L=None, a_c=0, l_3=None, edv=False, edv_prior=1e-3,
            extra_priors=None, log_sigma_w=None, log_omega0=None,
            log_S0=None, log_Q=None, unroll=True, nroll=3, unwrap=False,
            thin=1, init_scale=0.5, progress=True, backend=None):
        """
        Use emcee to fit the eclipses in the current datasets 

        Eclipse depths variations can be included in the fit using the keyword
        edv=True. In this case L must be a fixed parameter and the eclipse
        depth for dataset i is L_i, i=1, ..., N. The prior on the values of
        L_i is a Gaussian with mean value L and width edv_prior.

        """
        # Get a dictionary of all keyword arguments excluding 'self'
        kwargs = dict(locals())
        del kwargs['self']
        self.__fittype__ = 'eclipse'

        return self.__run_emcee__(**kwargs)

#--------------------------------------------------------------------------

    def fit_eblm(self, steps=128, nwalkers=64, burn=256, 
            T_0=None, P=None, D=None, W=None, b=None, f_c=None, f_s=None, 
            h_1=None, h_2=None, l_3=None, ttv=False, ttv_prior=3600, 
            L=None, a_c=0, edv=False, edv_prior=1e-3, extra_priors=None, 
            log_sigma_w=None, log_omega0=None, log_S0=None, log_Q=None,
            unroll=True, nroll=3, unwrap=False, thin=1, 
            init_scale=0.5, progress=True, backend=None):
        """
        Use emcee to fit the transits and eclipses in the current datasets
        using a model for an eclipsing binary with a low-mass companion.

        The model does not account for the thermal/reflected phase effect.

        If T_0 and P are both fixed parameters then ttv=True can be used to
        include the free parameters ttv_i, the offset in seconds from the
        predicted time of mid-transit for each dataset i = 1, ..., N. The
        prior on the values of ttv_i is a Gaussian with a width ttv_prior in
        seconds.

        Eclipse depths variations can be included in the fit using the keyword
        edv=True. In this case L must be a fixed parameter and the eclipse
        depth for dataset i is L_i, i=1, ..., N. The prior on the values of
        L_i is a Gaussian with mean value L and width edv_prior.

        """
        # Get a dictionary of all keyword arguments excluding 'self'
        kwargs = dict(locals())
        del kwargs['self']
        self.__fittype__ = 'eblm'

        return self.__run_emcee__(**kwargs)

#--------------------------------------------------------------------------

    def fit_hotplanet(self, steps=128, nwalkers=64, burn=256, 
            T_0=None, P=None, D=None, W=None, b=None, f_c=None, f_s=None, 
            h_1=None, h_2=None, l_3=None, ttv=False, ttv_prior=3600, 
            F_max=None, F_min=0, ph_off=0,
            a_c=None, edv=False, edv_prior=1e-3, extra_priors=None, 
            log_sigma_w=None, log_omega0=None, log_S0=None, log_Q=None,
            unroll=True, nroll=3, unwrap=False, thin=1, 
            init_scale=0.5, progress=True, backend=None):
        """
        Use emcee to fit the transits and eclipses in the current datasets
        using the HotPlanetModel model.

        If T_0 and P are both fixed parameters then ttv=True can be used to
        include the free parameters ttv_i, the offset in seconds from the
        predicted time of mid-transit for each dataset i = 1, ..., N. The
        prior on the values of ttv_i is a Gaussian with a width ttv_prior in
        seconds.

        Eclipse depths variations can be included in the fit using the keyword
        edv=True. In this case F_max must be a fixed parameter and the value of
        F_max for dataset i is F_max_i, i=1, ..., N. The prior on the values of
        F_max_i is a Gaussian with mean value F_max and width edv_prior.

        By default, this method assumes ph_off=0 and F_min=0. The initial
        value of F_max is calculated from the best-fit values of L in the
        input eclipse datasets, if possible.

        """
        # Get a dictionary of all keyword arguments excluding 'self'
        kwargs = dict(locals())
        del kwargs['self']
        self.__fittype__ = 'hotplanet'

        return self.__run_emcee__(**kwargs)


#--------------------------------------------------------------------------

    def fit_planet(self, steps=128, nwalkers=64, burn=256, 
            T_0=None, P=None, D=None, W=None, b=None, f_c=None, f_s=None, 
            h_1=None, h_2=None, l_3=None, ttv=False, ttv_prior=3600, 
            A_g=None, a_c=None, edv=False, edv_prior=1e-3, extra_priors=None, 
            log_sigma_w=None, log_omega0=None, log_S0=None, log_Q=None,
            unroll=True, nroll=3, unwrap=False, thin=1, 
            init_scale=0.5, progress=True, backend=None):
        """
        Use emcee to fit the transits and eclipses in the current datasets
        using the PlanetModel model.

        If T_0 and P are both fixed parameters then ttv=True can be used to
        include the free parameters ttv_i, the offset in seconds from the
        predicted time of mid-transit for each dataset i = 1, ..., N. The
        prior on the values of ttv_i is a Gaussian with a width ttv_prior in
        seconds.

        Eclipse depths variations can be included in the fit using the keyword
        edv=True. In this case A_g must be a fixed parameter and the value of
        A_g for dataset i is A_g_i, i=1, ..., N. The prior on the values of
        F_max_i is a Gaussian with mean value F_max and width edv_prior.

        The initial value of A_g is estimated from the best-fit values of L
        in the input eclipse datasets assuming a circular orbit, if possible,
        i.e. A_g = (a/R_*)^2 * L/D

        The output from this method includes the derived parameter L_0, which
        is the eclipse depth calculated from A_g using the same expression.

        """
        # Get a dictionary of all keyword arguments excluding 'self'
        kwargs = dict(locals())
        del kwargs['self']
        self.__fittype__ = 'planet'

        return self.__run_emcee__(**kwargs)

#--------------------------------------------------------------------------

    def fit_report(self, **kwargs):
        """
        Return a string summarizing the results of the last emcee fit
        """
        result = self.__result__
        report = lmfit_report(result, **kwargs)
        n = [len(d.lc['time']) for d in self.datasets]
        rms = np.sqrt(np.average(result.rms**2,weights=n))*1e6
        s = "    RMS residual       = {:0.1f} ppm\n".format(rms)
        j = report.index('[[Variables]]')
        report = report[:j] + s + report[j:]
        noPriors = True
        params = result.params
        parnames = list(params.keys())
        namelen = max([len(n) for n in parnames])
        if result.npriors > 0: report+="\n[[Priors]]"
        for p in result.priors:
            q = result.priors[p]
            report += "\n    %s:%s" % (p, ' '*(namelen-len(p)))
            report += '%s +/-%s' % (gformat(q.n), gformat(q.s))

        report += '\n[[Notes]]'
        if self.__unroll__:
            report += '\n    Implicit roll-angle decorrelation used'
            report += f' nroll={self.__nroll__} terms'
        else:
            report += f'\n    Implicit roll-angle decorrelation not used.'
        if self.__unwrap__:
            report += '\n    Best-fit roll-angle decorrelation was subtracted'
            report += ' from light curves (unwrap=True)'
        else:
            report += '\n    Best-fit roll-angle decorrelation was not used'
            report += ' (unwrap=False)'

        for i,s in enumerate(self.__scales__):
            if s is not None:
                report += f'\n    Dataset {i+1}: '
                if s:
                    report += 'decorrelation parameters were scaled)'
                else:
                    report +='decorrelation parameters were not scaled'


        report += '\n[[Software versions]]'
        pipe_vers = ""
        for s in set([d.pipe_ver for d in self.datasets]):
            pipe_vers += f"{s}, "
        report += '\n    CHEOPS DRP : %s' % pipe_vers[:-2]
        report += '\n    pycheops   : %s' % __version__
        report += '\n    lmfit      : %s' % _lmfit_version_
        return(report)

    # ----------------------------------------------------------------

    def ttv_plot(self, plot_kws=None, figsize=(8,5)):
        """
        Plot results of TTV analysis

        The keyword plot_kws can be used to set keyword options in the call to
        plt.errorbar().

        """

        result = self.__result__
        if plot_kws is None:
            plot_kws={'fmt':'bo', 'capsize':4}
        fig,ax = plt.subplots(figsize=figsize)
        for j in range(len(self.datasets)):
            t = self.datasets[j].lc['time'].mean() - 1900
            ttv = result.params[f'ttv_{j+1:02d}'].value
            ttv_err = result.params[f'ttv_{j+1:02d}'].stderr
            ax.errorbar(t,ttv,yerr=ttv_err, **plot_kws)
            plt.axhline(0,c='darkcyan',ls=':')
            ax.set_xlabel('BJD - 2458900')
            ax.set_ylabel(r'$\Delta T$')
        return fig

    # ----------------------------------------------------------------

    def trail_plot(self, plotkeys=None, 
            plot_kws={'alpha':0.1}, width=8, height=1.5):
        """
        Plot parameter values v. step number for each walker.

        These plots are useful for checking the convergence of the sampler.

        The parameters width and height specifiy the size of the subplot for
        each parameter.

        The parameters to be plotted at specified by the keyword plotkeys, or
        plotkeys='all' to plot every jump parameter.

        The keyword plot_kws can be used to set keyword options in the plots.

        """

        result = self.__result__
        params = result.params
        samples = self.__sampler__.get_chain()
        var_names = result.var_names
        n = len(self.datasets)

        if plotkeys == 'all':
            plotkeys = var_names
        elif plotkeys is None:
            if self.__fittype__ == 'transit':
                l = ['D', 'W', 'b', 'T_0', 'P', 'h_1', 'h_2']
            elif self.__fittype__ == 'planet':
                l = ['D', 'W', 'b', 'T_0', 'P', 'A_g']
            elif self.__fittype__ == 'hotplanet':
                l = ['D', 'W', 'b', 'T_0', 'P', 'F_max']
            elif self.__fittype__ == 'eblm':
                l = ['D', 'W', 'b', 'T_0', 'P', 'L']
            elif 'L_01' in var_names:
                l = ['D','W','b']+[f'L_{j+1:02d}' for j in range(n)]
            else:
                l = ['L']+[f'c_{j+1:02d}' for j in range(n)]
            plotkeys = list(set(var_names).intersection(l))
            plotkeys.sort()

        n = len(plotkeys)
        fig,ax = plt.subplots(nrows=n, figsize=(width,n*height), sharex=True)
        if n == 1: ax = [ax,]

        d0 = 0 
        if 'T_0' in plotkeys:
            d0 = np.floor(np.nanmedian(samples[:,:,var_names.index('T_0')]))
        extra_labels = {}
        for i,d in enumerate(self.datasets):
            if d.extra_decorr_vectors != None:
                for k in d.extra_decorr_vectors:
                    if k == 't':
                        continue
                    if 'label' in d.extra_decorr_vectors[k].keys():
                        label = d.extra_decorr_vectors[k]['label']
                        label += f'$_{{{i+1:02d}}}$'
                        extra_labels[f'dfd{k}_{i+1:02d}'] = label
        labels = _make_labels(plotkeys, d0, extra_labels)

        for i,key in enumerate(plotkeys):
            if key == 'T_0':
                ax[i].plot(samples[:,:,var_names.index(key)]-d0, **plot_kws)
            else:
                ax[i].plot(samples[:,:,var_names.index(key)], **plot_kws)
            ax[i].set_ylabel(labels[i])
            ax[i].yaxis.set_label_coords(-0.1, 0.5)
        ax[-1].set_xlim(0, len(samples)-1)
        ax[-1].set_xlabel("step number");

        fig.tight_layout()
        return fig

    # ----------------------------------------------------------------

    def corner_plot(self, plotkeys=None, custom_labels=None, 
            show_priors=True, show_ticklabels=False,  kwargs=None):
        """
        Parameter correlation plot 

        Use custom_labels to change the string used for the axis labels, e.g.
        custom_labels={'F_max':r'$F_{\rm pl}/F_{\star}$'}

        :param plotkeys: list of variables to include in the corner plot
        :param custom_labels: dict of custom labels 
        :param show_priors: show +-1-sigma limits for Gaussian priors
        :param show_ticklabels: Show sumerical labels for tick marks
        :param kwargs: dict of keywords to pass through to corner.corner

        See also  https://corner.readthedocs.io/en/latest/ 

        """


        result = self.__result__
        params = result.params
        var_names = result.var_names
        n = len(self.datasets)

        if plotkeys == 'all':
            plotkeys = var_names
        if plotkeys == None:
            if self.__fittype__ == 'transit':
                l = ['D', 'W', 'b', 'T_0', 'P', 'h_1', 'h_2']
            elif self.__fittype__ == 'planet':
                l = ['D', 'W', 'b', 'T_0', 'P', 'A_g']
            elif self.__fittype__ == 'hotplanet':
                l = ['D', 'W', 'b', 'T_0', 'P', 'F_max']
            elif self.__fittype__ == 'eblm':
                l = ['D', 'W', 'b', 'T_0', 'P', 'L']
            elif 'L_01' in var_names:
                l = ['D','W','b']+[f'L_{j+1:02d}' for j in range(n)]
            else:
                l = ['L']+[f'c_{j+1:02d}' for j in range(n)]
            plotkeys = list(set(var_names).intersection(l))
            plotkeys.sort()

        chain = self.__sampler__.get_chain(flat=True)
        xs = []
        if 'T_0' in plotkeys:
            d0 = np.floor(np.nanmedian(chain[:,var_names.index('T_0')]))
        else:
            d0 = 0 

        if 'D' in var_names:
            k = np.sqrt(chain[:,var_names.index('D')])
        else:
            k = np.sqrt(params['D'].value) # Needed for later calculations

        if 'b' in var_names:
            b = chain[:,var_names.index('b')]
        else:
            b = params['b'].value  # Needed for later calculations

        if 'W' in var_names:
            W = chain[:,var_names.index('W')]
        else:
            W = params['W'].value  # Needed for later calculations

        aR = np.sqrt((1+k)**2-b**2)/W/np.pi
        sini = np.sqrt(1 - (b/aR)**2)

        for key in plotkeys:
            if key in var_names:
                if key == 'T_0':
                    xs.append(chain[:,var_names.index(key)]-d0)
                else:
                    xs.append(chain[:,var_names.index(key)])

            elif key == 'sigma_w' and params['log_sigma_w'].vary:
                xs.append(np.exp(self.emcee.chain[:,-1])*1e6)

            elif key == 'k' and 'D' in var_names:
                xs.append(k)

            elif key == 'aR':
                xs.append(aR)

            elif key == 'sini':
                xs.append(sini)

            elif key == 'logrho':
                if 'P' in var_names:
                    P = chain[:,var_names.index('P')]
                else:
                    P = params['P'].value   # Needed for later calculations
                logrho = np.log10(4.3275e-4*((1+k)**2-b**2)**1.5/W**3/P**2)
                xs.append(logrho)

            elif key == 'L_0':
                if 'A_g' in var_names:
                    L = chain[:,var_names.index('A_g')]*(k/aR)**2
                else:
                    L = params['A_g'].value*(k/aR)**2
                xs.append(L)
            else:
                raise ValueError(f'Variable {key} not in emcee chain')

        kws = {} if kwargs is None else kwargs

        xs = np.array(xs).T
        if custom_labels is None:
            extra_labels = {}
        else:
            extra_labels = custom_labels
        for i,d in enumerate(self.datasets):
            if d.extra_decorr_vectors != None:
                for k in d.extra_decorr_vectors:
                    if k == 't':
                        continue
                    if 'label' in d.extra_decorr_vectors[k].keys():
                        label = d.extra_decorr_vectors[k]['label']
                        label += f'$_{{{i+1:02d}}}$'
                        extra_labels[f'dfd{k}_{i+1:02d}'] = label
        labels = _make_labels(plotkeys, d0, extra_labels)

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
                q = params[key].user_data
                if isinstance(q, UFloat):
                    if key == 'T_0': q -= d0
                    ax = axes[i, i]
                    ax.axvline(q.n - q.s, color="g", linestyle='--')
                    ax.axvline(q.n + q.s, color="g", linestyle='--')
        return figure
        
    # ------------------------------------------------------------
    
    def cds_data_export(self, title=None, author=None, authors=None,
            abstract=None, keywords=None, bibcode=None,
            acknowledgements=None):
        '''
        Save light curve, best fit, etc. to files suitable for CDS upload

        Generates ReadMe file and data files with the following columns..
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
        F7.3  ---     temp_2   thermFront_2 temperature sensor reading

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
        cds = cdspyreadme.CDSTablesMaker()
        cds.title = title if title is not None else ""
        cds.author = author if author is not None else ""
        cds.authors = authors if author is not None else ""
        cds.abstract = abstract if abstract is not None else ""
        cds.keywords = keywords if keywords is not None else ""
        cds.bibcode = bibcode if bibcode is not None else ""
        cds.date = Time.now().value.year

        result = self.__result__
        par = result.parbest
        for j,d in enumerate(self.datasets):

            T=Table()
            T['time'] = d.lc['time'] + 2457000
            T['time'].info.format = '16.6f'
            T['time'].description = 'Time of mid-exposure'
            T['time'].units = u.day
            T['flux'] = d.lc['flux']
            T['flux'].info.format = '8.6f'
            T['flux'].description = 'Normalized flux'
            T['e_flux'] = d.lc['flux_err']
            T['e_flux'].info.format = '8.6f'
            T['e_flux'].description = 'Normalized flux error'
            T['flux_d'] = self.__fluxes_det__[j]
            T['flux_d'].info.format = '8.6f'
            T['flux_d'].description = (
                    'Normalized flux corrected for instrumental trends' )
            T['xoff'] = d.lc['xoff']
            T['xoff'].info.format = '8.4f'
            T['xoff'].description = "Target position offset in x-direction"
            T['yoff'] = d.lc['yoff']
            T['yoff'].info.format = '8.4f'
            T['yoff'].description = "Target position offset in y-direction"
            T['roll'] = d.lc['roll_angle']
            T['roll'].info.format = '8.4f'
            T['roll'].description = "Spacecraft roll angle"
            T['roll'].units = u.degree
            T['contam'] = d.lc['contam']
            T['contam'].info.format = '9.7f'
            T['contam'].description = (
                    "Fraction of flux in aperture from nearby stars" )
            if np.ptp(d.lc['smear']) > 0:
                T['smear'] = d.lc['smear']
                T['smear'].info.format = '9.7f'
                T['smear'].description = (
                        "Fraction of flux in aperture from readout trails" )
            T['bg'] = d.lc['bg']
            T['bg'].info.format = '9.7f'
            T['bg'].description = (
                    "Fraction of flux in aperture from background" )
            if np.ptp(d.lc['deltaT']) > 0:
                T['temp_2'] = d.lc['deltaT'] - 12
                T['temp_2'].info.format = '7.3f'
                T['temp_2'].description = (
                        "thermFront_2 temperature sensor reading" )
                T['temp_2'].units = u.Celsius
            table = cds.addTable(T, f'lc{j+1:02d}.dat',
                        description=f"Data from archive file {d.file_key}" )
            # Set output format
            for p in T.colnames:
                c=table.get_column(p)
                c.set_format(f'F{T[p].format[:-1]}')
            # Units
            c=table.get_column('time'); c.unit = 'd'
            c=table.get_column('xoff'); c.unit = 'pix'
            c=table.get_column('yoff'); c.unit = 'pix'
            c=table.get_column('roll'); c.unit = 'deg'

            cds.writeCDSTables()

        templatename = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    'data','cdspyreadme','ReadMe.template')
        coo = SkyCoord(self.datasets[0].lc['header']['RA_TARG'],
                    self.datasets[0].lc['header']['DEC_TARG'],unit='deg')
        rastr = coo.ra.to_string(unit='hour',sep=' ',precision=1, pad=True)
        destr = coo.dec.to_string(unit='deg',sep=' ',precision=0,
                alwayssign=True, pad=True)
        desc = (f'CHEOPS photometry of {self.target} generated using pycheops '+
                f'version {__version__}.')
    
        templateValue = {
                'object':f'{rastr} {destr}   {self.target}',
                'description':desc,
                'acknowledgements':acknowledgements
                }
        cds.setReadmeTemplate(templatename, templateValue)
        with open("ReadMe", "w") as fd:
            cds.makeReadMe(out=fd)
        
    # ------------------------------------------------------------
    
    def plot_fit(self, title=None, detrend=False, 
            binwidth=0.005, add_gaps=True, gap_tol=0.005, 
            data_offset=None, res_offset=None, phase0=None,
            xlim=None, data_ylim=None, res_ylim=None, renorm=True, 
            show_gp=True, figsize=None, fontsize=12):
        """
        If there are gaps in the data longer than gap_tol phase units and
        add_gaps is True then put a gap in the lines used to plot the fit. The
        transit/eclipse model is plotted using a thin line in these gaps.

        Binned data are plotted in phase bins of width binwidth. Set
        binwidth=False to disable this feature.

        The data are plotted in the range phase0 to 1+phase0.

        The offsets between the light curves from different datasets can be
        set using the data_offset keyword. The offset between the residuals
        from different datasets can be  set using the res_offset keyword. The
        y-axis limits for the data and residuals plots can be set using the
        data_ylim and res_ylim keywords, e.g. res_ylim = (-0.001,0.001).

        With renorm=True and detrend=False, each data set is re-scaled by the
        value of c_01, c_02, for that data set.

        For fits to datasets containing a mixture of transits and eclipses,
        data_offset and res_offset can be 2-tuples with the offsets for
        transits and eclipses, respectively. 

        For fits to datasets containing a mixture of transits and eclipses,
        the x-axis and y-axis limits for the data plots are specifed in the
        form ((min_left,max_left),(min_right,max-right))

        For fits that include a Gaussian process (GP), use show_gp=True to
        plot residuals that show the GP fit to the residuals, otherwise the
        residuals from fit includding the GP are shown.

        """
        n = len(self.datasets)
        par = self.__parbest__
        result = self.__result__
        P = par['P'].value
        T_0 = par['T_0'].value
        ph_fluxes = []   # Phases for observed/detrended fluxes
        fluxes = []      # observed/detrended fluxes
        resids = []      # Residuals for plotting (with correlated noise)
        ph_fits = []     # For plotting fits with lines - may contain np.nan
        fits = []        # Best fit - may contain np.nan to get gaps
        rednoise = []    # SHO GP fits (with np.nan for gaps)
        # Phases for models same as ph_fits. May contain np.nans for gaps
        lcmodels = []    # Model fluxes with transit+/eclipse effects only 
        ph_grid = []     # Grid of phases across one cycle
        lc_grid  = []     # Models evaulated across ph_grid
        iqrmax = 0  
        phmin = np.inf
        phmax = -np.inf
        if phase0 is None: phase0 = -0.25
        for j,dataset in enumerate(self.datasets):
            modpar = copy(self.__modpars__[j])
            ph = phaser(dataset.lc['time'], P, T_0, phase0)
            phmin = min([min(ph), phmin])
            phmax = max([max(ph), phmax])
            ph_fluxes.append(ph)

            if detrend:
                flux = self.__fluxes_det__[j]
                fit = copy(self.__fluxes_sys__[j] + self.__fluxes_sho__[j])
            else:
                if renorm:
                    if f'c_{j+1:02d}' in self.__parbest__:
                        c = self.__parbest__[f'c_{j+1:02d}'].value
                    else:
                        c = 1
                else:
                    c = 1
                flux = copy(dataset.lc['flux'])/c
                fit = copy(self.__fluxes_fit__[j])/c
            fluxes.append(flux)
            iqrmax = np.max([iqrmax, iqr(flux)])
            f_sho = self.__fluxes_sho__[j]
            if show_gp:
                resids.append(flux - fit + f_sho)
            else:
                resids.append(flux - fit)

            # Insert np.nan where there are gaps in phase so that the plotted
            # lines have a break
            lcmodel = copy(self.__fluxes_sys__[j])
            g = np.where((ph[1:]-ph[:-1]) > gap_tol)[0]
            if add_gaps and len(g) > 0:
                phmid = 0.5*(ph[1:]+ph[:-1])
                ph = np.insert(ph, g+1, phmid[g])
                fit = np.insert(fit, g+1, np.nan)
                lcmodel = np.insert(lcmodel, g+1, np.nan)
                f_sho = np.insert(f_sho, g+1, np.nan)
                
            ph_fits.append(ph)
            fits.append(fit)
            lcmodels.append(lcmodel)
            rednoise.append(f_sho)
            
            t0 = T_0+phase0*P
            tp = np.linspace(t0,t0+P,65536,endpoint=False)
            ph_grid.append(phaser(tp,P,T_0,phase0))
            model = self.__models__[j]
            k = f'_{self.__fittype__}_func'
            lc_grid.append(model.eval_components(params=modpar,t=tp)[k])

        plt.rc('font', size=fontsize)    
        if self.__fittype__ in ['eblm', 'hotplanet', 'planet']:

            f_c = par['f_c'].value
            f_s = par['f_s'].value
            ecc = f_c**2 + f_s**2
            omdeg = np.arctan2(f_s, f_c)*180/np.pi
            sini = par['sini'].value
            ph_sec = eclipse_phase(sini,ecc,omdeg)
            is_ecl = [min(abs(ph-ph_sec)) < 0.05 for ph in ph_fluxes]
            n_ecl = sum(is_ecl)
            n_tr = n-n_ecl
            if figsize is None:
                figsize = (8, 2+1.5*max(n_ecl,n_tr))
            fig,axes=plt.subplots(nrows=2,ncols=2, figsize=figsize,
                    gridspec_kw={'height_ratios':[2,1]})
            if data_offset is None:
                doff_tr,doff_ecl  = 2.5*iqrmax,2.5*iqrmax
            else:
                if np.isscalar(data_offset):
                    doff_tr,doff_ecl = data_offset, data_offset
                else:
                    doff_tr,doff_ecl = data_offset

            phmin_tr, phmax_tr = np.inf, -np.inf
            phmin_ecl, phmax_ecl = np.inf, -np.inf
            j_ecl, j_tr = 0, 0
            for (ph,flx,i) in zip(ph_fluxes, fluxes, is_ecl):
                if i:
                    off = j_ecl*doff_ecl
                    j_ecl += 1
                    ax = axes[0,1]
                    phmin_ecl = min([min(ph), phmin_ecl])
                    phmax_ecl = max([max(ph), phmax_ecl])
                else:
                    off = j_tr*doff_tr
                    j_tr += 1
                    ax = axes[0,0]
                    phmin_tr = min([min(ph), phmin_tr])
                    phmax_tr = max([max(ph), phmax_tr])
                ax.plot(ph, flx+off,'o',c='skyblue',ms=2, zorder=1)
                if binwidth:
                    r_, f_, e_, n_ = lcbin(ph, flx, binwidth=binwidth)
                    ax.errorbar(r_, f_+off, yerr=e_, fmt='o',
                            c='midnightblue', ms=5, capsize=2, zorder=3)

            j_ecl, j_tr = 0, 0
            for (ph,fit,lcmod,i) in zip(ph_fits,fits,lcmodels,is_ecl):
                if i:
                    off = j_ecl*doff_ecl
                    j_ecl += 1
                    ax = axes[0,1]
                else:
                    off = j_tr*doff_tr
                    j_tr += 1
                    ax = axes[0,0]
                k = np.argsort(ph)
                ax.plot(ph[k],fit[k]+off,c='saddlebrown', lw=2, zorder=4)
                if not detrend:
                    ax.plot(ph[k],lcmod[k]+off,c='forestgreen',zorder=2,lw=2)

            j_ecl, j_tr = 0, 0
            for (ph, fp, i) in zip(ph_grid, lc_grid, is_ecl):
                if i:
                    off = j_ecl*doff_ecl
                    j_ecl += 1
                    ax = axes[0,1]
                else:
                    off = j_tr*doff_tr
                    j_tr += 1
                    ax = axes[0,0]
                k = np.argsort(ph)
                ax.plot(ph[k],fp[k]+off,c='forestgreen', lw=1, zorder=2)

            roff = 10*np.max(result.rms)
            if res_offset is None:
                roff_tr,roff_ecl = roff,roff
            else:
                if np.isscalar(res_offset):
                    roff_tr,roff_ecl = res_offset, res_offset
                else:
                    roff_tr,roff_ecl = res_offset
            j_ecl = 0
            j_tr = 0
            for (ph,res,i) in zip(ph_fluxes,resids,is_ecl):
                if i:
                    off = j_ecl*roff_ecl
                    j_ecl += 1
                    ax = axes[1,1]
                else:
                    off = j_tr*roff_tr
                    j_tr += 1
                    ax = axes[1,0]
                ax.plot(ph, res+off,'o',c='skyblue',ms=2)
                ax.axhline(off, color='saddlebrown',ls=':')
                if binwidth:
                    r_, f_, e_, n_ = lcbin(ph, res, binwidth=binwidth)
                    ax.errorbar(r_, f_+off, yerr=e_,
                            fmt='o', c='midnightblue', ms=5, capsize=2)

            if show_gp:
                j_ecl, j_tr = 0, 0
                for ph,rn,i in zip(ph_fits, rednoise, is_ecl):
                    if i:
                        off = j_ecl*roff_ecl
                        j_ecl += 1
                        ax = axes[1,1]
                    else:
                        off = j_tr*roff_tr
                        j_tr += 1
                        ax = axes[1,0]
                    ax.plot(ph, rn+off, lw=2, c='saddlebrown')

            axes[0,0].set_xticklabels([])
            axes[0,1].set_xticklabels([])

            if xlim is None:
                pad = (phmax_tr-phmin_tr)/10
                pht = max([abs(phmin_tr), abs(phmax_tr)])
                axes[0,0].set_xlim(-pht-pad,pht+pad)
                axes[1,0].set_xlim(-pht-pad,pht+pad)
                pad = (phmax_ecl-phmin_ecl)/10
                axes[0,1].set_xlim(phmin_ecl-pad,phmax_ecl+pad)
                axes[1,1].set_xlim(phmin_ecl-pad,phmax_ecl+pad)
            else:
                axes[0,0].set_xlim(*xlim[0])
                axes[1,0].set_xlim(*xlim[0])
                axes[0,1].set_xlim(*xlim[1])
                axes[1,1].set_xlim(*xlim[1])
        
            if data_ylim is not None:
                axes[0,0].set_ylim(*data_ylim[0])
                axes[0,1].set_ylim(*data_ylim[1])
            if detrend:
                axes[0,0].set_ylabel('Flux-trend')
            else:
                axes[0,0].set_ylabel('Flux')
            axes[0,0].set_title(title)
            if res_ylim is None:
                if roff_tr != 0:
                    axes[1,0].set_ylim(np.sort([-0.75*roff_tr,
                         roff_tr*(n_tr-0.25)]))
                else:
                    axes[1,0].set_ylim(-roff, roff)
                if roff_ecl != 0:
                    ax = axes[1,1]
                    ax.set_ylim(np.sort([-0.75*roff_ecl,roff_ecl*(n_ecl-0.25)]))
                else:
                    axes[1,1].set_ylim(-roff, roff)
            else:
                axes[1,0].set_ylim(*res_ylim[0])
                axes[1,1].set_ylim(*res_ylim[1])
        
            axes[1,0].set_xlabel('Phase')
            axes[1,1].set_xlabel('Phase')
            axes[1,0].set_ylabel('Residual')

        else: # Not EBLM or Planet or HotPlanet

            if figsize is None:
                figsize = (8, 2+1.5*n)
            fig,ax=plt.subplots(nrows=2,sharex=True, figsize=figsize,
                    gridspec_kw={'height_ratios':[2,1]})
        
            doff = 2.5*iqrmax if data_offset is None else data_offset
            for j, (ph, flx) in enumerate(zip(ph_fluxes, fluxes)):
                off = j*doff
                ax[0].plot(ph, flx+off,'o',c='skyblue',ms=2, zorder=1)
                if binwidth:
                    r_, f_, e_, n_ = lcbin(ph, flx, binwidth=binwidth)
                    ax[0].errorbar(r_, f_+off, yerr=e_, fmt='o',
                            c='midnightblue', ms=5, capsize=2, zorder=3)
        
            for j, (ph,fit,lcmod) in enumerate(zip(ph_fits,fits,lcmodels)):
                off = j*doff
                k = np.argsort(ph)
                ax[0].plot(ph[k],fit[k]+off,c='saddlebrown', lw=2, zorder=4)
                if not detrend:
                    ax[0].plot(ph[k],lcmod[k]+off,c='forestgreen',zorder=2,lw=2)
        
            for j, (ph, fp) in enumerate(zip(ph_grid, lc_grid)):
                off = j*doff
                k = np.argsort(ph)
                ax[0].plot(ph[k],fp[k]+off,c='forestgreen', lw=1, zorder=2)
        
            roff = 10*np.max(result.rms) if res_offset is None else res_offset
            for j, (ph,res) in enumerate(zip(ph_fluxes, resids)):
                off=j*roff
                ax[1].plot(ph, res+off,'o',c='skyblue',ms=2)
                ax[1].axhline(off, color='saddlebrown',ls=':')
                if binwidth:
                    r_, f_, e_, n_ = lcbin(ph, res, binwidth=binwidth)
                    ax[1].errorbar(r_, f_+off, yerr=e_,
                            fmt='o', c='midnightblue', ms=5, capsize=2)
            if show_gp:
                for j, (ph,rn) in enumerate(zip(ph_fits, rednoise)):
                    off=j*roff
                    ax[1].plot(ph, rn+off, lw=2, c='saddlebrown')
        
            if xlim is None:
                pad = (phmax-phmin)/10
                if self.__fittype__ == "transit":
                    pht = max([abs(phmin), abs(phmax)])
                    ax[1].set_xlim(-pht-pad,pht+pad)
                else:
                    ax[1].set_xlim(phmin-pad,phmax+pad)
            else:
                ax[1].set_xlim(*xlim)
        
            if data_ylim is not None: ax[0].set_ylim(*data_ylim)
            if detrend:
                ax[0].set_ylabel('Flux-trend')
            else:
                ax[0].set_ylabel('Flux')
            ax[0].set_title(title)
            if res_ylim is None:
                if roff != 0:
                    ax[1].set_ylim(np.sort([-0.75*roff, roff*(n-0.25)]))
                else:
                    rms = np.max(result.rms)
                    ax[1].set_ylim(-5*rms, 5*rms)
            else:
                ax[1].set_ylim(*res_ylim)
        
            ax[1].set_xlabel('Phase')
            ax[1].set_ylabel('Residual')

        fig.tight_layout()
        return fig
        
    # ------------------------------------------------------------

    def tzero(self, BJD_0, P):
        '''
        Return the time of mid-transit closest to the centre of the combined
        dataset as BJD-2457000, i.e., on the same time scale as the data.

        :param BJD_0: BJD of mid-transit - float or ufloat
        :param P: orbital period in days - float or ufloat

        Returns

        :param T_0: time of mid-transit, BJD-2457000, float or ufloat
        '''
        t = np.mean([d.lc['time'].mean() for d in self.datasets]) 
        c = (t-BJD_0+2457000)/P
        if isinstance(c, UFloat): c = c.n
        return BJD_0-2457000 + round(c)*P
        
    # ------------------------------------------------------------

    def massradius(self, m_star=None, r_star=None, K=None, q=0, 
            jovian=True, plot_kws=None, return_samples=False,
            verbose=True):
        '''
        Use the results from the previous transit light curve fit to estimate
        the mass and/or radius of the planet.

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

        flatchain = self.__sampler__.get_chain(flat=True)
        vn = self.__result__.var_names
        pars = self.__result__.params
        # Generate value(s) from previous emcee sampler run
        def _v(p):
            if (p in vn):
                v = flatchain[:,vn.index(p)]
            elif p in pars.valuesdict().keys():
                v = pars[p].value
            else:
                raise AttributeError(
                        'Parameter {} missing from dataset'.format(p))
            return v
    
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

    
        # Generate samples for derived parameters not specified by the user
        # from the chain rather than the summary  statistics 
        k = np.sqrt(_v('D'))
        b = _v('b')
        W = _v('W')
        P = _v('P')
        aR = np.sqrt((1+k)**2-b**2)/W/np.pi
        sini = np.sqrt(1 - (b/aR)**2)
        f_c = _v('f_c')
        f_s = _v('f_s')
        ecc = f_c**2 + f_s**2
        _q = _s(q, len(flatchain))
        rho_star = rhostar(1/aR,P,_q)
        # N.B. use of np.abs to cope with values with large errors
        if r_star is None and m_star is not None:
            _m = np.abs(_s(m_star, len(flatchain)))
            r_star = (_m/rho_star)**(1/3)
        if m_star is None and r_star is not None:
            _r = np.abs(_s(r_star, len(flatchain)))
            m_star = rho_star*_r**3

        if m_star is None and r_star is not None:
            if isinstance(r_star, tuple):
                _r = ufloat(r_star[0], r_star[1])
            else:
                _r = r_star
            m_star = rho_star*_r**3
        if verbose:
            print('[[Mass/radius]]')
       
        if plot_kws is None:
            plot_kws = {}
       
        return massradius(P=P, k=k, sini=sini, ecc=ecc,
                m_star=m_star, r_star=r_star, K=K, aR=aR,
                jovian=jovian, return_samples=return_samples,
                verbose=verbose, **plot_kws)
    
    #------

    def save(self, tag="", overwrite=False):
        """
        Save the current MultiVisit instance as a pickle file

        :param tag: string to tag different versions of the same MultiVisit

        :param overwrite: set True to overwrite existing version of file

        :returns: pickle file name
        """
        fl = self.target.replace(" ","_")+'_'+tag+'.multivisit'
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
        Load a MultiVisit from a pickle file

        :param filename: pickle file name

        :returns: MultiVisit object
        
        """
        with open(filename, 'rb') as fp:
            self = pickle.load(fp)
        return self

    #------

    def __getstate__(self):

        state = self.__dict__.copy()

        # Replace lmfit models with their string representation
        if '__models__' in state.keys():
            state['__models__'] = [m.__repr__() for m in state['__models__']]
        else:
            state['__models__'] = []

        return state

    #------

    def __setstate__(self, state):

        self.__dict__.update(state)

        models = []
        for model_repr,d in zip(self.__models__, self.datasets):
            t = d.lc['time']
            try:
                smear = d.lc['smear']
            except KeyError:
                smear = np.zeros_like(t)
            try:
                deltaT = d.lc['deltaT']
            except KeyError:
                deltaT = np.zeros_like(t)
            if d.__scale__:
                factor_model = FactorModel(
                    dx = _make_interp(t,d.lc['xoff'], scale='range'),
                    dy = _make_interp(t,d.lc['yoff'], scale='range'),
                    bg = _make_interp(t,d.lc['bg'], scale='range'),
                    contam = _make_interp(t,d.lc['contam'], scale='range'),
                    smear = _make_interp(t,smear, scale='range'),
                    deltaT = _make_interp(t,deltaT),
                    extra_basis_funcs=d.__extra_basis_funcs__)
            else:
                factor_model = FactorModel(
                    dx = _make_interp(t,d.lc['xoff']),
                    dy = _make_interp(t,d.lc['yoff']),
                    bg = _make_interp(t,d.lc['bg']),
                    contam = _make_interp(t,d.lc['contam']),
                    smear = _make_interp(t,smear),
                    deltaT = _make_interp(t,deltaT),
                    extra_basis_funcs=d.__extra_basis_funcs__)

            if self.__fittype__ == 'transit':
                model = TransitModel()*factor_model
            elif self.__fittype__ == 'eclipse':
                model = EclipseModel()*factor_model
            elif self.__fittype__ == 'eblm':
                model = EBLMModel()*factor_model
            elif self.__fittype__ == 'hotplanet':
                model = HotPlanetModel()*factor_model
            elif self.__fittype__ == 'planet':
                model = PlanetModel()*factor_model

            if 'glint_func' in model_repr:
                delta_t = d._old_bjd_ref - d.bjd_ref
                model += Model(_glint_func, independent_vars=['t'],
                    f_theta=d.f_theta, f_glint=d.f_glint, delta_t=delta_t)

            models.append(model)

        self.__models__ = models

