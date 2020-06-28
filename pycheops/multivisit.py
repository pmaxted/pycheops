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
from os import path
from .dataset import Dataset
from .starproperties import StarProperties
import re
from warnings import warn
from .dataset import _kw_to_Parameter,  _log_prior
from .dataset import _make_interp, _glint_func
from lmfit import Parameters, Parameter
from lmfit import fit_report as lmfit_report
from lmfit import __version__ as _lmfit_version_
from . import __version__
import autograd.numpy as ag
from lmfit.models import ExpressionModel, Model
from lmfit.minimizer import MinimizerResult
from celerite.terms import Term, SHOTerm, JitterTerm
from .models import TransitModel, FactorModel, EclipseModel
from celerite import GP
from .funcs import rhostar, massradius
from uncertainties import UFloat, ufloat
from emcee import EnsembleSampler
import corner
from sys import stdout
import matplotlib.pyplot as plt
from lmfit.printfuncs import gformat
from copy import copy
from .utils import phaser, lcbin
from scipy.stats import iqr


#--------

class CosineTerm(Term):
    def __init__(self, *args, **kwargs):
        self.parameter_names = ("log_omega_j", "log_sigma_j")
        super(CosineTerm, self).__init__(*args, **kwargs)
    def __repr__(self):
        return ("CosineTerm({0.log_omega_j}, {0.log_sigma_j})"
                .format(self))
    def get_complex_coefficients(self, params):
        log_omega_j, log_sigma_j = params
        return ag.exp(log_sigma_j), 0., 0., ag.exp(log_omega_j)

#--------

def fit_harmonics(t, y, omega, n):
    expr = 'c'
    init = {'c':0}
    for i in range(n):
        expr += f' + a_{i+1}*sin({i+1}*omega*x)'
        expr += f' + b_{i+1}*cos({i+1}*omega*x)'
        init[f'a_{i+1}'] = 0
        init[f'b_{i+1}'] = 0
    m = ExpressionModel(expr)
    p = m.make_params(**init)
    r = m.fit(y,p,x=x)
    return r.residual

#--------

SineModel = ExpressionModel('sin(2*pi*(x-x0)/P)')

#--------

def _make_model(model_repr, lc, f_theta=None, f_glint=None):
    t = lc['time']
    factor_model = FactorModel(
            dx = _make_interp(t,lc['xoff'], scale='range'),
            dy = _make_interp(t,lc['yoff'], scale='range'),
            bg = _make_interp(t,lc['bg'], scale='max'),
            contam = _make_interp(t,lc['contam'], scale='max'))
    if '_transit_func' in model_repr:
        model = TransitModel()*factor_model
    elif '_eclipse_func' in model_repr:
        model = EclipseModel()*factor_model
    if 'glint_func' in model_repr:
        model += Model(_glint_func, independent_vars=['t'],
            f_theta=f_theta, f_glint=f_glint)
    return model

#---------------

def _make_labels(plotkeys, d0):
    labels = []
    r = re.compile('dfd(.*)_([0-9]+)')
    r2 = re.compile('d2fd(.*)2_([0-9]+)')
    rt = re.compile('ttv_([0-9]+)')
    rl = re.compile('L_([0-9]+)')
    for key in plotkeys:
        if key == 'T_0':
            labels.append(r'T$_0-{:.0f}$'.format(d0))
        elif key == 'h_1':
            labels.append(r'$h_1$')
        elif key == 'h_2':
            labels.append(r'$h_2$')
        elif r.match(key):
            p,n = r.match(key).group(1,2)
            labels.append(r'$df\,/\,d{}_{}$'.format(p,n))
        elif r2.match(key):
            p,n = r2.match(key).group(1,2)
            labels.append(r'$d^2f\,/\,d{}^2_{}$'.format(p,n))
        elif rt.match(key):
            n = rt.match(key).group(1)
            labels.append(r'$\Delta\,T_{}$'.format(n))
        elif rl.match(key):
            n = rl.match(key).group(1)
            labels.append(r'$L_{}$'.format(n))
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
            labels.append(r'a\,/\,R$_{\star}$')
        elif key == 'sini':
            labels.append(r'\sin i')
        else:
            labels.append(key)
    return labels

#--------

def _log_posterior(pos, gps, lcs, models, modpars, noisemodel, priors, vn,
        return_fit):

    lnprob = 0 
    lc_mods = []    # model light curves per dataset
    lc_fits = []    # lc fit per dataset
    for i, (gp,lc,model,modpar) in enumerate(zip(gps,lcs,models,modpars)):

        for p in ('T_0', 'P', 'D', 'W', 'b', 'f_c', 'f_s', 'h_1', 'h_2', 'L'):
            if p in vn:
                v = pos[vn.index(p)]
                if (v < modpar[p].min) or (v > modpar[p].max): return -np.inf
                modpar[p].value = v

        # Check that none of the derived parameters are out of range
        for p in ('e', 'q_1', 'q_2', 'k', 'aR',  'rho',):
            if p in modpar:
                v = modpar[p].value
                if (v < modpar[p].min) or (v > modpar[p].max): return -np.inf

        df = ('c', 'dfdbg', 'dfdcontam', 'glint_scale',
                'dfdx', 'd2fdx2', 'dfdy', 'd2fdy2', 'dfdt', 'd2fdt2')
        for d in df:
            p = f'{d}_{i+1}' 
            if p in vn:
                v = pos[vn.index(p)]
                if (v < modpar[d].min) or (v > modpar[d].max): return -np.inf
                modpar[d].value = v

        p = f'ttv_{i+1}'
        if p in vn:
            v = pos[vn.index(p)]
            modpar['T_0'].value = modpar['T_0'].init_value + v/86400

        p = f'L_{i+1}'
        if p in vn:
            modpar['L'].value = pos[vn.index(p)]

        # Noise model parameters
        for p in ('log_sigma_w', 'log_omega0', 'log_S0', 'log_Q'):
            if p in vn:
                v = pos[vn.index(p)] if p in vn else noisemodel[p].value
                if (v < noisemodel[p].min) or (v > noisemodel[p].max):
                    return -np.inf
                gp.set_parameter(noisemodel[p].user_data['kname'], v)

        mod = model.eval(modpar, t=lc['time'])
        resid = lc['flux'] - mod
        lnprob += gp.log_likelihood(resid)

        if return_fit:
            k = noisemodel['log_sigma_w'].user_data['kname']
            if k == 'kernel:log_sigma':
                lc_fits.append(mod)
            else:
                lc_fits.append(gp.predict(resid, return_cov=False) + mod)

    if return_fit:
        return lc_fits

    args=[modpar[p] for p in ('D','W','b')]
    lnprior = _log_prior(*args)
    for p in priors:
        if p in vn:
            lnprob += -0.5*( (pos[vn.index(p)] - priors[p].n)/priors[p].s)**2

    return lnprob + lnprior

#--------

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
      e.g., dfdx=(-1,1). The initial value is taken as the the mid-point of
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

    N.B. The timescale for T_0 in BJD_TDB - 2457000.

    Priors on transit parameters are only set if they are specified in the
    call to the fitting method using either a ufloat, or as an lmfit Parameter
    object that includes a ufloat in its user_data.

    Priors on the derived parameters e, q_1, q_2, logrho, etc. can be
    specified as a dictionary of ufloat values using the extra_priors
    keyword, e.g., extra_priors={'e':ufloat(0.2,0.01)}
    
    Noise model
    ~~~~~~~~~~~

    The noise model is calculated with celerite using the a white-noise
    term JitterTerm(log_sigma_w) plus, optionally, a GP with kernel
    SHOTerm(log_omega0, log_S0, log_Q). The same values of log_sigma_w,
    log_omega0, log_S0 and log_Q are used for all the datasets in the
    combined fit.
    
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
    terms is set by nroll, e.g., setting nroll=3 (recommended) includes terms
    up to sin(3.phi) and cos(3.phi). This requires that phi is a linear
    function of time for each dataset, which is a very good approximation for
    individual CHEOPS visits. 
    
    Other decorrelation parameters not derived from the roll angle, e.g.,
    dfdx, dfdy, etc. are included in the fit to individual datasets only if
    they were free parameters in the last fit to that dataset. The
    decorrelation is done independently for each dataset. The free parameters
    are labelled dfdx_i, dfdy_i where i is the number of the dataset to which
    each decorrelation parameter applies. 

    Glint correction is done independently for each dataset if the glint
    correction was included in the last fit to that dataset. The glint
    scale factor for dataset i is labelled glint_scale_i. The glint
    scaling factor for each dataset can either be a fixed or a free
    parameter, depending on whether it was a fixed or a free parameter in
    the last fit to that dataset.

    """

    def __init__(self, target=None, datadir=None,
            ident=None, id_kws={'dace':True},
            verbose=True):

        self.target = target
        self.datadir = datadir
        self.datasets = []

        if target is None: return

        ptn = target.replace(" ","_")+'__CH*.dataset'
        if datadir is not None:
            ptn = join(datadir,ptn)

        g = glob(ptn)
        if len(g) == 0:
            warn(f'No matching dataset names for target {target}', UserWarning)
            return

        if ident is not 'none':
            if ident is None: ident = target 
            self.star = StarProperties(ident, **id_kws)

        if verbose:
            print(self.star)
            print('''
 N  file_key                   Aperture last_ GP  Glint pipe_ver
 ---------------------------------------------------------------------------''')

        for n,fl in enumerate(glob(ptn)):
            d = Dataset.load(fl)

            # Make time scales consistent
            dBJD = d.bjd_ref - 2457000
            d.bjd_ref = 2457000
            d.lc['time'] += dBJD
            d.lc['bjd_ref'] = 0
            if 'lmfit' in d.__dict__:
                p = d.lmfit.params['T_0']
                p._val += dBJD
                p.init_value += dBJD
                p.min += dBJD
                p.max += dBJD
                if 'T_0' in d.lmfit.var_names: 
                    d.lmfit.init_vals[d.lmfit.var_names.index('T_0')] += dBJD
                if 'T_0' in d.lmfit.init_values: 
                    d.lmfit.init_values['T_0'] += dBJD

            if 'emcee' in d.__dict__:
                p = d.emcee.params['T_0']
                p._val += dBJD
                p.init_value += dBJD
                p.min += dBJD
                p.max += dBJD
                if 'T_0' in d.emcee.var_names: 
                    d.emcee.init_vals[d.emcee.var_names.index('T_0')] += dBJD
                if 'T_0' in d.emcee.init_values: 
                    d.emcee.init_values['T_0'] += dBJD

            self.datasets.append(d)
            if verbose:
                dd = d.__dict__
                ap = d.lc['aperture'] if 'lc' in dd else '---'
                lf = d.__lastfit__ if '__lastfit__' in dd else '---'
                gp = 'Yes' if ('gp' in dd and d.gp is not None) else 'No'
                gl = 'Yes' if 'f_glint' in dd else 'No'
                pv = d.pipe_ver
                print(f' {n+1:2} {d.file_key} {ap:8} {lf:5} {gp:3} {gl:5} {pv}')

#--------------------------------------------------------------------------
# 
# Some big chunks of code here common to all the fitting routines, mostly just
# parameter handling and model creation.
#
#  "params" is an lmfit Parameters object that is used for storing
#   the results, initial values, etc. Not passed to the target
#   log-posterior function.
#
#  "models" is a list of lmfit models that get evaluated in the target
#  log-posterior function. These models including a TransitModel,
#  FactorModel and may also have a GlintModel. 
#
#  "modpars" is a list of Parameters objects, one for each dataset.
#  These parameters used to evaluate the models in "models". These all
#  have the same transit model parameters, but different decorrelation
#  parameters sent to FactorModel for each dataset. 
#
#  "noisemodel" is an lmfit Parameters object used for passing the
#  noise model parameters log_sigma_w, log_omega0, etc. to the target
#  log-posterior function. The user_data attribute of each parameter
#  is a dict that has at least one entry "kname" with the name of the
#  parameter as it appears in the celerite GP kernel object. It may
#  also contain an entry "prior" with the Gaussian prior on the value
#  stored as a ufloat. 
#
# "priors" is a list of priors stored as ufloat values. 
#
#  "vn" is a list of the free parameters in the combined fit.

    def __make_params__(self, vals, priors, pmin, pmax, step, extra_priors):

        plist = [d.emcee.params if d.__lastfit__ == 'emcee' else 
                 d.lmfit.params for d in self.datasets]
        params = Parameters()  
        vv,vs,vn  = [],[],[]     # Free params for emcee, name value, err
        for k in vals:
            if vals[k] is None:    # No user-defined value
                if k is 'T_0':
                    t = np.array([p[k].value for p in plist])
                    c = np.round((t-t[0])/params['P'])
                    c -= c.max()//2
                    t -= c*params['P']
                    val = t.mean()
                    vmin = val - params['W']*params['P']/2
                    vmax = val + params['W']*params['P']/2
                else:
                    val = np.array([p[k].value for p in plist]).mean()
                    vmin = np.array([p[k].min for p in plist]).max()
                    vmax = np.array([p[k].max for p in plist]).min()
                vary = True in [p[k].vary for p in plist]
                params.add(k, val, vary=vary, min=vmin, max=vmax)
                vals[k] = val
            else:
                params[k] = _kw_to_Parameter(k, vals[k])
                vals[k] = params[k].value

            if params[k].vary:
                vn.append(k)
                vv.append(params[k].value)
                if isinstance(params[k].user_data, UFloat):
                    priors[k] = params[k].user_data
                # Step size for setting up initial walker positions
                if params[k].stderr is None:
                    if params[k].user_data is None:
                        vs.append(step[k])
                    else:
                        vs.append(params[k].user_data.s)
                else:
                    if np.isfinite(params[k].stderr):
                        vs.append(params[k].stderr)
                    else:
                        vs.append(0.01*(params[k].max-params[k].min))
            else:
                # Needed to avoid errors when printing parameters
                params[k].stderr = None

        # Derived parameters
        params.add('k',expr='sqrt(D)',min=0,max=1)
        params.add('aR',expr='sqrt((1+k)**2-b**2)/W/pi',min=1)
        params.add('sini',expr='sqrt(1 - (b/aR)**2)')
        # Avoid use of aR in this expr for logrho - breaks error propogation.
        expr = 'log10(4.3275e-4*((1+k)**2-b**2)**1.5/W**3/P**2)'
        params.add('logrho',expr=expr,min=-9,max=6)
        params.add('e',min=0,max=1,expr='f_c**2 + f_s**2')
        if 'h_1' in params:
            params.add('q_1',min=0,max=1,expr='(1-h_2)**2')
            params.add('q_2',min=0,max=1,expr='(h_1-h_2)/(1-h_2)')
        if extra_priors is not None:
            for k in extra_priors:
                if k in params:
                    params[k].user_data = extra_priors[k]
        return vn,vv,vs,params

#--------------------------------------------------------------------------

    def __make_noisemodel__(self, log_sigma_w, log_S0, log_omega0, log_Q,
            params, priors, vn, vv, vs, unroll):

        noisemodel = Parameters()  
        k = 'log_sigma_w'
        if log_sigma_w is None:
            noisemodel.add(k, -6, min=-12, max=-2)
        else:
            noisemodel[k] = _kw_to_Parameter(k, log_sigma_w)
            # Avoid crazy-low values that are consistent with sigma_w = 0
            if not np.isfinite(noisemodel[k].min):
                noisemodel[k].min = np.min([noisemodel[k].value-10, -30])
        params[k] = copy(noisemodel[k])
        # Convert user_data to a dict so we can put kernel name there
        if isinstance(noisemodel[k].user_data, UFloat):
            priors[k] = noisemodel[k].user_data
            noisemodel[k].user_data = {'prior':noisemodel[k].user_data}
        else:
            noisemodel[k].user_data = {}
        if noisemodel[k].vary:
            vn.append(k)
            vv.append(noisemodel[k].value)
            vs.append(1)

        if log_S0 is not None and log_omega0 is not None:
            k = 'log_sigma_w'
            noisemodel[k].user_data['kname'] = 'kernel:terms[0]:log_sigma'
            if log_Q is None: log_Q = 1/np.sqrt(2)
            nvals = {'log_S0':log_S0, 'log_omega0':log_omega0, 'log_Q':log_Q}
            for k in nvals:
                noisemodel[k] = _kw_to_Parameter(k, nvals[k])
                params[k] = copy(noisemodel[k])
                if isinstance(noisemodel[k].user_data, UFloat):
                    priors[k] = noisemodel[k].user_data
                    noisemodel[k].user_data = {'prior':noisemodel[k].user_data}
                else:
                    noisemodel[k].user_data = {}
                if noisemodel[k].vary:
                    vn.append(k)
                    vv.append(noisemodel[k].value)
                    vs.append(1)
                noisemodel[k].user_data['kname'] = f'kernel:terms[1]:{k}'
        else:
            k = 'log_sigma_w'
            n = 'kernel:terms[0]:log_sigma' if unroll else 'kernel:log_sigma'
            noisemodel[k].user_data['kname'] = n

        return noisemodel, params, priors, vn, vv, vs

#--------------------------------------------------------------------------

    def __make_modpars__(self, model_type, vals, params, noisemodel,
             vn, vv, vs, priors, ttv_edv, ttv_edv_prior, unroll, nroll):
        plist = [d.emcee.params if d.__lastfit__ == 'emcee' else 
                 d.lmfit.params for d in self.datasets]

        ttv, edv = False, False
        if ttv_edv:
            if model_type == '_transit_func':
                ttv, ttv_prior = True, ttv_edv_prior
            if model_type == '_eclipse_func':
                edv, edv_prior = True, ttv_edv_prior
        
        gps = []      # Separate GPs because over o/h to compute matrices
        lcs = []
        models = []
        modpars = []
        glints = []
        omegas = []
        # FactorModel parameters excluding cos(j.phi), sin(j.phi) terms
        dfdp = ['c', 'dfdbg', 'dfdcontam', 'dfdx', 'd2fdx2', 'dfdy', 'd2fdy2',
                'dfdt', 'd2fdt2', 'glint_scale']
        for i,p in enumerate(plist):
            lc = self.datasets[i].lc
            lcs.append(lc)
            if 'glint_scale' in p:
                model_type += ' glint_func'
                glint = (self.datasets[i].f_theta, self.datasets[i].f_glint)
            else:
                glint = (None, None)
            glints.append(glint)
            m = _make_model(model_type, lc, *glint)
            models.append(m)
            modpar = m.make_params(verbose=False, **vals)
            if ttv: 
                modpar['T_0'].init_value = modpar['T_0'].value
            modpars.append(modpar)

            if ttv:
                t = f'ttv_{i+1}'
                params.add(t, 0)
                params[t].user_data = ufloat(0,ttv_prior)
                vn.append(t)
                vv.append(0)
                vs.append(30)
                priors[t] = params[t].user_data
                
            if edv:
                t = f'L_{i+1}'
                params.add(t, 0)
                params[t].user_data = ufloat(vals['L'], edv_prior)
                vn.append(t)
                vv.append(0)
                vs.append(1e-6)
                priors[t] = params[t].user_data
                
            for d in dfdp:
                if d in p and p[d].vary:
                    pj = f'{d}_{i+1}'
                    params.add(pj, p[d].value, min=p[d].min, max=p[d].max)
                    if pj in priors:
                        params[pj].user_data = priors[pj]
                    vn.append(f'{d}_{i+1}')
                    if d == 'glint_scale' or d == 'c':
                        vv.append(1)
                    else:
                        vv.append(0)
                    vs.append(1e-6)

            # Set up GP for each dataset
            kernel = JitterTerm(noisemodel['log_sigma_w'].value)
            if 'log_Q' in params:
                kernel += SHOTerm(
                        log_S0=params['log_S0'].value,
                        log_Q=params['log_Q'].value,
                        log_omega0=params['log_omega0'].value)
            if unroll:
                # log_sigma_j is the width of the Gaussian priors on the 
                # coefficients for the linear roll-noise model
                log_sigma_j = np.log(np.nanstd(lc['flux']))
                sinphi = np.sin(np.radians(lc['roll_angle']))
                # Best-fit spin period for this dataset
                s = SineModel.fit(sinphi, P=99/1440, x0=0, x=lc['time'])
                Omega= 2*np.pi/s.params['P']
                omegas.append(Omega)
                for j in range(1,nroll+1 if unroll else 0): 
                    kernel += CosineTerm(log_omega_j=np.log(j*Omega),
                        log_sigma_j=log_sigma_j)
            self.omegas = omegas
            gp = GP(kernel)
            gp.compute(lc['time'],yerr=lc['flux_err'])
            gps.append(gp)

        return params,gps,lcs,models,modpars,glints,omegas,vn,vv,vs,priors

#--------------------------------------------------------------------------

    def __make_result__(self, vn, pos, vv, params, fits, priors):
        # lmfit MinimizerResult object summary of results for printing and
        # plotting. Data/objects required to re-run the analysis, e.g., gps,
        # go directly into self.
        result = MinimizerResult()
        result.status = 0
        result.var_names = vn
        result.covar = np.cov(self.flatchain.T)
        result.init_vals = vv
        result.init_values = params.valuesdict()
        result.acceptance_fraction = self.sampler.acceptance_fraction.mean()
        steps, nwalkers, ndim = self.sampler.get_chain().shape
        result.nfev = int((nwalkers*steps/result.acceptance_fraction))
        result.nwalkers = nwalkers
        result.nvarys = ndim
        result.ndata = sum([len(d.lc['time']) for d in self.datasets])
        result.nfree = result.ndata - ndim
        result.method = 'emcee'
        result.errorbars = True
        result.bestfit = fits
        result.chisqr = -2*np.max(self.sampler.get_log_prob())
        result.redchi = result.chisqr/result.nfree
        result.aic = 2*result.nfree + result.chisqr
        result.bic = result.nfree*np.log(result.ndata) + result.chisqr
        result.residual = [(d.lc['flux']-f) for d,f in zip(self.datasets,fits)]
        result.rms = [r.std() for r in result.residual]
        result.npriors = len(priors)
        result.priors = priors
        
        quantiles = np.percentile(self.flatchain, [15.87, 50, 84.13], axis=0)
        parbest = params.copy()
        corrcoefs = np.corrcoef(self.flatchain.T)
        for i, n in enumerate(vn):
            std_l, median, std_u = quantiles[:, i]
            params[n].value = median
            params[n].stderr = 0.5 * (std_u - std_l)
            parbest[n].value = pos[i]
            parbest[n].stderr = 0.5 * (std_u - std_l)
            if n in self.noisemodel:
                self.noisemodel[n].value = median
                self.noisemodel[n].stderr = 0.5 * (std_u - std_l)
            correl = {}
            for j, n2 in enumerate(vn):
                if i != j:
                    correl[n2] = corrcoefs[i, j]
            params[n].correl = correl
            parbest[n].correl = correl
        result.params  = params
        result.parbest = parbest
        return result

#--------------------------------------------------------------------------

    def fit_transit(self, 
            steps=128, nwalkers=64, burn=256, 
            T_0=None, P=None, D=None, W=None, b=None, f_c=None, f_s=None,
            h_1=None, h_2=None, ttv=False, ttv_prior=3600, extra_priors=None, 
            log_sigma_w=None, log_omega0=None, log_S0=None, log_Q=None,
            unroll=True, nroll=3,
            init_scale=1e-2, progress=True):
        """
        Use emcee to fit the transits in the current datasets 

        If T_0 and P are both fixed parameters then ttv=True can be used to
        include the free parameters ttv_i, the offset in seconds from the
        predicted time of mid-transit for each dataset i = 1, ..., N. The
        prior on the values of ttv_i is a Gaussian with a width ttv_prior in
        seconds.

        """
        # Dict of initial parameter values for creation of models
        # Calculation of mean T_0 needs P and W so T_0 is not first in the list
        vals = {'D':D, 'W':W, 'b':b, 'P':P, 'T_0':T_0, 'f_c':f_c, 'f_s':f_s,
              'h_1':h_1, 'h_2':h_2}
        priors = {} if extra_priors is None else extra_priors
        pmin = {'P':0, 'D':0, 'W':0, 'b':0, 'f_c':-1, 'f_s':-1,
                'h_1':0, 'h_2':0}
        pmax = {'D':0.1, 'W':0.1, 'b':1, 'f_c':1, 'f_s':1,
                'h_1':1, 'h_2':1}
        step = {'D':1e-4, 'W':1e-4, 'b':1e-2, 'P':1e-6, 'T_0':1e-4,
                'f_c':1e-4, 'f_s':1e-3, 'h_1':1e-3, 'h_2':1e-2}

        vn,vv,vs,params = self.__make_params__(vals, priors, pmin, pmax, step,
                extra_priors)

        if ttv and (params['T_0'].vary or params['P'].vary):
            raise ValueError('TTV not allowed if P or T_0 are variables')

        _ = self.__make_noisemodel__(log_sigma_w, log_S0, log_omega0,
                log_Q, params, priors, vn, vv, vs, unroll)
        noisemodel, params, priors, vn, vv, vs = _

        _ = self.__make_modpars__('_transit_func', vals, params, noisemodel,
                vn, vv, vs, priors, ttv, ttv_prior, unroll, nroll)
        params,gps,lcs,models,modpars,glints,omegas,vn,vv,vs,priors  = _

        # Setup sampler
        vv = np.array(vv)
        vs = np.array(vs)
        pos = []
        n_varys = len(vv)
        return_fit = False
        args = (gps, lcs, models, modpars, noisemodel, priors, vn, return_fit)
        for i in range(nwalkers):
            lnlike_i = -np.inf
            while lnlike_i == -np.inf:
                pos_i = vv + vs*np.random.randn(n_varys)*init_scale
                lnlike_i = _log_posterior(pos_i, *args)
            pos.append(pos_i)

        sampler = EnsembleSampler(nwalkers, n_varys, _log_posterior,
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
        state = sampler.run_mcmc(pos, steps, 
            skip_initial_state_check=True, progress=progress)

        flatchain = sampler.get_chain(flat=True)
        pos = flatchain[np.argmax(sampler.get_log_prob()),:]

        return_fit = True
        args = (gps, lcs, models, modpars, noisemodel, priors, vn, return_fit)
        fits = _log_posterior(pos, *args)


        self.gps = gps
        self.noisemodel = noisemodel
        self.models = models
        self.modpars = modpars
        self.sampler = sampler
        self.__fittype__ = 'transit'
        self.flatchain = flatchain
        self.result = self.__make_result__(vn, pos, vv, params, fits, priors)

        return self.result

#--------------------------------------------------------------------------

    def fit_eclipse(self, 
            steps=128, nwalkers=64, burn=256, 
            T_0=None, P=None, D=None, W=None, b=None, f_c=None, f_s=None,
            L=None, a_c=0, edv=False, edv_prior=1e-3, extra_priors=None, 
            log_sigma_w=None, log_omega0=None, log_S0=None, log_Q=None,
            unroll=True, nroll=3,
            init_scale=1e-2, progress=True):
        """
        Use emcee to fit the eclipses in the current datasets 

        Eclipse depths variations can be included in the fit using the keyword
        edv=True. In this case L must be a fixed parameter and the eclipse
        depth for dataset i is L_i, i=1, ..., N. The prior on the values of
        L_i is a Gaussian with mean value L and width edv_prior.

        """

        # Dict of initial parameter values for creation of models
        # Calculation of mean T_0 needs P and W so T_0 is not first in the list
        vals = {'D':D, 'W':W, 'b':b, 'P':P, 'T_0':T_0, 'f_c':f_c, 'f_s':f_s,
                'L':L, 'a_c':a_c}
        priors = {} if extra_priors is None else extra_priors
        pmin = {'P':0, 'D':0, 'W':0, 'b':0, 'f_c':-1, 'f_s':-1,
                'L':0}
        pmax = {'D':0.1, 'W':0.1, 'b':1, 'f_c':1, 'f_s':1,
                'L':0.1}
        step = {'D':1e-4, 'W':1e-4, 'b':1e-2, 'P':1e-6, 'T_0':1e-4,
                'L':1e-5}

        vn,vv,vs,params = self.__make_params__(vals, priors, pmin, pmax, step,
                extra_priors)

        if edv and params['L'].vary:
            raise ValueError('L must be a fixed parameter of edv=True.')

        _ = self.__make_noisemodel__(log_sigma_w, log_S0, log_omega0,
                log_Q, params, priors, vn, vv, vs, unroll)
        noisemodel, params, priors, vn, vv, vs = _

        _ = self.__make_modpars__('_eclipse_func', vals, params, noisemodel,
                vn, vv, vs, priors, edv, edv_prior, unroll, nroll)
        params,gps,lcs,models,modpars,glints,omegas,vn,vv,vs,priors  = _

        # Setup sampler
        vv = np.array(vv)
        vs = np.array(vs)
        pos = []
        n_varys = len(vv)
        return_fit = False
        args = (gps, lcs, models, modpars, noisemodel, priors, vn, return_fit)
        for i in range(nwalkers):
            lnlike_i = -np.inf
            while lnlike_i == -np.inf:
                pos_i = vv + vs*np.random.randn(n_varys)*init_scale
                lnlike_i = _log_posterior(pos_i, *args)
            pos.append(pos_i)

        sampler = EnsembleSampler(nwalkers, n_varys, _log_posterior,
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
        state = sampler.run_mcmc(pos, steps, 
            skip_initial_state_check=True, progress=progress)

        flatchain = sampler.get_chain(flat=True)
        pos = flatchain[np.argmax(sampler.get_log_prob()),:]

        return_fit = True
        args = (gps, lcs, models, modpars, noisemodel, priors, vn, return_fit)
        fits = _log_posterior(pos, *args)

        self.gps = gps
        self.noisemodel = noisemodel
        self.models = models
        self.modpars = modpars
        self.sampler = sampler
        self.__fittype__ = 'eclipse'
        self.flatchain = flatchain
        self.result = self.__make_result__(vn, pos, vv, params, fits, priors)

        return self.result

    # ----------------------------------------------------------------

    def fit_report(self, **kwargs):
        """
        Return a string summarizing the results of the last emcee fit
        """
        report = lmfit_report(self.result, **kwargs)
        rms = np.array(self.result.rms).mean()*1e6
        s = "    RMS residual       = {:0.1f} ppm\n".format(rms)
        j = report.index('[[Variables]]')
        report = report[:j] + s + report[j:]
        noPriors = True
        params = self.result.params
        parnames = list(params.keys())
        namelen = max([len(n) for n in parnames])
        if self.result.npriors > 0: report+="\n[[Priors]]"
        for p in self.result.priors:
            u = self.result.priors[p]
            report += "\n    %s:%s" % (p, ' '*(namelen-len(p)))
            report += '%s +/-%s' % (gformat(u.n), gformat(u.s))
        report += '\n[[Software versions]]'
        pipe_vers = ""
        for s in set([d.pipe_ver for d in self.datasets]):
            pipe_vers += f"{s}, "
        report += '\n    CHEOPS DRP : %s' % pipe_vers[:-2]
        report += '\n    pycheops   : %s' % __version__
        report += '\n    lmfit      : %s' % _lmfit_version_
        return(report)

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

        params = self.result.params
        samples = self.sampler.get_chain()
        var_names = self.result.var_names
        n = len(self.datasets)

        if plotkeys == 'all':
            plotkeys = var_names
        elif plotkeys is None:
            if self.__fittype__ == 'transit':
                l = ['D', 'W', 'b', 'T_0', 'P', 'h_1', 'h_2']
            elif 'L_1' in var_names:
                l = ['D','W','b']+[f'L_{j+1}' for j in range(n)]
            else:
                l = ['L']+[f'c_{j+1}' for j in range(n)]
            plotkeys = list(set(var_names).intersection(l))
            plotkeys.sort()


        n = len(plotkeys)
        fig,ax = plt.subplots(nrows=n, figsize=(width,n*height), sharex=True)
        d0 = 0 
        if 'T_0' in plotkeys:
            d0 = np.floor(np.nanmedian(samples[:,:,var_names.index('T_0')]))
        labels = _make_labels(plotkeys, d0)
        for i,key in enumerate(plotkeys):
            if key == 'T_0':
                ax[i].plot(samples[:,:,var_names.index(key)]-d0, **plot_kws)
            else:
                ax[i].plot(samples[:,:,var_names.index(key)], **plot_kws)
            ax[i].set_ylabel(labels[i])
            ax[i].yaxis.set_label_coords(-0.1, 0.5)
        ax[-1].set_xlim(0, len(samples))
        ax[-1].set_xlabel("step number");

        fig.tight_layout()
        return fig

    # ----------------------------------------------------------------

    def corner_plot(self, plotkeys=None, 
            show_priors=True, show_ticklabels=False,  kwargs=None):

        params = self.result.params
        var_names = self.result.var_names
        n = len(self.datasets)

        if plotkeys == 'all':
            plotkeys = var_names
        if plotkeys == None:
            if self.__fittype__ == 'transit':
                l = ['D', 'W', 'b', 'T_0', 'P', 'h_1', 'h_2']
            elif 'L_1' in var_names:
                l = ['D','W','b']+[f'L_{j+1}' for j in range(n)]
            else:
                l = ['L']+[f'c_{j+1}' for j in range(n)]
            plotkeys = list(set(var_names).intersection(l))
            plotkeys.sort()

        chain = self.sampler.get_chain(flat=True)
        xs = []
        if 'T_0' in plotkeys:
            d0 = np.floor(np.nanmedian(chain[:,var_names.index('T_0')]))
        else:
            d0 = 0 
        for key in plotkeys:
            if key in var_names:
                if key == 'T_0':
                    xs.append(chain[:,var_names.index(key)]-d0)
                else:
                    xs.append(chain[:,var_names.index(key)])

            if key == 'sigma_w' and params['log_sigma_w'].vary:
                xs.append(np.exp(self.emcee.chain[:,-1])*1e6)

            if 'D' in var_names:
                k = np.sqrt(chain[:,var_names.index('D')])
            else:
                k = np.sqrt(params['D'].value) # Needed for later calculations

            if key == 'k' and 'D' in var_names:
                xs.append(k)

            if 'b' in var_names:
                b = chain[:,var_names.index('b')]
            else:
                b = params['b'].value  # Needed for later calculations

            if 'W' in var_names:
                W = chain[:,var_names.index('W')]
            else:
                W = params['W'].value

            aR = np.sqrt((1+k)**2-b**2)/W/np.pi
            if key == 'aR':
                xs.append(aR)

            sini = np.sqrt(1 - (b/aR)**2)
            if key == 'sini':
                xs.append(sini)

            if 'P' in var_names:
                P = chain[:,var_names.index('P')]
            else:
                P = params['P'].value   # Needed for later calculations

            if key == 'logrho':
                logrho = np.log10(4.3275e-4*((1+k)**2-b**2)**1.5/W**3/P**2)
                xs.append(logrho)

        kws = {} if kwargs is None else kwargs

        xs = np.array(xs).T
        labels = _make_labels(plotkeys, d0)
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
    
    def plot_fit(self, title=None, detrend=False, 
            binwidth=0.001, 
            add_gaps=True, gap_tol=0.005, renorm=True,
            data_offset=None, res_offset=None, phase0=None,
            xlim=None, ylim=None, figsize=None, fontsize=12):
        """

        If there are gaps in the data longer than gap_tol phase units and
        add_gaps is True then put a gap in the lines used to plot the fit. The
        transit model is plotted using a thin line in these gaps.

        Binned data are plotted in phase bins of width binwidth. Set
        binwidth=False to disable this feature.

        The data are plotted in the range phase0 to 1+phase0.

        The offsets between the light curves from different datasets can be
        set using the data_offset keyword. The offset between the residuals
        from different datasets can be  set using the res_offset keyword..

        Set renorm=False to prevent automatic re-scaling of fluxes.

        """
        n = len(self.datasets)
        result = self.result
        par = result.parbest
        P = par['P'].value
        T_0 = par['T_0'].value
        times = []
        phases = []
        phfits = []  # For plotting fits with lines - may contain np.nan
        fluxes = []
        fits = []   
        lcmods = []  
        trends = []  
        ph_plots = []
        lc_plots = []
        iqrmax = 0  
        phmin = np.inf
        phmax = -np.inf
        if phase0 is None: phase0 = -0.2
        for j,d in enumerate(self.datasets):
            t = d.lc['time']
            ph = phaser(t,P,T_0,phase0)
            phmax = max(ph)
            phmin = min(ph)
            phases.append(ph)
            flux = d.lc['flux']
            c = np.percentile(flux, 67) if renorm else 1
            fluxes.append(flux/c)
            iqrmax = np.max([iqrmax, iqr(flux)])
            fit = copy(result.bestfit[j])
            modpar = copy(self.modpars[j])
            for d in ('c', 'dfdbg', 'dfdcontam', 'glint_scale',
                    'dfdx', 'd2fdx2', 'dfdy', 'd2fdy2', 'dfdt', 'd2fdt2'):
                p = f'{d}_{j+1}'
                if p in result.var_names:
                     modpar[d].value = 1 if d == 'c' else 0
            model = self.models[j]
            lcmod = model.eval(modpar, t=t)
            trend = fit - lcmod
            trend -= np.nanmedian(trend)
            # Insert np.nan where there are gaps in phase so that the plotted
            # lines have a break
            g = np.where((ph[1:]-ph[:-1]) > gap_tol)[0]
            if add_gaps and len(g) > 0:
                ph = np.insert(ph, g+1, np.nan)
                fit = np.insert(fit, g+1, np.nan)
                lcmod = np.insert(lcmod, g+1, np.nan)
                trend = np.insert(trend, g+1, np.nan)
            phfits.append(ph)
            lcmods.append(lcmod)
            fits.append(fit/c)
            trends.append(trend)
            tp = np.linspace(T_0,T_0+P,65536,endpoint=False)
            ph = phaser(tp,P,T_0,phase0)
            lc = model.eval(modpar, t=tp)
            k = np.argsort(ph)
            ph_plots.append(ph[k])
            lc_plots.append(lc[k])

        if detrend:
            for j, (trend, flx, fit) in enumerate(zip(trends, fluxes, fits)):
                flx -= trend[np.isfinite(trend)]
                fit -= trend
                c = np.nanmax(fit)
                fluxes[j] = flx/c
                fits[j] = fit/c 

        plt.rc('font', size=fontsize)    
        if figsize is None:
            figsize = (8, 2+1.5*n)
        fig,ax=plt.subplots(nrows=2,sharex=True, figsize=figsize,
                gridspec_kw={'height_ratios':[2,1]})

        doff = 2.5*iqrmax if data_offset is None else data_offset
        for j, (ph, flx) in enumerate(zip(phases, fluxes)):
            off = j*doff
            ax[0].plot(ph, flx+off,'o',c='skyblue',ms=2, zorder=1)
            if binwidth:
                r_, f_, e_, n_ = lcbin(ph, flx, binwidth=binwidth)
                ax[0].errorbar(r_, f_+off, yerr=e_, fmt='o',
                        c='midnightblue', ms=5, capsize=2, zorder=3)

        for j, (ph,fit,lcmod) in enumerate(zip(phfits,fits,lcmods)):
            off = j*doff
            ax[0].plot(ph,fit+off,c='saddlebrown', lw=2, zorder=4)
            if not detrend:
                ax[0].plot(ph,lcmod+off,c='forestgreen',zorder=2, lw=2)

        for j, (ph, fp) in enumerate(zip(ph_plots, lc_plots)):
            off = j*doff
            ax[0].plot(ph,fp+off,c='forestgreen', lw=1, zorder=2)


        roff = 10*np.max(result.rms) if res_offset is None else res_offset
        for j, (ph,res) in enumerate(zip(phases, result.residual)):
            off=j*roff
            ax[1].plot(ph, res+off,'o',c='skyblue',ms=2)
            ax[1].axhline(off, color='saddlebrown',ls=':')
            if binwidth:
                r_, f_, e_, n_ = lcbin(ph, res, binwidth=binwidth)
                ax[1].errorbar(r_, f_+off, yerr=e_,
                        fmt='o', c='midnightblue', ms=5, capsize=2)

        if xlim is None:
            pad = (phmax-phmin)/10
            if self.__fittype__ == "transit":
                pht = max([abs(phmin), abs(phmax)])
                ax[1].set_xlim(-pht-pad,pht+pad)
            else:
                ax[1].set_xlim(phmin-pad,phmax+pad)
        else:
            ax[1].set_xlim(*xlim)

        if ylim is not None: ax[0].set_ylim(*ylim)
        ax[0].set_ylabel('Flux')
        if roff > 0:
            ax[1].set_ylim(-0.75*roff, roff*(n-0.25))
        else:
            rms = 10*np.max(result.rms)
            ax[1].set_ylim(-5*rms, 5*rms)

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
            jovian=True, plot_kws=None, verbose=True):
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

        # Generate value(s) from previous emcee sampler run
        def _v(p):
            vn = self.result.var_names
            chain = self.flatchain
            pars = self.result.params
            if (p in vn):
                v = chain[:,vn.index(p)]
            elif p in pars.valuesdict().keys():
                v = pars[p].value
            else:
                raise AttributeError(
                        'Parameter {} missing from dataset'.format(p))
            return v
    
    
        # Generate a sample of values for a parameter
        def _s(x, nm=100_000):
            if isinstance(x,float) or isinstance(x,int):
                return np.full(nm, x, dtype=np.float)
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
        _q = _s(q, len(self.flatchain))
        rho_star = rhostar(1/aR,P,_q)
        # N.B. use of np.abs to cope with values with large errors
        if r_star is None and m_star is not None:
            _m = np.abs(_s(m_star, len(self.flatchain)))
            r_star = (_m/rho_star)**(1/3)
        if m_star is None and r_star is not None:
            _r = np.abs(_s(r_star, len(self.flatchain)))
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
                jovian=jovian, verbose=verbose, **plot_kws)
    
        
    # ------------------------------------------------------------

