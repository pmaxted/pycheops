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
instrument
==========
 Constants, functions and data related to the CHEOPS instrument.

Functions 
---------

"""

from __future__ import (absolute_import, division, print_function,
                                unicode_literals)
import numpy as np
from .models import TransitModel, scaled_transit_fit, minerr_transit_fit
import warnings 


__all__ = [ 'response', 'visibility', 'exposure_time', 'transit_noise']

from astropy.table import Table
from os.path import join,abspath,dirname
from pickle import load
from numpy import int as np_int 
from numpy import round

_data_path = join(dirname(abspath(__file__)),'data')

with open(join(_data_path,'instrument','exposure_time.p'),'rb') as fp:
    _exposure_time_interpolator = load(fp)

with open(join(_data_path,'instrument','visibility_interpolator.p'),'rb') as fp:
    _visibility_interpolator = load(fp)

def visibility(ra, dec):
    """
    Estimate of target visibility 

    The target visibility estimated with this function is approximate. A more
    reliable estimate of the observing efficiency can be made with the 
    Feasibility Checker tool.

    :param ra: right ascension in degrees (scalar or array)

    :param dec: declination in degrees (scalar or array)

    :returns: target visibility (%)

    """
    return (_visibility_interpolator(ra, dec)*100).astype(np_int)

def response(passband='CHEOPS'):
    """
     Instrument response functions.

     The response functions have been digitized from Fig. 2 of
     https://www.cosmos.esa.int/web/cheops/cheops-performances

     The available passband names are 'CHEOPS', 'MOST', 
     'Kepler', 'CoRoT', 'Gaia', 'B', 'V', 'R', 'I',
     u_','g_','r_','i_','z_', and 'NGTS'

     :param passband: instrument/passband names (case sensitive).

     :returns: Instrument response function as an astropy Table object.

    """
    T = Table.read(join(_data_path,'response_functions',
        'response_functions.fits'))
    T.rename_column(passband,'Response')
    return T['Wavelength','Response']

def exposure_time(G):
    """
    Recommended minimum/maximum exposure times

    The function returns the exposure times that are estimated to provide 
    10% and 90% of the detector full well capacity in the brightest image
    pixel of the target. 

     :param G: Gaia G-band magnitude

     :returns: min,max recommended exposure time

    """

    r =  _exposure_time_interpolator(G)
    return round(r[0],2),round(r[1],2)


def transit_noise(time, flux, flux_err, T_0=None, width=3,
                  h_1=0.7224, h_2=0.6713, tol=0.1, 
                  method='scaled'):
    """
    Transit noise estimate

    The noise is calculated in a window of duration 'width' in hours centered 
    at time T_0 by finding the depth of a transit that gives S/N = 1.
    
    Two methods are available to estimate the transit depth and its standard
    error - 'scaled' or 'minerr'.

    If method='scaled', the transit depth and its standard error are
    calculated assuming that the true standard errors on the flux measurements
    are a factor f times the nominal standard error(s) provided in flux_err.

    If method='minerr', the transit depth and its standard error are
    calculated assuming that standard error(s) provided in flux_err are a
    lower bound to the true standard errors. This tends to be more
    conservative than using method='scaled'.

    The transit is calculated from an impact parameter b=0 using power-2 limb
    darkening parameters h_1 and h_2. Default values for h_1 and h_2 are solar
    values.

    If T_0 is not specifed that the median value of time is used.

    If there are insufficient data for the calculation the function returns
    values returned are np.nan, np.nan

     :param time: Array of observed times (days)

     :param flux: Array of normalised flux measurements

     :param flux_err: Standard error estimate(s) for flux - array of scalar

     :param T_0: Centre of time window for noise estimate

     :param width: Width of time window for noise estimate in hours

     :param h_1: Limb darkening parameter

     :param h_2: Limb darkening parameter

     :param tol: Tolerance criterion for convergence (ppm)

     :param method: 'scaled' or 'minerr'

     :returns: noise in ppm and, if method is 'scaled', noise scaling factor, f

    """

    assert (method in ('scaled', 'minerr')), "Invalid method value"

    if T_0 is None:
        T_0 = np.median(time)

    # Use orbital period = 10* data duration so there is certainly 1 transit
    P = 10*(max(time)-min(time))
    j = (np.abs(time-T_0) < (width/48)).nonzero()[0]
    if len(j) < 3:
        if method == 'scaled':
            return np.nan, np.nan
        else:
            return np.nan

    e_depth = np.median(flux_err[j])/np.sqrt(len(j))
    W = width/24/P   # Transit Width in phase units
    tm = TransitModel()
    depth_tol = tol*1e-6
    depth_in = 0

    mad =  np.median(np.abs(flux-np.median(flux)))
    if np.abs(np.median(flux)-1) > mad:
        warnings.warn ("Input flux values are not normalised")

    ITMAX = 100
    i = 1
    while abs(e_depth-depth_in) > depth_tol:
        depth_in = e_depth
        k = np.clip(np.sqrt(depth_in),1e-6,0.2)
        S = ((1-k)**2)/((1+k)**2)
        pars = tm.make_params(T_0=T_0, P=P, D=depth_in, W=W, S=S,
                h_1=h_1, h_2=h_2)
        model = tm.eval(params=pars, t=time)
        if method == 'scaled':
            s, f, sigma_s, sigma_f = scaled_transit_fit(flux,flux_err,model)
        else:
            s, sigma_s = minerr_transit_fit(flux,flux_err,model)

        if sigma_s is np.nan:
            if method == 'scaled':
                return np.nan, np.nan
            else:
                return np.nan

        e_depth = sigma_s*depth_in
        i = i + 1
        if i > ITMAX:
            warnings.warn ('Algorithm failed to converge.')
            break

    if method == 'scaled':
        return 1e6*depth_in, f
    else:
        return 1e6*depth_in

