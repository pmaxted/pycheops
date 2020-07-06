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

from os.path import join,abspath,dirname,isfile
import pickle 
from numpy import int as np_int 
from astropy.table import Table
from .core import load_config
from .models import TransitModel, scaled_transit_fit, minerr_transit_fit
import warnings 

__all__ = [ 'response', 'visibility', 'exposure_time', 'transit_noise',
        'count_rate', 'cadence', 'CHEOPS_ORBIT_MINUTES']

_data_path = join(dirname(abspath(__file__)),'data')
config = load_config()
_cache_path = config['DEFAULT']['data_cache_path']

# Parameters from spreadsheet ImageETCv1.4, 2020-04-01
FLUX_0 = 1851840480 
PSF_R90 = 16.2
PSF_HP = 0.0046
FWC = 114000

# From Benz et al., 2020
CHEOPS_ORBIT_MINUTES = 98.725

with open(join(_cache_path,'C_G_Teff_interpolator.p'),'rb') as fp:
    _C_G_Teff_interpolator = pickle.load(fp)

with open(join(_cache_path,'visibility_interpolator.p'),'rb') as fp:
    _visibility_interpolator = pickle.load(fp)

_cadence_Table = Table.read(join(_data_path,'instrument','cadence.csv'),
        format='ascii.csv', header_start=1)

#-----------------------------

def count_rate(G, Teff=6000):
    """
    Predicted count rates, c_tot, c_av, c_max 

    The count rates in e-/s based on the star's Gaia G magnitude and effective
    temperature, Teff. 

    * c_tot = total count rate
    * c_av  = average count rate
    * c_max = count rate in the brightest pixel

    :param G: Gaia G-band magnitude

    :param Teff: target effective temperature in K

    :returns: c_tot, c_av, c_max 

    """

    c_tot = round(FLUX_0*10**(-0.4*(G+_C_G_Teff_interpolator(Teff))))
    c_av  = round(0.90*c_tot/(np.pi*PSF_R90**2))
    c_max = round(PSF_HP*c_tot)
    return c_tot, c_av, c_max


#-----------------------------

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


#-----------------------------

def response(passband='CHEOPS'):
    """
     Instrument response functions.

     The response functions have been digitized from Fig. 2 of
     https://www.cosmos.esa.int/web/cheops/cheops-performances

     The available passband names are 'CHEOPS', 'MOST', 
     'Kepler', 'CoRoT', 'Gaia', 'B', 'V', 'R', 'I',
     'u\_','g\_','r\_','i\_','z\_', and 'NGTS'

     :param passband: instrument/passband names (case sensitive).

     :returns: Instrument response function as an astropy Table object.

    """
    T = Table.read(join(_data_path,'response_functions',
        'response_functions.fits'))
    T.rename_column(passband,'Response')
    return T['Wavelength','Response']

#------------------

def exposure_time(G, Teff=6000, frac=0.85):
    """
    Recommended exposure time.

    By default, calculates the exposure time required to obtain 85% of the
    full-well capacity in the brightest pixel for a star of a given Gaia
    G-band magnitude, G, and effective temperature, Teff in Kelvin.

    The value returned is restricted to the range  0.1 s < t_exp < 60 s.

    The exposure time can be adjusted by selecting a different value of frac,
    the fraction of the full-well capacity (FWC) in the brightest pixel. It is
    strongly recommended not to exceed frac=0.95 for CHEOPS observations.

    :param G: Gaia G-band magnitude

    :frac: target fraction of the FWC in the brightest pixel.

    :returns: t_exp

    """

    c_tot, c_av, c_max = count_rate(G, Teff)

    t_exp = round(np.clip(frac*FWC/c_max,0.1,60),2)

    return t_exp

#------------------


def cadence(exptime, G, Teff=6000):
    """
    Cadence and other observing informtion for a given exposure time.

    For a star of the specified Gaia G-band magnitude and effective
    temperature, return the following parameters for an exposure time of the
    specified length.

    * img  = image stacking order
    * igt  = imagette stacking order
    * cad  = stacked image cadence (in seconds)
    * duty = duty cycle (%) 
    * frac = maximim counts as a fraction of the full-well capacity 

    :param exptime: exposure time in seconds (0.1 .. 60)

    :param G: Gaia G-band magnitude

    :Teff: target effective temperature in Kelvin

    :returns: img, igt, cad, duty, frac

    """

    if exptime < 0.1 or exptime > 60:
        return int(np.nan), int(np.nan), np.nan, np.nan, np.nan

    R = _cadence_Table[np.searchsorted(_cadence_Table['t_hi'],exptime)]
    img = R['img']
    igt = R['igt']
    w = (R['t_hi']-exptime)/(R['t_hi']-R['t_lo'])  # interpolating weight
    duty = round(w*R['duty_lo'] + (1-w)*R['duty_hi'],2)
    cad = round(w*R['cad_lo'] + (1-w)*R['cad_hi'],2)
    c_tot, c_av, c_max = count_rate(G, Teff)
    frac = round(exptime*c_max/FWC,2)

    return img, igt, cad, duty, frac

#------------------

def transit_noise(time, flux, flux_err, T_0=None, width=3,
                  h_1=0.7224, h_2=0.6713, tol=0.1, 
                  method='scaled'):
    """
    Transit noise estimate

    The noise is calculated in a window of duration 'width' in hours centered 
    at time T_0 by first dividing out the best-fitting transit (even if this
    has a negative depth), and then finding the depth of an injected transit
    that gives S/N = 1.
    
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

    mad =  np.median(np.abs(flux-np.median(flux)))
    if np.abs(np.median(flux)-1) > mad:
        warnings.warn ("Input flux values are not normalised")

    if T_0 is None:
        T_0 = np.median(time)

    # Use orbital period = 10* data duration so there is certainly 1 transit
    P = 10*(max(time)-min(time))

    j = (np.abs(time-T_0) < (width/48)).nonzero()[0]
    if len(j) < 4:
        if method == 'scaled':
            return np.nan, np.nan
        else:
            return np.nan

    ITMAX = 10
    it = 1
    e_depth = np.median(flux_err[j])/np.sqrt(len(j))
    depth_in = 0
    W = width/24/P   # Transit Width in phase units
    tm = TransitModel()
    depth_tol = tol*1e-6
    while abs(e_depth-depth_in) > depth_tol:
        depth_in = e_depth
        k = np.clip(np.sqrt(depth_in),1e-6,0.2)
        pars = tm.make_params(T_0=T_0, P=P, D=depth_in, W=W, b=0,
                h_1=h_1, h_2=h_2)
        model = tm.eval(params=pars, t=time)

        # Calculate best-fit transit depth
        if method == 'scaled':
            s0, _, _, _ = scaled_transit_fit(flux,flux_err,model)
            if s0 == 0:
                s0, _, _, _ = scaled_transit_fit(2-flux,flux_err,model)
                s0 = -s0
        else:
            s0, _ = minerr_transit_fit(flux,flux_err,model)
            if s0 == 0:
                s0, _ = minerr_transit_fit(2-flux,flux_err,model)
                s0 = -s0

        # Subtract off best-fit transit depth and inject model transit
        _f = flux  - (s0-1)*(model-1) 

        if method == 'scaled':
            s, f, sigma_s, sigma_f = scaled_transit_fit(_f,flux_err,model)
        else:
            s, sigma_s = minerr_transit_fit(_f,flux_err,model)

        # If the input depth is too small then error can be 0, so ..
        if sigma_s > 0:
            e_depth = sigma_s*depth_in
        else:
            e_depth = depth_in*2
        #print(it,s0,s, sigma_s, depth_in, e_depth)
        it = it + 1
        if it > ITMAX:
            warnings.warn ('Algorithm failed to converge.')
            break

    if method == 'scaled':
        return 1e6*depth_in, f
    else:
        return 1e6*depth_in

