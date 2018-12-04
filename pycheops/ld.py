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
ld
==
Limb darkening functions 

The available passband names are:

* 'CHEOPS', 'MOST', 'Kepler', 'CoRoT', 'Gaia',

* 'U', 'B', 'V', 'R', 'I' (Bessell/Johnson)

* 'u\_', 'g\_', 'r\_', 'i\_', 'z\_'  (SDSS)

The power-2 limb-darkening law is described in Maxted (2018) [2].

Examples
--------

>>> from pycheops.ld import *
>>> import matplotlib.pyplot as plt
>>> import numpy as np
>>> T_eff = 5560
>>> log_g = 4.3
>>> Fe_H = -0.3
>>> passband = 'Kepler'
>>> cK = stagger_claret_interpolator(passband)
>>> ac = cK(T_eff, log_g, Fe_H)
>>> mK = stagger_mugrid_interpolator(passband)
>>> I_mu = mK(T_eff, log_g, Fe_H)
>>> p2K = stagger_power2_interpolator(passband)
>>> c2,a2,h1,h2 = p2K(T_eff, log_g, Fe_H)
>>> plt.plot(mK.mu, I_mu,'bo')
>>> mu = np.linspace(0,1)
>>> plt.plot(mu, ld_claret(mu,ac),label='Claret')
>>> plt.plot(mu, ld_power2(mu,[c2, a2]),label='power-2')
>>> plt.xlim(0,1)
>>> plt.ylim(0,1)
>>> plt.xlabel('$\mu$')
>>> plt.ylabel('$I_{\lambda}(\mu)$')
>>> plt.legend()
>>> plt.show()


.. rubric:: References
.. [2] Maxted, P.F.L., 2018, A&A, submitted 

"""

from __future__ import (absolute_import, division, print_function,
                                unicode_literals)
import numpy as np
from os.path import join,abspath,dirname
import pickle
from scipy.interpolate import pchip_interpolate
from scipy.optimize import minimize
from ellc import lc
from .funcs import transit_width

__all__ = ['ld_power2', 'ld_claret', 'stagger_power2_interpolator',
        'stagger_mugrid_interpolator', 'stagger_claret_interpolator',
        'ca_to_h1h2', 'h1h2_to_ca' ]

data_path = join(dirname(abspath(__file__)),'data','limbdarkening')

def ld_power2(mu, a):
    """
    Evaluate power-2 limb-darkening law 

    :param mu: cos of angle between surface normal and line of sight
    :param a: array or tuple [c, alpha]

    :returns:  1 - c * (1-mu**alpha)
    
    """

    c, alpha = a
    return  1 - c * (1-mu**alpha)

def ca_to_h1h2(c, alpha):
    """
    Transform for power-2 law coefficients
    h1 = 1 - c*(1-0.5**alpha)
    h2 = c*0.5**alpha

    :param c: power-2 law coefficient, c
    :param alpha: power-2 law exponent, alpha

    returns: h1, h2

    """
    return 1 - c*(1-0.5**alpha), c*0.5**alpha


def h1h2_to_ca(h1, h2):
    """
    Inverse transform for power-2 law coefficients
    c = 1 - h1 + h2
    alpha = log2(c/h2)

    :param h1: 1 - c*(1-0.5**alpha)
    :param h2: c*0.5**alpha

    returns: c, alpha

    """
    return 1 - h1 + h2, np.log2((1 - h1 + h2)/h2)


def ld_claret(mu, a):
    """
    Evaluate Claret 4-parameter limb-darkening law

    :param mu: cos of angle between surface normal and line of sight

    :param a: array or tuple [a_1, a_2, a_3, a_4]

    :returns:  1 - Sum(i=1,4) a_i*(1-mu**(i/2))
    
    """

    return 1-a[0]*(1-mu**0.5)-a[1]*(1-mu)-a[2]*(1-mu**1.5)-a[3]*(1-mu**2)

class stagger_mugrid_interpolator:
    """
    
    Limb-darkening as a function of mu from the Stagger grid.

    Returns NaN if interpolation outside the grid range is attempted

    :ivar mu: grid of mu values for most recent call


    """

    def __init__(self, passband='CHEOPS'):

        self._mu_default = np.array(
                [0,0.01,0.05,0.1,0.2,0.3,0.5,0.7,0.8,0.9,1.0])

        self.mu = self._mu_default

        pfile = join(data_path,passband+'_stagger_mugrid_interpolator.p')
        with open(pfile, 'rb') as fp:
            self._interpolator = pickle.load(fp)

    def __call__(self, T_eff, log_g, Fe_H, n_mu=None):

        """

        :parameter T_eff: effective temperature in Kelvin
        
        :parameter log_g: log of the surface gravity in cgs units
       
        :parameter Fe/H: [Fe/H] in dex
       
        :parameter n_mu: No. of points in the output intensity grid
       
        :returns: I(mu) on a grid of mu values from 0 to 1.
       
        """
        I_mu = self._interpolator(T_eff, log_g, Fe_H) 

        if n_mu is None:
            self.mu = self._mu_default
        else:
            self.mu = np.linspace(0,1,n_mu)
            I_mu = pchip_interpolate(self._mu_default, I_mu, self.mu )

        return I_mu

class stagger_claret_interpolator:
    """
    
    Coefficients of a Claret 4-parameter limb-darkening law interpolated
    from the Stagger grid.
    
    Returns NaN if interpolation outside the grid range is attempted


    """

    def __init__(self,passband='CHEOPS'):
        """

        :param passband: instrument/passband names (case sensitive).

        """
        pfile = join(data_path, passband+'_stagger_claret_interpolator.p')
        with open(pfile, 'rb') as fp:
            self._interpolator = pickle.load(fp)

    def __call__(self, T_eff, log_g, Fe_H):
        """

        :parameter T_eff: effective temperature in Kelvin
        
        :parameter log_g: log of the surface gravity in cgs units
       
        :parameter Fe/H: [Fe/H] in dex
       
        :returns: array of coefficients 
       
        """
        return self._interpolator(T_eff, log_g, Fe_H)


class _coefficient_optimizer:
    """
    
    Optimize coefficients of the limb darkening law specified by fitting a
    transit light curve.

    Available limb-darkening laws are "lin", "quad", "power-2", "exp",
    "sqrt", "log", "sing" and "claret"

    """
   
    def __init__(self, passband='CHEOPS'):
        """

        :param passband: instrument/passband names (case sensitive).

        """

        pfile = join(data_path,passband+'_stagger_mugrid_interpolator.p')
        with open(pfile, 'rb') as fp:
            self._interpolator = pickle.load(fp)


    def __call__(self, T_eff, log_g, Fe_H, k=0.1, b=0.0, 
            law='power-2', precision='low'):
        """

        :parameter T_eff: effective temperature in Kelvin
        
        :parameter log_g: log of the surface gravity in cgs units
       
        :parameter Fe/H: [Fe/H] in dex

        :parameter k: Radius ratio R_pl/R_star 

        :parameter b: Impact parameter (R_star/a)cos(incl) 
       
        :parameter law: Limb darkening law
       
        :param precision: 'low', 'medium' or 'high'
       
        :returns: array of coefficients 
       
        """

        self._mu_default = np.array(
                [0,0.01,0.05,0.1,0.2,0.3,0.5,0.7,0.8,0.9,1.0])
        precision_to_gridsize = {
            "low" :  "very_sparse",
            "medium": "sparse",
            "high": "default" 
            }
        self._gridsize = precision_to_gridsize.get(precision,None)
        if self._gridsize is None:
            raise ValueError("Invalid precision value",precision)

    # Fixed parameters
        n_mu = 51
        n_grid = 32

        mu = np.linspace(0,1,n_mu)
        I_mu =  self._interpolator(T_eff, log_g, Fe_H)
        ldc_1 = pchip_interpolate(self._mu_default, I_mu, mu )
        w = 0.5*transit_width(0.1, k, b)
        # Avoid last contact point - occasional numerical problems
        self._t = np.linspace(0,w,n_grid,endpoint=False)
        incl = 180*np.arccos(0.1*b)/np.pi
        self._lc_mugrid = lc(self._t, radius_1=0.1, radius_2=0.1*k, 
            sbratio=0, incl=incl, ld_1='mugrid', ldc_1 = ldc_1,
            grid_1=self._gridsize, grid_2=self._gridsize)

        if law in ("lin"):
            c = np.full(1, 0.5)
        elif law in ("quad", "log", "sqrt", "exp"):
            c = np.full(2, 0.5)
        elif law in ("power-2"):
            c = np.array([0.7,0.45])  # h1, h2
        elif law in ("sing"):
            c = np.full(3, 0.5)
        elif law in ("claret"):
            c = np.full(4, 0.5)
        else:
            raise Exception("Invalid limb darkening law")

        if law in ("power-2"):
            soln = minimize(self._f, c, args=(k, incl, law),
                    method='L-BFGS-B',bounds=((0.01,0.99),(0.01,0.99)))
            c2,a2 = h1h2_to_ca(soln.x[0],soln.x[1])
            c = np.array([c2, a2])
        else:
            soln = minimize(self._f, c, args=(k, incl, law))
            c = soln.x

        self._rms = soln.fun
        self._lc_fit = lc(self._t, radius_1=0.1, radius_2=0.1*k, 
            sbratio=0, incl=incl, ld_1=law, ldc_1 = c, 
            grid_1=self._gridsize, grid_2=self._gridsize)

        return c

    def _f(self, c, k, incl, law):
        if law in ("power-2"):
           c2,a2 = h1h2_to_ca(c[0],c[1]) 
           ldc_1 = [c2, a2]
        else:
           ldc_1 = c
           
        try:
            lc_fit = lc(self._t, radius_1=0.1, radius_2=0.1*k, 
                sbratio=0, incl=incl, ld_1=law, ldc_1 = ldc_1, 
                grid_1=self._gridsize, grid_2=self._gridsize)
        except:
            lc_fit = zero_like(self._t)
        rms =  np.sqrt(np.mean((lc_fit - self._lc_mugrid)**2))
        return rms

class stagger_power2_interpolator:
    """
    
    Parameters of a power-2 limb-darkening law interpolated
    from the Stagger grid.

    The power-2 limb darkening law is 
      I_X(mu) = 1 - c * (1-mu**alpha)

    It is often better to use the transformed coefficients

    *  h1 = 1 - c*(1-0.5**alpha) 

    and 

    *  h2 = c*0.5**alpha

    as free parameters in a least-squares fit and/or for applying priors.

    Returns NaN if interpolation outside the grid range is attempted


    """

    def __init__(self,passband='CHEOPS'):
        """

        :param passband: instrument/passband names (case sensitive).

        """
        pfile = join(data_path, passband+'_stagger_power2_interpolator.p')
        with open(pfile, 'rb') as fp:
            self._interpolator = pickle.load(fp)

    def __call__(self, T_eff, log_g, Fe_H):
        """

        :parameter T_eff: effective temperature in Kelvin
        
        :parameter log_g: log of the surface gravity in cgs units
       
        :parameter Fe/H: [Fe/H] in dex
       
        :returns: c, alpha, h_1, h_2
       
        """
        return self._interpolator(T_eff, log_g, Fe_H)


