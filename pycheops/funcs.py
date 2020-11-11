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
funcs
=====

 Functions related to observable properties of stars and exoplanets

Parameters
----------
Functions are defined in terms of the following parameters. [1]_

* a          - orbital semi-major axis in solar radii = a_1 + a_2 
* P          - orbital period in mean solar days
* Mass       - total system mass in solar masses, Mass = m_1 + m_2
* ecc        - orbital eccentricity
* omdeg      - longitude of periastron of star's orbit, omega, in _degrees_
* sini       - sine of the orbital inclination 
* K          - 2.pi.a.sini/(P.sqrt(1-e^2)) = K_1 + K_2
* K_1, K_2   - orbital semi-amplitudes in km/s
* q          - mass ratio = m_2/m_1 = K_1/K_2 = a_1/a_2
* f_m        - mass function = m_2^3.sini^3/(m_1+m_2)^2 in solar masses 
                             = K_1^3.P/(2.pi.G).(1-e^2)^(3/2)
* r_1        - radius of star 1 in units of the semi-major axis, r_1 = R_*/a
* r_2        - radius of companion in units of the semi-major axis, r_2 = R_2/a
* rhostar    - mean stellar density = 3.pi/(GP^2(1+q)r_1^3)
* rstar      - host star radius/semi-major axis, rstar = R_*/a
* k          - planet/star radius ratio, k = R_planet/R_star
* tzero      - time of mid-transit (minimum on-sky star-planet separation). 
* b          - impact parameter, b = a.cos(i)/R_star
  
.. rubric References
.. [1] Hilditch, R.W., An Introduction to Close Binary Stars, CUP 2001.

Functions 
---------

"""

from __future__ import (absolute_import, division, print_function,
                                unicode_literals)
from .constants import *
import numpy as np
from scipy.optimize import brent
from numba import vectorize
from uncertainties import ufloat, UFloat
from uncertainties.umath import sqrt as usqrt
import requests
from .utils import mode, parprint, ellpar
from random import sample as random_sample
from .constants import R_SunN, M_SunN, M_JupN, R_JupN, au, M_EarthN, R_EarthN
import matplotlib.pyplot as plt
from os.path import join, dirname, abspath
from astropy.table import Table
from pathlib import Path
from time import localtime, mktime
from os.path import getmtime
from .core import load_config
from matplotlib.patches import Ellipse
from scipy.signal import argrelextrema
import warnings

__all__ = [ 'a_rsun','f_m','m1sin3i','m2sin3i','asini','rhostar','g_2',
        'K_kms','m_comp','transit_width','esolve','t2z',
        'tperi2tzero','tzero2tperi', 'vrad', 'xyz_planet']

_arsun   = (GM_SunN*mean_solar_day**2/(4*np.pi**2))**(1/3.)/R_SunN
_f_m     = mean_solar_day*1e9/(2*np.pi)/GM_SunN
_asini   = mean_solar_day*1e3/2/np.pi/R_SunN
_rhostar = 3*np.pi*V_SunN/(GM_SunN*mean_solar_day**2)
_model_path = join(dirname(abspath(__file__)),'data','models')
_rho_Earth_cgs = M_EarthN/(4/3*np.pi*R_EarthN**3)/1000

config = load_config()
_cache_path = config['DEFAULT']['data_cache_path']
TEPCatPath = Path(_cache_path,'allplanets-csv.csv')

def a_rsun(P, Mass):
    """
    Semi-major axis in solar radii

    :param P: orbital period in mean solar days
    :param Mass: total mass in solar masses, M

    :returns: a = (G.M.P^2/(4.pi^2))^(1/3) in solar radii
    
    """

    return _arsun * P**(2/3.) * Mass**(1/3.)

def f_m(P, K, ecc=0):
    """
    Mass function in solar masses

    :param P: orbital period in mean solar days
    :param K: semi-amplitude of the spectroscopic orbit in km/s
    :param ecc: orbital eccentricity

    :returns: f_m =  m_2^3.sini^3/(m_1+m_2)^2  in solar masses
    """
    return _f_m * K**3 * P * (1 - ecc**2)**1.5

def m1sin3i(P, K_1, K_2, ecc=0):
    """
     Reduced mass of star 1 in solar masses

     :param K_1: semi-amplitude of star 1 in km/s
     :param K_2: semi-amplitude of star 2 in km/s
     :param P: orbital period in mean solar days
     :param ecc:  orbital eccentricity

     :returns: (m_2.sini)^3/(m_1+m_2)^2 in solar masses 
    """
    return _f_m * K_2 * (K_1 + K_2)**2 * P * (1 - ecc**2)**1.5

def m2sin3i(P, K_1, K_2, ecc=0):
    """
     Reduced mass of star 2 in solar masses

     :param K_1:  semi-amplitude of star 1 in km/s
     :param K_2:  semi-amplitude of star 2 in km/s
     :param P:   orbital period in mean solar days
     :param ecc:   orbital eccentricity

     :returns: m_2.sini^3 in solar masses 
    """
    return _f_m * K_1 * (K_1 + K_2)**2 * P * (1 - ecc**2)**1.5

def asini(K, P, ecc=0):
    """
     a.sini in solar radii

     :param K: semi-amplitude of the spectroscopic orbit in km/s
     :param P: orbital period in mean solar days

     :returns: a.sin(i) in solar radii

    """
    return _asini * K * P *np.sqrt(1-ecc**2)

def r_star(rho, P, q=0):
    """ 
    Scaled stellar radius R_*/a from mean stellar density 

    :param rho: Mean stellar density in solar units
    :param P: orbital period in mean solar days
    :param q: mass ratio, m_2/m_1

    :returns: radius of star in units of the semi-major axis, R_*/a

    """
    return (_rhostar/(rho*P**2/(1+q)))**(1/3.)

def rhostar(r_1, P, q=0):
    """ 
    Mean stellar density from scaled stellar radius.

    :param r_1: radius of star in units of the semi-major axis, r_1 = R_*/a
    :param P: orbital period in mean solar days
    :param q: mass ratio, m_2/m_1

    :returns: Mean stellar density in solar units

    """
    return _rhostar/(P**2*(1+q)*r_1**3)

def g_2(r_2, P, K, sini=1, ecc=0):
    """ 
    Companion surface gravity g = G.m_2/R_2**2 from P, K and r_2
    
    Calculated using equation (4) from Southworth et al., MNRAS
    2007MNRAS.379L..11S. The

    :param r_2: companion radius relative to the semi-major axis, r_2 = R_2/a
    :param P: orbital period in mean solar days
    :param K_1: semi-amplitude of star 1's orbit in km/s
    :param sini: sine of the orbital inclination
    :param ecc: orbital eccentrcity

    :returns: companion surface gravity in m.s-2

    """
    return 2*np.pi*np.sqrt(1-ecc**2)*K*1e3/(P*mean_solar_day*r_2**2*sini) 

def K_kms(m_1, m_2, P, sini, ecc):
    """
    Semi-amplitudes of the spectroscopic orbits in km/s
     - K = 2.pi.a.sini/(P.sqrt(1-ecc^2))
     - K_1 = K * m_2/(m_1+m_2)
     - K_2 = K * m_1/(m_1+m_2)

    :param m_1:  mass of star 1 in solar masses
    :param m_2:  mass of star 2 in solar masses
    :param P:  orbital period in mean solar days
    :param sini:  sine of the orbital inclination
    :param ecc:  orbital eccentrcity

    :returns: K_1, K_2 -- semi-amplitudes in km/s
    """
    M = m_1 + m_2
    a = a_rsun(P, M)
    K = 2*np.pi*a*R_SunN*sini/(P*mean_solar_day*np.sqrt(1-ecc**2))/1000
    K_1 = K * m_2/M
    K_2 = K * m_1/M
    return K_1, K_2

#---------------

def m_comp(f_m, m_1, sini):
    """
    Companion mass in solar masses given mass function and stellar mass

    :param f_m: = K_1^3.P/(2.pi.G).(1-ecc^2)^(3/2) in solar masses
    :param m_1: mass of star 1 in solar masses
    :param sini: sine of orbital inclination

    :returns: m_2 = mass of companion to star 1 in solar masses

    """
    DA = -f_m/sini**3
    DB = 2*DA*m_1
    DC = DA*m_1**2
    Q = (DA**2 - 3*DB)/9
    R = (2*DA**3 - 9*DA*DB + 27*DC)/54
    DAA = -np.sign(R)*(np.sqrt(R**2 - Q**3) + np.abs(R))**(1/3)
    DBB = Q/DAA
    return DAA + DBB - DA/3

#---------------

def transit_width(r, k, b, P=1):
    """
    Total transit duration for a circular orbit.

    See equation (3) from Seager and Malen-Ornelas, 2003ApJ...585.1038S.

    :param r: R_star/a
    :param k: R_planet/R_star
    :param b: impact parameter = a.cos(i)/R_star
    :param P: orbital period (optional, default P=1)

    :returns: Total transit duration in the same units as P.

    """

    return P*np.arcsin(r*np.sqrt( ((1+k)**2-b**2) / (1-b**2*r**2) ))/np.pi

#---------------

@vectorize(nopython=True)
def esolve(M, ecc):
    """
    Solve Kepler's equation M = E - ecc.sin(E) 

    :param M: mean anomaly (scalar or array)
    :param ecc: eccentricity (scalar or array)

    :returns: eccentric anomaly, E

    Algorithm is from Markley 1995, CeMDA, 63, 101 via pyAstronomy class
    keplerOrbit.py

    :Example:

    Test precision using random values::
    
     >>> from pycheops.funcs import esolve
     >>> from numpy import pi, sin, abs, max
     >>> from numpy.random import uniform
     >>> ecc = uniform(0,1,1000)
     >>> M = uniform(-2*np.pi,4*np.pi,1000)
     >>> E = esolve(M, ecc)
     >>> maxerr = max(abs(E - ecc*sin(E) - (M % (2*np.pi)) ))
     >>> print("Maximum error = {:0.2e}".format(maxerr))
     Maximum error = 8.88e-16

    """
    M = M % (2*np.pi)
    if ecc == 0:
        return M
    if M > np.pi:
        M = 2*np.pi - M
        flip = True
    else:
        flip = False
    alpha = (3*np.pi + 1.6*(np.pi-np.abs(M))/(1+ecc) )/(np.pi - 6/np.pi)
    d = 3*(1 - ecc) + alpha*ecc
    r = 3*alpha*d * (d-1+ecc)*M + M**3
    q = 2*alpha*d*(1-ecc) - M**2
    w = (np.abs(r) + np.sqrt(q**3 + r**2))**(2/3)
    E = (2*r*w/(w**2 + w*q + q**2) + M) / d
    f_0 = E - ecc*np.sin(E) - M
    f_1 = 1 - ecc*np.cos(E)
    f_2 = ecc*np.sin(E)
    f_3 = 1-f_1
    d_3 = -f_0/(f_1 - 0.5*f_0*f_2/f_1)
    d_4 = -f_0/(f_1 + 0.5*d_3*f_2 + (d_3**2)*f_3/6)
    E = E -f_0/(f_1 + 0.5*d_4*f_2 + d_4**2*f_3/6 - d_4**3*f_2/24)
    if flip:
        E =  2*np.pi - E
    return E

#---------------

def t2z(t, tzero, P, sini, rstar, ecc=0, omdeg=90, returnMask=False):
    """
    Calculate star-planet separation relative to scaled stellar radius, z

    Optionally, return a flag/mask to indicate cases where the planet is
    further from the observer than the star, i.e., whether phases with z<1 are
    transits (mask==True) or eclipses (mask==False)

    :param t: time of observation (scalar or array)
    :param tzero: time of inferior conjunction, i.e., mid-transit
    :param P: orbital period
    :param sini: sine of orbital inclination
    :param rstar: scaled stellar radius, R_star/a
    :param ecc: eccentricity (optional, default=0)
    :param omdeg: longitude of periastron in degrees (optional, default=90)
    :param returnFlag: return a flag to distinguish transits from eclipses.

    N.B. omdeg is the longitude of periastron for the star's orbit

    :returns: z [, mask]

    :Example:
    
    >>> from pycheops.funcs import t2z
    >>> from numpy import linspace
    >>> import matplotlib.pyplot as plt
    >>> t = linspace(0,1,1000)
    >>> sini = 0.999
    >>> rstar = 0.1
    >>> plt.plot(t, t2z(t,0,1,sini,rstar))
    >>> plt.xlim(0,1)
    >>> plt.ylim(0,12)
    >>> ecc = 0.1
    >>> for omdeg in (0, 90, 180, 270):
    >>>     plt.plot(t, t2z(t,0,1,sini,rstar,ecc,omdeg))
    >>> plt.show()
        
    """
    if ecc == 0:
        nu = 2*np.pi*(t-tzero)/P
        omrad = 0.5*np.pi
        z = np.sqrt(1 - np.cos(nu)**2*sini**2)/rstar
    else:
        tp = tzero2tperi(tzero,P,sini,ecc,omdeg)
        M = 2*np.pi*(t-tp)/P
        E = esolve(M,ecc)
        nu = 2*np.arctan(np.sqrt((1+ecc)/(1-ecc))*np.tan(E/2))
        omrad = np.pi*omdeg/180
        # Equation (5.63) from Hilditch
        z = (((1-ecc**2)/
            (1+ecc*np.cos(nu))*np.sqrt(1-np.sin(omrad+nu)**2*sini**2))/rstar)
    if returnMask:
        return z, np.sin(nu + omrad)*sini < 0
    else:
        return z

#---------

def tzero2tperi(tzero,P,sini,ecc,omdeg):
    """
    Calculate time of periastron from time of mid-transit

    Uses the method by Lacy, 1992AJ....104.2213L

    :param tzero: times of mid-transit
    :param P: orbital period
    :param sini: sine of orbital inclination 
    :param ecc: eccentricity 
    :param omdeg: longitude of periastron in degrees

    :returns: time of periastron prior to tzero

    :Example:
     >>> from pycheops.funcs import tzero2tperi
     >>> tzero = 54321.6789
     >>> P = 1.23456
     >>> sini = 0.987
     >>> ecc = 0.654
     >>> omdeg = 89.01
     >>> print("{:0.4f}".format(tzero2tperi(tzero,P,sini,ecc,omdeg)))
     54321.6784

    """
    def _delta(th, sin2i, omrad, ecc):
        # Equation (4.9) from Hilditch
        return (1-ecc**2)*(
                np.sqrt(1-sin2i*np.sin(th+omrad)**2)/(1+ecc*np.cos(th)))

    omrad = omdeg*np.pi/180
    sin2i = sini**2
    theta = 0.5*np.pi-omrad
    if (1-sin2i) > np.finfo(0.).eps :
        ta = theta-0.125*np.pi
        tb = theta
        tc = theta+0.125*np.pi
        fa = _delta(ta, sin2i, omrad, ecc)
        fb = _delta(tb, sin2i, omrad, ecc)
        fc = _delta(tc, sin2i, omrad, ecc)
        if ((fb>fa)|(fb>fc)):
            t_ = np.linspace(0,2*np.pi,1024)
            d_ = _delta(t_, sin2i, omrad, ecc)
            try:
                i_= argrelextrema(d_, np.less)[0]
                t_ = t_[i_]
                if len(t_)>1:
                    i_ = (np.abs(t_ - tb)).argmin()
                    t_ = t_[i_]
                ta,tb,tc = (t_-0.01, t_, t_+0.01)
            except:
                print(sin2i, omrad, ecc)
                print(ta, tb, tc)
                print(fa, fb, fc)
                raise ValueError('tzero2tperi grid search fail')
        try:
            theta = brent(_delta, args=(sin2i, omrad, ecc), brack=(ta, tb, tc))
        except ValueError:
            print(sin2i, omrad, ecc)
            print(ta, tb, tc)
            print(fa, fb, fc)
            raise ValueError('Not a bracketing interval.')

    if theta == np.pi:
        E = np.pi 
    else:
        E = 2*np.arctan(np.sqrt((1-ecc)/(1+ecc))*np.tan(theta/2))
    return tzero - (E - ecc*np.sin(E))*P/(2*np.pi)

#---------

def tperi2tzero(tperi,P,sini,ecc,omdeg,eclipse=False):
    """
    Calculate phase mid-eclipse from time of mid-transit

    :param tperi: times of periastron passage
    :param P: orbital period
    :param sini: sine of orbital inclination 
    :param ecc: eccentricity 
    :param omdeg: longitude of periastron in degrees
    :param eclipse: calculate time of mid-eclipse if True, else mid-transit

    :returns: time of mid-eclipse 

    :Example:
     >>> from pycheops.funcs import tperi2tzero
     >>> tperi = 54321.6784
     >>> P = 1.23456
     >>> sini = 0.987
     >>> ecc = 0.654
     >>> omdeg = 89.01
     >>> t_transit = tperi2tzero(tperi,P,sini,ecc,omdeg)
     >>> t_eclipse = tperi2tzero(tperi,P,sini,ecc,omdeg,eclipse=True)
     >>> print(f"{t_transit:0.4f}, {t_eclipse:0.4f}")

    """
    def _delta(th, sin2i, omrad, ecc):
        # Equation (4.9) from Hilditch
        return (1-ecc**2)*(
                np.sqrt(1-sin2i*np.sin(th+omrad)**2)/(1+ecc*np.cos(th)))

    omrad = omdeg*np.pi/180
    sin2i = sini**2
    theta = 0.5*np.pi-omrad + np.pi*eclipse
    if (1-sin2i) > np.finfo(0.).eps :
        ta = theta-0.125*np.pi
        tb = theta
        tc = theta+0.125*np.pi
        fa = _delta(ta, sin2i, omrad, ecc)
        fb = _delta(tb, sin2i, omrad, ecc)
        fc = _delta(tc, sin2i, omrad, ecc)
        if ((fb>fa)|(fb>fc)):
            t_ = np.linspace(0,2*np.pi,1024)
            d_ = _delta(t_, sin2i, omrad, ecc)
            try:
                i_= argrelextrema(d_, np.less)[0]
                t_ = t_[i_]
                if len(t_)>1:
                    i_ = (np.abs(t_ - tb)).argmin()
                    t_ = t_[i_]
                ta,tb,tc = (t_-0.01, t_, t_+0.01)
            except:
                print(sin2i, omrad, ecc)
                print(ta, tb, tc)
                print(fa, fb, fc)
                raise ValueError('tzero2tperi grid search fail')
        try:
            theta = brent(_delta, args=(sin2i, omrad, ecc), brack=(ta, tb, tc))
        except ValueError:
            print(sin2i, omrad, ecc)
            print(ta, tb, tc)
            print(fa, fb, fc)
            raise ValueError('Not a bracketing interval.')

    if theta == np.pi:
        E = np.pi 
    else:
        E = 2*np.arctan(np.sqrt((1-ecc)/(1+ecc))*np.tan(theta/2))
    return tperi + (E - ecc*np.sin(E))*P/(2*np.pi)

#---------------

def eclipse_phase (P,sini,ecc,omdeg):
    """
    Calculate time of mid-transit/mid-eclipse from time of periastron

    Uses the method by Lacy, 1992AJ....104.2213L

    :param tzero: times of mid-transit
    :param P: orbital period
    :param sini: sine of orbital inclination 
    :param ecc: eccentricity 
    :param omdeg: longitude of periastron in degrees

    :returns: phase of mid-eclipse

    :Example:
     >>> from pycheops.funcs import eclipse_phase
     >>> P = 1.23456
     >>> sini = 0.987
     >>> ecc = 0.654
     >>> omdeg = 89.01
     >>> ph_ecl = eclipse_phase(tzero,P,sini,ecc,omdeg)
     >>> print(f"Phase of eclipse = {ph_ecl:0.4f}")

    """
    t_peri = tzero2tperi(0,P,sini,ecc,omdeg)
    t_ecl = tperi2tzero(t_peri,P,sini,ecc,omdeg,eclipse=True)
    return t_ecl/P % 1

#---------------


def nu_max(Teff, logg):
    """
    Peak frequency in micro-Hz for solar-like oscillations.

    From equation (17) of Campante et al., (2016)[2]_.

    :param logg: log of the surface gravity in cgs units.
    :param Teff: effective temperature in K

    :returns: nu_max in micro-Hz

    .. rubric References
    .. [2] Campante, 2016, ApJ 830, 138.

    """
    return 3090 * 10**(logg-4.438)/np.sqrt(Teff/5777)


#---------------

def vrad(t,tzero,P,K,ecc=0,omdeg=90,sini=1, primary=True):
    """
    Calculate radial velocity, V_r, for body in a Keplerian orbit

    :param t: array of input times 
    :param tzero: time of inferior conjunction, i.e., mid-transit
    :param P: orbital period
    :param K: radial velocity semi-amplitude 
    :param ecc: eccentricity (optional, default=0)
    :param omdeg: longitude of periastron in degrees (optional, default=90)
    :param sini: sine of orbital inclination (to convert tzero to t_peri)
    :param primary: if false calculate V_r for companion

    :returns: V_r in same units as K relative to the barycentre of the binary

    """
    tp = tzero2tperi(tzero,P,sini,ecc,omdeg)
    M = 2*np.pi*(t-tp)/P
    E = esolve(M,ecc)
    nu = 2*np.arctan(np.sqrt((1+ecc)/(1-ecc))*np.tan(E/2))
    omrad = np.pi*omdeg/180
    if not primary:
        omrad = omrad + np.pi
    return K*(np.cos(nu+omrad)+ecc*np.cos(omrad))

#---------------

def xyz_planet(t, tzero, P, sini, ecc=0, omdeg=90):
    """
    Position of the planet in Cartesian coordinates.

    The position of the ascending node is taken to be Omega=0 and the
    semi-major axis is taken to be a=1.

    :param t: time of observation (scalar or array)
    :param tzero: time of inferior conjunction, i.e., mid-transit
    :param P: orbital period
    :param sini: sine of orbital inclination
    :param ecc: eccentricity (optional, default=0)
    :param omdeg: longitude of periastron in degrees (optional, default=90)

    N.B. omdeg is the longitude of periastron for the star's orbit

    :returns: (x, y, z)

    :Example:
    
    >>> from pycheops.funcs import phase_angle
    >>> from numpy import linspace
    >>> import matplotlib.pyplot as plt
    >>> t = linspace(0,1,1000)
    >>> sini = 0.9
    >>> ecc = 0.1
    >>> omdeg = 90
    >>> x, y, z = xyz_planet(t, 0, 1, sini, ecc, omdeg)
    >>> plt.plot(x, y)
    >>> plt.plot(x, z)
    >>> plt.show()
        
    """
    if ecc == 0:
        nu = 2*np.pi*(t-tzero)/P
        r = 1
        cosw = 0
        sinw = -1
    else:
        tp = tzero2tperi(tzero,P,sini,ecc,omdeg)
        M = 2*np.pi*(t-tp)/P
        E = esolve(M,ecc)
        nu = 2*np.arctan(np.sqrt((1+ecc)/(1-ecc))*np.tan(E/2))
        r = (1-ecc**2)/(1+ecc*np.cos(nu))
        omrad = np.pi*omdeg/180
        # negative here since om_planet = om_star + pi
        cosw = -np.cos(omrad)
        sinw = -np.sin(omrad)
    sinv = np.sin(nu) 
    cosv = np.cos(nu)
    cosi = np.sqrt(1-sini**2)
    x = r*(-sinv*sinw + cosv*cosw)
    y = r*cosi*(cosv*sinw + sinv*cosw)
    z = -r*sini*(cosw*sinv + cosv*sinw)
    return x, y, z
        
#----------------------------------------------------------------------------

def massradius(P=None, k=None, sini=None, ecc=None,
        m_star=None, r_star=None, K=None, aR=None,
        jovian=False, solar=False, verbose=True, return_samples=False,
        plot=True, figsize=(8,6), xlim=None, ylim=None, 
        errorbar=True, err_kws={'capsize':4, 'color':'darkred', 'fmt':'o'},
        logmass=False, logradius=False, title=None,
        ellipse=True, ell_kws={'facecolor':'None','edgecolor':'darkblue'},
        ell_sigma=[1,2,3], tepcat=True, tepcat_kws={'s':8, 'c':'cadetblue'},
        show_legend=True, legend_kws={},
        zeng_models=['R100H2O','Rrock'], zeng_kws={}, 
        baraffe_models=['Z0.02_5Gyr','Z0.50_5Gyr'],
        baraffe_kws={}, lab_kws={}, tick_kws={}):
    """ 
    Calculate planet mass and/or radius

    Stellar mass and/or radius (m_star, r_star) are assumed to have solar
    units. The radial velocity semi-amplitude of the stars orbit, K, is
    assumed to have units of m/s. P is assumed to have units of days.
    
    Parameters can be specified in one of the following ways.
    - single value (zero error assumed)
    - ufloat values, i.e., m_star=ufloat(1.1, 0.05)
    - 2-tuple with value and standard deviation, e.g., m_star=(1.1, 0.05)
    - a numpy array of values sampled from the parameter's probability
      distribution

    If input values are numpy arrays of the same size, e.g., outputs from the
    same run of an emcee sampler, then they are sampled in the same way to
    ensure any correlations between these input parameters are preserved.

    If the orbital eccentricity is not given then it is assumed to be e=0
    (circular orbit).

    In the table below, the input and output quantities are
    - k = planet-star radius ratio r_p/r_star
    - sini = sine of orbital inclination
    - ecc = orbital eccentricity
    - K = semi-amplitude of star's spectroscopic orbit in m/s
    - aR = a/r_star
    - r_pl = planet radius
    - m_pl = planet mass
    - a = semi-major axis of the planet's orbit in solar radii
    - q = mass ratio = m_pl/m_star
    - g_p = planet's surface gravity (m.s-2)
    - rho_p = planet's mean density 
    - rho_star = mean stellar density in solar units

    +----------------------------+---------------+
    | Input                      | Output        |
    +============================+===============+
    | r_star, k                  | r_p           |
    | m_star, K, sini, P         | m_p, a, q     |
    | aR, k, sini, P, K          | g_p           |
    | aR, k, sini, P, K, m_star  | rho_p         |
    | aR, P, m_star, K, sini     | rho_star      |
    +----------------------------+---------------+

    The planet surface gravity, g_p, is calculated directly from k and aR
    using equation (4) from Southworth et al., MNRAS 2007MNRAS.379L..11S. The
    mean stellar density, rho_star, is calculated directly from aR using
    the equation from section 2.2 of Maxted et al. 2015A&A...575A..36M.

    By default, the units for the planet mass, radius and density are Earth
    mass, Earth radius and Earth density. Jovian mass, radius and density
    units can be selected by setting jovian=True. In both cases, the radius
    units are those for a sphere with the same volume as the Earth or Jupiter.
    Alternatively, solar units can be selected using solar=True.

    The following statistics are calculated for each of the input and output
    quantities and are returned as a python dict.  
    - mean
    - stderr (standard error)
    - mode (estimated using half-sample method)
    - median
    _ e_hi (84.1%-ile - median)
    _ e_lo (median - 15.9%-ile)
    - c95_up (95& upper confidence limit) 
    - c95_lo (95& lower confidence limit) 
    - sample (sample used to calculate statistics, if return_samples=True)

    An output plot showing the planet mass and radius relative to models
    and/or other known planets is generated if both the planet mass and
    radius can be calculated (unless plot=False is specified). Keyword options
    can be sent to ax.tick_params using the tick_kws option and similarly for
    ax.set_xlabel and ax.set_ylabel with the lab_kws option. The plot title
    can be set with the title keyword.

    The following models from Zeng et al. (2016ApJ...819..127Z) can be
    selected using the zeng_models keyword. 
     R100Fe,R50Fe,R30Fe,R25Fe,R20Fe,Rrock,R25H2O,R50H2O,R100H2O
    Set zeng_models=None to skip plotting of these models, or 'all' to plot
    them all. Keyword argument to the plot command for these models can be
    added using the zeng_kws option.

    Models from Baraffe et al., (2008A&A...482..315B) are available for
    metalicities Z=0.02, 0.10, 0.50 and 0.90, and ages 0.5Gyr, 1Gyr and 5Gyr.
    Models can be selected using the baraffe_models option using model names
    Z0.02_0.5Gyr, Z0.02_1Gyr, Z0.02_5Gyr, Z0.02_0.5Gyr, etc. Set
    baraffe_models=None to skip plotting of these models, or 'all' to plot
    them all. Keyword argument to the plot command for these models can be
    added using the baraffe_kws option.

    The keyword show_legend can be used to include a legend for the models
    plotted with keyword arguments legend_kws. 

    Well-studied planets from TEPCat will also be shown in the plot if
    tepcat=True. The appearance of the points can be controlled using
    kws_tepcat keyword arguments that are passed to plt.scatter.

    If errorbar=True the planet mass and radius are plotted as an error bar
    using plt.errorbar with optional keyword arguments err_kws. Logarithmic
    scales for the mass and radius axes can be selected with the logmass and
    logradius keywords.

    If ellipse=True then the planet mass and radius are shown using ellipses
    with semi-major axes set by the ell_sigma keyword. The appearance of these
    ellipses can be specified using the ell_kws keyword. These options are
    sent to the plt.add_patch command. 

    The return value of this function is "result, fig" or, if plot=False,
    "result",  where "result" is a python dict containing the statistics for
    each parameter and "fig" is a matplotlib Figure object.

    """

    NM=100_000  # No. of Monte Carlo simulations.

    # Generate a sample of values for a parameter
    def _s(x, nm=NM):
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

    # Generate dict of parameter statistics
    def _d(x):
        d = {}
        d['mean'] = x.mean()
        d['stderr'] = x.std()
        d['mode'] = mode(x)
        q = np.percentile(x, [5,15.8655,50,84.1345,95])
        d['median'] = q[2]
        d['e_hi'] = q[3]-q[2]
        d['e_lo'] = q[2]-q[1]
        d['c95_up'] = q[4]
        d['c95_lo'] = q[0]
        return d

    result = {}
    fig = None

    # Use e=0 if input value is none, otherwise sample in the range [0,1)
    _e = 0 if ecc is None else np.clip(np.abs(_s(ecc)),0,0.999999)

    # Look for input values that are numpy arrays of the same length, in which
    # case sample these together.
    pv = [P, k, sini, _e, m_star, r_star, K, aR]
    pn = ['P', 'k', 'sini', 'e', 'm_star', 'r_star', 'K', 'aR']
    ps = {}  # dictionary of samples for each input parameter
    _n = [len(p) if isinstance(p, np.ndarray) else 0 for p in pv]
    _u = np.unique(_n)
    for _m in _u[_u>0]:
        _i = np.where(_n == _m)[0]
        if len(_i) > 1:
            if _m == NM:
                _j = range(_m)
            elif _m > NM:
                _j = random_sample(range(_m), NM)
            else:
                _j = (np.random.random(NM)*_m).astype(int)
            for _k in _i:
                ps[pn[_k]] = pv[_k][_j]

    # Generate samples for input parameters not already sampled
    # N.B. All parameters assumed to be strictly positive so use abs() to
    # avoid negative values.
    for n in set(pn) - set(ps.keys()):
        _i = pn.index(n)
        ps[n] = None if pv[_i] is None else np.abs(_s(pv[_i]))

    if jovian:
        if solar: raise ValueError("Cannot specify both jovian and solar units")
        mfac = M_SunN/M_JupN 
        rfac = R_SunN/R_JupN 
        mstr = ' M_Jup'
        rstr = ' R_Jup'
    elif solar:
        mfac, rfac = 1, 1
        mstr = ' M_Sun'
        rstr = ' R_Sun'
    else:
        mfac = M_SunN/M_EarthN
        rfac = R_SunN/R_EarthN
        mstr = ' M_Earth'
        rstr = ' R_Earth'

    if ps['m_star'] is not None:
        result['m_star'] = _d(ps['m_star'])
        if verbose:
                print(parprint(ps['m_star'],'m_star',wn=8,w=10) + ' M_Sun')

    if ps['r_star'] is not None:
        result['r_star'] = _d(ps['r_star'])
        if verbose:
                print(parprint(ps['r_star'],'r_star',wn=8,w=10) + ' R_Sun')

    result['e'] = _d(ps['e'])
    if verbose:
        print(parprint(ps['e'],'e',wn=8,w=10))

    # Calculations start here. Intermediate variables names in result
    # dictionary start with "_" so we can remove/ignore them later.

    if ps['k'] is not None and ps['r_star'] is not None:
        ps['_rp'] = ps['k']*ps['r_star']   # in solar units
        ps['r_p'] = ps['_rp']*rfac         # in output units
        result['r_p'] = _d(ps['r_p'])
        if verbose:
            print(parprint(ps['r_p'],'r_p',wn=8,w=10) + rstr)
    
    if not True in [p is None for p in [m_star, sini, P, K]]:
        # Mass function in solar mass - careful to use K in km/s here
        _K = ps['K']/1000  # K in km/s
        ps['_fm'] = f_m(ps['P'], _K, ps['e']) 
        ps['_mp'] = m_comp(ps['_fm'], ps['m_star'], ps['sini'])  # solar units
        ps['m_p'] = ps['_mp']*mfac                  # in output units
        result['m_p'] = _d(ps['m_p']) 
        ps['q'] = ps['_mp']/ps['m_star']
        result['q'] = _d(ps['q'])
        ps['a'] = asini(_K*(1+1/ps['q']), ps['P'], ps['e']) / ps['sini']
        result['a'] = _d(ps['a'])
        if verbose:
            print(parprint(ps['m_p'],'m_p',wn=8,w=10) + mstr)
            print(parprint(ps['q'],'q',wn=8,w=10))
            print(parprint(ps['a'],'a',wn=8,w=10) + ' R_Sun')
            print(parprint(ps['a']*R_SunN/au,'a',wn=8,w=10) + ' au')
        if aR is not None:
            ps['rho_star'] = rhostar(1/ps['aR'], ps['P'], ps['q'])
            result['rho_star'] = _d(ps['rho_star'])
            if verbose:
                print(parprint(ps['rho_star'],'rho_star',wn=8,w=10)+' rho_Sun')

    if not True in [p is None for p in [k, aR, K, sini, P]]:
        _K = ps['K']/1000  # K in km/s
        ps['g_p'] = g_2(ps['k']/ps['aR'],ps['P'],_K,ps['sini'],ps['e'])
        result['g_p'] = _d(ps['g_p'])
        if verbose:
            print(parprint(ps['g_p'],'g_p',wn=8,w=10) + ' m.s-2')
            _loggp = np.log10(ps['g_p'])+2
            print(parprint(_loggp,'log g_p',wn=8,w=10)+' [cgs]')
        if m_star is not None:
            _rho = (3 * ps['g_p']**1.5 / 
                    ( 4*np.pi * G_2014**1.5 * (ps['_mp']*M_SunN)**0.5) )
            if jovian:
                rho_Jup  = M_JupN / (4/3*np.pi*R_JupN**3)
                ps['rho_p'] = _rho/rho_Jup
                rhostr = ' rho_Jup'
            elif solar:
                ps['rho_p'] = _rho
                rhostr = ' rho_Sun'
            else:
                rho_Earth  = M_EarthN / (4/3*np.pi*R_EarthN**3)
                ps['rho_p'] = _rho/rho_Earth
                rhostr = ' rho_Earth'
            if verbose:
                print(parprint(ps['rho_p'],'rho_p',wn=8,w=10) + rhostr)
                print(parprint(_rho*1e-3,'rho_p',wn=8,w=10)+' [g.cm-3]')
            result['rho_p'] = _d(ps['rho_p'])

    if return_samples:
        for k in result.keys():
            result[k]['sample'] = ps[k]

    if plot is False or not 'm_p' in result or not 'r_p' in result:
        return result
        
    # Plotting starts here
    
    _m = result['m_p']['median']
    _r = result['r_p']['median']
    fig, ax = plt.subplots(figsize=figsize)
    ax.tick_params(**tick_kws)
    if logmass: ax.set_xscale('log')
    if logradius: ax.set_yscale('log')
    if xlim: 
        ax.set_xlim(xlim)
    else:
        if logmass:
            ax.set_xlim(_m/10,_m*10)
        else:
            ax.set_xlim(0, _m*2)
    if ylim:
        ax.set_ylim(ylim)
    else:
        if logradius:
            ax.set_ylim(_r/10,_r*10)
        else:
            ax.set_ylim(0, _r*2)
    if jovian:
        ax.set_xlabel(r"$M/M_{\rm Jup}$", **lab_kws)
        ax.set_ylabel(r"$R/R_{\rm Jup}$", **lab_kws)
    elif solar:
        ax.set_xlabel(r"$M/M_{\odot}$", **lab_kws)
        ax.set_ylabel(r"$R/R_{\odot}$", **lab_kws)
    else:
        ax.set_xlabel(r"$M/M_{\oplus}$", **lab_kws)
        ax.set_ylabel(r"$R/R_{\oplus}$", **lab_kws)
    if title is not None:
        ax.set_title(title)


    if zeng_models is not None:
        if jovian:
            mfac, rfac = M_EarthN/M_JupN, R_EarthN/R_JupN
        elif solar:
            mfac, rfac = M_EarthN/M_SunN, R_EarthN/R_SunN
        else:
            mfac, rfac = 1,1
        mfile = join(_model_path,'apj522803t2_mrt.txt')
        T = Table.read(mfile, format='ascii.cds')
        if zeng_models == 'all':
            zeng_models = T.colnames[1:]
        for c in T.colnames[1:] if zeng_models == 'all' else zeng_models:
            ax.plot(T['Mass']*mfac,T[c]*rfac,**zeng_kws,label=c)

    if baraffe_models is not None:
        if jovian:
            mfac, rfac = M_EarthN/M_JupN, 1
        elif solar:
            mfac, rfac = M_EarthN/M_SunN, R_JupN/R_SunN
        else:
            mfac, rfac = 1, R_JupN/R_EarthN
        mfile = join(_model_path,'aa9321-07_table4.csv')
        T = Table.read(mfile, format='csv')
        if baraffe_models == 'all':
            baraffe_models = T.colnames[1:]
        for c in T.colnames[1:] if baraffe_models == 'all' else baraffe_models:
            ax.plot(T['Mass']*mfac,T[c]*rfac,**baraffe_kws,label=c)
    
    if show_legend:
        ax.legend(**legend_kws)

    if tepcat:
        if TEPCatPath.is_file():
            file_age = mktime(localtime())-getmtime(TEPCatPath)
            if file_age > int(config['TEPCat']['update_interval']):
                download = True
            else:
                download = False
        else:
            download = True
        if download:
            url = config['TEPCat']['download_url']
            try:
                req=requests.post(url)
            except:
                warnings.warn("Failed to update TEPCat data file from server")
            else:
                with open(TEPCatPath, 'wb') as file:
                    file.write(req.content)
                if verbose:
                    print('TEPCat data downloaded from \n {}'.format(url))
        # Awkward table to deal with because of repeated column names
        T = Table.read(TEPCatPath,format='ascii.no_header')
        M_b=np.array(T[T.colnames[list(T[0]).index('M_b')]][1:],dtype=np.float)
        R_b=np.array(T[T.colnames[list(T[0]).index('R_b')]][1:],dtype=np.float)
        ok = (M_b > 0) & (R_b > 0)
        M_b = M_b[ok]
        R_b = R_b[ok]
        if jovian:
            R_b = R_b*R_eJupN/R_JupN 
        elif solar:
            M_b = M_b*M_JupN/M_SunN
            R_b = R_b*R_eJupN/R_SunN
        else:
            M_b = M_b*M_JupN/M_EarthN
            R_b = R_b*R_eJupN/R_EarthN 
        ax.scatter(M_b,R_b, **tepcat_kws)

    if errorbar:
        ax.errorbar(_m, _r, 
                xerr=[[result['m_p']['e_lo']],[result['m_p']['e_hi']]],
                yerr=[[result['r_p']['e_lo']],[result['r_p']['e_hi']]],
                **err_kws)
    if ellipse:
        for nsig in ell_sigma:
            xy, w, h, theta = ellpar(ps['m_p'],ps['r_p'],nsig)
            ax.add_patch(Ellipse(xy, w, h, theta, **ell_kws))

    return result, fig
        
