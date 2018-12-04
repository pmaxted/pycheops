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
Functions relating observable properties of binary stars and exoplanet 
systems to their fundamental properties, and vice versa. Also functions
related to Keplerian orbits.


Parameters
----------
Functions are defined in terms of the following parameters. [1]

* a          - orbital semi-major axis in solar radii = a_1 + a_2 
* P          - orbital period in mean solar days
* M          - total system mass in solar masses, M = m_1 + m_2
* e          - orbital eccentricity
* om         - longitude of periastron, omega, in _degrees_
* sini       - sine of the orbital inclination 
* K          - 2.pi.a.sini/(P.sqrt(1-e^2)) = K_1 + K_2
* K_1, K_2   - orbital semi-amplitudes in km/s
* q          - mass ratio = m_2/m_1 = K_1/K_2 = a_1/a_2
* f_m        - mass function = m_2^3.sini^3/(m_1+m_2)^2 in solar masses 
                             = K_1^3.P/(2.pi.G).(1-e^2)^(3/2)
* r_1        - radius of star 1 in units of the semi-major axis, r_1 = R_*/a
* rho_1      - mean stellar density = 3.pi/(GP^2(1+q)r_1^3)
  
.. rubric References
.. [1] Hilditch, R.W., An Introduction to Close Binary Stars, CUP 2001.


Functions 
---------

"""

from __future__ import (absolute_import, division, print_function,
                                unicode_literals)
from .constants import *
from numpy import roots, imag, real, vectorize, isscalar, isfinite, array, abs
from numpy import arcsin, sqrt, pi, sin, cos, tan, arctan, nan, empty_like
from scipy.optimize import brent
from numba import vectorize


__all__ = [ 'a_rsun','f_m','m1sin3i','m2sin3i','asini','rhostar',
        'K_kms','m_comp','transit_width','esolve','t2z',
        'tzero2tperi', 'vrad']

_arsun   = (GM_SunN*mean_solar_day**2/(4*pi**2))**(1/3.)/R_SunN
_f_m     = mean_solar_day*1e9/(2*pi)/GM_SunN
_asini   = mean_solar_day*1e3/2/pi/R_SunN
_rhostar = 3*pi*V_SunN/(GM_SunN*mean_solar_day**2)

def a_rsun(P, M):
    """
    Semi-major axis in solar radii

    :param P: orbital period in mean solar days
    :param M: total mass in solar masses

    :returns: a = (G.M.P^2/(4.pi^2))^(1/3) in solar radii
    
    """

    return _arsun * P**(2/3.) * M**(1/3.)

def f_m(P, K, e=0):
    """
    Mass function in solar masses

    :param P: orbital period in mean solar days
    :param K: semi-amplitude of the spectroscopic orbit in km/s
    :param e: orbital eccentricity

    :returns: f_m =  m_2^3.sini^3/(m_1+m_2)^2  in solar masses
    """
    return _f_m * K**3 * P * (1 - e**2)**1.5

def m1sin3i(P, K_1, K_2, e=0):
    """
     Reduced mass of star 1 in solar masses

     :param K_1: semi-amplitude of star 1 in km/s
     :param K_2: semi-amplitude of star 2 in km/s
     :param P: orbital period in mean solar days
     :param e:  orbital eccentricity

     :returns: m_1.sini^3 in solar masses 
    """
    return _f_m * K_2 * (K_1 + K_2)**2 * P * (1 - e**2)**1.5

def m2sin3i(P, K_1, K_2, e=0):
    """
     Reduced mass of star 2 in solar masses

     :param K_1:  semi-amplitude of star 1 in km/s
     :param K_2:  semi-amplitude of star 2 in km/s
     :param P:   orbital period in mean solar days
     :param e:   orbital eccentricity

     :returns: m_2.sini^3 in solar masses 
    """
    return _f_m * K_1 * (K_1 + K_2)**2 * P * (1 - e**2)**1.5

def asini(K, P, e=0):
    """
     a.sini in solar radii

     :param K: semi-amplitude of the spectroscopic orbit in km/s
     :param P: orbital period in mean solar days

     :returns: a.sin(i) in solar radii

    """
    return _asini * K * P 

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

def K_kms(m_1, m_2, P, sini, e):
    """
    Semi-amplitudes of the spectroscopic orbits in km/s
     - K = 2.pi.a.sini/(P.sqrt(1-e^2))
     - K_1 = K * m_2/(m_1+m_2)
     - K_2 = K * m_1/(m_1+m_2)

    :param m_1:  mass of star 1 in solar masses
    :param m_2:  mass of star 2 in solar masses
    :param P:  orbital period in mean solar days
    :param sini:  sine of the orbital inclination
    :param e:  orbital eccentrcity

    :returns: K_1, K_2 -- semi-amplitudes in km/s
    """
    M = m_1 + m_2
    a = a_rsun(P, M)
    K = 2*pi*a*R_SunN*sini/(P*mean_solar_day*sqrt(1-e**2))/1000
    K_1 = K * m_2/M
    K_2 = K * m_1/M
    return K_1, K_2

#---------------

def m_comp(f_m, m_1, sini):
    """
    Companion mass in solar masses given mass function and stellar mass

    :param f_m: = K_1^3.P/(2.pi.G).(1-e^2)^(3/2) in solar masses
    :param m_1: mass of star 1 in solar masses
    :param sini: sine of orbital inclination

    :returns: m_2 = mass of companion to star 1 in solar masses

    """

    def _m_comp_scalar(f_m, m_1, sini):
        if not isfinite(f_m*m_1*sini):
            return nan
        for r in roots([sini**3, -f_m,-2*f_m*m_1, -f_m*m_1**2]):
            if imag(r) == 0:
                return real(r)
        raise ValueError("No finite companion mass for input values.")

    _m_comp_vector = vectorize(_m_comp_scalar )

    if isscalar(f_m) & isscalar(m_1) & isscalar(sini):
        return float(_m_comp_scalar(f_m, m_1, sini))
    else:
        return _m_comp_vector(f_m, m_1, sini)

#---------------

def transit_width(r, k, b, p=1):
    """
    Total transit duration.

    See equation (3) from Seager and Malen-Ornelas, 2003ApJ...585.1038S.

    :param r: R_star/a
    :param k: R_planet/R_star
    :param b: impact parameter = a.cos(i)/R_star
    :param p: orbital period (optional, default=1)

    :returns: Total transit duration in the same units as p.

    """

    return p*arcsin(r*sqrt( ((1+k)**2-b**2) / (1-b**2*r**2) ))/pi

#---------------

@vectorize(nopython=True)
def esolve(M, e):
    """
    Solve Kepler's equation M = E - e.sin(E) 

    :param M: mean anomaly (scalar or array)
    :param e: eccentricity (scalar or array)

    :returns: eccentric anomaly, E

    Algorithm is from Markley 1995, CeMDA, 63, 101 via pyAstronomy class
    keplerOrbit.py

    :Example:

    Test precision using random values::
    
     >>> from pycheops.funcs import esolve
     >>> from numpy import pi, sin, abs, max
     >>> from numpy.random import uniform
     >>> e = uniform(0,1,1000)
     >>> M = uniform(-2*pi,4*pi,1000)
     >>> E = esolve(M, e)
     >>> maxerr = max(abs(E - e*sin(E) - (M % (2*pi)) ))
     >>> print("Maximum error = {:0.2e}".format(maxerr))
     Maximum error = 8.88e-16

    """
    M = M % (2*pi)
    if e == 0:
        return M
    if M > pi:
        M = 2*pi - M
        flip = True
    else:
        flip = False
    alpha = (3*pi + 1.6*(pi-abs(M))/(1+e) )/(pi - 6/pi)
    d = 3*(1 - e) + alpha*e
    r = 3*alpha*d * (d-1+e)*M + M**3
    q = 2*alpha*d*(1-e) - M**2
    w = (abs(r) + sqrt(q**3 + r**2))**(2/3)
    E = (2*r*w/(w**2 + w*q + q**2) + M) / d
    f_0 = E - e*sin(E) - M
    f_1 = 1 - e*cos(E)
    f_2 = e*sin(E)
    f_3 = 1-f_1
    d_3 = -f_0/(f_1 - 0.5*f_0*f_2/f_1)
    d_4 = -f_0/(f_1 + 0.5*d_3*f_2 + (d_3**2)*f_3/6)
    E = E -f_0/(f_1 + 0.5*d_4*f_2 + d_4**2*f_3/6 - d_4**3*f_2/24)
    if flip:
        E =  2*pi - E
    return E

#---------------

def t2z(t,t0,p,sini,rs,e=0,om=90):
    """
    Calculate star-planet separation

    :param t: time of observation (scalar or array)
    :param t0: time of inferior conjunction, i.e., mid-transit
    :param p: orbital period
    :param sini: sine of orbital inclination
    :param rs: scaled stellar radius, R_star/a
    :param e: eccentricity (optional, default=0)
    :param om: longitude of periastron in degrees (optional, default=90)

    :returns: star-planet separation relative to scaled stellar radius

    :Example:
    
    >>> from pycheops.funcs import t2z
    >>> from numpy import linspace
    >>> import matplotlib.pyplot as plt
    >>> t = linspace(0,1,1000)
    >>> sini = 0.999
    >>> rs = 0.1
    >>> plt.plot(t, t2z(t,0,1,sini,rs))
    >>> plt.xlim(0,1)
    >>> plt.ylim(0,12)
    >>> ecc = 0.1
    >>> for om in (0, 90, 180, 270):
    >>>     plt.plot(t, t2z(t,0,1,sini,rs,ecc,om))
    >>> plt.show()
        
    """
    if e == 0:
        return sqrt(1 - cos(2*pi*(t-t0)/p)**2*sini**2)/rs
    tp = tzero2tperi(t0,p,sini,e,om)
    M = 2*pi*(t-tp)/p
    E = esolve(M,e)
    nu = 2*arctan(sqrt((1+e)/(1-e))*tan(E/2))
    omrad = pi*om/180
    return ((1-e**2)/(1+e*cos(nu))*sqrt(1-sin(omrad+nu)**2*sini**2))/rs

#---------

def tzero2tperi(t0,p,sini,e,om):
    """
    Calculate time of periastron from time of mid-eclipse

    Uses the method by Lacy, 1992AJ....104.2213L

    :param t0: times of mid-eclipse
    :param p: orbital period
    :param sini: sine of orbital inclination 
    :param e: eccentricity 
    :param om: longitude of periastron in degrees

    :returns: time of periastron prior to t0

    :Example:
     To do

    """

    def _delta(th, sin2i, om, e):
        # Separation of centres of mass in units of a - equation (8) from
        # Lacy, 1992. 
        # theta = nu + om - pi/2 (7)
        return (1-e**2)*sqrt(1-sin2i*sin(th+om)**2)/(1+e*cos(th))

    omrad = om*pi/180
    theta = brent(_delta, 
            args = (sini**2, omrad, e),
            brack = (0.25*pi-omrad,0.5*pi-omrad,0.75*pi-omrad))
    if theta == pi:
        E = pi 
    else:
        E = 2*arctan(sqrt((1-e)/(1+e))*tan(theta/2))
    return t0 - (E - e*sin(E))*p/(2*pi)

#---------------

def vrad(t,t0,p,sini,K,e=0,om=90):
    """
    Calculate radial velocity, V_r, for body in a Keplerian orbit

    :param t: array of input times 
    :param t0: time of inferior conjunction, i.e., mid-transit
    :param p: orbital period
    :param sini:  sine of the orbital inclination
    :param K: radial velocity semi-amplitude 
    :param e: eccentricity (optional, default=0)
    :param om: longitude of periastron in degrees (optional, default=90)

    :returns: V_r in same units as K relative to the barycentre of the binary

    """
    tp = tzero2tperi(t0,p,sini,e,om)
    M = 2*pi*(t-tp)/p
    E = esolve(M,e)
    nu = 2*arctan(sqrt((1+e)/(1-e))*tan(E/2))
    omr = om*pi/180
    return K*(cos(nu+omr)+e*cos(omr))

