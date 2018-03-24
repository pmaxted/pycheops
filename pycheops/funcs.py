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
from numpy import roots, imag, real, vectorize, isscalar, ceil
from numpy import arcsin, sqrt, pi, sin, cos, tan, arctan
from scipy.optimize import minimize_scalar

__all__ = [ 'a_rsun','f_m','m1sin3i','m2sin3i','asini','rhostar',
        'K_kms','m_comp','transit_width','esolve']

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

def esolve(M, e):
    """
    Solve Kepler's equation M = E - e.sin(E) 

    :param M: mean anomaly
    :param e: eccentricity

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

    def _esolve_scalar(M, e):
        if (e < 0) or (e >= 1):
            raise ValueError("Invalid eccentricity value")

        if e == 0:
            return M

        m = M % (2*pi)
        if m > pi:
            m = 2*pi - m
            flip = True
        else:
            flip = False

        alpha = (3*pi + 1.6*(pi-abs(m))/(1+e) )/(pi - 6/pi)
        d = 3*(1 - e) + alpha*e
        r = 3*alpha*d * (d-1+e)*m + m**3
        q = 2*alpha*d*(1-e) - m**2
        w = (abs(r) + sqrt(q**3 + r**2))**(2/3)
        E = (2*r*w/(w**2 + w*q + q**2) + m) / d
        f_0 = E - e*sin(E) - m
        f_1 = 1 - e*cos(E)
        f_2 = e*sin(E)
        f_3 = 1-f_1
        d_3 = -f_0/(f_1 - 0.5*f_0*f_2/f_1)
        d_4 = -f_0/(f_1 + 0.5*d_3*f_2 + (d_3**2)*f_3/6)
        E = E -f_0/(f_1 + 0.5*d_4*f_2 + d_4**2*f_3/6 - d_4**3*f_2/24)
        if flip:
            E =  2*pi - E
        return E

    _esolve_vector = vectorize(_esolve_scalar )

    if isscalar(M) & isscalar(e):
        return float(_esolve_scalar(M, e))
    else:
        return _esolve_vector(M, e)

