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
models
======
Models and likelihood functions for use with lmfit

"""

from __future__ import (absolute_import, division, print_function,
                                unicode_literals)
import numpy as np
from lmfit.model import Model
from lmfit.models import COMMON_INIT_DOC
from numba import jit
from warnings import warn

__all__ = ['qpower2', 'ueclipse', 'TransitModel']

@jit()
def qpower2(z,k,c,a):
    """
    Fast and accurate transit light curves for the power-2 limb-darkening law

    The power-2 limb-darkening law is I(mu) = 1 - c (1 - mu**a)

    Light curves are calculated using the qpower2 approximation [1]. The
    approximation is accurate to better than 100ppm for radius ratio k < 0.1.

    N.B. qpower2 is untested/inaccurate for values of k > 0.2

    .. [1] Maxted, P.F.L. & Gill, S., 2019, accepted for publication in A&A.

    :param z: star-planet separation on the sky cf. star radius (array)
    :param k: planet-star radius ratio (scalar, k<1) 
    :param c: power-2 limb darkening coefficient
    :param a: power-2 limb darkening exponent

    :returns: light curve (observed flux)  

    :Example:
    >>> from pycheops.models import qpower2
    >>> from pycheops.funcs import t2z
    >>> from numpy import linspace
    >>> import matplotlib.pyplot as plt
    >>> t = linspace(-0.025,0.025,1000)
    >>> sini = 0.999
    >>> rstar = 0.05
    >>> ecc = 0.2
    >>> om = 120
    >>> tzero = 0.0
    >>> P = 0.1
    >>> z=t2z(t,tzero,P,sini,rstar,ecc,om)
    >>> c = 0.5
    >>> a = 0.7
    >>> k = 0.1
    >>> f = qpower2(z,k,c,a)
    >>> plt.plot(t,f)
    >>> plt.show()

    """

    if (k > 1):
        raise ValueError("qpower2 requires k < 1")

    if (k > 0.2):
        warn ("qpower2 is untested/inaccurate for values of k > 0.2")

    f = np.ones_like(z)
    I_0 = (a+2)/(np.pi*(a-c*a+2))
    g = 0.5*a
    for i,zi in enumerate(z):
        zt = np.abs(zi)
        if zt <= (1-k):
            s = 1-zt**2
            c0 = (1-c+c*s**g)
            c2 = 0.5*a*c*s**(g-2)*((a-1)*zt**2-1)
            f[i] = 1-I_0*np.pi*k**2*(
                    c0 + 0.25*k**2*c2 - 0.125*a*c*k**2*s**(g-1) )
        elif np.abs(zt-1) < k:
            d = (zt**2 - k**2 + 1)/(2*zt)
            ra = 0.5*(zt-k+d)
            rb = 0.5*(1+d)
            sa = 1-ra**2
            sb = 1-rb**2
            q = min(max(-1.,(zt-d)/k),1.)
            w2 = k**2-(d-zt)**2
            w = np.sqrt(w2)
            b0 = 1 - c + c*sa**g
            b1 = -a*c*ra*sa**(g-1)
            b2 = 0.5*a*c*sa**(g-2)*((a-1)*ra**2-1)
            a0 = b0 + b1*(zt-ra) + b2*(zt-ra)**2
            a1 = b1+2*b2*(zt-ra)
            aq = np.arccos(q)
            J1 = ( (a0*(d-zt)-(2/3)*a1*w2 + 
                0.25*b2*(d-zt)*(2*(d-zt)**2-k**2))*w
                 + (a0*k**2 + 0.25*b2*k**4)*aq )
            J2 = a*c*sa**(g-1)*k**4*(
                0.125*aq + (1/12)*q*(q**2-2.5)*np.sqrt(max(0.,1-q**2)) )
            d0 = 1 - c + c*sb**g
            d1 = -a*c*rb*sb**(g-1)
            K1 = ((d0-rb*d1)*np.arccos(d) + 
                    ((rb*d+(2/3)*(1-d**2))*d1 - d*d0) * 
                    np.sqrt(max(0.,1-d**2)) )
            K2 = (1/3)*c*a*sb**(g+0.5)*(1-d)
            f[i] = 1 - I_0*(J1 - J2 + K1 - K2)
    return f

@jit()
def ueclipse(z,k,f):
    """
    Eclipse light curve for a planet with uniform surface brightness by a star

    :param z: star-planet separation on the sky cf. star radius (array)
    :param k: planet-star radius ratio (scalar, k<1) 
    :param f: planet-star flux ratio (scalar) 

    :returns: light curve (observed flux)  
    """
    if (k > 1):
        raise ValueError("ueclipse requires k < 1")

    fl = np.ones_like(z)
    for i,zi in enumerate(z):
        zt = np.abs(zi)
        if zt <= (1-k):
            fl[i] = 1/(1+f)
        elif np.abs(zt-1) < k:
            t1 = np.arccos(min(max(-1,(zt**2+k**2-1)/(2*zt*k)),1))
            t2 = np.arccos(min(max(-1,(zt**2+1-k**2)/(2*zt)),1))
            t3 = 0.5*np.sqrt(max(0,(1+k-zt)*(zt+k-1)*(zt-k+1)*(zt+k+1)))
            fl[i] = 1 - f/(1+f)*(k**2*t1 + t2 - t3)/(np.pi*k**2)
    return fl

def _pdsv_func(t, F, dFdx, dFdy, d2Fdxdy, d2Fdx2, d2Fdy2, xy=None):
    if xy is None:
        return 1
    else:
        x = xy['x'](t)
        y = xy['y'](t)
        return F + dFdx*x + dFdy*y + d2Fdxdy*x*y + d2Fdx2*x**2 + d2Fdy2*y**2

def _transit_func(t, T_0, P, D, W, S, f_c, f_s, h_1, h_2, 
        F, dFdx, dFdy, d2Fdxdy, d2Fdx2, d2Fdy2, xy=None):
    # Note: x, y is included in the list of keyword arguments to avoid
    # UserWarning when called from lmfit, it is not used in this model.
    # 
    from pycheops.funcs import t2z
    from pycheops.models import qpower2

    k = np.sqrt(D)
    r_star = 0.5*np.pi*np.sqrt(W**2*(1-S**2)/k)
    sini = np.sqrt(1 - ((1-k)**2 - S**2*(1+k)**2)/(1-S**2)*r_star**2)
    c2 = 1 - h_1 + h_2
    a2 = np.log2(c2/h_2)
    z = t2z(t, T_0, P, sini, r_star, signFlag = True)
    # Set z values where planet is behind star to a large nominal value
    z[z < 0]  = 9999
    pdsv = _pdsv_func(t, F, dFdx, dFdy, d2Fdxdy, d2Fdx2, d2Fdy2, xy=xy)
    return qpower2(z, k, c2, a2)*pdsv

class TransitModel(Model):
    r"""Light curve model for the transit of a spherical star by an opaque
    spherical body (planet).

    Limb-darkening is described by the power-2 law:
    .. math::
        I(\mu; c, \alpha) = 1 - c (1 - mu^\alpha)

    The light curve depth, width and shape are parameterised by D, W, and S as
    defined below in terms of the star and planet radii, R_s and R_p,
    respectively, the semi-major axis, a, and the orbital inclination, i. The
    following parameters are used for convenience - k = R_p/R_s,
    b=a.cos(i)/R_s. The shape parameter is approximately (t_F/t_T)^2 where
    t_T=W*P is the duration of the transit (1st to 4th contact points) and t_F
    is the duration of the "flat" part of the transit between the 2nd and 3rd
    contact points. The eccentricity and longitude of periastron for the
    planet's orbit are ecc and omega, respectively. These parameters are all
    available as constraints within the model.

    The model includes a position-dependent sensitivity variation model
    F + dFdx*x + dFdy*y + d2Fdxdy*x*y + d2Fdx2*x**2 + d2Fdy2*y**2

    :param t:    - independent variable (time)
    :param T_0:  - time of mid-transit
    :param P:    - orbital period
    :param D:    - (R_p/R_s)^2 = k^2
    :param W:    - (R_*/a)*sqrt((1+k)^2 - b^2)/pi
    :param S:    - ((1-k)^2-b^2)/((1+k)^2 - b^2)
    :param f_c:  - sqrt(ecc).cos(omega)
    :param f_s:  - sqrt(ecc).sin(omega)
    :param h_1:  - I(0.5) = 1 - c*(1-0.5**alpha)
    :param h_2:  - I(0.5) - I(0) = c*0.5**alpha

    The flux value outside of transit is 1. The light curve is calculated using
    the qpower2 algorithm, which is fast but only accurate for k < ~0.3.

    """

    def __init__(self, independent_vars=['t'], prefix='', nan_policy='raise',
                 **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})
        super(TransitModel, self).__init__(_transit_func, **kwargs)
        self._set_paramhints_prefix()

    def _set_paramhints_prefix(self):
        self.set_param_hint('P', min=1e-15)
        self.set_param_hint('D', min=0, max=1)
        self.set_param_hint('W', min=0, max=0.3)
        self.set_param_hint('S', min=0, max=1)
        self.set_param_hint('f_c', value=0, min=-1, max=1)
        self.set_param_hint('f_s', value=0, min=-1, max=1)
        self.set_param_hint('h_1', min=0, max=1)
        self.set_param_hint('h_2', min=0, max=1)
        expr = "sqrt({prefix:s}D)".format(prefix=self.prefix)
        self.set_param_hint('k', expr=expr, min=0, max=1)
        self.set_param_hint('R_s',min=0,max=1, expr=
          "0.5*pi*{p:s}W*sqrt((1-{p:s}S)/k)".format(p=self.prefix) )
        self.set_param_hint('bsq', min=0,  expr = 
          "((1-k)**2-{p:s}S*(1+k)**2)/(1-{p:s}S)".format(p=self.prefix) )
        self.set_param_hint('b', max=1.3, expr = 
          "sqrt(bsq)".format(p=self.prefix) )
        self.set_param_hint('F', min=0, value=1)
        self.set_param_hint('dFdx', value=0, vary=False)
        self.set_param_hint('dFdy', value=0, vary=False)
        self.set_param_hint('d2Fdxdy', value=0, vary=False)
        self.set_param_hint('d2Fdx2', value=0, vary=False)
        self.set_param_hint('d2Fdy2', value=0, vary=False)

