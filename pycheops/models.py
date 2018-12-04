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
Models for use within the celerite framework

"""

from __future__ import (absolute_import, division, print_function,
                                unicode_literals)
import numpy as np
from celerite.modeling import Model
from collections import OrderedDict
from numba import jit

__all__ = ['pModel','qpower2','ueclipse']

class pModel(Model):
    """

    A subclass of the celerite abstract class Model with Gaussian priors and
    default values and initial states (frozen/thawed) for parameters.

    Initial parameter values must be set by name as keyword arguments.

    Args:
        bounds (Optional[list or dict]): Bounds can be given for each
            parameter setting their minimum and maximum allowed values.
            This parameter can either be a ``list`` (with length
            ``full_size``) or a ``dict`` with named parameters. Any parameters
            that are omitted from the ``dict`` will be assumed to have no
            bounds. These bounds can be retrieved later using the
            :func:`celerite.Model.get_parameter_bounds` method and, by
            default, they are used in the :func:`celerite.Model.log_prior`
            method.

        priors (Optional[list or dict]): priors can be given for each
            parameter setting the mean and standard deviation of the Gaussian
            prior probability distribution.
            This parameter can either be a ``list`` (with length
            ``full_size``) or a ``dict`` with named parameters. Any parameters
            that are omitted from the ``dict`` will be assumed to have no
            priors (but bounds on the parameter, if specified, are applied).
            These priors can be retrieved later using the
            :func:`celerite.Model.get_parameter_priors` method and, by
            default, they are used in the :func:`celerite.Model.log_prior`
            method.
    """

    parameter_names = tuple()
    parameter_defaults = tuple()
    parameter_initial_thawed = tuple()  # True = thawed, False=frozen

    def __init__(self, *args, **kwargs):

        if len(self.parameter_names) != len(self.parameter_defaults):
            raise ValueError("the number of defaults must equal the number of "
                             "parameters")

        if len(self.parameter_names) != len(self.parameter_initial_thawed):
            raise ValueError("the number of initial states must equal the "
                             "number of parameters")

        self.unfrozen_mask = np.array(self.parameter_initial_thawed)
        self.dirty = True

        # Deal with bounds
        self.parameter_bounds = []
        bounds = kwargs.pop("bounds", dict())
        try:
            # Try to treat 'bounds' as a dictionary
            for name in self.parameter_names:
                self.parameter_bounds.append(bounds.get(name, (None, None)))
        except AttributeError:
            # 'bounds' isn't a dictionary - it had better be a list
            self.parameter_bounds = list(bounds)
        if len(self.parameter_bounds) != self.full_size:
            raise ValueError("the number of bounds must equal the number of "
                             "parameters")
        if any(len(b) != 2 for b in self.parameter_bounds):
            raise ValueError("the bounds for each parameter must have the "
                             "format: '(min, max)'")

        # Deal with priors
        self.parameter_priors = []
        priors = kwargs.pop("priors", dict())
        try:
            # Try to treat 'priors' as a dictionary
            for name in self.parameter_names:
                self.parameter_priors.append(priors.get(name, (None, None)))
        except AttributeError:
            # 'priors' isn't a dictionary - it had better be a list
            self.parameter_priors = list(priors)
        if len(self.parameter_priors) != self.full_size:
            raise ValueError("the number of priors must equal the number of "
                             "parameters")
        if any(len(b) != 2 for b in self.parameter_priors):
            raise ValueError("the priors for each parameter must have the "
                             "format: '(min, max)'")

        # Parameter values must be specified keywords
        params = list(self.parameter_defaults)
        # Loop over the kwargs and set the parameter values
        for k in kwargs:
            try:
                params[self.parameter_names.index(k)] = kwargs[k]
            except ValueError:
                raise ValueError("No such parameter name {}".format(k))
        self.parameter_vector = params

        # Check the initial prior value
        quiet = kwargs.get("quiet", False)
        if not quiet and not np.isfinite(self.log_prior()):
             raise ValueError("non-finite log prior value")


    def get_parameter_priors(self, include_frozen=False):
        """
        Get a list of the parameter priors
        Args:
            include_frozen (Optional[bool]): Should the frozen parameters be
                included in the returned value? (default: ``False``)
        """
        if include_frozen:
            return self.parameter_priors
        return list(p
                for p, f in zip(self.parameter_priors, self.unfrozen_mask)
                if f)

    def log_prior(self):
        """Compute the log prior probability of the current parameters"""
        for p, b in zip(self.parameter_vector, self.parameter_bounds):
            if b[0] is not None and p < b[0]:
                return -np.inf
            if b[1] is not None and p > b[1]:
                return -np.inf
        lp = 0.0
        for p,g in zip(self.parameter_vector, self.parameter_priors):
            if g[0] is not None and g[1] is not None:
                lp -= 0.5*((p-g[0])/g[1])**2

        return lp

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

#class TransitModel(pModel):
#    parameter_names = ("T_0", "P", "D", "W", "S", "F", "h_1", "h_2",
#            "dFdx", "dFdy","d2Fdx2", "d2Fdy2", "d2Fdxdy",
#            "dFdt", "dFdt2",
#            "Fratio", "f_s", "f_c")
#


#class EclipseModel(pModel):
#    parameter_names = ("Fratio", "f_s", "f_c", "asini",
#            "T_0", "P", "D", "W", "S", "F", 
#            "dFdx", "dFdy","d2Fdx2", "d2Fdy2", "d2Fdxdy",
#            "dFdt", "dFdt2")
#



#class RVModel(pModel):
#    parameter_names = ("T_0", "P", "K", "f_s", "f_c", V, dVdt)
#

