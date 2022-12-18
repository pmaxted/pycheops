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
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from os.path import join,abspath,dirname
from astropy.table import Table
from uncertainties import ufloat, UFloat

class Power2(object):
    """Object class for handling power-2 limb-darkening law

    The power-2 limb-darkening law has the form 
    
    I(mu) = 1 - c * (1-mu**alpha)
    
    Maxted (2018) [1]_ defines the following parameters that are uncorrelated
    for fits to typical transit light curves:
    
    h1 = I(0.5) = 1 - c*(1-0.5**alpha)
    h2 = h1 - I(0) = c*0.5**alpha

    Short et al. (2019) [2]_ describe the following parameters that span the
    range of valid c,alpha values for 0 < q1 < 1 and 0 < q2 < 1
    
    q1 = (1 - h2)**2
    q2 = (h1 - h2)/(1-h2)
    
    Different definitions for the radius at which mu=0 exist so the value of
    h2 is ambiguous. To avoid this issue, we can use the following parameters
    instead:

    h1p = I(2/3) = 1 - c*(1-(2/3)**alpha)
    h2p = h1p - I(1/3) = c*(2/3)**alpha - c*(1/3)**alpha

    Priors based on these parameters and the results published in Maxted 2022
    can be used to calculate semi-empirical priors on the limb-darkening for
    fits to light curves of solar-type stars.
    
    Attributes
    ----------

    c : float
        coefficient c

    alpha : float
        exponent alpha. For numerical stability, restrict 0.001 < alpha < 1000.

    h1 : float
        Parameter h1 = I(0.5) = 1 - c*(1-0.5**alpha)

    h2 : float
        Parameter h2 = h1 - I(0) = c*0.5**alpha

    h1p : float
        Parameter h1p = I(2/3) = 1 - c*(1 - (2/3)**alpha)
        
    h2p : float
        Parameter h2p = h1p - I(1/3) = c*(2/3)**alpha - c*(1/3)**alpha
        
    q1 : float
        Parameter q1 = (1 - h2)**2

    q2 : float
        Parameter q2 = (h1 - h2)/(1-h2)

    Returns
    -------
     Instance of Power2 object or raise ValueError if input values are
     invalid (negative flux or positive flux gradient df/dr).

    .. rubric:: References
    .. [1] Maxted, P.F.L., 2018, A&A, 616, A39
    .. [2] Short, D.R., et al., 2019, RNAAS, 3, 117

    """
    
    __interpolator_passband__ = None

    def _badq(self, q1,q2):
        return (q1<0) | (q1>1) | (q2<0) | (q2>1)

    def _ca_to_h1h2(self, c, a):
        return 1 - c*(1-0.5**a), c*0.5**a

    def _h1h2_to_ca(self, h1, h2):
        return 1 - h1 + h2, np.log2((1 - h1 + h2)/h2)

    def _h1h2_to_q1q2(self, h1, h2):
        return (1 - h2)**2, (h1 - h2)/(1-h2)

    def _q1q2_to_h1h2(self, q1, q2):
        if self._badq(q1, q2):
            return np.nan, np.nan
        return 1 - np.sqrt(q1) + q2*np.sqrt(q1), 1 - np.sqrt(q1)

    def _ca_to_h1ph2p(self, c, a):
        return 1 - c*(1-(2/3)**a), c*((2/3)**a - (1/3)**a)

    def _h1ph2p_to_ca(self, h1p, h2p):
        def _f(a, h1p, h2p):
            return ((2/3)**a - (1/3)**a)*(1-h1p)/(1-(2/3)**a) - h2p    
        try:
            r = root_scalar(_f, 
                        bracket=(self._amin, self._amax), 
                        x0=0.8, 
                        method='bisect', 
                        args=(h1p, h2p))
        except ValueError:
            return np.nan, np.nan
        return (1-h1)/(1-(2/3)**r.root), r.root

    def __init__(self, c=None, alpha=None, h1=None, h2=None,
            h1p=None, h2p=None, q1=None, q2=None, passband=None,
            source='User'):

        if sum(x is not None for x in [alpha,c,h1,h2,h1p,h2p,q1,q2]) != 2:
            raise ValueError('specify two input values to initialize object.')

        self._amin = 0.0001
        self._amax = 10000
            
        if c is not None and alpha is not None:
            if any([h1, h2, h1p, h2p, q1, q2]):
                raise ValueError('only two input values can be specified')
            if (c < 0) | (c > 1):
                raise ValueError('c outside range 0 < c < 1')
            if (alpha < self._amin) | (alpha > self._amax):
                raise ValueError('alpha outside range 0.01 < alpha < 100')
            h1, h2 = self._ca_to_h1h2(c, alpha)
            q1, q2 = self._h1h2_to_q1q2(h1, h2)
            if self._badq(q1, q2):
                raise ValueError('invalid c, alpha combination')
            a = alpha
            h1p, h2p = self._ca_to_h1ph2p(c, alpha)

        elif h1 is not None and h2 is not None:
            if any([c, alpha, h1p, h2p, q1, q2]):
                raise ValueError('only two input values can be specified')
            q1, q2 = self._h1h2_to_q1q2(h1, h2)
            if self._badq(q1, q2):
                raise ValueError('invalid h1, h2 combination')
            c, a = self._h1h2_to_ca(h1, h2)
            _, h2pp = self._ca_to_h1ph2p(c, a)

        elif h1p is not None and h2p is not None:
            if any([c, alpha, h1, h2, q1, q2]):
                raise ValueError('only two input values can be specified')
            c, a  = self._h1ph2p_to_ca(h1p, h2p)
            if np.isnan(c) | np.isnan(a):
                raise ValueError('invalid h1p, h2p combination')
            h1, h2 = self._ca_to_h1h2(c, a)
            q1, q2 = self._h1h2_to_q1q2(h1, h2)
            if self._badq(q1, q2):
                raise ValueError('invalid h1p, h2p combination')

        elif q1 is not None and q2 is not None:
            if any([c, alpha, h1, h2, h1p, h2p]):
                raise ValueError('only two input values can be specified')
            if self._badq(q1, q2):
                raise ValueError('invalid q1, q2 value(s).')
            h1, h2 = self._q1q2_to_h1h2(q1, q2)
            c, a = self._h1h2_to_ca(h1, h2)
            h1p, h2p = self._ca_to_h1ph2p(c, a)

        else:
            raise ValueError('invalid combination of initial parameters')

        self.c = c
        self.alpha = a
        self.h1 = h1
        self.h2 = h2
        self.q1 = q1
        self.q2 = q2
        self.h1p = h1p
        self.h2p = h2p
        self.passband = passband
        self.source = source

    def __call__(self, mu):
        """
        Evaulate power-2 limb-darkening law

        Parameters
        ----------
        mu : float, array-like
            cosine of the angle between the line-of-sight and the surface normal
        Returns
        -------
            I(\mu) = 1 - c (1 - \mu^\alpha)

        """
        if np.isscalar(mu):
            return 1 - self.c*(1-mu**self.alpha)
        else:
            return 1 - self.c*(1-np.array(mu)**self.alpha)
        
    def __repr__(self):
        s = ''
        s += f'c        : {self.c:7.4f}\n'
        s += f'alpha    : {self.alpha:7.4f}\n'
        s += f'h_1      : {self.h1:7.4f}\n'
        s += f'h_2      : {self.h2:7.4f}\n'
        s += f'h\'_1     : {self.h1p:7.4f}\n'
        s += f'h\'_2     : {self.h2p:7.4f}\n'
        s += f'q_1      : {self.q1:7.4f}\n'
        s += f'q_2      : {self.q2:7.4f}\n'
        s += f'Passband :  {self.passband}\n'
        s += f'Source   :  {self.source}'
        return s
    
    @classmethod
    def lookup(self, teff, logg, metal, passband='CHEOPS'):
        """

        Limb-darkening parameters from MPS-ATLAS models using linear
        interpolation in Table 5 from Kostogryz et al. (arXiv:2206.06641).

        The available passband names are 'CHEOPS', 'Kepler', 'TESS' and
        'PLATO'

        Input values for T_eff, log g and metallicity [M/H] can be float or
        ufloat values.

        """

        if not passband in ['CHEOPS', 'Kepler', 'TESS', 'PLATO']:
            raise ValueError('no such passband')

        if passband != self.__interpolator_passband__:
            d = join(dirname(abspath(__file__)), 'data','limbdarkening')
            f = join(d, 'table5.txt.fits')
            T = Table.read(f)
            T = T[T['Band'] == passband]
            p = np.array([T['Teff'],T['logg'],T['met']]).T
            v = np.array([T['c'], T['alpha']]).T
            self.__table5_interpolator__ = LinearNDInterpolator(p,v)
            self.__interpolator_passband__ = passband

        if np.isnan(c) | np.isnan(a):
            raise ValueError('parameters out of range')

        def isu(x):
            return isinstance(x, UFloat)

        if isu(teff) | isu(logg) | isu(metal):
            teff0 = teff.n if isu(teff) else teff
            logg0 = logg.n if isu(logg) else logg
            metal0 = metal.n if isu(metal) else metal
            c0,a0 = self.__table5_interpolator__(teff0, logg0, metal0)
        else:
            c,a = self.__table5_interpolator__(teff, logg, metal)

        return self(c=c, alpha=a, passband=passband, source='MPS-ATLAS')

    def priors(self, empirical=True):
        """
        Priors for h'_1 h'_2.

        If empirical=True (default) then apply empirical corrections and error
        budget from Maxted (2022).

        Empirical corrections for the CHEOPS and PLATO bands are assumed to be
        equal to the observed offsets for the Kepler band.

        Returns
        -------
        h1p, h2p: ufloat,ufloat

        """

        if self.passband == 'TESS':
            dh1p = ufloat(+0.004, 0.004) + ufloat(0, 0.002)
            dh2p = ufloat(-0.009, 0.004) + ufloat(0, 0.002) + ufloat(0, 0.005)
        else:
            dh1p = ufloat(+0.006, 0.002) + ufloat(0, 0.004)
            dh2p = ufloat(-0.012, 0.004) + ufloat(0, 0.012) + ufloat(0, 0.005)

        h1p = self.h1p + dh1p
        h2p = self.h2p + dh2p
        return h1p, h2p
