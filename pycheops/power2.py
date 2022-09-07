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
from os.path import join,abspath,dirname,isfile
from astropy.table import Table

raise NotImplemented("Work in progress")

_data_path_ = join(dirname(abspath(__file__)),'data','limbdarkening')

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

    h1p = I(2/3) = 
    h2p = h1 - I(1/3) = 0.5**alpha - 0.1**alpha
    
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
        Parameter h2p = h1p - I(1/3) = (2/3)**alpha - (1/3)**alpha
        
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
        s += f'h\'_1    : {self.h1p:7.4f}\n'
        s += f'h\'_2    : {self.h2p:7.4f}\n'
        s += f'q_1      : {self.q1:7.4f}\n'
        s += f'q_2      : {self.q2:7.4f}\n'
        s += f'Passband :  {self.passband}\n'
        s += f'Source   :  {self.source}'
        return s
    
    @classmethod
    def lookup(cls, teff, logg, metal=0, passband='CHEOPS',
            table='stagger', empirical=True):
        """

        The available passband names are:

        * 'CHEOPS', 'MOST', 'Kepler', 'CoRoT', 'Gaia', 'TESS', 'PLATO'

        * 'U', 'B', 'V', 'R', 'I' (Bessell/Johnson)

        * 'u\_', 'g\_', 'r\_', 'i\_', 'z\_'  (SDSS)

        * 'NGTS'

        The available table names are:

        * 'stagger' - all passbands, see Maxted (2018) [1]
        * 'atlas' - CHEOPS, UBVRI, ugriz, Gaia, Kepler, and TESS
        * 'phoenix-cond' - CHEOPS, metal=0 only.

        Data for atlas tables are from Claret & Southworth (arXiv:2206.11098)
        and Claret (2021RNAAS...5...13C). See import_atlas_claret-2021.py for
        details.

         Data for phoenix-cond are from Claret (2021RNAAS...5...13C). The
        coefficients from these spherical atmosphere models cannot be used
        directly but have to be adjusted to a radius scale r'=r0.r, where
        r0=sqrt(1-mucri**2) is the radius at the stellar limb. This is done by
        matching the intensity profile at mu=2/3 and mu=1/3, or setting
        I(mu=0)=0 if the latter condition leads to negative intensities at
        the limb. See import_phoenix-cond-claret-2021.py for details.

        """
        if table == 'stagger':
            datfile = join(_data_path_, 'power2.dat')
            Tpower2 = Table.read(datfile,format='ascii',
                names=['Tag','T_eff','log_g','Fe_H','c','alpha','h1','h2'])
            tag = passband[0:min(len(passband),2)]
            T = Tpower2[(Tpower2['Tag'] == tag)]
            p = np.array([T['T_eff'],T['log_g'],T['Fe_H']]).T
            v = np.array([T['h1'], T['h2']]).T            
            f = LinearNDInterpolator(p,v)
            h1, h2 = f(teff, logg, metal)
            if np.isnan(h1) | np.isnan(h2):
                raise ValueError('teff, logg and/or metal out of range')
            return cls(h1=h1,h2=h2, passband=passband, source='STAGGER-grid')

        elif table == 'atlas':
            datfile = join(_data_path_, 'atlas-claret-2021.dat')
            Tatlas = Table.read(datfile,format='ascii',guess=False,
                names=['Tag','T_eff','log_g','Fe_H','c','alpha'])
            tag = passband[0:min(len(passband),2)]
            T = Tatlas[(Tatlas['Tag'] == tag)]
            if len(T) == 0:
                raise ValueError('no such passband')
            p = np.array([T['T_eff'],T['log_g'],T['Fe_H']]).T
            v = np.array([T['c'], T['alpha']]).T            
            f = LinearNDInterpolator(p,v)
            c, a = f(teff, logg, metal)
            if np.isnan(c) | np.isnan(a):
                raise ValueError('teff, logg and/or metal out of range')
            s = 'atlas-claret-2021'
            return cls(c=c, alpha=a, passband=passband, source=s)

        elif table == 'phoenix-cond':
            if metal != 0:
                raise ValueError('metal=0 only for phoenix-cond')
            if passband != 'CHEOPS':
                raise ValueError('no such passband')
            fl = join(_data_path_, 'phoenix-cond-claret-2021.dat')
            teff_, logg_, c_, a_ = np.loadtxt(fl, unpack=True)
            p = np.array([teff_, logg_]).T
            v = np.array([c_, a_]).T
            f = LinearNDInterpolator(p,v)
            c,a = f(teff, logg)
            if np.isnan(c) | np.isnan(a):
                raise ValueError('teff and/or logg  out of range')
            s = 'phoenix-cond-claret-2021'
            return cls(c=c, alpha=a, passband=passband, source=s)
        else:
            raise ValueError('no such table')
