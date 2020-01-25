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
StarProperties
==============
 Object class to obtain/store observed properties of a star and to infer
 parameters such as radius and density.

"""

from __future__ import (absolute_import, division, print_function,
                                unicode_literals)
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
import requests
from .core import load_config
from pathlib import Path
from os.path import getmtime
from time import localtime, mktime
from uncertainties import ufloat, UFloat
from .ld import stagger_power2_interpolator
from numpy.random import normal


class StarProperties(object):
    """
    CHEOPS StarProperties object

    """

    def __init__(self, identifier, force_download=False, 
            match_arcsec=5, configFile=None,
            teff=None, logg=None, metal=None, 
            verbose=True):

        self.identifier = identifier
        coords = SkyCoord.from_name(identifier)
        self.ra = coords.ra.to_string(precision=2,unit='hour',sep=':',pad=True)
        self.dec = coords.dec.to_string(precision=1,sep=':',unit='degree',
                alwayssign=True,pad=True)

        config = load_config(configFile)
        _cache_path = config['DEFAULT']['data_cache_path']
        sweetCatPath = Path(_cache_path,'sweetcat.tsv')

        download_sweetcat = False
        if force_download:
            download_sweetcat = True
        elif sweetCatPath.is_file():
            file_age = mktime(localtime())-getmtime(sweetCatPath)
            if file_age > int(config['SWEET-Cat']['update_interval']):
                download_sweetcat = True
        else:
            download_sweetcat = True

        if download_sweetcat:
            url = config['SWEET-Cat']['download_url']
            req=requests.post(url)
            with open(sweetCatPath, 'wb') as file:
                file.write(req.content)
            if verbose:
                print('SWEET-Cat data downloaded from \n {}'.format(url))

        names = ['star', 'hd', 'ra', 'dec', 'vmag', 'e_vmag', 'par', 'e_par',
                'parsource', 'teff', 'e_teff', 'logg', 'e_logg', 'logglc',
                'e_logglc', 'vt', 'e_vt', 'metal', 'e_metal', 'mass', 'e_mass',
                'author', 'source', 'update', 'comment']
        sweetCat = Table.read(sweetCatPath,format='ascii.no_header', 
                delimiter="\t", fast_reader=False, names=names,
                encoding='utf-8')

        catalog_c = SkyCoord(sweetCat['ra'],sweetCat['dec'],unit='hour,degree')
        idx, sep, _ = coords.match_to_catalog_sky(catalog_c)
        if sep.arcsec[0] > match_arcsec:
            raise ValueError('No matching star in SWEET-Cat')

        entry = sweetCat[idx]
        ld_data_missing = False
        try:
            self.teff = ufloat(float(entry['teff']),float(entry['e_teff']))
            self.teff_note = "SWEET-Cat"
        except:
            self.teff = None
            ld_data_missing = True
        try:
            self.logg = ufloat(float(entry['logg']),float(entry['e_logg']))
            self.logg_note = "SWEET-Cat"
        except:
            self.logg = None
            ld_data_missing = True
        try:
            self.metal = ufloat(float(entry['metal']),float(entry['e_metal']))
            self.metal_note = "SWEET-Cat"
        except:
            self.metal = None
            ld_data_missing = True

        # User defined values
        if teff is not None:
           if  isinstance(teff, UFloat):
               self.teff = teff
               self.teff_note = "User"
           else:
               raise ValueError("teff keyword is not ufloat")
        if logg is not None:
           if  isinstance(logg, UFloat):
               self.logg = logg
               self.logg_note = "User"
           else:
               raise ValueError("logg keyword is not ufloat")
        if metal is not None:
           if  isinstance(metal, UFloat):
               self.metal = metal
               self.metal_note = "User"
           else:
               raise ValueError("metal keyword is not ufloat")

        # log rho from log g using method of Moya et al.
        # (2018ApJS..237...21M). Accuracy is 4.4%
        self.logrho = None
        if self.logg is not None:
            if (self.logg.n > 3.697) and (self.logg.n < 4.65):
                logrho = -7.352 + 1.6580*self.logg
                self.logrho = ufloat(logrho.n, np.hypot(logrho.s, 0.044))

        self.h_1 = None
        self.h_2 = None
        if not ld_data_missing:
            power2 = stagger_power2_interpolator()
            _,_,h_1,h_2 = power2(self.teff.n,self.logg.n,self.metal.n)
            if not np.isnan(h_1):
                Xteff = normal(self.teff.n, self.teff.s, 256)
                Xlogg = normal(self.logg.n, self.logg.s, 256)
                Xmetal = normal(self.metal.n, self.metal.s, 256)
                X = power2(Xteff,Xlogg,Xmetal)
                # Additinal error derived in Maxted, 2019
                e_h_1 = np.hypot(0.01,np.sqrt(np.nanmean((X[:,2]-h_1)**2)))
                e_h_2 = np.hypot(0.05,np.sqrt(np.nanmean((X[:,3]-h_2)**2)))
                self.h_1 = ufloat(round(h_1,3),round(e_h_1,3))
                self.h_2 = ufloat(round(h_2,3),round(e_h_2,3))



    def __repr__(self):
        s =  'Identifier : {}\n'.format(self.identifier)
        s +=  'Coordinates: {} {}\n'.format(self.ra, self.dec)
        if self.teff is not None:
            s += 'T_eff : {:5.0f} +/- {:0.0f} K    [{}]\n'.format(
                    self.teff.n, self.teff.s,self.teff_note)
        if self.logg is not None:
            s += 'log g : {:5.2f} +/- {:0.2f}    [{}]\n'.format(
                    self.logg.n, self.logg.s, self.logg_note)
        if self.metal is not None:
            s += '[M/H] : {:+5.2f} +/- {:0.2f}    [{}]\n'.format(
                    self.metal.n, self.metal.s, self.metal_note)
        if self.logrho is not None:
            s += 'log rho : {:5.2f} +/- {:0.2f} [solar]\n'.format(
                    self.logrho.n, self.logrho.s)
        if self.h_1 is not None:
            s += 'h_1 : {:5.3f} +/- {:0.3f}\n'.format(
                    self.h_1.n, self.h_1.s)
        if self.h_2 is not None:
            s += 'h_2 : {:5.3f} +/- {:0.3f}\n'.format(
                    self.h_2.n, self.h_2.s)
        return s






