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
from .ld import stagger_power2_interpolator, atlas_h1h2_interpolator
from .ld import phoenix_h1h2_interpolator
from numpy.random import normal


class StarProperties(object):
    """
    CHEOPS StarProperties object

    The observed properties T_eff, log_g and [Fe/H] are obtained from
    DACE or SWEET-Cat, or can be specified by the user. 

    Set match_arcsec=None to skip extraction of parameters from SWEET-Cat.

    By default properties are obtained from SWEET-Cat.
    
    Set dace=True to obtain parameters from the stellar properties table at
    DACE.
    
    User-defined properties are specified either as a ufloat or as a 2-tuple
    (value, error), e.g., teff=(5000,100).
    
    User-defined properties over-write values obtained from SWEET-Cat or DACE.

    The stellar density is estimated using an linear relation between log(rho)
    and log(g) derived using the method of Moya et al. (2018ApJS..237...21M)

    Limb darkening parameters in the CHEOPS band are interpolated from Table 2
    of Maxted (2018A&A...616A..39M). The error on these parameters is
    propogated from the errors in Teff, log_g and [Fe/H] plus an additional
    error of 0.01 for h_1 and 0.05 for h_2, as recommended in Maxted (2018).
    If [Fe/H] for the star is not specified, the value 0.0 +/- 0.3 is assumed.

    If the stellar parameters are outside the range covered by Table 2 of
    Maxted (2018), then the results from ATLAS model from Table 10 of Claret
    (2019RNAAS...3...17C) are used instead. For stars cooler than 3500K the
    PHOENIX models for solar metalicity  from Table 5 of Claret (2019) are
    used. The parameters h_1 and h_2 are both given nominal errors of 0.1 for
    both ATLAS model, and 0.15 for PHOENIX models.

    """

    def __init__(self, identifier, force_download=False, dace=False, 
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

        if force_download:
            download_sweetcat = True
        elif dace:
            download_sweetcat = False
        elif sweetCatPath.is_file():
            file_age = mktime(localtime())-getmtime(sweetCatPath)
            if file_age > int(config['SWEET-Cat']['update_interval']):
                download_sweetcat = True
            else:
                download_sweetcat = False
        else:
            download_sweetcat = True

        if download_sweetcat:
            url = config['SWEET-Cat']['download_url']
            req=requests.post(url)
            with open(sweetCatPath, 'wb') as file:
                file.write(req.content)
            if verbose:
                print('SWEET-Cat data downloaded from \n {}'.format(url))

        if dace:
            from dace.cheops import Cheops
            db = Cheops.query_catalog("stellar")
            cat_c = SkyCoord(db['obj_pos_ra_deg'], db['obj_pos_dec_deg'],
                    unit='degree,degree')
            idx, sep, _ = coords.match_to_catalog_sky(cat_c)
            if sep.arcsec[0] > match_arcsec:
                raise ValueError(
                        'No matching star in DACE stellar properties table')
            self.teff = ufloat(db['obj_phys_teff_k'][idx],99) 
            self.teff_note = "DACE"
            self.logg = ufloat(db['obj_phys_logg'][idx],0.09) 
            self.logg_note = "DACE"
            self.metal = ufloat(db['obj_phys_feh'][idx],0.09) 
            self.metal_note = "DACE"
            self.gaiadr2 = db['obj_id_gaiadr2'][idx]

        else:
            names = ['star', 'hd', 'ra', 'dec', 'vmag', 'e_vmag', 'par',
                    'e_par', 'parsource', 'teff', 'e_teff', 'logg', 'e_logg',
                    'logglc', 'e_logglc', 'vt', 'e_vt', 'metal', 'e_metal',
                    'mass', 'e_mass', 'author', 'source', 'update', 'comment']
            sweetCat = Table.read(sweetCatPath,format='ascii.no_header',
                    delimiter="\t", fast_reader=False, names=names,
                    encoding='utf-8')

            if match_arcsec is None:
                entry = None
            else:
                cat_c = SkyCoord(sweetCat['ra'],sweetCat['dec'],
                            unit='hour,degree')
                idx, sep, _ = coords.match_to_catalog_sky(cat_c)
                if sep.arcsec[0] > match_arcsec:
                    raise ValueError('No matching star in SWEET-Cat')
                entry = sweetCat[idx]

            try:
                self.teff = ufloat(float(entry['teff']),float(entry['e_teff']))
                self.teff_note = "SWEET-Cat"
            except:
                self.teff = None
            try:
                self.logg = ufloat(float(entry['logg']),float(entry['e_logg']))
                self.logg_note = "SWEET-Cat"
            except:
                self.logg = None
            try:
                self.metal=ufloat(float(entry['metal']),float(entry['e_metal']))
                self.metal_note = "SWEET-Cat"
            except:
                self.metal = None

        # User defined values
        if teff:
           self.teff = teff if isinstance(teff, UFloat) else ufloat(*teff)
           self.teff_note = "User"
        if logg:
           self.logg = logg if isinstance(logg, UFloat) else ufloat(*logg)
           self.logg_note = "User"
        if metal:
           self.metal = metal if isinstance(metal, UFloat) else ufloat(*metal)
           self.metal_note = "User"

        # log rho from log g using method of Moya et al.
        # (2018ApJS..237...21M). Accuracy is 4.4%
        self.logrho = None
        if self.logg:
            if (self.logg.n > 3.697) and (self.logg.n < 4.65):
                logrho = -7.352 + 1.6580*self.logg
                self.logrho = ufloat(logrho.n, np.hypot(logrho.s, 0.044))

        self.h_1 = None
        self.h_2 = None
        self.ld_ref = None
        if self.teff and self.logg:
            metal = self.metal if self.metal else ufloat(0,0.3)
            power2 = stagger_power2_interpolator()
            _,_,h_1,h_2 = power2(self.teff.n,self.logg.n,metal.n)
            if not np.isnan(h_1):
                self.ld_ref = 'Stagger'
                Xteff = normal(self.teff.n, self.teff.s, 256)
                Xlogg = normal(self.logg.n, self.logg.s, 256)
                Xmetal = normal(metal.n, metal.s, 256)
                X = power2(Xteff,Xlogg,Xmetal)
                # Additional error derived in Maxted, 2019
                e_h_1 = np.hypot(0.01,np.sqrt(np.nanmean((X[:,2]-h_1)**2)))
                e_h_2 = np.hypot(0.05,np.sqrt(np.nanmean((X[:,3]-h_2)**2)))
                self.h_1 = ufloat(round(h_1,3),round(e_h_1,3))
                self.h_2 = ufloat(round(h_2,3),round(e_h_2,3))
            if self.ld_ref is None:
                atlas = atlas_h1h2_interpolator()
                h_1,h_2 = atlas(self.teff.n,self.logg.n,metal.n)
                if not np.isnan(h_1): 
                    self.h_1 = ufloat(round(h_1,3),0.1)
                    self.h_2 = ufloat(round(h_2,3),0.1)
                    self.ld_ref = 'ATLAS'
            if self.ld_ref is None:
                phoenix = phoenix_h1h2_interpolator()
                h_1,h_2 = phoenix(self.teff.n,self.logg.n)
                if not np.isnan(h_1): 
                    self.h_1 = ufloat(round(h_1,3),0.15)
                    self.h_2 = ufloat(round(h_2,3),0.15)
                    self.ld_ref = 'PHOENIX-COND'

    def __repr__(self):
        s =  'Identifier : {}\n'.format(self.identifier)
        s +=  'Coordinates: {} {}\n'.format(self.ra, self.dec)
        if self.teff:
            s += 'T_eff : {:5.0f} +/- {:3.0f} K   [{}]\n'.format(
                    self.teff.n, self.teff.s,self.teff_note)
        if self.logg:
            s += 'log g : {:5.2f} +/- {:0.2f}    [{}]\n'.format(
                    self.logg.n, self.logg.s, self.logg_note)
        if self.metal:
            s += '[M/H] : {:+5.2f} +/- {:0.2f}    [{}]\n'.format(
                    self.metal.n, self.metal.s, self.metal_note)
        if self.logrho:
            s += 'log rho : {:5.2f} +/- {:0.2f}  (solar units)\n'.format(
                    self.logrho.n, self.logrho.s)
        if self.ld_ref:
            s += 'h_1 : {:5.3f} +/- {:0.3f}     [{}]\n'.format(
                    self.h_1.n, self.h_1.s,self.ld_ref)
            s += 'h_2 : {:5.3f} +/- {:0.3f}     [{}]\n'.format(
                    self.h_2.n, self.h_2.s,self.ld_ref)
        return s
