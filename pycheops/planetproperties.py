# -*- coding: utf-8 -*-
#
#   pycheops - Tools for the analysis of data from the ESA CHEOPS mission
#
#   Copyright (C) 2018 Dr Pierre Maxted, Keele University
#                 2020 Prof Andrew Cameron, University of St Andrews
#                 2020 Dr Thomas Wilson, University of St Andrews
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
PlanetProperties
================
 Object class to obtain/store observed properties of a planet and to infer
 parameters such as surface gravity and density.

"""

from __future__ import (absolute_import, division, print_function,
                                unicode_literals)
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord, match_coordinates_sky
import requests
from pycheops.core import load_config
from pathlib import Path
from time import localtime, mktime
from uncertainties import ufloat, UFloat
from numpy.random import normal
import astropy.units as u
from pycheops import StarProperties
from uncertainties.umath import sqrt as usqrt
from uncertainties.umath import sin as usin
from uncertainties.umath import cos as ucos
from uncertainties.umath import atan2 as uatan2
import os
import warnings
from contextlib import redirect_stderr
from dace_query.cheops import Cheops

class PlanetProperties(object):
    """
    CHEOPS PlanetProperties object
    
    The observed properties T0, P, ecosw, esinw, depth (ppm), width (days) and
    K (km/s) are obtained from one of the following sources (listed here in
    priority order).
    - specified by the user
    - DACE planet table (unless query_dace=False)
    - TEPCat (unless query_tepcat=False)

    """

    def __init__(self, identifier, force_download=False, configFile=None,
            query_dace=True, query_tepcat=True, T0=None, P=None, ecosw=None,
            esinw=None, depth=None, width=None, K=None, verbose=True):

        self.identifier = identifier
        self.T0 = T0
        self.P = P
        self.ecosw = ecosw
        self.esinw = esinw
        self.depth = depth
        self.width = width
        self.K = K
        
        config = load_config(configFile)
        _cache_path = config['DEFAULT']['data_cache_path']
        
        if query_dace: 
            f = {"obj_id_planet_catname":{"contains":[identifier]}}
            planet_data = Cheops.query_catalog('planet', filters=f)
            target = planet_data['obj_id_planet_catname']
            if len(target) == 1:
                if verbose: 
                    print('Target ',target[0],' found in DACE-Planets.')
                T0_val = planet_data['obj_trans_t0_bjd'][0]
                T0_err = planet_data['obj_trans_t0_bjd_err'][0]
                P_val = planet_data['obj_trans_period_days'][0]
                P_err = planet_data['obj_trans_period_days_err'][0]
                ecosw_val = planet_data['obj_trans_ecosw'][0]
                ecosw_err = planet_data['obj_trans_ecosw_err'][0]
                esinw_val = planet_data['obj_trans_esinw'][0]
                esinw_err = planet_data['obj_trans_esinw_err'][0]
                depth_val = planet_data['obj_trans_depth_ppm'][0]
                depth_err = planet_data['obj_trans_depth_ppm_err'][0]
                width_val = planet_data['obj_trans_duration_days'][0]
                width_err = planet_data['obj_trans_duration_days_err'][0]
                # 'obj_rv_k_mps' is in km/s so need to convert to m/s
                # Note use of float() to avoid problems with 'NaN' values
                K_val = float(planet_data['obj_rv_k_mps'][0])*1000
                K_err = float(planet_data['obj_rv_k_mps_err'][0])*1000
                
                # Still need to get errors on these parameters and replace
                # np.nan with None
                
                try:
                    self.T0 = ufloat(float(T0_val),float(T0_err))
                    self.T0_note = "DACE-Planets"
                except:
                    self.T0 = None
                try:
                    self.P = ufloat(float(P_val),float(P_err))
                    self.P_note = "DACE-Planets"
                except:
                    self.P = None
                try:
                    self.ecosw=ufloat(float(ecosw_val),float(ecosw_err))
                    self.ecosw_note = "DACE-Planets"
                except:
                    self.ecosw = None
                try:
                    self.esinw=ufloat(float(esinw_val),float(esinw_err))
                    self.esinw_note = "DACE-Planets"
                except:
                    self.esinw = None
                try:
                    self.depth=ufloat(float(depth_val),float(depth_err))
                    self.depth_note = "DACE-Planets"
                except:
                    self.depth = None
                try:
                    self.width=ufloat(float(width_val),float(width_err))
                    self.width_note = "DACE-Planets"
                except:
                    self.width = None
                try:
                    self.K=ufloat(float(K_val),float(K_err))
                    self.K_note = "DACE-Planets"
                except:
                    self.K = None

                # Tidy up any missing values stored as NaNs in DACE
        
                if np.isnan(self.depth.n): 
                    self.depth = None
                if np.isnan(self.width.n): 
                    self.width = None
                if np.isnan(self.ecosw.n): 
                    self.ecosw = None
                if np.isnan(self.esinw.n): 
                    self.esinw = None
                if np.isnan(self.K.n): 
                    self.K = None
    
            elif len(target) < 1:
                print('No matching planet in DACE-Planets.')
                if verbose:
                    print('List of valid planet_id keys:')
                    l = Cheops.query_catalog('planet')['obj_id_planet_catname']
                    print(l)
            else:
                print(r'Target ',identifier,'not defined uniquely: ',target)
            
        if query_tepcat:    
            TEPCatObsPath = Path(_cache_path,'observables.csv')
            download_tepcat = False
            if force_download:
                download_tepcat = True
            elif TEPCatObsPath.is_file():
                file_age = mktime(localtime())-os.path.getmtime(TEPCatObsPath)
                if file_age > int(config['TEPCatObs']['update_interval']):
                    download_tepcat = True
                else:
                    download_tepcat = False
            else:
                download_tepcat = True
            if download_tepcat:
                try:
                    url = config['TEPCatObs']['download_url']
                except:
                    raise KeyError("TEPCatObs table not found in config file."
                            " Run core.setup_config")
                try:
                    req=requests.post(url)
                except:
                    warnings.warn("Failed to update TEPCatObs from server")
                else:
                    with open(TEPCatObsPath, 'wb') as file:
                        file.write(req.content)
                    if verbose:
                        print('TEPCat data downloaded from \n {}'.format(url))
            # Awkward table to deal with because of repeated column names
            T = Table.read(TEPCatObsPath,format='ascii.no_header')
            hdr = list(T[0])
            targets=np.array(T[T.colnames[hdr.index('System')]][1:],
                    dtype=np.str_)
            RAh=np.array(T[T.colnames[hdr.index('RAh')]][1:],
                    dtype=np.str_)
            RAm=np.array(T[T.colnames[hdr.index('RAm')]][1:],
                    dtype=np.str_)
            RAs=np.array(T[T.colnames[hdr.index('RAs')]][1:],
                    dtype=np.str_)
            Decd=np.array(T[T.colnames[hdr.index('Decd')]][1:],
                    dtype=np.str_)
            Decm=np.array(T[T.colnames[hdr.index('Decm')]][1:],
                    dtype=np.str_)
            Decs=np.array(T[T.colnames[hdr.index('Decs')]][1:],
                    dtype=np.str_)
            T0_vals=np.array(T[T.colnames[hdr.index('T0(HJDorBJD)')]][1:],
                    dtype=float)
            T0_errs=np.array(T[T.colnames[hdr.index('T0err')]][1:],
                    dtype=float)
            periods=np.array(T[T.colnames[hdr.index('Period(day)')]][1:],
                    dtype=float)
            perrors=np.array(T[T.colnames[hdr.index('Perioderr')]][1:],
                    dtype=float)
            lengths=np.array(T[T.colnames[hdr.index('length')]][1:],
                    dtype=float)
            depths =np.array(T[T.colnames[hdr.index('depth')]][1:],
                    dtype=float)

            ok = [t.startswith(identifier.replace(' ','_')) for t in targets]
            if sum(ok) > 1:
                print('Matching planet names: ', *targets[ok])
                raise ValueError('More than one planet matches identifier.')
            elif sum(ok) == 1:
                T0_val=T0_vals[ok][0]
                T0_err=T0_errs[ok][0]
                period=periods[ok][0]
                perror=perrors[ok][0]
                length=lengths[ok][0]
                depth_val=depths[ok][0]*10000
            else:
                try:
                    tar_coords = SkyCoord.from_name(identifier) 
                    all_coords = []
                    for index, i in enumerate(RAh):
                        all_coords.append(
                                RAh[index]+":"+RAm[index]+":"+RAs[index]+" "+
                                Decd[index]+":"+Decm[index]+":"+Decs[index])
                    TEPCat_coords = SkyCoord(all_coords, frame="icrs", 
                            unit=(u.hourangle, u.deg))
                    ok = tar_coords.separation(TEPCat_coords) < 5*u.arcsec
                    if sum(ok) > 1:
                        print('Matching planets: ', *targets[ok])
                        raise ValueError(
                                'More than one planet matches coordinates.')
                    elif sum(ok)==1:
                        T0_val=T0_vals[ok][0]
                        T0_err=T0_errs[ok][0]
                        period=periods[ok][0]
                        perror=perrors[ok][0]
                        length=lengths[ok][0]
                        depth_val=depths[ok][0]*10000
            
                    else:
                        print('No matching planet in TEPCat.')   
                except: 
                    print('No coordinate match for planet in TEPCat.')
                
            if sum(ok)==1:        
                if self.T0 == None: 
                    try:
                        self.T0 = ufloat(float(T0_val),float(T0_err))
                        self.T0_note = "TEPCat"
                    except:
                        self.T0 = None
                if self.P == None:
                    try:
                        self.P = ufloat(float(period),float(perror))
                        self.P_note = "TEPCat"
                    except:
                        self.P = None
                if self.depth == None:
                    try:
                        self.depth=ufloat(float(depth_val),1e2)
                        self.depth_note = "TEPCat"
                    except:
                        self.depth = None
                if self.width == None:
                    try:
                        self.width=ufloat(float(length),0.01)
                        self.width_note = "TEPCat"
                    except:
                        self.width = None
                    
        # User defined values
        if T0:
            if  isinstance(T0, UFloat):
                self.T0 = T0
                self.T0_note = "User"
            else:
                raise ValueError("T0 keyword is not ufloat")
        if P:
            if  isinstance(P, UFloat):
                self.P = P
                self.P_note = "User"
            else:
                raise ValueError("P keyword is not ufloat")
        if ecosw:
            if  isinstance(ecosw, UFloat):
                self.ecosw = ecosw
                self.ecosw_note = "User"
            else:
                raise ValueError("ecosw keyword is not ufloat")
        if esinw:
            if  isinstance(esinw, UFloat):
                self.esinw = esinw
                self.esinw_note = "User"
            else:
                raise ValueError("esinw keyword is not ufloat")
        if depth:
            if  isinstance(depth, UFloat):
                self.depth = depth
                self.depth_note = "User"
            else:
                raise ValueError("depth keyword is not ufloat")
        if width:
            if  isinstance(width, UFloat):
                self.width = width
                self.width_note = "User"
            else:
                raise ValueError("width keyword is not ufloat")
        if K:
            if  isinstance(K, UFloat):
                self.K = K
                self.K_note = "User"
            else:
                raise ValueError("K keyword is not ufloat")
            
            
        # Eccentricity and omega from ecosw and esinw
        
        self.ecc = None
        self.omega = None
        self.f_s = None
        self.f_c = None
        if self.ecosw and self.esinw:
            ecosw = self.ecosw
            esinw = self.esinw
            ecc = usqrt(ecosw*ecosw+esinw*esinw)
            if ecc.n != 0:
                omega = uatan2(esinw,ecosw)
                f_s = usqrt(ecc)*usin(omega)
                f_c = usqrt(ecc)*ucos(omega)
            elif ecc.s != 0:
                # Work-around to avoid NaNs for e=0 with finite error. 
                eps = .0001
                ecc = usqrt((ecosw+eps)**2+(esinw+eps)**2)-eps
                omega = None
                f_s = ufloat(0,np.sqrt(esinw.s))
                f_c = ufloat(0,np.sqrt(ecosw.s))
            else:
                ecc = None
                omega = None
                f_s = None
                f_c = None
            self.ecc = ecc
            self.ecc_note = "Derived"
            self.omega = omega
            self.omega_note = "Derived"
            
            self.f_s = f_s
            self.f_s_note = "Derived"
            self.f_c = f_c
            self.f_c_note = "Derived"
        
        ##############################################################################################################
        ### Calculate system properties (b, aR (or just a), and planetary mass, radius, density, surface gravity, and 
        ### equilibrium temperature) when we can import stellar radius/mass from StarProperties
        ##############################################################################################################
        
        
        # starproperties = StarProperties(identifier)
        
        # aR = funcs.a_rsun(P, starproperties.M)*astrocon.R_sun.value/starproperties.R ### in stellar radii
        # b_pl = np.sqrt((1+np.sqrt(D*1.e-6))**2 - (aR*starproperties.R*astrocon.R_sun.value*W*24*np.pi/P)**2)
        # g_pl = funcs.g_2(np.sqrt(D*1.e-6)/aR,P,K/1000,ecc=ecc) ### in m/s^2
        # m_pl = funcs.m_comp(funcs.f_m(P,K/1000,ecc=ecc), starproperties.M, sini=usqrt(1-(b_pl/aR)**2))/astrocon.M_sun.value ### in kg
        # r_pl = np.sqrt(D*1.e-6)*starproperties.R*astrocon.R_sun.value ### in m
        # p_pl = m_pl/((4*np.pi/3)*r_pl**3) ### in kg/m^3
        # T_pl = starproperties.T*np.sqrt(1/(2*aR))*(1-A_b)**(1/4)
        
            
    def __repr__(self):
        s =  'Identifier : {}\n'.format(self.identifier)
        if self.T0:
            s += 'T0 : {:12.4f} +/- {:0.4f} BJD       [{}]\n'.format(
                    self.T0.n, self.T0.s,self.T0_note)
        if self.P:
            s += 'P : {:13.7f} +/- {:0.7f} days   [{}]\n'.format(
                    self.P.n, self.P.s, self.P_note)
        if self.depth:
            s += 'depth : {:10.4f} +/- {:0.4f} ppm         [{}]\n'.format(
                    self.depth.n, self.depth.s, self.depth_note)
        if self.width:
            s += 'width : {:7.4f} +/- {:0.4f} days            [{}]\n'.format(
                    self.width.n, self.width.s, self.width_note)
        if self.K:
            s += 'K : {:7.4f} +/- {:0.4f} m/s      [{}]\n'.format(
                    self.K.n, self.K.s, self.K_note)
        if self.ecosw:
            s += 'ecosw : {:+7.4f} +/- {:0.4f}             [{}]\n'.format(
                    self.ecosw.n, self.ecosw.s, self.ecosw_note)
        if self.esinw:
            s += 'esinw : {:+7.4f} +/- {:0.4f}             [{}]\n'.format(
                    self.esinw.n, self.esinw.s, self.esinw_note)
        if self.ecc:
            s += 'ecc : {:6.4f} +/- {:0.4f}                [{}]\n'.format(
                    self.ecc.n, self.ecc.s, self.ecc_note)
        if self.omega:
            s += 'omega : {:+8.5f} +/- {:0.5f} radian    [{}]\n'.format(
                    self.omega.n, self.omega.s, self.omega_note)
        if self.f_c:
            s += 'f_c : {:+7.4f} +/- {:0.4f}               [{}]\n'.format(
                    self.f_c.n, self.f_c.s, self.f_c_note)
        if self.f_s:
            s += 'f_s : {:+7.4f} +/- {:0.4f}               [{}]\n'.format(
                    self.f_s.n, self.f_s.s, self.f_s_note)
        return s
