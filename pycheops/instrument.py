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
instrument
==========
 Constants, functions and data related to the CHEOPS instrument.

Functions 
---------

"""

from __future__ import (absolute_import, division, print_function,
                                unicode_literals)

__all__ = [ 'response', 'visibility', 'exposure_time']

from astropy.table import Table
from os.path import join,abspath,dirname
from pickle import load
from numpy import int as np_int 
from numpy import round

_data_path = join(dirname(abspath(__file__)),'data')

with open(join(_data_path,'instrument','exposure_time.p'),'rb') as fp:
    _exposure_time_interpolator = load(fp)

with open(join(_data_path,'instrument','visibility_interpolator.p'),'rb') as fp:
    _visibility_interpolator = load(fp)

def visibility(ra, dec):
    """
    Estimate of target visibility 

    The target visibility estimated with this function is approximate. A more
    reliable estimate of the observing efficiency can be made with the 
    Feasibility Checker tool.

    :param ra: right ascension in degrees (scalar or array)

    :param dec: declination in degrees (scalar or array)

    :returns: target visibility (%)

    """
    return (_visibility_interpolator(ra, dec)*100).astype(np_int)

def response(passband='CHEOPS'):
    """
     Instrument response functions.

     The response functions have been digitized from Fig. 2 of
     https://www.cosmos.esa.int/web/cheops/cheops-performances

     The available passband names are 'CHEOPS', 'MOST', 
     'Kepler', 'CoRoT', 'Gaia', 'B', 'V', 'R', 'I'

     :param passband: instrument/passband names (case sensitive).

     :returns: Instrument response function as an astropy Table object.

    """
    T = Table.read(join(_data_path,'response_functions',
        'response_functions.fits'))
    T.rename_column(passband,'Response')
    return T['Wavelength','Response']

def exposure_time(G):
    """
    Recommended minimum/maximum exposure times

    The function returns the exposure times that are estimated to provide 
    10% and 90% of the detector full well capacity in the brightest image
    pixel of the target. 

     :param G: Gaia G-band magnitude

     :returns: min,max recommended exposure time

    """

    r =  _exposure_time_interpolator(G)
    return round(r[0],2),round(r[1],2)
