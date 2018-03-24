"""
instrument
==========
 Constants, functions and data related to the CHEOPS instrument.

Functions 
---------

"""

from __future__ import (absolute_import, division, print_function,
                                unicode_literals)

__all__ = [ 'response',]

from astropy.table import Table
from os.path import join,abspath,dirname


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
    dir_path = dirname(abspath(__file__))
    T = Table.read(join(dir_path,'data','response_functions',
        'response_functions.fits'))
    T.rename_column(passband,'Response')
    return T['Wavelength','Response']
