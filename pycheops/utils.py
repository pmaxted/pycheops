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
utils
======
 Utility functions 

Functions 
---------

"""

from __future__ import (absolute_import, division, print_function,
                                unicode_literals)
import numpy as np

__all__ = [ 'eformat']

# Print value and error in final digit with specified number of sig. fig. in
# the error value. 
def eformat(val,err,sf=2):
    ndp = np.int(np.floor(np.log10(err))) + sf - 1
    if (ndp < 0):
      ndp = -ndp
      edigit = np.int(np.round((err*10**ndp)))
      if (edigit == 10) :
        ndp = ndp - 1
        edigit = 1
      return  '{:.{ndp}f}({})'.format(round(val,ndp),edigit,ndp=ndp)
    else:
      return ''

