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

__all__ = [ 'parprint']
def parprint(x,n, w=8, sf=2, wn=None, short=False):
    """
    Print the value and error of a parameter based on a sample

    The number of decimal places in the value and error are set such that the
    error has the specified number of significant figures.

    The parameter value is set to the sample median and the error is based on
    the 15.87% and 84.13% percentiles of the sample.

    :param x:  input sample for probability distribution of the parameter
    :param n:  parameter name
    :param w:  field width for values
    :param wn:  field width for name
    :param sf: number of sig. fig. in the error

    :returns: formatted string

    """
    if wn is None:
        wn = len(n)+1
    std_l, val, std_u = np.percentile(x, [15.87, 50, 84.13])
    err = 0.5*(std_u-std_l)
    e_hi = std_u - val
    e_lo = val - std_l
    ndp = sf - np.int(np.floor(np.log10(err))) - 1
    if ndp < 0:
        ndp = -ndp
        b = 10**ndp
        val = round(val/b)*b
        err = round(err/b)*b
        e_lo = round(e_lo/b)*b
        e_hi = round(e_hi/b)*b
    else:
        val = round(val,ndp)
        err = round(err,ndp)
        e_lo = round(e_lo,ndp)
        e_hi = round(e_hi,ndp)
    if short:
        b = 10**ndp
        err = round(err,ndp)*b
        e_lo = round(e_lo,ndp)*b
        e_hi = round(e_hi,ndp)*b
        f='{:{wn}s} = {:{w}.{ndp}f} ({:{sf}.0f}) (-{:{sf}.0f},+{:{sf}.0f})'
        s = f.format(n, val,err,e_lo,e_hi,ndp=ndp,w=w,wn=wn,sf=sf)
    else:
        f='{:{wn}s} = {:{w}.{ndp}f} +/- {:{w}.{ndp}f} (-{:.{ndp}f},+{:.{ndp}f})'
        s = f.format(n, val,err,e_lo,e_hi,ndp=ndp,w=w,wn=wn)
    return s

