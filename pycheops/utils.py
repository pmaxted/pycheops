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

__all__ = [ 'parprint', 'lcbin']

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

#----------

def lcbin(time, flux, binwidth=0.06859, nmin=4, time0=None,
        robust=False, tmid=False):
    """
    Calculate average flux and error in time bins of equal width.

    The default bin width is equivalent to one CHEOPS orbit in units of days.

    To avoid binning data on either side of the gaps in the light curve due to
    the CHEOPS orbit, the algorithm searches for the largest gap in the data
    shorter than binwidth and places the bin edges so that they fall at the
    centre of this gap. This behaviour can be avoided by setting a value for
    the parameter time0.

    The time values for the output bins can be either the average time value
    of the input points or, if tmid is True, the centre of the time bin.

    If robust is True, the output bin values are the median of the flux values
    of the bin and the standard error is estimated from their mean absolute
    deviation. Otherwise, the mean and standard deviation are used.

    The output values are as follows.
    * t_bin - average time of binned data points or centre of time bin.
    * f_bin - mean or median of the input flux values.
    * e_bin - standard error of flux points in the bin.
    * n_bin - number of flux points in the bin.

    :param time: time
    :param flux: flux (or other quantity to be time-binned)
    :param binwidth:  bin width in the same units as time
    :param nmin: minimum number of points for output bins
    :param time0: time value at the lower edge of one bin
    :param robust: use median and robust estimate of standard deviation
    :param tmid: return centre of time bins instead of mean time value

    :returns: t_bin, f_bin, e_bin, n_bin

    """
    if time0 is None:
        tgap = (time[1:]+time[:-1])/2
        gap = time[1:]-time[:-1]
        j = gap < binwidth
        gap = gap[j]
        tgap = tgap[j]
        time0 = tgap[np.argmax(gap)]
        time0 = time0 - binwidth*np.ceil((time0-min(time))/binwidth)

    n = np.int(1+np.ceil(np.ptp(time)/binwidth))
    r = (time0,time0+n*binwidth)
    n_in_bin,bin_edges = np.histogram(time,bins=n,range=r)
    bin_indices = np.digitize(time,bin_edges)

    t_bin = np.zeros(n)
    f_bin = np.zeros(n)
    e_bin = np.zeros(n)
    n_bin = np.zeros(n, dtype=np.int)

    for i,n in enumerate(n_in_bin):
        if n >= nmin:
            j = bin_indices == i+1
            n_bin[i] = n
            if tmid:
                t_bin[i] = (bin_edges[i]+bin_edges[i+1])/2
            else:
                t_bin[i] = np.mean(time[j])
            if robust:
                f_bin[i] = np.median(flux[j])
                e_bin[i] = 1.25*np.mean(abs(flux[j] - f_bin[i]))/np.sqrt(n)
            else:
                f_bin[i] = np.mean(flux[j])
                e_bin[i] = np.std(flux[j])/np.sqrt(n-1)

    j = (n_bin >= nmin)
    return t_bin[j], f_bin[j], e_bin[j], n_bin[j]
