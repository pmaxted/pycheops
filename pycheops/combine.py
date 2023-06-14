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
combine
=======
 Calculate weighted mean and its standard error allowing for external noise

Functions 
---------
 main() - combine


"""

from __future__ import (absolute_import, division, print_function,
                                unicode_literals)
import argparse
from astropy.table import Table
import numpy as np
from . import __version__
from .utils import uprint as up
from uncertainties import ufloat as uf
import emcee
import textwrap
from scipy.special import gammainc
from numba import jit

def combine(y, yerr, walkers=64, steps=256, discard=128):

    @jit(nopython=True)
    def log_prob(p, y, yvar, mulo, muhi, lnsig_lo, lnsig_hi):
        mu = p[0]
        lnsig = p[1]
        if (mu < mulo) or (mu > muhi): return -np.inf
        if (lnsig < lnsig_lo) or (lnsig > lnsig_hi): return -np.inf
        sigma2 = yvar + np.exp(2*lnsig)
        return -0.5 * np.sum((y - mu)**2/sigma2 + np.log(sigma2))

    y = np.array(y)
    yerr = np.array(yerr)

    mulo = y.min() - yerr.max()
    muhi = y.max() + yerr.max()
    mu_i = np.random.normal(y.mean(),y.std()+yerr.min(), walkers)
    mu_i = np.clip(mu_i, mulo, muhi)
    lnsig_0 = np.log(y.std()+yerr.min())
    lnsig_lo = lnsig_0 - 15
    lnsig_hi = lnsig_0 + 5
    lnsig_i = np.random.normal(lnsig_0-5,1,walkers)
    lnsig_i = np.clip(lnsig_i, lnsig_lo, lnsig_hi)
    pos = np.array( [mu_i, lnsig_i]).T
    nwalkers, ndim = pos.shape

    yvar = yerr**2
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob,
            args=(y, yvar, mulo, muhi, lnsig_lo, lnsig_hi))
    sampler.run_mcmc(pos, steps)
    chain = sampler.get_chain(flat=True, discard=discard)

    mu = chain[:,0].mean()
    mu_err = chain[:,0].std()
    sigext = np.exp(chain[:,1]).mean()
    sigext_err = np.exp(chain[:,1]).std()
    return mu, mu_err, sigext, sigext_err,  sampler

def main():

    # Set up command line switches
    parser = argparse.ArgumentParser(
        description='Weighted mean and error allowing for external noise',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog = textwrap.dedent(f'''\

        This is version {__version__}

        Reads a table of values with standard error estimates and calculates
        the weighted mean and error, allowing for the possibility that there
        is an additional source of uncertainty.

        The input table can be any format suitable for reading with the
        command astropy.table.Table.read(), e.g., CSV.
        
        By default, the calculation is done using columns 1 and 2 in the
        table. Use the flag --val_col and --err_col to specify alternative
        column names or numbers.

        If --val_col is used and --err-col is not specified, the program will ..
         - use the column val_col+1 if val_col is an integer
         - look for a column named "e_"+val_col
         - look for a column named val_col+"_err"
         - give up...

        '''))

    parser.add_argument('table', nargs='?',
        help='Table of values and errors'
    )

    parser.add_argument('-f', '--format',
        help='Table format - passed to Table.read()'
    )

    parser.add_argument('-v', '--val_col', 
        default=1, 
        help='''Column with values
        (default: %(default)d)
        '''
    )

    parser.add_argument('-e', '--err_col', 
        default=None, 
        help='Column with errors'
    )

    parser.add_argument('-1', '--one_line',  
        action='store_const',
        dest='one_line',
        const=True,
        default=False,
        help='Output results on one line'
    )

    parser.add_argument('-p', '--precise', 
        action='store_const',
        dest='precise',
        const=True,
        default=False,
        help='More precise (but slower) calculation'
    )

    args = parser.parse_args()

    if args.table is None:
        parser.print_usage()
        exit(1)

    table = Table.read(args.table, format=args.format)

    try:
        val_col = int(args.val_col) - 1
        name = 'y'
    except ValueError:
        val_col = str(args.val_col)
        name = args.val_col
    y = table[val_col][:]

    if args.err_col is None:
        if isinstance(val_col, int):
            yerr = table[val_col+1][:]
        else:
            try:
                err_col = 'e_'+str(args.val_col)
                yerr = table[err_col][:]
            except KeyError:
                err_col = str(args.val_col)+'_err'
                yerr = table[err_col][:]
    else:
        yerr = table[args.err_col][:]


    if args.precise:
        nw, ns, nb, sf = 128, 1500, 500, 2
    else:
        nw, ns, nb, sf = 64, 150, 50, 1

    mu, e_mu, sig, e_sig, sampler = combine(y, yerr,
            walkers=nw, steps=ns, discard=nb)

    if not args.one_line:
        n = len(y)
        print (f'\nRead {n} values from {args.table}')
        print (up(uf(y.max(), yerr[np.argmax(y)]),
            f'Maximum value of {name}', sf=sf))
        print (up(uf(y.min(), yerr[np.argmin(y)]),
            f'Minimum value of {name}', sf=sf))

        wt = 1/yerr**2
        wsum = wt.sum()
        wmean = (y*wt).sum()/wsum
        chisq = ((y-wmean)**2*wt).sum()
        e_int = np.sqrt(1/wsum)
        e_ext = np.sqrt(chisq/(n-1)/wsum)
        print(f' Weighted mean = {wmean:0.4f}')
        print (up(uf(wmean, e_int),f'{name}', sf=sf), '(Internal error)')
        print (up(uf(wmean, e_ext),f'{name}', sf=sf), '(External error)')
        print(f' Chi-square = {chisq:0.2f}')

        p = 1-gammainc(0.5*(n-1),0.5*chisq)
        print(f' P(chi-sq. > observed chi-sq. if mean is constant) = {p:0.2}')
        print ("--")
    
    print(up(uf(mu,e_mu), name, sf=sf) + '; ' +
            up(uf(sig, e_sig),'sigma_ext', sf=sf))
    


