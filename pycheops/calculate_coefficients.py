#!/usr/bin/env python

import numpy as np
from scipy.interpolate import pchip_interpolate
from scipy.optimize import minimize
import argparse
import textwrap
from ellc import lc

def q1q2_to_h1h2(q1, q2):
    return 1 - np.sqrt(q1) + q2*np.sqrt(q1), 1 - np.sqrt(q1)

def h1h2_to_ca(h1, h2):
    return 1 - h1 + h2, np.log2((1 - h1 + h2)/h2)

def transit_width(r, k, b, P=1):
    return P*np.arcsin(r*np.sqrt( ((1+k)**2-b**2) / (1-b**2*r**2) ))/np.pi

def func(c, t, r_1, k, incl, grid_size, lc_mugrid):
    h1,h2 = q1q2_to_h1h2(c[0],c[1])
    c2,a2 = h1h2_to_ca(h1,h2)
    ldc_1 = [c2, a2]
    try:
        lc_fit = lc(t, radius_1=r_1, radius_2=r_1*k,
            sbratio=0, incl=incl, ld_1='power-2', ldc_1=ldc_1,
            grid_1=grid_size, grid_2=grid_size)
    except:
        lc_fit = zero_like(t)
    return (lc_fit - lc_mugrid).std()

#---------------

def main():


    parser = argparse.ArgumentParser(
        description='Optimize limb-darkening coefficients',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog = textwrap.dedent('''\

        Optimize limb-darkening coefficients by fitting a transit light curve

        Input files containing centre-to-limb intensity profile must have three
        columns with mu in the _2nd_ column and intensity in the _3rd_column.

        N.B. input data must be in ascending order of mu.

        Currently only implements power-2 law - others to be added.
        
        The output is a single line with the parameters in the following order

            profile c alpha h_1 h_2 q_1 q_2 rms

        rms is the root-mean-square residual of the fit to the transit light
        curve in ppm.

        
        Notes
        ~~~~~
        - If repeated intensity values with mu=0 are present at the start of
           the data file then only the last one in the list is used.
        - If the data file does not contain data at mu=0, I(0)=0 will be used. 
        - If the data file does not contain data at mu=1, I(1)=1 will be used. 

        --
    '''))

    parser.add_argument('profile', nargs='+', 
            help='File with intensity profile, mu in col. 2, I(mu) in col. 3')

    parser.add_argument('-r', '--resample', default=101, type=int,
        help='''Re-sample input data  
         Input data are interpolated onto regular grid with specifed number of
        points. Set this value to 0 to avoid re-sampling if the input data are
        already on a regular grid of mu values. 
        (default: %(default)d)
        ''')

    parser.add_argument('-n', '--n_lc', default=64, type=int,
        help='''Number of points in the simulated light curve.
        (default: %(default)d)
        ''')

    parser.add_argument('-b', '--impact', default=0.0, type=float,
        help='''Impact parameter for simulated light curve
        (default: %(default)f)
        ''')

    parser.add_argument('-k', '--ratio', default=0.1, type=float,
        help='''Planet-star radius ratio for simulated light curve
        (default: %(default)f)
        ''')

    parser.add_argument('-g', '--grid', default='sparse', type=str,
        help='''Density of numerical grid for simulation
        Options are 'very_sparse', 'sparse', 'default', 'fine' and 'very_fine'
        (default: %(default)s)
        ''')

    args= parser.parse_args()

    # Fixed parameters, might change these to input options later, idk
    r_1 = 0.1  # star radius/a

    w = 0.5*transit_width(r_1, args.ratio, args.impact)
    t = np.linspace(0,w,args.n_lc,endpoint=False)
    for profile in args.profile: 
        mu,I_mu=np.loadtxt(profile,unpack=True, usecols=[1,2])
        # Deal with repeated data at mu=0 
        if sum(mu == 0) > 1:
            j = np.arange(len(mu))[mu ==0].max()
            mu = mu[j:]
            I_mu = I_mu[j:]
        elif sum(mu == 0) == 0:
            mu = np.array([0,*mu])
            I_mu = np.array([0,*I_mu])
        if sum(mu == 1) == 0:
            mu = np.array([*mu, 1])
            I_mu = np.array([*I_mu, 1])
        if args.resample > 0:
            mu_1 = np.linspace(0,1,args.resample)
            ldc_1 = pchip_interpolate(mu, I_mu, mu_1)
        else:
            ldc_1 = mu
        incl = 180*np.arccos(r_1*args.impact)/np.pi
        lc_mugrid = lc(t, radius_1=r_1, radius_2=r_1*args.ratio,
            sbratio=0, incl=incl, ld_1='mugrid', ldc_1 = ldc_1,
            grid_1=args.grid, grid_2=args.grid)

        c = np.array([0.3,0.45])   # q1, q2
        smol = np.sqrt(np.finfo(float).eps)
        soln = minimize(func, c, 
                args=(t, r_1, args.ratio, incl, args.grid, lc_mugrid),
                method='L-BFGS-B', bounds=((smol, 1-smol),(smol, 1-smol)))
        q1,q2 = soln.x
        h1,h2 = q1q2_to_h1h2(q1,q2)
        c2,a2 = h1h2_to_ca(h1,h2)
        print(f"{profile} {c2:0.5f} {a2:0.5f} {h1:0.5f} {h2:0.5f} {q1:0.5f} {q2:0.5f} {soln.fun*1e6:5.2f}")






#-------------------------------

if __name__ == "__main__":
    main()
