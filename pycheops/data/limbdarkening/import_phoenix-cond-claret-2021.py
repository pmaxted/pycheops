import numpy as np
from scipy.optimize import root_scalar

_amin = 0.001
_amax = 1000

def _h1ph2p_to_ca(h1p, h2p):
    def _f(a, h1p, h2p):
        return ((2/3)**a - (1/3)**a)*(1-h1p)/(1-(2/3)**a) - h2p
    try:
        r = root_scalar(_f, 
                    bracket=(_amin, _amax), 
                    x0=0.8, 
                    method='bisect', 
                    args=(h1p, h2p))
    except ValueError:
        return np.nan, np.nan
    return (1-h1p)/(1-(2/3)**r.root), r.root


url = 'https://cdsarc.cds.unistra.fr/ftp/J/other/RNAAS/5.13/table5.dat'
cols = [0,1,3,4,5]
logg,teff,c,a,mucri = np.genfromtxt(url, unpack=True, usecols=cols)
h1 = 1-c*(1-np.sqrt(1-0.75*(1-mucri**2))**a)
a1 = -np.log2(h1)
raise
h2p = h1p - (1-c*(1-np.sqrt(1-0.99*(1-mucri**2))**a))
h2pp_max = 0.5**a1 - 0.1**a1
h2pp = np.minimum(h2pp, h2pp_max)

f = open("phoenix-cond-claret-2021.dat", "w")
for i,(h1_, h2pp_, t, g) in enumerate(zip(h1, h2pp, teff, logg)):
    c_,a_ = _h1h2pp_to_ca(h1_, h2pp_)
    f.write(f'{t:5.0f} {g:4.1f} {c_:0.8f} {a_:0.8f}\n')
f.close()

