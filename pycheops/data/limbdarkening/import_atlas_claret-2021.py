import numpy as np

def c_a_ok(c,a):
    h1,h2 = 1 - c*(1-0.5**a), c*0.5**a
    q1, q2 = (1 - h2)**2, (h1 - h2)/(1-h2)
    return ((abs(q1-0.5) < 0.5) & (abs(q2-0.5) < 0.5)).nonzero()[0]

f = open("atlas-claret-2021.dat", "w")

# CHEOPS
url = 'https://cdsarc.cds.unistra.fr/ftp/J/other/RNAAS/5.13/table11.dat'
cols = [0,1,2,4,5]
g,t,m,c,a = np.loadtxt(url, unpack=True, usecols=cols)
for i in c_a_ok(c,a):
    f.write(f'CH {t[i]:5.0f} {g[i]:4.1f} {m[i]:4.1f} {c[i]:0.4f} {a[i]:0.4f}\n')

# Gaia, Kepler and TESS
fl = 'arXiv2206.11098/Table1'
cols = [0,1,2,5,7,8]
g,t,m,Ga,Ke,TE = np.loadtxt(fl,usecols=cols,unpack=True,skiprows=3)
g=g[::3]
t=t[::3]
m=m[::3]
for tag,dat in zip(['Ga','Ke','TE'],[Ga,Ke,TE]):
    c = dat[::3]
    a = dat[1::3]
    for i in c_a_ok(c,a):
        f.write(f'{tag} {t[i]:5.0f} {g[i]:4.1f} {m[i]:4.1f} {c[i]:0.4f} {a[i]:0.4f}\n')

# ugriz
fl = 'arXiv2206.11098/Table2'
cols = [0,1,2,4,5,6,7,8]
g,t,m,u_,g_,r_,i_,z_ = np.loadtxt(fl,usecols=cols,unpack=True,skiprows=3)
g=g[::3]
t=t[::3]
m=m[::3]
for tag,dat in zip(["u_","g_","r_","i_","z_"],[u_,g_,r_,i_,z_]):
    c = dat[::3]
    a = dat[1::3]
    for i in c_a_ok(c,a):
        f.write(f'{tag} {t[i]:5.0f} {g[i]:4.1f} {m[i]:4.1f} {c[i]:0.4f} {a[i]:0.4f}\n')

# UBVRI
fl = 'arXiv2206.11098/Table3'
cols = [0,1,2,8,9,10,11,12]
g,t,m,U,B,V,R,I= np.loadtxt(fl,usecols=cols,unpack=True,skiprows=3)
g=g[::3]
t=t[::3]
m=m[::3]
for tag,dat in zip(["U","B","V","R","I"],[U,B,V,R,I]):
    c = dat[::3]
    a = dat[1::3]
    for i in c_a_ok(c,a):
        f.write(f'{tag} {t[i]:5.0f} {g[i]:4.1f} {m[i]:4.1f} {c[i]:0.4f} {a[i]:0.4f}\n')


f.close()
