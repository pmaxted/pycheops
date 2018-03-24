
from math import pi, sqrt

from unittest import TestCase

from cheops.constants import *
from cheops.funcs import *

class TestConstants(TestCase):

    def test_rhostar(self):
        m_star = 1.234 * M_SunN
        r_star = 0.987 * R_SunN
        rho_star = m_star/(4*pi*r_star**3/3.)
        m_planet = 5.678 * M_JupN
        P = 9.876
        M = (m_star+m_planet)/M_SunN
        a = a_rsun(P, M)
        r_1 = r_star/R_SunN/a
        q = m_planet/m_star
        rho_Sun = M_SunN/V_SunN
        assert abs(rho_star/rho_Sun - rhostar(r_1, P, q=q)) < 1e-9


    def test_m_comp(self):
        m_1 = 1.23456
        m_2 = 0.23456 
        P =   3.45678
        sini = 0.8765 
        e = 0.45678
 
        K_1,K_2 = K_kms(m_1, m_2, P, sini, e)
        fm1 = f_m(P, K_1, e)
        fm2 = f_m(P, K_2, e)
        t_1 = abs(m_2 - m_comp(fm1,m_1,sini))
        t_2 = abs(m_1 - m_comp(fm2,m_2,sini))
        assert  t_1+t_2 < 1e-9


