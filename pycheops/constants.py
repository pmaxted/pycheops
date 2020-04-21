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
constants
=========
Nominal values of solar and planetary constants in SI units from IAU
Resolution B3 [1]_ plus related constants

Masses in SI units are derived using the 2014 CODATA value for the 
Newtonian constant, :math:`G=6.67408\\times 10^{-11}\,m^3\,kg^{-1}\,s^{-2}`.

The following conversion constants are defined.

Solar conversion constants 
--------------------------
* R_SunN     - solar radius
* S_SunN     - total solar irradiance
* L_SunN     - luminosity
* Teff_SunN  - solar effective temperature
* GM_SunN    - solar mass parameter
* M_SunN     - solar mass derived from GM_SunN and G_2014
* V_SunN     - solar volume = (4.pi.R_SunN**3/3)

Planetary conversion constants
------------------------------
* R_eEarthN  - equatorial radius of the Earth
* R_pEarthN  - polar radius of the Earth
* R_eJupN    - equatorial radius of Jupiter
* R_pJupN    - polar radius of Jupiter
* GM_EarthN  - terrestrial mass parameter
* GM_JupN    - jovian mass parameter
* M_EarthN   - mass of the Earth from GM_EarthN and G_2014
* M_JupN     - mass of Jupiter from GM_JupN and G_2014
* V_EarthN   - volume of the Earth (4.pi.R_eEarthN^2.R_pEarthN/3) 
* V_JupN     - volume of Jupiter (4.pi.R_eJupN^2.R_pJupN/3)
* R_EarthN   - volume-average radius of the Earth  (3.V_EarthN/4.pi)^(1/3)
* R_JupN     - volume-average radius of Jupiter  (3.V_JupN/4.pi)^(1/3)


Related constants
-----------------
* G_2014          - 2014 CODATA value for the Newtonian constant
* mean_solar_day  - 86,400.002 seconds [2]_
* au              - IAU 2009 value for astronomical constant in metres. [3]_
* pc              - 1 parsec = 3600*au*180/pi
* c               - speed of light = 299,792,458 m / s

Example
-------
Calculate the density relative to Jupiter for a planet 1/10 the radius of the
Sun with a mass 1/1000 of a solar mass.  Note that we use the volume-average
radius for Jupiter in this case::

 >>> from pycheops.constants import M_SunN, R_SunN, M_JupN, R_JupN
 >>> M_planet_Jup = M_SunN/1000 / M_JupN
 >>> R_planet_Jup = R_SunN/10 / R_JupN
 >>> rho_planet_Jup = M_planet_Jup / (R_planet_Jup**3)
 >>> print ("Planet mass    = {:.3f} M_Jup".format(M_planet_Jup))
 >>> print ("Planet radius  = {:.3f} R_Jup".format(R_planet_Jup))
 >>> print ("Planet density = {:.3f} rho_Jup".format(rho_planet_Jup))
 Planet mass    = 1.048 M_Jup
 Planet radius  = 0.995 R_Jup
 Planet density = 1.063 rho_Jup

.. rubric:: References
.. [1] https://www.iau.org/static/resolutions/IAU2015_English.pdf
.. [2] http://tycho.usno.navy.mil/leapsec.html
.. [3] Luzum et al., Celest Mech Dyn Astr (2011) 110:293-304

"""

from __future__ import (absolute_import, division, print_function,
                                unicode_literals)
__all__ = [ 'G_2014', 
        'R_SunN','S_SunN','L_SunN','Teff_SunN','GM_SunN','M_SunN','V_SunN',
        'R_eEarthN','R_pEarthN','GM_EarthN','M_EarthN','V_EarthN','R_EarthN',
        'R_eJupN','R_pJupN','GM_JupN','M_JupN','V_JupN','R_JupN',
        'mean_solar_day','au','pc','c']

from math import pi

G_2014 = 6.67408E-11          # m3.kg-1.s-2, 2014 CODATA value
                            
R_SunN    = 6.957E8           # m, solar radius
S_SunN    = 1361.             # w.m-2 total solar irradiance
L_SunN    = 3.828E26          # W, solar luminosity
Teff_SunN = 5772.             # K, solar effective temperature
GM_SunN   = 1.3271244E20      # m3.s-2, solar mass parameter
M_SunN    = GM_SunN/G_2014    # kg, solar mass derived from GM_SunN and G_2014
V_SunN    = 4*pi*R_SunN**3/3  # m3,  solar volume 

R_eEarthN  = 6.3781E6         # m, equatorial radius of the Earth
R_pEarthN  = 6.3568E6         # m, polar radius of the Earth
R_eJupN    = 7.1492E7         # m, equatorial radius of Jupiter
R_pJupN    = 6.6854E7         # m, polar radius of Jupiter
GM_EarthN  = 3.986004E14      # m3.s-2, terrestrial mass parameter
GM_JupN    = 1.2668653E17     # m3.s-2, jovian mass parameter
M_EarthN   = GM_EarthN/G_2014 # kg,  mass of the Earth from GM_EarthN and G_2014
M_JupN     = GM_JupN/G_2014   # kg,  mass of Jupiter from GM_JupN and G_2014
V_EarthN   = 4*pi*R_eEarthN**2*R_pEarthN/3 # m^3, volume of the Earth 
V_JupN     = 4*pi*R_eJupN**2*R_pJupN/3        # m^3, volume of Jupiter 
R_EarthN   = (R_eEarthN**2*R_pEarthN)**(1/3.) # m, mean radius of the Earth  
R_JupN     = (R_eJupN**2*R_pJupN)**(1/3.)     # m, mean radius of Jupiter

mean_solar_day = 86400.002    # seconds

au =  1.49597870700E11        # m, IAU 2009 Astronomical unit
pc =  3600*au*180/pi          # m, parsec
c  = 299792458                # m.s-1, speed of light

