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
quantities
==========
Nominal values of solar and planetary constants from IAU Resolution B3 [1] 
plus related constants as astropy quantities.

Masses in SI units are derived using the 2014 CODATA value for the Newtonian
constant, G=6.67408E-11 m3.kg-1.s-2.

The following conversion constants are defined.

Solar conversion constants 
--------------------------
* R_SunN     - solar radius
* S_SunN     - total solar irradiance
* L_SunN     - solar luminosity
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
* mean_solar_day  - 86,400.002 seconds [2] 
* au              - IAU 2009 value for astronomical constant in metres. [3]
* pc              - 1 parsec = 3600*au*180/pi

Fundamental constants
---------------------
* c               - speed of light in m.s-1 [3]

Example
-------
Calculate the density relative to Jupiter for a planet 1/10 the radius of 
the Sun with a mass 1/1000 of a solar mass. Note that we use the 
volume-average radius for Jupiter in this case::

 >>> from pycheops.quantities import M_SunN, R_SunN, M_JupN, R_JupN
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

import astropy.units as u

__all__ = [ 'G_2014', 
        'R_SunN','S_SunN','L_SunN','Teff_SunN','GM_SunN','M_SunN','V_SunN',
        'R_eEarthN','R_pEarthN','GM_EarthN','M_EarthN','V_EarthN','R_EarthN',
        'R_eJupN','R_pJupN','GM_JupN','M_JupN','V_JupN','R_JupN',
        'mean_solar_day','au','pc']

from math import pi

G_2014 = 6.67408E-11* u.m**3/u.kg/u.s**2   # 2014 CODATA value
                            
R_SunN    = 6.957E8  *u.m                # Solar radius
S_SunN    = 1361 *u.W/u.m**2             # Total solar irradiance
L_SunN    = 3.828E26 *u.W                # Solar luminosity
Teff_SunN = 5772 *u.K                    # Solar effective temperature
GM_SunN   = 1.3271244E20 *u.m**3/u.s**2 # Solar mass parameter
M_SunN    = GM_SunN/G_2014               # Solar mass 
V_SunN    = 4*pi*R_SunN**3/3             # Solar volume 

R_eEarthN  = 6.3781E6 *u.m               # Equatorial radius of the Earth
R_pEarthN  = 6.3568E6 *u.m               # Polar radius of the Earth
R_eJupN    = 7.1492E7 *u.m               # Equatorial radius of Jupiter
R_pJupN    = 6.6854E7 *u.m               # Polar radius of Jupiter
GM_EarthN  = 3.986004E14  *u.m**3/u.s**2 # Terrestrial mass parameter
GM_JupN    = 1.2668653E17 *u.m**3/u.s**2 # Jovian mass parameter
M_EarthN   = GM_EarthN/G_2014            # Earth mass
M_JupN     = GM_JupN/G_2014              # Jupiter mass
V_EarthN   = 4*pi*R_eEarthN**2*R_pEarthN/3 # Volume of the Earth 
V_JupN     = 4*pi*R_eJupN**2*R_pJupN/3     # Volume of Jupiter 
R_EarthN   = (R_eEarthN**2*R_pEarthN)**(1/3) # Mean radius of the Earth  
R_JupN     = (R_eJupN**2*R_pJupN)**(1/3)     # Mean radius of Jupiter

mean_solar_day = 86400.002 *u.s  # seconds

au =  1.49597870700E11 *u.m  # IAU 2009 Astronomical unit
pc =  3600*au*180/pi         # parsec

c = 2.99792458e8 *u.m/u.s     #  Speed of light
