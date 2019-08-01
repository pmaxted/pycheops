0.0.3 (2018-03-24)
~~~~~~~~~~~~~~~~~~
 Renamed pycheops from cheops

0.0.4 (2018-08-06)
~~~~~~~~~~~~~~~~~~
 Added targets.py (make_xml_files)

0.0.5 (2018-08-06)
~~~~~~~~~~~~~~~~~~
 Added version requirements to install_requires in setup.py

0.0.6 (2018-08-13)
~~~~~~~~~~~~~~~~~~
 Added --ignore-gaia-id-check flag
 Changed Gaia DR2 lookup from Vizier to gea.esac.esa.int with TapPlus
 targets.py -> make_xml_files.py

0.0.7 (2018-08-13)
~~~~~~~~~~~~~~~~~~
 Fixed errors in documentation for make_xml_files re: source of Gaia DR2 ID

0.0.8 (2018-08-15)
~~~~~~~~~~~~~~~~~~
 Added "V" to spectral type in output xml file so it is picked up by PHT2 tool

0.0.9 (2018-08-16)
~~~~~~~~~~~~~~~~~~
 Added license to source files
 funcs.m_comp now return np.nan for non-finite input values.

0.0.10 (2018-09-04)
~~~~~~~~~~~~~~~~~~~
  Updated PSF contamination file.

0.0.11 (2018-09-04)
~~~~~~~~~~~~~~~~~~~
  Added warning flag 16 for magnitude out of range when -a flag not set
  Added -e/--example-file-copy to copy over example files

0.0.12 (2018-09-18)
~~~~~~~~~~~~~~~~~~~
 Added numba to package requirements and re-factored the following functions
 to enable @jit acceleration - funcs.esolve, lc.qpower2, lc.ueclipse

0.0.12 (2018-09-19)
~~~~~~~~~~~~~~~~~~~
 Replaced corrupted file TESS_stagger_power2_interpolator.p
 Removed nopython=True from numbba @jit function decorators in lc.py to
 enable python warnings. 

0.0.13 (2018-09-19)
~~~~~~~~~~~~~~~~~~~
 Moved functions from lc.py into models.py

0.0.14 (2018-12-04)
~~~~~~~~~~~~~~~~~~~
 Refactored funcs.py to use numba @vectorize decorators. Added vrad().

0.0.15 (2018-12-04)
~~~~~~~~~~~~~~~~~~~
 Moved development back to github 

0.0.16 (2018-12-04)
~~~~~~~~~~~~~~~~~~~
 Various bug-fixes and homogenization of parameters in funcs.py and models.py

0.0.17 (2018-12-05)
~~~~~~~~~~~~~~~~~~~
 Removed link to readthedocs from README

0.0.18 (2018-12-09)
~~~~~~~~~~~~~~~~~~~
 Added missing data files

0.1.0 (2018-12-20)
~~~~~~~~~~~~~~~~~~
 Promote previous version to 0.1.0 since it has been used for CHEOPS SV3
 exercise.

0.1.1 (2019-03-02)
~~~~~~~~~~~~~~~~~~
 Replaced missing make_xml_files.py file
 Includes some development work on models with lmfit (incomplete)

0.1.2 (2019-03-04)
~~~~~~~~~~~~~~~~~~
 Update make_xml_files for new version of Feasibility Checker

0.1.3 (2019-03-05)
~~~~~~~~~~~~~~~~~~
 Bug fix in  make_xml_files for non time-critical observations.

0.1.4 (2019-03-08)
~~~~~~~~~~~~~~~~~~
 No Fulfil_all_Phase_Ranges in make_xml_files if there are no phase ranges. 

0.1.4 (2019-03-14)
~~~~~~~~~~~~~~~~~~
 lmfit...

0.1.5 (2019-05-06)
~~~~~~~~~~~~~~~~~~
 Use string comparison for Gaia ID check (issue #39)
 Gaia epoch change 2015.0 -> 2015.5 (issue #40)

0.1.6 (2019-06-17)
~~~~~~~~~~~~~~~~~~
 Print contamination value to 3 d.p. (up from 2).

0.1.7 (2019-06-18)
~~~~~~~~~~~~~~~~~~
 Added models.scaled_transit_fit and instrument.transit_noise

0.1.8 (2019-06-19)
~~~~~~~~~~~~~~~~~~
 Remove debug output from instrument.transit_noise

0.1.9 (2019-07-09)
~~~~~~~~~~~~~~~~~~
 Added method='minerr' option to instrument.transit_noise and new function 
 models.minerr_transit_fit

0.2.0 (2019-07-15)
~~~~~~~~~~~~~~~~~~
  Added pycheops.dataset

0.2.1 (2019-07-15)
~~~~~~~~~~~~~~~~~~
  Fix dataset list output for python < 3.6

0.2.2 (2019-07-29)
~~~~~~~~~~~~~~~~~~
 Added q1q2 to ld.py

0.2.3 (2019-07-30)
~~~~~~~~~~~~~~~~~~
 Added NGTS to response_functions
 Change free parameters in ld._coefficient_optimizer to (q1, q2)
 Improved precision of values in limb-darkening tables

0.2.4 (2019-07-30)
~~~~~~~~~~~~~~~~~~
 Added NGTS to ld inline documentation.

0.2.5 (2019-08-01)
~~~~~~~~~~~~~~~~~~
 Added ftp download to dataset - temporarily using obsftp.unige.ch
