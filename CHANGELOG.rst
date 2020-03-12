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

0.3.0 (2019-09-29)
~~~~~~~~~~~~~~~~~~
 Dataset transit fitting methods added

0.3.1 (2019-10-02)
~~~~~~~~~~~~~~~~~~
 Added dataset_fit_transit_from_simulation.ipynb to examples/Notebooks

0.3.2 (2019-10-02)
~~~~~~~~~~~~~~~~~~
 Update requirements in setup.py

0.3.3 (2019-10-02)
~~~~~~~~~~~~~~~~~~
Fix version requirements problem in setup.py

0.3.4 (2019-10-03)
~~~~~~~~~~~~~~~~~~
Remove username and password from config

0.3.5 (2019-10-03)
~~~~~~~~~~~~~~~~~~
Previous upload failed

0.3.6 (2019-10-03)
~~~~~~~~~~~~~~~~~~
config bug fix 

0.3.7 (2019-10-03)
~~~~~~~~~~~~~~~~~~
Remove ellc from requirements

0.3.8 (2019-10-03)
~~~~~~~~~~~~~~~~~~
Move "from ellc import ld" to avoid import if not needed in ld.py

0.3.9 (2019-10-03)
~~~~~~~~~~~~~~~~~~
Second attempt to avoid ellc import (try/except)

0.4.0(2019-10-20)
~~~~~~~~~~~~~~~~~
 Subversion change - previous version presented at CST meeting.
 Bug fix in calculation of e,om from f_c, f_s in pychepos/models.py
 Dataset() now downloads from DACE
 Changed dataset_id to file_key in Dataset for consistency with DACE
 Changed transit_noise to subtract of best-fit transit depth before
 calculation. 
 Made minerr_transit_fit more robust
 Fixed calculation of transit parameters in EclipseModel

0.4.1 (2019-10-22)
~~~~~~~~~~~~~~~~~~
 Make elements of lc['xoff'], lc['yoff'] and lc['roll_angle'] consistent with
 lc['time'], lc['flux'], etc. following high point rejection. 
 Added detrending coeffs dfdsin2phi and dfdcos2phi
 Added requirement keyword to Dataset.transit_noise_plot()

0.5.0 (2019-11-01)
~~~~~~~~~~~~~~~~~~
 Replace parameter S in TransitModel and EclipseModel with b.
 Bug fix for parameters dfdcos2phi and dfdsin2phi in dataset.
 Added dataset.lmfit_eclipse and renamed emcee_transit to emcee_sampler.
 Added "detrend" option to dataset.plot_lmfit and dataset.plot_emcee.
 Put dace.cheops import inside try:/except

0.6.0 (2019-11-06)
~~~~~~~~~~~~~~~~~~
 Generate pickle files in data directory at run time when first needed. 
 Single-source version number from pycheops/VERSION
 Removed stagger_claret_interpolator and stagger_mugrid_interpolator from ld.

0.6.1 (2019-11-22)
~~~~~~~~~~~~~~~~~~
 Remove error message if there is no imagette data in the dataset.
 Remove DACE import warning in dataset
 Added calculation of prior on P(D, W, b) for transit/eclipse fitting assuming
 uniform priors on cos(i), log(k) and log(a/R*).  

0.6.2 (2020-01-25)
~~~~~~~~~~~~~~~~~~
 Store pickle files in user's cache directory to avoid permissions issues
 with root user installations. (not finished)
 Added --scaling-factor-percent option to make_xml_files.
 Fix bug in make_xml_files where T_exp is stored as an integer - now float
 Improved interpolation of exposure times. (not finished)
 Updated spectral-type T_eff G-V values in make_xml_files (not finished)
 Bug fix for cases where log_g, [Fe/H] not defined in sweetcat.
 Add option for user-defined parameters in starproperties.

0.6.3 (2020-02-01)
~~~~~~~~~~~~~~~~~~
 Completed the changes from version 0.6.2 - store pickle files in user's cache
 directory, interpolation of exposure times, update spectral-type T_eff G-V
 values.
 Fixed J=L/D in EclipseModel
 Added EBLMModel to models.
 Added a few examples of TESS analysis to  examples/Notebooks
 Changed target TESS_fit_EB.ipynb to TESS_fit_EBLM.ipynb  fit to EBLM J0113+31.

0.6.4  (2020-02-19)
~~~~~~~~~~~~~~~~~~~
 Simplified call to astroquery.gaia in make_xml_files - fixes HTTPError 302
 problem that started happening since the last update.
 change at the server(?)

0.6.5
~~~~~~
 Change BJD_late to 2460000.5 in example make_xml_file input files.
 Add --count_rate option to make_xml_files

0.6.6
~~~~~
 Added numba version requirement to setup.py
 Added V magnitude and spectral type information to dataset object
 Add light curve stats to dataset objects
