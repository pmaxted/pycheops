Changes since 0.6.0 onwards.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

0.6.8 (2020-04-02)
~~~~~~~~~~~~~~~~~~

* Fixed bug for new users - not possible to run setup_config()
* Fixed bug in instrument.py - log_exposure_time.p not used anymore

0.6.7 (2020-04-02)
~~~~~~~~~~~~~~~~~~

* Set vary=False default for f_c and f_s in TransitModel.
* Replaced vectorize in func/m_comp() with map.
* Fixed bug in dataset.lmfit_transit() and dataset.lmfit_eclipse() for fitting 
  d2fdx2, d2fdy2 and d2fdt2.
* Added dfdcontam to models/FactorModel() 
* Added dfdbg and dfdcontam to dataset.lmfit_transit and dataset.lmfit_eclipse()
* Changed CHANGELOG format
* Improved/simplified dataset.clip_outliers()
* Removed broken pool option from dataset.emcee_sampler()
* Additional parameter checks in EclipseModel and TransitModel
* Change default to reject_highpoints=False in dataset
* Include pycheops version with fit reports in dataset
* Added nu_max to funcs
* Updated instrument.count_rate and instrument.exposure_time to make them
  consistent with spreadsheet ImageETCv1.4, 2020-04-01
* Added instrument.cadence()
* Updated make_xml_files
* Updated pycheops/examples/Notebooks/TestThermalPhaseModel.ipynb 

0.6.6
~~~~~
* Added numba version requirement to setup.py.
* Added V magnitude and spectral type information to dataset object.
* Add light curve stats to dataset objects.
* Added "local" option to dataset.transit_noise_plot.
* Set max value of D to 0.25 in models.TransitModel and models.EBLMModel.
* Fixed bug with missing prefix in expr for param hints in models..
* Added model.PlanetModel.
* Added dataset.lc['bg'].
* Updated conf.py for sphinx documentation.

0.6.5
~~~~~~
* Change BJD_late to 2460000.5 in example make_xml_file input files.
* Add --count_rate option to make_xml_files

0.6.4  (2020-02-19)
~~~~~~~~~~~~~~~~~~~
* Simplified call to astroquery.gaia in make_xml_files - fixes HTTPError 302
  problem that started happening since the last update. Change at the server(?)

0.6.3 (2020-02-01)
~~~~~~~~~~~~~~~~~~
* Completed the changes from version 0.6.2 - store pickle files in user's cache
  directory, interpolation of exposure times, update spectral-type T_eff G-V
  values.
* Fixed J=L/D in EclipseModel
* Added EBLMModel to models.
* Added a few examples of TESS analysis to  examples/Notebooks
* Changed target TESS_fit_EB.ipynb to TESS_fit_EBLM.ipynb  fit to EBLM J0113+31.

0.6.2 (2020-01-25)
~~~~~~~~~~~~~~~~~~
* Store pickle files in user's cache directory to avoid permissions issues
  with root user installations. (not finished)
* Added --scaling-factor-percent option to make_xml_files.
* Fix bug in make_xml_files where T_exp is stored as an integer - now float
* Improved interpolation of exposure times. (not finished)
* Updated spectral-type T_eff G-V values in make_xml_files (not finished)
* Bug fix for cases where log_g, [Fe/H] not defined in sweetcat.
* Add option for user-defined parameters in starproperties.

0.6.1 (2019-11-22)
~~~~~~~~~~~~~~~~~~
* Remove error message if there is no imagette data in the dataset.
* Remove DACE import warning in dataset
* Added calculation of prior on P(D, W, b) for transit/eclipse fitting assuming
  uniform priors on cos(i), log(k) and log(a/R*).  

0.6.0 (2019-11-06)
~~~~~~~~~~~~~~~~~~
* Generate pickle files in data directory at run time when first needed. 
* Single-source version number from pycheops/VERSION
* Removed stagger_claret_interpolator and stagger_mugrid_interpolator from ld.

