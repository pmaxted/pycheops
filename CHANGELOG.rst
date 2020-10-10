Changes since 0.6.0 onwards.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

0.9.3 (2020-10-10)
~~~~~~~~~~~~~~~~~~
* Fixed missing Bayes factor for d2fdt2 (#159)
* Changed aperture used to extract metadata to DEFAULT in dataset

0.9.2 (2020-09-25)
~~~~~~~~~~~~~~~~~~
* Removes autograd from requirements in setup.py
* Added solar options to funcs.massradius()
* Changed default thin=4 to thin=1 in dataset.emcee_sampler()
* Fixed bug in Multivisit for default log_Q value (#155)
* Added PlanetProperties
* Updated KELT-11b-tutorial.ipynb to show use of PlanetProperties
* Update example TESS notebooks to celerite2

0.9.1 (2020-09-10)
~~~~~~~~~~~~~~~~~~
* celerite -> celerite2
* Fix missing DRP report due to new file structure for simulation data (#146)

0.9.0 (2020-09-09)
~~~~~~~~~~~~~~~~~~
* Added tqdm to requirements in setup.py
* Added "unwrap" option to Multivisit fit routines fit_transit(), etc. 
* Set mean value of glint function to 0 in dataset.add_glint().
* Fixed bug with evaluation of glint function in Multivisit 
* Fixed bug in Multivisit.plot_fit() - model plotted using old parameters

0.8.5 (2020-09-02)
~~~~~~~~~~~~~~~~~~
* Added funcs.tperi2tzero() and funcs.eclipse_phase()
* Added "Bayes factors" section to dataset.lmfit_report()
* Added multivisit.fit_eblm
* Added pycheops/examples/Notebooks/KELT-11b-tutorial.ipynb

0.8.4 (2020-08-30)
~~~~~~~~~~~~~~~~~~
* Fix parameter hint prefix problem in models (#141)
* Fix -ve offset ylimit problem in multivisit (#139)
* Added warning is failed to update TEPCat in funcs.massradius (#137)
* Fix bug in dataset and multivisit if only 1 variable in trailplot (#130)
  
0.8.3 (2020-07-30)
~~~~~~~~~~~~~~~~~~
* Fix astype(int) problem in __init__.py for windows users
* Fix bug in multivisit where priors on derived parameter were ignored.
  
0.8.2 (2020-07-26)
~~~~~~~~~~~~~~~~~~
* Read datasets into multivisit object in a logical order (#133)
* Update T0 in dataset.emcee.params_best and dataset.emcee.chain in multivisit
* Fix copy.copy bug in dataset.should_I_decorr() 

0.8.1 (2020-06-29)
~~~~~~~~~~~~~~~~~~
* Added multivisit.ttv_plot()
* Changed parameter names to ttv_01, L_01, etc. in multivisit to cope with
  multivisit objects with >9 datasets.
* Added min/max values from params to modpars in multivisit
* multivisit datadir join bug fix
* Fixed title keyword option in multivisit.plot_fit()

0.8.0 (2020-06-28)
~~~~~~~~~~~~~~~~~~
* Added Multivisit class
* Added load() and save() to dataset
* Added dace keywords to StarProperties
* Added option to set user-defined values using a 2-tuple in StarProperties 
* Bug fixes for animate_frames 
* Add requirement for matplotlib 3.2.2 to setup.py
* Get fits extensions by name in dataset
* Updated notebooks in examples/Notebooks

0.7.8 (2020-06-03)
~~~~~~~~~~~~~~~~~~
* Suppress warnings from matplotlib.animate in dataset
* Subarray metadata search fix (#110)
* Add check for finite flux values in dataset.get_lightcurve()
* should_I_decorr bug fix, code cleanup and expansion (#115)
  
0.7.7 (2020-05-12)
~~~~~~~~~~~~~~~~~~
*N.B.* New behaviour for dataset.get_lightcurve()

* dataset.get_lightcurve() now subtracts contaminating flux by default
* added decontaminate keyword to dataset.get_lightcurve() (#82)
* dataset.add_glint() function is now  periodic (#87)
* Added outlier rejection to dataset.diagnostic_plot (#84)
* Add functions to dataset to view/animate images (#83)
* Updated comments re: decorrelation in example notebooks 
* Bug fix to moon angle calculation in dataset.py
* Fix math errors in funcs.massradius caused by negative values (#104)
* Fix math errors in dataset.massradius caused by negative values (#104)
* dataset.get_subarray adapted to allow use of simulated data

0.7.6 (2020-05-01)
~~~~~~~~~~~~~~~~~~
* Fixed y-axis title bug in dataset.rollangle_plot (#85).
* Added robust grid search to funcs.tzero2tperi

0.7.5 (2020-04-27)
~~~~~~~~~~~~~~~~~~
* Bug fix in dataset for d2fdx2, d2fdy2, d2fdt2
* Reduced size of initial bracketing interval in funcs.tzero2tperi
* Wrong units on stellar mass/radius in funcs.massradius fixed
* Fixed decorr with bg, contam, sin3phi, cos3phi bug (#80)
* Added fallback in utils/parprint() if error is 0

0.7.4 (2020-04-23)
~~~~~~~~~~~~~~~~~~
* Added dataset.planet_check
* Added moon option to add_glint
* Dropped angle0 option from dataset.rollangle_plot
* Bug fix in funcs.massradius for calls without m_star or r_star

0.7.3 (2020-04-22)
~~~~~~~~~~~~~~~~~~
* Documentation update for funcs.massradius
* Bug fix in decorr and should_I_decorr (#73)

0.7.2 (21-04-2021)
~~~~~~~~~~~~~~~~~~
* Improved edge behaviour of dataset.clip_outliers
* Added option in starproperties to not raise error if star not in SWEET-Cat
* Added plot_model to dataset.plot_lmfit
* Fixed offset problem for transit model in dataset.plot_emcee
* Added sini to derived parameters listed in dataset
* Improved funcs.m_comp using closed-form solution of cubic polynomials.
* Added funcs.massradius and dataset.massradius
* Added catch for e>0.999 in models

0.7.1 (14-04-2020)
~~~~~~~~~~~~~~~~~~
* Fixed dataset flux.nanmean issue caused by merge on github.

0.7.0 (13-04-2020)
~~~~~~~~~~~~~~~~~~
* Added kwargs to dataset.corner_plot
* Added binned data points to dataset.plot_lmfit and dataset.plot_emcee
* Added utils.lcbin and utils.parprint
* Moved priors appended to dataset.lmfit.residual to their own object
  dataset.lmfit.prior_residual and added dataset.npriors
* Fixed bug on models.FactorModel for dfdsin3phi and dfdcos3phi
* Tidied-up/improved interpolation of dependent variables in dataset
* Fixed bug with xoff being assigned to yoff in dataset.lmfit_transit() and
  dataset.lmfit_eclipse()
* Added dataset.rollangle_plot()
* Set stderr and correl values for dataset.emcee.params_best - breaks printing
  otherwise.
* Changed logic in dataset.emcee_sampler() so add_shoterm works if param
  keyword is specified.
* Enabled show_priors option in dataset.corner_plot()
* Added kwargs to dataset.lmfit_report() and dataset.emcee_report
* Added RMS residual to dataset.lmfit_report() and dataset.emcee_report()
* Added dataset.mask_data()
* Added dataset.plot_fft()
* Added dataset.trail_plot()
* Updated dataset examples in pycheops/examples/Notebooks
* Removed bug in dataset when setting h_1, h_2 from tuple.
* Removed bug when plotting GPs in dataset that caused an offset ("flux0=flux
  is not a copy" issue).
* Added ld.atlas_h1h2_interpolator and used it in starproperties
* Added ld.phoenix_h1h2_interpolator and used it in starproperties
* Moved pickle files used in ld.py to user's cache directory instead of the
  installation data directory.
* Added dataset.add_glint() and scaled glint correction to lmfit/emcee fits

0.6.9 (2020-04-02)
~~~~~~~~~~~~~~~~~~
* Bug fix for use of bg and contam in dataset.py 
* Changed to interp1d from InterpolatedUnivariateSpline in dataset.py

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

