Changes since 1.0.0 onwards.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1.1.14 (2024-11-05)
~~~~~~~~~~~~~~~~~~~
* Changed instances of BJD_TDB to BJD_TT in the docstrings for dataset.py and
  multivisit.py. (Difference is < 1 ms)
* Updated import commands in dataset.py and __init__.py for
  photutils.CircularAperture and photutils.aperture_photometry

1.1.13 (2024-09-17)
~~~~~~~~~~~~~~~~~~~
* Added SpotCrossingModel,TransitModel1Spot and TransitModel2Spot to models.
* Fix to plotting with glint in dataset (#320)  
* Avoid need for aperture and decontaminate keywords in
  dataset.get_lightcurve() for Kepler/TESS data from pycdata
* Avoid "No metadata" warning when loading Kepler/TESS data from pycdata
* Added omega as a derived parameter for all models that have parameters f_c,
  f_s, and updated other models to have the same expression for this value.
* Updated pycheops cookbook to version 1.1 from Pia Cortes Zuleta

1.1.12 (2024-06-13)
~~~~~~~~~~~~~~~~~~~
* Update dataset.animate_frames() (Tom Wilson)

1.1.11 (2024-06-05)
~~~~~~~~~~~~~~~~~~~
* Cookbook updated (Pia Cortes Zuleta)

1.1.10 (2024-04-24)
~~~~~~~~~~~~~~~~~~~
* Example notebooks updated.

1.1.9 (2024-04-16)
~~~~~~~~~~~~~~~~~~
* Updated make_xml_files to cope with change to data table format returned by
  astroquery.gaia. 

1.1.8 (2024-03-16)
~~~~~~~~~~~~~~~~~~
* Added 'L_0' to list of valid plotkeys in multivisit.corner_plot() and
  tidied-up logic so that invalid keys can return a helpful error message.

1.1.7 (2024-03-06)
~~~~~~~~~~~~~~~~~~
* Update make_xml_files and fix bug with N_Ranges being ignored.

1.1.6 (2024-02-09)
~~~~~~~~~~~~~~~~~~
* Added (insecure) work-around for SSL: CERTIFICATE_VERIFY_FAILED problem in
  StarProperties when downloading SWEET-Cat.

1.1.5 (2023-11-10)
~~~~~~~~~~~~~~~~~~
* Add fix for SSLCertVerificationError to README.rst

1.1.4 (2023-09-27)
~~~~~~~~~~~~~~~~~~
* PlanetModel renamed to HotPlanetModel
* Multivisit.fit_planet() renamed to Multivisit.fit_hotplanet()
* New PlanetModel that assumes flux from planet is due to reflection
* Multivisit.fit_planet() now assumes flux from planet is due to reflection
* Ensure funcs.eclipse_phase() returns phases in the range [0, 1)  
* Change default keyword value a_c=0 to a_c=None so that it gets properly
  initialised from the input datasets.

1.1.3 (2023-09-20)
~~~~~~~~~~~~~~~~~~
* Improved visibility calculation in make_xml_files. Visibility for stars with
  ecliptic latitude <-60,>+60 is now 0 (CHEOPS-UGE-PSO-MAN-001  section 1.3.2)

1.1.2 (2023-09-01)
~~~~~~~~~~~~~~~~~~
* make_xml_files bug fix.

1.1.1 (2023-08-31)
~~~~~~~~~~~~~~~~~~
* Correction to tzero parameter definitions in docstrings for module funcs 
* Correction to description of mask in funcs.t2z() docstring
* Removed argument "P" in call to funcs.eclipse_phase and updated docstring
* Removed "-c" option from make_xml_files
* Added funcs/contact_points()

1.1.0 (2023-07-14)
~~~~~~~~~~~~~~~~~~
* New extra_decorr_vectors option in Dataset and Multivisit fitting routines.
* New Dataset.select_detrend() feature, parameter selection from Bayes factors
* New MultiVisit.fit_planet() method for transit+eclipse fitting.
* Added 'tag' option to Dataset.save() and Multivisit.save()
* Added Dataset.from_pipe_file()
* Added MultiVisit.save() and MultiVisit.load() (#176)
* Added "copy=False" in call to interp1d in Dataset._make_interp().
* Changed zero-point of scaling for xoff, yoff, bg, etc. to median instead of
  mid-point of the values - should reduce correlation with 'c'. 
* Updated description of parameter scaling in  Dataset.lmfit_transit().
* In Dataset, set source automatically from file_key if not specified by user.
* Added xlim option to Dataset.plot_lmfit() and Dataset.plot_emcee().
* Added esinw, ecosw, T_tot, etc. to parameters for eccentric orbits in 
  the fitting routines in Dataset and MultiVisit.
* Added notes on unwrap and nroll to fit report in MultiVisit (#285)
* Raise error if initial value is out of range for Dataset or MultiVisit.
* Scaling of contam, smear and bg in MultiVisit, now consistent with Dataset
* Added target location on CCD to verbose output for Dataset.get_lightcurve()
* In Dataset, yoff was measured relative to the wrong value - fixed.
* Improved initialisation of walkers in MultiVisit fit routines - use standard
  deviation based on previous fits rather than arbitrary values.
* Added overwrite=False keyword option to MultiVisit.save() and Dataset.save() 
* Fix problem with automatic selection of x limits in MultiVisit.plot_fit()
* Fix problem on 'c' missing from parameters if fixed for MultiVisit
* Fix bug in calculation of rms for MultiVisit
* Fix display of prior for T_0 in MultiVisit.corner_plot()
* Allow list input to combine.combine()
* Added custom_labels option to MultiVisit.corner_plot()

1.0.19 (2023-05-12)
~~~~~~~~~~~~~~~~~~~
* Added "aperture" attribute to Dataset to store aperture name.
* Added scaling of detrending functions to Dataset.aperture_scan() 
* Added N_data to output of Dataset.aperture_scan()
* Added copy_initial option to Dataset.aperture_scan()
* Added "ramp" in Dataset.aperture_scan() - was documented but not implemented
* Dataset.aperture_scan(return_full=true) now also returns time,flux,flux_err 
  
1.0.18 (2023-05-06)
~~~~~~~~~~~~~~~~~~~~
* Fixed bug in calculation of the Moon - target separation for planet_check()  
* Added funcs.delta_t_sec(), light travel time correction for eclipses.
  
1.0.17 (2023-05-05)
~~~~~~~~~~~~~~~~~~~~
* Replaced np.int and np.float with int and float everywhere. (#292) 

1.0.16 (2023-02-01)
~~~~~~~~~~~~~~~~~~~~
* Added Dataset.aperture_scan() to help users find the best aperture choice
* Changed scaling of bg, contam and smear basis functions for decorrelation
  from (0,1) to (-1,1). This reduces the strong correlations between the
  constant scaling factor "c" and the decorrelation coefficients dfdbg,
  dfdsmear and dfdcontam. 
* Update examples/Notebooks/KELT-11b for consistency with changes above.
* Change examples/Notebooks/WASP-189 to download data from DACE. Remove
  example data examples/Notebooks/CH_PR100041_TG00020?_V0102.tgz
* Catch decorrelation against parameters with zero range in
  dataset.lmfit_transit and dataset.lmfit_eclipse. (#207)
* Remove power2
* Fixed "warnings is not defined" bug in planetproperties.
* Replace python-dace-client dependency with dace-query.
* Suppress UnitsWarning in Dataset when reading from FITS files.
* Add advice to update config file if psf_file generates KeyError
* Add IPython to requirements in setup.py

1.0.15 (2022-10-14)
~~~~~~~~~~~~~~~~~~~~
* Fix bug in dataset.load() for datasets with no defined model
  
1.0.14 (2022-09-07)
~~~~~~~~~~~~~~~~~~~~
* Fixed bug in Dataset that prevents import of R25 aperture lightcurve.
* Temporarily disabled power2

1.0.13 (2022-08-28)
~~~~~~~~~~~~~~~~~~~~
* Use parameter stderr values to initialize walkers in Dataset. 
* Default init_scale value in Dataset fit functions changed from 0.01 to 0.5
  
1.0.12 (2022-08-18)
~~~~~~~~~~~~~~~~~~~~
* Enable Dataset to load old saved datasets with no __scale__ attribute

1.0.11 (2022-08-17)
~~~~~~~~~~~~~~~~~~~~
* Starproperties - use Logg if Logg_gaia missing from SWEETCat
* Added Dataset.list_apertures()
* Updated Dataset to allow for new DRP14 aperture names
  
1.0.10 (2022-08-04)
~~~~~~~~~~~~~~~~~~~~
* Added Power2 class for improved handling of power-2 limb darkening
* Bug fix for missing argument "q" in funcs.RVCompanion
* Update reference to Maxted et al. in README.rst
* Added PLATO passband to ld.py
* Use Logg_gaia from SWEET-Cat instead of Logg (#276)
* In utils.pprint fix short format error where sf=1 appears as '(10)'
* Removed redundant _make_models function from multivisit
* Added "scale" option to dataset and multivisit fitting routines. 

1.0.9 (2022-05-19)
~~~~~~~~~~~~~~~~~~~
* Fix bug os.mkdirs() -> os.makedirs() in core.py

1.0.8 (2022-05-18)
~~~~~~~~~~~~~~~~~~~
* Added show_gp option to multivisit.plot_fit() for eclipse and transit fits
* Removed spurious line lc_fits.append(mod) at line 1029 of the multivisit.py
  file. (#271).

1.0.7 (2022-05-12)
~~~~~~~~~~~~~~~~~~~
* Changing the input file formats so that it can accepts files from other
  sources (PR #250, issue #249)
* Updated make_xml_files example files
* Added show_gp option to multivisit.plot_fit() for results of eblm_fit()
* In core, use os.makedirs(path, exist_ok=True) to avoid FileNotFoundError
  when creating cache directory requiring subdirectories.

1.0.5 (2022-03-14)
~~~~~~~~~~~~~~~~~~~
* Update planetproperties to use new header format for TEPCat
* Fixed typos in output of dataset.get_lightcurve (#256)
* Added teff attribute to Dataset, if T_EFF present in header (#266)
* Fixed problem using backends to restart MultiVisit (#263)  
* Catch cases where "c" is not a free parameter for datasets when plotting in
  MultiVisit (#251)

1.0.4 (2022-02-14)
~~~~~~~~~~~~~~~~~~~
* Added Dataset.bright_star_check()
* Included relativistic corrections in Models.RVModel() (experimental)
* Added note to inline help for instrument.response that TESS is available

1.0.3 (2022-01-19)
~~~~~~~~~~~~~~~~~~~
* BUG FIX. In dataset.py, decontaminate=True should apply the correction 
  flux = flux/(1 + contam), not flux = flux*(1 - contam). Fixed.
* Avoid "Warning: converting a masked element to nan." in starproperties.py
* Clarified definition of L in EclipseModel and EBLMModel
* Fixed retrieving psf_file bug in init.py (#255)
* Updated PSF reference file to average of in-flight PSFs measured at 9 CCD
  locations during IOC.
* Added l_3 option to models.py, dataset.py and multivisit.py.
* Added l_3, f_c and f_s to _make_labels in dataset.py and multivisit.py
* Fixed "SyntaxWarning: "is" with a literal." from multivisit.py and core.py
* Update Contamination_33arcsec_aperture.p if older than the reference
  psf_file in __init__.py

1.0.2 (2021-12-09)
~~~~~~~~~~~~~~~~~~~
* Fix problem with SWEET-Cat encoding (#252)
* Add decontaminate method to dataset (experimental)
* Fix issue in WASP-189 notebook with missing text files for cds_data_export

1.0.1 (2021-11-21)
~~~~~~~~~~~~~~~~~~~
* Attempted fix in 0.9.18 to avoid hidden files in dataset() failed - fixed.

1.0.0 (2021-11-17)
~~~~~~~~~~~~~~~~~~~
* Updated readme, notebooks and cookbook for release of version 1.0.0
