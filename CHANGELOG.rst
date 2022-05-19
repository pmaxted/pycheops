Changes since 1.0.0 onwards.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
  sources (PR #250)
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
