PYCHEOPS
========

Python package for the analysis of light curves from the `ESA CHEOPS mission <http://cheops.unibe.ch/>`_.

Use ``pip install pycheops`` to install.

See `pycheops/examples/Notebooks <https://github.com/pmaxted/pycheops/tree/master/pycheops/examples/Notebooks>`_ for examples.

For discussion and announcements, please join the `pycheops google group <https://groups.google.com/forum/#!forum/pycheops>`_

See pycheops/docs/PyCheops_Cookbook.pdf for a guide to using pycheops.

See pycheops/examples/Notebooks for Jupyter notebook that illustrate the
features of pycheops.

See Maxted et al. 2021 (arxiv:2111.08828) for a full description of the
algorithms and assumptions used in pycheops. 

Troubleshooting
***************

Installation fails with "ModuleNotFoundError: No module named 'pybind11'"
--------------------------------------------------------------------------

Run ``pip install pybind11`` then try again

StarProperties(dataset.target) produces "Segmentation fault: 11"
-----------------------------------------------------------------

You are running the wrong version of python, e.g., anaconda2 instead of anaconda3

"TypeError: 'str' object is not callable" in animate frames 
------------------------------------------------------------
Install "pillow", e.g., conda install pillow.

"No matching distribution found for matplotlib>3,2 (from pycheops)" 
--------------------------------------------------------------------
 This error message may appear when upgrading to pycheops version 0.8.0 or
 later.  Two solutions have been found ...

 1. "conda update --all" then "pip install pycheops --upgrade"

 2. "pip install matplotlib --upgrade" then "pip install pycheops --upgrade"

Installation fails with "ERROR: Could not build wheels for celerite2 which use PEP 517 and cannot be installed directly" 
-------------------------------------------------------------------------------------------------------------------------
 This error message may appear when upgrading to pycheops version 0.9.1 or
 later. The working solution is to install celerite2 prior to installing/
 updating pycheops using:

 ``git clone --recursive https://github.com/dfm/celerite2.git``

 ``cd celerite2``

 ``python -m pip install celerite2==0.0.1rc1``
