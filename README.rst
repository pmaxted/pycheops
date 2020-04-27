PYCHEOPS
========

Python package for the analysis of light curves from the ESA CHEOPS mission <http://cheops.unibe.ch/>.

Use *pip install pycheops* to install.

See examples/Notebooks for examples.


Troubleshooting
***************


Installation fails with "ModuleNotFoundError: No module named 'pybind11'"
--------------------------------------------------------------------------
Run "pip install pybind11" then try again

StarProperties(dataset.target) produces "Segmentation fault: 11"
-----------------------------------------------------------------
 You are running the wrong version of python, e.g., anaconda2 instead of anaconda3
