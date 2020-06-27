PYCHEOPS
========

Python package for the analysis of light curves from the `ESA CHEOPS mission <http://cheops.unibe.ch/>`_.

Use ``pip install pycheops`` to install.

See `pycheops/examples/Notebooks <https://github.com/pmaxted/pycheops/tree/master/pycheops/examples/Notebooks>`_ for examples.

For discussion and announcements, please join the `pycheops google group <https://groups.google.com/forum/#!forum/pycheops>`_

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

