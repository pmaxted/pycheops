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
core
====
Core functions for pycheops

"""

from __future__ import (absolute_import, division, print_function,
                                unicode_literals)
import os 
from configparser import ConfigParser
from sys import platform
import getpass

__all__ = ['load_config', 'setup_config', 'get_cache_path']



def find_config():
    r"""
    Find pycheops.cfg from a hierarchy of places
    
    First, try `~/pycheops.cfg`
    if that fails, path is platform dependent
    Linux: `$XDG_CONFIG_HOME/pycheops.cfg` (defaults to `~/.config/pycheops.cfg` if `$XDG_DATA_HOME` is not set)
    Windows: `%APPDATA%\pycheops\pycheops.cfg` (usually `C:\Users\user\AppData\Roaming\pycheops\pycheops.cfg`)
    Other: `~/pycheops/pycheops.cfg`
    """

    dirname='~'
    fname='pycheops.cfg'

    tryConfigFile = os.path.expanduser(os.path.join(dirname, fname))
    if os.path.isfile(tryConfigFile):
        configFile = tryConfigFile
        return configFile


    if platform == "linux" or platform == "linux2":
        dirname = os.getenv('XDG_CONFIG_HOME', os.path.expanduser(os.path.join('~', '.config')))
    elif platform == "win32":
        dirname = os.path.expandvars(os.path.join('%APPDATA%', 'pycheops'))
    else:
        dirname = os.path.expanduser(os.path.join('~', 'pycheops'))


    tryConfigFile = os.path.join(dirname, fname)
    if not os.path.isdir(dirname):
        os.makedirs(dirname, exist_ok=True)
    configFile = tryConfigFile
    return configFile


def load_config(configFile=None):
    """
    Load module configuration from configFile 
    
    If configFile is None, find pycheops.cfg

    :param configFile: Full path to configuration file

    """


    if configFile is None:
        configFile = find_config()

    if not os.path.isfile(configFile):
        raise ValueError('Configuration file not found - run core.setup_config')

    c = ConfigParser()
    c.read(configFile)
    return c


def setup_config(configFile=None, overwrite=False, mode=0o600, 
        data_cache_path=None, pdf_cmd=None):
    """
    Create module configuration 
    
    If configFile is None, find pycheops.cfg

    :param configFile: Full path to configuration file

    :param overwrite: overwrite values in existing configFile

    :param mode: mode (permission) settings for configFile

    :param data_cache_path: user is prompted if None, use '' for default

    :param pdf_cmd: user is prompted if None, use '' for default

    """

    if configFile is None:
        configFile = find_config()
    print('Creating configuration file {}'.format(configFile))

    if os.path.isfile(configFile) and not overwrite:
        raise ValueError('Configuration file exists and overwrite is not set')

    r"""
    `data_cache_default` is platform dependent and not in `~`
    Linux: `$XDG_DATA_HOME/pycheops` (defaults to `~/.local/share/pycheops` if `$XDG_DATA_HOME` is not set)
    Windows: `%APPDATA%\pycheops\data` (usually `C:\Users\user\AppData\Roaming\pycheops\data`)
    Other: `~/pycheops/data`
    """
    if platform == "linux" or platform == "linux2":
        data_cache_default = os.path.join(os.getenv('XDG_DATA_HOME', os.path.expanduser(os.path.join('~', '.local', 'share'))), 'pycheops')
    elif platform == "win32":
        data_cache_default = os.path.expandvars(os.path.join('%APPDATA%', 'pycheops', 'data'))
    else:
        data_cache_default = os.path.expanduser(os.path.join('~', 'pycheops', 'data'))

    prompt = "Enter data cache directory [{}] > ".format(data_cache_default)
    if data_cache_path is None:
        data_cache_path = input(prompt)
    if data_cache_path == '':
        data_cache_path = data_cache_default
    if not os.path.isdir(data_cache_path):
        os.makedirs(data_cache_path, exist_ok=True)

    if platform == "linux" or platform == "linux2":
        pdf_cmd_default = r'okular {} &'
    elif platform == "darwin":
        pdf_cmd_default = r'open -a preview {}'
    elif platform == "win32":
        pdf_cmd_default = r'AcroRd32.exe {}'
    prompt = ("Enter command to view PDF with {{}} as file name placeholder "
    "[{}] > ".format(pdf_cmd_default))
    if pdf_cmd is None:
        pdf_cmd = input(prompt)
    if pdf_cmd == '':
        pdf_cmd = pdf_cmd_default

    c = ConfigParser()
    c['DEFAULT'] = {'data_cache_path': data_cache_path, 
                    'pdf_cmd': pdf_cmd}

    # SweetCat location and update interval in seconds
    url = 'https://sweetcat.iastro.pt/catalog/SWEETCAT_Dataframe.csv'
    c['SWEET-Cat'] = {'update_interval': 86400, 'download_url': url}
    
    # TEPCat location and update interval in seconds
    url = 'https://www.astro.keele.ac.uk/jkt/tepcat/allplanets-csv.csv' 
    c['TEPCat'] = {'update_interval': 86400, 'download_url': url}
    url = 'https://www.astro.keele.ac.uk/jkt/tepcat/observables.csv' 
    c['TEPCatObs'] = {'update_interval': 86400, 'download_url': url}

    #N.B. The archive username and password are stored in plain text so the
    #default mode value is 0o600 = user read/write permission only.

    # Reference PSF file
    psf_file = 'CHEOPS_IT_PSFwhite_CH_TU2018-01-01.txt'
    c['psf_file'] = {'psf_file': psf_file, 'x0':99.5, 'y0':99.5}
    
    with open(os.open(configFile, os.O_CREAT | os.O_WRONLY, mode), 'w') as cf:
        c.write(cf)

