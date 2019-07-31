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
import  os 
from configparser import ConfigParser
import getpass

__all__ = ['load_config', 'setup_config', 'get_cache_path']


def load_config(configFile=None):
    """
    Load module configuration from configFile 
    
    If configFile is None, look for pycheops.cfg in the user's home directory

    :param configFile: Full path to configuration file

    """

    if configFile is None:
        configFile = os.path.expanduser(os.path.join('~','pycheops.cfg'))

    if not os.path.isfile(configFile):
        raise ValueError('Configuration file not found - run setup_config')

    c = ConfigParser()
    c.read(configFile)
    return c


def setup_config(configFile=None, overwrite=False, mode=0o600):
    """
    Create module configuration 
    
    If configFile is None, use pycheops.cfg in the user's home directory

    N.B. The archive username and password are stored in plain text so the
    default mode value is 0o600 = user read/write permission only.

    :param configFile: Full path to configuration file

    :param overwrite: overwrite values in existing configFile

    :param mode: mode (permission) settings for configFile

    """

    if configFile is None:
        configFile = os.path.expanduser(os.path.join('~','pycheops.cfg'))
    print('Creating configuration file {}'.format(configFile))

    if os.path.isfile(configFile) and not overwrite:
        raise ValueError('Configuration file exists and overwrite is not set')

    data_cache_default = os.path.expanduser(os.path.join('~','pycheops_data'))
    prompt = "Enter data cache directory [{}] > ".format(data_cache_default)
    data_cache_path = input(prompt)
    if data_cache_path is '':
        data_cache_path = data_cache_default
    if not os.path.isdir(data_cache_path):
        os.mkdir(data_cache_path)

    default_username = getpass.getuser()
    prompt = "Enter CHEOPS archive username [{}] > ".format(default_username)
    username = input(prompt)
    if username is '':
        username = default_username

    password = getpass.getpass("Enter CHEOPS archive password > ")
    c = ConfigParser()
    c['DEFAULT'] = {'data_cache_path': data_cache_path,
            'archive_username': username,
            'archive_password': password}
    with open(os.open(configFile, os.O_CREAT | os.O_WRONLY, mode), 'w') as cf:
        c.write(cf)

