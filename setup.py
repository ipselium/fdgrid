#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2016-2019 Cyril Desjouy <cyril.desjouy@univ-lemans.fr>
#
# This file is part of fdgrid
#
# fdgrid is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# fdgrid is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with fdgrid. If not, see <http://www.gnu.org/licenses/>.
#
#
# Creation Date : 2018-04-10 - 17:52:42
"""
-----------

fdgrid setup file.

@author: Cyril Desjouy
"""

from setuptools import setup, find_packages

setup(

    name='fdgrid',
    description="Grid generator",
    long_description=open('README.rst').read(),
    long_description_content_type='text/x-rst',
    version='0.8.7',
    license="GPL",
    url='http://github.com/ipselium/fdgrid',
    author="Cyril Desjouy",
    author_email="cyril.desjouy@univ-lemans.fr",
    packages=find_packages(),
    include_package_data=True,
    install_requires=['numpy', 'scipy', 'matplotlib', 'ofdlib2', 'mplutils'],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
    ]
)
