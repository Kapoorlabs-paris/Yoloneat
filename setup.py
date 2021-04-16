#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 14:28:22 2020

@author: aimachine
"""

from __future__ import absolute_import
from setuptools import setup, find_packages
from os import path

_dir = path.abspath(path.dirname(__file__))

with open(path.join(_dir,'version.py')) as f:
    exec(f.read())

with open(path.join(_dir,'README.md')) as f:
    long_description = f.read()


setup(name='ONT',
      version=__version__,
      description='ONEAT-Online Offline network for event aware topology detection (ONEAT)',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='',
      author='Varun Kapoor',
      author_email='vkapoor@curie.fr, randomaccessiblekapoor@gmail.com',
      license='BSD 3-Clause License',
      packages=find_packages(),

      project_urls={
          'Documentation': '',
          'Repository': 'https://github.com/kapoorlab/ONT',
      },

      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering',
          'License :: OSI Approved :: BSD License',

          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
      ],

      install_requires=[
          "numpy",
          "scipy",
          "matplotlib",
          "six",
          "keras>=2.1.2,<2.4",
          "h5py",
          "imagecodecs-lite<=2020; python_version<'3.6'",
          "tifffile",
          "tqdm",
          "elasticdeform"
          "pathlib2; python_version<'3'",
          "backports.tempfile; python_version<'3.4'",
      ]
      )