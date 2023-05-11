# python
import os
import io
import re
import os.path as p
# 3rd party
from pathlib import Path
from setuptools import setup, Distribution, Extension, find_packages
from setuptools.command.build_ext import build_ext as build_ext_orig
from distutils.cmd import Command


def get_version():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    version_file = os.path.join(current_dir, 'ganymede.proj', 'ganymede', '__init__.py')
    with io.open(version_file, encoding='utf-8') as f:
        return re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', f.read(), re.M).group(1)


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


LONG_DESCRIPTION = ''
with open('README.md') as fh:
    LONG_DESCRIPTION = fh.read()
    

setup(
    name = 'ganymede-aux',
    version = get_version(),
    description = 'Auxiliary library for computer vision and deep learning',
    author = 'Anton Vasyev',
    author_email = 'vasyaevanton@gmail.com',
    license = 'MIT',
    keywords = '',
    url = 'https://github.com/Anton-Vasyaev/ganymede',
    install_requires = [
        'autofast>=0.2.1',
        'Pillow>=9.0.0',
        'pywin32>=306'
    ],
    requires_python='>=3.9.0',
    packages = find_packages(where='ganymede.proj'),
    package_dir = {'': 'ganymede.proj'},
    package_data = {'': ['*.dll', '*.so']},
    distclass=BinaryDistribution
)