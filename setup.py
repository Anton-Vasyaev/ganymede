import os
import io
import re

from setuptools import setup, find_packages, find_namespace_packages

def get_version():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    version_file = os.path.join(current_dir, 'src', 'ganymede', '__init__.py')
    with io.open(version_file, encoding='utf-8') as f:
        return re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', f.read(), re.M).group(1)

setup(
    name='ganymede',
    version=get_version(),
    packages=find_packages(where='src'),
    package_dir={'': 'src'}
)