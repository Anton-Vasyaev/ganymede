# python
import os
import io
import re
import os.path as p
# 3rd party
from pathlib import Path
from setuptools import setup, Distribution, Extension, find_packages
from setuptools.command.build_ext import build_ext as build_ext_orig
from setuptools import setup, find_packages, find_namespace_packages


def get_version():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    version_file = os.path.join(current_dir, 'ganymede.proj', 'ganymede', '__init__.py')
    with io.open(version_file, encoding='utf-8') as f:
        return re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', f.read(), re.M).group(1)


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


class CMakeConanExtension(Extension):

    def __init__(
        self, 
        name, 
        conan_update_dir, 
        target
    ):
        # don't invoke the original build_ext for this special extension
        super().__init__(name, sources=[])

        self.conan_update_dir = conan_update_dir
        self.target           = target


class build_ext(build_ext_orig):

    def run(self):
        for ext in self.extensions:
            self.build_cmake(ext)
        super().run()


    def build_cmake(self, ext):
        setup_path = p.abspath(p.dirname(__file__))

        # these dirs will be created in build_py, so if you don't have
        # any python sources to bundle, the dirs will be missing
        build_temp = Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)
        extdir = Path(self.get_ext_fullpath(ext.name))
        extdir.mkdir(parents=True, exist_ok=True)

        # update conan
        conan_update_dir = p.abspath(ext.conan_update_dir)
        os.chdir(conan_update_dir)
        if os.name == 'nt':
            self.spawn(['conan_update_x64.bat'])
        elif os.name == 'posix':
            self.spawn(['conan_update_x64.sh'])
        else:
            raise Exception(f'Not expected os:{os.name}')
        os.chdir(setup_path)

        
        # cmake
        cmake_dir_p = Path(ext.name).absolute()
        cmake_build_dir_p = cmake_dir_p / 'build'

        # configuration
        cmake_args = [
            '--no-warn-unused-cli',
            '-DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE',
            '-DCMAKE_BUILD_TYPE:STRING=Release',
            '-S', str(cmake_dir_p), 
            '-B', str(cmake_build_dir_p)
        ]
        self.spawn(['cmake'] + cmake_args)

        # build
        cmake_args = [
            '--build', str(cmake_build_dir_p),
            '--config', 'Release',
            '--target', ext.target
        ]
        self.spawn(['cmake'] + cmake_args)

        os.chdir(setup_path)


setup(
    name='ganymede',
    version=get_version(),
    packages=find_packages(where='ganymede.proj'),
    package_dir={'': 'ganymede.proj'},
    package_data={'': ['*.dll']},
    distclass=BinaryDistribution,
    ext_modules=[
        CMakeConanExtension('auxml.export.proj', 'submodules/auxml', 'auxml_export')
    ],
    cmdclass={
        'build_ext': build_ext,
    }
)