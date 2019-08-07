from setuptools import setup,find_packages
import os

setup(name='openbte',
      version='0.9.63',
      description='Boltzmann Transport Equation for Phonons',
      author='Giuseppe Romano',
      author_email='romanog@mit.edu',
      classifiers=['Programming Language :: Python :: 3.6'],
      #long_description=open('README.rst').read(),
      install_requires=['numpy',
                        'scipy',
                        'sparse',
                        'shapely',
                        'pyvtk',
                        'unittest2',
                        'ipython',
                        'future',
                        'termcolor',
                        'alabaster',
                        'deepdish',
                        'mpi4py==3.0.0',
                        'matplotlib',
                         ],
      license='GPLv2',\
      packages = ['openbte'],
      package_data = {'openbte':['materials/*.dat']},
      entry_points = {
     'console_scripts': [
      'openbte=openbte.__main__:main','shengbte2openbte=openbte.shengbte2openbte:main'],
      },
      zip_safe=False)
