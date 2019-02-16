from setuptools import setup,find_packages
import os

setup(name='openbte',
      version='0.9.37',
      description='Boltzmann Transport Equation for Phonons',
      author='Giuseppe Romano',
      author_email='romanog@mit.edu',
      classifiers=['Programming Language :: Python :: 2.7',\
                   'Programming Language :: Python :: 3.6'],
      long_description=open('README.rst').read(),
      install_requires=['numpy',
                        'scipy',
                        'sparse',
                        'shapely',
                        'pyvtk',
                        'unittest2',
                        'twisted',
                        'ipython',
                        'grin',
                        'termcolor',
                        'future',
                        'alabaster',
                        'deepdish',
                        'mpi4py',
                        'networkx',
                        'matplotlib',
                         ],
      license='GPLv2',\
      packages = ['openbte'],
      package_data = {'openbte':['materials/*.dat','../extern/gmsh']},
      #packages = find_packages(exclude=['openbte_new']),
      entry_points = {
     'console_scripts': [
      'openbte=openbte.__main__:main','shengbte2openbte=openbte.shengbte2openbte:main'],
      },
      zip_safe=False)
