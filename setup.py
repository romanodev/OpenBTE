from setuptools import setup,find_packages
import os


setup(name='openbte',
      version='0.9.19',
      description='Boltzmann Transport Equation for Phonons',
      author='Giuseppe Romano',
      author_email='romanog@mit.edu',
      classifiers=['Programming Language :: Python :: 2.7',\
                   'Programming Language :: Python :: 3.6'],
      long_description=open('README.rst').read(),
      install_requires=['numpy', 
                        'scipy',
                        'shapely',
                        'h5py',
                        'pyvtk',
                        'unittest2',
                        'termcolor',
                        'pyclipper',
                        'future',
                        'deepdish',
                        'mpi4py',
                        'sympy',
                        'matplotlib',         
                         ],
      license='GPLv2',\
      packages = ['openbte'],
      package_data = {'openbte':['materials/*.dat']},
      #packages = find_packages(exclude=['openbte_new']),
      entry_points = {
     'console_scripts': [
      'openbte=openbte.__main__:main','shengbte2openbte=openbte.shengbte2openbte:main',
      'download_openbte_example=openbte.download_openbte_example:main'],
      },
      zip_safe=False)
