from setuptools import setup,find_packages
import os

datafiles= []
rootDir = 'examples'
for dirName, subdirList, fileList in os.walk(rootDir):
    for fname in fileList:
     datafiles.append(dirName + '/' + fname)


rootDir = 'materials'
for dirName, subdirList, fileList in os.walk(rootDir):
    for fname in fileList:
     datafiles.append(dirName + '/' + fname)


setup(name='openbte',
      version='0.9.8',
      description='Boltzmann Transport Equation for Phonons',
      author='Giuseppe Romano',
      author_email='romanog@mit.edu',
      classifiers=['Programming Language :: Python :: 2.7'],
      long_description=open('README.rst').read(),
      install_requires=['numpy', 
                        'scipy',
                        'shapely',
                        'h5py',
                        'pyvtk',
                        'termcolor',
                        'mpi4py',
                        'matplotlib',         
                         ],
      license='GPLv2',\

      packages = find_packages(exclude=['openbte_new']),
      entry_points = {
     'console_scripts': [
      'openbte=openbte.__main__:main','shengbte2openbte=openbte.shengbte2openbte:main'],
      },
      zip_safe=False)
