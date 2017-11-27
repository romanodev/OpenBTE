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
      version='0.9.0',
      description='Boltzmann Transport Equation for Phonons',
      author='Giuseppe Romano',
      author_email='romanog@mit.edu',
      install_requires=['numpy', 
                        'scipy',
                        'shapely',
                        'h5py',
                        'pyvtk',
                        'mpi4py',
                        'matplotlib',         
                         ],
      license='MIT',\
      #scripts=['src/convertShengBTE.py'],\
      packages = find_packages(),
      #data_files = datafiles,
      entry_points = {
     'console_scripts': [
      'openbte=src.__main__:main'],
      },
      zip_safe=False)
