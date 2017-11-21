from setuptools import setup
import os

datafiles= []
rootDir = 'openbte/examples'
for dirName, subdirList, fileList in os.walk(rootDir):
    #print('Found directory: %s' % dirName)
    for fname in fileList:
     datafiles.append(dirName + '/' + fname)


rootDir = 'openbte/materials'
for dirName, subdirList, fileList in os.walk(rootDir):
    #print('Found directory: %s' % dirName)
    for fname in fileList:
     datafiles.append(dirName + '/' + fname)



setup(name='openbte',
      version='0.1',
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
      scripts=['openbte/convertShengBTE.py'],\
      packages=['openbte'],
      data_files = datafiles,
      entry_points = {
     'console_scripts': [
      'openbte=openbte.openbte:main'],
      },
      zip_safe=False)
