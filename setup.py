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
                        'termcolor',
                        'mpi4py',
                        'matplotlib',         
                         ],
      license='MIT',\
      #scripts=['src/convertShengBTE.py'],\
      packages = find_packages('openbte',exclude=['solver_full.py',\
                                                  'fourier_graded.py',\
                                                  'GenerateInterface2D.py',\
                                                  'solver_interface.py',\
                                                  'nanowire.py',\
                                                  'GenerateRandomPores.py',\
                                                  'GenerateRandomPoresOverlap.py',\
                                                  'solver_old.py',\
                                                  'fourier.py']),
   #   packages = ['openbte',exclude=['solver_full.py']],
      #data_files = datafiles,
      entry_points = {
     'console_scripts': [
      'openbte=openbte.__main__:main','shengbte2openbte=openbte.shengbte2openbte:main'],
      },
      zip_safe=False)
