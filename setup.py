from setuptools import setup,find_packages
import os

setup(name='openbte',
      version='1.122',
      description='Boltzmann Transport Equation for Phonons',
      author='Giuseppe Romano',
      author_email='romanog@mit.edu',
      classifiers=['Programming Language :: Python :: 3.6'],
      #long_description=open('README.rst').read(),
      install_requires=['shapely',
                        'pyvtk',
                        'googledrivedownloader',
                        'unittest2',
                        'nbsphinx',
                        'future',
                        'pytest',
                        'pubeasy',
                        'termcolor',
                        'alabaster',
                        'cachetools',
                        'mpi4py',
                        'plotly',
                        'numpy',
                        'dash_core_components',
                        'dash',
                        'dash_html_components',
                        'dash_bootstrap_components',
                        'nbsphinx',
                        'recommonmark',
                        'sphinx',
                        'sphinx_rtd_theme'
                         ],
      license='GPLv3',\
      packages = ['openbte'],
      package_data = {'openbte':['materials/*.npz']},
      entry_points = {
     'console_scripts': ['AlmaBTE2OpenBTE=openbte.almabte2openbte:main',\
                         'Phono3py2OpenBTE=openbte.phono3py2openbte:main',\
                         'rta2mfp=openbte.rta2mfp:main',\
                         'bundle_data=openbte.bundle_data:main',\
                         'gui=openbte.gui:main',\
                         'app=openbte.app:App',\
                         'OpenBTE=openbte.openbte:main'],
      },
      zip_safe=False)
