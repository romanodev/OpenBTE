from setuptools import setup,find_packages
import os

setup(name='openbte',
      version='2.46',
      description='Boltzmann Transport Equation for Phonons',
      author='Giuseppe Romano',
      author_email='romanog@mit.edu',
      classifiers=['Programming Language :: Python :: 3.6'],
      #long_description=open('README.rst').read(),
      install_requires=['shapely',
                        'pandas',
                        'docutils==0.16',
                        'dash',
                        'unittest2',
                        'nbsphinx',
                        'pytest',
                        'termcolor',
                        'matplotlib',
                        'alabaster',
                        'cachetools',
                        'mpi4py',
                        'numpy',
                        'scipy',
                        'dash_core_components',
                        'dash_html_components',
                        'dash_bootstrap_components',
                        'plotly >=4.14.0',
                        'nbsphinx',
                        'recommonmark',
                        'sphinx',
                        'sphinx_rtd_theme'
                         ],
      license='GPLv3',\
      packages = ['openbte'],
      package_data = {'openbte':['materials/*.npz','efunx.yaml']},
      entry_points = {
     'console_scripts': ['AlmaBTE2OpenBTE=openbte.almabte2openbte:main',\
                         'AlmaBTE2OpenBTE2=openbte.almabte2openbte_new:main',\
                         'Phono3py2OpenBTE=openbte.phono3py2openbte:main',\
                         'rta2mfp=openbte.rta2mfp:main',\
                         'bundle_data=openbte.bundle_data:main',\
                         'gui=openbte.gui:main',\
                         'app=openbte.app:App',\
                         'OpenBTE=openbte.openbte:main'],
      },
      zip_safe=False)
