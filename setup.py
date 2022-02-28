from setuptools import setup,find_packages
import os

setup(name='openbte',
      version='2.71.0',
      description='Boltzmann Transport Equation for Phonons',
      author='Giuseppe Romano',
      author_email='romanog@mit.edu',
      classifiers=['Programming Language :: Python :: 3.6'],
      #long_description=open('README.rst').read(),
      install_requires=['shapely',
                        'pandas',
                        'docutils==0.16',
                        'numpy',
                        'dash',
                        'unittest2',
                        'nbsphinx',
                        'pytest',
                        'termcolor',
                        'matplotlib',
                        'alabaster',
                        'cachetools',
                        'mpi4py',
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
      package_data = {'openbte':['materials/*.npz']},
      include_package_data=True,
      entry_points = {
     'console_scripts': ['AlmaBTE2OpenBTE=openbte.almabte2openbte:main',\
                         'Phono3py2OpenBTE=openbte.phono3py2openbte:main']\
      },
      zip_safe=True)
