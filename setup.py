from setuptools import setup,find_packages

setup(name='openbte',
      version='3.0.3',
      description='Boltzmann Transport Equation for Phonons',
      author='Giuseppe Romano',
      author_email='romanog@mit.edu',
      classifiers=['Programming Language :: Python :: 3.8'],
      install_requires=['shapely',
                        'numpy',
                        'sqlitedict',
                        'cachetools',
                        'sqlitedict',
                        'icosphere',
                        'nptyping',
                        'pickle5',
                        'plotly',
                        'jax',
                        'jaxlib',
                        'nlopt',
                        'scikit-image',
                        'matplotlib',
                        'scipy',
                        'gmsh'],
      license='GPLv3',\
      packages = find_packages(),
      entry_points = {
     'console_scripts': ['AlmaBTE2OpenBTE=openbte.almabte2openbte:almabte2openbte']},
      package_data = {'openbte':['materials/*.db']},
      include_package_data=True
      )
