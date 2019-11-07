from setuptools import setup

setup(name='estaudel',
      version='0.1',
      description='Nested Darwinian population simulations',
      url='https://gitlab.com/ecoevomath/estaudel',
      author='Guilhem Doulcier',
      author_email='guilhem.doulcier@ens.fr',
      license='GPLv3+',
      python_requires='>=3',
      packages=['estaudel'],
      install_requires=[
          'numpy',
          'scipy',
          'pandas',
          'matplotlib',
          'pycairo',
          'Pillow'],
      zip_safe=False)
