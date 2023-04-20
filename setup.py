from setuptools import setup

setup(name='py_lpa',
      version='0.1',
      description='A package for running gaussian mixture models and latent profile analyses in Python',
      url='https://github.com/johanna-einsiedler/PyGMM',
      author='Johanna Einsiedler',
      author_email='johanna.einsiedler@sodas.ku.dk',
      license='MIT',
      packages=['py_lpa'],
      install_requires=[
          'numpy','scipy','tqdm'
      ],
      zip_safe=False)
