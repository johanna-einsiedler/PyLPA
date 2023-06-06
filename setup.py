from setuptools import setup
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()
setup(name='py-latent-profiles',
      version='0.3',
      description='A package for running gaussian mixture models and latent profile analyses in Python',
      url='https://github.com/johanna-einsiedler/PyGMM',
      author='Johanna Einsiedler',
      author_email='johanna.einsiedler@sodas.ku.dk',
      license='MIT',
      packages=['py_lpa'],
      install_requires=[
          'numpy','scipy','tqdm'
      ],
        # other arguments omitted
    long_description=long_description,
    long_description_content_type='text/markdown',
      zip_safe=False)
