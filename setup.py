from setuptools import setup

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()


setup(
  name = 'samplefit',
  packages = ['samplefit'],
  version = '0.3.1',
  license = 'MIT',
  description = 'samplefit library implements the Sample Fit Reliability algorithm for the assessment of sample fit in econometric models.',
  long_description_content_type='text/markdown',
  long_description=long_description,
  author = 'Gabriel Okasa and Kenneth A. Younge',
  author_email = 'okasa.gabriel@gmail.com',
  url = 'https://github.com/okasag/samplefit',
  download_url = 'https://github.com/okasag/samplefit/archive/v0.3.1.tar.gz',
  keywords = ['sample fit', 'reliability', 'econometrics'],
  install_requires=[
          'numpy>=1.22.0',
          'pandas>=1.3.5',
          'scipy>=1.7.2',
          'statsmodels>=0.12.2',
          'matplotlib>=3.4.2',
          'joblib>=1.0.1',
          'psutil>=5.8.0',
      ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Science/Research',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
  ],
)
