"""
 Description
 ----------------------------
 A Python implementation of the Random Sample Reliability (RSR) algorithm as
 developed in Okasa & Younge (2022). The RSR estimates the reliability scores
 for each observation within the sample. The RSR reliability scores reflect
 the reverse of an average estimation loss resulting from a resampling
 procedure. These scores can be further used for an annealing sensitivity
 analysis to assess the impact of unreliable observations, or for a weighted
 regression or a consensus estimation to reduce the impact of less reliable
 observations.

 Installation
 ----------------------------
 
 To clone the repo for the `samplefit` library run:

 ```
 git clone https://github.com/okasag/samplefit.git
 ```

 `samplefit` relies on Python 3 and requires the following dependencies:
     
 * numpy (>=1.22.0)
 * pandas (>=1.3.5)
 * scipy (>=1.7.2)
 * statsmodels (>=0.12.2)
 * matplotlib (>=3.4.2)
 
 The required modules can be installed by navigating to the root of the cloned
 project and executing the following command:
 `pip install -r requirements.txt`. 

 Examples
 ----------------------------

 The example below demonstrates the workflow of using the `samplefit` library
 in conjunction with the well-known `statsmodels` library.

 Import libraries:
 ```python
 import samplefit as sf
 import statsmodels.api as sm
 ```

 Get data:
 ```python
 boston = sm.datasets.get_rdataset("Boston", "MASS")
 Y = boston.data['medv']
 X = boston.data['rm']
 X = sm.add_constant(X)
 ```

 Assess model fit:
 ```python
 model = sm.OLS(endog=Y, exog=X)
 model_fit = model.fit()
 model_fit.summary()
 ```

 Assess sample fit:
 ```python
 sample = sf.RSR(model=model)
 sample_fit = sample.fit()
 sample_fit.summary()
 ```

 Assess sample reliability:
 ```python
 sample_scores = sample.score()
 sample_scores.plot()
 ```

 Assess sample sensitivity:
 ```python
 sample_annealing = sample.anneal()
 sample_annealing.plot()
 ```

 Authors
 ----------------------------
 Gabriel Okasa & Kenneth A. Younge

 References
 ----------------------------
 - Okasa, Gabriel, and Kenneth A. Younge. ???Random Sample Reliability.???
 Working Paper. 2022.
 - Seabold, Skipper, and Josef Perktold. ???statsmodels: Econometric and 
 statistical modeling with python.??? Proceedings of the 9th Python in Science 
 Conference. 2010.
"""

from samplefit.Reliability import RSR
from samplefit.Reliability import RSRFitResults
from samplefit.Reliability import RSRAnnealResults
from samplefit.Reliability import RSRScoreResults

__all__ = ["RSR", "RSRFitResults", "RSRAnnealResults", "RSRScoreResults"]
__version__ = "0.0.9000"
__module__ = 'samplefit'
__author__ = "Gabriel Okasa & Kenneth A. Younge"
__copyright__ = "Copyright (c) 2022, Gabriel Okasa & Kenneth A. Younge"
__license__ = "MIT License"
