"""

 `samplefit` is a Python library to assess sample fit, as opposed to model fit,
 via the *Sample Fit Reliability* algorithm as developed by Okasa & Younge (2022).
 `samplefit` is linked to the `statsmodels` library (Seabold & Perktold, 2010)
 and follows the same command workflow.

 
 Description
 ----------------------------
 `samplefit` is a Python library for the assessment of sample fit in
 econometric models. In particular, `samplefit` implements the Sample Fit
 Reliability (SFR) algorithm, a re-sampling procedure to estimate the
 reliability of data and check the sensitivity of results. To that end,
 SFR is a computational approach with three aspects: *Scoring*, to estimate a 
 point-wise reliability score for every observation in a sample based on the
 expected estimation loss over sub-samples; *Annealing*, to test the sensitivity
 of results to the sequential removal of unreliable data points; and *Fitting*,
 to estimate a weighted regression that adjusts for the reliability of the data.

 Installation
 ----------------------------
 
 To install the `samplefit` library from `PyPi` run:

 ```
 pip install samplefit
 ```
 
 or alternatively, to clone the repo run:

 ```
 git clone https://github.com/okasag/samplefit.git
 ```

 `samplefit` relies on Python 3 and requires the following dependencies:
     
 * numpy (>=1.22.0)
 * pandas (>=1.3.5)
 * scipy (>=1.7.2)
 * statsmodels (>=0.12.2)
 * matplotlib (>=3.4.2)
 * joblib (>=1.0.1)
 * psutil (>=5.8.0)
 
 The required modules can be installed by navigating to the root of
 the cloned project and executing the following command:
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
 Y = boston.data['crim']
 X = boston.data['lstat']
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
 sample = sf.SFR(model=model)
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
 - Okasa, Gabriel, and Kenneth A. Younge. “Sample Fit.” Working Paper. 2022.
 - Seabold, Skipper, and Josef Perktold. “statsmodels: Econometric and 
 statistical modeling with python.” Proceedings of the 9th Python in Science 
 Conference. 2010.
"""

from samplefit.Reliability import SFR
from samplefit.Reliability import SFRFitResults
from samplefit.Reliability import SFRAnnealResults
from samplefit.Reliability import SFRScoreResults

__all__ = ["SFR", "SFRFitResults", "SFRAnnealResults", "SFRScoreResults"]
__version__ = "0.3.1"
__module__ = 'samplefit'
__author__ = "Gabriel Okasa & Kenneth A. Younge"
__copyright__ = "Copyright (c) 2022, Gabriel Okasa & Kenneth A. Younge"
__license__ = "MIT License"
