# `samplefit`: Random Sample Reliability

`samplefit` is a Python library to assess sample fit, as opposed to model fit, via the Random Sample Reliability algorithm as developed by Okasa & Younge (2022). `samplefit` is built upon the `statsmodels` library (Seabold & Perktold, 2010) and follows the same command workflow.

Copyright (c) 2022 Gabriel Okasa & Kenneth A. Younge.

	AUTHOR:  Gabriel Okasa & Kenneth A. Younge
	SOURCE:  https://github.com/okasag/samplefit
	LICENSE: Access to this code is provided under an MIT License.

Repo maintainer: Gabriel Okasa ([gabriel.okasa@epfl.ch](mailto:gabriel.okasa@epfl.ch))

## Introduction

Researchers frequently test model fit by holding data constant and varying the model. We propose Random Sample Reliability (RSR) as a computational framework to test sample fit by holding the model constant and varying the data. Random Sample Reliability re-samples data to estimate the reliability of observations of a sample. RSR can be used to score the reliability of every observation within the sample, test the sensitivity of results to atypical observations via annealing procedure, and estimate a weighted fit where the analysis is more robust.

Detailed documentation of the `samplefit` library is available [here](https://okasag.github.io/samplefit/).

Poster describing the RSR algorithm is available [here](https://okasag.github.io/assets/pdf/Okasa_Younge_RSR_Poster_SciPy.pdf).

## Installation

To clone this repo for the `samplefit` library run:

```
git clone https://github.com/okasag/samplefit.git
```

The required modules can be installed by navigating to the root of this project and
executing the following command: `pip install -r requirements.txt`.

## Example

The example below demonstrates the workflow of using the `samplefit` library in conjunction with the well-known `statsmodels` library.

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
sample = sf.RSR(linear_model=model)
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

## References

- Okasa, Gabriel, and Kenneth A. Younge. “Random Sample Reliability.” Working Paper. 2022.
- Seabold, Skipper, and Josef Perktold. “statsmodels: Econometric and statistical modeling with python.” Proceedings of the 9th Python in Science Conference. 2010.
