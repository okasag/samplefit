# `samplefit`: Random Sample Reliability

`samplefit` is a Python library to assess sample fit, as opposed to model fit, via the Random Sample Reliability algorithm as developed by Okasa & Younge (2022). `samplefit` is built upon the `statsmodels` library (Seabold & Perktold, 2010) and follows the same command workflow.

## Introduction

Researchers frequently test model fit by holding data constant and varying the model. We propose Random Sample Reliability (RSR) as a computational framework to test sample fit by holding the model constant and varying the data. Random Sample Reliability re-samples data to estimate the reliability of observations of a sample. RSR can be used to score the reliability of every observation within the sample, test the sensitivity of results to atypical observations via annealing procedure, and estimate a weighted fit where the analysis is more robust.

## Installation

To install the latest `PyPi` released version run

```
pip install samplefit
```

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

## References

- Okasa, Gabriel, and Kenneth A. Younge. “Sample Fit: Random Sample Reliability.” arXiv preprint arXiv:xxxx.xxxxx. 2022.
- Seabold, Skipper, and Josef Perktold. “statsmodels: Econometric and statistical modeling with python.” Proceedings of the 9th Python in Science Conference. 2010.
