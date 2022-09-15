[![PyPI version](https://img.shields.io/pypi/v/samplefit.svg)](https://pypi.org/project/samplefit/)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/samplefit.svg)](https://pypi.org/project/samplefit/)
[![Licence](https://img.shields.io/pypi/l/samplefit.svg)](https://pypi.org/project/samplefit/)

# `samplefit`

`samplefit` is a Python library to assess sample fit, as opposed to model fit, via the *Sample Fit Reliability* algorithm as developed by Okasa & Younge (2022). `samplefit` is linked to the `statsmodels` library (Seabold & Perktold, 2010) and follows the same command workflow.

Copyright (c) 2022 Gabriel Okasa & Kenneth A. Younge.

	AUTHOR:  Gabriel Okasa & Kenneth A. Younge
	SOURCE:  https://github.com/okasag/samplefit
	LICENSE: Access to this code is provided under an MIT License.

Repo maintainer: Gabriel Okasa ([okasa.gabriel@gmail.com](mailto:okasa.gabriel@gmail.com))

## Introduction

`samplefit` is a Python library for the assessment of sample fit in
econometric models. In particular, `samplefit` implements the Sample Fit
Reliability (SFR) algorithm, a re-sampling procedure to estimate the reliability
of the data, check the sensitivity of the results, and improve the robustness of
the analysis. To that end, SFR is a computational approach with three aspects:
*Scoring*, to estimate a reliability score for every observation in a sample
based on the expected estimation loss over sub-samples; *Annealing*, to test the
sensitivity of results to the sequential removal of unreliable data; and *Fitting*,
to estimate a weighted regression that adjusts for the reliability of the sample.

Detailed documentation of the `samplefit` library is available [here](https://okasag.github.io/samplefit/).

Paper describing the SFR algorithm is available [here](https://arxiv.org/pdf/2209.06631.pdf).

## Installation

To install the `samplefit` library from `PyPi` run:

```
pip install samplefit
```

or alternatively, to clone the repo run:

```
git clone https://github.com/okasag/samplefit.git
```

The required modules can be installed by navigating to the root of
the cloned project and executing the following command:
`pip install -r requirements.txt`. 

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
sample = sf.SFR(linear_model=model)
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

- Okasa, Gabriel, and Kenneth A. Younge. “Sample Fit Reliability.” arXiv preprint arXiv:2209.06631. <https://arxiv.org/abs/2209.06631>. 2022.
- Seabold, Skipper, and Josef Perktold. “statsmodels: Econometric and statistical modeling with python.” Proceedings of the 9th Python in Science Conference. 2010.
