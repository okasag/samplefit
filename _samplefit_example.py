# samplefit example #

# import libraries
import samplefit as sf
import statsmodels.api as sm

# get data
boston = sm.datasets.get_rdataset("Boston", "MASS")
Y = boston.data['crim'] # crime rate
X = boston.data['lstat'] # % of lower status
X = sm.add_constant(X)

# assess model fit
model = sm.OLS(endog=Y, exog=X)
model_fit = model.fit()
model_fit.summary()

# assess sample fit
sample = sf.RSR(linear_model=model)
sample_fit = sample.fit()
sample_fit.summary()

# assess sample sensitivity
sample_annealing = sample.anneal()
sample_annealing.plot()

# assess sample reliability
sample_scores = sample.score()
sample_scores.plot()
