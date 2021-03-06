"""
samplefit: Random Sample Reliability.

Python library to assess Sample Fit via the Random Sample Reliability
algorithm as developed by Okasa & Younge (2022).

Definition of base RSR class and fit, score and anneal methods.

"""

# import statsmodels and samplefit
import statsmodels
import samplefit.Reliability as Reliability

# import modules
import numpy as np # (hast to be 1.22.0 at least, due to np.percentile changes)
import pandas as pd

# import submodules and functions
from scipy import stats
# TODO: from joblib import Parallel, delayed, parallel_backend
from multiprocessing import cpu_count, Lock
# TODO: add check_random_state from statsmodels 0.14.0


# %% RSR Class definition
# define BaseRSR class
class BaseRSR:
    """
    Base RSR class of samplefit.
    This class should not be used directly. Use derived classes instead.
    """

    # define init function
    def __init__(self,
                 linear_model=None,
                 n_samples=1000,
                 min_samples=None,
                 loss=None,
                 boost=None,
                 n_jobs=-1,
                 random_state=None):

        # assign input values
        self.linear_model = linear_model
        self.n_samples = n_samples
        self.min_samples = min_samples
        self.loss = loss
        self.boost = boost
        self.n_jobs = n_jobs
        self.random_state = random_state
        
        # check the self inputs and set defaults
        self._input_checks()


    def _input_checks(self):
        """Input checks for the RSR class init."""
        
        # check and define the input parameters
        linear_model = self.linear_model
        n_samples = self.n_samples
        min_samples = self.min_samples
        loss = self.loss
        boost = self.boost
        n_jobs = self.n_jobs
        random_state = self.random_state

        # check if the model is supported (currently only linear model)
        # define supported models (Gaussian)
        family_set = [statsmodels.genmod.families.family.Gaussian]

        # check the model input
        if isinstance(linear_model,
                      (statsmodels.genmod.generalized_linear_model.GLM,
                       statsmodels.regression.linear_model.OLS)):
            # if GLM, check if its Gaussian
            if isinstance(linear_model,
                          (statsmodels.genmod.generalized_linear_model.GLM)):
                # check if GLM family is supported
                if type(linear_model.family) in family_set:
                    # assign the value
                    self.linear_model = linear_model
                else:
                    # raise type error
                    raise TypeError(f'{type(linear_model.family)} is not '
                                    'supported. '
                                    f'Only: {family_set} are available.')
            else:
                # assign the value (OLS class has same structure as GLM class)
                self.linear_model = linear_model
        else:
            # raise value error
            raise ValueError('linear_model must be a class of '
                             'statsmodels.regression.linear_model.OLS or '
                             'statsmodels.genmod.generalized_linear_model.GLM'
                             f', got {type(linear_model)}.')

        # get model inputs
        # get exog and endog
        self.exog = self.linear_model.data.exog
        self.endog = self.linear_model.data.endog
        
        # initiliaze exog names, observations, etc
        self.exog_names = self.linear_model.data.cov_names
        self.n_obs = int(self.linear_model.nobs)
        self.n_exog = self.linear_model.exog.shape[1]

        # check the number of subsampling iterations
        if isinstance(n_samples, int):
            # check if its at least 1
            if n_samples >= 1:
                # assign the input value
                self.n_samples = n_samples
            else:
                # raise value error
                raise ValueError("n_samples must be at least 1"
                                 ", got %s" % n_samples)
        else:
            # raise value error
            raise ValueError("n_samples must be an integer"
                             ", got %s" % n_samples)

        # subsampling fraction
        if min_samples is None:
            # take as default minimum number to estimate the model
            self.min_samples = self.n_exog + 1
        # otherwise chekc if its float or integer
        elif isinstance(min_samples, (float, int)):
            # if float, then take it as a share
            if isinstance(min_samples, float):
                # check if its within (0,1]
                if (min_samples > 0 and min_samples <= 1):
                    # assign the input value
                    self.min_samples = int(np.ceil(min_samples * self.n_obs))
                else:
                    # raise value error
                    raise ValueError("min_samples must be within (0,1]"
                                     ", got %s" % min_samples)
            # if int, then take it as a absolute number
            else:
                # check if its within [p,N-1]
                if ((min_samples >= self.n_exog + 1) and
                    (min_samples <= self.n_obs - 1)):
                    # assign the input value
                    self.min_samples = min_samples
                else:
                    # raise value error
                    raise ValueError("min_samples must be within [p+1,N-1]"
                                     ", got %s" % min_samples)
        else:
            # raise value error
            raise ValueError("min_samples must be either float or integer"
                             ", got %s" % min_samples)
        
        # define the loss function
        if loss is None:
            # set default for absolute loss
            self.loss = 'absolute_error'
            self.loss_function = (lambda outcome, outcome_pred:
                                  np.abs(outcome - outcome_pred))
        # otherwise check if its specified as a string
        elif isinstance(loss, str):
            # if loss is admissible
            if loss in ('absolute_error', 'squared_error'):
                # if squared loss
                if (loss == 'squared_error'):
                    # specify the function
                    self.loss_function = (lambda outcome, outcome_pred:
                                          (outcome - outcome_pred) ** 2)
                else:
                    # otherwise take absolute loss
                    self.loss_function = (lambda outcome, outcome_pred:
                                          np.abs(outcome - outcome_pred))
            else:
                # else raise an error
                raise ValueError("Input for 'loss' must be either"
                                 " 'absolute_error' or 'squared_error', "
                                 "got %s" % loss)
        else:
            raise ValueError("Input for 'loss' must be a string"
                             ", got %s" % loss)

        # if boosting should be applied
        if boost is None:
            # set None as default, no boosting performed
            self.boost = None
        # otherwise check the percentage of boosting
        elif isinstance(boost, float):
            # check if its within (0,1)
            if (boost >= 0 and boost < 1):
                # assign the input value
                self.boost = int(np.ceil(boost * self.n_obs))
            else:
                # raise value error
                raise ValueError("boost must be within [0,1)"
                                 ", got %s" % boost)
        else:
            # raise value error
            raise ValueError("boost must be a float"
                             ", got %s" % boost)

        # check whether n_jobs is integer
        if isinstance(n_jobs, int):
            # check max available cores
            max_jobs = cpu_count()
            # check if it is -1
            if n_jobs == -1:
                # set max - 1 as default
                self.n_jobs = max_jobs - 1
            # check if jobs are admissible for the machine
            elif (n_jobs >= 1 and n_jobs <= max_jobs):
                # assign the input value
                self.n_jobs = n_jobs
            else:
                # throw an error
                raise ValueError("n_jobs must be greater than 0 and less than"
                                 "available cores, got %s" % n_jobs)
        # check whether n_jobs is set to None
        elif n_jobs is None:
            # set n_jobs to 1 - sequential execution
            self.n_jobs = 1
        else:
            # raise value error
            raise ValueError("n_jobs must be of type integer or None"
                             ", got %s" % n_jobs)

        # check whether seed is set
        self.random_state = random_state
        # TODO: self.random_state = check_random_state(random_state)
        # get max np.int32 based on machine limit
        max_int = np.iinfo(np.int32).max

        # initiate the reliability scores
        self.scores = None


    # %% score function
    # function to estimate RSR reliability scores
    def score(self):
        """
        RSR reliability scores estimation.
        """
        
        # check if boosting should be performed
        if self.boost is None:
            # run the standard RSR algorithm to estimate reliability scores  
            rsr_scores = self._estimate_scores(endog=self.endog,
                                               exog=self.exog,
                                               n_samples=self.n_samples,
                                               min_samples=self.min_samples,
                                               loss=self.loss_function)
        else:
            # run the boosted RSR algorithm to estimate reliability scores  
            rsr_scores = self._estimate_boosted_scores(endog=self.endog,
                                                       exog=self.exog,
                                                       n_samples=self.n_samples,
                                                       min_samples=self.min_samples,
                                                       loss=self.loss_function)
        
        # compute gini coefficient of the scores
        self.gini = self._gini_coef(rsr_scores)

        # assign estimated reliability scores
        self.scores = rsr_scores
        
        # return the scores as own ScoreClass
        return Reliability.RSRScoreResults(sample=self)
    
    
    # %% fit
    # function to fit the linear model: weighted or consensus fit
    def fit(self, weights=None, consensus=None, n_boot=None):
        """
        RSR fit, either weighted or consensus estimation.

        """

        # check fit inputs
        rsr_scores = self._check_fit_inputs(weights, consensus, n_boot)
        
        # compute gini coefficient of the scores
        self.gini = self._gini_coef(rsr_scores)
        
        # fit the reliable model: either weighted fit or consensus fit
        if self.weighted:
            # weighted fit
            betas = self._weighted_fit(endog=self.endog,
                                       exog=self.exog,
                                       scores=self.weights)
        else:
            # consensus fit
            betas, threshold, inliers, outliers = self._consensus_fit(
                endog=self.endog,
                exog=self.exog,
                scores=rsr_scores)
        
        # get fitted values and residuals
        fittedvalues = np.dot(self.exog, betas)
        resid = self.endog - fittedvalues
        
        # estimate standard errors
        if self.n_boot is None:
            # asymptotic approximation
            betas_se, boot_betas = self._asym_se(residuals=resid,
                                                 weights=self.weights,
                                                 weighted=self.weighted)
        else:
            # bootstrap approximation
            betas_se, boot_betas = self._boot_se()

        # return the result as RSRFitResults class
        return Reliability.RSRFitResults(sample=self,
                                         params=betas,
                                         params_boot=boot_betas,
                                         stand_err=betas_se,
                                         fittedvalues=fittedvalues)


    # %% anneal function
    # function for annealing sensitivity analysis based on reliability scores
    def anneal(self, share=0.05, n_boot=None):
        """
        RSR annealing sensitivity analysis.
        """
        
        # check anneal inputs
        self._check_anneal_inputs(share, n_boot)
        
        # get reliability scores first
        rsr_scores = self._check_fitted_scores()
        
        # compute gini coefficient of the scores
        self.gini = self._gini_coef(rsr_scores)
        
        # remove sequantially share*n_obs of most unreliable datapoints
        n_drop = int(np.ceil(self.share * self.n_obs))
        
        # annealing fit
        betas, betas_se, drop_idx = self._annealing_fit(endog=self.endog,
                                                        exog=self.exog,
                                                        scores=rsr_scores,
                                                        n_drop=n_drop)
        
        # estimate standard errors if inference needed
        if self.n_boot is None:
            # asymptotic approximation
            boot_betas = None
        else:
            # bootstrap approximation
            betas_se, boot_betas = self._boot_se(anneal=True)
        
        # return the annealing results as RSRAnnealResults class
        return Reliability.RSRAnnealResults(sample=self,
                                            params=betas,
                                            params_boot=boot_betas,
                                            stand_err=betas_se,
                                            drop_idx=drop_idx)


    # %% in-class internal functions definitions
    # function to estimate the reliability scores
    def _estimate_scores(self, endog, exog, n_samples, min_samples, loss):
        """Estimate the reliability scores via RSR algorithm."""
    
        # get indices to sample from
        indices = np.arange(len(endog), dtype=int)
            
        # controls for the optimization
        iter_loss_value = {}
        sample_idx = 0
        
        # start sampling loop
        while sample_idx < n_samples:
            
            # initiate loss vector with NAs
            loss_value = np.empty(len(endog))
            loss_value[:] = np.nan
    
            # subsample data without replacement
            # get in bag indices
            in_idx = np.random.choice(indices, size=min_samples, replace=False)
            # get out of bag indices
            oob_idx = np.setdiff1d(indices, in_idx)
            
            # in bag observations
            endog_in = endog[in_idx]
            exog_in = exog[in_idx, :]
            # out of bag observations
            endog_out = endog[oob_idx]
            exog_out = exog[oob_idx, :]
    
            # check if the sample is valid
            if self._is_sample_valid(exog_in):
                # update sample_idx
                sample_idx +=1
            else:
                # continue with new draw
                continue
            
            # estimate the betas
            betas = np.linalg.inv(exog_in.T @ exog_in) @ (exog_in.T @ endog_in)
            # predict out-of-bag
            oob_pred = exog_out @ betas
            # evaluate the error for out-of-bag observations
            loss_value[oob_idx] = loss(endog_out, oob_pred)
            # save the results
            iter_loss_value[sample_idx] = loss_value
        
        # resampling loss: rows are observations, columns are iterations
        resampling_loss = pd.DataFrame(iter_loss_value)
        # average over the losses row-wise
        average_loss = np.array(resampling_loss.mean(axis=1))
        # reverse the relationship and scale between 0 and 1 for scores
        # take absolute value to prevent -0
        scores = np.abs((average_loss - np.max(average_loss))/
                        (np.min(average_loss) - np.max(average_loss)))
        
        # return the reliability scores
        return scores
    

    # function to estimate the boosted reliability scores
    def _estimate_boosted_scores(self,
                                 endog, exog, n_samples, min_samples, loss):
        """Estimate the reliability scores via boosted RSR algorithm."""
    
        # boost the RSR algorithm
        boost_drops = []
        boost_indices = np.arange(self.n_obs, dtype=int)
        
        # start annealing
        for drop_idx in range(self.boost):
    
            
            #in_idx = np.setdiff1d(np.arange(self.n_obs, dtype=int),
            #                      boost_drops)
            endog_boost = self.endog[boost_indices]
            exog_boost = self.exog[boost_indices, :]
    
            # run the standard RSR algorithm to estimate reliability scores  
            rsr_scores = self._estimate_scores(endog=endog_boost,
                                               exog=exog_boost,
                                               n_samples=self.n_samples,
                                               min_samples=self.min_samples,
                                               loss=self.loss_function)
        
            # identify the most unreliable observation
            most_unreliable = int(np.where(rsr_scores == 0)[0])
            # add it to drop indices
            boost_drops.append(boost_indices[most_unreliable])
            # get updated sample, without the least reliable score
            boost_indices = np.delete(boost_indices, most_unreliable)
            
        # estimate boosted rsr scores now for all observations 
        # controls for the optimization
        iter_loss_value = {}
        sample_idx = 0
        
        # start sampling loop
        while sample_idx < n_samples:
            
            # initiate loss vector with NAs
            loss_value = np.empty(len(endog))
            loss_value[:] = np.nan
    
            # subsample data without replacement
            # get in bag indices from the reliable part of the sample
            in_idx = np.random.choice(boost_indices, size=min_samples,
                                      replace=False)
            # get out of bag indices
            oob_idx = np.setdiff1d(boost_indices, in_idx)
            # plus out-of-sample indices (unreliable part of the sample)
            out_idx = np.insert(oob_idx, 0, boost_drops)
            
            # in bag observations
            endog_in = endog[in_idx]
            exog_in = exog[in_idx, :]
            # out of bag plus out of sample observations
            endog_out = endog[out_idx]
            exog_out = exog[out_idx, :]
            
            # check if the sample is valid
            if self._is_sample_valid(exog_in):
                # update sample_idx
                sample_idx +=1
            else:
                # continue with new draw
                continue
    
            # estimate the betas
            betas = np.linalg.inv(exog_in.T @ exog_in) @ (exog_in.T @ endog_in)
            # predict out-of-sample
            out_pred = exog_out @ betas
            # evaluate the error for out-of-sample observations
            loss_value[out_idx] = loss(endog_out, out_pred)
            # save the results
            iter_loss_value[sample_idx] = loss_value
        
        # resampling loss: rows are observations, columns are iterations
        resampling_loss = pd.DataFrame(iter_loss_value)
        # average over the losses row-wise
        average_loss = np.array(resampling_loss.mean(axis=1))
        # reverse the relationship and scale between 0 and 1 for scores
        # take absolute value to prevent -0
        scores = np.abs((average_loss - np.max(average_loss))/
                        (np.min(average_loss) - np.max(average_loss)))
        
        # return the reliability scores
        return scores
    
    
    # function to compute gini coefficient for reliability scores
    def _gini_coef(self, scores=None):
        """Compute the Gini coefficient for the RSR scores."""
        
        # sort the scores
        scores = np.sort(scores)
        # get the indices
        idx = np.arange(self.n_obs) + 1
        # compute gini according to formula
        gini = ((np.sum((2 * idx - self.n_obs  - 1) * scores)) /
                (self.n_obs * np.sum(scores)))
        
        # return the gini coef
        return gini


    # function to chekc if sampled data is valid
    def _is_sample_valid(self, matrix=None):
        """Check if sampled covariate matrix has full rank."""
        
        # compute the matrix rank
        rank = np.linalg.matrix_rank(matrix)
        
        # if the matrix has full rank
        if (rank == self.n_exog):
            # update sampling index
            valid = True
        else:
            # otherwise do not update
            valid = False
        
        # return the valid value
        return valid

    
    # function to estimate the standard errors via asymptotic approximation
    def _asym_se(self, residuals=None, weights=None, weighted=None):
        """Estimate the standard errors via asymptotics."""
        
        # check if weighted or consensus fit and compute matrix accordingly
        if weighted:
            # compute weight matrix
            W_matrix = np.diag(weights)
            # compute (x'wx)-1
            x_inv = np.linalg.inv(self.exog.T @ W_matrix @ self.exog)
            # compute sum of squared residuals
            sigma_sq = (residuals.T @ W_matrix @ residuals)
        else:
            # compute (x'x)-1
            x_inv = np.linalg.inv(np.dot(self.exog.T, self.exog))
            # compute sum of squared residuals
            sigma_sq = np.dot(residuals.T, residuals)
        
        # compute se as (sqrt diag((u'u)(x'x)^(-1)/(N-p))
        betas_se = np.sqrt(np.diagonal(
            np.dot(sigma_sq, x_inv)/
            (self.n_obs - self.n_exog)))
        # set boot betas to None
        boot_betas = None
        
        # return the standard erros
        return betas_se, boot_betas
    
        
    # function to estimate the standard errors via bootstrapping
    def _boot_se(self, anneal=False):
        """Estimate the standard errors via bootstrapping."""
    
        # controls for the optimization
        boot_betas = {}
        
        # fix the seed
        np.random.seed(self.random_state)
        
        # start sampling loop
        for boot_idx in range(self.n_boot):
    
            # resample data with replacement
            # get in bag indices
            in_idx = np.random.choice(np.arange(self.n_obs, dtype=int),
                                      size=self.n_obs, replace=True)
            
            # in bag observations
            endog_in = self.endog[in_idx]
            exog_in = self.exog[in_idx, :]
    
            # estimate the scores
            if self.boost is None:
                # estimate standard scores
                scores_idx = self._estimate_scores(endog=endog_in,
                                                   exog=exog_in,
                                                   n_samples=self.n_samples,
                                                   min_samples=self.min_samples,
                                                   loss=self.loss_function)
            else:
                # run the boosted RSR algorithm to estimate reliability scores  
                scores_idx = self._estimate_boosted_scores(endog=endog_in,
                                                           exog=exog_in,
                                                           n_samples=self.n_samples,
                                                           min_samples=self.min_samples,
                                                           loss=self.loss_function)
            
            # check if bootstrap should be used for annealing
            if not anneal:
                # fit the reliable model: either weighted fit or consensus fit
                if self.weighted:
                    # get weights
                    if not self.user_weights:
                        # default squared weights (reflected in inference)
                        weights_idx = scores_idx ** 2
                    else:
                        # user-supplied weights (not reflected in inference)
                        weights_idx = self.weights[in_idx]
                    # weighted fit (1 x n_params)
                    betas_idx = self._weighted_fit(endog=endog_in,
                                                   exog=exog_in,
                                                   scores=weights_idx)
                else:
                    # consensus fit (1 x n_params)
                    betas_idx = self._consensus_fit(endog=endog_in,
                                                    exog=exog_in,
                                                    scores=scores_idx)[0]
            # otherwise do the annealing fit
            else:
                # get the number of drop observations
                n_drop = int(np.ceil(self.share * self.n_obs))
                # get the annealing fit (n_drop+1 x n_params)
                betas_idx = self._annealing_fit(endog=endog_in,
                                                exog=exog_in,
                                                scores=scores_idx,
                                                n_drop=n_drop)[0]

            # save the results
            boot_betas[boot_idx] = betas_idx
            
        # different output shapes for annealing
        if not anneal:
            # reshape boot betas: rows are observations, columns are iters
            boot_betas_reshape = np.array(pd.DataFrame(boot_betas))
            # take bootstrap std as approxamition of standard error
            betas_se = boot_betas_reshape.std(axis=1)
        # otherwise coefficient paths
        else:
            # create empty storage for betas_se
            betas_se = np.zeros([n_drop + 1, self.n_exog])
            # loop through each drop in dictionary and get std (incl. full)
            for drop_idx in range(n_drop + 1):
                # create temp storage (n_boot x n_params)
                boot_drop = np.zeros([self.n_boot, self.n_exog])
                # loop through each key in a dictionary
                for boot_idx in range(self.n_boot):
                    # collect all values of boot params
                    boot_drop[boot_idx, :] = boot_betas[boot_idx][drop_idx, :]
                # take bootstrap std as approxamition of standard error
                betas_se[drop_idx, :] = boot_drop.std(axis=0)
        
        # return the se and bootstrap results of reliability fit
        return betas_se, boot_betas


    # Function for weighted fit
    def _weighted_fit(self, endog, exog, scores):
        """Weighted fit based on reliability scores."""
    
        # create the weight matrix based on reliability scores
        W_matrix = np.diag(scores)
        # estimate the betas
        betas = (np.linalg.inv(exog.T @ W_matrix @ exog) @
                 (exog.T @ W_matrix @ endog))
        # return betas
        return betas


    # Function for threshold estimation
    def _threshold(self, scores):
        """Threshold based on reliability scores."""
        
        # fit the threshold by the largest change in the slope of the scores
        scores_sorted = np.sort(scores)
        # take double diff for each sorted observation (derivative approx)
        scores_sorted_diff = np.diff(np.diff(scores_sorted))
        # get the index of the largest slope change shift index by 1 due to the
        # central differences to get the index of point where the biggest
        # change starts (including that point)
        max_slope_idx = np.argmin(scores_sorted_diff) + 1
        # get the error threshold for oultiers
        threshold = scores_sorted[max_slope_idx]
        # return threshold
        return threshold


    # Function for consensus fit
    def _consensus_fit(self, endog, exog, scores):
        """Consensus fit based on reliability scores."""

        # estimate threshold
        threshold = self._threshold(scores=scores)
        # get inliers
        inliers = (scores >= threshold)
        # get outliers
        outliers = (scores < threshold)
        # estimate consensus fit
        betas = (np.linalg.inv(exog[inliers].T @ exog[inliers]) @
                 (exog[inliers].T @ endog[inliers]))
        # return betas, threshold, in and outlier mask
        return betas, threshold, inliers, outliers
    
    
    # Function for annealing fit
    def _annealing_fit(self, endog, exog, scores, n_drop):
        """Annealing fit for sequential subsamples."""
        
        # compute (x'x)-1
        x_inv = np.linalg.inv(exog.T @ exog)
        # get the full sample fit first
        params = (x_inv @ (exog.T @ endog))
        # get residuals
        residuals = endog - (exog @ params)
        # compute sum of squared residuals
        sigma_sq = (residuals.T @ residuals)
        # compute se as (sqrt diag((u'u)(x'x)^(-1)/(N-p))
        params_se = np.sqrt(np.diagonal(
            np.dot(sigma_sq, x_inv)/(self.n_obs - self.n_exog)))
        
        # sort the scores in increasing manner and get indices
        sorted_scores = np.argsort(scores)
        # store indices of dropped observations
        obs_drop = []
        
        # loop through the first share% of observations
        for obs_idx in sorted_scores[0:n_drop]:
            # update obs_drop
            obs_drop.append(obs_idx)

            # get the current sample
            endog_sample = np.delete(endog, obs_drop)
            exog_sample = np.delete(exog, obs_drop, axis=0)
            
            # compute (x'x)-1
            exog_inv = np.linalg.inv(exog_sample.T @ exog_sample)
            # fit the linear model excluding the observations
            betas = (exog_inv @ (exog_sample.T @ endog_sample))
            
            # get residuals
            res = endog_sample - (exog_sample @ betas)
            # compute sum of squared residuals
            ssq = (res.T @ res)
            # compute se as (sqrt diag((u'u)(x'x)^(-1)/(N-p))
            betas_se = np.sqrt(np.diagonal(
                np.dot(ssq, exog_inv)/(len(endog_sample) - self.n_exog)))

            # save the parameter and se estimates
            params = np.vstack([params, betas])
            params_se = np.vstack([params_se, betas_se])
        
        # return parameter path (n_drop+1 x n_params) and dropped obs idx
        return params, params_se, obs_drop


    # check if rsr scores have been fitted
    def _check_fitted_scores(self):
        """Checks if RSR scores have been fitted and returns scores"""
        
        # check if reliability scores have been fitted already
        if self.scores is None:
            # estimate the reliability scores first
            if self.boost is None:
                # run the standard RSR algorithm to estimate reliability scores 
                rsr_scores = self._estimate_scores(endog=self.endog,
                                                   exog=self.exog,
                                                   n_samples=self.n_samples,
                                                   min_samples=self.min_samples,
                                                   loss=self.loss_function)
            else:
                # run the boosted RSR algorithm to estimate reliability scores  
                rsr_scores = self._estimate_boosted_scores(endog=self.endog,
                                                           exog=self.exog,
                                                           n_samples=self.n_samples,
                                                           min_samples=self.min_samples,
                                                           loss=self.loss_function)
            # and assign scores to self
            self.scores = rsr_scores
        else:
            # take the estimated scores
            rsr_scores = self.scores
        
        # return the scores
        return rsr_scores
    
    
    # function for fit inputs checks
    def _check_fit_inputs(self, weights, consensus, n_boot):
        """Input checks for the .fit() function."""
        
        # check fitted scores
        rsr_scores = self._check_fitted_scores()
        
        # check weighting option
        if weights is None:
            # take squared RSR reliability scores as weights as default
            self.weights = rsr_scores ** 2
            # and document this
            self.user_weights = False
        # otherwise check user supplied weights
        elif isinstance(weights, np.ndarray):
            # check if the dimension fits
            if weights.shape == self.endog.shape:
                # assign weights instead of rsr_scores
                self.weights = weights
                # and document this
                self.user_weights = True
            else:
                # raise value error
                raise ValueError("weight must be of the same dimension "
                                 "as endog: " + str(self.endog.shape) + ""
                                 ", got %s" % weights.shape)
        else:
            # raise value error
            raise ValueError("weight must be of type numpy array"
                             ", got %s" % type(weights))

        # define consensus methods
        consensus_methods = ['second_derivative']
        # check the consensu fit
        if consensus is None:
            # no consensus fit performed
            self.consensus = None
        elif isinstance(consensus, str):
            # check admissible options
            if consensus in consensus_methods:
                # perform consensus fit according to second derivative rule
                self.consensus == 'second_derivative'
            else:
                # raise value error
                raise ValueError("consensus must be one of " +
                                 str(consensus_methods) + ""
                                 ", got %s" % consensus)
        else:
            # raise value error
            raise ValueError("consensus must be NoneType or string "
                             ", got %s" % type(consensus))
        
        # decide if weighted or consesus fit gets estimated
        if (weights is None and consensus is None):
            # then perform weighted fit with RSR scores
            self.weighted = True
        elif (weights is not None and consensus is None):
            # then perform weighted fit with supplied weights
            self.weighted = True
        elif (weights is None and consensus is not None):
            # then perform consensus fit with supplied method
            self.weighted = False
        elif (weights is not None and consensus is not None):
            # then perform weighted fit with supplied weights
            self.weighted = True
        else:
            # raise value error
            raise ValueError("weights and consensus are not compatible, "
                             "check the documentation.")
    
        # check if bootstrapping or asymptotic inference should be done
        if n_boot is None:
            # assign None
            self.n_boot = None
        # otherwise check how many replication should be used for inference
        elif isinstance(n_boot, int):
            # check if its at least 2
            if n_boot >= 2:
                # assign the input value
                self.n_boot = n_boot
            else:
                # raise value error
                raise ValueError("n_boot must be at least 2"
                                 ", got %s" % n_boot)
        else:
            # raise value error
            raise ValueError("n_boot must be an integer or NoneType"
                             ", got %s" % type(n_boot))
            
        # return rsr scores
        return rsr_scores


    # function for anneal inputs checks
    def _check_anneal_inputs(self, share, n_boot):
        """Input checks for the .anneal() function."""
        
        # check the share option
        if isinstance(share, float):
            # check if its within (0,1]
            if (share > 0 and share <= 1):
                # assign the input value
                self.share = share
            else:
                # raise value error
                raise ValueError("share must be within (0,1]"
                                 ", got %s" % share)
        else:
            # raise value error
            raise ValueError("share must be a float"
                             ", got %s" % type(share))

        # check if bootstrapping or asymptotic inference should be done
        if n_boot is None:
            # assign None
            self.n_boot = None
        # otherwise check how many replication should be used for inference
        elif isinstance(n_boot, int):
            # check if its at least 2
            if n_boot >= 2:
                # assign the input value
                self.n_boot = n_boot
            else:
                # raise value error
                raise ValueError("n_boot must be at least 2"
                                 ", got %s" % n_boot)
        else:
            # raise value error
            raise ValueError("n_boot must be an integer or NoneType"
                             ", got %s" % type(n_boot))
