"""
samplefit.

Python library to assess sample fit in econometric models via
the Sample Fit Reliability (SFR) approach as developed by
Okasa & Younge (2022).

Definition of base SFR class and fit, score and anneal methods.

"""

# import statsmodels and samplefit
import statsmodels
import samplefit.Reliability as Reliability

# import modules
import numpy as np
import pandas as pd

# import submodules and functions
from joblib import Parallel, delayed, parallel_backend
from psutil import cpu_count


# %% SFR Class definition
# define BaseSFR class
class BaseSFR:
    """
    Base SFR class of samplefit.
    This class should not be used directly. Use derived user classes instead.
    """

    # define init function
    def __init__(self,
                 linear_model=None,
                 n_samples=1000,
                 min_samples=None,
                 loss=None,
                 n_jobs=-1,
                 random_state=None):

        # assign input values
        self.linear_model = linear_model
        self.n_samples = n_samples
        self.min_samples = min_samples
        self.loss = loss
        self.n_jobs = n_jobs
        self.random_state = random_state
        
        # check the self inputs and set defaults
        self._input_checks()


    def _input_checks(self):
        """Input checks for the SFR class init."""
        
        # check and define the input parameters
        linear_model = self.linear_model
        n_samples = self.n_samples
        min_samples = self.min_samples
        loss = self.loss
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
                if (min_samples > 0 and min_samples < 1):
                    # get the integer value for sampling size
                    min_samples_int = int(np.ceil(min_samples * self.n_obs))
                    # check if its within [p,N-1]
                    if ((min_samples_int >= self.n_exog + 1) and
                        (min_samples_int <= self.n_obs - 1)):
                        # assign the input value
                        self.min_samples = min_samples_int
                    else:
                        # raise value error
                        raise ValueError("min_samples must be within [p+1,N-1]"
                                         ", increase the min_samples share or "
                                         "specify number of minimum samples "
                                         "directly as an integer.")
                else:
                    # raise value error
                    raise ValueError("min_samples must be within (0,1)"
                                     ", got %s" % min_samples)
            # if int, then take it as a absolute number
            else:
                # check if its within [p+1,N-1]
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

        # check whether n_jobs is integer
        if isinstance(n_jobs, int):
            # check max available physical cores
            max_jobs = cpu_count(logical=False)
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
                             ", got %s" % type(n_jobs))

        # check whether seed is set
        if random_state is None:
            # initiate seed as zero
            self.random_state = 0
        # chekc if its integer
        elif isinstance(random_state, int):
            # if its non-negative
            if random_state >= 0:
                # assign the user supplied value
                self.random_state = random_state
            else:
                # throw an error
                raise ValueError("random_state must be equal or greater than 0"
                                 ", got %s" % random_state)
        else:
            # raise value error
            raise ValueError("random_state must be of type integer or None"
                             ", got %s" % type(random_state))
        # TODO: self.random_state = check_random_state(random_state)
        # get max np.int32 based on machine limit
        max_int = np.iinfo(np.int32).max

        # initiate the reliability scores and bootstrapping
        self.scores = None
        self.n_boot = None


    # %% score function
    # function to estimate SFR reliability scores
    def score(self):
        """
        SFR Scoring.
        """
        
        # restore bootstrapping
        self.n_boot = None
        
        # run the SFR algorithm to estimate reliability scores  
        sfr_scores = self._estimate_scores(endog=self.endog,
                                           exog=self.exog,
                                           n_samples=self.n_samples,
                                           min_samples=self.min_samples,
                                           loss=self.loss_function)
        
        # compute gini coefficient of the scores
        self.gini = self._gini_coef(sfr_scores)

        # assign estimated reliability scores
        self.scores = sfr_scores
        
        # return the scores as own ScoreClass
        return Reliability.SFRScoreResults(sample=self)
    
    
    # %% fit
    # function to fit the linear model via weighted least squares
    def fit(self, weights=None, n_boot=None):
        """
        SFR Fitting.

        """

        # check fit inputs
        sfr_scores = self._check_fit_inputs(weights, n_boot)
        
        # compute gini coefficient of the scores
        self.gini = self._gini_coef(sfr_scores)
        
        # fit the reliable model: weighted fit
        betas = self._weighted_fit(endog=self.endog,
                                   exog=self.exog,
                                   scores=self.weights)
        
        # get fitted values and residuals
        fittedvalues = np.dot(self.exog, betas)
        resid = self.endog - fittedvalues
        
        # estimate standard errors
        if self.n_boot is None:
            # asymptotic approximation
            betas_se, boot_betas = self._asym_se(residuals=resid,
                                                 weights=self.weights)
        else:
            # bootstrap approximation
            betas_se, boot_betas = self._boot_se()

        # return the result as SFRFitResults class
        return Reliability.SFRFitResults(sample=self,
                                         params=betas,
                                         params_boot=boot_betas,
                                         stand_err=betas_se,
                                         fittedvalues=fittedvalues)


    # %% anneal function
    # function for annealing sensitivity analysis based on reliability scores
    def anneal(self, share=0.05, n_boot=None):
        """
        SFR Annealing.
        """
        
        # check anneal inputs
        sfr_scores = self._check_anneal_inputs(share, n_boot)
        
        # compute gini coefficient of the scores
        self.gini = self._gini_coef(sfr_scores)
        
        # remove sequantially share*n_obs of most unreliable datapoints
        self.n_drop = int(np.ceil(self.share * self.n_obs))
        
        # annealing fit
        betas, betas_se, drop_idx = self._annealing_fit(endog=self.endog,
                                                        exog=self.exog,
                                                        scores=sfr_scores,
                                                        n_drop=self.n_drop)
        
        # estimate standard errors if inference needed
        if self.n_boot is None:
            # asymptotic approximation already done as default
            boot_betas = None
        else:
            # bootstrap approximation
            betas_se, boot_betas = self._boot_se(anneal=True)
        
        # return the annealing results as SFRAnnealResults class
        return Reliability.SFRAnnealResults(sample=self,
                                            params=betas,
                                            params_boot=boot_betas,
                                            stand_err=betas_se,
                                            drop_idx=drop_idx)


    # %% in-class internal functions definitions
    # function to estimate the reliability scores
    def _estimate_scores(self, endog, exog, n_samples, min_samples, loss):
        """Estimate the reliability scores via SFR algorithm."""
    
        # controls for the optimization
        iter_loss_value = {}
        
        # check if resampling should be done parallel (avoid double parallel)
        if ((self.n_jobs > 1) and (self.n_boot is None)):
            # Loop over samples in parallel using joblib
            with parallel_backend('loky',
                                  n_jobs=self.n_jobs):
                loss_value = Parallel()(
                    delayed(self._estimate_loss)(
                        endog=endog,
                        exog=exog,
                        min_samples=min_samples,
                        loss=loss,
                        seed=(self.random_state + sample_idx)
                        ) for sample_idx in range(n_samples))
            # assign the results
            iter_loss_value = np.vstack(loss_value).T
        else:
            # sequential execution
            for sample_idx in range(n_samples):
                # estimate resample loss
                loss_value = self._estimate_loss(endog=endog,
                                                 exog=exog,
                                                 min_samples=min_samples,
                                                 loss=loss,
                                                 seed=(self.random_state +
                                                 sample_idx))
                # save the results for current iteration
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
    
    
    # function to estimate the individual loss
    def _estimate_loss(self, endog, exog, min_samples, loss, seed):
        """Estimate the individual sample loss."""
        
        # fix the seed
        np.random.seed(seed)
        
        # get indices to sample from
        indices = np.arange(len(endog), dtype=int)
        
        # initiate loss vector with NAs
        loss_value = np.empty(len(endog))
        loss_value[:] = np.nan
            
        # controls for the optimization
        full_rank = False
        
        # start sampling loop
        while not full_rank:
    
            # subsample data without replacement
            # get in bag indices
            in_idx = np.random.choice(indices, size=min_samples, replace=False)
            # in bag observations
            endog_in = endog[in_idx]
            exog_in = exog[in_idx, :]
            
            # check if the sample is valid
            if self._is_sample_valid(exog_in):
                # update condition
                full_rank = True
            else:
                # continue with new draw
                continue
        
        # get out of bag indices
        oob_idx = np.setdiff1d(indices, in_idx)
        # out of bag observations
        endog_out = endog[oob_idx]
        exog_out = exog[oob_idx, :]
            
        # estimate the betas
        betas = np.linalg.inv(exog_in.T @ exog_in) @ (exog_in.T @ endog_in)
        # predict out-of-bag
        oob_pred = exog_out @ betas
        # evaluate the error for out-of-bag observations
        loss_value[oob_idx] = loss(endog_out, oob_pred)
        
        # return the individual sample loss
        return loss_value
    
    
    # function to compute gini coefficient for reliability scores
    def _gini_coef(self, scores=None):
        """Compute the Gini coefficient for the SFR scores."""
        
        # sort the scores
        scores = np.sort(scores)
        # get the indices
        idx = np.arange(self.n_obs) + 1
        # compute gini according to formula
        gini = ((np.sum((2 * idx - self.n_obs  - 1) * scores)) /
                (self.n_obs * np.sum(scores)))
        
        # return the gini coef
        return gini


    # function to check if sampled data is valid
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
    def _asym_se(self, residuals=None, weights=None):
        """Estimate the standard errors via asymptotics."""
        
        # compute weight matrix
        W_matrix = np.diag(weights)
        # compute (x'wx)-1
        x_inv = np.linalg.inv(self.exog.T @ W_matrix @ self.exog)
        # compute sum of squared residuals
        sigma_sq = (residuals.T @ W_matrix @ residuals)
        
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
        
        # check if parallelisation should be used
        if self.n_jobs > 1:
            # use multiprocessing using joblib
            with parallel_backend('loky',
                                  n_jobs=self.n_jobs):
                boot_betas = Parallel()(
                    delayed(self._boot_iter)(
                        anneal=anneal,
                        seed=(self.random_state + boot_idx)
                        ) for boot_idx in range(self.n_boot))
            # assign the results
            boot_betas = {key:value for key, value in enumerate(boot_betas)}
        else:
            # start sampling loop in serial
            for boot_idx in range(self.n_boot):
                # save the results in a dictionary
                boot_betas[boot_idx] = self._boot_iter(anneal=anneal,
                                                       seed=(self.random_state
                                                             + boot_idx))
            
        # different output shapes for annealing
        if not anneal:
            # reshape boot betas: rows are observations, columns are iters
            boot_betas_reshape = np.array(pd.DataFrame(boot_betas))
            # take bootstrap std as approxamition of standard error
            betas_se = boot_betas_reshape.std(axis=1)
        # otherwise coefficient paths
        else:
            # create empty storage for betas_se
            betas_se = np.zeros([self.n_drop + 1, self.n_exog])
            # loop through each drop in dictionary and get std (incl. full)
            for drop_idx in range(self.n_drop + 1):
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

    
    # bootstrap iterations
    def _boot_iter(self, anneal, seed):
        """Perform a bootstrap iteration."""
        
        # fix the seed
        np.random.seed(seed)
        # resample data with replacement
        # get in bag indices
        in_idx = np.random.choice(np.arange(self.n_obs, dtype=int),
                                  size=self.n_obs, replace=True)
        
        # in bag observations
        endog_in = self.endog[in_idx]
        exog_in = self.exog[in_idx, :]

        # estimate the scores
        scores_idx = self._estimate_scores(endog=endog_in,
                                           exog=exog_in,
                                           n_samples=self.n_samples,
                                           min_samples=self.min_samples,
                                           loss=self.loss_function)
        
        # check if bootstrap should be used for annealing
        if not anneal:
            # fit the reliable model: weighted fit
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
        # otherwise do the annealing fit
        else:
            # get the annealing fit (n_drop+1 x n_params)
            betas_idx = self._annealing_fit(endog=endog_in,
                                            exog=exog_in,
                                            scores=scores_idx,
                                            n_drop=self.n_drop)[0]

        # return the results
        return betas_idx

        
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
        # this needs to be sequential - no pralallelisation
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


    # check if reliability scores have been fitted
    def _check_fitted_scores(self):
        """Checks if reliability scores have been fitted and returns scores"""
        
        # check if reliability scores have been fitted already
        if self.scores is None:
            # estimate the reliability scores first
            sfr_scores = self._estimate_scores(endog=self.endog,
                                               exog=self.exog,
                                               n_samples=self.n_samples,
                                               min_samples=self.min_samples,
                                               loss=self.loss_function)
            # and assign scores to self
            self.scores = sfr_scores
        else:
            # take the estimated scores
            sfr_scores = self.scores
        
        # return the scores
        return sfr_scores
    
    
    # function for fit inputs checks
    def _check_fit_inputs(self, weights, n_boot):
        """Input checks for the .fit() function."""
        
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

        # check fitted scores
        sfr_scores = self._check_fitted_scores()
        
        # check weighting option
        if weights is None:
            # take squared reliability scores as weights as default
            self.weights = sfr_scores ** 2
            # and document this
            self.user_weights = False
        # otherwise check user supplied weights
        elif isinstance(weights, np.ndarray):
            # check if the dimension fits
            if weights.shape == self.endog.shape:
                # assign weights instead of sfr_scores
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
            
        # return sfr scores
        return sfr_scores


    # function for anneal inputs checks
    def _check_anneal_inputs(self, share, n_boot):
        """Input checks for the .anneal() function."""
        
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

        # get reliability scores
        sfr_scores = self._check_fitted_scores()
        
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
        
        return sfr_scores
