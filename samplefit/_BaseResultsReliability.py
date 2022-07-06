"""
samplefit: Random Sample Reliability.

Python library to assess Sample Fit via the Random Sample Reliability
algorithm as developed by Okasa & Younge (2022).

Definition of Reliability Results Classes and the methods.

"""

# import statsmodels and samplefit
import statsmodels
import samplefit.Reliability as Reliability

# import modules
import numpy as np # (hast to be 1.22.0 at least, due to np.percentile changes)
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# import submodules and functions
from scipy import stats
# TODO: from joblib import Parallel, delayed, parallel_backend
from multiprocessing import cpu_count, Lock
from matplotlib.colors import is_color_like
# TODO: add check_random_state from statsmodels 0.14.0


# %% RSR Results Classes
# define BaseFitResults class
class BaseRSRFitResults:
    """
    Base class for RSRFitResults.
    This class should not be used directly. Use derived classes instead.
    """

    # define init function
    def __init__(self,
                 sample=None,
                 params=None,
                 params_boot=None,
                 stand_err=None,
                 fittedvalues=None
                 ):
        
        # assign input values
        self.sample = sample
        self.params = params
        self.params_boot = params_boot
        self.stand_err = stand_err
        self.fittedvalues = fittedvalues
        
        # check the self outputs and set defaults
        self._output_checks()


    def _output_checks(self):
        """Output checks for the BaseRSRFitResults class init."""
        
        # check and define the output parameters
        sample = self.sample
        params = self.params
        params_boot = self.params_boot
        stand_err = self.stand_err
        fittedvalues = self.fittedvalues

        # check if the sample is of class RSR
        if isinstance(sample, Reliability.RSR):
            # assign the value
            self.sample = sample
        else:
            # raise value error
            raise ValueError('sample must be a class of '
                             'SampleFit.Reliability.RSR '
                             f', got {type(sample)}.')

        # check the estimated parameters is an array
        if isinstance(params, np.ndarray):
            # check if its not empty
            if not params.size == 0:
                # assign the value
                self.params = params
            else:
                # raise value error
                raise ValueError("params must not be empty"
                                 ", got %s" % params.size)
        else:
            # raise value error
            raise ValueError("params must be a numpy array"
                             ", got %s" % type(params))

        
        # check the estimated boot parameters is a dictionary
        if isinstance(params_boot, dict):
            # check if its not empty
            if bool(params_boot):
                # assign the value
                self.params_boot = params_boot
            else:
                # raise value error
                raise ValueError("params_boot must not be empty"
                                 ", got %s" % bool(params_boot))
        # check if its None
        elif params_boot is None:
            # no bootstrapping has been done - assign the value
            self.params_boot = params_boot
        else:
            # raise value error
            raise ValueError("params_boot must be a dictionary or a NoneType"
                             ", got %s" % type(params_boot))
        
        # check the estimated standard error is an array
        if isinstance(stand_err, np.ndarray):
            # check if its not empty
            if not stand_err.size == 0:
                # assign the value
                self.stand_err = stand_err
            else:
                # raise value error
                raise ValueError("stand_err must not be empty"
                                 ", got %s" % stand_err.size)
        else:
            # raise value error
            raise ValueError("stand_err must be a numpy array"
                             ", got %s" % type(stand_err))
        
        # check the fittedvalues is an array
        if isinstance(fittedvalues, np.ndarray):
            # check if its not empty
            if not fittedvalues.size == 0:
                # assign the value
                self.fittedvalues = fittedvalues
            else:
                # raise value error
                raise ValueError("fittedvalues must not be empty"
                                 ", got %s" % fittedvalues.size)
        else:
            # raise value error
            raise ValueError("fittedvalues must be a numpy array"
                             ", got %s" % type(fittedvalues))
        
        # get exog and endog
        self.exog = self.sample.exog
        self.endog = self.sample.endog
        
        # initiliaze exog names, observations, etc
        self.exog_names = self.sample.exog_names
        self.n_obs = int(self.sample.n_obs)
        self.n_exog = self.sample.n_exog
        
        # get the class inputs
        self.min_samples = self.sample.min_samples
        self.loss = self.sample.loss
        self.boost = self.sample.boost
        self.n_boot = self.sample.n_boot
        self.weights = self.sample.weights

        # get the scores and compute gini coef
        self.scores = self.sample.scores
        self.gini = self.sample.gini
        
        # compute residuals, degrees of freedom, rsquared, tvalues and pvalues
        self.resid = self.endog - self.fittedvalues
        self.df_resid = self.n_obs - self.n_exog
        self.rsquared = 1 - ((self.resid.T @ self.resid)/
                             ((self.endog - np.mean(self.endog)).T @
                              (self.endog - np.mean(self.endog))))
        self.tvalues = self.params / self.stand_err
        self.pvalues = stats.t.sf(np.abs(self.tvalues), self.df_resid) * 2


    # %% in-class functions
    # function to predict yhats based on the weighted or consensus fit
    def predict(self, params=None, exog=None):
        """
        RSR prediction after fit.
        """
        
        # check inputs for predict
        params, exog = self._check_predict_inputs(params, exog)
        
        # get predictions
        preds = np.dot(exog, params)

        # return the preds
        return preds
    
    
    # function to compute confidence intervals
    def conf_int(self, alpha=0.05, percentile=False):
        """
        RSR confidence intervals after fit.
        """
        
        # check the inputs
        self._check_conf_int_inputs(alpha, percentile)
        
        # check if asymptotic or bootstrap have been done
        if self.params_boot is None:
            # compute asymptotic conf intervals
            lower_ci, upper_ci = self._asym_ci(betas=self.params,
                                               betas_se=self.stand_err,
                                               alpha=self.alpha)
        else:
            # compute bootstrap conf intervals
            lower_ci, upper_ci = self._boot_ci(betas=self.params,
                                               betas_se=self.stand_err,
                                               boot_betas=self.params_boot,
                                               alpha=self.alpha,
                                               percentile=self.percentile)
        # return confidence intervals
        return lower_ci, upper_ci
    
    
    # function to print summary
    def summary(self, yname=None, xname=None, title=None, alpha=0.05,
                percentile=False):
        """
        RSR fit summary.
        """
        
        # check inputs for summary
        self._check_summary_inputs(yname, xname, title, alpha, percentile)
        
        # get the preamble
        # info abour RSR
        rsr_info = ['No. Samples:', 'Min. Samples:', 'Loss:', 'Gini:']
        rsr_out = [self.sample.n_samples, self.min_samples,
                   self.loss, np.round(self.gini, 4)]
        # info about fit
        fit_info = ['Dep. Variable:', 'No. Observations:', 'Df Residuals:',
                    'R-squared:']
        fit_out = [self.yname, self.n_obs, self.df_resid,
                   np.round(self.rsquared, 4)]
        
        # get header for table
        header = ['coef', 'std err', 't', 'P>|t|',
                  '[' + str(alpha/2), str(1-alpha/2) + ']']
        # compute conf int
        ci_low, ci_up = self.conf_int(alpha=self.alpha,
                                      percentile=self.percentile)
        # concatenate arrays with outputs
        output = np.vstack([self.params, self.stand_err, self.tvalues,
                            self.pvalues, ci_low, ci_up]).T
        # create dataframe table
        summary_table = pd.DataFrame(np.round(output, 4),
                                     index=self.xname, columns=header)
        
        # print the preamble title
        print('\n',
              f"{self.title : ^80}",
              '=' * 80,
              '-' * 10 + ' RSR Results ' + '-' * 34 + ' Fit Results ' + '-' * 10,
              '=' * 80,
              sep='\n')
        # print the preamble data
        for idx in range(len(rsr_info)):
            # print the preambe
            print(f"{rsr_info[idx]:<15}{rsr_out[idx]:>20}          {fit_info[idx]:<20}{fit_out[idx]:>15}")
        # print params summary
        print('=' * 80,
              summary_table.to_string(justify='right', line_width=80, col_space=10),
              '=' * 80,
              sep='\n')

        # return None
        return None
    
    
    #%% non-user in-class functions
    # function to compute conf intervals via asymptotic approximation
    def _asym_ci(self, betas=None, betas_se=None, alpha=0.05):
        """Compute the confidence intervals via asymptotics."""
        
        # compute confidence intervals via asymptotics
        # take lower tail
        lower_ci = betas - stats.norm.ppf(1-alpha/2) * betas_se
        # take upper tail
        upper_ci = betas + stats.norm.ppf(1-alpha/2) * betas_se
                
        # return confidence intervals
        return lower_ci, upper_ci
    
    
    # function for bootstrap confidence intervals
    def _boot_ci(self, betas=None, betas_se=None, boot_betas=None, alpha=0.05,
                 percentile=False):
        """Compute confidence intervals via bootstrapping."""

        # get the number of estimated betas (models)
        n_betas = boot_betas[0].shape[0]

        # compute confidence intervals based on bootstrap
        if percentile:
            # check if coefficients alone or coefficienth paths
            if not n_betas > 1:
                # single betas reshape
                boot_betas = np.array(pd.DataFrame(boot_betas))
                # sort bootstrapped betas
                sorted_betas = np.sort(boot_betas, axis=1)
                # take lower percentile
                lower_ci = np.percentile(sorted_betas,
                                         q=alpha/2,
                                         method='closest_observation',
                                         axis=1)
                # take upper percentile
                upper_ci = np.percentile(sorted_betas,
                                         q=1-alpha/2,
                                         method='closest_observation',
                                         axis=1)
            else:
                # create empty storage for CIs
                lower_ci = np.zeros([n_betas, self.n_exog])
                upper_ci = np.zeros([n_betas, self.n_exog])
                # loop through annealing drops
                for drop_idx in range(n_betas):
                    # create temp storage (n_boot x n_params)
                    boot_drop = np.zeros([self.n_boot, self.n_exog])
                    # loop through each key in a dictionary
                    for boot_idx in range(self.n_boot):
                        # collect all values of boot params
                        boot_drop[boot_idx, :] = boot_betas[boot_idx][drop_idx, :]
                    # sort the betas
                    sorted_betas = np.sort(boot_drop, axis=0)
                    # take bootstrap percentile as approxamition of CIs
                    # lower
                    lower_ci[drop_idx, :] = np.percentile(sorted_betas,
                                                          q=alpha/2,
                                                          method='closest_observation',
                                                          axis=0)
                    # upper
                    upper_ci[drop_idx, :] = np.percentile(sorted_betas,
                                                          q=1-alpha/2,
                                                          method='closest_observation',
                                                          axis=0)
        else:
            # take bootstrap std as approxamition of standard error
            # take lower tail
            lower_ci = betas - stats.norm.ppf(1-alpha/2) * betas_se
            # take upper tail
            upper_ci = betas + stats.norm.ppf(1-alpha/2) * betas_se
                
        # return confidence intervals
        return lower_ci, upper_ci


    # function for predict inputs checks
    def _check_predict_inputs(self, params, exog):
        """Input checks for the .predict() function."""
        
        # check if params is None
        if params is None:
            # assign estimated parameters
            params = self.params
        else:
            # if supplied, check if types are the same
            if (type(params) == type(self.params)):
                # if supplied, check if dimensions are the same
                if params.shape == self.params.shape:
                    # assign the value
                    params = params
                else:
                    # raise value error
                    raise ValueError("params must have same dimensions as "
                                     "estimated params: " + 
                                     str(self.params.shape) + ""
                                     ", got %s" % params.shape)
            else:
                # raise value error
                raise ValueError("params must have same type as "
                                 "estimated params: " + 
                                 str(type(self.params)) + ""
                                 ", got %s" % type(params))
        
        # check if exog is None
        if exog is None:
            # assign in-sample
            exog = self.exog
        else:
            # if supplied, check if types are the same
            if (type(exog) == type(self.exog)):
                # if supplied, check if dimensions are the same
                if exog.shape == self.exog.shape:
                    # assign the value
                    exog = exog
                else:
                    # raise value error
                    raise ValueError("exog must have same dimensions as "
                                     "in-sample data: " + 
                                     str(self.exog.shape) + ""
                                     ", got %s" % exog.shape)
            else:
                # raise value error
                raise ValueError("exog must have same type as "
                                 "in-sample data: " + 
                                 str(type(self.exog)) + ""
                                 ", got %s" % type(exog))
        
        # return checked params and exog
        return params, exog
    
    
    # function for conf_int inputs checks
    def _check_conf_int_inputs(self, alpha, percentile):
        """Input checks for the .conf_int() function."""
        
        # check what confidence level should be tested
        if isinstance(alpha, float):
            # check if its within (0,1)
            if (alpha > 0 and alpha < 1):
                # assign the input value
                self.alpha = alpha
            else:
                # raise value error
                raise ValueError("alpha must be between (0,1)"
                                 ", got %s" % alpha)
        else:
            # raise value error
            raise ValueError("alpha must be a float"
                             ", got %s" % type(alpha))
        
        # check whether to conduct percentile method for CIs
        if isinstance(percentile, bool):
            # assign to self
            self.percentile = percentile
        else:
            # raise value error
            raise ValueError("percentile must be of type boolean"
                             ", got %s" % type(percentile))
                
        
    # function for summary inputs checks
    def _check_summary_inputs(self, yname, xname, title, alpha, percentile):
        """Input checks for the .summary() function."""
        
        # check name for y
        if yname is None:
            # set default as 'y'
            self.yname = 'y'
        # if supplied check if its valid
        elif isinstance(yname, str):
            # set value to user supplied
            self.yname = yname
        else:
            # raise value error
            raise ValueError("yname must be a string"
                             ", got %s" % type(yname))
        
        # check name for x
        if xname is None:
            # set default as exog names
            self.xname = self.exog_names
        # if supplied check if its valid
        elif isinstance(xname, (list, tuple)):
            # check if dimension fits
            if len(xname) == self.n_exog:
                # set value to user supplied
                self.xname = xname
            else:
                # raise value error
                raise ValueError("xname must be of exog length: " +
                                 str(self.n_exog) + ""
                                 ", got %s" % len(xname))
        else:
            # raise value error
            raise ValueError("xname must be a list or tuple of strings"
                             ", got %s" % type(xname))
        
        # check name for title
        if title is None:
            # set default as RSR title
            self.title = 'Random Sample Reliability Fit Results'
        # if supplied check if its valid
        elif isinstance(title, str):
            # set value to user supplied
            self.title = title
        else:
            # raise value error
            raise ValueError("title must be a string"
                             ", got %s" % type(title))
        
        # check what confidence level should be tested
        if isinstance(alpha, float):
            # check if its within (0,1)
            if (alpha > 0 and alpha < 1):
                # assign the input value
                self.alpha = alpha
            else:
                # raise value error
                raise ValueError("alpha must be between (0,1)"
                                 ", got %s" % alpha)
        else:
            # raise value error
            raise ValueError("alpha must be a float"
                             ", got %s" % type(alpha))
        
        # check whether to conduct percentile method for CIs
        if isinstance(percentile, bool):
            # assign to self
            self.percentile = percentile
        else:
            # raise value error
            raise ValueError("percentile must be of type boolean"
                             ", got %s" % percentile)


# define BaseAnnealResults class
class BaseRSRAnnealResults:
    """
    Base class for RSRAnnealResults.
    This class should not be used directly. Use derived classes instead.
    """

    # define init function
    def __init__(self,
                 sample=None,
                 params=None,
                 params_boot=None,
                 stand_err=None,
                 drop_idx=None
                 ):
        
        # assign input values
        self.sample = sample
        self.params = params
        self.params_boot = params_boot
        self.stand_err = stand_err
        self.drop_idx = drop_idx
        
        # check the self outputs and set defaults
        self._output_checks()


    def _output_checks(self):
        """Output checks for the BaseRSRAnnealResults class init."""
        
        # check and define the output parameters
        sample = self.sample
        params = self.params
        params_boot = self.params_boot
        stand_err = self.stand_err
        drop_idx = self.drop_idx

        # check if the sample is of class RSR
        if isinstance(sample, Reliability.RSR):
            # assign the value
            self.sample = sample
        else:
            # raise value error
            raise ValueError('sample must be a class of '
                             'SampleFit.Reliability.RSR '
                             f', got {type(sample)}.')

        # check the estimated parameters is an array
        if isinstance(params, np.ndarray):
            # check if its not empty
            if not params.size == 0:
                # check if its multidimensional
                if params.shape[0] > 1:
                    # assign the value
                    self.params = params
                else:
                    # raise value error
                    raise ValueError("params must have at least 2 rows"
                                     ", got %s" % params.shape[0])
            else:
                # raise value error
                raise ValueError("params must not be empty"
                                 ", got %s" % params.size)
        else:
            # raise value error
            raise ValueError("params must be a numpy array"
                             ", got %s" % type(params))

        
        # check the estimated boot parameters is a dictionary
        if isinstance(params_boot, dict):
            # check if its not empty
            if bool(params_boot):
                # assign the value
                self.params_boot = params_boot
            else:
                # raise value error
                raise ValueError("params_boot must not be empty"
                                 ", got %s" % bool(params_boot))
        # check if its None
        elif params_boot is None:
            # no bootstrapping has been done - assign the value
            self.params_boot = params_boot
        else:
            # raise value error
            raise ValueError("params_boot must be a dictionary or a NoneType"
                             ", got %s" % type(params_boot))
        
        # check the estimated standard error is an array
        if isinstance(stand_err, np.ndarray):
            # check if its not empty
            if not stand_err.size == 0:
                # check if its multidimensional
                if stand_err.shape[0] > 1:
                    # assign the value
                    self.stand_err = stand_err
                else:
                    # raise value error
                    raise ValueError("stand_err must have at least 2 rows"
                                     ", got %s" % stand_err.shape[0])
                # assign the value
                self.stand_err = stand_err
            else:
                # raise value error
                raise ValueError("stand_err must not be empty"
                                 ", got %s" % stand_err.size)
        else:
            # raise value error
            raise ValueError("stand_err must be a numpy array"
                             ", got %s" % type(stand_err))
        
        # check the fittedvalues is a list
        if isinstance(drop_idx, list):
            # check if its not empty
            if len(drop_idx) > 0:
                # assign the value
                self.drop_idx = drop_idx
            else:
                # raise value error
                raise ValueError("drop_idx must not be empty"
                                 ", got %s" % len(drop_idx))
        else:
            # raise value error
            raise ValueError("drop_idx must be a list"
                             ", got %s" % type(drop_idx))
        
        # get exog and endog
        self.exog = self.sample.exog
        self.endog = self.sample.endog
        
        # initiliaze exog names, observations, etc
        self.exog_names = self.sample.exog_names
        self.n_obs = int(self.sample.n_obs)
        self.n_exog = self.sample.n_exog
        
        # get the class inputs
        self.min_samples = self.sample.min_samples
        self.loss = self.sample.loss
        self.boost = self.sample.boost
        self.share = self.sample.share
        self.n_boot = self.sample.n_boot

        # get the scores and compute gini coef
        self.scores = self.sample.scores
        self.gini = self.sample.gini


    # %% in-class functions
    # function to plot annealing sensitivity
    def plot(self,
             xname=None, title=None, alpha=0.05, percentile=False, color=None,
             path=None, figsize=None, ylim=None, xlabel=None, dpi=None):
        """
        RSR Annealing Sensitivity Plot.
        """
        
        # check inputs for plot
        self._check_plot_inputs(
            xname, title, color, path, figsize, ylim, xlabel, dpi)
        
        # set resolution
        plt.rcParams['figure.dpi'] = self.dpi
        plt.rcParams['savefig.dpi'] = self.dpi
        
        # check if confidence intervals should be plotted too
        if not alpha is None:
            # get confidence bands
            ci_low, ci_up = self.conf_int(alpha=alpha, percentile=percentile)
        else:
            # no CIs
            ci_low = None
            ci_up = None
        
        # get plot x axis width
        x_width = len(self.drop_idx) + 1
        
        # get storage
        figures = {}
        iter_idx = 0
        
        # plot for each xname
        for var_name in self.xname:
            # get the variable idx
            var_idx = self.exog_names.index(var_name)
            
            # define the plot layout
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize = self.figsize)

            # plot annealing fit
            ax.plot(range(x_width), self.params[:, var_idx], color='black',
                    label=self.xlabel[iter_idx])
            # set limits if specified
            if not self.ylim is None:
                ax.set_ylim(self.ylim)
            # ax.set_ylim([np.min(self.params[:, var_idx]),
            #              np.max(self.params[:, var_idx])])
            # allow buffer on left and right
            ax.set_xlim([-0.065*x_width, x_width + 0.065*x_width])
            
            # plot confidence bands
            if not ci_low is None:
                # fill in
                ax.fill_between(range(x_width),
                                ci_up[:, var_idx],
                                ci_low[:, var_idx],
                                color=self.color, alpha=.2,
                                label= str(int((1-self.alpha)*100)) + '% CI')
            
            # dashed line at zero
            # plt.axhline(y=0, color='black', linestyle='dashed')
            
            # add titles, ticks, etc.
            ax.title.set_text(self.title)
            ax.set_xlabel('Sample Share Dropped')
            ax.set_ylabel('Effect')
            labels = list(np.arange(0, self.share, 0.01))
            labels.append(self.share)
            ticks = np.arange(0, x_width, np.floor(x_width/(len(labels)-1)))
            plt.xticks(ticks, labels)
            plt.legend(loc="best")
            plt.show()
            
            # save plot
            if not self.path is None:
                # save plot
                fig.savefig(self.path + '/' + var_name + '_annealing.png',
                            bbox_inches='tight')
            
            # save figure
            figures[var_name] = (fig, ax)
            # update index
            iter_idx += 1
        
        # asign figures to self
        self.figures = figures
        
        # reset resolution
        plt.rcParams['figure.dpi'] = 80
        plt.rcParams['savefig.dpi'] = 100

        # empty return
        return None
    
    
    # function to compute confidence intervals for annealing sensitivity
    def conf_int(self, alpha=0.05, percentile=False):
        """
        RSR confidence intervals after anneal.
        """
        
        # check the inputs
        self._check_conf_int_inputs(alpha, percentile)
        
        # check if asymptotic or bootstrap have been done
        if self.params_boot is None:
            # compute asymptotic conf intervals
            lower_ci, upper_ci = self._asym_ci(betas=self.params,
                                               betas_se=self.stand_err,
                                               alpha=self.alpha)
        else:
            # compute bootstrap conf intervals
            lower_ci, upper_ci = self._boot_ci(betas=self.params,
                                               betas_se=self.stand_err,
                                               boot_betas=self.params_boot,
                                               alpha=self.alpha,
                                               percentile=self.percentile)
        # return confidence intervals
        return lower_ci, upper_ci
    
    
    # fucntion to check inputs for conf_int function
    def _check_conf_int_inputs(self, alpha, percentile):
        """Input checks for the .conf_int() function."""
        
        # check what confidence level should be tested
        if isinstance(alpha, float):
            # check if its within (0,1)
            if (alpha > 0 and alpha < 1):
                # assign the input value
                self.alpha = alpha
            else:
                # raise value error
                raise ValueError("alpha must be between (0,1)"
                                 ", got %s" % alpha)
        else:
            # raise value error
            raise ValueError("alpha must be a float"
                             ", got %s" % type(alpha))
        
        # check whether to conduct percentile method for CIs
        if isinstance(percentile, bool):
            # assign to self
            self.percentile = percentile
        else:
            # raise value error
            raise ValueError("percentile must be of type boolean"
                             ", got %s" % type(percentile))
    
    
    # fucntion to check inputs for plot function
    def _check_plot_inputs(self, xname, title, color, path, figsize, ylim,
                           xlabel, dpi):
        """Input checks for the .plot() function."""
        
        # check xname
        if xname is None:
            # plot all as default
            self.xname = self.exog_names
        # otherwise check inputs
        elif isinstance(xname, (list, tuple, str)):
            # check if they are admissible
            if isinstance(xname, str):
                # check in
                if xname in self.exog_names:
                    # assign the input value as list
                    self.xname = [xname]
                else:
                    # raise value error
                    raise ValueError("xname must be one of:" +
                                     str(self.exog_names) + ""
                                     ", got %s" % xname)
            else:
                # check set in
                if set(xname) <= set(self.exog_names):
                    # assign the input value
                    self.xname = xname
                else:
                    # raise value error
                    raise ValueError("xname must be one of:" +
                                     str(self.exog_names) + ""
                                     ", got %s" % xname)
        else:
            # raise value error
            raise ValueError("xname must be one of tuple, list or string"
                             ", got %s" % type(xname))
        
        # check path for the plot
        if path is None:
            # do not save
            self.path = None
        elif isinstance(path, str):
            # assign the input value
            self.path = path
        else:
            # raise value error
            raise ValueError("path must be a string"
                             ", got %s" % type(path))
        
        # check path for the plot
        if figsize is None:
            # set default as 10,5
            self.figsize = (10, 5)
        elif isinstance(figsize, tuple):
            # assign the input value
            self.figsize = figsize
        else:
            # raise value error
            raise ValueError("figsize must be a tuple"
                             ", got %s" % type(path))
        
        # check for color
        if color is None:
            # define grey as default
            self.color = 'grey'
        # check if it is float
        elif isinstance(color, str):
            # check if its in the available colors
            if is_color_like(color):
                # get the user specified values
                self.color = color
            else:
                # raise error
                raise ValueError("The 'color' argument "
                                 "must be one of matplotlib colors."
                                 ", got %s" % color)
        else:
            # raise error
            raise TypeError("The 'color' argument must be a string."
                            ", got %s" % type(color))
        
        # check for title
        if title is None:
            # define default
            self.title = 'RSR: Annealing Sensitivity'
        # check if it is float
        elif isinstance(title, str):
            # assign input value
            self.title = title
        else:
            # raise error
            raise TypeError("The 'title' argument must be a string."
                            ", got %s" % type(title))
        
        # check name for ylim
        if ylim is None:
            # set default auto
            self.ylim = ylim
        # if supplied check if its valid
        elif isinstance(ylim, (tuple, list)):
            # set value to user supplied
            self.ylim = ylim
        else:
            # raise value error
            raise ValueError("ylim must be a tuple or a list"
                             ", got %s" % type(ylim))
        
        # check xname
        if xlabel is None:
            # plot xname as default
            self.xlabel = self.xname
        # otherwise check inputs
        elif isinstance(xlabel, (list, tuple, str)):
            # check if they are admissible
            if isinstance(xlabel, str):
                # check in
                if len(self.xname) == 1:
                    # assign the input value as list
                    self.xlabel = [xlabel]
                else:
                    # raise value error
                    raise ValueError("xlabel must be of same length as xname"
                                     ", got a single string.")
            else:
                # check if length of tuple and list matches xname
                if len(self.xname) == len(xlabel):
                    # assign the input value
                    self.xlabel = xlabel
                else:
                    # raise value error
                    raise ValueError("xlabel must be of same length as xname"
                                     ", got %s" % len(xlabel))
        else:
            # raise value error
            raise ValueError("xlabel must be one of tuple, list or string"
                             ", got %s" % type(xlabel))
        
        # check dpi
        if dpi is None:
            # set default auto as in matplotlib
            self.dpi = 100
        # if supplied check if its valid
        elif isinstance(dpi, (float, int)):
            # set value to user supplied
            self.dpi = dpi
        else:
            # raise value error
            raise ValueError("dpi must be float or int"
                             ", got %s" % type(dpi))
    
    
    # function to compute conf intervals via asymptotic approximation
    def _asym_ci(self, betas=None, betas_se=None, alpha=0.05):
        """Compute the confidence intervals via asymptotics."""
        
        # compute confidence intervals via asymptotics
        # take lower tail
        lower_ci = betas - stats.norm.ppf(1-alpha/2) * betas_se
        # take upper tail
        upper_ci = betas + stats.norm.ppf(1-alpha/2) * betas_se
                
        # return confidence intervals
        return lower_ci, upper_ci
    
    
    # function for bootstrap confidence intervals
    def _boot_ci(self, betas=None, betas_se=None, boot_betas=None, alpha=0.05,
                 percentile=False):
        """Compute confidence intervals via bootstrapping."""

        # get the number of estimated betas (models)
        n_betas = boot_betas[0].shape[0]

        # compute confidence intervals based on bootstrap
        if percentile:
            # check if coefficients alone or coefficienth paths
            if not n_betas > 1:
                # single betas reshape
                boot_betas = np.array(pd.DataFrame(boot_betas))
                # sort bootstrapped betas
                sorted_betas = np.sort(boot_betas, axis=1)
                # take lower percentile
                lower_ci = np.percentile(sorted_betas,
                                         q=alpha/2,
                                         method='closest_observation',
                                         axis=1)
                # take upper percentile
                upper_ci = np.percentile(sorted_betas,
                                         q=1-alpha/2,
                                         method='closest_observation',
                                         axis=1)
            else:
                # create empty storage for CIs
                lower_ci = np.zeros([n_betas, self.n_exog])
                upper_ci = np.zeros([n_betas, self.n_exog])
                # loop through annealing drops
                for drop_idx in range(n_betas):
                    # create temp storage (n_boot x n_params)
                    boot_drop = np.zeros([self.n_boot, self.n_exog])
                    # loop through each key in a dictionary
                    for boot_idx in range(self.n_boot):
                        # collect all values of boot params
                        boot_drop[boot_idx, :] = boot_betas[boot_idx][drop_idx, :]
                    # sort the betas
                    sorted_betas = np.sort(boot_drop, axis=0)
                    # take bootstrap percentile as approxamition of CIs
                    # lower
                    lower_ci[drop_idx, :] = np.percentile(sorted_betas,
                                                          q=alpha/2,
                                                          method='closest_observation',
                                                          axis=0)
                    # upper
                    upper_ci[drop_idx, :] = np.percentile(sorted_betas,
                                                          q=1-alpha/2,
                                                          method='closest_observation',
                                                          axis=0)
        else:
            # take bootstrap std as approxamition of standard error
            # take lower tail
            lower_ci = betas - stats.norm.ppf(1-alpha/2) * betas_se
            # take upper tail
            upper_ci = betas + stats.norm.ppf(1-alpha/2) * betas_se
                
        # return confidence intervals
        return lower_ci, upper_ci


# define BaseScoreResults class
class BaseRSRScoreResults:
    """
    Base class for RSRScoreResults.
    This class should not be used directly. Use derived classes instead.
    """

    # define init function
    def __init__(self,
                 sample=None
                 ):
        
        # assign input values
        self.sample = sample
        
        # check the self outputs and set defaults
        self._output_checks()


    def _output_checks(self):
        """Output checks for the BaseRSRScoreResults class init."""
        
        # check and define the output parameters
        sample = self.sample

        # check if the sample is of class RSR
        if isinstance(sample, Reliability.RSR):
            # assign the value
            self.sample = sample
        else:
            # raise value error
            raise ValueError('sample must be a class of '
                             'SampleFit.Reliability.RSR '
                             f', got {type(sample)}.')
        
        # get exog and endog
        self.exog = self.sample.exog
        self.endog = self.sample.endog
        
        # initiliaze exog names, observations, etc
        self.exog_names = self.sample.exog_names
        self.n_obs = int(self.sample.n_obs)
        self.n_exog = self.sample.n_exog
        
        # get the class inputs
        self.min_samples = self.sample.min_samples
        self.loss = self.sample.loss
        self.boost = self.sample.boost

        # get the scores and compute gini coef
        self.scores = self.sample.scores
        self.gini = self.sample.gini


    # %% in-class functions
    # function to plot annealing sensitivity
    def plot(self,yname=None, xname=None, title=None, cmap=None, path=None,
             figsize=None, s=None, ylim=None, xlabel=None, dpi=None):
        """
        RSR Reliability Scores Plot.
        """
        
        # check inputs for plot
        self._check_plot_inputs(yname, xname, title, cmap, path, figsize, s,
                                ylim, xlabel, dpi)
        
        # set resolution
        plt.rcParams['figure.dpi'] = self.dpi
        plt.rcParams['savefig.dpi'] = self.dpi

        # get storage
        figures = {}
        iter_idx = 0

        # plot for each xname
        for var_name in self.xname:
            # get the variable idx
            var_idx = self.exog_names.index(var_name)
            
            # define the plot layout
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize = self.figsize)

            # plot scores
            plot = ax.scatter(x=self.exog[:, var_idx],
                              y=self.endog,
                              c=self.scores,
                              cmap=self.cmap,
                              s=self.s)
            # add titles, ticks, etc.
            ax.title.set_text(self.title)
            ax.set_xlabel(self.xlabel[iter_idx])
            ax.set_ylabel(self.yname)
            # set limits if specified
            if not self.ylim is None:
                ax.set_ylim(self.ylim)
            # add legend
            legend = ax.legend(*plot.legend_elements(),
                               title="Reliability Score",
                               bbox_to_anchor=(0., -0.3, 1., .102),
                               loc=3,
                               ncol=12,
                               mode="expand",
                               borderaxespad=0.,
                               fancybox=True,
                               shadow=True,
                               handletextpad=0.1)
            ax.add_artist(legend)
            plt.show()
            
            # save plot
            if not self.path is None:
                # save plot
                fig.savefig(self.path + '/' + var_name + '_scores.png',
                            bbox_inches='tight')
            
            # save figure
            figures[var_name] = (fig, ax, plot)
            # update index
            iter_idx += 1
        
        # asign figures to self
        self.figures = figures
        
        # restore the original resolution
        plt.rcParams['figure.dpi'] = 80
        plt.rcParams['savefig.dpi'] = 100

        # empty return
        return None


    # check inputs for score plot
    def _check_plot_inputs(self, yname, xname, title, cmap, path, figsize, s,
                           ylim, xlabel, dpi):
        """Input checks for the .plot() function."""
        
        # check name for y
        if yname is None:
            # set default as 'y'
            self.yname = 'y'
        # if supplied check if its valid
        elif isinstance(yname, str):
            # set value to user supplied
            self.yname = yname
        else:
            # raise value error
            raise ValueError("yname must be a string"
                             ", got %s" % type(yname))
        
        # check xname
        if xname is None:
            # plot all as default
            self.xname = self.exog_names
        # otherwise check inputs
        elif isinstance(xname, (list, tuple, str)):
            # check if they are admissible
            if isinstance(xname, str):
                # check in
                if xname in self.exog_names:
                    # assign the input value wrapped in a list
                    self.xname = [xname]
                else:
                    # raise value error
                    raise ValueError("xname must be one of:" +
                                     str(self.exog_names) + ""
                                     ", got %s" % xname)
            else:
                # check set in
                if set(xname) <= set(self.exog_names):
                    # assign the input value
                    self.xname = xname
                else:
                    # raise value error
                    raise ValueError("xname must be one of:" +
                                     str(self.exog_names) + ""
                                     ", got %s" % xname)
        else:
            # raise value error
            raise ValueError("xname must be one of tuple, list or string"
                             ", got %s" % type(xname))
        
        # check path for the plot
        if path is None:
            # do not save
            self.path = None
        elif isinstance(path, str):
            # assign the input value
            self.path = path
        else:
            # raise value error
            raise ValueError("path must be a string"
                             ", got %s" % type(path))
        
        # check path for the plot
        if figsize is None:
            # set default as 10,5
            self.figsize = (10, 5)
        elif isinstance(figsize, tuple):
            # assign the input value
            self.figsize = figsize
        else:
            # raise value error
            raise ValueError("figsize must be a tuple"
                             ", got %s" % type(path))
        
        # check for color
        if cmap is None:
            # define RdYlGn as default
            self.cmap = 'RdYlGn'
        # check if it is float
        elif isinstance(cmap, str):
            # check if its in the available colors maps
            if cmap in plt.colormaps():
                # get the user specified values
                self.cmap = cmap
            else:
                # raise error
                raise ValueError("The 'cmap' argument "
                                 "must be one of matplotlib color maps."
                                 ", got %s" % cmap)
        else:
            # raise error
            raise TypeError("The 'cmap' argument must be a string."
                            ", got %s" % type(cmap))
        
        # check for title
        if title is None:
            # define default
            self.title = 'RSR: Reliability Scores'
        # check if it is float
        elif isinstance(title, str):
            # assign input value
            self.title = title
        else:
            # raise error
            raise TypeError("The 'title' argument must be a string."
                            ", got %s" % type(title))
        
        # check name for ylim
        if ylim is None:
            # set default auto
            self.ylim = ylim
        # if supplied check if its valid
        elif isinstance(ylim, (tuple, list)):
            # set value to user supplied
            self.ylim = ylim
        else:
            # raise value error
            raise ValueError("ylim must be a tuple or a list"
                             ", got %s" % type(ylim))
        
        # check markersize s
        if s is None:
            # set default auto
            self.s = mpl.rcParams['lines.markersize'] ** 2
        # if supplied check if its valid
        elif isinstance(s, (float, int)):
            # set value to user supplied
            self.s = s
        else:
            # raise value error
            raise ValueError("s must be float or int"
                             ", got %s" % type(s))
        
        # check xname
        if xlabel is None:
            # plot xname as default
            self.xlabel = self.xname
        # otherwise check inputs
        elif isinstance(xlabel, (list, tuple, str)):
            # check if they are admissible
            if isinstance(xlabel, str):
                # check in
                if len(self.xname) == 1:
                    # assign the input value as list
                    self.xlabel = [xlabel]
                else:
                    # raise value error
                    raise ValueError("xlabel must be of same length as xname"
                                     ", got a single string.")
            else:
                # check if length of tuple and list matches xname
                if len(self.xname) == len(xlabel):
                    # assign the input value
                    self.xlabel = xlabel
                else:
                    # raise value error
                    raise ValueError("xlabel must be of same length as xname"
                                     ", got %s" % len(xlabel))
        else:
            # raise value error
            raise ValueError("xlabel must be one of tuple, list or string")
        
        # check dpi
        if dpi is None:
            # set default auto as in matplotlib
            self.dpi = 100
        # if supplied check if its valid
        elif isinstance(dpi, (float, int)):
            # set value to user supplied
            self.dpi = dpi
        else:
            # raise value error
            raise ValueError("dpi must be float or int"
                             ", got %s" % type(dpi))
