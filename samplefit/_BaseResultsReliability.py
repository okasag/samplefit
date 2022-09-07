"""
samplefit.

Python library to assess sample fit in econometric models via
the Sample Fit Reliability (SFR) approach as developed by
Okasa & Younge (2022).

Definition of SFR results classes and the methods.

"""

# import statsmodels, matplotlib and samplefit
import statsmodels
import matplotlib
import samplefit.Reliability as Reliability

# import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import submodules and functions
from scipy import stats
from matplotlib.colors import is_color_like


# %% SFR Results Classes
# define BaseFitResults class
class BaseSFRFitResults:
    """
    Base class for SFRFitResults.
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
        """Output checks for the BaseSFRFitResults class init."""
        
        # check and define the output parameters
        sample = self.sample
        params = self.params
        params_boot = self.params_boot
        stand_err = self.stand_err
        fittedvalues = self.fittedvalues

        # check if the sample is of class SFR
        if isinstance(sample, Reliability.SFR):
            # assign the value
            self.sample = sample
        else:
            # raise value error
            raise ValueError('sample must be a class of '
                             'samplefit.Reliability.SFR '
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
    # function to predict yhats based on the weighted fit
    def predict(self, params=None, exog=None):
        """
        SFR prediction after fit.
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
        SFR confidence intervals after fit.
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
                percentile=False, get_table=False, verbose=True):
        """
        SFR fit summary.
        """
        
        # check inputs for summary
        self._check_summary_inputs(yname, xname, title, alpha, percentile,
                                   get_table, verbose)
        
        # get the preamble
        # info about SFR
        sfr_info = ['No. Samples:', 'Min. Samples:', 'Loss:', 'Gini:']
        sfr_out = [self.sample.n_samples, self.min_samples,
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
        
        # check if printing to a console
        if self.verbose:
            # print the preamble title
            print('\n',
                  f"{self.title : ^80}",
                  '=' * 80,
                  '-' * 10 + ' SFR Results ' + '-' * 34 + ' Fit Results ' + '-' * 10,
                  '=' * 80,
                  sep='\n')
            # print the preamble data
            for idx in range(len(sfr_info)):
                # print the preambe
                print(f"{sfr_info[idx]:<15}{sfr_out[idx]:>20}          {fit_info[idx]:<20}{fit_out[idx]:>15}")
            # print params summary
            print('=' * 80,
                  summary_table.to_string(justify='right', line_width=80, col_space=10),
                  '=' * 80,
                  sep='\n')

        # return table or None
        return summary_table if self.get_table else None
    
    
    #%% non-user in-class functions
    # function to compute conf intervals via asymptotic approximation
    def _asym_ci(self, betas=None, betas_se=None, alpha=0.05):
        """Compute the confidence intervals via asymptotics."""
        
        # compute confidence intervals via asymptotics
        # take lower tail
        lower_ci = betas - stats.t.ppf(1-alpha/2, self.df_resid) * betas_se
        # take upper tail
        upper_ci = betas + stats.t.ppf(1-alpha/2, self.df_resid) * betas_se
                
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
            # take bootstrap std as approximation of standard error
            # take lower tail
            lower_ci = betas - stats.t.ppf(1-alpha/2, self.df_resid) * betas_se
            # take upper tail
            upper_ci = betas + stats.t.ppf(1-alpha/2, self.df_resid) * betas_se
                
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
    def _check_summary_inputs(self, yname, xname, title, alpha, percentile,
                              get_table, verbose):
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
            # set default as SFR title
            self.title = 'SFR: Fitting'
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
                             ", got %s" % type(percentile))
        
        # check whether to return table or not
        if isinstance(get_table, bool):
            # assign to self
            self.get_table = get_table
        else:
            # raise value error
            raise ValueError("get_table must be of type boolean"
                             ", got %s" % type(get_table))
        
        # check whether to print to console or not
        if isinstance(verbose, bool):
            # assign to self
            self.verbose = verbose
        else:
            # raise value error
            raise ValueError("verbose must be of type boolean"
                             ", got %s" % type(verbose))


# define BaseAnnealResults class
class BaseSFRAnnealResults:
    """
    Base class for SFRAnnealResults.
    This class should not be used directly. Use derived user classes instead.
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
        """Output checks for the BaseSFRAnnealResults class init."""
        
        # check and define the output parameters
        sample = self.sample
        params = self.params
        params_boot = self.params_boot
        stand_err = self.stand_err
        drop_idx = self.drop_idx

        # check if the sample is of class SFR
        if isinstance(sample, Reliability.SFR):
            # assign the value
            self.sample = sample
        else:
            # raise value error
            raise ValueError('sample must be a class of '
                             'SampleFit.Reliability.SFR '
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
        self.df_resid = self.n_obs - self.n_exog
        
        # get the class inputs
        self.min_samples = self.sample.min_samples
        self.loss = self.sample.loss
        self.share = self.sample.share
        self.n_boot = self.sample.n_boot

        # get the scores and compute gini coef
        self.scores = self.sample.scores
        self.gini = self.sample.gini


    # %% in-class functions
    # function to plot annealing sensitivity
    def plot(self,
             yname=None, xname=None, title=None, alpha=0.05, percentile=False,
             color=None, path=None, figsize=None, ylim=None, xlabel=None,
             dpi=None, fname=None):
        """
        SFR Annealing Plot.
        """
        
        # check inputs for plot
        self._check_plot_inputs(
            yname, xname, title, color, path, figsize, ylim, xlabel, dpi, fname)
        
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
            
            # add titles, ticks, etc.
            ax.title.set_text(self.title)
            ax.set_xlabel('Sample Share Dropped')
            ax.set_ylabel(self.yname)
            labels = list(np.round(np.linspace(0, self.share, 11), 2))
            ticks = list(np.round(np.linspace(0, x_width, 11), 0))
            plt.xticks(ticks, labels)
            plt.legend(loc="best")
            plt.show()
            
            # save plot
            if not self.path is None:
                # save plot
                if not self.fname is None:
                    fig.savefig(self.path + '/' + self.fname,
                                bbox_inches='tight')
                else:
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
        SFR confidence intervals after annealing.
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
    def _check_plot_inputs(self, yname, xname, title, color, path, figsize,
                           ylim, xlabel, dpi, fname):
        """Input checks for the .plot() function."""
        
        # check name for y
        if yname is None:
            # set default as 'Effect'
            self.yname = 'Effect'
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
        
        # check fname for plot
        if fname is None:
            # set default as None
            self.fname = None
        # if supplied check if its valid
        elif isinstance(fname, str):
            # set value to user supplied
            self.fname = fname
        else:
            # raise value error
            raise ValueError("fname must be a string"
                             ", got %s" % type(fname))
        
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
            self.title = 'SFR: Annealing'
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
        lower_ci = betas - stats.t.ppf(1-alpha/2, self.df_resid) * betas_se
        # take upper tail
        upper_ci = betas + stats.t.ppf(1-alpha/2, self.df_resid) * betas_se
                
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
            lower_ci = betas - stats.t.ppf(1-alpha/2, self.df_resid) * betas_se
            # take upper tail
            upper_ci = betas + stats.t.ppf(1-alpha/2, self.df_resid) * betas_se
                
        # return confidence intervals
        return lower_ci, upper_ci


# define BaseScoreResults class
class BaseSFRScoreResults:
    """
    Base class for SFRScoreResults.
    This class should not be used directly. Use derived user classes instead.
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
        """Output checks for the BaseSFRScoreResults class init."""
        
        # check and define the output parameters
        sample = self.sample

        # check if the sample is of class SFR
        if isinstance(sample, Reliability.SFR):
            # assign the value
            self.sample = sample
        else:
            # raise value error
            raise ValueError('sample must be a class of '
                             'SampleFit.Reliability.SFR '
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

        # get the scores and compute gini coef
        self.scores = self.sample.scores
        self.gini = self.sample.gini


    # %% in-class functions
    # function to plot reliability scores
    def plot(self, yname=None, xname=None, title=None, cmap=None, path=None,
             figsize=None, s=None, ylim=None, xlim=None, xlabel=None, dpi=None,
             fname=None, jitter=False):
        """
        SFR Scoring Plot.
        """
        
        # check inputs for plot
        self._check_plot_inputs(yname, xname, title, cmap, path, figsize, s,
                                ylim, xlim, xlabel, dpi, fname, jitter)
        
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

            # check if X is categorical
            if np.sum(self.exog[:, var_idx].astype(int) - 
                      self.exog[:, var_idx]) == 0:
                # get distinct values
                cat_values = list(np.sort(np.unique(self.exog[:, var_idx])
                                          ).astype(int))
                # check if jitter should be applied
                if self.jitter:
                    # apply jitter random noise for visualisation purposes
                    exog_jitter = (self.exog[:, var_idx].copy() + 
                                   np.random.uniform(-0.1, 0.1,
                                                     len(self.endog)))
                else:
                    # keep original values
                    exog_jitter = self.exog[:, var_idx].copy()
                # scatter plot
                plot = ax.scatter(x=exog_jitter,
                                  y=self.endog,
                                  c=self.scores,
                                  cmap=self.cmap,
                                  s=self.s)
                # plot ticks
                cat_values.insert(0, (np.min(cat_values) - 0.5))
                cat_values.append((np.max(cat_values) + 0.5))
                ticks = cat_values.copy()
                # add ticks
                plt.xticks(ticks, cat_values)
            else:
                # scatter plot
                plot = ax.scatter(x=self.exog[:, var_idx],
                                  y=self.endog,
                                  c=self.scores,
                                  cmap=self.cmap,
                                  s=self.s)

            # add titles, labels, etc.
            ax.title.set_text(self.title)
            ax.set_xlabel(self.xlabel[iter_idx])
            ax.set_ylabel(self.yname)
            # set limits if specified
            if not self.ylim is None:
                ax.set_ylim(self.ylim)
            if not self.xlim is None:
                ax.set_xlim(self.xlim)
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
                if not self.fname is None:
                    fig.savefig(self.path + '/' + self.fname,
                                bbox_inches='tight')
                else:
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
                           ylim, xlim, xlabel, dpi, fname, jitter):
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
        
        # check fname for plot
        if fname is None:
            # set default as None
            self.fname = None
        # if supplied check if its valid
        elif isinstance(fname, str):
            # set value to user supplied
            self.fname = fname
        else:
            # raise value error
            raise ValueError("fname must be a string"
                             ", got %s" % type(fname))
        
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
        elif isinstance(cmap, (str, matplotlib.colors.ListedColormap)):
            # if string, check admissibility
            if isinstance(cmap, str):
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
                # get the user specified value
                    self.cmap = cmap
        else:
            # raise error
            raise TypeError("The 'cmap' argument must be a string or "
                            "ListedColormap, got %s" % type(cmap))
        
        # check for title
        if title is None:
            # define default
            self.title = 'SFR: Scoring'
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
        
        # check name for xlim
        if xlim is None:
            # set default auto
            self.xlim = xlim
        # if supplied check if its valid
        elif isinstance(xlim, (tuple, list)):
            # set value to user supplied
            self.xlim = xlim
        else:
            # raise value error
            raise ValueError("xlim must be a tuple or a list"
                             ", got %s" % type(xlim))
        
        # check markersize s
        if s is None:
            # set default auto
            self.s = matplotlib.rcParams['lines.markersize'] ** 2
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
        
        # check jitter
        if isinstance(jitter, bool):
            # assign value
            self.jitter = jitter
        else:
            # raise value error
            raise ValueError("jitter must be boolean"
                             ", got %s" % type(jitter))
