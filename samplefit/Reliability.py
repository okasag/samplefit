"""
samplefit.

Python library to assess sample fit in econometric models via
the Sample Fit Reliability (SFR) approach as developed by
Okasa & Younge (2022).

Definition of main user classes.

"""

from samplefit._BaseReliability import BaseSFR
from samplefit._BaseResultsReliability import BaseSFRFitResults
from samplefit._BaseResultsReliability import BaseSFRAnnealResults
from samplefit._BaseResultsReliability import BaseSFRScoreResults

class SFR(BaseSFR):
    """
    Sample Fit Reliability class labeled `SFR()`. Initializes
    parameters for sample fit.

    Parameters
    ----------
    linear_model : statsmodels class
        Linear model specified via statsmodels OLS or GLM class.
    n_samples : int
        The number of sub-samples in the re-sampling procedure.
        The default is 1000.
    min_samples : int, float or NoneType
        Minimum number of observations for each sub-sample, i.e. number of
        observations to draw from the data without replacement. If integer
        supplied, exact number of observation is sampled. If float, share of
        full sample is considered (rounded up). If None, the minimum number of
        observations to estimate the model is selected, i.e p+1 (reccommended),
        where p is number of model parameters. The default is None.
    loss : str or lambda function
        Loss function for evaluation of the estimation errors. Loss must be
        either 'absolute_error' (reccommended) or 'squared_error'. For a user
        defined loss function, user can directly supply own lambda function of
        type: 'lambda y, yhat:'. Default is 'absolute_error'.
    n_jobs : int or NoneType
        The number of parallel jobs to be used for multithreading in
        [`.fit()`](#samplefit.Reliability.SFR.fit),
        [`.score()`](#samplefit.Reliability.SFR.score) and
        [`.anneal()`](#samplefit.Reliability.SFR.anneal).
        Follows
        [`joblib`](https://joblib.readthedocs.io){:target="_blank"} semantics:

        - `n_jobs=-1` means all - 1 available cpu physical cores.
        - `n_jobs=None` and `n_jobs=1` means no parallelism.

        The default is -1.
    random_state : int, NoneType or numpy.random.RandomState object
        Random seed used to initialize the pseudo-random number
        generator. See
        [`numpy` documentation](https://numpy.org/doc/stable/reference/random/legacy.html){:target="_blank"}
        for details. If None specified, 0 is used. The default is None.

    Returns
    -------
    Initializes SFR class. Following methods are available:
    .fit(), .score() and .anneal().


    Notes
    -----
    `SFR()` includes methods to [`.fit()`](#samplefit.Reliability.SFR.fit),
    [`.score()`](#samplefit.Reliability.SFR.score) and
    [`.anneal()`](#samplefit.Reliability.SFR.anneal).

    For further details, see examples below.

    Examples
    --------
    ```py
    # import libraries
    import samplefit as sf
    import statsmodels.api as sm
    
    # get data 
    boston = sm.datasets.get_rdataset("Boston", "MASS")
    Y = boston.data['crim'] # per capita crime rate
    X = boston.data['lstat'] # % lower status population
    X = sm.add_constant(X)
    
    # assess model fit
    model = sm.OLS(endog=Y, exog=X)
    model_fit = model.fit()
    model_fit.summary()
    
    # assess sample fit
    sample = sf.SFR(linear_model=model)
    sample_fit = sample.fit()
    sample_fit.summary()
    
    # assess sample sensitivity
    sample_annealing = sample.anneal()
    sample_annealing.plot()
    
    # assess sample reliability
    sample_scores = sample.score()
    sample_scores.plot()
    ```
    """

    # define init function
    def __init__(self,
                 linear_model=None,
                 n_samples=1000,
                 min_samples=None,
                 loss=None,
                 n_jobs=-1,
                 random_state=None):
        # access inherited methods
        super().__init__(
            linear_model=linear_model,
            n_samples=n_samples,
            min_samples=min_samples,
            loss=loss,
            n_jobs=n_jobs,
            random_state=random_state
        )


    def fit(self,
            weights=None,
            n_boot=None):
        """
        Sample fit based on the reliability scores via the SFR algorithm.

        Parameters
        ----------
        weights : array-like of shape (n_obs, 1) or NoneType
            An array of weights for weighted regression. If None, squared
            reliability scores will be used as weights as a default. Note, that
            if bootstrapping is used for inference, the estimation of
            user-supplied weights is not reflected. Default is None.
        n_boot : int or NoneType
            Number of bootstrap replications for inference. If None specified,
            asymptotic approximation is used for inference instead. For valid
            inference, bootstrapping is recommended. Note that bootstrapping
            requires longer computation time. Default is None.

        Returns
        -------
        Results of class SFRFitResults. Following methods are available:
        .summary(), .conf_int() and .predict().

        Notes
        -----
        [`.fit()`](#samplefit.Reliability.SFR.fit) estimates the reliability
        scores via the SFR algorithm in the first step and estimates weighted
        regression in the second step, with the squared reliability scores as
        weights if not specified otherwise.

        Examples
        --------
        ```py
        # import libraries
        import samplefit as sf
        import statsmodels.api as sm
        
        # get data 
        boston = sm.datasets.get_rdataset("Boston", "MASS")
        Y = boston.data['crim'] # per capita crime rate
        X = boston.data['lstat'] # % lower status population
        X = sm.add_constant(X)
        
        # specify model
        model = sm.OLS(endog=Y, exog=X)
        
        # specify sample
        sample = sf.SFR(linear_model=model)
        
        # sample fit with defaults
        sample_fit = sample.fit()
        
        # sample fit with bootstrapping
        sample_fit = sample.fit(n_boot=1000)
        
        # get summary of sample fit
        sample_fit.summary()
        
        # get confidence intervals
        ci_low, ci_up = sample_fit.conf_int()
        
        # get predictions (in-sample)
        preds = sample_fit.predict()
        ```
        """
        return super().fit(
            weights=weights,
            n_boot=n_boot
            )


    def score(self):
        """
        Estimation of reliability scores via the SFR algorithm.

        Parameters
        ----------
        None.

        Returns
        -------
        Results of class SFRScoreResults. Following methods are available:
        .plot().

        Notes
        -----
        [`.score()`](#samplefit.Reliability.SFR.score) estimates the
        reliability scores via the SFR algorithm. Each observation is scored
        for the reliability with 0 being the most unreliable observation and
        with 1 being the most reliable observation within a sample.

        Examples
        --------
        ```py
        # import libraries
        import samplefit as sf
        import statsmodels.api as sm
        
        # get data 
        boston = sm.datasets.get_rdataset("Boston", "MASS")
        Y = boston.data['crim'] # per capita crime rate
        X = boston.data['lstat'] # % lower status population
        X = sm.add_constant(X)
        
        # specify model
        model = sm.OLS(endog=Y, exog=X)
        
        # specify sample
        sample = sf.SFR(linear_model=model)
        
        # score reliability
        sample_scores = sample.score()
        
        # extract reliability scores
        scores = sample_scores.scores
        
        # plot reliability scores
        sample_scores.plot()
        ```
        """
        return super().score()


    def anneal(self, share=0.1, n_boot=None):
        """
        Sample annealing based on the reliability scores via the SFR algorithm.

        Parameters
        ----------
        share : float or NoneType
            Share of sample that gets annealed based on the most unreliable
            observations. Default is 0.1.
        n_boot : int or NoneType
            Number of bootstrap replications for inference. If None specified,
            asymptotic approximation is used for inference instead. For valid
            inference, bootstrapping is recommended. Note that bootstrapping
            requires longer computation time. Default is None.

        Returns
        -------
        Results of class SFRAnnealResults. Following methods are available:
        .conf_int() and .plot().

        Notes
        -----
        [`.anneal()`](#samplefit.Reliability.SFR.anneal) re-estimates the model
        while sequentially dropping the most unreliable observations. Such
        annealing procedure helps to assess the sample sensitivity and detect
        how much the parameters depend on particularly unreliable observations.

        Examples
        --------
        ```py
        # import libraries
        import samplefit as sf
        import statsmodels.api as sm
        
        # get data 
        boston = sm.datasets.get_rdataset("Boston", "MASS")
        Y = boston.data['crim'] # per capita crime rate
        X = boston.data['lstat'] # % lower status population
        X = sm.add_constant(X)
        
        # specify model
        model = sm.OLS(endog=Y, exog=X)
        
        # specify sample
        sample = sf.SFR(linear_model=model)
        
        # sample annealing with defaults
        sample_annealing = sample.anneal()
        
        # sample annealing with specified share
        sample_annealing = sample.anneal(share=0.1)
        
        # sample annealing with bootstrapping
        sample_annealing = sample.anneal(n_boot=1000)
        
        # get confidence intervals
        ci_low, ci_up = sample_annealing.conf_int()
        
        # get annealing plot
        sample_annealing.plot()
        ```
        """
        return super().anneal(
            share=share,
            n_boot=n_boot
            )


# class for FitResults
class SFRFitResults(BaseSFRFitResults):
    """
    Fit Results class labeled `SFRFitResults()`.
    Initializes output of SFR.fit().

    """

    # define init function
    def __init__(self,
                 sample=None,
                 params=None,
                 params_boot=None,
                 stand_err=None,
                 fittedvalues=None
                 ):
        # access inherited methods
        super().__init__(
            sample=sample,
            params=params,
            params_boot=params_boot,
            stand_err=stand_err,
            fittedvalues=fittedvalues
        )


    def predict(self,
                params=None,
                exog=None):
        """
        Predict outcomes based on the sample fit via the SFR algorithm.
        
        Parameters
        ----------
        params : array-like or NoneType
            Array of parameters to predict with. If None supplied, the
            estimated parameters from the sample fit will be used.
            Default is None.
        exog : array-like or NoneType
            Matrix of features/covariates for which the outcomes should be
            predicted (out-of-sample). Column dimensions must be identical
            to the training data supplied to the statsmodels model class.
            If None supplied, in-sample predictions (fitted values) are
            returned. Default is None.

        Returns
        -------
        Array of predictions.

        Notes
        -----
        [`.predict()`](#samplefit.Reliability.SFRFitResults.predict) constructs
        predictions for outcome variable based on the estimated parameters.
        Predictions are based on the parameters of weighted fit. If no new
        values for exogeneous variables are supplied, fitted values are
        returned.

        Examples
        --------
        ```py
        # import libraries
        import samplefit as sf
        import statsmodels.api as sm
        
        # get data 
        boston = sm.datasets.get_rdataset("Boston", "MASS")
        Y = boston.data['crim'] # per capita crime rate
        X = boston.data['lstat'] # % lower status population
        X = sm.add_constant(X)
        
        # specify model
        model = sm.OLS(endog=Y, exog=X)
        
        # specify sample
        sample = sf.SFR(linear_model=model)
        
        # sample fit
        sample_fit = sample.fit()
        
        # predict in-sample
        preds = sample_fit.predict()
        
        # predict out-of-sample
        preds = sample_fit.predict(exog=X[0, :])
        ```
        """
        return super().predict(
            params=params,
            exog=exog
            )
    
    
    def conf_int(self,
                 alpha=0.05,
                 percentile=False):
        """
        Confidence intervals based on the sample fit via the SFR algorithm.
        
        Parameters
        ----------
        alpha : float or NoneType
            Confidence level alpha. Default is 0.05.
        percentile : bool
            Percentile method for confidence intervals based on bootstrapping.
            If bootstrapping has not been used for fitting, it is ignored.
            Default is False.

        Returns
        -------
        Tuple of arrays of confidence bounds, lower and upper.

        Notes
        -----
        [`.conf_int()`](#samplefit.Reliability.SFRFitResults.conf_int)
        constructs confidence intervals for estimated paramaters. If fitted
        without bootstrapping, asymptotic approximations are used. If fitted
        with bootstrapping, the standard deviation of bootstrapped parameters
        is used for standard error approximation. If percentile=True, the
        percentile method is used instead of standard deviation.

        Examples
        --------
        ```py
        # import libraries
        import samplefit as sf
        import statsmodels.api as sm
        
        # get data 
        boston = sm.datasets.get_rdataset("Boston", "MASS")
        Y = boston.data['crim'] # per capita crime rate
        X = boston.data['lstat'] # % lower status population
        X = sm.add_constant(X)
        
        # specify model
        model = sm.OLS(endog=Y, exog=X)
        
        # specify sample
        sample = sf.SFR(linear_model=model)
        
        # sample fit
        sample_fit = sample.fit()
        
        # compute confidence intervals with default settings
        ci_low, ci_up = sample_fit.conf_int()
        
        # compute confidence intervals with custom alpha
        ci_low, ci_up = sample_fit.conf_int(alpha=0.1)
        ```
        """
        return super().conf_int(
            alpha=alpha,
            percentile=percentile
            )
    
    
    def summary(self,
                yname=None,
                xname=None,
                title=None,
                alpha=0.05,
                percentile=False,
                get_table=False,
                verbose=True):
        """
        Summary of the sample fit via the SFR algorithm.
        
        Parameters
        ----------
        yname : str or NoneType
            Name of the endog variable. Default is 'y'.
        xname : list, tuple or NoneType
            List of name of the exog variables. Must have the same dimension as
            exog columns. Default are the supplied exog names.
        title : str or NoneType
            Title for the summary table. Default is 'SFR: Fitting'.
        alpha : float or NoneType
            Confidence level alpha. Default is 0.05.
        percentile : bool
            Percentile method for confidence intervals based on bootstrapping.
            If bootstrapping has not been used for fitting, it is ignored.
            Default is False.
        get_table : bool
            If a summary table should be returned or not. If True, a pandas
            DataFrame with estimation results is returned. Default is False.
        verbose : bool
            If a summary table should be printed to console or not.
            Default is True.

        Returns
        -------
        None. Prints summary table.

        Notes
        -----
        [`.summary()`](#samplefit.Reliability.SFRFitResults.summary) produces
        a summary table including information on sample fit as well as model
        fit together with parameters, standard errors, t-values, p-values
        and confidence intervals.

        Examples
        --------
        ```py
        # import libraries
        import samplefit as sf
        import statsmodels.api as sm
        
        # get data 
        boston = sm.datasets.get_rdataset("Boston", "MASS")
        Y = boston.data['crim'] # per capita crime rate
        X = boston.data['lstat'] # % lower status population
        X = sm.add_constant(X)
        
        # specify model
        model = sm.OLS(endog=Y, exog=X)
        
        # specify sample
        sample = sf.SFR(linear_model=model)
        
        # sample fit
        sample_fit = sample.fit()
        
        # default summary
        sample_fit.summary()
        
        # custom summary title
        sample_fit.summary(title='My Title')
        ```
        """
        return super().summary(
            yname=yname,
            xname=xname,
            title=title,
            alpha=alpha,
            percentile=percentile,
            get_table=get_table,
            verbose=verbose
            )


# class for AnnealResults
class SFRAnnealResults(BaseSFRAnnealResults):
    """
    Annealing results class labeled `SFRAnnealResults()`.
    Initializes output of SFR.anneal().

    """

    # define init function
    def __init__(self,
                 sample=None,
                 params=None,
                 params_boot=None,
                 stand_err=None,
                 drop_idx=None
                 ):
        # access inherited methods
        super().__init__(
            sample=sample,
            params=params,
            params_boot=params_boot,
            stand_err=stand_err,
            drop_idx=drop_idx
        )


    def plot(self,
             yname=None,
             xname=None,
             title=None,
             alpha=0.05,
             percentile=False,
             color=None,
             path=None,
             figsize=None,
             ylim=None,
             xlabel=None,
             dpi=None,
             fname=None):
        """
        Plot the Annealing based on the reliability scores from the SFR
        algorithm.
        
        Parameters
        ----------
        yname : str or NoneType
            Name of the y axis. Default is 'Effect'.
        xname : list, tuple, str or NoneType
            Name or list of names of the exog variables for which parameter
            an annealing plot should be constructed. Must be one of the exog 
            variable names. If not supplied annealing plots for all parameters
            are constructed. Default are the supplied exog names.
        title : str or NoneType
            Title for the annealing plot. Default is 'SFR: Annealing'.
        alpha : float or NoneType
            Confidence level alpha. Default is 0.05.
        percentile : bool
            Percentile method for confidence intervals based on bootstrapping.
            If bootstrapping has not been used for annealing, it is ignored.
            Default is False.
        color : str or NoneType
            Color used for the confidence interval. Must be one of the
            matplotlib supported colors. Default is grey.
        path : str or NoneType
            Valid path to save the plot. If None, plot is not saved. Default
            is None.
        figsize : tuple or NoneType
            Tuple of x and y axis size for matplotlib figsize argument.
            Default is (10,5).
        ylim : tuple, list or NoneType
            Tuple of upper and lower limits of y axis. Default is automatic.
        xlabel : str or NoneType
            Label for the x axis for the exog variable. Default is 'xname'.
        dpi : float, int or NoneType
            The resolution for matplotlib scatter plot. Default is 100.
        fname : str or NoneType
            Valid figure name to save the plot. If None, generic name is used.
            Default is None.
            

        Returns
        -------
        Dictionary of matplotlib figures and axes. Prints annealing plots.

        Notes
        -----
        [`.plot()`](#samplefit.Reliability.SFRAnnealResults.plot) produces
        an annealing plot for assessment of sample fit sensitivity, together
        with parameters and confidence intervals.

        Examples
        --------
        ```py
        # import libraries
        import samplefit as sf
        import statsmodels.api as sm
        
        # get data 
        boston = sm.datasets.get_rdataset("Boston", "MASS")
        Y = boston.data['crim'] # per capita crime rate
        X = boston.data['lstat'] # % lower status population
        X = sm.add_constant(X)
        
        # specify model
        model = sm.OLS(endog=Y, exog=X)
        
        # specify sample
        sample = sf.SFR(linear_model=model)
        
        # sample annealing
        sample_annealing = sample.anneal()
        
        # default annealing plot
        sample_annealing.plot()
        
        # custom annealing
        sample_annealing.plot(title='My Title')
        ```
        """
        return super().plot(
            yname=yname,
            xname=xname,
            title=title,
            alpha=alpha,
            percentile=percentile,
            color=color,
            path=path,
            figsize=figsize,
            ylim=ylim,
            xlabel=xlabel,
            dpi=dpi,
            fname=fname
            )
    
    
    def conf_int(self,
                 alpha=0.05,
                 percentile=False):
        """
        Confidence intervals based on the annealing via the SFR algorithm.
        
        Parameters
        ----------
        alpha : float or NoneType
            Confidence level alpha. Default is 0.05.
        percentile : bool
            Percentile method for confidence intervals based on bootstrapping.
            If bootstrapping has not been used for annealing, it is ignored.
            Default is False.

        Returns
        -------
        Tuple of arrays of confidence bounds, lower and upper.

        Notes
        -----
        [`.conf_int()`](#samplefit.Reliability.SFRAnnealResults.conf_int)
        constructs confidence intervals for estimated paramaters. If annealed
        without bootstrapping, asymptotic approximations are used. If annealed
        with bootstrapping, the standard deviation of bootstrapped parameters
        is used for standard error approximation. If percentile=True, the
        percentile method is used instead of standard deviation.

        Examples
        --------
        ```py
        # import libraries
        import samplefit as sf
        import statsmodels.api as sm
        
        # get data 
        boston = sm.datasets.get_rdataset("Boston", "MASS")
        Y = boston.data['crim'] # per capita crime rate
        X = boston.data['lstat'] # % lower status population
        X = sm.add_constant(X)
        
        # specify model
        model = sm.OLS(endog=Y, exog=X)
        
        # specify sample
        sample = sf.SFR(linear_model=model)
        
        # sample annealing
        sample_annealing = sample.anneal()
        
        # compute confidence intervals with default settings
        ci_low, ci_up = sample_annealing.conf_int()
        
        # compute confidence intervals with custom alpha
        ci_low, ci_up = sample_fit.conf_int(alpha=0.1)
        ```
        """
        return super().conf_int(
            alpha=alpha,
            percentile=percentile
            )


# class for ScoreResults
class SFRScoreResults(BaseSFRScoreResults):
    """
    Scoring results class labeled `SFRScoreResults()`.
    Initializes output of SFR.score().

    """

    # define init function
    def __init__(self,
                 sample=None
                 ):
        # access inherited methods
        super().__init__(
            sample=sample
            )


    def plot(self,
             yname=None,
             xname=None,
             title=None,
             cmap=None,
             path=None,
             figsize=None,
             s=None,
             ylim=None,
             xlim=None,
             xlabel=None,
             dpi=None,
             fname=None,
             jitter=False):
        """
        Plot the reliability scores based on the SFR algorithm.
        
        Parameters
        ----------
        yname : str or NoneType
            Name of the endog variable. Default is 'y'.
        xname : list, tuple, str or NoneType
            Name or list of names of the exog variables for which parameter
            an scoring plot should be constructed. Must be one of the exog 
            variable names. If not supplied scoring plots for all parameters
            are constructed. Default are the supplied exog names.
        title : str or NoneType
            Title for the scoring plot. Default is 'SFR: Scoring'.
        cmap : str or NoneType
            Color map used for the reliability score. Must be one of the
            matplotlib supported color maps. Default is 'RdYlGn'.
        path : str or NoneType
            Valid path to save the plot. If None, plot is not saved. Default
            is None.
        figsize : tuple or NoneType
            Tuple of x and y axis size for matplotlib figsize argument.
            Default is (10,5).
        s : float, int or NoneType
            The marker size in points**2 as for in matplotlib scatter plot.
            Default is automatic.
        ylim : tuple, list or NoneType
            Tuple of upper and lower limits of y axis. Default is automatic.
        xlim : tuple, list or NoneType
            Tuple of upper and lower limits of x axis. Default is automatic.
        xlabel : str or NoneType
            Label for the x axis for the exog variable. Default is 'xname'.
        dpi : float, int or NoneType
            The resolution for matplotlib scatter plot. Default is 100.
        fname : str or NoneType
            Valid figure name to save the plot. If None, generic name is used.
            Default is None.
        jitter : bool
            Logical, if scatterplot should be jittered for categorical
            features. Note, that this involves random perturbation of the
            values of features along X axis, fixing seed is thus necessary for
            reproducibility. Default is False.
    
        Returns
        -------
        Dictionary of matplotlib figures and axes. Prints scoring plots.

        Notes
        -----
        [`.plot()`](#samplefit.Reliability.SFRScoreResults.plot) produces
        a scoring plot for assessment of sample fit reliability.

        Examples
        --------
        ```py
        # import libraries
        import samplefit as sf
        import statsmodels.api as sm
        
        # get data 
        boston = sm.datasets.get_rdataset("Boston", "MASS")
        Y = boston.data['crim'] # per capita crime rate
        X = boston.data['lstat'] # % lower status population
        X = sm.add_constant(X)
        
        # specify model
        model = sm.OLS(endog=Y, exog=X)
        
        # specify sample
        sample = sf.SFR(linear_model=model)
        
        # sample reliability
        sample_scores = sample.score()
        
        # default scoring plot
        sample_scores.plot()
        
        # custom scoring
        sample_scores.plot(title='My Title')
        ```
        """
        return super().plot(
            yname=yname,
            xname=xname,
            title=title,
            cmap=cmap,
            path=path,
            figsize=figsize,
            s=s,
            ylim=ylim,
            xlim=xlim,
            xlabel=xlabel,
            dpi=dpi,
            fname=fname,
            jitter=jitter
            )
