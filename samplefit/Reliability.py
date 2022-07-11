"""
samplefit: Random Sample Reliability.

Python library to assess Sample Fit via the Random Sample Reliability
algorithm as developed by Okasa & Younge (2022).

Definition of main user classes.

"""

from samplefit._BaseReliability import BaseRSR
from samplefit._BaseResultsReliability import BaseRSRFitResults
from samplefit._BaseResultsReliability import BaseRSRAnnealResults
from samplefit._BaseResultsReliability import BaseRSRScoreResults

class RSR(BaseRSR):
    """
    Random Sample Reliability class labeled `RSR()`. Initializes
    parameters for sample fit.

    Parameters
    ----------
    linear_model : statsmodels class
        Linear model specified via statsmodels OLS or GLM class.
    n_samples : int
        The number of samples in the sub-sampling. The default is 1000.
    min_samples : int, float or NoneType
        Minimum number of observations for each subsample, i.e. number of
        observations to draw from the data without replacement. If integer
        supplied, exact number of observation is sampeld. If float, share of
        full sample is considered (rounded up). If None, the minimum number of
        observations to estimate the model is selected, i.e p+1, where p is
        number of model parameters. The default is None.
    loss : str or lambda function
        Loss function for evaluation of the estimation errors. Loss must be
        either 'squared_error' or 'absolute_error'. For a user defined loss
        function, user can directly supply own lambda function of type:
        'lambda y, yhat:'. Default is 'absolute_error'.
    boost : float or NoneType
        Share of sample that should be boosted, i.e. the share of observations
        that is sequentially dropped from the sample based on the iteratively
        lowest reliability scores. The final boosted reliability scores are
        then estimated based on the boosted sample that excludes boost \% of
        the most unreliable observations from the sample. Boosting should be
        applied only when there is a prior knowledge of the outlier share.
        The default is None.
    n_jobs : int or NoneType
        The number of parallel jobs to be used for multithreading in
        [`.fit()`](#samplefit.Reliability.RSR.fit),
        [`.score()`](#samplefit.Reliability.RSR.score) and
        [`.anneal()`](#samplefit.Reliability.RSR.anneal).
        Follows
        [`joblib`](https://joblib.readthedocs.io){:target="_blank"} semantics:

        - `n_jobs=-1` means all - 1 available cpu cores.
        - `n_jobs=None` and `n_jobs=1` means no parallelism.

        The default is -1.
    random_state : int, None or numpy.random.RandomState object
        Random seed used to initialize the pseudo-random number
        generator. See
        [`numpy` documentation](https://numpy.org/doc/stable/reference/random/legacy.html){:target="_blank"}
        for details. The default is None.

    Returns
    -------
    Initializes RSR class. Following methods are available:
    .fit(), .score() and .anneal().


    Notes
    -----
    `RSR()` includes methods to [`.fit()`](#samplefit.Reliability.RSR.fit),
    [`.score()`](#samplefit.Reliability.RSR.score) and
    [`.anneal()`](#samplefit.Reliability.RSR.anneal).

    For further details, see examples below.

    Examples
    --------
    ```py
    # import libraries
    import samplefit as sf
    import statsmodels.api as sm
    
    # get data 
    boston = sm.datasets.get_rdataset("Boston", "MASS")
    Y = boston.data['medv'] # median house price
    X = boston.data['rm'] # number of rooms
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
    ```
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
        # access inherited methods
        super().__init__(
            linear_model=linear_model,
            n_samples=n_samples,
            min_samples=min_samples,
            loss=loss,
            boost=boost,
            n_jobs=n_jobs,
            random_state=random_state
        )


    def fit(self,
            weights=None,
            consensus=None,
            n_boot=None):
        """
        Sample fit based on the reliability scores via the RSR algorithm.

        Parameters
        ----------
        weights : array-like of shape (n_obs, 1) or NoneType
            An array of weights for weighted regression. If None, squared
            reliability scores will be used as weights as a default. Note, that
            if bootstrapping is used for inference, the estimation of
            user-supplied weights is not reflected. Default is None.
        consensus : str or NoneType
            Type of optimization for consensus fit. Currently only
            'second_derivative' is supported. If None, weighted fit is
            performed. For consensus fit, set consensus='second_derivative' and
            weights=None. In all other cases, weighted fit is performed.
            Default is None.
        n_boot : int or NoneType
            Number of bootstrap replications for inference. If None specified,
            asymptotic approximation is used for inference instead. For valid
            inference, bootstrapping is recommended. Note that bootstrapping
            requires longer computation time. Default is None.

        Returns
        -------
        Results of class RSRFitResults. Following methods are available:
        .summary(), .conf_int() and .predict().

        Notes
        -----
        [`.fit()`](#samplefit.Reliability.RSR.fit) estimates the reliability
        scores via the RSR algorithm in the first step and estimates weighted
        regression in the second step, with the reliability scores as weights
        if not specified otherwise.

        Examples
        --------
        ```py
        # import libraries
        import samplefit as sf
        import statsmodels.api as sm
        
        # get data 
        boston = sm.datasets.get_rdataset("Boston", "MASS")
        Y = boston.data['medv'] # median house price
        X = boston.data['rm'] # number of rooms
        X = sm.add_constant(X)
        
        # specify model
        model = sm.OLS(endog=Y, exog=X)
        
        # specify sample
        sample = sf.RSR(linear_model=model)
        
        # sample fit with defaults
        sample_fit = sample.fit()
        
        # sample fit with consensus
        sample_fit = sample.fit(consensus='second_derivative')
        
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
            consensus=consensus,
            n_boot=n_boot
            )


    def score(self):
        """
        Estimation of reliability scores via the RSR algorithm.

        Parameters
        ----------
        None.

        Returns
        -------
        Results of class RSRScoreResults. Following methods are available:
        .plot().

        Notes
        -----
        [`.score()`](#samplefit.Reliability.RSR.score) estimates the
        reliability scores via the RSR algorithm. Each observation is scored
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
        Y = boston.data['medv'] # median house price
        X = boston.data['rm'] # number of rooms
        X = sm.add_constant(X)
        
        # specify model
        model = sm.OLS(endog=Y, exog=X)
        
        # specify sample
        sample = sf.RSR(linear_model=model)
        
        # score reliability
        sample_scores = sample.score()
        
        # extract reliability scores
        scores = sample_scores.scores
        
        # plot reliability scores
        sample_scores.plot()
        ```
        """
        return super().score()


    def anneal(self, share=0.05, n_boot=None):
        """
        Sample annealing based on the reliability scores via the RSR algorithm.

        Parameters
        ----------
        share : float or NoneType
            Share of sample that gets annealed based on the most unreliable
            observations. Default is 0.05.
        n_boot : int or NoneType
            Number of bootstrap replications for inference. If None specified,
            asymptotic approximation is used for inference instead. For valid
            inference, bootstrapping is recommended. Note that bootstrapping
            requires longer computation time. Default is None.

        Returns
        -------
        Results of class RSRAnnealResults. Following methods are available:
        .conf_int() and .plot().

        Notes
        -----
        [`.anneal()`](#samplefit.Reliability.RSR.anneal) re-estimates the model
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
        Y = boston.data['medv'] # median house price
        X = boston.data['rm'] # number of rooms
        X = sm.add_constant(X)
        
        # specify model
        model = sm.OLS(endog=Y, exog=X)
        
        # specify sample
        sample = sf.RSR(linear_model=model)
        
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
class RSRFitResults(BaseRSRFitResults):
    """
    Reliability Results class labeled `RSRFitResults()`.
    Initializes output of RSR.fit().

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
        Predict outcomes based on the sample fit via the RSR algorithm.
        
        Parameters
        ----------
        params : array-like or NoneType
            Array of parameters to predict with. If None supplied, the
            estimated parameters from the sample fit (weighted or consensus)
            will be used. Default is None.
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
        [`.predict()`](#samplefit.Reliability.RSRFitResults.predict) constructs
        predictions for outcome variable based on the estimated parameters.
        Predictions are based on the parameters of weighted fit or consensus
        fit, depending on the prior .fit() specification. If no new values for
        exogeneous variables are supplied, fitted values are returned.

        Examples
        --------
        ```py
        # import libraries
        import samplefit as sf
        import statsmodels.api as sm
        
        # get data 
        boston = sm.datasets.get_rdataset("Boston", "MASS")
        Y = boston.data['medv'] # median house price
        X = boston.data['rm'] # number of rooms
        X = sm.add_constant(X)
        
        # specify model
        model = sm.OLS(endog=Y, exog=X)
        
        # specify sample
        sample = sf.RSR(linear_model=model)
        
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
        Confidence intervals based on the sample fit via the RSR algorithm.
        
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
        [`.conf_int()`](#samplefit.Reliability.RSRFitResults.conf_int) constructs
        confidence intervals for estimated paramaters. If fitted without
        bootstrapping, asymptotic approximations are used. If fitted with
        bootstrapping, the standard deviation of bootstrapped parameters
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
        Y = boston.data['medv'] # median house price
        X = boston.data['rm'] # number of rooms
        X = sm.add_constant(X)
        
        # specify model
        model = sm.OLS(endog=Y, exog=X)
        
        # specify sample
        sample = sf.RSR(linear_model=model)
        
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
                percentile=False):
        """
        Summary of the sample fit via the RSR algorithm.
        
        Parameters
        ----------
        yname : str or NoneType
            Name of the endog variable. Default is 'y'.
        xname : list, tuple or NoneType
            List of name of the exog variables. Must have the same dimension as
            exog columns. Default are the supplied exog names.
        title : str or NoneType
            Title for the summary table. Default is
            'Random Sample Reliability Fit Results'.
        alpha : float or NoneType
            Confidence level alpha. Default is 0.05.
        percentile : bool
            Percentile method for confidence intervals based on bootstrapping.
            If bootstrapping has not been used for fitting, it is ignored.
            Default is False.

        Returns
        -------
        None. Prints summary table.

        Notes
        -----
        [`.summary()`](#samplefit.Reliability.RSRFitResults.summary) produces
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
        Y = boston.data['medv'] # median house price
        X = boston.data['rm'] # number of rooms
        X = sm.add_constant(X)
        
        # specify model
        model = sm.OLS(endog=Y, exog=X)
        
        # specify sample
        sample = sf.RSR(linear_model=model)
        
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
            percentile=percentile
            )


# class for AnnealResults
class RSRAnnealResults(BaseRSRAnnealResults):
    """
    Reliability Results class labeled `RSRAnnealResults()`.
    Initializes output of RSR.anneal().

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
             xname=None,
             title=None,
             alpha=0.05,
             percentile=False,
             color=None,
             path=None,
             figsize=None,
             ylim=None,
             xlabel=None,
             dpi=None):
        """
        Plot the Annealing based on the reliability scores via the RSR
        of class `Sample()`.
        
        Parameters
        ----------
        xname : list, tuple, str or NoneType
            Name or list of names of the exog variables for which parameter
            an annealing plot should be constructed. Must be one of the exog 
            variable names. If not supplied annealing plots for all parameters
            are constructed. Default are the supplied exog names.
        title : str or NoneType
            Title for the annealing plot. Default is
            'RSR: Annealing Sensitivity'.
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
            

        Returns
        -------
        Dictionary of matplotlib figures. Prints annealing plots.

        Notes
        -----
        [`.plot()`](#samplefit.Reliability.RSRAnnealResults.plot) produces
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
        Y = boston.data['medv'] # median house price
        X = boston.data['rm'] # number of rooms
        X = sm.add_constant(X)
        
        # specify model
        model = sm.OLS(endog=Y, exog=X)
        
        # specify sample
        sample = sf.RSR(linear_model=model)
        
        # sample annealing
        sample_annealing = sample.anneal()
        
        # default annealing plot
        sample_annealing.plot()
        
        # custom annealing
        sample_annealing.plot(title='My Title')
        ```
        """
        return super().plot(
            xname=xname,
            title=title,
            alpha=alpha,
            percentile=percentile,
            color=color,
            path=path,
            figsize=figsize,
            ylim=ylim,
            xlabel=xlabel,
            dpi=dpi
            )
    
    
    def conf_int(self,
                 alpha=0.05,
                 percentile=False):
        """
        Fitting the model based on the reliability scores via the RSR algorithm
        of class `Sample()`.
        
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
        [`.conf_int()`](#samplefit.Reliability.RSRAnnealResults.conf_int)
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
        Y = boston.data['medv'] # median house price
        X = boston.data['rm'] # number of rooms
        X = sm.add_constant(X)
        
        # specify model
        model = sm.OLS(endog=Y, exog=X)
        
        # specify sample
        sample = sf.RSR(linear_model=model)
        
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
class RSRScoreResults(BaseRSRScoreResults):
    """
    Reliability Results class labeled `RSRScoreResults()`.
    Initializes output of RSR.score().

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
             xlabel=None,
             dpi=None):
        """
        Plot the Reliability Scores based on the RSR algorithm.
        
        Parameters
        ----------
        yname : str or NoneType
            Name of the endog variable. Default is 'y'.
        xname : list, tuple, str or NoneType
            Name or list of names of the exog variables for which parameter
            an annealing plot should be constructed. Must be one of the exog 
            variable names. If not supplied annealing plots for all parameters
            are constructed. Default are the supplied exog names.
        title : str or NoneType
            Title for the annealing plot. Default is
            'RSR: Annealing Sensitivity'.
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
            The marker size in points**2 for in matplotlib scatter plot.
            Default is automatic.
        ylim : tuple, list or NoneType
            Tuple of upper and lower limits of y axis. Default is automatic.
        xlabel : str or NoneType
            Label for the x axis for the exog variable. Default is 'xname'.
        dpi : float, int or NoneType
            The resolution for matplotlib scatter plot. Default is 100.
            

        Returns
        -------
        Dictionary of matplotlib figures. Prints annealing plots.

        Notes
        -----
        [`.plot()`](#samplefit.Reliability.RSRAnnealResults.plot) produces
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
        Y = boston.data['medv'] # median house price
        X = boston.data['rm'] # number of rooms
        X = sm.add_constant(X)
        
        # specify model
        model = sm.OLS(endog=Y, exog=X)
        
        # specify sample
        sample = sf.RSR(linear_model=model)
        
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
            cmap=cmap,
            path=path,
            figsize=figsize,
            s=s,
            ylim=ylim,
            xlabel=xlabel,
            dpi=dpi
            )
