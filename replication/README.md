# Replication: Sample Fit Reliability

Herein we present the workflow for replicating the results of the paper.

## Requirements

Main data analysis and estimation has been conducted using Python (v3.8.8), whereas the data preparation part has been carried out in R (v4.2.1).

In order to install all required libraries for Python, we recommend setting up a virtual environment (see e.g. `virtualenv` package) and running `pip install -r requirements.txt`.
The required packages for R will be automatically installed from within the provided scripts.

## Data

We provide the main data used for the analysis in the subfolder `\data` containing the following files: `data_lalonde.csv`, `data_microcredit.csv`, `data_microcredit_ppp.csv` and `data_charity.csv`.

The following two datasets, i.e. `data_microcredit.csv` and `data_microcredit_ppp.csv`, are modifications (specifically a subset transformed to a `.csv` format) of the original dataset `microcredit_project_data.RData` by Meager (2019) distributed under the following Public License: *Creative Commons Attribution 4.0 International Public License*, see the full licence material [here](https://www.openicpsr.org/openicpsr/project/116357/version/V1/view;jsessionid=6002C97DB4AD8FB5782D8F4B38DA21F2?path=/openicpsr/116357/fcr:versions/V1/LICENSE.txt&type=file). The original data are accesible for download [here](https://www.openicpsr.org/openicpsr/project/116357/version/V1/view?path=/openicpsr/116357/fcr:versions/V1&type=project). Please, save the corresponding data file `microcredit_project_data.RData` within the `\data` subfolder.

The `data_charity.csv` dataset is a modification (specifically a subset transformed to a `.csv` format) of the original dataset `AER merged.dta` by Karlan and List (2014) distributed under the following Public License: *reative Commons CC0 1.0 Universal Public Domain Dedication*, see the full licence material [here](https://creativecommons.org/publicdomain/zero/1.0/). The original data are accessible for download [here](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi%3A10.7910/DVN/27853). Please, save the corresponding data file `AER merged.dta` within the `\data` subfolder.

The `data_lalonde.csv` dataset is a modification (specifically a subset transformed to a `.csv` format) of the original dataset `lalonde` accessible from the R package `Matching` by Sekhon (2011) available [here](https://CRAN.R-project.org/package=Matching) and is directly downloaded from within the corresponding Python and R scripts. The original dataset refers to works by Lalonde (1986) and Dehejia and Wahba (1999).

The Boston housing dataset used in the paper is also readily accessible from the R package `MASS` by Venables and Ripley (2002) available [here](https://CRAN.R-project.org/package=MASS) and is directly downloaded from within the corresponding Python and R scripts. The original dataset refers to works by Harrison and Rubinfeld (1978) and Belsley, Kuh and Welsch (1980).

We hereby thank the authors for generously providing an open access with Public License to their research data.

To reproduce the data preparation to generate the above-mentioned `.csv` files, run the following notebook: `dataprep_replication.ipynb` (runtime: <1min). This will re-generate the `.csv` files within the `\data` subfolder.
Note that this step is not necessary to replicate the results, but serves for completeness of the replication exercise.

## Illustration

For replicating the results of the empirical illustration from Section 3 of the paper, run the following notebook: `illustration_replication.ipynb` (runtime: <1min).
The notebook generates all figures and tables and saves them in the corresponding subfolders `\figures` and `\results`.

## Simulation

To replicate the simulation results from Section 4 of the paper, as well as from Appendix B, run the following notebook `simulation_replication.ipynb` (runtime: > 2h).
The notebook generates all figures and tables and saves them in the corresponding subfolders `\figures` and `\results`.

## Replications

In order to reproduce the results from the replications in Section 5 of the paper, as well as from Appendix B, run the following notebook: `application_replication.ipynb` (runtime: > 5h).
The notebook generates all figures and tables and saves them in the corresponding subfolders `\figures` and `\results`.

## References

- Belsley D.A., Kuh, E. and Welsch, R.E. (1980) Regression Diagnostics. Identifying Influential Data and Sources of Collinearity. New York: Wiley
- Dehejia, Rajeev and Sadek Wahba. 1999.“Causal Effects in Non-Experimental Studies: Re-Evaluating the Evaluation of Training Programs.” Journal of the American Statistical Association 94 (448): 1053-1062.
- Harrison, D. and Rubinfeld, D.L. (1978) Hedonic prices and the demand for clean air. J. Environ. Economics and Management 5, 81–102.
- Karlan, Dean; List, John A., 2014, "Does Price Matter in Charitable Giving? Evidence from a Large-Scale Natural Field Experiment", https://doi.org/10.7910/DVN/27853, Harvard Dataverse, V4, UNF:5:C63Hp69Cn88CkFCoOA0N/w== [fileUNF]
- LaLonde, Robert. 1986. “Evaluating the Econometric Evaluations of Training Programs.” Ameri- can Economic Review 76:604-620.
- Meager, Rachael. Replication data for: Understanding the Average Impact of Microcredit Expansions: A Bayesian Hierarchical Analysis of Seven Randomized Experiments. Nashville, TN: American Economic Association [publisher], 2019. Ann Arbor, MI: Inter-university Consortium for Political and Social Research [distributor], 2019-12-07. https://doi.org/10.3886/E116357V1
- Sekhon JS (2011). “Multivariate and Propensity Score Matching Software with Automated Balance Optimization: The Matching Package for R.” Journal of Statistical Software, 42(7), 1–52. doi: 10.18637/jss.v042.i07.
- Venables WN, Ripley BD (2002). Modern Applied Statistics with S, Fourth edition. Springer, New York. ISBN 0-387-95457-0, https://www.stats.ox.ac.uk/pub/MASS4/.
