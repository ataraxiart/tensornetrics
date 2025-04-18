% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/torch_rnm_helper_functions.R
\name{rnm_lasso_explore}
\alias{rnm_lasso_explore}
\title{LASSO for RNM}
\usage{
rnm_lasso_explore(
  mod,
  criterion = "BIC",
  v_values = pracma::logspace(log10(0.01), log10(100), 30),
  lrate = 0.01,
  epsilon = 1e-04,
  gamma = 0.5,
  batch_size = batch_size
)
}
\arguments{
\item{mod}{original torch_rnm/torch_rnm_stepwise module undergoing stepping up}

\item{criterion}{(Optional) string indicating criterion to use - "AIC", "BIC or "EBIC". Default is BIC.}

\item{v_values}{(Optional) values of the hyperparameter v to be used in the search. Default range 
of values is 30 values of v from 0.01 to 100 spread out on a log scale.}

\item{lrate}{(Optional) learning rate for lasso. Default value is 0.01.}

\item{epsilon}{(Optional) threshold under which partial correlations will be set to 0. Default is 0.0001.}

\item{gamma}{(Optional) value of gamma if "EBIC" is used. Default value is 0.5.}
}
\value{
a list containing the following: 1)value of v of the model which 
gives the best criterion score and 2)free parameters for omega_theta
}
\description{
Performs a lasso search to find the value of hyperparameter v which correspond to the model 
which gives the lowest criterion scores (AIC/BIC/EBIC). At the end, the free parameters
for omega_theta (parameters which have been set to zero) will also be returned. Each model 
is fit with a different value of v and partial correlations which are lower than the threshold value are set to 0
These partial correlations are removed from the pool of free parameters before the criterion score is calculated.
}
