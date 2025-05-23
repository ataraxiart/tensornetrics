% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/helper_functions.R
\name{lasso_explore}
\alias{lasso_explore}
\title{LASSO for Network Models}
\usage{
lasso_explore(
  mod,
  criterion = "BIC",
  v_values = NULL,
  lrate = 0.01,
  epsilon = 1e-04,
  gamma = 0.5,
  batch_size = NULL
)
}
\arguments{
\item{mod}{torch module representing an lnm or rnm}

\item{criterion}{(Optional) string indicating criterion to use - "AIC", "BIC or "EBIC". Default is BIC.}

\item{v_values}{(Optional) values of the hyperparameter v to be used in the search. Default range 
of values is 30 values of v from 0.01 to 100 spread out on a log scale.}

\item{lrate}{(Optional) learning rate for lasso. Default value is 0.01.}

\item{epsilon}{(Optional) threshold under which partial correlations will be set to 0. Default is 0.0001.}

\item{gamma}{(Optional) value of gamma if "EBIC" is used. Default value is 0.5.}
}
\value{
a list containing the following: 1)value of v of the model which 
gives the best criterion score and 2)either constraints or free partial correlations in the model
}
\description{
Performs a lasso search to find the value of hyperparameter v which correspond to the model 
which gives the lowest criterion scores (AIC/BIC/EBIC). At the end, the function also 
returns constraints or free parameters to be indicated for the network models. For lnm, it 
returns constraints for omega_psi while for rnm, free parameters for omega_theta are returned.
}
