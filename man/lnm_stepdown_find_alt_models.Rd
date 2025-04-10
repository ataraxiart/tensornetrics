% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/torch_lnm_helper_functions.R
\name{lnm_stepdown_find_alt_models}
\alias{lnm_stepdown_find_alt_models}
\title{Helper function for lnm_stepdown. Given an original torch_lnm/torch_lnm_stepwise module, the function fits all possible models
during one iteration of the stepping down process and returns either the best model based on criterion score
after this fitting process or the original torch_lnm/torch_lnm_stepwise module if all new models give a
worse model fit based on criterion score. A possible model here is a model with one of the
originally non-zero partial correlations now being set to zero.}
\usage{
lnm_stepdown_find_alt_models(
  mod,
  criterion = "BIC",
  gamma = 0.5,
  batch_size = NULL
)
}
\arguments{
\item{mod}{original torch_lnm/torch_lnm_stepwise module undergoing stepping down}

\item{criterion}{string indicating criterion to use - "AIC", "BIC or "EBIC"}

\item{gamma}{(Optional) value of gamma if "EBIC" is used. Default value is 0.5.}
}
\value{
either the original torch_lnm/torch_lnm_stepwise module or 
a torch_lnm_stepwise module with a better criterion score
}
\description{
Helper function for lnm_stepdown. Given an original torch_lnm/torch_lnm_stepwise module, the function fits all possible models
during one iteration of the stepping down process and returns either the best model based on criterion score
after this fitting process or the original torch_lnm/torch_lnm_stepwise module if all new models give a
worse model fit based on criterion score. A possible model here is a model with one of the
originally non-zero partial correlations now being set to zero.
}
