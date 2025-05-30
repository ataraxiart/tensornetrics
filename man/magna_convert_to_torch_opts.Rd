% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/torch_magna_helper_functions.R
\name{magna_convert_to_torch_opts}
\alias{magna_convert_to_torch_opts}
\title{Converts input information to the necessary data structures to start the torch optimization procedure
for random effects MAGNA}
\usage{
magna_convert_to_torch_opts(
  corMats,
  missing = FALSE,
  params_free_filter = NULL,
  params_value_filter = NULL,
  dtype = NULL,
  device = NULL
)
}
\arguments{
\item{corMats}{list of torch tensors}

\item{missing}{indicator for whether missing values are present , by default FALSE}

\item{params_free_filter}{torch tensor of booleans (ones and zeros) to indicate which parameters are free,
by default NULL}

\item{params_value_filter}{torch tensor of user-defined starting values to use at start of optimization,
by default NULL}

\item{dtype}{data type to use for torch operations}

\item{device}{device to use for torch operations}
}
\value{
list containing the following information to be input into the torch module for
random effects MAGNA: params_vec, params_free, params_value, omega_params_length,
tau_params_length
}
\description{
Example:
magna_convert_to_torch_opts(corMats,missing = FALSE,params_free_filter = NULL,
params_value_filter = NULL, dtype = torch_float32(), device = torch_device("cpu"))
Output: list of information to be used in random effects MAGNA estimation via torch
}
\details{
Explanation: Converts input information for the start of the torch optimization procedure
}
