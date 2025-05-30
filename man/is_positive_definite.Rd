% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/helper_functions.R
\name{is_positive_definite}
\alias{is_positive_definite}
\title{Checks if the given matrix is positive definite}
\usage{
is_positive_definite(matrix)
}
\arguments{
\item{matrix}{matrix such as the covariance matrix estimated from lnm/rnm}
}
\value{
boolean indicating if matrix is positive definite
}
\description{
Example: is_positive_definite(matrix(c(1,1,1,0),byrow=T,ncol = 2))
Output: FALSE
      
       
Explanation: The given matrix is not positive definite.
}
