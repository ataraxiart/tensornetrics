% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/helper_functions.R
\name{flip_one_0_to_1}
\alias{flip_one_0_to_1}
\title{Given a tensor of 1s and 0s, it returns a list of the possible tensors after one of the 0s 
is now set to 1}
\usage{
flip_one_0_to_1(tensor)
}
\arguments{
\item{tensor}{torch tensor of 1s and 0s}
}
\value{
list of possible torch tensors
}
\description{
Example: flip_one_1_to_0(torch_tensor(c(0,1,1)))
Output: [[1]]
       torch_tensor
       1
       1
       1
       [ CPUFloatType{3} ]
       
Explanation: There is only one possibility from torch_tensor(c(0,1,1)), namely
torch_tensor(c(1,1,1))
}
