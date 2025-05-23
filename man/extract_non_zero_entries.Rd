% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/helper_functions.R
\name{extract_non_zero_entries}
\alias{extract_non_zero_entries}
\title{Given a symmetric matrix outputs a list containing 3 vectors:  The first and second vector represent the 
row and column of the non-zero entries inside the lower triangle of the symmetric matrix. 
The last vector is just a vector of zeros with length which matches the number of non-zero entries.}
\usage{
extract_non_zero_entries(mat)
}
\arguments{
\item{mat}{symmetric matrix}
}
\value{
list of 3 vectors as described above
}
\description{
Example: extract_non_zero_entries(matrix(c(0,1,1,1,0,1,1,1,0),byrow=F,nrow = 3))
Output:      
        $rows
         2 3 3
        $cols
         1 1 2
        $values
         0 0 0
}
\details{
Explanation: vector with titles "rows" and "cols" colectively indicate the position of the non-zero entries
of the matrix. The remaining vector "values" is just a zero vector.
}
