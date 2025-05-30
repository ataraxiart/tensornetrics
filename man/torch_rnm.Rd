% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/torch_rnm.R
\name{torch_rnm}
\alias{torch_rnm}
\alias{tensor_rnm}
\title{Residual Network Model with a Torch backend}
\usage{
tensor_rnm(
  data,
  lambda,
  vars,
  latents,
  B_matrix = NULL,
  lambda_constraint_lst = NULL,
  psi_constraint_lst = NULL,
  delta_theta_constraint_lst = NULL,
  omega_theta_constraint_lst = NULL,
  omega_theta_free_lst = NULL,
  lasso = FALSE,
  custom_loss = NULL,
  dtype = torch_float32(),
  device = torch_device("cpu"),
  identification = "variance"
)
}
\arguments{
\item{data}{data for the SEM model}

\item{lambda}{design matrix for factor loadings, lambda matrix}

\item{vars}{vector containing names of observed variables}

\item{latents}{vector containing names of latent variables}

\item{lambda_constraint_lst}{list containing the row, column and value vectors of the constraints of the lambda matrix}

\item{psi_constraint_lst}{list containing the row, column and value vectors of the constraints of the psi matrix}

\item{delta_theta_constraint_lst}{list containing the row, column and value vectors of the constraints of the delta_theta matrix}

\item{omega_theta_constraint_lst}{list containing the row, column and value vectors of the constraints of the omega_theta matrix}

\item{omega_theta_free_lst}{list containing the row, column and value vectors of the free parameters of the omega_theta matrix}

\item{lasso}{(optional) boolean to specify if model wants to use LASSO}

\item{custom_loss}{(optional) function supplied to allow the model to be fitted to a custom loss function}

\item{dtype}{(optional) torch dtype for the model (default torch_float32())}

\item{device}{(optional) device type to put the model on. see [torch::torch_device()]}

\item{identification}{(optional)  method to identify the model, either by variance or by loadings (default "variance")}
}
\value{
A `torch_rnm` object, which is an `nn_module` (torch object)
}
\description{
Function for creating a Residual Network Model
}
\details{
This function instantiates a torch object for computing the model-implied covariance matrix
based on a Residual Network Model. Through `torch`, gradients of this forward model can then
be computed using backpropagation, and the parameters can be optimized using gradient-based
optimization routines from the `torch` package.

Due to this, it is straightforward to add additional penalties to the standard objective function,
or to write a new objective function altogether.
}
\section{Methods}{


## `$initialize(data, lambda, vars, latents)`
The initialize method. Don't use this, just use [tensor_rnm()]

### Arguments
- `data` data for the SEM model
- `lambda` design matrix for factor loadings, lambda matrix
- `vars` vector containing names of observed variables
- `latents` vector containing names of latent variables
- `lambda_constraint_lst` list containing the row, column and value vectors of the constraints of the lambda matrix
- `psi_constraint_lst` list containing the row, column and value vectors of the constraints of the psi matrix
- `delta_theta_constraint_lst` list containing the row, column and value vectors of the constraints of the delta_theta matrix
- `omega_theta_constraint_lst` list containing the row, column and value vectors of the constraints of the omega_theta matrix
- `omega_theta_free_lst` list containing the row, column and value vectors of the free parameters of the omega_theta matrix
- `lasso` (optional) boolean to specify if model wants to use LASSO
- `custom_loss` (optional) function supplied to allow the model to be fitted to a custom loss function
- `dtype` (optional) torch dtype for the model (default torch_float32())
- `device` (optional) device type to put the model on. see [torch::torch_device()]
- `identification` (optional)  method to identify the model, either by variance or by loadings (default "variance")

### Value
A `torch_rnm` object, which is an `nn_module` (torch object)



## `$forward()`
Compute the model-implied covariance matrix.
Don't use this; `nn_modules` are callable, so access this method by calling
the object itself as a function, e.g., `my_torch_rnm()`.
In the forward pass, we apply constraints to the parameter vector, and we
create matrix views from it to compute the model-implied covariance matrix.

### Value
A `torch_tensor` of the model-implied covariance matrix



## `$loglik()`
Multivariate normal log-likelihood of the data.


### Value
Log-likelihood value (torch scalar)



## `$lasso_loss()`
Returns lasso loss which is -2 times Multivariate normal log-likelihood of the data + Penalty.

### Arguments
- `v` hyperparameter which controls for the penalty term inside the lasso loss function

### Value
lasso loss value (torch scalar)



## `$inverse_Hessian()`
Compute and return the asymptotic covariance matrix of the parameters with
respect to the loss function, which is limited to the -2 times the log-likelihood function


### Value
A `torch_tensor`, representing the ACOV of the free parameters



## `$fit()`
Fit a torch_rnm model using the default maximum likelihood objective.
This function uses the Adam optimizer to estimate the parameters of a torch_rnm

### Arguments
- `lrate` (Optional) learning rate of the Adam optimizer. Default is 0.05.
- `maxit` (Optional) maximum number of epochs to train the model. Default is 5000.
- `verbose` (Optional) whether to print progress to the console.  Default is TRUE.
- `tol` (Optional) parameter change tolerance for stopping training. Default is 1e-20.
- `batch_size` (Optional) change the number of samples used for training 
to reduce computational time. Default is NULL

### Value
Self, i.e., the `torch_rnm` object with updated parameters



## `$lasso_fit()`
Fit a torch_lnm model using the lasso loss function.
This function uses the Adam optimizer to estimate the parameters of a torch_lnm

### Arguments
- `verbose`(Optional) whether to print progress to the console.  Default is TRUE.
- `lrate` (Optional) learning rate of the Adam optimizer. Default is 0.05.
- `maxit` (Optional) maximum number of epochs to train the model. Default is 5000.
- `tol` (Optional) parameter change tolerance for stopping training. Default is 1e-20.
- `v` (Optional) hyperparameter which controls for the penalty term inside the lasso loss function. Default is 1.
- `epsilon` (Optional) Cutoff for lasso to set parameter to 0. Default is 0.0001.
- `batch_size` (Optional) change the number of samples used for training 
to reduce computational time. Default is NULL

### Value
Self, i.e., the `torch_rnm` object with updated parameters



## `$custom_fit()`
Fit a torch_lnm model using a custom loss function supplied to the torch_lnm module.
The custom loss function has to have 2 input parameters, the model covariance matrix and 
the data. (See Example Code for clarification)
This function uses the Adam optimizer to estimate the parameters of a torch_lnm

### Arguments
- `lrate` (Optional) learning rate of the Adam optimizer. Default is 0.05.
- `maxit` (Optional) maximum number of epochs to train the model. Default is 5000.
- `verbose` (Optional) whether to print progress to the console.  Default is TRUE.
- `tol` (Optional) parameter change tolerance for stopping training. Default is 1e-20.
- `batch_size` (Optional) change the number of samples used for training 
to reduce computational time. Default is NULL

### Value
Self, i.e., the `torch_rnm` object with updated parameters



## `$lasso_update_params_added(v,epsilon)`
Update the model attribute, self$lasso_num_params_added This is will be done automatically if
is called after `$lasso_fit()`.

### Arguments
- `v` hyperparameter which controls for the penalty term inside the lasso loss function.
- `epsilon`  Cutoff for lasso to set parameter to 0. 

### Value
None



## `$get_all_latent_pairings()`
Get all the possible combinations of latent pairs, or the pairs of latent variables as prescribed
in the model. For example, if there are 3 latent variables (a,b,c), we have a total of 3 choose 2, or 3
latent pairs possible and the function will return c('a~b', 'a~c', 'b~c').

### Value
vector of all possible combinations of latent pairs



## `$get_all_residual_pairings()`
Get all the possible combinations of residual pairs, or the pairs of observed variables as prescribed
in the model. For example, if there are 3 observed variables (a,b,c), we have a total of 3 choose 2, or 3
pairs possible and the function will return c('a~b', 'a~c', 'b~c').

### Value
vector of all possible combinations of residual/observed variable pairs



## `$get_df()`
Get the number of degrees of freedom in the model which equals n(n+1)/2 - number of free parameters.
where n is the dimension of the sample covariance matrix.

### Value
Degrees of freedom



## `$get_loadings()`
Get all the free factor loadings (entries of lambda matrix) determined after model fit. 
This includes fit, custom_fit and lasso_fit. 

### Value
Dataframe of factor loadings and if the default log-likelihood fn is used in to fit,
standard errors and p-values are also provided



## `$get_psi()`
Get all the free partial correlations (entries of lower triangle of omega_theta matrix) 
determined after model fit. This includes fit, custom_fit and lasso_fit. 



### Value
Dataframe of partial correlations and if the default log-likelihood fn is used in to fit,
standard errors and p-values are also provided



## `$get_delta_theta()`
Get all the free residuals (diagonal entries of theta matrix) determined after model fit. 
This includes fit, custom_fit and lasso_fit. 

### Value
Dataframe of residuals and if the default log-likelihood fn is used in to fit,
standard errors and p-values are also provided



## `$get_partial_correlations(epsilon)`
Get all the free partial correlations (entries of lower triangle of omega_theta matrix) 
determined after model fit. This includes fit, custom_fit and lasso_fit. 

### Arguments
- `epsilon` (Optional) If lasso fit was used, this determines the threshold, under which
a partial correlation is no longer considered a free parameter and is set to 0. Default value is
0.00001



### Value
Dataframe of partial correlations and if the default log-likelihood fn is used in to fit,
standard errors and p-values are also provided



## `$get_criterion_value()`

Get the criterion value of the model according to the type of criterion specified.
The possibilities include AIC (Akaike Information Criterion), 
BIC (Bayesian Information Criterion), EBIC (Extended Bayesian Information Criterion) 
and the chisquare statistic.

### Arguments
- `criterion` Name of the Criterion used, AIC, BIC, EBIC, or chisq.
- `gamma` (optional) Gamma hyperparmaeter for the EBIC if the EBIC is used. (default value is 0.5)

### Value
Criterion Value



## `$get_fit_metrics()`

Get the fit metrics after the model is fit. Metrics returned are the CFI (Comparative Fit Index), 
TLI (Tucker-Lewis Index), the RMSEA (Root Mean Square Error of Approximation) 
and the confidence interval of the RMSEA. These metrics compare the fitted model to
the baseline model which assumes all observed variables are uncorrelated.

### Arguments

### Value
Dataframe containing the fit metrics
}

