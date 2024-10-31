

# 

# lambda
# omega_theta
# psi
# 

#initialize rnm module
#fit -> Just for no partial correlations
#stepup -> fit until 





#' Residual Network Model with a Torch backend
#'
#' Function for creating a Latent Network Model
#'
#' @param data data for the SEM model
#' @param lambda design matrix for factor loadings, lambda matrix
#' @param vars vector containing names of observed variables
#' @param latents vector containing names of latent variables
#' @param lambda_constraint_lst list containing the row, column and value vectors of the constraints of the lambda matrix
#' @param psi_constraint_lst list containing the row, column and value vectors of the constraints of the psi matrix
#' @param theta__omega_constraint_lst list containing the row, column and value vectors of the constraints of the theta_omega matrix
#' @param lasso (optional) boolean to specify if model wants to use LASSO
#' @param custom_loss (optional) function supplied to allow the model to be fitted to a custom loss function
#' @param dtype (optional) torch dtype for the model (default torch_float32())
#' @param device (optional) device type to put the model on. see [torch::torch_device()]
#'
#' @return A `torch_rnm` object, which is an `nn_module` (torch object)
#'
#' @details
#' This function instantiates a torch object for computing the model-implied covariance matrix
#' based on a Residual Network Model. Through `torch`, gradients of this forward model can then
#' be computed using backpropagation, and the parameters can be optimized using gradient-based
#' optimization routines from the `torch` package.
#'
#' Due to this, it is straightforward to add additional penalties to the standard objective function,
#' or to write a new objective function altogether.
#'
#' @import torch
#' @import lavaan
#' @importFrom R6 R6Class
#'
#' @name torch_rnm
#'
#'
#' @export
tensor_rnm <- torch::nn_module(
  classname = "torch_network_model",
  #' @section Methods:
  #'
  #' ## `$initialize(data, lambda, vars, latents)`
  #' The initialize method. Don't use this, just use [torch_sem()]
  #'
  #' ### Arguments
  #' - `data` data for the SEM model
  #' - `lambda` design matrix for factor loadings, lambda matrix
  #' - `vars` vector containing names of observed variables
  #' - `latents` vector containing names of latent variables
  #' - `lambda_constraint_lst` list containing the row, column and value vectors of the constraints of the lambda matrix
  #' - `psi_constraint_lst` list containing the row, column and value vectors of the constraints of the psi matrix
  #' - `theta_omega_constraint_lst` list containing the row, column and value vectors of the constraints of the theta_omega matrix
  #' - `lasso` (optional) boolean to specify if model wants to use LASSO
  #' - `custom_loss` (optional) function supplied to allow the model to be fitted to a custom loss function
  #' - `dtype` (optional) torch dtype for the model (default torch_float32())
  #' - `device` (optional) device type to put the model on. see [torch::torch_device()]
  #'
  #' ### Value
  #' A `torch_rnm` object, which is an `nn_module` (torch object)
  initialize = function(data,lambda,vars,latents,B_matrix = NULL,lambda_constraint_lst = NULL,
                        psi_constraint_lst = NULL,delta_theta_constraint_lst = NULL,omega_theta_constraint_lst = NULL,
                        lasso=FALSE,custom_loss=NULL,dtype = torch_float32(), device = torch_device("cpu")){
    if (is.null(vars) || is.null(latents)) {
      warning('Must indicate names of observed and latent variables')
    }
    else{
      if (any(colSums(is.na(data[vars])) > 0)) warning('Data has missing values!')
    self$data <- df_to_tensor(data[,vars])} #Centered data
    self$vars <- vars
    self$latents <- latents
    self$custom_loss <- custom_loss
    self$lasso<-lasso
    self$device <- device
    self$dtype <- dtype
    # compute torch settings
    opt <-rnm_mod_to_torch_opts(lambda, lambda_constraint_lst,
                                psi_constraint_lst,delta_theta_constraint_lst,omega_theta_constraint_lst)
    # initialize the dimensions of the model:
    self$n <- dim(lambda)[1]
    self$m <- dim(lambda)[2]
    self$num_obs <- nrow(data)
    # initialize the parameter vector
    self$params_vec <- nn_parameter(torch_tensor(opt$params_start, dtype = self$dtype, requires_grad = TRUE, device = self$device))
    # initialize the parameter constraints
    self$params_free <- torch_tensor(opt$params_free, dtype = torch_bool(), requires_grad = FALSE, device = self$device)
    self$params_value <- torch_tensor(opt$params_value, dtype = self$dtype, requires_grad = FALSE, device = self$device)
    # other attributes
    self$cov_matrix <- torch_tensor(cov(data[,vars]))
    self$lambda_matrix_free <- opt$lambda_matrix_free
    self$params_sizes <- opt$params_sizes
    self$params_free_sizes <- opt$params_free_sizes
    self$params_free_sizes_max <- opt$params_free_sizes_max
    self$params_free_starting_pts <- opt$params_free_starting_pts
    self$lasso_num_params_removed  <- 0
    # duplication indices transforming vech to vec for psi and omega_theta
    self$psi_dup_idx <- torch_tensor(vech_dup_idx(self$m), dtype = torch_long(), requires_grad = FALSE, device = self$device)
    self$delta_theta_dup_idx <- torch_tensor(vech_dup_idx(self$n), dtype = torch_long(), requires_grad = FALSE, device = self$device)
    self$omega_theta_dup_idx <- torch_tensor(vech_dup_idx(self$n), dtype = torch_long(), requires_grad = FALSE, device = self$device)
    # B matrix
    if(is.null(B_matrix)){self$B <- matrix(0,nrow=self$m,ncol = self$m)}
    else{
    self$B <- B_matrix
    }
    # tensor identity matrices
    self$I_mat_m <- torch_eye(self$m, dtype = self$dtype, requires_grad = FALSE, device = self$device)
    self$I_mat_n <- torch_eye(self$n, dtype = self$dtype, requires_grad = FALSE, device = self$device)
    # mean is fixed to 0/centering
    self$mu <- torch_zeros(self$n, dtype = self$dtype, requires_grad = FALSE, device = self$device) 
  },
  
  #' @section Methods:
  #'
  #' ## `$forward()`
  #' Compute the model-implied covariance matrix.
  #' Don't use this; `nn_modules` are callable, so access this method by calling
  #' the object itself as a function, e.g., `my_torch_rnm()`.
  #' In the forward pass, we apply constraints to the parameter vector, and we
  #' create matrix views from it to compute the model-implied covariance matrix.
  #'
  #' ### Value
  #' A `torch_tensor` of the model-implied covariance matrix
  forward = function(compute_sigma = TRUE){
    #Apply constraints for non-free parameters
    self$params <- torch_where(self$params_free, self$params_vec, self$params_value)
    #Get separate tensors for lambda, theta, delta_psi, and omega_psi
    self$params_split <- torch_split(self$params, self$params_sizes)
    self$lambda <- self$params_split[[1]]$view(c(self$m, self$n))$t()
    self$psi <- torch_index_select(self$params_split[[2]], 1, self$psi_dup_idx)$view(c(self$m,self$m)) 
    self$delta_theta <- torch_index_select(self$params_split[[3]], 1, self$delta_theta_dup_idx)$view(c(self$n,self$n))
    self$omega_theta <- torch_index_select(self$params_split[[4]], 1, self$omega_theta_dup_idx)$view(c(self$n,self$n))
    
    
    #Compute the model-implied covariance matrix 
    self$B_0 <- (self$I_mat_m - self$B)
    self$B_0_inv <- torch_inverse(self$B_0)
    self$psi_middle_term <- self$B_0_inv$mm(self$psi$mm(self$B_0_inv$t()))
    self$cov_first_term <- self$lambda$mm(self$psi_middle_term$mm(self$lambda$t()))
    
    self$omega_theta_inv <- torch_inverse(self$I_mat_n - self$omega_theta)
    self$residual_term <- self$delta_theta$mm(self$omega_theta_inv$mm(self$delta_theta)) 
    
    
  if (compute_sigma == FALSE) return(invisible(self))
  self$sigma <- self$cov_first_term + 
    self$residual_term
  return(self$sigma)},
  #' @section Methods:
  #'
  #' ## `$loglik()`
  #' Multivariate normal log-likelihood of the data.
  #'
  #'
  #' ### Value
  #' Log-likelihood value (torch scalar)
  loglik = function() {
    px <- distr_multivariate_normal(loc = self$mu, covariance_matrix = self$forward())
    return(px$log_prob(self$data)$sum())
    
  },
  #' @section Methods:
  #'
  #' ## `$inverse_Hessian()`
  #' Compute and return the asymptotic covariance matrix of the parameters with
  #' respect to the loss function, which is limited to the -2 times the log-likelihood function
  #'
  #'
  #' ### Value
  #' A `torch_tensor`, representing the ACOV of the free parameters
  inverse_Hessian = function() {
    g <- autograd_grad(-2*self$loglik(), self$params_vec, create_graph = TRUE)[[1]]
    H <- torch_jacobian(g, self$params_vec)
    free_idx <- torch_nonzero(self$params_free)$view(-1)
    self$Hinv <- torch_inverse(H[free_idx, ][, free_idx])
    return(self$Hinv)
  },
  
  #' @section Methods:
  #'
  #' ## `$fit()`
  #' Fit a torch_rnm model using the default maximum likelihood objective.
  #' This function uses the Adam optimizer to estimate the parameters of a torch_rnm
  #'
  #' ### Arguments
  #' - `lrate` (Optional) learning rate of the Adam optimizer. Default is 0.05.
  #' - `maxit` (Optional) maximum number of epochs to train the model. Default is 5000.
  #' - `verbose` (Optional) whether to print progress to the console.  Default is TRUE.
  #' - `tol` (Optional) parameter change tolerance for stopping training. Default is 1e-20.
  #'
  #' ### Value
  #' Self, i.e., the `torch_rnm` object with updated parameters 
  fit = function(lrate = 0.05, maxit = 5000, verbose = TRUE, tol = 1e-20) {
    if (verbose) cat("Fitting SEM with Adam optimizer and MVN log-likelihood loss\n")
    optim <- optim_adam(self$params_vec, lr = lrate)
    prev_loss <- 0.0
    for (epoch in 1:maxit) {
      optim$zero_grad()
      loss <- -2*self$loglik()
      if (verbose) {
        cat("\rEpoch:", epoch, " loglik:", -loss$item())
        flush.console()
      }
      loss$backward()
      optim$step()
      if (epoch > 1 && abs(loss$item() - prev_loss) < tol) {
        if (verbose) cat("\n")
        break
      }
      prev_loss <- loss$item()
    }
    if (epoch == maxit) warning("maximum iterations reached")
    
    return(invisible(self))
  },
  
  #' @section Methods:
  #'
  #' ## `$identify_via_loadings()`
  #' Fit a torch_rnm model using the default maximum likelihood objective.
  #' This function uses the Adam optimizer to estimate the parameters of a torch_rnm
  #'
  #' ### Arguments
  #' - `lrate` (Optional) learning rate of the Adam optimizer. Default is 0.05.
  #' - `maxit` (Optional) maximum number of epochs to train the model. Default is 5000.
  #' - `verbose` (Optional) whether to print progress to the console.  Default is TRUE.
  #' - `tol` (Optional) parameter change tolerance for stopping training. Default is 1e-20.
  #'
  #' ### Value
  #' Self, i.e., the `torch_rnm` object with updated parameters  
  identify_via_loadings = function() {
 
  },
  ,
  #' @section Methods:
  #'
  #' ## `$get_df()`
  #' Get the number of degrees of freedom in the model which equals n(n+1)/2 - number of free parameters.
  #' where n is the dimension of the sample covariance matrix.
  #'
  #' ### Value
  #' Degrees of freedom
  
  get_df=function(){
    self$df <- self$n*(self$n+1)/2 - sum(self$params_free_sizes)
    if (self$df < 0) warning('Over identified model!')
    return(self$df)
  }
)












