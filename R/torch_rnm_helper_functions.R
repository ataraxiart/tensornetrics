
library(torch)
library(lavaan)
library(pracma)


#' Given a number n (number of observations in dataset), returns a the torch tensor
#' to represent the free params of the omega delta matrix assuming a saturated model
#' 
#' Example: get_saturated_omega_theta_free(3)
#' Output: 
#'        torch_tensor
#'        0
#'        1
#'        1
#'        0
#'        1
#'        0
#'        [ CPUFloatType{6} ]
#'        
#' Explanation: There are in total 3 free params inside the matrix, the first 2
#' belong to the first column in the lower triangle of the matrix and the last
#' one belongs to the second column
#'
#' @param  n
#' 
#' 
#' @return tensor
get_saturated_omega_theta_free <- function(n){
  result <- matrix(1,ncol=n,nrow=n)
  diag(result) <- 0
  result <- torch_tensor(lavaan::lav_matrix_vech(result))
  return(result)
}




#' Based on the type of matrix, returns a list containing 1)mat_value, a matrix containing the 
#' starting values of the matrix parameters and 2)mat_free, a matrix of 1s and 0s which 
#' indicate which entries of the matrix are free parameters.  
#'
#' Example: rnm_constraint_matrices(type='psi',n_rows=3,n_cols = 3, entries=list(c(2),c(1),c(0.8)),identification = 'variance')
#' 
#' Output: $mat_value
#'         1      0    0
#'         0.8    1    0
#'         0.0    0    1
#'         $mat_free
#'         1    1    1
#'         0    1    1
#'         1    1    1
#' 
#' Explanation: The selected matrix is the 'psi' matrix with 3 rows and 3 columns.
#' The (2,1) entry is set to be 0.8 in mat_value. It is no longer free and so (2,1) entry
#' in mat_free will be a value of 0. Note that for symmetric matrices like psi,
#' if (i,j) entry is set to be fixed, then (j,i) entry will also be set to be fixed in the model, 
#' but the output of this function will not reflect that.
#'
#'
#' @param type string to indicate whether selected matrix is a "lambda", "omega_psi" or "theta" matrix
#' @param omega_theta indicate if omega_theta matrix return corresponds to constraints or free parameters
#' @param lambda design matrix for lambda if matrix type selected is "lambda"
#' @param n_rows number of rows of the selected matrix in the lnm model
#' @param n_cols number of columns of the selected matrix in the lnm model
#' @param entries (Optional) list containing 3 vectors which contain information about 
#' which parameters to fix. The first and second vector indicate the row and column of the 
#' parameter inside the selected matrix to be fixed. The last vector contains the prescribed value
#' for the parameter. 
#' @param identification (Optional) method to identify the model, either by variance or by loadings (default "variance")
#'
#' @return list containing mat_value and mat_free
#'
rnm_constraint_matrices <- function(type = "lambda",omega_theta = "free",lambda = NULL,n_rows, n_cols, entries = NULL,identification = 'variance') {
  
  if(identification == "variance"){
  if (type == "lambda"){
    mat_value <- matrix(0, nrow = n_rows, ncol = n_cols)
    mat_value <- mat_value + 0.7*lambda
    mat_free <- lambda
  }
  else if (type == "psi"){
    mat_value <- diag(1,n_rows)
    mat_free <- matrix(1, nrow = n_rows, ncol = n_cols)
    diag(mat_free) <- 0
  }
  else if (type == "delta_theta"){
    mat_value <- diag(0.5,n_rows)
    mat_free <- matrix(0, nrow = n_rows, ncol = n_cols)
    diag(mat_free) <- 1
  }
  else if (type == "omega_theta") {
    mat_value <- matrix(0, nrow = n_rows, ncol = n_cols)
    mat_free <- matrix(0, nrow = n_rows, ncol = n_cols)
  }
  }
  else if (identification == "loadings"){
    if (type == "lambda"){
      helper_mat_1 <- format_lambda_mat(lambda,type='value')
      helper_mat_2 <- format_lambda_mat(lambda,type = 'free')
      mat_value <- helper_mat_1 
      mat_free <- helper_mat_2 
    }
    else if (type == "psi"){
      mat_value <- diag(1,n_rows)
      mat_free <- matrix(1, nrow = n_rows, ncol = n_cols)
    }
    else if (type == "delta_theta"){
      mat_value <- diag(0.5,n_rows)
      mat_free <- matrix(0, nrow = n_rows, ncol = n_cols)
      diag(mat_free) <- 1
    }
    else if (type == "omega_theta") {
      mat_value <- matrix(0, nrow = n_rows, ncol = n_cols)
      mat_free <- matrix(0, nrow = n_rows, ncol = n_cols)
    }
    
    
  }


  if (!is.null(entries) && omega_theta == "free"){
    for (i in 1:length(entries[[1]])) {
      row <- entries[[1]][i]
      col <- entries[[2]][i]
      value <- entries[[3]][i]
      mat_value[row, col] <- value
      if (type != "omega_theta"){
      mat_free[row, col] <- 0
      mat_free[col, row] <- 0
      } else {
        mat_free[row, col] <- 1
        mat_free[col, row] <- 1
      }
      
    }
  }
    
    if (!is.null(entries) && omega_theta == "constraint"){
      for (i in 1:length(entries[[1]])) {
        row <- entries[[1]][i]
        col <- entries[[2]][i]
        value <- entries[[3]][i]
        mat_value[row, col] <- value
        mat_free[row, col] <- 0
        mat_free[col, row] <- 0
        
      }
    }

  return(list('mat_value' = mat_value,'mat_free' = mat_free))
}



#' Returns a list containing necessary information for the initialization of the torch rnm
#' model. The list contains: 1) params_start -  a vector of starting values for the parameters, 
#' 2) params_value - a vector of default values for non-free parameters and 3) params_free - a 
#' vector of 1s and 0s which indicate which parameters are free. Also included are: 1)
#' params_sizes - a vector which indicate the number of parameters in the lambda, psi, delta_theta and
#' omega_theta matrices respectively, 2) params_free_sizes - a vector which indicate the number of 
#' free parameters in each of the four matrices respectively, 3) params_free_sizes_max - a vector 
#' which indicate the maximum possible number of free parameters in the 
#' four matrices, 4) params_free_starting_pts - a vector which indicate the starting index of 
#' each group of parameters and 5) lambda_matrix_free - a matrix
#' of 1s and 0s which indicate which parameters in the lambda matrix are free.
#'
#' Example: lambda <- matrix(0, 10, 3)
#' lambda[1:3,1] <- 1
#' lambda[4:6,2] <- 1
#' lambda[7:9,3] <- 1
#' rnm_mod_to_torch_opts(lambda)
#' Output: list containing the aforementioned vectors
#' 
#' Explanation: The design matrix for the factor loadings, lambda is provided to the function 
#' while setting no constraints for the lambda, omega_psi and theta matrices. The aforementioned vectors
#' returned will reflect these conditions set.
#'
#'
#' @param lambda Design matrix for the factor loadings, the lambda matric
#' @param lambda_constraint_lst (optional)  list containing 3 vectors - the row, column indices and values of the fixed parameters 
#' @param psi_constraint_lst (optional)  list containing 3 vectors - the row, column indices and values of the fixed parameters
#' @param delta_theta_constraint_lst (optional)  list containing 3 vectors - the row, column indices and values of the fixed parameters
#' @param omega_theta_constraint_lst (optional) list containing 3 vectors - the row, column indices and values of the fixed parameters
#' @param omega_theta_free_lst (optional) list containing 3 vectors - the row, column indices and values of the free parameters
#' @param identification (optional) method to identify the model, either by variance or by loadings (default "variance")
#' 
#' 
#' @return list containing params_start, params_free, params_value, params_sizes, params_free_sizes,
#' params_free_sizes_max, params_free_starting_pts, lambda_matrix_Free
#'
rnm_mod_to_torch_opts <- function(lambda,
                                  lambda_constraint_lst = NULL,
                                  psi_constraint_lst = NULL,
                                  delta_theta_constraint_lst = NULL,
                                  omega_theta_constraint_lst = NULL,
                                  omega_theta_free_lst = NULL,
                                  identification = 'variance'
                                  ){
  
  n <- dim(lambda)[1]
  m <- dim(lambda)[2]
  

  if(is.null(omega_theta_constraint_lst)){
  omega_theta_free_matrices <- rnm_constraint_matrices('omega_theta',omega_theta = "free",lambda=NULL,n_rows=n,n_cols=n,entries=omega_theta_free_lst,identification = identification)}
  else{
    omega_theta_free_matrices <- rnm_constraint_matrices('omega_theta',omega_theta = "constraint",lambda=NULL,n_rows=n,n_cols=n,entries=omega_theta_constraint_lst,identification = identification)
    
  }
  lambda_constraint_matrices <- rnm_constraint_matrices('lambda',omega_theta = "free",lambda=lambda,n_rows=n,n_cols=m,entries=lambda_constraint_lst,identification = identification)
  psi_constraint_matrices <- rnm_constraint_matrices('psi',omega_theta = "free",lambda=NULL,n_rows=m,n_cols=m,entries=psi_constraint_lst,identification = identification)
  delta_theta_constraint_matrices <- rnm_constraint_matrices('delta_theta',omega_theta = "free",lambda=NULL,n_rows=n,n_cols=n,entries=delta_theta_constraint_lst,identification = identification)
  
  lambda_start <- lavaan::lav_matrix_vec(lambda_constraint_matrices[[1]])
  lambda_matrix_free <- lambda_constraint_matrices[[2]]
  lambda_free <- lavaan::lav_matrix_vec(lambda_matrix_free )
  lambda_free_idx <- which(lambda_free!= 0)
  lambda_free_size <- length(lambda_free_idx)
  
  psi_start <- lavaan::lav_matrix_vech(psi_constraint_matrices[[1]])
  psi_free <- lavaan::lav_matrix_vech(psi_constraint_matrices[[2]])
  psi_free_idx <- which(psi_free!= 0)
  psi_free_size <- length(psi_free_idx)
  
  
  delta_theta_start <- lavaan::lav_matrix_vech(delta_theta_constraint_matrices[[1]])
  delta_theta_free <- lavaan::lav_matrix_vech(delta_theta_constraint_matrices[[2]])
  delta_theta_free_idx <- which(delta_theta_free!= 0)
  delta_theta_free_size <- length(delta_theta_free_idx)
  
  
  omega_theta_start <- lavaan::lav_matrix_vech(omega_theta_free_matrices[[1]])
  omega_theta_free <- lavaan::lav_matrix_vech(omega_theta_free_matrices[[2]])
  omega_theta_free_idx <- which(omega_theta_free!= 0)
  omega_theta_free_size <- length(omega_theta_free_idx)
  
  
  params_sizes <- sapply(list(lambda_start,psi_start,delta_theta_start,omega_theta_start), length)
  
  torch_list <- list(
    params_start = c(lambda_start,psi_start,delta_theta_start,omega_theta_start),
    params_free = c(lambda_free,psi_free,delta_theta_free,omega_theta_free),
    params_value = c(lambda_start,psi_start,delta_theta_start,omega_theta_start),
    params_sizes =   params_sizes,
    params_free_sizes = c(lambda_free_size, psi_free_size,delta_theta_free_size, omega_theta_free_size),
    params_free_sizes_max = c(length(which(lambda!= 0)), m*(m+1)/2 - m, n ,n*(n+1)/2 - n),
    params_free_starting_pts = c(1,1+params_sizes[1],1+sum(params_sizes[1:2]),1+sum(params_sizes[1:3])),
    lambda_matrix_free = lambda_matrix_free )
  

return(torch_list)
}



#' Torch rnm module created during stepping down/stepping up/pruning 
#'
#' Function for creating an rnm model
#'
#' @param params_vec param values for the initializing of the module
#'
#' @return A `torch_sem` object, which is an `nn_module` (torch object)
#'
#' @details
#' This function instantiates a torch object for computing the model-implied covariance matrix
#' based on an rnm model during the stepping down/stepping up/pruning processes. The methods are identical 
#' to torch_rnm and information about these methods can be looked up under torch_rnm
#' 
#' @import torch
#' @import lavaan
#' @importFrom R6 R6Class
#'
tensor_rnm_stepwise <- torch::nn_module(
  classname = "torch_network_model",
  initialize = function(params_vec){
    self$params_vec<-nn_parameter(params_vec)
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
    
    #Get separate tensors for lambda, theta, delta_psi, and omega_theta
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
  loglik = function(data = NULL) {
    if (is.null(data)){
      data = self$data
    }
    px <- distr_multivariate_normal(loc = self$mu, covariance_matrix = self$forward(),validate_args=FALSE)
    return(px$log_prob(data)$sum())
    
  },
  
  #' @section Methods:
  #'
  #' ## `$lasso_loss()`
  #' Returns lasso loss which is -2 times Multivariate normal log-likelihood of the data + Penalty.
  #'
  #' ### Arguments
  #' - `v` hyperparameter which controls for the penalty term inside the lasso loss function
  #'
  #' ### Value
  #' lasso loss value (torch scalar)
  lasso_loss = function(v,data = NULL) {
    if (is.null(data)){
      data = self$data
    }
    return(-2*self$loglik(data) + v*sum(abs(self$omega_theta$t()$reshape(-1)[strict_lower_triangle_idx(self$n)])))
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
    if(any(diag(as.matrix(self$Hinv)) < 0)) {print("Heywood Cases detected")}
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
  #' - `batch_size` (Optional) change the number of samples used for training 
  #' to reduce computational time. Default is NULL
  #'
  #' ### Value
  #' Self, i.e., the `torch_rnm` object with updated parameters 
  fit = function(lrate = 0.05, maxit = 5000, verbose = TRUE, tol = 1e-20, batch_size = NULL) {
    if (verbose) cat("Fitting SEM with Adam optimizer and MVN log-likelihood loss\n")
    optim <- optim_adam(self$params_vec, lr = lrate)
    scheduler <- lr_reduce_on_plateau(optim, factor = 0.5, patience = 5)
    prev_loss <- 0.0
    
    data_size <- nrow(self$data)  # Assuming self$data contains dataset
    if (is.null(batch_size) || batch_size >= data_size) {
      batch_size <- data_size  # Use full dataset if batch_size is NULL or too large
    }
    for (epoch in 1:maxit) {
      permuted_indices <- as_array(torch_randperm(data_size)) + 1  # Shuffle data at the start of each epoch
      num_batches <- ceiling(data_size / batch_size)
      
      for (batch in 1:num_batches) {
        batch_start <- (batch - 1) * batch_size + 1
        batch_end <- min(batch_start + batch_size - 1, data_size)
        batch_indices <- permuted_indices[batch_start:batch_end] 
        batch_data <- self$data[batch_indices]
        optim$zero_grad()
        loss <- -2 * self$loglik(batch_data)
        loss$backward()
        optim$step()
        counter <- 1
        while (!is_positive_definite(self$forward()) && counter < maxit) {
          with_no_grad(self$params_vec$copy_(get_alternative_update(self$params_vec)))
          counter <- counter + 1
          if (counter == maxit) {warning("No convergence due to issues with positive definiteness")}
          
        }
      }
      
      current_loss <- -2*self$loglik()$item()  
      if (verbose) {
        cat("\rEpoch:", epoch, " loss:", current_loss )
        flush.console()
      }
      
      if (epoch > 1 && abs(current_loss - prev_loss) < tol) {
        if (verbose) cat("\n")
        break
      }
      prev_loss <- current_loss
      scheduler$step(prev_loss)
    }
    
    if (epoch == maxit) warning("maximum iterations reached")
    
    return(invisible(self))
  }
  ,
  #' @section Methods:
  #'
  #' ## `$lasso_fit()`
  #' Fit a torch_lnm model using the lasso loss function.
  #' This function uses the Adam optimizer to estimate the parameters of a torch_lnm
  #'
  #' ### Arguments
  #' - `verbose`(Optional) whether to print progress to the console.  Default is TRUE.
  #' - `lrate` (Optional) learning rate of the Adam optimizer. Default is 0.05.
  #' - `maxit` (Optional) maximum number of epochs to train the model. Default is 5000.
  #' - `tol` (Optional) parameter change tolerance for stopping training. Default is 1e-20.
  #' - `v` (Optional) hyperparameter which controls for the penalty term inside the lasso loss function. Default is 1.
  #' - `epsilon` (Optional) Cutoff for lasso to set parameter to 0. Default is 0.0001.
  #' - `batch_size` (Optional) change the number of samples used for training 
  #' to reduce computational time. Default is NULL
  #' 
  #' ### Value
  #' Self, i.e., the `torch_rnm` object with updated parameters  
  
  lasso_fit = function(verbose=FALSE, lrate = 0.05, maxit = 5000, tol = 1e-20, v=1, epsilon = 0.0001, batch_size = NULL) {
    optim <- optim_adam(self$params_vec, lr = lrate, amsgrad = TRUE)
    scheduler <- lr_reduce_on_plateau(optim, factor = 0.5, patience = 5)
    prev_loss <- 0.0
    data_size <- nrow(self$data)  # Assuming self$data contains dataset
    if (is.null(batch_size) || batch_size >= data_size) {
      batch_size <- data_size  # Use full dataset if batch_size is NULL or too large
    }
    for (epoch in 1:maxit) {
      permuted_indices <- as_array(torch_randperm(data_size)) + 1  
      # Shuffle data at the start of each epoch
      
      num_batches <- ceiling(data_size / batch_size)
      
      for (batch in 1:num_batches) {
        batch_start <- (batch - 1) * batch_size + 1
        batch_end <- min(batch_start + batch_size - 1, data_size)
        batch_indices <- permuted_indices[batch_start:batch_end]
        batch_data <- self$data[batch_indices]
        
        optim$zero_grad()
        loss <- self$lasso_loss(v, batch_data)  # Compute loss on batch
        loss$backward()
        optim$step()
        counter <- 1
        while (!is_positive_definite(self$forward()) && counter < maxit) {
          with_no_grad(self$params_vec$copy_(get_alternative_update(self$params_vec)))
          counter <- counter + 1
          if (counter == maxit) {warning("No convergence due to issues with positive definiteness")}
          
        }
      }
      
      current_loss <- self$lasso_loss(v)$item()
      if (verbose) {
        cat("\rEpoch:", epoch, " loss:", current_loss)
        flush.console()
      }
      if (epoch > 1 && abs(current_loss - prev_loss) < tol) {
        break
      }
      prev_loss <- current_loss
      scheduler$step(prev_loss)
    }
    
    self$lasso_update_params_added(v=v, epsilon=epsilon)
    
    if (epoch == maxit) warning("maximum iterations reached") 
    
    return(invisible(self))
  },
  
  #' @section Methods:
  #'
  #' ## `$custom_fit()`
  #' Fit a torch_lnm model using a custom loss function supplied to the torch_lnm module.
  #' The custom loss function has to have 2 input parameters, the model covariance matrix and 
  #' the data. (See Example Code for clarification)
  #' This function uses the Adam optimizer to estimate the parameters of a torch_lnm
  #'
  #' ### Arguments
  #' - `lrate` (Optional) learning rate of the Adam optimizer. Default is 0.05.
  #' - `maxit` (Optional) maximum number of epochs to train the model. Default is 5000.
  #' - `verbose` (Optional) whether to print progress to the console.  Default is TRUE.
  #' - `tol` (Optional) parameter change tolerance for stopping training. Default is 1e-20.
  #' - `batch_size` (Optional) change the number of samples used for training 
  #' to reduce computational time. Default is NULL
  #'
  #' ### Value
  #' Self, i.e., the `torch_rnm` object with updated parameters  
  
  custom_fit = function(lrate = 0.05, maxit = 5000,verbose = TRUE, tol = 1e-20, batch_size = NULL){
    if(is.null(self$custom_loss)){warning('No Custom Loss function provided!')}
    if (verbose) cat("Fitting SEM with Adam optimizer and custom loss\n")
    optim <- optim_adam(self$params_vec, lr = lrate)
    scheduler <- lr_reduce_on_plateau(optim, factor = 0.5, patience = 5)
    prev_loss <- 0.0
    data_size <- nrow(self$data)  # Assuming self$data contains dataset
    if (is.null(batch_size) || batch_size >= data_size) {
      batch_size <- data_size  # Use full dataset if batch_size is NULL or too large
    }
    for (epoch in 1:maxit) {
      permuted_indices <- as_array(torch_randperm(data_size)) + 1   
      # Shuffle data at the start of each epoch
      
      num_batches <- ceiling(data_size / batch_size)
      
      for (batch in 1:num_batches) {
        batch_start <- (batch - 1) * batch_size + 1
        batch_end <- min(batch_start + batch_size - 1, data_size)
        batch_indices <- permuted_indices[batch_start:batch_end]
        batch_data <- self$data[batch_indices]
        optim$zero_grad()
        loss <- self$custom_loss(self$sigma, batch_data)  # Compute loss on batch
        loss$backward()
        optim$step()
        counter <- 1
        while (!is_positive_definite(self$forward()) && counter < maxit) {
          with_no_grad(self$params_vec$copy_(get_alternative_update(self$params_vec)))
          counter <- counter + 1
          if (counter == maxit) {warning("No convergence due to issues with positive definiteness")}
          
        }
      } 
      
      current_loss <-  self$custom_loss(self$forward(), self$data)$item()
      if (verbose) {
        cat("\rEpoch:", epoch, " loss:", current_loss )
        flush.console()
      }
      if (epoch > 1 && abs(current_loss - prev_loss) < tol) {
        break
      }
      prev_loss <- current_loss 
      scheduler$step(prev_loss)
    }
    
    
    if (epoch == maxit) warning("maximum iterations reached")
    return(invisible(self))
    
  },
  
  
  #' @section Methods:
  #'
  #' ## `$lasso_update_params_added(v,epsilon)`
  #' Update the model attribute, self$lasso_num_params_added This is will be done automatically if
  #' is called after `$lasso_fit()`.
  #' 
  #' ### Arguments
  #' - `v` hyperparameter which controls for the penalty term inside the lasso loss function.
  #' - `epsilon`  Cutoff for lasso to set parameter to 0. 
  #'
  #' ### Value
  #' None 
  lasso_update_params_added = function(v,epsilon){
    if (is.null(self$sigma)) warning('Data must be fitted first')
    if (self$lasso == FALSE) warning('Set lasso = TRUE first!')
    lower_triangle_idx <- lower_triangle_idx(self$n)
    partial_correlations <- as_array(self$omega_theta$t()$reshape(-1)[lower_triangle_idx]$reshape(-1))
    ind_idx <- which(as_array(self$params_free[self$params_free_starting_pts[4]:(self$params_free_starting_pts[4]+length(lower_triangle_idx)-1)]$reshape(-1))==1)
    partial_correlations <- partial_correlations[ind_idx]
    ind <- which(abs(partial_correlations) > epsilon)
    self$lasso_num_params_added <- length(ind) 
  },
  
  
  #' @section Methods:
  #'
  #' ## `$get_all_latent_pairings()`
  #' Get all the possible combinations of latent pairs, or the pairs of latent variables as prescribed
  #' in the model. For example, if there are 3 latent variables (a,b,c), we have a total of 3 choose 2, or 3
  #' latent pairs possible and the function will return c('a~b', 'a~c', 'b~c').
  #'
  #' ### Value
  #' vector of all possible combinations of latent pairs
  get_all_latent_pairings=function(){
    self$latent_pairings <- vector(mode='character',length = factorial(self$m)/(factorial(2)*factorial(self$m-2)))
    counter <- 1
    for (i in 1:(self$m)){
      for (j in (i):self$m){
        self$latent_pairings[counter] <- paste0(stringr::str_sub(self$latents[i],1,3),'~',stringr::str_sub(self$latents[j],1,3))
        counter <- counter + 1
      }
    }
    return(self$latent_pairings)
  },
  
  #' @section Methods:
  #'
  #' ## `$get_all_residual_pairings()`
  #' Get all the possible combinations of residual pairs, or the pairs of observed variables as prescribed
  #' in the model. For example, if there are 3 observed variables (a,b,c), we have a total of 3 choose 2, or 3
  #' pairs possible and the function will return c('a~b', 'a~c', 'b~c').
  #'
  #' ### Value
  #' vector of all possible combinations of residual/observed variable pairs
  get_all_residual_pairings=function(){
    self$residual_pairings <- vector(mode='character',length = factorial(self$n)/(factorial(2)*factorial(self$n-2)))
    counter <- 1
    for (i in 1:(self$n)){
      for (j in (i):self$n){
        self$residual_pairings[counter] <- paste0(stringr::str_sub(self$vars[i],1,3),'~',stringr::str_sub(self$vars[j],1,3))
        counter <- counter + 1
      }
    }
    return(self$residual_pairings)
  },
  
  #' @section Methods:
  #'
  #' ## `$get_df()`
  #' Get the number of degrees of freedom in the model which equals n(n+1)/2 - number of free parameters.
  #' where n is the dimension of the sample covariance matrix.
  #'
  #' ### Value
  #' Degrees of freedom
  get_df=function(){
    if (self$lasso==TRUE){
      if(is.null(self$sigma)){
        print('Fit using lasso first to get the number of df of the model selected by lasso. If not, df reflected is not accurate.')}
      self$df <- self$n*(self$n+1)/2 - sum(self$params_free_sizes[1:3]) - self$lasso_num_params_added 
    }
    else {
      self$df <- self$n*(self$n+1)/2 - sum(self$params_free_sizes[1:4]) 
    }
    return(self$df)
  },
  
  #' @section Methods:
  #'
  #' ## `$get_loadings()`
  #' Get all the free factor loadings (entries of lambda matrix) determined after model fit. 
  #' This includes fit, custom_fit and lasso_fit. 
  #' 
  #' ### Value
  #' Dataframe of factor loadings and if the default log-likelihood fn is used in to fit,
  #' standard errors and p-values are also provided
  get_loadings=function(){
    if (is.null(self$sigma)) warning('Data must be fitted first')
    ind_idx <- vector(mode='numeric',length=self$params_free_sizes[1])
    loading_names <- vector(mode='numeric',length=self$params_free_sizes[1])
    total_length <- 0
    counter <- 1
    
    for (i in 1:self$m){
      col_idx <- which(self$lambda_matrix_free[,i] != 0)
      if (counter != 1){
        start <- total_length + 1
        end <- start + length(col_idx) - 1
      } else{
        start <- 1
        end <- length(col_idx)
      }
      loading_names[start:end] <- sapply(self$vars[col_idx], 
                                         function(x) paste0(stringr::str_sub(x,1,3),'~',
                                                            stringr::str_sub(self$latents[i],1,3)))
      
      ind_idx[start:end] <- col_idx + (counter-1)*self$n
      counter <- counter + 1
      total_length <- total_length + length(col_idx)
      
    }
    
    
    col1<-as_array(self$lambda$t()$reshape(-1)[ind_idx])
    
    if (self$lasso == TRUE || !is.null(self$custom_loss)){
      self$loadings <- data.frame(`Factor loadings` = round(col1,3))
      rownames(self$loadings) <- loading_names
      return(self$loadings)
    }
    H_ind1 <- 1
    H_ind2 <- self$params_free_sizes[1]
    col2 <- as_array(torch_diag(self$inverse_Hessian()[H_ind1 :H_ind2,][,H_ind1 :H_ind2])$reshape(-1))
    col2<-sqrt(col2)
    col3<-sapply(col1/col2, get_p_value)
    col4<-ifelse(col3<0.05, "*","")
    self$loadings <- data.frame(`Factor loadings` = round(col1,3),`Standard error`= round(col2,3),
                                `P value`= round(col3,4),`Significant`=col4)
    rownames(self$loadings) <- loading_names
    return(self$loadings)
  },
  #' @section Methods:
  #'
  #' ## `$get_psi()`
  #' Get all the free partial correlations (entries of lower triangle of omega_theta matrix) 
  #' determined after model fit. This includes fit, custom_fit and lasso_fit. 
  #' 
  #'
  #'
  #' ### Value
  #' Dataframe of partial correlations and if the default log-likelihood fn is used in to fit,
  #' standard errors and p-values are also provided 
  get_psi = function(){
    if (is.null(self$sigma)) warning('Data must be fitted first')
    lower_triangle_idx <- lower_triangle_idx(self$m)
    psi <- as_array(self$psi$t()$reshape(-1)[lower_triangle_idx]$reshape(-1))
    latent_pairings <- self$get_all_latent_pairings()
    ind_idx <- which(as_array(self$params_free[self$params_free_starting_pts[2]:(self$params_free_starting_pts[2]+length(lower_triangle_idx)-1)]$reshape(-1))==1)
    psi <-  psi[ind_idx]
    latent_pairings <- latent_pairings[ind_idx]
    
    
    if (self$lasso == TRUE || !is.null(self$custom_loss)){
      col1 <- psi
      self$psi_abridged <- data.frame(`Covariance Pairs` = round(col1,3))
      rownames(self$psi_abridged) <- latent_pairings
      return(self$psi_abridged)
    }
    
    col1 <-  psi
    H_ind1 <- (sum(self$params_free_sizes[1])+1)
    H_ind2 <- (sum(self$params_free_sizes[1:2]))
    col2 <- sqrt(as_array(torch_diag(self$inverse_Hessian()[H_ind1:H_ind2,][,H_ind1:H_ind2])$reshape(-1)))
    
    col3<-sapply(col1/col2, get_p_value)
    col4<-ifelse(col3<0.05, "*","")
    self$psi_abridged <- data.frame(`Covariance Pairs` = round(col1,3),`Standard error`= round(col2,3),
                                    `P value`= round(col3,4),`Significant`=col4)
    rownames(self$psi_abridged) <- latent_pairings
    return(self$psi_abridged)
    
  },
  
  
  #' @section Methods:
  #'
  #' ## `$get_delta_theta()`
  #' Get all the free residuals (diagonal entries of theta matrix) determined after model fit. 
  #' This includes fit, custom_fit and lasso_fit. 
  #'
  #' ### Value
  #' Dataframe of residuals and if the default log-likelihood fn is used in to fit,
  #' standard errors and p-values are also provided
  get_delta_theta=function(){
    
    if (is.null(self$sigma)) warning('Data must be fitted first')
    ind_idx <- which(as_array(self$params_free$reshape(-1))[(sum(self$params_sizes[1:2]) + 1):sum(self$params_sizes[1:3])] == TRUE)
    names_ind_idx <- find_correct_indices(diagonal_idx(self$n),ind_idx)
    residual_names <- self$vars[names_ind_idx]
    counter <- 1
    H_ind1 <- (sum(self$params_free_sizes[1:2])+1)
    H_ind2  <- sum(self$params_free_sizes[1:3])
    col1<- lavaan::lav_matrix_vech(as.matrix(self$delta_theta))[ind_idx]
    col2 <- as_array(torch_diag(self$inverse_Hessian()[H_ind1 :H_ind2 ,]
                                [,H_ind1 :H_ind2 ])$reshape(-1))
    
    if (self$lasso == TRUE || !is.null(self$custom_loss)){
      self$residuals <- data.frame(`Residuals` = round(col1,3))
      rownames(self$residuals) <- residual_names
      return(self$residuals)
    }
    
    col2<-sqrt(col2)
    col3<-sapply(col1/col2, get_p_value)
    col4<-ifelse(col3<0.05, "*","")
    self$residuals <- data.frame(`Residuals` = round(col1,5),`Standard error`= round(col2,5),
                                 `P value`= round(col3,5),`Significant`=col4)
    rownames(self$residuals) <- residual_names
    return(self$residuals)
  },
  
  #' @section Methods:
  #'
  #' ## `$get_partial_correlations(epsilon)`
  #' Get all the free partial correlations (entries of lower triangle of omega_theta matrix) 
  #' determined after model fit. This includes fit, custom_fit and lasso_fit. 
  #' 
  #' ### Arguments
  #' - `epsilon` (Optional) If lasso fit was used, this determines the threshold, under which
  #' a partial correlation is no longer considered a free parameter and is set to 0. Default value is
  #' 0.00001
  #'
  #' 
  #'
  #' ### Value
  #' Dataframe of partial correlations and if the default log-likelihood fn is used in to fit,
  #' standard errors and p-values are also provided 
  get_partial_correlations = function(epsilon = 0.0001,silent = FALSE){
    is_integer0 <- function(x) {
      is.integer(x) && length(x) == 0
    }
    if (is.null(self$sigma)) warning('Data must be fitted first')
    if (!is.null(self$omega_theta_constraint_lst)){
      print("Returning the omega_theta matrix")
      return(self$omega_theta)
    }
    lower_triangle_idx <- lower_triangle_idx(self$n)
    partial_correlations <- as_array(self$omega_theta$t()$reshape(-1)[lower_triangle_idx]$reshape(-1))
    residual_pairings <- self$get_all_residual_pairings()
    ind_idx <- which(as_array(self$params_free[self$params_free_starting_pts[4]:(self$params_free_starting_pts[4]+length(lower_triangle_idx)-1)]$reshape(-1))==1)
    
    if(is_integer0(ind_idx)){warning("No partial correlations in model!")}
    
    partial_correlations <- partial_correlations[ind_idx]
    residual_pairings <- residual_pairings[ind_idx]
    
    
    if (self$lasso == TRUE || !is.null(self$custom_loss)){
      if(self$lasso == TRUE){
        ind <- which(abs(partial_correlations) > epsilon)
        to_be_added_idx <- which(abs(partial_correlations) > epsilon)
        to_be_added_indicator <- ifelse(abs(partial_correlations) > epsilon,1,0)
        self$added_partial_correlations <- residual_pairings[to_be_added_idx]
        residual_pairings_remained <- residual_pairings[ind]
        col1 <- partial_correlations[ind]
        self$partial_corr <- data.frame(`Partial Corr` = round(col1,3))
        rownames(self$partial_corr) <- residual_pairings_remained 
        self$lasso_num_params_added <- length(ind) 
        
        return(list(self$partial_corr,self$lasso_num_params_added,self$added_partial_correlations,
                    extract_non_zero_entries(vech_to_symmetric_zero_diag(to_be_added_indicator,self$n))))
      } else{
        col1 <- partial_correlations
        self$partial_corr <- data.frame(`Partial Corr` = round(col1,3))
        rownames(self$partial_corr) <- residual_pairings
        if (silent == TRUE){
          return(NULL)
        } else{
          return(self$partial_corr)
        }
      }
    }
    
    if (torch_sum(partial_correlations)$item() == 0) {return("No partial correlations in model!")}
    
    col1 <- partial_correlations
    H_ind1 <- (sum(self$params_free_sizes[1:3])+1)
    H_ind2 <- sum(self$params_free_sizes)
    col2 <- sqrt(as_array(torch_diag(self$inverse_Hessian()[H_ind1:H_ind2,][,H_ind1:H_ind2])$reshape(-1)))
    
    col3<-sapply(col1/col2, get_p_value)
    col4<-ifelse(col3<0.05, "*","")
    self$partial_corr <- data.frame(`Partial Corr` = round(col1,3),`Standard error`= round(col2,3),
                                    `P value`= round(col3,4),`Significant`=col4)
    rownames(self$partial_corr) <- residual_pairings
    return(self$partial_corr)
    
  },
  
  #' @section Methods:
  #'
  #' ## `$get_criterion_value()`
  #' 
  #' Get the criterion value of the model according to the type of criterion specified.
  #' The possibilities include AIC (Akaike Information Criterion), 
  #' BIC (Bayesian Information Criterion), EBIC (Extended Bayesian Information Criterion) 
  #' and the chisquare statistic.
  #' 
  #' ### Arguments
  #' - `criterion` Name of the Criterion used, AIC, BIC, EBIC, or chisq.
  #' - `gamma` (optional) Gamma hyperparmaeter for the EBIC if the EBIC is used. (default value is 0.5)
  #'
  #' ### Value
  #' Criterion Value 
  get_criterion_value = function(criterion,gamma = 0.5){
    if (criterion == "AIC") return(2*(self$n*(self$n+1)/2 - self$get_df()) - 2*self$loglik())
    else if (criterion == "BIC") return((self$n*(self$n+1)/2 - self$get_df())*log(self$num_obs) - 2*self$loglik())
    else if (criterion == "EBIC"){
      num_params_free <- (self$n*(self$n+1)/2 - self$get_df())
      return((num_params_free) *log(self$num_obs) - 2*self$loglik() + 2*gamma*log(num_params_free))
    }
    else if (criterion == "chisq") {
      if (self$lasso == TRUE) {warning('Not Applicable when lasso = TRUE')}
      if (is.null(self$sigma)){self$forward()}
      return((self$num_obs-1)*(torch_det(self$sigma)$log() + torch_trace(self$cov_matrix$mm(torch_inverse(self$sigma))) - torch_det(self$cov_matrix)$log()- self$n ))
    }
  },
  #' @section Methods:
  #'
  #' ## `$get_fit_metrics()`
  #' 
  #' Get the fit metrics after the model is fit. Metrics returned are the CFI (Comparative Fit Index), 
  #' TLI (Tucker-Lewis Index), the RMSEA (Root Mean Square Error of Approximation) 
  #' and the confidence interval of the RMSEA. These metrics compare the fitted model to
  #' the baseline model which assumes all observed variables are uncorrelated.
  #'
  #' ### Arguments
  #'
  #' ### Value
  #' Dataframe containing the fit metrics
  get_fit_metrics = function(){
    metric_names <- c('CFI', 'TLI', 'RMSEA', 'RMSEA.lower.0.05', 'RMSEA.upper.0.95')
    std_dev <- torch_sqrt(torch_diag(self$cov_matrix))
    outer_product <- torch_outer(std_dev, std_dev)
    cor_matrix <- self$cov_matrix / outer_product
    cor_matrix[torch_isnan(cor_matrix)] <- 0
    chisq_baseline <- (-log(linalg_det(cor_matrix))*(self$num_obs - 1))
    df_baseline <- self$n*(self$n-1)/2
    chisq_model <- self$get_criterion_value('chisq')
    df_model <- self$get_df()
    self$cfi <- (1 - (chisq_model-df_model)/(chisq_baseline-df_baseline))$item()
    self$tli <- ((chisq_baseline/df_baseline - chisq_model/df_model)/(chisq_baseline/df_baseline - 1))$item()
    self$rmsea <- sqrt((chisq_model/df_model-1)/(self$num_obs - 1))$item()
    if ((qchisq(0.05,df = df_model,ncp = (chisq_model-df_model)$item()) - df_model) <= 0){self$rmsea_lower <- 0}
    else{
      self$rmsea_lower <- sqrt(((qchisq(0.05,df = df_model,ncp = (chisq_model-df_model)$item()) - df_model)/df_model/(self$num_obs-1)))
    }
    self$rmsea_upper <- max(sqrt(((qchisq(0.95,df = df_model,ncp = (chisq_model-df_model)$item()) - df_model)/df_model/(self$num_obs-1))),0,na.rm=TRUE)
    self$metrics <- data.frame("Metrics" = metric_names, "Values" = c(self$cfi, self$tli,self$rmsea,self$rmsea_lower,self$rmsea_upper))
    return(self$metrics)
  }
  
)



#' Given a torch_rnm module, mod, with params_vec, this copies all the attributes of mod 
#' into a new torch_rnm_stepwise module (except for params_free and params_free_sizes) which 
#' is instantiated during stepping up/stepping down/pruning.
#' 
#'
#' @param torch_rnm torch_rnm module which is undergoing stepping down/stepping up/pruning
#' @param params_vec params_vec which belongs to the torch_enm module. If name of module is mod,
#' "mod$params_vec" should be input here.
#' 
#' @return torch_rnm_stepwise module with the exact same attributes as the original torch_enm module
#' 
copy_rnm_attributes <- function(mod,params_vec){
  output_model <- tensor_rnm_stepwise(params_vec)
  output_model$model_type <- mod$model_type
  output_model$data <- mod$data
  output_model$vars <- mod$vars
  output_model$latents <- mod$latents
  output_model$identification <- mod$identification
  output_model$custom_loss <- mod$custom_loss
  output_model$lasso <- mod$lasso
  output_model$device <- mod$device
  output_model$dtype <- mod$dtype
  output_model$n <- mod$n
  output_model$m <- mod$m
  output_model$num_obs <- mod$num_obs
  output_model$params_value <- mod$params_value
  output_model$lambda_matrix_free <- mod$lambda_matrix_free
  output_model$params_sizes <- mod$params_sizes
  output_model$params_free_sizes_max <- mod$params_free_sizes_max
  output_model$params_free_starting_pts <- mod$params_free_starting_pts
  output_model$cov_matrix <- mod$cov_matrix 
  output_model$lasso_num_params_added  <- mod$lasso_num_params_added
  output_model$psi_dup_idx <- mod$psi_dup_idx
  output_model$delta_theta_dup_idx <- mod$delta_theta_dup_idx
  output_model$omega_theta_dup_idx <- mod$omega_theta_dup_idx
  output_model$B <-  mod$B 
  output_model$I_mat_m <- mod$I_mat_m
  output_model$I_mat_n <- mod$I_mat_n
  output_model$mu <- mod$mu
  
  
  return(output_model)
}

  
#' NOT RECOMMENDED FOR RNM DUE TO LONG EXECUTION TIMES
#' 
#' Helper function for rnm_stepdown. Given an original torch_rnm/torch_rnm_stepwise module, the function fits all possible models
#' during one iteration of the stepping down process and returns either the best model based on criterion score
#' after this fitting process or the original torch_rnm/torch_rnm_stepwise module if all new models give a
#' worse model fit based on criterion score. A possible model here is a model with one of the
#' originally non-zero partial correlations now being set to zero.
#' 
#'
#' @param mod original torch_rnm/torch_rnm_stepwise module undergoing stepping down
#' @param criterion string indicating criterion to use - "AIC", "BIC or "EBIC"
#' @param gamma (Optional) value of gamma if "EBIC" is used. Default value is 0.5.
#' 
#' 
#' @return either the original torch_rnm/torch_rnm_stepwise module or 
#' a torch_rnm_stepwise module with a better criterion score
#' 
rnm_stepup_find_alt_models <- function(mod,criterion = 'BIC',gamma = 0.5,batch_size = NULL){
  omega_theta_ind_1 <- mod$params_free_starting_pts[4]
  omega_theta_ind_2 <- omega_theta_ind_1 + mod$params_sizes[4]
  omega_theta_free <- mod$params_free[omega_theta_ind_1:omega_theta_ind_2]
  sum_omega_theta_free = torch_sum(omega_theta_free)$item()
  if ((sum_omega_theta_free - 1) == 0) {warning('Cant stepdown anymore! Left with one partial correlation!')}
  other_params_free <- mod$params_free[1:(omega_theta_ind_1-1)]
  other_params_vec <- mod$params_vec[1:(omega_theta_ind_1-1)]
  other_models_free_params <- flip_one_0_to_1(omega_theta_free)
  num_omega_theta_params <- length(omega_theta_free)
  num_possible_models <- length(other_models_free_params)
  list_of_criterion_values <- vector("numeric", length = num_possible_models)
  models <- vector("list", length = num_possible_models)
  current_criterion_value <- mod$get_criterion_value(criterion,gamma)$item()
  for(i in 1:num_possible_models){
    new_params_vec <- torch_cat(list(mod$params_vec[1:(omega_theta_ind_1-1)],torch_zeros(num_omega_theta_params)),dim = 1)
    clone <- copy_rnm_attributes(mod,new_params_vec$detach())
    clone$params_free <- torch_cat(list(other_params_free,other_models_free_params[[i]]),dim=1)
    clone$params_free_sizes <- c(mod$params_free_sizes[1:3], mod$params_free_sizes[4] - 1) 
    clone$fit(verbose=F,batch_size = batch_size)
    models[[i]] <- clone
    list_of_criterion_values[i] <- clone$get_criterion_value(criterion,gamma)$item()
  }
}

#' NOT RECOMMENDED FOR RNM DUE TO LONG EXECUTION TIMES
#' 
#' Step Down function for torch_rnm/torch_rnm_stepwise module
#' 
#' Performs stepping down to give the best model based on the selected criterion score
#' At the end of each iteration, one of the non-zero partial correlations will be set to zero and this
#' corresponds to the model with the best model fit based on criterion score 
#' out of all possibilities (number of non-zero partial correlations at the start). If the models
#' fit at the end of 1 iteration with 1 partial correlation removed each do not lead to
#' a better criterion score, the model found at the end of the previous iteration is returned.
#' 
#' The process terminates if the model is left with only 1 partial correlation at the end of 
#' all the iterations.
#'
#' @param mod original torch_rnm/torch_rnm_stepwise module undergoing stepping down
#' @param criterion string indicating criterion to use - "AIC", "BIC or "EBIC"
#' @param gamma (Optional) value of gamma if "EBIC" is used. Default value is 0.5.
#' 
#' 
#' @return either the original torch_rnm/torch_rnm_stepwise module or 
#' a torch_rnm_stepwise module with a best criterion score after stepping down
#' 
#' 
rnm_stepup<- function(mod,criterion = 'BIC',gamma = 0.5, batch_size = NULL){
  current_value <- mod$get_criterion_value(criterion,gamma)$item()
  stepup_model <- rnm_stepup_find_alt_models(mod,criterion,gamma,batch_size = batch_size)
  if(stepup_model$get_criterion_value(criterion,gamma)$item() == current_value){
    return(mod)
  }else{
    rnm_stepup(stepup_model,criterion,batch_size = batch_size)
    
  }
}

#' NOT RECOMMENDED FOR RNM DUE TO LONG EXECUTION TIMES
#' 
#' #' Helper function for rnm_stepup. Given an original torch_rnm/torch_rnm_stepwise module, the function fits all possible models
#' during one iteration of the stepping up process and returns either the best model based on criterion score
#' after this fitting process or the original torch_rnm/torch_rnm_stepwise module if all new models give a
#' worse model fit based on criterion score. A possible model here is a model with one of the
#' originally zero partial correlations now being set to non-zero.
#' 
#'
#' @param mod original torch_rnm/torch_rnm_stepwise module undergoing stepping up
#' @param criterion string indicating criterion to use - "AIC", "BIC or "EBIC"
#' @param gamma (Optional) value of gamma if "EBIC" is used. Default value is 0.5.
#' 
#' 
#' @return either the original torch_rnm/torch_rnm_stepwise module or 
#' a torch_rnm_stepwise module with a better criterion score
#' 
rnm_stepdown_find_alt_models <- function(mod,criterion = 'BIC',gamma = 0.5, batch_size = NULL){
  omega_theta_ind_1 <- mod$params_free_starting_pts[4]
  omega_theta_ind_2 <- omega_theta_ind_1 + mod$params_sizes[4]
  omega_theta_free <- mod$params_free[omega_theta_ind_1:omega_theta_ind_2]
  sum_omega_theta_free = torch_sum(omega_theta_free)$item()
  if ((sum_omega_theta_free - 1) == 0) {warning('Cant stepdown anymore! Left with one partial correlation!')}
  other_params_free <- mod$params_free[1:(omega_theta_ind_1-1)]
  other_params_vec <- mod$params_vec[1:(omega_theta_ind_1-1)]
  other_models_free_params <- flip_one_1_to_0(omega_theta_free)
  num_omega_theta_params <- length(omega_theta_free)
  num_possible_models <- length(other_models_free_params)
  list_of_criterion_values <- vector("numeric", length = num_possible_models)
  models <- vector("list", length = num_possible_models)
  current_criterion_value <- mod$get_criterion_value(criterion,gamma)$item()
  for(i in 1:num_possible_models){
    new_params_vec <- torch_cat(list(mod$params_vec[1:(omega_theta_ind_1-1)],torch_zeros(num_omega_theta_params)),dim = 1)
    clone <- copy_rnm_attributes(mod,new_params_vec$detach())
    clone$params_free <- torch_cat(list(other_params_free,other_models_free_params[[i]]),dim=1)
    clone$params_free_sizes <- c(mod$params_free_sizes[1:3], mod$params_free_sizes[4] - 1) 
    clone$fit(verbose=F,batch_size = batch_size)
    models[[i]] <- clone
    list_of_criterion_values[i] <- clone$get_criterion_value(criterion,gamma)$item()
  }
  
  min_index <- which.min(list_of_criterion_values)
  
  if (list_of_criterion_values[min_index] < current_criterion_value) return(models[[min_index]])
  
  else {return(mod)}
}

#' NOT RECOMMENDED FOR RNM DUE TO LONG EXECUTION TIMES
#' 
#' Step Up function for torch_rnm/torch_rnm_stepwise module
#' 
#' Performs stepping up to give the best model based on the selected criterion score
#' At the end of each iteration, one of the originally zero partial correlations will be
#' turned to non-zero and this corresponds to the model with the best model fit based on criterion score
#' corresponds to the model with the best model fit based on criterion score 
#' out of all possibilities (number of zero partial correlations at the start). 
#' If the models fit at the end of 1 iteration with 1 partial correlation set to non-zero each 
#' do not lead to a better criterion score, the model found at the end of the previous iteration is returned.
#' 
#' The process terminates if the model is saturated (all partial correlations are non-zero) at the end of the iteration.
#'
#' @param mod original torch_rnm/torch_rnm_stepwise module undergoing stepping up
#' @param criterion string indicating criterion to use - "AIC", "BIC or "EBIC"
#' @param gamma (Optional) value of gamma if "EBIC" is used. Default value is 0.5.
#' 
#' 
#' @return either the original torch_rnm/torch_rnm_stepwise module or 
#' 
rnm_stepdown <- function(mod,criterion = 'BIC',gamma = 0.5, batch_size = NULL){
  
  
  current_value <- mod$get_criterion_value(criterion,gamma)$item()
  stepdown_model <- rnm_stepdown_find_alt_models(mod,criterion,gamma,batch_size=batch_size)
  
  if(stepdown_model$get_criterion_value(criterion,gamma)$item() == current_value){
    return(mod)
  }else{
    rnm_stepdown(stepdown_model,criterion,gamma,batch_size=batch_size)
    
  }
}


#' Helper function for rnm_prune. Given an rnm model, it removes all partial correlations
#' found to be insignificant from the model and refits the model without these partial correlations.
#' Thereafter, it returns the updated model.
#' 
#'
#' @param mod original torch_rnm/torch_rnm_stepwise module 
#' 
#' 
#' @return new refitted model with insignificant partial correlations removed
#' 
rnm_prune_helper <- function(mod,batch_size = NULL){
  
  index <- which(mod$partial_corr$Significant == "")
  num_params_removed <- length( index)  
  if( num_params_removed == 0){return(mod)}
  omega_theta_ind_1 <- mod$params_free_starting_pts[4]
  omega_theta_ind_2 <- omega_theta_ind_1 + mod$params_sizes[4]
  omega_theta_free <- mod$params_free$clone()[omega_theta_ind_1:omega_theta_ind_2]
  one_positions <- torch_nonzero(omega_theta_free)$squeeze(2)  
  omega_theta_free[one_positions[index]] <- 0 
  sum_omega_theta_free <- torch_sum(omega_theta_free)$item()
  other_params_free <- mod$params_free[1:(omega_theta_ind_1-1)]
  other_params_vec <- mod$params_vec[1:(omega_theta_ind_1-1)]
  new_free_params <- omega_theta_free
  num_omega_theta_params <- length(omega_theta_free)
  new_params_vec <- torch_cat(list(mod$params_vec[1:(omega_theta_ind_1-1)],torch_zeros(num_omega_theta_params)),dim = 1)
  clone <- copy_rnm_attributes(mod,new_params_vec$detach())
  clone$params_free <- torch_cat(list(other_params_free,new_free_params),dim=1)
  clone$params_free_sizes <- c(mod$params_free_sizes[1:3], mod$params_free_sizes[4] - num_params_removed) 
  clone$fit(verbose=F,batch_size=batch_size)
  return(clone) 
  
}

#' Pruning function for torch_rnm/torch_rnm_stepwise module
#' 
#' Given an r nm model, it iteratively removes insignificant partial correlations and refits models until
#' all partial correlations are found to be significant. Thereafter ,it returns this model
#' with only significant partial correlations.
#' 
#'
#' @param mod original torch_rnm/torch_rnm_stepwise module 
#' 
#' 
#' @return new refitted model with only significant partial correlations 
#' 
rnm_prune <- function(mod,batch_size = NULL){
  pruned_model <-  rnm_prune_helper(mod,batch_size = batch_size)
  
  if(length(which(pruned_model$partial_corr$Significant == "")) == 0){
    return(pruned_model)
  }
  else{
    rnm_prune(pruned_model,batch_size = batch_size)
  }
}


#' LASSO for RNM
#' 
#' Performs a lasso search to find the value of hyperparameter v which correspond to the model 
#' which gives the lowest criterion scores (AIC/BIC/EBIC). At the end, the free parameters
#' for omega_theta (parameters which have been set to zero) will also be returned. Each model 
#' is fit with a different value of v and partial correlations which are lower than the threshold value are set to 0
#' These partial correlations are removed from the pool of free parameters before the criterion score is calculated.
#'  
#' @import pracma
#' 
#' @param mod original torch_rnm/torch_rnm_stepwise module undergoing stepping up
#' @param criterion (Optional) string indicating criterion to use - "AIC", "BIC or "EBIC". Default is BIC.
#' @param v_values (Optional) values of the hyperparameter v to be used in the search. Default range 
#' of values is 30 values of v from 0.01 to 100 spread out on a log scale.
#' @param lrate (Optional) learning rate for lasso. Default value is 0.01.
#' @param epsilon (Optional) threshold under which partial correlations will be set to 0. Default is 0.0001.
#' @param gamma (Optional) value of gamma if "EBIC" is used. Default value is 0.5.
#' 
#' 
#' @return a list containing the following: 1)value of v of the model which 
#' gives the best criterion score and 2)free parameters for omega_theta
#' 
rnm_lasso_explore <- function(mod, criterion = "BIC", v_values = pracma::logspace(log10(0.01), log10(100), 30) ,lrate= 0.01, epsilon = 0.0001, gamma = 0.5, batch_size = batch_size){
  if (mod$lasso == FALSE) {warning('Set lasso to TRUE first!')}
  criterion_values <- vector(mode = 'numeric',length=length(v_values))
  output_params <- list()
  df_values <- vector(mode = 'numeric',length=length(v_values))
  for (i in 1:length(v_values)){
    mod$lasso_fit(v = v_values[i],lrate = lrate,epsilon = epsilon, batch_size = batch_size)
    mod$lasso_update_params_added(epsilon = epsilon)
    criterion_values[i] <- mod$get_criterion_value(criterion,gamma)$item()
    df_values[i] <- mod$get_df() 
    output_params[[i]] <- mod$get_partial_correlations()[[4]]
  }
  ind <- which.min(criterion_values)
  if (df_values[ind] < 0) {print("Model will be underidentified, consider increasing penalty parameter")}
  return(list(penalty_value = v_values[ind], free_omega_theta_params = output_params[[ind]]))
}








