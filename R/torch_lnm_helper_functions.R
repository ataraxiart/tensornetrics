library(torch)
library(lavaan)
library(pracma)


#' Based on the type of matrix, returns a list containing 1)mat_value, a matrix containing the 
#' starting values of the matrix parameters and 2)mat_free, a matrix of 1s and 0s which 
#' indicate which entries of the matrix are free parameters.  
#'
#' Example: lnm_constraint_matrices(type='omega_psi',n_rows=3,n_cols = 3, entries=list(c(2),c(1),c(0.8)),identification = 'variance')
#' 
#' Output: $mat_value
#'         0.0    0    0
#'         0.8    0    0
#'         0.0    0    0
#'         $mat_free
#'         0    1    1
#'         0    0    1
#'         1    1    0
#' 
#' Explanation: The selected matrix is the 'omega_psi' matrix with 3 rows and 3 columns.
#' The (2,1) entry is set to be 0.8 in mat_value. It is no longer free and so (2,1) entry
#' in mat_free will be a value of 0. Note that for omega_psi and theta, symmetric matrices,
#' if (i,j) entry is set to be fixed, then (j,i) entry will also be set to be fixed in the model, 
#' but the output of this function will not reflect that.
#'
#'
#' @param type string to indicate whether selected matrix is a "lambda", "omega_psi" or "theta" matrix
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
lnm_constraint_matrices <- function(type = "lambda",lambda = NULL,n_rows, n_cols, entries = NULL,
                                    identification='variance') {
  
  if (identification == "variance"){
    
    if (type == "lambda"){
      mat_value <- matrix(0, nrow = n_rows, ncol = n_cols)
      mat_value <- mat_value + 0.7*lambda
      mat_free <- lambda
    }
    else if (type == "omega_psi") {
      mat_value <- matrix(0, nrow = n_rows, ncol = n_cols)
      mat_free <- matrix(1, nrow = n_rows, ncol = n_cols)
      diag(mat_free) <- 0
    }
    else if (type == "delta_psi"){
      mat_value <- diag(rep(1, n_rows))
      mat_free <- matrix(0, nrow = n_rows, ncol = n_cols)
    }
    else {
      mat_value <- diag(1, n_rows)
      mat_free <- matrix(0, nrow = n_rows, ncol = n_cols)
      diag(mat_free) <- 1
    }
  }
  else if (identification == "loadings"){
    if (type == "lambda"){
      helper_mat_1 <- format_lambda_mat(lambda,type='value')
      helper_mat_2 <- format_lambda_mat(lambda,type = 'free')
      mat_value <- helper_mat_1 
      mat_free <- helper_mat_2 
    }
    else if (type == "omega_psi") {
      mat_value <- matrix(0, nrow = n_rows, ncol = n_cols)
      mat_free <- matrix(1, nrow = n_rows, ncol = n_cols)
      diag(mat_free) <- 0
    }
    else if (type == "delta_psi"){
      mat_value <- diag(rep(1, n_rows))
      mat_free <- diag(rep(1, n_rows))
    }
    else {
      mat_value <- diag(1, n_rows)
      mat_free <- matrix(0, nrow = n_rows, ncol = n_cols)
      diag(mat_free) <- 1
    }
    
  }
  
  if (!is.null(entries)){
    for (i in 1:length(entries[[1]])) {
      row <- entries[[1]][i]
      col <- entries[[2]][i]
      value <- entries[[3]][i]
      mat_value[row, col] <- value
      mat_free[row, col] <- 0
      
    }
  }
  return(list('mat_value' = mat_value,'mat_free' = mat_free))
  
  
}






#' Returns a list containing necessary information for the initialization of the torch lnm
#' model. The list contains: 1) params_start -  a vector of starting values for the parameters, 
#' 2) params_value - a vector of default values for non-free parameters and 3) params_free - a 
#' vector of 1s and 0s which indicate which parameters are free. Also included are: 1)
#' params_sizes - a vector which indicate the number of parameters in the lambda, theta, delta_psi and
#' omega_psi matrices respectively, 2) params_free_sizes - a vector which indicate the number of 
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
#' lnm_mod_to_torch_opts(lambda)
#' Output: list containing the aforementioned vectors
#' 
#' Explanation: The design matrix for the factor loadings, lambda is provided to the function 
#' while setting no constraints for the lambda, omega_psi and theta matrices. The aforementioned vectors
#' returned will reflect these conditions set.
#'
#'
#' @param lambda Design matrix for the factor loadings, the lambda matric
#' @param lambda_constraint_lst (optional)  list containing 3 vectors - the row, column indices and values of the fixed parameters 
#' @param theta_constraint_lst (optional)  list containing 3 vectors - the row, column indices and values of the fixed parameters
#' @param delta_psi_constraint_lst (optional)  list containing 3 vectors - the row, column indices and values of the fixed parameters
#' @param omega_psi_constraint_lst (optional) list containing 3 vectors - the row, column indices and values of the fixed parameters
#' @param identification (optional) method to identify the model, either by variance or by loadings (default "variance")
#' 
#' 
#' @return list containing params_start, params_free, params_value, params_sizes, params_free_sizes,
#' params_free_sizes_max, params_free_starting_pts, lambda_matrix_Free
#'
lnm_mod_to_torch_opts <- function(lambda,
                                  lambda_constraint_lst = NULL,
                                  theta_constraint_lst = NULL,
                                  delta_psi_constraint_lst = NULL,
                                  omega_psi_constraint_lst = NULL,
                                  identification = "variance"){
  
  n <- dim(lambda)[1]
  m <- dim(lambda)[2]
  
  lambda_constraint_matrices <- lnm_constraint_matrices('lambda',lambda=lambda,n,m,entries=lambda_constraint_lst,identification = identification)
  omega_psi_constraint_matrices <- lnm_constraint_matrices('omega_psi',lambda=NULL,m,m,entries=omega_psi_constraint_lst,identification = identification)
  delta_psi_constraint_matrices <- lnm_constraint_matrices('delta_psi',lambda=NULL,m,m,entries=delta_psi_constraint_lst,identification = identification)
  theta_constraint_matrices <- lnm_constraint_matrices('theta',lambda=NULL,n,n,entries=theta_constraint_lst,identification = identification)
  
  lambda_start <- lavaan::lav_matrix_vec(lambda_constraint_matrices[[1]])
  lambda_matrix_free <- lambda_constraint_matrices[[2]]
  lambda_free <- lavaan::lav_matrix_vec(lambda_matrix_free )
  lambda_free_idx <- which(lambda_free!= 0)
  lambda_free_size <- length(lambda_free_idx)
  
  theta_start <- lavaan::lav_matrix_vech(theta_constraint_matrices[[1]])
  theta_free <- lavaan::lav_matrix_vech(theta_constraint_matrices[[2]])
  theta_free_idx <- which(theta_free!= 0)
  theta_free_size <- length(theta_free_idx)
  
  delta_psi_start <- lavaan::lav_matrix_vech(delta_psi_constraint_matrices[[1]])
  delta_psi_free <- lavaan::lav_matrix_vech(delta_psi_constraint_matrices[[2]])
  delta_psi_free_idx <- which(delta_psi_free!= 0)
  delta_psi_free_size <- length(delta_psi_free_idx)
  
  omega_psi_start <- lavaan::lav_matrix_vech(omega_psi_constraint_matrices[[1]])
  omega_psi_free <- lavaan::lav_matrix_vech(omega_psi_constraint_matrices[[2]])
  omega_psi_free_idx <- which(omega_psi_free!= 0)
  omega_psi_free_size <- length(omega_psi_free_idx)
  
  params_sizes <- sapply(list(lambda_start,theta_start, delta_psi_start ,omega_psi_start), length)
  
  torch_list <- list(
    params_start = c(lambda_start,theta_start,delta_psi_start ,omega_psi_start),
    params_free = c(lambda_free,theta_free,delta_psi_free ,omega_psi_free),
    params_value = c(lambda_start,theta_start,delta_psi_start,omega_psi_start),
    params_sizes =   params_sizes ,
    params_free_sizes = c(lambda_free_size,theta_free_size,delta_psi_free_size,omega_psi_free_size),
    params_free_sizes_max = c(length(which(lambda!= 0)),n,m,m*(m+1)/2 - m),
    params_free_starting_pts = c(1,1+params_sizes[1],1+sum(params_sizes[1:2]),1+sum(params_sizes[1:3])),
    lambda_matrix_free = lambda_matrix_free )
  
  return(torch_list)
}





#' Torch lnm module created during stepping down/stepping up/pruning 
#'
#' Function for creating an lnm model
#'
#' @param params_vec param values for the initializing of the module
#'
#' @return A `torch_sem` object, which is an `nn_module` (torch object)
#'
#' @details
#' This function instantiates a torch object for computing the model-implied covariance matrix
#' based on an lnm model during the stepping down/stepping up/pruning processes. The methods are identical 
#' to torch_lnm and information about these methods can be looked up under torch_lnm
#' 
#' @import torch
#' @import lavaan
#' @importFrom R6 R6Class
#'
tensor_lnm_stepwise <- torch::nn_module(
  classname = "torch_network_model",
  initialize = function(params_vec){
    self$params_vec<-nn_parameter(params_vec)
  },
  #' @section Methods:
  #'
  #' ## `$forward()`
  #' Compute the model-implied covariance matrix.
  #' Don't use this; `nn_modules` are callable, so access this method by calling
  #' the object itself as a function, e.g., `my_torch_lnm()`.
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
    self$theta <- torch_index_select(self$params_split[[2]], 1, self$theta_dup_idx)$view(c(self$n,self$n))
    self$delta_psi <- torch_index_select(self$params_split[[3]], 1, self$delta_psi_dup_idx)$view(c(self$m,self$m)) 
    self$omega_psi <- torch_index_select(self$params_split[[4]], 1, self$omega_psi_dup_idx)$view(c(self$m,self$m)) 
    #Compute the model-implied covariance matrix 
    self$omega <- (self$I_mat - self$omega_psi)
    self$omega_inv <- torch_inverse(self$omega)
    if (compute_sigma == FALSE) return(invisible(self))
    self$sigma <-self$lambda$mm(self$delta_psi$mm(self$omega_inv$mm(self$delta_psi$mm(self$lambda$t())))) + self$theta
    return(self$sigma)
  }
  ,
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
    return(-2*self$loglik(data) + v*sum(abs(self$omega_psi$t()$reshape(-1)[strict_lower_triangle_idx(self$m)])))
  }
  ,
  
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
  #' - `batch_size` (Optional) change the number of samples used for training 
  #' to reduce computational time. Default is NULL
  #'
  #' ### Value
  #' Self, i.e., the `torch_lnm` object with updated parameters 
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
    
    lasso_update_params_removed(v=v, epsilon=epsilon)
    
    if (epoch == maxit) warning("maximum iterations reached") 
    
    return(invisible(self))
  },
  
  #' @section Methods:
  #'
  #' ## `$fit()`
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
  #' Self, i.e., the `torch_lnm` object with updated parameters  
  
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
        loss <- self$custom_loss(self$forward(), batch_data)  # Compute loss on batch
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
  ,
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
    self$latent_pairings <- vector(mode='character',length = factorial(self$m)/factorial(2))
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
  #' ## `$get_residuals()`
  #' Get all the free residuals (diagonal entries of theta matrix) determined after model fit. 
  #' This includes fit, custom_fit and lasso_fit. 
  #'
  #' ### Value
  #' Dataframe of residuals and if the default log-likelihood fn is used in to fit,
  #' standard errors and p-values are also provided
  get_residuals=function(){
    if (is.null(self$sigma)) warning('Data must be fitted first')
    ind_idx <- which(as_array(self$params_free$reshape(-1))[(self$params_sizes[1] + 1):sum(self$params_sizes[1:2])] == TRUE)
    names_ind_idx <- find_correct_indices(diagonal_idx(self$n),ind_idx)
    residual_names <- self$vars[names_ind_idx]
    counter <- 1
    H_ind1 <- (self$params_free_sizes[1]+1)
    H_ind2  <- sum(self$params_free_sizes[1:2])
    col1<- lavaan::lav_matrix_vech(as.matrix(self$theta))[ind_idx]
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
    self$residuals <- data.frame(`Residuals` = round(col1,3),`Standard error`= round(col2,3),
                                 `P value`= round(col3,4),`Significant`=col4)
    rownames(self$residuals) <- residual_names
    return(self$residuals)
  },
  #' @section Methods:
  #'
  #' ## `$lasso_update_params_removed(v,epsilon)`
  #' Update the model attribute, self$lasso_num_params_removed. This is will be done automatically if
  #' after `$lasso_fit()`.
  #'
  #' ### Arguments
  #' - `v` hyperparameter which controls for the penalty term inside the lasso loss function.
  #' - `epsilon`  Cutoff for lasso to set parameter to 0. 
  #' 
  #'
  #' ### Value
  #' None 
  lasso_update_params_removed = function(v,epsilon){
    if (is.null(self$sigma)) warning('Data must be fitted first')
    if (self$lasso == FALSE) warning('Set lasso = TRUE first!')
    lower_triangle_idx <- lower_triangle_idx(self$m)
    partial_correlations <- as_array(self$omega_psi$t()$reshape(-1)[lower_triangle_idx]$reshape(-1))
    ind_idx <- which(as_array(self$params_free[self$params_free_starting_pts[4]:(self$params_free_starting_pts[4]+length(lower_triangle_idx)-1)]$reshape(-1))==1)
    partial_correlations <- partial_correlations[ind_idx]
    ind <- which(abs(partial_correlations) > epsilon)
    self$lasso_num_params_removed <- (self$m*(self$m +1)/2 - self$m) - length(ind) 
  },
  #' @section Methods:
  #'
  #' ## `$get_partial_correlations(epsilon)`
  #' Get all the free partial correlations (entries of lower triangle of omega_psi matrix) 
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
  get_partial_correlations = function(epsilon = 0.0001){
    if (is.null(self$sigma)) warning('Data must be fitted first')
    lower_triangle_idx <- lower_triangle_idx(self$m)
    partial_correlations <- as_array(self$omega_psi$t()$reshape(-1)[lower_triangle_idx]$reshape(-1))
    latent_pairings <- self$get_all_latent_pairings()
    ind_idx <- which(as_array(self$params_free[self$params_free_starting_pts[4]:(self$params_free_starting_pts[4]+length(lower_triangle_idx)-1)]$reshape(-1))==1)
    partial_correlations <- partial_correlations[ind_idx]
    latent_pairings <- latent_pairings[ind_idx]
    
    
    if (self$lasso == TRUE || !is.null(self$custom_loss)){
      if(self$lasso == TRUE){
        ind <- which(abs(partial_correlations) > epsilon)
        to_be_removed_idx <- which(abs(partial_correlations) <= epsilon)
        to_be_removed_indicator <- ifelse(abs(partial_correlations) <= epsilon,1,0)
        self$removed_partial_correlations <- latent_pairings[to_be_removed_idx]
        latent_pairings_remained <- latent_pairings[ind]
        col1 <- partial_correlations[ind]
        self$partial_corr <- data.frame(`Partial Corr` = round(col1,3))
        rownames(self$partial_corr) <- latent_pairings_remained 
        self$lasso_num_params_removed <- (self$m*(self$m +1)/2 - self$m) - length(ind) 
        
        return(list(self$partial_corr,self$lasso_num_params_removed,self$removed_partial_correlations,
                    extract_non_zero_entries(vech_to_symmetric_zero_diag(to_be_removed_indicator,self$m))))
      } else{
        col1 <- partial_correlations
        self$partial_corr <- data.frame(`Partial Corr` = round(col1,3))
        rownames(self$partial_corr) <- latent_pairings
        return(self$partial_corr)
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
    rownames(self$partial_corr) <- latent_pairings
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
    if (is.null(self$sigma)) warning('Data must be fitted first')
    if (criterion == "AIC") return(2*(sum(self$params_free)-self$lasso_num_params_removed) - 2*self$loglik())
    else if (criterion == "BIC") return((sum(self$params_free)-self$lasso_num_params_removed)*log(self$num_obs) - 2*self$loglik())
    else if (criterion == "EBIC"){
      num_params_free <- sum(self$params_free) -self$lasso_num_params_removed
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
    if (is.null(self$sigma)) warning('Data must be fitted first')
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


#' Given a torch_lnm module, mod, with params_vec, this copies all the attributes of mod 
#' into a new torch_lnm_stepwise module (except for params_free and params_free_sizes) which 
#' is instantiated during  stepping up/stepping down/pruning.
#' 
#'
#' @param torch_lnm torch_lnm module which is undergoing s stepping up/stepping down/pruning
#' @param params_vec params_vec which belongs to the torch_lnm module. If name of module is mod,
#' "mod$params_vec" should be input here.
#' 
#' @return torch_lnm_stepwise module with the exact same attributes as the original torch_lnm module
#' 
copy_lnm_attributes <- function(mod,params_vec){
  output_model <- tensor_lnm_stepwise(params_vec)
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
  output_model$lasso_num_params_removed  <- mod$lasso_num_params_removed
  output_model$theta_dup_idx <- mod$theta_dup_idx
  output_model$delta_psi_dup_idx <- mod$delta_psi_dup_idx
  output_model$omega_psi_dup_idx <- mod$omega_psi_dup_idx
  output_model$I_mat <- mod$I_mat
  output_model$mu <- mod$mu
  
  return(output_model)
}



#' Helper function for lnm_stepdown. Given an original torch_lnm/torch_lnm_stepwise module, the function fits all possible models
#' during one iteration of the stepping down process and returns either the best model based on criterion score
#' after this fitting process or the original torch_lnm/torch_lnm_stepwise module if all new models give a
#' worse model fit based on criterion score. A possible model here is a model with one of the
#' originally non-zero partial correlations now being set to zero.
#' 
#'
#' @param mod original torch_lnm/torch_lnm_stepwise module undergoing stepping down
#' @param criterion string indicating criterion to use - "AIC", "BIC or "EBIC"
#' @param gamma (Optional) value of gamma if "EBIC" is used. Default value is 0.5.
#' 
#' 
#' @return either the original torch_lnm/torch_lnm_stepwise module or 
#' a torch_lnm_stepwise module with a better criterion score
#' 
lnm_stepdown_find_alt_models <- function(mod,criterion = 'BIC',gamma = 0.5,batch_size = NULL){
  omega_psi_ind_1 <- mod$params_free_starting_pts[4]
  omega_psi_ind_2 <- omega_psi_ind_1 + mod$params_sizes[4]
  omega_psi_free <- mod$params_free[omega_psi_ind_1:omega_psi_ind_2]
  sum_omega_psi_free = torch_sum(omega_psi_free)$item()
  if ((sum_omega_psi_free - 1) == 0) {warning('Cant stepdown anymore! Left with one partial correlation!')}
  other_params_free <- mod$params_free[1:(omega_psi_ind_1-1)]
  other_params_vec <- mod$params_vec[1:(omega_psi_ind_1-1)]
  other_models_free_params <- flip_one_1_to_0(omega_psi_free)
  num_omega_psi_params <- length(omega_psi_free)
  num_possible_models <- length(other_models_free_params)
  list_of_criterion_values <- vector("numeric", length = num_possible_models)
  models <- vector("list", length = num_possible_models)
  current_criterion_value <- mod$get_criterion_value(criterion,gamma)$item()
  for(i in 1:num_possible_models){
    new_params_vec <- torch_cat(list(mod$params_vec[1:(omega_psi_ind_1-1)],torch_zeros(num_omega_psi_params)),dim = 1)
    clone <- copy_lnm_attributes(mod,new_params_vec$detach())
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

#' Step Down function for torch_lnm/torch_lnm_stepwise module
#' 
#' Performs stepping down to give the best model based on the selected criterion score
#' At the end of each iteration, one of the partial correlations will be removed and this
#' corresponds to the model with the best model fit based on criterion score 
#' out of all possibilities (number of non-zero partial correlations at the start). 
#' If the models fit at the end of 1 iteration with 1 partial correlation removed each do not lead to
#' a better criterion score, the model found at the end of the previous iteration is returned.
#' 
#' The process terminates if the model is left with only 1 partial correlation at the end of 
#' all the iterations.
#'
#' @param mod original torch_lnm/torch_lnm_stepwise module undergoing stepping down
#' @param criterion string indicating criterion to use - "AIC", "BIC or "EBIC"
#' @param gamma (Optional) value of gamma if "EBIC" is used. Default value is 0.5.
#' 
#' 
#' @return either the original torch_lnm/torch_lnm_stepwise module or 
#' a torch_lnm_stepwise module with a best criterion score after stepping down
#' 
#' 
lnm_stepdown <- function(mod,criterion = 'BIC',gamma = 0.5,batch_size = NULL){
  
  
  current_value <- mod$get_criterion_value(criterion,gamma)$item()
  stepdown_model <- lnm_stepdown_find_alt_models(mod,criterion,gamma,batch_size = batch_size)
  
  if(stepdown_model$get_criterion_value(criterion,gamma)$item() == current_value){
    return(mod)
  }else{
    lnm_stepdown(stepdown_model,criterion,gamma,batch_size = batch_size)
    
  }
}

#' Helper function for lnm_stepup. Given an original torch_lnm/torch_lnm_stepwise module, the function fits all possible models
#' during one iteration of the stepping up process and returns either the best model based on criterion score
#' after this fitting process or the original torch_lnm/torch_lnm_stepwise module if all new models give a
#' worse model fit based on criterion score. A possible model here is a model with one of the
#' originally zero partial correlations now being set to non-zero.
#' 
#'
#' @param mod original torch_lnm/torch_lnm_stepwise module undergoing stepping up
#' @param criterion string indicating criterion to use - "AIC", "BIC or "EBIC"
#' @param gamma (Optional) value of gamma if "EBIC" is used. Default value is 0.5.
#' 
#' 
#' @return either the original torch_lnm/torch_lnm_stepwise module or 
#' a torch_lnm_stepwise module with a better criterion score
#' 
lnm_stepup_find_alt_models<- function(mod,criterion = 'BIC',gamma = 0.5,batch_size = NULL){
  omega_psi_ind_1 <- mod$params_free_starting_pts[4]
  omega_psi_ind_2 <- omega_psi_ind_1 + mod$params_sizes[4]
  omega_psi_free <- mod$params_free[omega_psi_ind_1:omega_psi_ind_2]
  sum_omega_psi_free = torch_sum(omega_psi_free)$item()
  if ((mod$get_df() - sum_omega_psi_free - 1) < 0) {
  print('Model is about to be underidentified, returning the model before 
                                                                   underidentification') 
    return(mod)}
  if (sum_omega_psi_free == mod$params_free_sizes_max[4]) {warning('Cant stepup anymore! Saturated!')}
  other_params_free <- mod$params_free[1:(omega_psi_ind_1-1)]
  other_params_vec <- mod$params_vec[1:(omega_psi_ind_1-1)]
  other_models_free_params <- flip_one_0_to_1(omega_psi_free)
  num_omega_psi_params <- length(omega_psi_free)
  num_possible_models <- length(other_models_free_params)
  list_of_criterion_values <- vector("numeric", length = num_possible_models)
  models <- vector("list", length = num_possible_models)
  current_criterion_value <- mod$get_criterion_value(criterion,gamma)$item()
  
  
  for(i in 1:num_possible_models){
    new_params_vec <- torch_cat(list(mod$params_vec[1:(omega_psi_ind_1-1)],torch_zeros(num_omega_psi_params)),dim = 1)
    clone <- copy_lnm_attributes(mod,new_params_vec$detach())
    clone$params_free <- torch_cat(list(other_params_free,other_models_free_params[[i]]),dim=1)
    clone$params_free_sizes <- c(mod$params_free_sizes[1:3], mod$params_free_sizes[4] + 1) 
    clone$fit(verbose=F,batch_size = batch_size)
    models[[i]] <- clone
    list_of_criterion_values[i] <- clone$get_criterion_value(criterion,gamma)$item()
  }
  
  min_index <- which.min(list_of_criterion_values)
  
  if (list_of_criterion_values[min_index] < current_criterion_value) return(models[[min_index]])
  
  else {return(mod)}
}


#' Step Up function for torch_lnm/torch_lnm_stepwise module
#' 
#' Performs stepping up to give the best model based on the selected criterion score
#' At the end of each iteration, one of the originally zero partial correlations will be
#' turned to non-zero and this corresponds to the model with the best model fit based on criterion score
#' out of all possibilities (number of zero partial correlations at the start). 
#' If the models fit at the end of 1 iteration with 1 partial correlation set to non-zero each 
#' do not lead to a better criterion score, the model found at the end of the previous iteration is returned.
#' 
#' The process terminates if the model is saturated (all partial correlations are non-zero) at the end of the iteration.
#'
#' @param mod original torch_lnm/torch_lnm_stepwise module undergoing stepping up
#' @param criterion string indicating criterion to use - "AIC", "BIC or "EBIC"
#' @param gamma (Optional) value of gamma if "EBIC" is used. Default value is 0.5.
#' 
#' 
#' @return either the original torch_lnm/torch_lnm_stepwise module or 
#' a torch_lnm_stepwise module with a best criterion score after stepping down
#' 
lnm_stepup<- function(mod,criterion = 'BIC',gamma = 0.5,batch_size = NULL){
  current_value <- mod$get_criterion_value(criterion,gamma)$item()
  stepup_model <- lnm_stepup_find_alt_models(mod,criterion,gamma,batch_size = batch_size)
  
  if(stepup_model$get_criterion_value(criterion,gamma)$item() == current_value){
    return(mod)
  }else{
    lnm_stepup(stepup_model,criterion,batch_size = batch_size)
    
  }
}

#' Helper function for lnm_prune. Given an lnm model, it removes all partial correlations
#' found to be insignificant from the model and refits the model without these partial correlations.
#' Thereafter, it returns the updated model.
#' 
#'
#' @param mod original torch_lnm/torch_lnm_stepwise module 
#' 
#' 
#' @return new refitted model with insignificant partial correlations removed
#' 
lnm_prune_helper <- function(mod,batch_size = NULL){
  
  index <- which(mod$partial_corr$Significant == "")
  num_params_removed <- length(index)  
  if( num_params_removed == 0){return(mod)}
  omega_psi_ind_1 <- mod$params_free_starting_pts[4]
  omega_psi_ind_2 <- omega_psi_ind_1 + mod$params_sizes[4]
  omega_psi_free <- mod$params_free$clone()[omega_psi_ind_1:omega_psi_ind_2]
  sum_omega_psi_free = torch_sum(omega_psi_free)$item()
  other_params_free <- mod$params_free[1:(omega_psi_ind_1-1)]
  other_params_vec <- mod$params_vec[1:(omega_psi_ind_1-1)]
  one_positions <- torch_nonzero(omega_psi_free)$squeeze(2)  
  omega_psi_free[one_positions[index]] <- 0 
  new_free_params <- omega_psi_free
  num_omega_psi_params <- length(omega_psi_free)
  new_params_vec <- torch_cat(list(mod$params_vec[1:(omega_psi_ind_1-1)],torch_zeros(num_omega_psi_params)),dim = 1)
  clone <- copy_lnm_attributes(mod,new_params_vec$detach())
  clone$params_free <- torch_cat(list(other_params_free,new_free_params),dim=1)
  clone$params_free_sizes <- c(mod$params_free_sizes[1:3], mod$params_free_sizes[4] - num_params_removed) 
  clone$fit(verbose=F,batch_size = batch_size)
  return(clone) 
  
}

#' Pruning fortorch_lnm/torch_lnm_stepwise module
#' 
#' Given an  model, it iteratively removes insignificant partial correlations and refits models until
#' all partial correlations are found to be significant. Thereafter ,it returns this model
#' with only significant partial correlations.
#' 
#'
#' @param mod original torch_lnm/torch_lnm_stepwise module 
#' 
#' 
#' @return new refitted model with only significant partial correlations 
#' 
lnm_prune <- function(mod, batch_size = NULL){
  pruned_model <-  lnm_prune_helper(mod,batch_size = batch_size)
  
  if(length(which(pruned_model$partial_corr$Significant == "")) == 0){
    return(pruned_model)
  }
  else{
    lnm_prune(pruned_model,batch_size = batch_size)
  }
}



#' LASSO for LNM
#' 
#' Performs a lasso search to find the value of hyperparameter v which correspond to the model 
#' which gives the lowest criterion scores (AIC/BIC/EBIC). At the end, the constraints
#' for omega_psi (parameters which have been set to zero) will also be returned. Each model 
#' is fit with a different value of v and partial correlations which are lower than the threshold value are set to 0.
#' These partial correlations are removed from the pool of free parameters before the criterion score is calculated.
#'  
#' @import pracma
#' 
#' @param mod original torch_lnm/torch_lnm_stepwise module undergoing stepping up
#' @param criterion (Optional) string indicating criterion to use - "AIC", "BIC or "EBIC". Default is BIC.
#' @param v_values (Optional) values of the hyperparameter v to be used in the search. Default range 
#' of values is 30 values of v from 0.01 to 100 spread out on a log scale.
#' @param lrate (Optional) learning rate for lasso. Default value is 0.01.
#' @param epsilon (Optional) threshold under which partial correlations will be set to 0. Default is 0.0001.
#' @param gamma (Optional) value of gamma if "EBIC" is used. Default value is 0.5.
#' 
#' 
#' @return a list containing the following: 1)value of v of the model which 
#' gives the best criterion score and 2)Constraints for omega_psi
#' 
lnm_lasso_explore <- function(mod, criterion = "BIC", v_values  = pracma::logspace(log10(0.01), log10(100), 30) ,lrate = 0.01,epsilon = 0.0001, gamma = 0.5,batch_size = NULL){
  if (mod$lasso == FALSE) {warning('Set lasso to TRUE first!')}
  criterion_values <- vector(mode = 'numeric',length=length(v_values))
  output_params <- list()
  for (i in 1:length(v_values)){
    mod$lasso_fit(v = v_values[i],lrate = lrate,epsilon = epsilon,batch_size=batch_size)
    mod$lasso_update_params_removed(epsilon = epsilon)
    criterion_values[i] <- mod$get_criterion_value(criterion,gamma)$item()
    output_params[[i]] <- mod$get_partial_correlations()[[4]]
  }
  ind <- which.min(criterion_values)
  return(list(penalty_parameter = v_values[ind],omega_psi_constraints = output_params[[ind]] ))
}


