#' Residual Network Model with a Torch backend
#'
#' Function for creating a Residual Network Model
#'
#' @param data data for the SEM model
#' @param lambda design matrix for factor loadings, lambda matrix
#' @param vars vector containing names of observed variables
#' @param latents vector containing names of latent variables
#' @param lambda_constraint_lst list containing the row, column and value vectors of the constraints of the lambda matrix
#' @param psi_constraint_lst list containing the row, column and value vectors of the constraints of the psi matrix
#' @param delta_theta_constraint_lst list containing the row, column and value vectors of the constraints of the delta_theta matrix
#' @param omega_theta_constraint_lst list containing the row, column and value vectors of the constraints of the omega_theta matrix
#' @param omega_theta_free_lst list containing the row, column and value vectors of the free parameters of the omega_theta matrix
#' @param lasso (optional) boolean to specify if model wants to use LASSO
#' @param custom_loss (optional) function supplied to allow the model to be fitted to a custom loss function
#' @param dtype (optional) torch dtype for the model (default torch_float32())
#' @param device (optional) device type to put the model on. see [torch::torch_device()]
#' @param identification (optional)  method to identify the model, either by variance or by loadings (default "variance")
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
#' @import stringr
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
  #' - `delta_theta_constraint_lst` list containing the row, column and value vectors of the constraints of the delta_theta matrix
  #' - `omega_theta_constraint_lst` list containing the row, column and value vectors of the constraints of the omega_theta matrix
  #' - `omega_theta_free_lst` list containing the row, column and value vectors of the free parameters of the omega_theta matrix
  #' - `lasso` (optional) boolean to specify if model wants to use LASSO
  #' - `custom_loss` (optional) function supplied to allow the model to be fitted to a custom loss function
  #' - `dtype` (optional) torch dtype for the model (default torch_float32())
  #' - `device` (optional) device type to put the model on. see [torch::torch_device()]
  #' - `identification` (optional)  method to identify the model, either by variance or by loadings (default "variance")
  #'
  #' ### Value
  #' A `torch_rnm` object, which is an `nn_module` (torch object)
  initialize = function(data,lambda,vars,latents,B_matrix = NULL,lambda_constraint_lst = NULL,
                        psi_constraint_lst = NULL,delta_theta_constraint_lst = NULL,omega_theta_constraint_lst = NULL,omega_theta_free_lst = NULL,
                        lasso=FALSE,custom_loss=NULL,dtype = torch_float32(), device = torch_device("cpu"),
                        identification = 'variance'){
    if (is.null(vars) || is.null(latents)) {
      warning('Must indicate names of observed and latent variables')
    }
    else{
      if (any(colSums(is.na(data[vars])) > 0)) warning('Data has missing values!')
    self$model_type <- "rnm"
    self$data <- df_to_tensor(data[,vars])} 
    self$vars <- vars
    self$latents <- latents
    self$identification <- identification
    self$custom_loss <- custom_loss
    self$lasso<-lasso
    self$device <- device
    self$dtype <- dtype
    # compute torch settings
    opt <-rnm_mod_to_torch_opts(lambda, lambda_constraint_lst,
                                psi_constraint_lst,delta_theta_constraint_lst,
                                omega_theta_constraint_lst,omega_theta_free_lst,
                                identification = identification)
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
    self$lasso_num_params_added  <- 0
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
    
    #Change params_free if lasso is turned on
    if(self$lasso == TRUE){
   self$params_free[self$params_free_starting_pts[4]:length(self$params_free)] <- get_saturated_omega_theta_free(self$n)
    self$params_free_sizes[4] <- self$n*(self$n-1)/2
    }
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
        cat("\rEpoch:", epoch, " loss:", current_loss)
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
    if (qchisq(0.05,df = df_model,ncp = (chisq_model-df_model)) <= 0){self$rmsea_lower <- 0}
    else{
      self$rmsea_lower <- sqrt(((qchisq(0.05,df = df_model,ncp = (chisq_model-df_model)$item()) - df_model)/df_model/(self$num_obs-1)))
    }
    self$rmsea_upper <- max(sqrt(((qchisq(0.95,df = df_model,ncp = (chisq_model-df_model)$item()) - df_model)/df_model/(self$num_obs-1))),0,na.rm=TRUE)
    self$metrics <- data.frame("Metrics" = metric_names, "Values" = c(self$cfi, self$tli,self$rmsea,self$rmsea_lower,self$rmsea_upper))
    return(self$metrics)
  }
  
)














