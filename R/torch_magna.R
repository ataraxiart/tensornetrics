#' Random effects MAGNA with a Torch backend
#'
#' Function for creating a random effects MAGNA model
#'
#'
#' @param corMats list containing correlation matrices of the studies of the model
#' @param nobs vector containing the sample sizes of the studies of the model
#' @param Vmethod specification for treatment of V matrix, either "individual" or "pooled"
#' @param Vestimation specification for estimation of V matrix, either "per_study" or "averaged"
#' @param missing (optional) indicator for whether missing values are present in the correlation matrices
#' @param params_free_filter (optional) torch_tensor of booleans (1s/0s) indicating which omega parameters are free or constrained 
#' @param params_value_filter (optional) user-defined torch_tensor of values indicating the starting values of the omega parameters to be used for optimization 
#' @param dtype (optional) torch dtype for the model (default torch_float32())
#' @param device (optional) device type to put the model on. (default torch_device("cpu")) see [torch::torch_device()]
#' 
#' 
#' @return A `torch_random_effects_MAGNA` object, which is an `nn_module` (torch object)
#'
#' @details
#' This function instantiates a torch object for the estimation of omega parameters and tau parameters
#' in a random effects MAGNA model. Through `torch`, gradients of this forward model can then
#' be computed using backpropagation, and the parameters can be optimized using gradient-based
#' optimization routines from the `torch` package.
#'
#'
#' @import torch
#' @import lavaan
#' @import stringr
#' @import dplyr
#' @importFrom R6 R6Class
#'
#' @name tensor_meta_ggm
#'
#'
#' @export
tensor_meta_ggm <- torch::nn_module(
  classname = "torch_random_effects_MAGNA",
  #' @section Methods:
  #'
  #' ## `$initialize(corMats,...)`
  #' 
  #' The initialize method. Don't use this, just use [tensor_lnm()]
  #'
  #' ### Arguments
  #' - `corMats` list containing correlation matrices of the studies of the model
  #' - `nobs` vector containing the sample sizes of the studies of the model
  #' - `Vmethod` specification for treatment of V matrix, either "individual" or "pooled"
  #' - `Vestimation` specification for estimation of V matrix, either "per_study" or "averaged"
  #' - `missing` (optional) indicator for whether missing values are present in the correlation matrices
  #' - `params_free_filter` (optional) torch_tensor of booleans (1s/0s) indicating which omega parameters are free or constrained 
  #' - `params_value_filter` (optional) user-defined torch_tensor of values indicating the starting values of the omega parameters to be used for optimization 
  #' - `dtype` (optional) torch dtype for the model (default torch_float32())
  #' - `device` (optional) device type to put the model on. (default torch_device("cpu")) see [torch::torch_device()]
  #' 
  #' ### Value
  #' A `torch_random_effects_MAGNA` object, which is an `nn_module` (torch object)
  initialize = function(corMats=NULL, nobs = NULL, Vmethod = NULL, Vestimation = NULL,
                        missing = FALSE,params_free_filter = NULL,
                        params_value_filter = NULL,
                        dtype = torch_float32(), device = torch_device("cpu")){
    if (is.null(corMats) || is.null(nobs)) {
      warning('Must input correlations and number of observations per study')
    }
    self$model_type <- "random_effects_magna"
    self$device <- device
    self$dtype <- dtype
    self$corMats <- get_corMats_as_tensors(corMats)
    self$params_free_filter = params_free_filter
    self$params_value_filter = params_value_filter
    # compute torch settings
    opts <-magna_convert_to_torch_opts(corMats,missing = FALSE,params_free_filter = NULL,
                                                  params_value_filter = NULL,
                                       dtype = self$dtype, device = self$device)
    # initialize the dimensions of the model:
    self$n <- get_network_dim(corMats)
    self$m <- length(corMats)
    self$nobs <- nobs
    # initialize the parameter vector
    self$params_vec <- nn_parameter(opts[[1]])
    # initialize the parameter constraints
    self$params_free <- opts[[2]]
    self$params_value <- opts[[3]]
    # other attributes
    if (Vmethod == "individual"){self$individual = TRUE}
    else {self$individual = FALSE}
    if (Vestimation == "per_study"){self$per_study = TRUE}
    else{self$per_study = FALSE}
    self$omega_params_length <- opts[[4]]
    self$tau_params_length <- opts[[5]]
    self$V_matrix <- self$get_V_i_or_V_asterisk()
  },
  
  #' @section Methods:
  #'
  #' ## `$get_V_i_or_V_asterisk()`
  #' 
  #' Compute either V_i or V_asterisk used in random effects MAGNA
  #' 
  #' ### Value
  #' Either a torch tensor (V matrix) or a list of torch tensors (V matrices)
  get_V_i_or_V_asterisk = function(){
    individual_estimates <- vector("list", length = self$m)
    
    get_block_diag <- function(tensor_list,nobs) {
      n_blocks <- length(tensor_list)
      block_shape <- tensor_list[[1]]$size()
      b_rows <- block_shape[1]
      b_cols <- block_shape[2]
      total_rows <- n_blocks * b_rows
      total_cols <- n_blocks * b_cols
      result <- torch_zeros(c(total_rows, total_cols))
      factors <- nobs/sum(nobs)
      for (i in 1:n_blocks) {
        for (j in 1:b_rows) {
          for (k in 1:b_cols){
            row_idx <- ((i - 1) * b_rows + j) 
            col_idx <- ((i - 1) * b_cols + k)
            result[row_idx, col_idx] <- tensor_list[[i]][j,k]* factors[i]
          }
        }
      }
      
      return(result)
    }
    if (self$individual){
      counter <- 1
      for (mat in self$corMats){
        kappa <- mat$inverse()
        D_s <- strict_duplication_matrix(self$n)
        distribution_hessian <- D_s$t()$matmul(kron(kappa,kappa))$matmul(D_s)
        fisher_information <- 0.5*distribution_hessian 
        V <- fisher_information$inverse()/self$nobs[counter]
        individual_estimates[[counter]] <- V
        counter <- counter + 1
      }
      if (self$per_study) return(individual_estimates)
      else{
        factors <- nobs/sum(self$nobs)
        return(Reduce(`+`, lapply(seq_along(individual_estimates), function(i) {
          individual_estimates[[i]] * factors[i]
        })))
      }
    }
    else {
      counter <- 1
      for (mat in self$corMats){
        kappa <- mat$inverse()
        D_s <- strict_duplication_matrix(self$n)
        individual_distribution_hessian <- D_s$t()$matmul(kron(kappa,kappa))$matmul(D_s)
        individual_estimates[[counter]] <- individual_distribution_hessian
        counter <- counter + 1
      }
      distribution_hessian <- get_block_diag(individual_estimates,self$nobs)
      I_list <- replicate(self$m, torch_eye(self$omega_params_length), simplify = FALSE)
      I_vert <- torch_cat(I_list, dim = 1)
      fisher_information <- 0.5*I_vert$t()$matmul(distribution_hessian)$matmul(I_vert)
      V <- fisher_information$inverse()/sum(self$nobs)
      if (!self$per_study)
      {
        return(V)
      }
      else{
        factors <- mean(self$nobs)/self$nobs
        return(lapply(seq_along(factors), function(i) {
          V * factors[i]
        }))
      }
    }
  }
  
  #' @section Methods:
  #'
  #' ## `$get_loss()`
  #' Get the value of the fit function.  The fit function has 2 forms corresponding to when  
  #' a single V matrix is used and when multiple V matrices for each individual study are used
  #'
  #'
  #' ### Value
  #' Value of fit function (torch scalar)
  ,get_loss= function(){
    self$params <- torch_where(self$params_free, self$params_vec, self$params_value)
    self$params_split <- torch_split(self$params, c(self$omega_params_length,self$tau_params_length))
    self$omega_params <- self$params_split[[1]]
    self$tau_params <- self$params_split[[2]]
    if (!self$per_study){
      mean_corr <- get_mean_corr(self$corMats,n=self$n,m=self$m)
      S_matrix <- get_sample_corr_of_corr(corMats=self$corMats,mean=mean_corr,omega_params_length = self$omega_params_length,n=self$n,m = self$m)
      self$tau_matrix <- self$tau_params[vech_dup_idx(self$omega_params_length)]$reshape(c(self$omega_params_length,
                                                                                      self$omega_params_length))
      sigma_matrix <- self$V_matrix + self$tau_matrix$matmul(self$tau_matrix$t())
      kappa_matrix <- sigma_matrix$inverse()
      self$omega_matrix <- transform_omega_params(self$omega_params,self$omega_params_length,self$n)[vech_dup_idx(self$n)]$reshape(c(self$n,self$n))
      self$omega_asterisk <- (torch_eye(self$n) - self$omega_matrix)$inverse()
      self$delta_matrix <- torch_diag(self$omega_asterisk$diag()$sqrt())
      mu_vector <- self$delta_matrix$matmul(self$omega_asterisk$matmul(self$delta_matrix))$reshape(-1)[strict_lower_triangle_idx(self$n)]
      trace_SK <- torch_trace(S_matrix$matmul(kappa_matrix)) 
      difference_vector<- (mean_corr - mu_vector)
      difference_squares <- (difference_vector)$matmul(kappa_matrix$matmul(difference_vector))
      ln_kappa <- torch_slogdet(kappa_matrix)
      loss <- trace_SK + difference_squares -  ln_kappa[[2]]
      return(loss)
      
    } else
    {
      loss <- 0
      self$omega_matrix <- transform_omega_params(self$omega_params,self$omega_params_length,self$n)[vech_dup_idx(self$n)]$reshape(c(self$n,self$n))
      self$omega_asterisk <- (torch_eye(self$n) - self$omega_matrix)$inverse()
      self$delta_matrix <- torch_diag(self$omega_asterisk$diag()$sqrt())
      mu_vector <- self$delta_matrix$matmul(self$omega_asterisk$matmul(self$delta_matrix))$reshape(-1)[strict_lower_triangle_idx(self$n)]
      self$tau_matrix <- self$tau_params[vech_dup_idx(self$omega_params_length)]$reshape(c(self$omega_params_length,self$omega_params_length))
      for (i in 1:length(self$corMats)){
        corr <- self$corMats[[i]]$reshape(-1)[strict_lower_triangle_idx(self$n)]
        current_sigma_matrix <- self$V_matrix[[i]] + self$tau_matrix$matmul(self$tau_matrix$t())
        current_kappa_matrix <- current_sigma_matrix$inverse()
        difference_vector<- (corr - mu_vector)
        current_difference_squares <- (difference_vector)$matmul(current_kappa_matrix$matmul(difference_vector))
        current_ln_kappa <- torch_slogdet(current_kappa_matrix)
        loss <- loss + current_difference_squares - current_ln_kappa[[2]]
      }
      return(loss)
    }
  }
  ,
  
  #' @section Methods:
  #'
  #' ## `$fit()`
  #' Fit a torch_random_effects_MAGNA model using the appropriate fit function.
  #' This function uses the Adam optimizer to estimate the omega and tau parameters.
  #' Note that very often, the inverse Hessian obtained after optimization is not positive definite
  #' and this is often due to premature convergence during numerical optimization. 
  #' To combat this, try running "fit()" again with a different learning rate 
  #' if necessary to obtain estimates closer to the argmin of the fit function.
  #'
  #' ### Arguments
  #' - `lrate` (Optional) learning rate of the Adam optimizer. Default is 0.05.
  #' - `maxit` (Optional) maximum number of epochs to train the model. Default is 5000.
  #' - `verbose` (Optional) whether to print progress to the console.  Default is TRUE.
  #' - `tol` (Optional) parameter change tolerance for stopping training. Default is 1e-20.
  #'
  #' ### Value
  #' Self, i.e., the `torch_random_effects_MAGNA` object with updated parameters 
  fit = function(lrate = 0.05, maxit = 5000, verbose = TRUE, tol = 1e-20) {
    if (verbose) cat("Performing Random Effects MAGNA estimation \n")
    optim <- optim_adam(params = self$params_vec, lr = lrate)
    scheduler <- lr_reduce_on_plateau(optim, factor = 0.5, patience = 5)
    prev_loss <- 0.0
    for (epoch in 1:maxit) {
      optim$zero_grad()
      loss <- self$get_loss()
      loss$backward()
      optim$step()
      
      current_loss <- loss$item() 
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
    
    print('Full Inverse Hessian might not be positive definite; if needed, run another round of the estimation procedure')
  
    return(invisible(self))
  }
  #' @section Methods:
  #'
  #' ## `$inverse_Hessian()`
  #' Compute and return the asymptotic covariance matrix of the parameters with
  #' respect to the appropriate fit function in random effects MAGNA
  #'
  #'
  #' ### Value
  #' A `torch_tensor`, representing the ACOV of the free parameters
  ,inverse_Hessian = function() {
    g <- autograd_grad(self$get_loss(), self$params_vec, create_graph = TRUE)[[1]]
    H <- torch_jacobian(g, self$params_vec)
    free_idx <- torch_nonzero(self$params_free)$view(-1)
    self$Hinv <- torch_inverse(H[free_idx, ][, free_idx])
    return(self$Hinv)
  },
  
  
  #' @section Methods:
  #'
  #' ## `$get_all_pairings()`
  #' Get all the possible combinations of pairs of a specified set of variables. 
  #' For example, if there are 3 variables varaibles = (a,b,c), we have a total of 3 choose 2, or 3
  #' pairs possible and the function get_all_pairings(variables) will return c('a~b', 'a~c', 'b~c').
  #'
  #' ### Arguments
  #' - `variables` (Optional) Set of variables to get the pairings from
  #'
  #' ### Value
  #' vector of all possible combinations of pairs  
  get_all_pairings=function(variables){
    pairings <- vector(mode='character',length = factorial(self$n)/factorial(2))
    counter <- 1
    for (i in 1:(self$n)){
      for (j in (i):self$n){
        pairings[counter] <- paste0(stringr::str_sub(variables[i],1,3),'~',stringr::str_sub(variables[j],1,3))
        counter <- counter + 1
      }
    }
    pairings <- pairings[which(pairings != "")]
    return(pairings)
  },
  
  #' @section Methods:
  #'
  #' ## `$get_summary(epsilon)`
  #' Get all the free omega parameters and if needed, tau parameters 
  #' determined after model fit. Standard errors and p-values are also provided.
  #' 
  #' ### Arguments
  #' - `omega_params_only` (Optional) Indicate only whether to return omega parameters,
  #' by default only omega parameters will be returned
  #'
  #' ### Value
  #' Dataframe of parameters, standard errors, p-values , and asterisks to indicate significant at
  #' alpha = 0.05
  get_summary = function(omega_params_only = TRUE){
    self$rownames <- c(self$get_all_pairings(sapply(1:self$omega_params_length,function(i) paste0("o",i)))[-diagonal_idx(self$n)])
    self$rownames <- self$rownames[as_array(self$params_free[1:self$omega_params_length]$nonzero())]
    col1 <- as_array(self$params_vec)[as_array(self$params_free[1:self$omega_params_length]$nonzero())]
    col2 <- as_array(self$inverse_Hessian()$diag()$sqrt())[as_array(self$params_free[1:self$omega_params_length]$nonzero())]
    col3 <- sapply(col1/col2, get_p_value)
    col4<-ifelse(col3<0.05, "*","")
    self$summary <- data.frame(`Parameters` = round(col1,4),`Standard error`= round(col2,4),
                              `P value`= round(col3,4),`Significant`=col4)
    rownames(self$summary) <- self$rownames
    print(self$summary)
    
    
    if(!omega_params_only){
    print("Full Inverse Hessian might not be positive definite!")
    self$rownames <- c(self$get_all_pairings(sapply(1:self$omega_params_length,function(i) paste0("o",i)))[-diagonal_idx(self$n)],
                       self$get_all_pairings(sapply(1:self$omega_params_length,function(i) paste0("t",i))))
    self$rownames <- self$rownames[as_array(self$params_free$nonzero())]
    col1 <- as_array(self$params_vec)[as_array(self$params_free$nonzero())]
    self$inverse_Hessian <- self$inverse_Hessian()
    col2 <- as_array(self$inverse_Hessian$diag()$sqrt())[as_array(self$params_free$nonzero())]
    col3 <- sapply(col1/col2, get_p_value)
    col4<-ifelse(col3<0.05, "*","")
    self$summary <- data.frame(`Parameters` = round(col1,4),`Standard error`= round(col2,4),
                              `P value`= round(col3,4),`Significant`=col4)
    rownames(self$summary) <- self$rownames
    print(self$summary)
    }
  }
  )


 







