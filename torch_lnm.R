

source(file="torch_lnm_helper_functions.R")


### Warning: This is Beta Code! There are potentially a lot of errors!



tensor_lnm <- torch::nn_module(
  classname = "torch_network_model",
  initialize = function(data,lambda,lambda_constraint_lst = NULL,
                        theta_constraint_lst = NULL,omega_psi_constraint_lst = NULL,
                        vars = NULL, latents = NULL, lasso=FALSE,custom_loss=NULL,dtype = torch_float32(), device = torch_device("cpu")){
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
    self$opt <-lnm_mod_to_torch_opts(lambda, lambda_constraint_lst,
                                     theta_constraint_lst,omega_psi_constraint_lst)
    # initialize the dimensions of the model:
    self$n <- dim(lambda)[1]
    self$m <- dim(lambda)[2]
    self$num_obs <- nrow(data)
    # initialize the parameter vector
    self$params_vec <- nn_parameter(torch_tensor(self$opt$params_start, dtype = self$dtype, requires_grad = TRUE, device = self$device))
    # initialize the parameter constraints
    self$params_free <- torch_tensor(self$opt$params_free, dtype = torch_bool(), requires_grad = FALSE, device = self$device)
    self$params_value <- torch_tensor(self$opt$params_value, dtype = self$dtype, requires_grad = FALSE, device = self$device)
    # other attributes
    self$cov_matrix <- torch_tensor(cov(data[,vars]))
    self$lambda_matrix_free <- self$opt$lambda_matrix_free
    self$params_sizes <- self$opt$params_sizes
    self$params_free_sizes <- self$opt$params_free_sizes
    self$params_free_sizes_max <- self$opt$params_free_sizes_max
    self$params_free_starting_pts <- self$opt$params_free_starting_pts
    self$lasso_num_params_removed  <- 0
    # duplication indices transforming vech to vec for theta and omega_psi
    self$theta_dup_idx <- torch_tensor(vech_dup_idx(self$n), dtype = torch_long(), requires_grad = FALSE, device = self$device)
    self$omega_psi_dup_idx <- torch_tensor(vech_dup_idx(self$m), dtype = torch_long(), requires_grad = FALSE, device = self$device)
    
    # tensor identity matrix
    self$I_mat <- torch_eye(self$m, dtype = self$dtype, requires_grad = FALSE, device = self$device)
    # mean is fixed to 0/centering
    self$mu <- torch_zeros(self$n, dtype = self$dtype, requires_grad = FALSE, device = self$device)
  },
  
  
  forward = function(compute_sigma = TRUE){
    #Apply constraints for non-free parameters
    self$params <- torch_where(self$params_free, self$params_vec, self$params_value)
    #Get separate tensors for lambda, theta, delta_psi, and omega_psi
    self$params_split <- torch_split(self$params, self$params_sizes)
    self$lambda <- self$params_split[[1]]$view(c(self$m, self$n))$t()
    self$theta <- torch_index_select(self$params_split[[2]], 1, self$theta_dup_idx)$view(c(self$n,self$n))
    self$omega_psi <- torch_index_select(self$params_split[[3]], 1, self$omega_psi_dup_idx)$view(c(self$m,self$m)) 
    #Compute the model-implied covariance matrix 
    self$omega <- (self$I_mat - self$omega_psi)
    self$omega_inv <- torch_inverse(self$omega)
    if (compute_sigma == FALSE) return(invisible(self))
    self$sigma <-self$lambda$mm(self$omega_inv$mm(self$lambda$t())) + self$theta
    return(self$sigma)
  }
  ,
  loglik = function() {
    px <- distr_multivariate_normal(loc = self$mu, covariance_matrix = self$forward())
    return(px$log_prob(self$data)$sum())
    
  },
  lasso_loss = function(v) {
    return(self$loglik() - v*sum(abs(self$omega_psi$t()$reshape(-1)[strict_lower_triangle_idx(self$m)])))
  }
  ,
  inverse_Hessian = function() {
    g <- autograd_grad(-self$loglik(), self$params_vec, create_graph = TRUE)[[1]]
    H <- torch_jacobian(g, self$params_vec)
    free_idx <- torch_nonzero(self$params_free)$view(-1)
    self$Hinv <- torch_inverse(H[free_idx, ][, free_idx])
    return(self$Hinv)
  },
  
  
  
  fit = function(lrate = 0.05, maxit = 5000, verbose = TRUE, tol = 1e-20) {
    if (verbose) cat("Fitting SEM with Adam optimizer and MVN log-likelihood loss\n")
    optim <- optim_adam(self$params_vec, lr = lrate)
    prev_loss <- 0.0
    for (epoch in 1:maxit) {
      optim$zero_grad()
      loss <- -self$loglik()
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
  }
  ,
  
  lasso_fit = function(lrate = 0.05, maxit = 5000, tol = 1e-20,v=1){
    optim <- optim_adam(self$params_vec, lr = lrate)
    prev_loss <- 0.0
    for (epoch in 1:maxit) {
      optim$zero_grad()
      loss <- -self$lasso_loss(v)
      loss$backward()
      optim$step()
      if (epoch > 1 && abs(loss$item() - prev_loss) < tol) {
        break
      }
      prev_loss <- loss$item()
    }    
    if (epoch == maxit) warning("maximum iterations reached")
    return(invisible(self))
  },
  
  custom_fit = function(lrate = 0.05, maxit = 5000, tol = 1e-20,verbose = TRUE){
    if(is.null(self$custom_loss)){warning('No Custom Loss function provided!')}
    if (verbose) cat("Fitting SEM with Adam optimizer and MVN log-likelihood loss\n")
    optim <- optim_adam(self$params_vec, lr = lrate)
    prev_loss <- 0.0
    for (epoch in 1:maxit) {
      optim$zero_grad()
      self$forward()
      loss <- self$custom_loss(self$sigma,self)
      if (verbose) {
        cat("\rEpoch:", epoch, " Value of Loss Function:", loss$item())
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
  
  get_df=function(){
    self$df <- self$n*(self$n+1)/2 - sum(self$params_free_sizes)
    return(self$df)
  }
  ,
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
  find_correct_indices = function(reference_array, input_array) {
    
    return(which(reference_array %in% input_array))
  },
  get_residuals=function(){
    
    if (is.null(self$sigma)) warning('Data must be fitted first')
    ind_idx <- which(as_array(self$params_free$reshape(-1))[(self$params_sizes[1] + 1):sum(self$params_sizes[1:2])] == TRUE)
    ind_idx <- self$find_correct_indices(diagonal_idx(self$n),ind_idx)
    residual_names <- self$vars[ind_idx]
    counter <- 1
    H_ind1 <- (self$params_free_sizes[1]+1)
    H_ind2  <- sum(self$params_free_sizes[1:2])
    col1<- as_array(self$theta$t()$reshape(-1)[ind_idx])
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
  
  lasso_update_params_removed = function(v,epsilon){
    if (is.null(self$sigma)) warning('Data must be fitted first')
    if (self$lasso == FALSE) warning('Set lasso = TRUE first!')
    lower_triangle_idx <- lower_triangle_idx(self$m)
    partial_correlations <- as_array(self$omega_psi$t()$reshape(-1)[lower_triangle_idx]$reshape(-1))
    ind_idx <- which(as_array(self$params_free[self$params_free_starting_pts[3]:(self$params_free_starting_pts[3]+length(lower_triangle_idx)-1)]$reshape(-1))==1)
    partial_correlations <- partial_correlations[ind_idx]
    ind <- which(abs(partial_correlations) > epsilon)
    self$lasso_num_params_removed <- (self$m*(self$m +1)/2 - self$m) - length(ind) 
  },
  
  get_partial_correlations = function(epsilon = 0.00001){
    if (is.null(self$sigma)) warning('Data must be fitted first')
    lower_triangle_idx <- lower_triangle_idx(self$m)
    partial_correlations <- as_array(self$omega_psi$t()$reshape(-1)[lower_triangle_idx]$reshape(-1))
    latent_pairings <- self$get_all_latent_pairings()
    ind_idx <- which(as_array(self$params_free[self$params_free_starting_pts[3]:(self$params_free_starting_pts[3]+length(lower_triangle_idx)-1)]$reshape(-1))==1)
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
    H_ind1 <- (sum(self$params_free_sizes[1:2])+1)
    H_ind2 <- sum(self$params_free_sizes)
    col2 <- sqrt(as_array(torch_diag(self$inverse_Hessian()[H_ind1:H_ind2,][,H_ind1:H_ind2])$reshape(-1)))
    
    col3<-sapply(col1/col2, get_p_value)
    col4<-ifelse(col3<0.05, "*","")
    self$partial_corr <- data.frame(`Partial Corr` = round(col1,3),`Standard error`= round(col2,3),
                                    `P value`= round(col3,4),`Significant`=col4)
    rownames(self$partial_corr) <- latent_pairings
    return(self$partial_corr)
    
  },
  
  get_criterion_value = function(criterion,gamma = 0.5){
    if (criterion == "AIC") return(2*(sum(self$params_free)-self$lasso_num_params_removed) - 2*self$loglik())
    else if (criterion == "BIC") return((sum(self$params_free)-self$lasso_num_params_removed) *log(self$num_obs) - 2*self$loglik())
    else if (criterion == "EBIC"){
      num_params_free <- sum(self$params_free) -self$lasso_num_params_removed
      return((num_params_free) *log(self$num_obs) - 2*self$loglik() + 2*gamma*log(num_params_free))
    }
    else if (criterion == "chisq") {
      if (self$lasso == TRUE) {warning('Not Applicable when lasso = TRUE')}
      if (is.null(self$sigma)){self$forward()}
      return((self$num_obs-1)*(torch_det(self$sigma)$log() + torch_trace(self$cov_matrix$mm(torch_inverse(self$sigma))) - torch_det(self$cov_matrix)$log()- self$n ))
    }
    
  }
  
) 






