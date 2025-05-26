
#' For a list of correlation matrices (as R matrices) provided, returns the same list
#' but with matrices as torch tensors
#' 
#' Example: 
#' 
#' corMats <- list(
#'matrix(
#'  c(1, 0.31, -0.09,
#'    0.31, 1, -0.23,
#'    -0.09, -0.23, 1),
#'  nrow = 3, ncol = 3),
#'matrix(c(
#'  1, 0.27, -0.01,
#'  0.27, 1, -0.18,
#'  -0.01, -0.18, 1),
#'  nrow = 3, ncol = 3),
#'matrix(c(
#'  1, 0.23, -0.01,
#'  0.23, 1, -0.25,
#'  -0.01, -0.25, 1),
#'  nrow = 3, ncol = 3)
#')
#' get_corMats_as_tensors(corMats) 
#' Output: Same list but matrices are now torch tensors
#' 
#'
#' @param corMats list of matrices as base R matrices
#' @return list of matrices as torch tensors
get_corMats_as_tensors <- function(corMats){
  return(lapply(corMats,function(mat) torch_tensor(mat))
  )
}

#' For a list of correlation matrices (as torch tensors) provided, returns the dimension of the network
#' which is assumed to be the same for each correlation matrix.
#' 
#' Example: 
#' 
#' corMats <- list(
#'matrix(
#'  c(1, 0.31, -0.09,
#'    0.31, 1, -0.23,
#'    -0.09, -0.23, 1),
#'  nrow = 3, ncol = 3),
#'matrix(c(
#'  1, 0.27, -0.01,
#'  0.27, 1, -0.18,
#'  -0.01, -0.18, 1),
#'  nrow = 3, ncol = 3),
#'matrix(c(
#'  1, 0.23, -0.01,
#'  0.23, 1, -0.25,
#'  -0.01, -0.25, 1),
#'  nrow = 3, ncol = 3)
#')
#' corMats <- get_corMats_as_tensors(corMats) 
#' 
#' get_network_dim(corMats) 
#' Output: 3
#' 
#' Explanation: the dimension of the network is 3
#'
#' @param corMats list of torch tensors
#' @return dimension of network
get_network_dim <- function(corMats){
  return(dim(corMats[[1]])[1])
}


#' For a list of correlation matrices (as torch tensors), the dimension of the network (n) and the number of studies (m) provided, 
#' returns the mean vector of the correlations present in the network
#' 
#' Example: 
#' 
#' corMats <- get_corMats_as_tensors(corMats) 
#' get_mean_corr(corMats,3,3) 
#' 
#' Output: Mean vector of the correlations present in the network
#' 
#' Explanation: Extracts and sums the correlations from the matrices by using the dimension network and then divides
#' the derived sum over the number of studies
#'
#' @param corMats list of torch tensors
#' @param n dimension of the network 
#' @param m number of studies
#' @return mean vector of correlations present in the network
get_mean_corr <- function(corMats,n,m){
  return((Reduce('+',corMats)/m)$reshape(-1)[strict_lower_triangle_idx(n)])
}


#' For a list of correlation matrices (as torch tensors), the vector of correlation means,
#' the number of omega params, the dimension of the network (n) and the number of studies (m) provided, 
#' returns the sample correlation matrix of the correlations
#' 
#' Example:
#' 
#' get_sample_corr_of_corr(corMats,mean = NULL, omega_params_length = 3,3,3) 
#' Output: sample correlation matrix of the correlations
#' 
#' Explanation: Uses information provided to calculate sample correlations
#'
#' @param corMats list of torch tensors 
#' @param mean mean vector of correlations, by default NULL 
#' @param omega_params_length number of omega parameters under consideration
#' @param n dimension of the network 
#' @param m number of studies
#' @return sample correlation matrix of correlations
get_sample_corr_of_corr <- function(corMats,mean,omega_params_length,n,m){
  if(is.null(mean)){
    mean <- get_mean_corr(corMats, n, m)
  }
  sum_squares <- lapply(corMats,function(mat,mean,n,params_length){
    exp <- (mat$reshape(-1)[strict_lower_triangle_idx(n)] - mean)$reshape(c(params_length,1))
    return(exp$matmul(exp$t()))},mean=mean, n=n, params_length=omega_params_length)
  return(Reduce('+',sum_squares)/m)
}


#' Converts input information to the necessary data structures to start the torch optimization procedure
#' for random effects MAGNA
#' 
#' Example:
#' magna_convert_to_torch_opts(corMats,missing = FALSE,params_free_filter = NULL,
#' params_value_filter = NULL, dtype = torch_float32(), device = torch_device("cpu"))
#' Output: list of information to be used in random effects MAGNA estimation via torch
#' 
#' Explanation: Converts input information for the start of the torch optimization procedure
#'
#' @param corMats list of torch tensors 
#' @param missing indicator for whether missing values are present , by default FALSE 
#' @param params_free_filter torch tensor of booleans (ones and zeros) to indicate which parameters are free,
#' by default NULL
#' @param params_value_filter torch tensor of user-defined starting values to use at start of optimization,
#' by default NULL
#' @param dtype data type to use for torch operations
#' @param device device to use for torch operations
#' @return list containing the following information to be input into the torch module for
#' random effects MAGNA: params_vec, params_free, params_value, omega_params_length,
#' tau_params_length
magna_convert_to_torch_opts <- function(corMats,missing = FALSE,params_free_filter = NULL,
                                        params_value_filter = NULL,dtype = NULL, device = NULL){
  get_network_dim <- function(corMats){
    return(dim(corMats[[1]])[1])
  }
  network_dim <- get_network_dim(corMats)
  get_sum_to_n <- function(n){
    result <- 0
    counter <- 1
    while (counter <= n){
      result <- result + counter
      counter <- counter + 1
    }
    return(result)
  }
  get_num_of_params_in_lower_triag <- function(n){
    result <- 0
    counter <- 1
    while (counter < n){
      result <- result + counter
      counter <- counter + 1
    }
    return(result)
  }
  
  if (missing == FALSE){
    total_omega_params <- 0
    total_tau_params <- 0
    total_omega_params <- get_num_of_params_in_lower_triag(dim(corMats[[1]])[1])
    if(total_omega_params != 1) total_tau_params <- total_omega_params*(total_omega_params+1)/2
    else total_tau_params <- 1
    params_vec <- torch_cat(list(torch_zeros(total_omega_params,dtype = dtype, device = device),
                                 torch_tensor(rep(0.1,total_tau_params),dtype = dtype, device = device)))
    if(is.null(params_free_filter)){
      params_free <- torch_ones(total_omega_params+total_tau_params,dtype = torch_bool(), device = device,
                                requires_grad = FALSE)
    } else{
      params_free <- torch_tensor(params_free_filter,dtype = torch_bool(), device = device,requires_grad = FALSE)
    }
    
    if(is.null(params_value_filter)){
      params_value <-  torch_zeros(total_omega_params+total_tau_params,dtype = dtype, device = device,
                                   requires_grad = FALSE)
    } else {
      params_value <- torch_tensor(params_value_filter,dtype = dtype, device = device,requires_grad = FALSE)
    }
    
    return(list(params_vec = params_vec, params_free = params_free,
                params_value = params_value,
                omega_params_length = total_omega_params,
                tau_params_length=total_tau_params))
  }
}



#' Outputs a strict duplication matrix as a torch tensor for any requested dimension n
#' 
#' Example:
#' strict_duplication_matrix(2)
#' 
#' Output: 
#'  0
#'  1
#'  1
#'  0
#' 
#' Explanation: Left multiplying the strict duplication matrix for any vechs(X) gives vech(X)
#'
#' @param n dimension as requested
#' @return strict duplication matrix of dimension n as a torch tensor
strict_duplication_matrix <- function(n) {
  total_elements <- n^2
  num_strict_lower <- n * (n - 1) / 2
  
  D_s <- torch_zeros(total_elements, num_strict_lower)
  col <- 1
  
  for (j in 1:(n - 1)) {
    for (i in (j + 1):n) {
      idx1 <- (j - 1) * n + i
      idx2 <- (i - 1) * n + j
      D_s[idx1, col] <- 1
      D_s[idx2, col] <- 1
      col <- col + 1
    }
  }
  
  return(D_s)
}


#' Outputs the Kronecker product of A and B, 2 torch tensors
#' 
#' Example:
#' kron(torch_eye(2),torch_eye(2))
#' 
#' Output: 
#' 1  0  0  0
#' 0  1  0  0
#' 0  0  1  0
#' 0  0  0  1
#' 
#' Explanation: Recall how the Kronecker product works, where each entry in the left matrix is multiplied 
#' with the right matrix - these individual "sub" products each form a block matrix in the new output matrix
#'
#' @param A left torch tensor
#' @param B right torch tensor
#' @return Kronecker product of A and B 
kron <- function(A, B) {
  a_dim <- A$size()
  b_dim <- B$size()
  A_exp <- A$unsqueeze(3)$unsqueeze(4) 
  B_exp <- B$unsqueeze(1)$unsqueeze(2) 
  C <- (A_exp * B_exp)$permute(c(1, 3, 2, 4)) 
  C <- C$reshape(c(a_dim[1] * b_dim[1], a_dim[2] * b_dim[2]))  
  
  return(C)
}


#' Converts a vector of omega params into a vector containing omega params and zeros along the diagonal
#' in the original omega matrix
#' 
#' Example:
#' transform_omega_params(torch_tensor(c(1,1,1)),3,3)
#' 
#' Output: 
#' 0
#' 1
#' 1
#' 0
#' 1
#' 0
#' 
#' Explanation: zeros are inserted at indices corresponding to the diagonal in the omega matrix, 
#' where the indices of the omega matrix start from the leftmost column and continue 
#' downwards and rightwards 
#'
#' @param omega_params torch tensor or R base vector of omega parameters
#' @param dim number of omega parameters
#' @param n dimension of network/omega matrix
#' @return torch tensor containing omega parameters and inputed 0s
transform_omega_params <- function(omega_params,dim,n){
  transformed_omega_params <- torch_zeros(dim+n)
  zero_idx <- diagonal_idx(n)
  counter <- 1
  for (i in 1:length(transformed_omega_params)){
    if (!i %in% zero_idx){
      transformed_omega_params[i] <- omega_params[counter]  
      counter <- counter + 1
    } else{
      transformed_omega_params[i] <- 0
    }
  }
  return(transformed_omega_params)
}


#' Prunes a given random effects MAGNA model and refits the pruned model, assuming standard errors for all
#' initial omega parameters are available
#' 
#' Example:
#' random_effects_MAGNA_prune(mod)
#' 
#' Output: Random effects MAGNA model mod with non-significant omega parameters removed
#' 
#'
#' @param mod Random effects MAGNA model
#' @param lrate learning rate for refitting the model, default 0.05
#' @param maxit maximum number of iterations for refitting the model, default 5000
#' @param verbose indicate whether loss function value should be printed for each iteration of optimization, default TRUE
#' @param tol threshold needed to be met for convergence, default 1e-20
#' @return Random effects MAGNA model mod with non-significant omega parameters removed
#' 
#' @export
random_effects_MAGNA_prune <- function(mod,lrate = 0.05, maxit = 5000, verbose = TRUE, tol = 1e-20){
  if(is.null(mod$summary)){
    print("Run get_summary() first before pruning")
  }
  if(anyNA(mod$summary$`Standard error`)){
    return("Not all standard errors for omega parameters are properly estimated! (Some are negative)")
  }
  for (i in which(mod$summary$Significant[1:mod$omega_params_length]=="")){
    mod$params_free[i] <- 0
  }
  print('Reestimating pruned model...')
  mod$fit(lrate = lrate, maxit = maxit, verbose = verbose, tol = tol)
  return(mod)
}










