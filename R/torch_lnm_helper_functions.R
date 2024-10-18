library(torch)
library(lavaan)
library(pracma)



### Warning: This is Beta Code! 





#' For a symmetric matrix A, indicates the indices of 
#' the entries of the strict lower triangle of A inside vec(A)
#' 
#' Example: strict_lower_triangle_idx(3) 
#' Output: 2 3 6
#' Explanation: the entries referred to by the indices
#' correspond to (2,1), (3,1) and (3,2) 
#' entries of the symmetric matrix A 
#' which are the 2, 3 and 6th entries
#' of vec(A) when read from top to bottom
#'
#' @param n dimension of the adjacency matrix
#' @return array of indices
strict_lower_triangle_idx <- function(n){
  if (n < 2){
    warning("n must be at least 2")
  }
  idx <- c()
  current_idx <- 1
  for (i in 1:(n-1)){
    idx <- c(idx,seq(from = current_idx + i ,to = current_idx +n-1))
    current_idx <- current_idx + n
  }
  return(idx)
}




#' For a symmetric matrix A, indicates the indices of 
#' the entries of the lower triangle of A inside vec(A)
#' 
#' Example: lower_triangle_idx(3) 
#' Output: 1 2 3 5 6 9
#' Explanation: the entries referred to by the indices
#' correspond to (1,1), (2,1), (3,1), (2,2), (3,2), (3,3) 
#' entries of the symmetric matrix A 
#' which are the 1, 2, 3, 5, 6 and 9th entries
#' of vec(A) when read from top to bottom
#'
#' @param n dimension of the adjacency matrix
#' @return array of indices
lower_triangle_idx <- function(n){
  if (n < 2){
    warning("n must be at least 2")
  }
  idx <- c()
  current_idx <- 1
  for (i in 1:n){
    idx <- c(idx,seq(from = current_idx + i -1 ,to = current_idx +n-1))
    current_idx <- current_idx + n
  }
  return(idx)
}




#' For an adjacency matrix A, indicates the indices of 
#' the entries of the diagonal of A inside vec(A)
#' 
#' Example: diagonal_idx(3) 
#' Output: 1 4 6
#' Explanation: the entries referred to by the indices
#' correspond to (1,1), (2,2) and (3,3)
#' entries of the symmmetric matrix A 
#' which are the 1, 4 and 6th entries
#' of vec(A) when read from top to bottom
#'
#' @param n dimension of the matrix
#' @return array of indices
diagonal_idx <- function(n){
  if (n < 1){
    warning("n must be at least 1")
  }
  if (n ==1){
    return(1)
  }
  idx <- c(1)
  current_idx <- 1
  last_idx <- 0
  for (i in 1:(n-1)){
    idx <- c(idx,current_idx+(n-i + 1))
    current_idx <- current_idx+(n-i + 1)
  }
  return(idx)
}








#' Constructs index vector for transforming a vech vector
#' into a vec vector to create an n*n symmetric matrix
#' from the vech vector.
#' 
#' 
#' Example: idx <-  vech_dup_idx(3) 
#'         tensor <- torch_tensor(c(1,2,3,4,5,6)) #tensor is a vech vector
#'         tensor$index_select(0, idx)$view(3,3)
#' Output: 3 x 3 symmetric matrix:
#'         1 2 3
#'         2 4 5
#'         3 5 6
#' Explanation: From the vech vector (1,2,3,4,5,6), we from the
#' vec vector (1,2,3,2,4,5,3,5,6) and from there, create the 
#' 3 x 3 symmetric matrix
#'
#' @param n size of the resulting square matrix
#' @return array containing the indices
vech_dup_idx <- function(n) {
  indices <- integer(n^2)
  cur_idx <- 0
  for (row in 0:(n-1)) {
    for (col in 0:(n-1)) {
      cur_idx <- cur_idx + 1
      if (row == col) indices[cur_idx] <- row * (2 * n - row + 1) / 2
      if (row < col) indices[cur_idx] <- row * (2 * n - row + 1) / 2 + col - row
      if (row > col) indices[cur_idx] <- col * (2 * n - col + 1) / 2 + row - col
    }
  }
  return(indices + 1)
}





#' Compute Jacobian of output wrt input tensor
#'
#' Example: example_function <- function(x) {
#' y <- x[1]^2 + x[2]^2
#' z <- x[1] + x[2]
#' torch_stack(list(y,z))
#' }
#' example_tensor <- torch_tensor(c(1,2),requires_grad = TRUE)
#' example_function(example_tensor)$shape[1]
#' torch_jacobian(example_function(example_tensor),example_tensor)
#' 
#' Output: 2 x 2 transpose of Jacobian Matrix:
#'         2 4
#'         1 1
#' Explanation: We define an example function (target function)
#' which outputs an example tensor. Placing both into torch_jacobian
#' returns the Jacobian of the target function
#'
#' @param output Tensor vector denoting output of target function 
#' @param input Tensor vector denoting input of target function 
#'
#' @return Jacobian matrix 
torch_jacobian <- function(output, input) {
  jac <- torch_zeros(output$shape[1], input$shape[1], dtype = input$dtype)
  for (i in 1:output$shape[1])
    jac[i] <- autograd_grad(output[i], input, retain_graph = TRUE)[[1]]
  
  return(jac)
}





#' Prepares a dataframe for the lnm/rnm model. This function
#' first converts the variables to a design matrix, then centers
#' it, and lastly converts it to a torch tensor
#'
#'
#'#' Example: df_to_tensor(mtcars)
#' 
#' Output: torch tensor with columns corresponding to mtcars variables
#' 
#' Explanation: Each variable mtcars is centered and the resulting 
#' dataframe is first converted to a torch tensor before it is
#' returned.
#'
#'
#' @param df data frame
#' @param dtype data type of the resulting tensor
#' @param device device to store the resulting tensor on
#'
#' @return Torch tensor of scaled and processed data
#'
#' @importFrom stats model.matrix
#'
#' @seealso [torch::torch_tensor()], [stats::model.matrix()]
df_to_tensor <- function(df, dtype = NULL, device = NULL) {
  torch::torch_tensor(
    data = scale(model.matrix(~ . - 1, df), scale = FALSE),
    requires_grad = FALSE,
    dtype = dtype,
    device = device
  )
}




#' Based on the type of matrix, returns a list containing 1)mat_value, a matrix containing the 
#' starting values of the matrix parameters and 2)mat_free, a matrix of 1s and 0s which 
#' indicate which entries of the matrix are free parameters.  
#'
#' Example: lnm_constraint_matrices(type='omega_psi',n_rows=3,n_cols = 3, entries=list(c(2),c(1),c(0.8)))
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
#'
#' @return list containing mat_value and mat_Free
#'
lnm_constraint_matrices <- function(type = "lambda",lambda = NULL,n_rows, n_cols, entries = NULL) {
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
  else if (type == "theta"){
    mat_value <- diag(1, n_rows)
    mat_free <- matrix(0, nrow = n_rows, ncol = n_cols)
    diag(mat_free) <- 1
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
#' params_sizes - a vector which indicate the number of parameters in the lambda, omega_psi and theta
#' matrices respectively, 2) params_free_sizes - a vector which indicate the number of free parameters in the lambda, omega_psi and theta
#' matrices respectively, 3) params_free_sizes_max - a vector which indicate the maximum possible number of free parameters in the lambda, omega_psi and theta
#' matrices respectively, 4) params_free_starting_pts - a vector which indicate the starting index of 
#' each group of parameters, lambda, omega_psi and theta respectively and 5) lambda_matrix_free - a matrix
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
#' @param lambda_constraint_lst (Optional) list containing 3 vectors - the row, column indices and values of the fixed parameters 
#' @param theta_constraint_lst (Optional) list containing 3 vectors - the row, column indices and values of the fixed parameters
#' @param omega_psi_constraint_lst (Optional) list containing 3 vectors - the row, column indices and values of the fixed parameters
#'
#' @return list containing params_start, params_free, params_value, params_sizes, params_free_sizes,
#' params_free_sizes_max, params_free_starting_pts, lambda_matrix_Free
#'
lnm_mod_to_torch_opts <- function(lambda,
                                  lambda_constraint_lst = NULL,
                                  theta_constraint_lst = NULL,
                                  omega_psi_constraint_lst = NULL){
  
  n <- dim(lambda)[1]
  m <- dim(lambda)[2]
  
  lambda_constraint_matrices <- lnm_constraint_matrices('lambda',lambda=lambda,n,m,entries=lambda_constraint_lst)
  omega_psi_constraint_matrices <- lnm_constraint_matrices('omega_psi',lambda=NULL,m,m,entries=omega_psi_constraint_lst)
  theta_constraint_matrices <- lnm_constraint_matrices('theta',lambda=NULL,n,n,entries=theta_constraint_lst)
  
  
  lambda_start <- lavaan::lav_matrix_vec(lambda_constraint_matrices[[1]])
  lambda_matrix_free <- lambda_constraint_matrices[[2]]
  lambda_free <- lavaan::lav_matrix_vec(lambda_matrix_free )
  lambda_free_idx <- which(lambda_free!= 0)
  lambda_free_size <- length(lambda_free_idx)
  
  theta_start <- lavaan::lav_matrix_vech(theta_constraint_matrices[[1]])
  theta_free <- lavaan::lav_matrix_vech(theta_constraint_matrices[[2]])
  theta_free_idx <- which(theta_free!= 0)
  theta_free_size <- length(theta_free_idx)
  
  omega_psi_start <- lavaan::lav_matrix_vech(omega_psi_constraint_matrices[[1]])
  omega_psi_free <- lavaan::lav_matrix_vech(omega_psi_constraint_matrices[[2]])
  omega_psi_free_idx <- which(omega_psi_free!= 0)
  omega_psi_free_size <- length(omega_psi_free_idx)
  
  params_sizes <- sapply(list(lambda_start,theta_start, omega_psi_start), length)
  
  torch_list <- list(
    params_start = c(lambda_start,theta_start,omega_psi_start),
    params_free = c(lambda_free,theta_free,omega_psi_free),
    params_value = c(lambda_start,theta_start,omega_psi_start),
    params_sizes =   params_sizes ,
    params_free_sizes = c(lambda_free_size,theta_free_size,omega_psi_free_size),
    params_free_sizes_max = c(length(which(lambda!= 0)),n,m*(m+1)/2 - m),
    params_free_starting_pts = c(1,1+params_sizes[1],1+sum(params_sizes[1:2])),
    lambda_matrix_free = lambda_matrix_free )
  
  return(torch_list)
}



#' Returns the p-value of a 2-tailed Z test
#' 
#' 
#' Example: get_p_value(1.96)
#' Output: 0.04999579
#' 
#' Explanation: Z-statistic of 1.96 roughly corresponds to a p-value of 0.05 from a 2-tailed Z test.
#'
#'
#' @param x Z-statistic observed 
#' @return p-value from a 2-tailed Z test
#' 
get_p_value=function(x){
  return(pnorm(abs(x),mean=0,sd=1,lower.tail=FALSE) + pnorm(-abs(x),mean=0,sd=1,lower.tail=TRUE))
}



#' Transforms vechs vector into an n x n symmetric matrix with 0s along its diagonal
#' 
#' 
#' Example: vech <- c(1, 2, 3, 4, 5, 6)
#'          n <- 4  
#'          vech_to_symmetric_zero_diag(vech, n)
#' Output:      
#'         0    1    2    3
#'         1    0    4    5
#'         2    4    0    6
#'         3    5    6    0
#' 
#' Explanation: We have a vechs of c(1,2,3,4,5,6) which correspond to the lower triangle of our 
#' output matrix. The diagonal are filled with 0s and the rest of the entries are filled to ensure 
#' symmetry.
#'
#'
#' @param x vector vechs
#' @param n n denoting dimension of output matrix
#' 
#' @return p-value from a 2-tailed Z test
#' 
vech_to_symmetric_zero_diag <- function(vech, n) {
  if (length(vech) != (n * (n - 1)) / 2) {
    stop("Length of vech does not match matrix size")
  }
  mat <- matrix(0, n, n)
  mat[lower.tri(mat)] <- vech
  mat <- mat + t(mat)
  
  return(mat)
}


#' Given a symmetric matrix outputs a list containing 3 vectors:  The first and second vector represent the 
#' row and column of the non-zero entries inside the lower triangle of the symmetric matrix. 
#' The last vector is just a vector of zeros with length which matches the number of non-zero entries. 
#' 
#' 
#' Example: extract_non_zero_entries(matrix(c(0,1,1,1,0,1,1,1,0),byrow=F,nrow = 3))
#' Output:      
#'         $rows
#'          2 3 3
#'         $cols
#'          1 1 2
#'         $values
#'          0 0 0
#' 
#' Explanation: vector with titles "rows" and "cols" colectively indicate the position of the non-zero entries
#' of the matrix. The remaining vector "values" is just a zero vector.
#'
#'
#' @param mat symmetric matrix
#' 
#' @return list of 3 vectors as described above
#' 
extract_non_zero_entries <- function(mat) {
  non_zero_indices <- which(mat != 0, arr.ind = TRUE)
  non_zero_indices <- non_zero_indices[non_zero_indices[, 1] < non_zero_indices[, 2], ]
  rows <- non_zero_indices[, 2]
  cols <- non_zero_indices[, 1]
  values <- vector(mode='numeric',length = length(rows))
  
  # Return as a list
  return(list(rows = rows, cols = cols, values = values))
}


#' Returns the indices of an reference array which are found in an input array. If none are found,
#' Null is returned
#' 
#' 
#' Example: find_correct_indices(c(0,2,3),c(4,5,6,0))
#' 
#' Output: 1
#'  
#' 
#' Explanation: The value 0 in the reference array (first array) is found in the second array and the 
#' position of 0 in the reference array is 1. So 1 is returned.
#'
#'
#' @param x reference array
#' @param n input array
#' 
#' @return vector of indices denoting positions of entries in reference array found
#' in input array
#' 
find_correct_indices = function(reference_array, input_array) {
  return(which(reference_array %in% input_array))
}


#' Torch lnm module created during pruning/stepping up 
#'
#' Function for creating an lnm model
#'
#' @param params_vec param values for the initializing of the module
#'
#' @return A `torch_sem` object, which is an `nn_module` (torch object)
#'
#' @details
#' This function instantiates a torch object for computing the model-implied covariance matrix
#' based on an lnm model during the pruning/stepping up processes. The methods are identical 
#' to torch_lnm and information about these methods can be looked up under torch_lnm
#' 
#' @import torch
#' @import lavaan
#' @importFrom R6 R6Class
#'
#' @name torch_lnm_stepwise
#'
#' @seealso [df_to_tensor()]
#'
#' @export
tensor_lnm_stepwise <- torch::nn_module(
  classname = "torch_network_model",
  initialize = function(params_vec){
    self$params_vec<-nn_parameter(params_vec)
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


#' Given a torch_lnm module, mod, with params_vec, this copies all the attributes of mod 
#' into a new torch_lnm_stepwise module (except for params_free and params_free_sizes) which 
#' is instantiated during pruning/stepping up.
#' 
#'
#' @param torch_lnm torch_lnm module which is undergoing pruning/stepping up
#' @param params_vec params_vec which belongs to the torch_lnm module. If name of module is mod,
#' "mod$params_vec" should be input here.
#' 
#' @return torch_lnm_stepwise module with the exact same attributes as the original torch_lnm module
#' 
copy_lnm_attributes <- function(mod,params_vec){
  output_model <- tensor_lnm_stepwise(params_vec) 
  output_model$data <- mod$data
  output_model$vars <- mod$vars
  output_model$latents <- mod$latents
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
  output_model$omega_psi_dup_idx <- mod$omega_psi_dup_idx
  output_model$I_mat <- mod$I_mat
  output_model$mu <- mod$mu
  
  return(output_model)
}


#' Given an original torch_lnm/torch_lnm_stepwise module, the function fits all possible models
#' during one iteration of the pruning process and returns either the best model based on criterion score
#' after this fitting process or the original torch_lnm/torch_lnm_stepwise module if all new models give a
#' worse model fit based on criterion score. A possible model here is a model with one of the
#' originally non-zero partial correlations now being set to zero.
#' 
#'
#' @param mod original torch_lnm/torch_lnm_stepwise module undergoing pruning
#' @param criterion string indicating criterion to use - "AIC", "BIC or "EBIC"
#' @param gamma (Optional) value of gamma if "EBIC" is used. Default value is 0.5.
#' 
#' 
#' @return either the original torch_lnm/torch_lnm_stepwise module or 
#' a torch_lnm_stepwise module with a better criterion score
#' 
prune_find_alt_models <- function(mod,criterion = 'BIC',gamma = 0.5){
  omega_psi_ind_1 <- mod$params_free_starting_pts[3]
  omega_psi_ind_2 <- omega_psi_ind_1 + mod$params_sizes[3]
  omega_psi_free <- mod$params_free[omega_psi_ind_1:omega_psi_ind_2]
  sum_omega_psi_free = torch_sum(omega_psi_free)$item()
  if (sum_omega_psi_free == 0) {warning('Cant prune anymore! Left with one partial correlation!')}
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
    clone$params_free_sizes <- c(mod$params_free_sizes[1:2], mod$params_free_sizes[3] - 1) 
    clone$fit(verbose=F)
    models[[i]] <- clone
    list_of_criterion_values[i] <- clone$get_criterion_value(criterion,gamma)$item()
  }
  
  min_index <- which.min(list_of_criterion_values)
  
  if (list_of_criterion_values[min_index] < current_criterion_value) return(models[[min_index]])
  
  else {return(mod)}
}

#' Performs pruning to give the best model based on the selected criterion score
#' At the end of each iteration, one of the partial correlations will be removed and this
#' corresponds to the model with the better model fit based on criterion score. If the models
#' fit at the end of 1 iteration with 1 partial correlation removed each do not lead to
#' a better criterion score, the model found at the end of the previous iteration is returned.
#' 
#' The process terminates if the model is left with only 1 partial correlation at the end of 
#' the iteration.
#'
#' @param mod original torch_lnm/torch_lnm_stepwise module undergoing pruning
#' @param criterion string indicating criterion to use - "AIC", "BIC or "EBIC"
#' @param gamma (Optional) value of gamma if "EBIC" is used. Default value is 0.5.
#' 
#' 
#' @return either the original torch_lnm/torch_lnm_stepwise module or 
#' a torch_lnm_stepwise module with a best criterion score after pruning
#' 
#' 
#' @name prune
#' 
#' 
#' @export
prune <- function(mod,criterion = 'BIC',gamma = 0.5){
  
  
  current_value <- mod$get_criterion_value(criterion,gamma)$item()
  pruned_model <- prune_find_alt_models(mod,criterion,gamma)
  
  if(pruned_model$get_criterion_value(criterion,gamma)$item() == current_value){
    return(mod)
  }else{
    prune(pruned_model,criterion,gamma)
    
  }
}

#' Given an original torch_lnm/torch_lnm_stepwise module, the function fits all possible models
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
stepup_find_alt_models <- function(mod,criterion = 'BIC',gamma = 0.5){
  omega_psi_ind_1 <- mod$params_free_starting_pts[3]
  omega_psi_ind_2 <- omega_psi_ind_1 + mod$params_sizes[3]
  omega_psi_free <- mod$params_free[omega_psi_ind_1:omega_psi_ind_2]
  sum_omega_psi_free = torch_sum(omega_psi_free)$item()
  if (sum_omega_psi_free == mod$params_free_sizes_max[3]) {warning('Cant stepup anymore! Saturated!')}
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
    clone$params_free_sizes <- c(mod$params_free_sizes[1:2], mod$params_free_sizes[3] - 1) 
    clone$fit(verbose=F)
    models[[i]] <- clone
    list_of_criterion_values[i] <- clone$get_criterion_value(criterion,gamma)$item()
  }
  
  min_index <- which.min(list_of_criterion_values)
  
  if (list_of_criterion_values[min_index] < current_criterion_value) return(models[[min_index]])
  
  else {return(mod)}
}


#' Performs stepping up to give the best model based on the selected criterion score
#' At the end of each iteration, one of the originally zeropartial correlations will be
#' turned to non-zero and this corresponds to the model with the better model fit based on criterion score. 
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
#' a torch_lnm_stepwise module with a best criterion score after pruning
#' 
#' @name stepup
#' 
#' @export
stepup<- function(mod,criterion = 'BIC',gamma = 0.5){
  current_value <- mod$get_criterion_value(criterion,gamma)$item()
  stepup_model <- stepup_find_alt_models(mod,criterion,gamma)
  
  if(stepup_model$get_criterion_value(criterion,gamma)$item() == current_value){
    return(mod)
  }else{
    stepup(stepup_model,criterion)
    
  }
}


#' Given a tensor of 1s and 0s, it returns a list of the possible tensors after one of the 1s 
#' is now set to 0
#' 
#' Example: flip_one_1_to_0(torch_tensor(c(1,0,0)))
#' Output: [[1]]
#'        torch_tensor
#'        0
#'        0
#'        0
#'        [ CPUFloatType{3} ]
#'        
#' Explanation: There is only one possibility from torch_tensor(c(1,0,0)), namely
#' torch_tensor(c(0,0,0))
#'
#' @param tensor torch tensor of 1s and 0s
#' 
#' 
#' @return list of possible torch tensors
flip_one_1_to_0 <- function(tensor) {
  one_indices <- torch_nonzero(tensor)
  result_tensors <- list()
  for (i in seq_len(one_indices$size(1))) {
    new_tensor <- tensor$clone()  
    idx <- one_indices[i]
    new_tensor[idx] <- 0  
    result_tensors[[length(result_tensors) + 1]] <- new_tensor
  }
  return(result_tensors)
}


#' Given a tensor of 1s and 0s, it returns a list of the possible tensors after one of the 0s 
#' is now set to 1
#' 
#' Example: flip_one_1_to_0(torch_tensor(c(0,1,1)))
#' Output: [[1]]
#'        torch_tensor
#'        1
#'        1
#'        1
#'        [ CPUFloatType{3} ]
#'        
#' Explanation: There is only one possibility from torch_tensor(c(0,1,1)), namely
#' torch_tensor(c(1,1,1))
#'
#' @param tensor torch tensor of 1s and 0s
#' 
#' 
#' @return list of possible torch tensors
flip_one_0_to_1 <- function(tensor) {
  zero_indices <- torch_nonzero(tensor == 0)
  result_tensors <- list()
  for (i in seq_len(zero_indices$size(1))) {
    new_tensor <- tensor$clone()  
    idx <- zero_indices[i]
    new_tensor[idx] <- 1 
    result_tensors[[length(result_tensors) + 1]] <- new_tensor
  }
  return(result_tensors)
}


#' Performs a lasso search to find the value of hyperparameter v which correspond to the model 
#' which gives the lowest criterion scores (AIC/BIC/EBIC). Each model is fit with a different 
#' value of v and partial correlations which are lower than the threshold value are set to 0/
#' removed from the pool of free parameters before the criterion score is calculated. 
#' 
#' 
#' @param mod original torch_lnm/torch_lnm_stepwise module undergoing stepping up
#' @param criterion string indicating criterion to use - "AIC", "BIC or "EBIC"
#' @param criterion (Optional) values of the hyperparameter v to be used in the search. Default range 
#' of values is 30 values of v from 0.01 to 100 spread out on a log scale.
#' @param epsilon threshold under which partial correlations will be set to 0
#' @param gamma (Optional) value of gamma if "EBIC" is used. Default value is 0.5.
#' 
#' 
#' @return value of v of the model which gives the best criterion score
#' 
#' @name lasso_explore
#' 
#' @export
lasso_explore <- function(mod, criterion = "BIC", v_values = NULL ,epsilon = 0.00001, gamma = 0.5){
  if (mod$lasso == FALSE) {warning('Set lasso to TRUE first!')}
  if (is.null(v_values)) {v_values <- logspace(log10(0.01), log10(100), 30)}
  criterion_values <- vector(mode = 'numeric',length=length(v_values))
  for (i in 1:length(v_values)){
    mod$lasso_fit(v = v_values[i])
    mod$lasso_update_params_removed(epsilon = epsilon)
    criterion_values[i] <- mod$get_criterion_value(criterion,gamma)$item()
  }
  ind <- which.min(criterion_values)
  return(v_values[ind])
}



#' Indicates the positions of the free parameters from
#' an adjacency matrix of the latent variables with 1s 
#' and non-free parameters with 0s.
#' The free parameters correspond to the non-diagonal
#' entries from the lower triangle of the adjacency matrix.
#' The returned vector of 0s and 1s start from the top of 
#' the first column and proceeds downwards.Thereafter, it
#' starts from the diagonal entry of the second column 
#' and so on.
#' 
#' Example: omega_psi_free_idx(3) 
#' Output: 0 1 1 0 1 0 
#' Explanation: the free entries correspond to (2,1), (3,1) and (3,2) 
#' entries of the adjacency matrix
#'
#' @param n dimension of the adjacency matrix
#' @return array containing 1s and 0s
# omega_psi_free_idx <- function(n){
#   if (n < 2){
#     warning("n must be at least 2")
#   }
#   idx <- c()
#   current_idx <- 1
#   for (i in 1:n){
#     idx <- c(idx,c(0,rep(1, n-i)))
#     current_idx <- current_idx + n
#   }
#   return(idx)
# }
