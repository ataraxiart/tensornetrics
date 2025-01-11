library(torch)
library(lavaan)
library(pracma)



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

library(Rcpp)



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
  non_zero_indices <- matrix(non_zero_indices[non_zero_indices[, 1] < non_zero_indices[, 2], ],ncol=2)
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




#' Given the lambda matrix belonging to an lnm/rnm, and the type of matrix needed specified,
#' it returns the matrix either representing the free parameters of the lambda matrix
#' of the matrix of the starting values of the lambda matrix.
#' 
#' Example: lambda <- matrix(0,10,2)
#'          lambda[c(1,2,4,6,7),1] <- 1 
#'          lambda[c(3,5,8,9,10),2] <- 1 
#'          format_lambda_mat(lambda,type = "free")
#'          format_lambda_mat(lambda,type = "value")
#'          
#' Output: 
#'        [,1] [,2]
#'  [1,]    0    0
#'  [2,]    1    0
#'  [3,]    0    0
#'  [4,]    1    0
#'  [5,]    0    1
#'  [6,]    1    0
#'  [7,]    1    0
#'  [8,]    0    1
#'  [9,]    0    1
#'  [10,]   0    1
#'  
#'  
#'        [,1] [,2]
#'  [1,]     1    0
#'  [2,]     0    0
#'  [3,]     0    1
#'  [4,]     0    0
#'  [5,]     0    0
#'  [6,]     0    0
#'  [7,]     0    0
#'  [8,]     0    0
#'  [9,]     0    0
#'  [10,]    0    0
#'  
#'        
#' Explanation: There is only one possibility from torch_tensor(c(0,1,1)), namely
#' torch_tensor(c(1,1,1))
#'
#' @param matrix lambda matrix of 
#' @param string "free" or "value"
#' 
#' @return binary matrix which represents the free parameters of the lambda matrix
#' or the matrix which contains the starting values of the lambda matrix.
format_lambda_mat <- function(mat,type = NULL){
  n <- nrow(mat)
  m <- ncol(mat)
  if (type == "free"){
    for (j in 1:m){
      i <- 1
      while (mat[i,j] == 0){
        i <- i + 1
      }
      mat[i,j] <- 0
    }
    return(mat)
  }
  else if (type == "value"){
    for (j in 1:m){
      i <- which.max(mat[,j])
      mat[,j] <- 0
      mat[i,j] <- 1
    }
    return(mat)
  }
}




#' Checks if the given matrix is positive definite
#' 
#' Example: is_positive_definite(matrix(c(1,1,1,0),byrow=T,ncol = 2))
#' Output: FALSE
#'       
#'        
#' Explanation: The given matrix is not positive definite. 
#'
#' @param matrix matrix such as the covariance matrix estimated from lnm/rnm
#' 
#' 
#' @return boolean indicating if matrix is positive definite
is_positive_definite <- function(matrix) {
  tryCatch({
    linalg_cholesky(matrix)
    TRUE
  }, error = function(e) {
    FALSE
  })
}


#' Scales the current parameter estimates by a specified scale factor in the case
#' the current estimates lead to a covariance matrix which is not positive definite. 
#' The default value of the scaling factor is 0.9
#' 
#' Example: get_alternative_update(torch_tensor(c(0,1,1)))
#' Output: 
#'        torch_tensor
#'        0
#'        0.9
#'        0.9
#'        [ CPUFloatType{3} ]
#'        
#' Explanation: The individual values are scaled by a factor of 0.9
#'
#' @param tensor current parameters of the lnm/rnm model
#' @param numeric factor to scale the current parameters
#' 
#' @return torch tensor containing the new parameters 
get_alternative_update <- function(current_params, scale_factor = 0.9) {
  current_params * scale_factor
}






#' Step Up function for Network Models
#' 
#' #' NOT RECOMMENDED FOR RNM  due to long computation times
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
#'
#' @param mod torch module
#' @param criterion string indicating criterion to use - "AIC", "BIC or "EBIC"
#' @param gamma (Optional) value of gamma if "EBIC" is used. Default value is 0.5.
#' 
#' 
#' @return either the original torch module or 
#' another torch module with the best criterion score after stepping down
#' 
#' @name stepup
#'
#'
#' @export
stepup <- function(mod,criterion = "BIC", gamma = 0.5){
  type <- mod$model_type
  if (type == "lnm"){
    lnm_stepup(mod,criterion,gamma)
  } 
  else if (type == "rnm"){
    rnm_stepup(mod,criterion,gamma)
  } 
}



#' Step Down function for Network Models
#' 
#' NOT RECOMMENDED FOR RNM  due to long computation times
#' 
#' Performs stepping down to give the best model based on the selected criterion score
#' At the end of each iteration, one of the originally free partial correlations will be
#' turned to zero and this corresponds to the model with the best model fit based on criterion score 
#' out of all possibilities (number of non-zero partial correlations at the start). 
#' If the models fit at the end of 1 iteration with 1 partial correlation set to zero each 
#' do not lead to a better criterion score, the model found at the end of the previous iteration is returned.
#' 
#' The process automatically terminates if the model is only left with one partial correlation
#'at the end of all the iterations.
#'
#' @param mod torch module
#' @param criterion string indicating criterion to use - "AIC", "BIC or "EBIC"
#' @param gamma (Optional) value of gamma if "EBIC" is used. Default value is 0.5.
#' 
#' 
#' @return either the original torch module or 
#' another torch module with the best criterion score after stepping down
#' 
#' @name stepdown
#'
#'
#' @export
stepdown <- function(mod,criterion = "BIC", gamma = 0.5){
    type <- mod$model_type
    if (type == "lnm"){
      lnm_stepdown(mod,criterion,gamma)
    } 
    else if (type == "rnm"){
      rnm_stepdown(mod,criterion,gamma)
    } 
  }
  



#' Pruning function for Network Models
#' 
#' Given an  model, it iteratively removes insignificant partial correlations and refits models until
#' all partial correlations are found to be significant. Thereafter ,it returns this model
#' with only significant partial correlations.
#' 
#'
#' @param mod torch module
#' 
#' 
#' @return new refitted model with only significant partial correlations 
#' 
#' @name prune
#'
#'
#' @export
prune <- function(mod){
  type <- mod$model_type
  if (type == "lnm"){
    lnm_prune(mod)
  } 
  else if (type == "rnm"){
    rnm_prune(mod)
  } 
}



#' LASSO for Network Models
#' 
#' Performs a lasso search to find the value of hyperparameter v which correspond to the model 
#' which gives the lowest criterion scores (AIC/BIC/EBIC). At the end, the function also 
#' returns constraints or free parameters to be indicated for the network models. For lnm, it 
#' returns constraints for omega_psi while for rnm, free parameters for omega_theta are returned.
#' 
#' 
#' 
#' @param mod torch module representing an lnm or rnm
#' @param criterion (Optional) string indicating criterion to use - "AIC", "BIC or "EBIC". Default is BIC.
#' @param v_values (Optional) values of the hyperparameter v to be used in the search. Default range 
#' of values is 30 values of v from 0.01 to 100 spread out on a log scale.
#' @param lrate (Optional) learning rate for lasso. Default value is 0.01.
#' @param epsilon (Optional) threshold under which partial correlations will be set to 0. Default is 0.0001.
#' @param gamma (Optional) value of gamma if "EBIC" is used. Default value is 0.5.
#' 
#' 
#' @return a list containing the following: 1)value of v of the model which 
#' gives the best criterion score and 2)either constraints or free partial correlations in the model
#' 
#' @name lasso_explore
#'
#'
#' @export
lasso_explore <- function(mod, criterion = "BIC", v_values = NULL ,lrate = 0.01,epsilon = 0.0001, gamma = 0.5){
    type <- mod$model_type
    if (type == "lnm"){
      lnm_lasso_explore(mod, criterion, v_values, lrate, epsilon, gamma)
    } 
    else if (type == "rnm"){
      rnm_lasso_explore(mod, criterion, v_values, lrate, epsilon, gamma)
    } 
  }
  



#' Format omega_theta matrix for rnm module
#' 
#' Given the omega_theta matrix of an existing rnm model either as an R matrix or as
#' a torch tensor, the function returns the omega_theta matrix in a format for the training
#' of a new rnm model on another set of data but with the same omega_theta matrix
#' 
#' 
#' @param mat omega_theta matrix
#' 
#' @return a list containing the following: 1)rows of nonzero partial correlations,
#' 2)cols of nonzero partial correlations and 3)initial values 
#' 
#' @name format_omega_theta
#'
#' @export
format_omega_theta <- function(mat){
  if (!inherits(mat,'torch_tensor')) {
    input_matrix <- torch_tensor(mat)
  }
  lower_triangle <- mat*torch_tril(torch_ones_like(mat))
  non_zero_indices <- torch_nonzero(lower_triangle)
  rows <- as.numeric(non_zero_indices[,1]) 
  cols <- as.numeric(non_zero_indices[,2])
  total_length <- length(rows)
  values <- vector(mode = 'numeric',length = total_length )
  counter <- 1
  for (i in 1:total_length ){
    values[counter] <- lower_triangle[rows[i],cols[i]]$item()
    counter <- counter + 1
  }
  result <- list(
    rows = rows,
    cols = cols,
    values = values
  )
  return(result)
}







