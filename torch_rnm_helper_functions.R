
library(torch)
library(lavaan)
library(pracma)


source("R/torch_lnm_helper_functions.R", echo = TRUE)

rnm_constraint_matrices <- function(type = "lambda",lambda = NULL,n_rows, n_cols, entries = NULL) {
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
    diag(mat_free) <- 0
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




rnm_mod_to_torch_opts <- function(lambda,
                                  lambda_constraint_lst = NULL,
                                  psi_constraint_lst = NULL,
                                  delta_theta_constraint_lst = NULL,
                                  omega_theta_constraint_lst = NULL
                                  ){
  
  n <- dim(lambda)[1]
  m <- dim(lambda)[2]
  
  lambda_constraint_matrices <- rnm_constraint_matrices('lambda',lambda=lambda,n,m,entries=lambda_constraint_lst)
  psi_constraint_matrices <- rnm_constraint_matrices('psi',lambda=NULL,m,m,entries=psi_constraint_lst)
  delta_theta_constraint_matrices <- rnm_constraint_matrices('delta_theta',lambda=NULL,n,n,entries=delta_theta_constraint_lst)
  omega_theta_constraint_matrices <- rnm_constraint_matrices('omega_theta',lambda=NULL,n,n,entries=omega_theta_constraint_lst)
 
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
  
  
  omega_theta_start <- lavaan::lav_matrix_vech(omega_theta_constraint_matrices[[1]])
  omega_theta_free <- lavaan::lav_matrix_vech(omega_theta_constraint_matrices[[2]])
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


lambda <- matrix(0,10,2)
lambda[c(1,2,4,6,7),1] <- 1 # Positive items
lambda[c(3,5,8,9,10),2] <- 1 # Negative items
data <- read.delim("C:/Users/samue/Documents/Masters/Sem 1/Sacha Work/R codes/selfEsteem.txt",sep='\t',header=TRUE)
vars <- paste0("Q",1:10)
trainData <- data[c(TRUE,FALSE),]
latents <- c('latent1','latent2')

twofac_train <- psychonetrics::rnm(trainData, vars = items, estimator = "FIML",
                                   lambda = Lambda,identification = 'variance') %>% runmodel
getmatrix(twofac_train , "sigma_zeta")
getmatrix(twofac_train , "lambda")
getmatrix(twofac_train , "omega_epsilon")
diag(getmatrix(twofac_train , "delta_epsilon"))
getmatrix(twofac_train , "sigma_epsilon")

rnm <- tensor_rnm(data=trainData,lambda,vars=vars,latents=latents)
rnm$fit(verbose=TRUE,lr=0.1)
rnm$get_df()
rnm$forward()
rnm$psi
rnm$lambda
torch_diag(rnm$delta_theta)
rnm$omega_theta
rnm$loglik()








rnm$lambda_matrix_free

ref_lst <- list(x = vector(length = 2, mode='numeric'),y=vector(length = 2, mode='numeric'))
for (j in 1:2){
  for (i in 1:10){
    if(rnm$lambda_matrix_free[i,j]  == 1){
      ref_lst$x[j] <- i
      ref_lst$y[j] <- j
      break
    }
  }
}

factor_matrix <- torch_tensor(diag(diag(matrix(1,nrow=2,ncol=2))/
                                     diag(rnm$lambda_matrix_free[ref_lst$x ,ref_lst$y]))) 
factor_matrix_inv <- torch_inverse(factor_matrix)
rnm$lambda$mm(factor_matrix)
factor_matrix_inv$mm()

rnm$B_0_inv$mm(factor_matrix_inv$mm(rnm$psi_middle_term$mm(factor_matrix_inv$mm(rnm$B_0))))




