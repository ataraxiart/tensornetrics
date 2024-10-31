# devtools::install_github('ataraxiart/tensornetrics')
library("tensornetrics")
library("psych")
library("psychonetrics")
library("torch")
library("bootnet")
library("mvtnorm")
library("dplyr")


# Simple model:
n_factor <- 8
n_indicator <- 3
n_observation <- 250

# Generate factor loadings:
lambda <- simplestructure(rep(1:n_factor,each=n_indicator)) * runif(n_factor*n_indicator,0.5,1.5)

# Generate residual variances:
sigma_epsilon <- diag(runif(n_factor*n_indicator,0.5,1.5))

# Generate latent network:
omega_zeta <- genGGM(n_factor)

# generate latent variables:
sigma_zeta <- cov2cor(solve(diag(n_factor)-omega_zeta))
eta <- rmvnorm(n_observation,rep(0,n_factor),sigma_zeta)

# Generate residuals:
epsilon <- rmvnorm(n_observation,rep(0,n_factor*n_indicator),sigma_epsilon)

# Generate data:
data <- as.data.frame(eta %*% t(lambda) + epsilon)

# Input:
vars <- names(data)
latents <- paste0("F",1:n_factor) # <- tensor_lnm gives bug now with long names

# Fit pruned psychonetrics model:
lambda_mod <- 1*(lambda!=0)

mod <- psychonetrics::lnm(data,
                          vars=vars,
                          latents=latents,
                          lambda = lambda_mod,
                          estimator = "FIML") %>% runmodel

# Get omega_zeta estimate:
omega_zeta_est_psychonetrics <- getmatrix(mod,"omega_zeta")

# Psychonetrics prune:
mod_prune <- mod %>% prune
omega_zeta_est_psychonetrics_pruned <- getmatrix(mod_prune,"omega_zeta")

# Now with Torch:
lnm <-  tensor_lnm(data=data,
                   lasso=FALSE,
                   lambda=lambda_mod,
                   vars=vars,
                   latents=latents,
                   device=torch_device('gpu'))

#Fit using standard log-likelihood fit function
lnm$fit(verbose = TRUE) 

#Access partial correlations
lnm$get_partial_correlations() # <- probably better as matrix

omega_zeta_est_tensor <- matrix(0,n_factor,n_factor)
omega_zeta_est_tensor[lower.tri(omega_zeta_est_tensor)] <- lnm$partial_corr[,1]
omega_zeta_est_tensor <- t(omega_zeta_est_tensor) + omega_zeta_est_tensor

# Tensornetrix prune (I think this is not significance based?)
pruned_model <- tensornetrics::prune(lnm,criterion='BIC')
pruned_model$get_partial_correlations()

# Setup empty matrix:
omega_zeta_est_tensor_pruned <- matrix(0,n_factor,n_factor)
rownames(omega_zeta_est_tensor_pruned) <- colnames(omega_zeta_est_tensor_pruned) <- latents

# Obtain estimates:
pcors <- pruned_model$partial_corr[,1]
labs <- do.call(rbind,strsplit(rownames(pruned_model$partial_corr),split="~"))

# Fill estimates (I know this is ugly...)
for (i in 1:nrow(labs)){
  omega_zeta_est_tensor_pruned[labs[i,1],labs[i,2]] <- pcors[i]
  omega_zeta_est_tensor_pruned[labs[i,2],labs[i,1]] <- pcors[i]
}

# Now also LASSO estimate:
lnm_lasso <-  tensor_lnm(data=data,
                   lasso=TRUE,
                   lambda=lambda_mod,
                   vars=vars,
                   latents=latents,
                   device=torch_device('gpu'))

# Find tuning:
optimal_value_of_v <- lasso_explore(lnm_lasso,epsilon = 0.01)

# Select mode:
lnm_lasso$lasso_fit(v=optimal_value_of_v)

# Obtain latent network
lnm_lasso$get_partial_correlations()

# Setup empty matrix:
omega_zeta_est_tensor_lasso <- matrix(0,n_factor,n_factor)
rownames(omega_zeta_est_tensor_lasso) <- colnames(omega_zeta_est_tensor_lasso) <- latents

# Obtain estimates:
pcors <- lnm_lasso$partial_corr[,1]
labs <- do.call(rbind,strsplit(rownames(lnm_lasso$partial_corr),split="~"))

# Fill estimates (I know this is ugly...)
for (i in 1:nrow(labs)){
  omega_zeta_est_tensor_lasso[labs[i,1],labs[i,2]] <- pcors[i]
  omega_zeta_est_tensor_lasso[labs[i,2],labs[i,1]] <- pcors[i]
}


# Compare:
library("qgraph")
layout(matrix(1:6,2,3,byrow=TRUE))

qgraph(omega_zeta,layout="circle",
       title="true latent network", 
       theme = "colorblind",
       labels = FALSE, 
       vsize = 20,
       color = "gray",
       mar = rep(6,4))
box("figure")

qgraph(omega_zeta_est_psychonetrics,layout="circle",
       title="psychonetrics (saturated)", 
       theme = "colorblind",
       labels = FALSE, 
       vsize = 20,
       color = "gray",
       mar = rep(6,4))
box("figure")

qgraph(omega_zeta_est_tensor,layout="circle",
       title="tensornetrics (saturated)", 
       theme = "colorblind",
       labels = FALSE, 
       vsize = 20,
       color = "gray",
       mar = rep(6,4))
box("figure")


qgraph(omega_zeta_est_psychonetrics_pruned,layout="circle",
       title="psychonetrics (pruned)", 
       theme = "colorblind",
       labels = FALSE, 
       vsize = 20,
       color = "gray",
       mar = rep(6,4))
box("figure")

qgraph(omega_zeta_est_tensor_pruned,layout="circle",
       title="tensornetrics (pruned)", 
       theme = "colorblind",
       labels = FALSE, 
       vsize = 20,
       color = "gray",
       mar = rep(6,4))
box("figure")

qgraph(omega_zeta_est_tensor_lasso,layout="circle",
       title="tensornetrics (lasso)", 
       theme = "colorblind",
       labels = FALSE, 
       vsize = 20,
       color = "gray",
       mar = rep(6,4))
box("figure")


