library(psychonetrics)
library(dplyr)
data("bfi")

# Extraversion and Neuroticism items:
data <- bfi[,11:20]
latents <- c("extraversion","neuroticism")
lambda <- matrix(0,10,2)
lambda[1:5,1] <- lambda[6:10,2] <- 1

# RNM model:
mod_rnm <- rnm(data, lambda = lambda, estimator = "FIML") %>% 
  runmodel %>% 
  prune %>% 
  modelsearch

parameters(mod_rnm)

# RNM with only a extraversion factor:
lambda <- matrix(rep(1:0,each=5),ncol=1)
latents <- "extraversion"
mod_rnm_extraversion <- rnm(data, lambda = lambda, estimator = "FIML") %>% 
  runmodel %>% 
  prune %>% 
  modelsearch


# RNM with only a neuroticism factor:
lambda <- matrix(rep(0:1,each=5),ncol=1)
latents <- "neuroticism"
mod_rnm_neuroticism <- rnm(data, lambda = lambda, estimator = "FIML") %>% 
  runmodel %>% 
  prune %>% 
  modelsearch