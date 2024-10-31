library(psychonetrics)
library(dplyr)
library(qgraph)





#RNM on self-esteem dataset

#Using a two-factor model 

data <- read.delim("C:/Users/samue/Documents/Masters/Sem 1/Sacha Work/R codes/selfEsteem.txt",sep='\t',header=TRUE)
items <- paste0("Q",1:10)
trainData <- data[c(TRUE,FALSE),]
testData <- data[c(FALSE,TRUE),]

# Factor loadings matrix (two factors):
Lambda <- matrix(0,10,2)
Lambda[c(1,2,4,6,7),1] <- 1 # Positive items
Lambda[c(3,5,8,9,10),2] <- 1 # Negative items

# Fit 2-factor models:
twofac_train <- psychonetrics::rnm(trainData, vars = items, estimator = "FIML",
                    lambda = Lambda,identification = 'variance') %>% runmodel




# Fit 2-factor RNM to train data:
RNM_twofac_train <- twofac_train %>% 
  psychonetrics::stepup(criterion = "bic",verbose = FALSE) %>% 
  modelsearch(verbose = FALSE)

# Fit 2-factor  RNM to test data:
structure <- 1*(getmatrix(RNM_twofac_train,"omega_epsilon")!=0)
RNM_twofac_test <- psychonetrics::rnm(testData, vars = items, estimator = "FIML",
                       lambda = Lambda, omega_epsilon = structure,identification='variance') %>% 
  runmodel

# Two-factor RNM:
# Obtain residual network:

residnet_test <- getmatrix(RNM_twofac_test, "kappa_epsilon")



# Obtain factor loadings:
factorloadings <- getmatrix(RNM_twofac_test, "lambda")

# Obtain correlations:
factorCors <- getmatrix(RNM_twofac_test, "sigma_zeta")

layout(t(1:2))
qgraph(residnet_test, theme = "colorblind", layout = "spring",
       title = "Residual network", vsize = 8)
qgraph.loadings(factorloadings, theme = "colorblind", model = "reflective",
                title = "Factor loadings", vsize = c(8,13), asize = 5,
                factorCors = factorCors)




#RNM on bfi  dataset

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



























