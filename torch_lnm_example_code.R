

devtools::install_github('ataraxiart/tensornetrics')
library(tensornetrics)
library(psych)
library(psychonetrics)
library(torch)


#Starwars dataset
data('StarWars')

#Get Design Matrix for lambda
lambda <- matrix(0, 10, 3)
lambda[1:4,1] <- 1
lambda[c(1,5:7),2] <- 1
lambda[c(1,8:10),3] <- 1

#Indicate names of observed and latent variables
observedvars <- colnames(StarWars[,1:10])
latents <- c('Prequels','Originals','Sequels')

#Instantiate lnm module
lnm <-  tensor_lnm(data=StarWars[1:10],lasso=F,lambda=lambda,vars = observedvars, latents= latents,device=torch_device('cpu'))

#Fit using standard log-likelihood fit function
lnm$fit(verbose = T) 

#Access partial correlations
lnm$get_partial_correlations()
#After calling get_partial_correlations, correlations can be accessed as lnm$partial_corr
print(lnm$partial_corr)


#Access factor loadings
lnm$get_loadings()
#After calling get_loadings, correlations can be accessed as lnm$loadings
print(lnm$loadings)


#Access residuals
lnm$get_residuals()
#After calling residuals, correlations can be accessed as lnm$loadings
print(lnm$residuals)


#Perform stepwise procedures, takes a while!
pruned_model <- prune(lnm,criterion='BIC')
stepup_model <- pruned_model %>% stepup(criterion = 'BIC')

#View partial correlations/loadings of stepup_model

stepup_model$get_partial_correlations()
stepup_model$get_loadings()

#Get criterion of model:
lnm$get_criterion_value('EBIC',gamma=0)
stepup_model$get_criterion_value('EBIC',gamma=0)

#############################################################################

#bfi dataset

lambda <- matrix(0, 25, 5)
lambda[1:5,1] <- 1
lambda[6:10,2] <- 1
lambda[11:15,3] <- 1
lambda[16:20,4] <- 1
lambda[21:25,5] <- 1

latents <- c('Agreeableness','Conscientiousness','Extraversion','Neuroticism','Openness')
observedvars <- colnames(bfi)[1:25]

lnm <-  tensor_lnm(data=bfi%>%na.omit(),lasso=F,lambda=lambda,vars = observedvars, latents= latents,device=torch_device('cpu'))
lnm$fit(verbose=F)
lnm$get_partial_correlations()


# Example Code to show the use of a custom loss function
# We define a custom loss function, LAD estimator with inputs, sigma 
#(the model covariance matrix) and mod (the lnm module to access its sample covariance attribute)

lad_estimator <- function(sigma, mod){
  torch_sum(torch_abs(sigma - mod$cov_matrix))
}

lnm_custom <-  tensor_lnm(data=bfi%>%na.omit(),lambda=lambda,vars = observedvars, latents= latents,device=torch_device('cpu'),
                   custom_loss = lad_estimator)

lnm_custom$custom_fit(lrate = 0.005, maxit = 500) #Cut off at 500 iterations, due to oscillating behavior
lnm_custom$custom_fit(lrate = 0.0001, maxit = 200) #Usage of smaller learning rate for convergence
#Final value of lad_estimator loss function is about 68

lad_estimator(lnm$sigma,lnm) #In comparison, lad_estimator loss function for when using standard fit
                            # function is about 73.461


#Using LASSO to select for pairs of latent variables to include in our network

#Instantiate an lnm module but set lasso to TRUE
lnm_lasso <- tensor_lnm(data=bfi%>%na.omit(),lasso=TRUE,lambda=lambda,vars = observedvars, latents= latents,device=torch_device('cpu'))

#Set optimal value of v where algo chooses among 30 values from 0.01 to 100 on a logscale by default
#We set cutoff for partial correlations to be 0.01
#Under the hood the lasso_explore function uses just 1 lnm module to search for v so information
#about which partial correlations are set to 0 are lost with each new iteration
optimal_value_of_v <- lasso_explore(lnm_lasso,epsilon = 0.01)


#Refit lasso model to get the constraints (partial correlations now set to 0) 
lnm_lasso <- tensor_lnm(data=bfi%>%na.omit(),lasso=T,lambda=lambda,vars = observedvars, latents= latents,device=torch_device('cpu'))
lnm_lasso$lasso_fit(v=optimal_value_of_v)

#When lasso is set to T, the function get_partial_correlations will return a list
# 1-Non-zero Partial Corr, 2-Num of Partial Corr removed, 3-Names of Partial Corr removed,
# 4-Constraint List we can use to fit the new model

omega_psi_constraint_lst <- lnm_lasso$get_partial_correlations(epsilon = 0.01)[[4]]


#Refit new model with the constraints 
new_lnm <- tensor_lnm(data=bfi%>%na.omit(),lasso=F,lambda=lambda,vars = observedvars, latents= latents,device=torch_device('cpu'),
                      omega_psi_constraint_lst = omega_psi_constraint_lst)

new_lnm$fit(verbose = F)

#partial correlations of new model
new_lnm$get_partial_correlations()

#comparing criterion of models:
lnm$get_criterion_value('EBIC')
new_lnm$get_criterion_value('EBIC')

