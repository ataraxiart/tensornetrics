library(psych)
library(psychonetrics)




#Starwars dataset

data('StarWars')
lambda <- matrix(0, 10, 3)
lambda[1:4,1] <- 1
lambda[c(1,5:7),2] <- 1
lambda[c(1,8:10),3] <- 1

observedvars <- colnames(StarWars[,1:10])
latents <- c('Prequels','Originals','Sequels')

lnm <-  tensor_lnm(data=StarWars[1:10],lasso=F,lambda=lambda,vars = observedvars, latents= latents,device=torch_device('cpu'))
lnm$fit(verbose = T) 
pruned_model <- lnm %>% prune()
stepup_model <- pruned_model %>% stepup()



#bfi dataset

source(file="torch_lnm_helper_functions.R")


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
lnm$get_criterion_value('EBIC')$item()


pruned_model <- lnm %>% prune('EBIC')
pruned_model$get_criterion_value('EBIC')$item()
stepup_model <- pruned_model %>% stepup('EBIC')
stepup_model$get_criterion_value('EBIC')$item()


lnm_lasso <- tensor_lnm(data=bfi%>%na.omit(),lasso=T,lambda=lambda,vars = observedvars, latents= latents,device=torch_device('cpu'))

lasso_explore(lnm_lasso,epsilon = 0.01)

lnm_lasso <- tensor_lnm(data=bfi%>%na.omit(),lasso=T,lambda=lambda,vars = observedvars, latents= latents,device=torch_device('cpu'))
lnm_lasso$lasso_fit(v=28.07216)
lnm_lasso$get_partial_correlations(epsilon = 0.01)[[4]]
new_lnm <- tensor_lnm(data=bfi%>%na.omit(),lasso=F,lambda=lambda,vars = observedvars, latents= latents,device=torch_device('cpu'),
                      omega_psi_constraint_lst = lnm_lasso$get_partial_correlations(epsilon = 0.01)[[4]])

new_lnm$fit(verbose = F)
# Example Code to show the use of a custom loss function
new_lnm$get_partial_correlations()


lad_estimator <- function(sigma, mod){
  torch_sum(torch_abs(sigma - mod$cov_matrix))
}

lnm$forward()
lnm$cov_matrix

lnm <-  tensor_lnm(data=bfi%>%na.omit(),lambda=lambda,vars = observedvars, latents= latents,device=torch_device('cpu'),
                   custom_loss = lad_estimator)

lnm$custom_fit(lrate = 0.005, maxit = 500) #Cut off at 500 iterations, due to oscillating behavior
lnm$custom_fit(lrate = 0.0001, maxit = 100) #Usage of smaller learning rate for convergence
lnm$get_partial_correlations()



