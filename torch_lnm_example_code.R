

remotes::install_github("ataraxiart/tensornetrics")

library(psych)
library(psychonetrics)
library(torch)
library(tensornetrics)
library(dplyr)
library(qgraph)

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
#variance of latents set to 1
lnm <-  tensor_lnm(data=StarWars[1:10],lasso=F,lambda=lambda,vars = observedvars, latents= latents,device=torch_device('cpu'),
                   identification = "variance")

#Or alternatively identify via loadings, default is via variance
#first factor loading of each latent variable is set to 1
lnm <-  tensor_lnm(data=StarWars[1:10],lasso=F,lambda=lambda,vars = observedvars, latents= latents,device=torch_device('cpu'),
                   identification = "loadings")


#Fit using standard log-likelihood fit function
lnm$fit(verbose = T,lr=0.05) 


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
#Use tensornetrics:: to avoid clash issues with psychonetrics
stepdown_model <- stepdown(lnm,criterion='BIC')
stepup_model <- stepdown_model %>% stepup(criterion = 'BIC')

#View partial correlations/loadings of stepup_model

stepup_model$get_partial_correlations()
stepup_model$get_loadings()

#Get criterion of model:
lnm$get_criterion_value('BIC',gamma=0)
stepup_model$get_criterion_value('EBIC',gamma=0)
stepup_model$get_fit_metrics()

lnm$get_fit_metrics()

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
lnm$fit(verbose=T)
lnm$get_partial_correlations()

#We can remove the insignificant partial correlations via the prune function
lnm <- prune(lnm)
lnm$get_partial_correlations()

qgraph(lnm$omega_psi,layout="circle",
       title="Big 5 Personality Traits", 
       theme = "colorblind",
       labels = latents, 
       vsize = 20,
       color = "gray",
       mar = rep(6,4))
box("figure")


# Example Code to show the use of a custom loss function
# We define a custom loss function, LAD estimator with inputs, sigma 
#(the model covariance matrix) and mod (the lnm module to access its sample covariance attribute)

lad_estimator <- function(sigma, data){
  get_cov_matrix <- function(data) {
    data_matrix <- as_array(data)
    if (!is.null(data_matrix )) {
      return(cov(data_matrix , use = "pairwise.complete.obs"))  # Handles NAs if present
    } else {
      stop("data_matrix is NULL")
    }
  }
  cov_matrix <- get_cov_matrix(data)
  torch_sum(torch_abs(sigma - cov_matrix))
}

lnm_custom <-  tensor_lnm(data=bfi%>%na.omit(),lambda=lambda,vars = observedvars, latents= latents,device=torch_device('cpu'),
                   custom_loss = lad_estimator)


lad_estimator(lnm$sigma,lnm$data) #In comparison, lad_estimator loss function for when using standard fit
                            # function is about 73.461

lnm_custom$custom_fit()
lnm_custom$get_partial_correlations()


#Using LASSO to select for pairs of latent variables to include in our network

#Instantiate an lnm module but set lasso to TRUE
lnm_lasso <- tensor_lnm(data=bfi%>%na.omit(),lasso=TRUE,lambda=lambda,vars = observedvars, latents= latents,device=torch_device('cpu'))

#We use lasso_explore to perform lasso
#By default algo chooses among 30 values from 0.01 to 100 on a logscale 
#We set cutoff for partial correlations to be 0.0001
optimal_value_of_v <- lasso_explore(lnm_lasso,epsilon = 0.0001,v_values = pracma::logspace(log10(100), log10(10000), 30))


#Here, we get a list containing value of v selected and the constraints for omega_psi
print(optimal_value_of_v) 

#Fit the model selected by lasso 
lnm_lasso <- tensor_lnm(data=bfi%>%na.omit(),lasso=F,lambda=lambda,vars = observedvars, latents= latents,device=torch_device('cpu'),
                        omega_psi_constraint_lst = optimal_value_of_v[[2]],identification = 'variance')
lnm_lasso$fit(verbose = T)

#partial correlations of new model
lnm_lasso$get_partial_correlations()

#comparing criterion of models:
lnm$get_criterion_value('BIC')$item()
lnm_lasso$get_criterion_value('BIC')$item()

#comparing fit metrics of models like CFI, TLI and RMSEA:
lnm$get_fit_metrics() #model obtained after pruning
lnm_lasso$get_fit_metrics() #model obtained after lasso


