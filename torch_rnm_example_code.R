remotes::install_github("ataraxiart/tensornetrics")


library(psychonetrics)
library(psych)
#library(tensornetrics)
library(dplyr)

#RNM on self-esteem dataset

lambda <- matrix(0,10,2)
lambda[c(1,2,4,6,7),1] <- 1 # Positive items
lambda[c(3,5,8,9,10),2] <- 1 # Negative items
file_path <- system.file( "selfEsteem.txt", package = "tensornetrics")
data <- read.table(file_path,header =TRUE)

vars <- paste0("Q",1:10)
trainData <- data[c(TRUE,FALSE),]
testData <- data[c(FALSE,TRUE),]
latents <- c('l1','l2')

#Instantiate and fit rnm module 
#variance of latents set to 1
rnm <- tensor_rnm(data=trainData,lambda,vars=vars,latents=latents,
                  lasso=FALSE,identification = 'variance')
rnm$fit(verbose = T)
rnm$get_loadings()

#Or alternatively identify via loadings, default is via variance
#first factor loading of each latent variable is set to 1
rnm <- tensor_rnm(data=trainData,lambda,vars=vars,latents=latents,
                  lasso=FALSE,identification = 'loadings')
rnm$fit(verbose = T)
rnm$get_loadings()

#By default/initially, no partial correlations are in the model/
#omega_theta matrix is just a zero matrix
#We have to perform model selection via lasso 
rnm$get_partial_correlations()


#Using LASSO to select for residual partial correlations to include in our network

#Instantiate an rnm module but set lasso to TRUE

rnm <- tensor_rnm(data=trainData,lambda,vars=vars,latents=latents,
                  lasso=TRUE,identification = 'loadings')

#We use lasso_explore to perform lasso
#By default algo chooses among 30 values from 0.01 to 100 on a logscale 
#We try from 10000 to 100000 on a logscale here
#We set cutoff for partial correlations to be 0.0001
optimal_value_of_v <- lasso_explore(rnm,epsilon = 0.0001,v_values = pracma::logspace(log10(10000), log10(100000), 30))

#Here, we get a list containing value of v selected and the free parameters for omega_theta
#Note that the lasso_explore function for rnm functions work differently from lnm functions
#since parameters returned here are the free params and not the constraints
print(optimal_value_of_v) 

#Fit the model selected by lasso 
model_selected_by_lasso<- tensor_rnm(data=trainData,lambda,vars=vars,latents=latents,
                                     lasso=FALSE,identification = 'variance',
                                     omega_theta_free_lst = optimal_value_of_v[[2]] )

model_selected_by_lasso$fit(verbose = TRUE)

#partial correlations of new model
model_selected_by_lasso$get_partial_correlations()

#other things you can get
model_selected_by_lasso$get_loadings()
model_selected_by_lasso$get_psi()



#We can remove the insignificant partial correlations via the prune function
final_model <- tensornetrics::prune(model_selected_by_lasso)
final_model$get_partial_correlations()

#Get final model criterion value and fit metrics
final_model$get_criterion_value("BIC")
final_model$get_fit_metrics()

#Test our omega_theta matrix obtained on our test data
formatted_omega_mat <-format_omega_theta(final_model$omega_theta)
model_on_test <- tensor_rnm(data=testData,lambda,vars=vars,latents=latents,
                                     lasso=FALSE,identification = 'variance',
                                     omega_theta_constraint_lst = formatted_omega_mat )

model_on_test$fit(verbose = TRUE)
model_on_test$get_fit_metrics() #fits v well





#Possible problems one might encounter when using LASSO

#1) Insufficient DF
#The best model returned here is still underidentified so the penalty parameters to consider
#should be increased even more
#Theoretically, since the objective function is convex, the solution returned by lasso
#each time is unique but the number of params estimated to be below the threshold might be
#insufficient
rnm_insufficient_df <- tensor_rnm(data=trainData,lambda,vars=vars,latents=latents,
                  lasso=TRUE,identification = 'loadings')
optimal_value_of_v_insufficient <- 
  lasso_explore(rnm_insufficient_df,epsilon = 0.0001,v_values = pracma::logspace(log10(0.01), log10(100), 30))



#2) Heywood Cases
#We fit the model with the constraints returned by lasso but when we attempt to calculate
#the standard errors, we get negative variances
#A possible reason is that the model selected by lasso might still be a misspecification
#and alternate values of the penalty value v should be considered again
rnm_heywood <- tensor_rnm(data=trainData,lambda,vars=vars,latents=latents,
                  lasso=TRUE,identification = 'loadings')
optimal_value_of_v_heywood <- lasso_explore(rnm_heywood ,epsilon = 0.0001,v_values = pracma::logspace(log10(100), log10(10000), 30))
rnm_heywood  <- tensor_rnm(data=trainData,lambda,vars=vars,latents=latents,
                  lasso=FALSE,identification = 'loadings',omega_theta_free_lst = optimal_value_of_v_heywood[[2]])
rnm_heywood$fit(verbose=T)
rnm_heywood$get_partial_correlations()


#Another example: RNM on bfi dataset

data("bfi")


# Extraversion and Neuroticism items:
data <- bfi[,11:20]
latents <- c("extraversion","neuroticism")
lambda <- matrix(0,10,2)
lambda[1:5,1] <- lambda[6:10,2] <- 1
vars <- c(paste0("E",seq(1,5)),paste0("N",seq(1,5)))

# RNM model:

#Whenever we run lasso, we must set lasso to true: 
rnm <- tensor_rnm(data=data%>%na.omit(),lambda,vars=vars,latents=latents,
                  lasso=TRUE,identification = 'loadings')

selected_v_params <- lasso_explore(rnm,epsilon=0.0001,lrate = 0.01,
                                       v_values=pracma::logspace(log10(1000), log10(100000), 5))

model_selected_by_lasso<- tensor_rnm(data=data%>%na.omit(),lambda,vars=vars,latents=latents,
                                     lasso=FALSE,identification = 'loadings',
                                     omega_theta_free_lst = selected_v_params[[2]] )

#Fit model chosen by lasso
model_selected_by_lasso$fit(verbose = TRUE)
model_selected_by_lasso$get_partial_correlations()

#Prune to remove insignificant partial correlations
final_model <- tensornetrics::prune(model_selected_by_lasso)
final_model$get_partial_correlations()




















