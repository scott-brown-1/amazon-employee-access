#########################
### Imports and setup ###
#########################

library(tidyverse)
library(tidymodels)
library(embed) # for target encoding

# SHOULD HAVE 112 ANSWERS

setup_train_recipe <- function(train, other_threshold = 0.01, form=ACTION~.){
  prelim_ft_eng <- recipe(form, data=train) %>%
    step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
    step_other(all_nominal_predictors(), threshold = other_threshold) %>% # combines categorical values that occur <5% into an "other" value
    step_dummy(all_nominal_predictors()) %>% # dummy variable encoding
    step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) #target encoding
  # also step_lencode_glm() and step_lencode_bayes()
  # NOTE: some of these step functions are not appropriate to use together
  
  # Set up preprocessing
  prepped_recipe <- prep(prelim_ft_eng, new_data=train)
  
  return(prepped_recipe)
}

###########################
####### Examine Data ######
###########################

## Load data
train <- vroom::vroom('./data/train.csv')

# # Set up preprocessing
prepped_recipe <- setup_train_recipe(train)

# Bake recipe
dim(bake(prepped_recipe, new_data=train))

