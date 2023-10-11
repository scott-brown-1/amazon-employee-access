#########################
### Imports and setup ###
#########################

library(tidyverse)
library(tidymodels)
library(embed) # for target encoding

prep_df <- function(df) {
  if('ACTION' %in% colnames(df)) {
    df <- df %>% mutate(ACTION = factor(ACTION))
  }
  return(df)
}

setup_train_recipe <- function(df, other_threshold = 0.001, form=ACTION~.){
  prelim_ft_eng <- recipe(form, data=df) %>%
    step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
    step_other(all_nominal_predictors(), threshold = other_threshold) %>% # combines categorical values that occur <5% into an "other" value
    step_dummy(all_nominal_predictors()) %>% # dummy variable encoding 
    step_lencode_bayes(all_nominal_predictors(), outcome = vars(ACTION)) # Target encoding
    #step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) #target encoding
  # also step_lencode_glm()
  # NOTE: some of these step functions are not appropriate to use together
  
  # Set up preprocessing
  prepped_recipe <- prep(prelim_ft_eng, new_data=df)
  
  return(prepped_recipe)
}
