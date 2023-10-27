#########################
### Imports and setup ###
#########################

library(tidyverse)
library(tidymodels)
library(embed) # for target encoding
library(themis) # for smote

prep_df <- function(df) {
  if('ACTION' %in% colnames(df)) {
    df <- df %>% mutate(ACTION = factor(ACTION))
  }
  return(df)
}

## TODO: tune recipes!
setup_train_recipe <- function(df,other_threshold = 0.01, 
                               smote_K = 10, pca_threshold = 0.85){
  
  ############
  ### SAVE ###
  ############
  # prelim_ft_eng <- recipe(form, data=df) %>%
  #   step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  #   step_other(all_nominal_predictors(), threshold = other_threshold) %>% # combines categorical values that occur <5% into an "other" value
  #   step_dummy(all_nominal_predictors()) %>% # dummy variable encoding 
  #   step_lencode_bayes(all_nominal_predictors(), outcome = vars(ACTION)) # Target encoding
  # NOTE: some of these step functions are not appropriate to use together
  # also step_lencode_glm()
  ############
  
  prelim_ft_eng <- recipe(ACTION~., data=df) %>%
    step_mutate_at(all_numeric_predictors(), fn = factor) %>%
    step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
    step_normalize(all_numeric_predictors())
  
  ## Dimension reduce with principal component analysis if pca_threshold > 0
  if(pca_threshold > 0){
    prelim_ft_eng <- prelim_ft_eng %>%
      step_pca(all_predictors(), threshold=pca_threshold)
  }
  
  ## SMOTE upsample if K nearest neighbors > 0
  if(smote_K > 0){
    prelim_ft_eng <- prelim_ft_eng %>% step_smote(all_outcomes(), neighbors = smote_K)
  }
  
  # Set up preprocessing
  prepped_recipe <- prep(prelim_ft_eng, new_data=df)
  
  return(prepped_recipe)
}
