#########################
### Imports and setup ###
#########################

# NOTE: This script is optimized for running as a BATCH job

library(tidyverse)
library(tidymodels)
library(discrim)
library(doParallel)

setwd('..')
source('./scripts/amazon_analysis.R')
PARALLEL <- F

#########################
####### Load Data #######
#########################

## Load data
train <- prep_df(vroom::vroom('./data/train.csv'))
test <- prep_df(vroom::vroom('./data/test.csv'))

#########################
## Feature Engineering ## 
#########################

set.seed(42)

## parallel tune grid

if(PARALLEL){
  cl <- makePSOCKcluster(15)
  registerDoParallel(cl)
}

## Set up preprocessing
prepped_recipe <- setup_train_recipe(train)

## Bake recipe
bake(prepped_recipe, new_data=train)
bake(prepped_recipe, new_data=test)

#########################
## Fit Classifer Model ##
#########################

## Define model
bayes_model <- naive_Bayes(
  Laplace=tune(),
  smoothness=tune()) %>%
set_engine("naivebayes") %>%
set_mode("classification")

## Define workflow
bayes_wf <- workflow() %>%
  add_recipe(prepped_recipe) %>%
  add_model(bayes_model)

## Grid of values to tune over
tuning_grid <- grid_regular(
  Laplace(),
  smoothness(),
  levels = 5)

## Split data for CV
folds <- vfold_cv(train, v = 5, repeats=1)

## Run the CV
cv_results <- bayes_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc))

## Find optimal tuning params
best_params <- cv_results %>%
  select_best("roc_auc")

## Fit workflow
final_wf <- bayes_wf %>%
  finalize_workflow(best_params) %>%
  fit(data = train)

## Predict new y
output <- predict(final_wf, new_data=test, type='prob') %>%
  bind_cols(., test) %>%
  rename(ACTION=.pred_1) %>%
  select(id, ACTION)

#LS: penalty, then mixture
vroom::vroom_write(output,'./outputs/naive_bayes_predictions.csv',delim=',')

if(PARALLEL){
  stopCluster(cl)
}