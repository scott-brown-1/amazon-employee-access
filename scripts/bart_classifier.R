#########################
### Imports and setup ###
#########################

# NOTE: This script is optimized for running as a BATCH job

library(tidyverse)
library(tidymodels)
library(doParallel)

setwd('..')
source('./scripts/amazon_analysis.R')
PARALLEL <- T

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
  cl <- makePSOCKcluster(10)
  registerDoParallel(cl)
}

## Set up preprocessing
prepped_recipe <- setup_train_recipe(train)

## Bake recipe
bake(prepped_recipe, new_data=train)
bake(prepped_recipe, new_data=test)

#########################
## Fit Regression Model #
#########################

# Define model
bart_model <- 
  parsnip::bart(
    trees = 250,
    prior_terminal_node_coef = 0.75,#tune(), #0.75,
    prior_terminal_node_expo = 1.75#tune()  #1.75,
  ) %>% 
  set_engine("dbarts") %>% 
  set_mode("classification")

## Define workflow
# Transform response to get different cutoff
bart_workflow <-
  workflow(prepped_recipe) %>%
  add_model(bart_model)

# Grid of values to tune over
tuning_grid <- grid_regular(
  trees(),
  prior_terminal_node_coef(),
  prior_terminal_node_expo(),
  levels = 1#7#0 #10^2 tuning possibilities
)

## Split data for CV
folds <- vfold_cv(train, v = 2, repeats=1)

## Run the CV
cv_results <- bart_workflow %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc))

## Find optimal tuning params
best_params <- cv_results %>%
  select_best("roc_auc")

## Fit workflow
final_wf <- bart_workflow %>%
  finalize_workflow(best_params) %>%
  fit(data = train)

## Predict new y
output <- predict(final_wf, new_data=test, type='prob') %>%
  bind_cols(., test) %>%
  rename(ACTION=.pred_1) %>%
  select(id, ACTION)

#LS: penalty, then mixture
vroom::vroom_write(output,'./outputs/bart_predictions.csv',delim=',')

if(PARALLEL){
  stopCluster(cl)
}
