#########################
### Imports and setup ###
#########################

# NOTE: This script is optimized for running as a BATCH job

library(tidyverse)
library(tidymodels)
library(doParallel)

setwd('..')
source('./scripts/amazon_analysis.R')

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

cl <- makePSOCKcluster(15)
registerDoParallel(cl)

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
    trees = tune(), #250,
    prior_terminal_node_coef = tune(), #0.75,
    prior_terminal_node_expo = tune()  #1.75,
  ) %>% 
  set_engine("dbarts") %>% 
  set_mode("classification")

## Define workflow
# Transform response to get different cutoff
bart_workflow <-
  workflow(prepped_recipe) %>%
  add_model(bart_model)

## Grid of values to tune over
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

## Predict new rentals
y_pred <- predict(final_wf, new_data=test, type='prob')

# Create output df in Kaggle format
output <- data.frame(
  Id=test$id,
  Action=y_pred$.pred_1
)

#LS: penalty, then mixture
vroom::vroom_write(output,'./outputs/bart_predictions.csv',delim=',')

stopCluster(cl)
