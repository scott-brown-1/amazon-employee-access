library(tidyverse)
library(tidymodels)
library(discrim)
source('./scripts/amazon_analysis.R')
## Create recipe here
## This ommitted section is the same as other working scripts
## Load data
train <- prep_df(vroom::vroom('./data/train.csv'))
test <- prep_df(vroom::vroom('./data/test.csv'))

#########################
## Feature Engineering ##
#########################

set.seed(42)

## parallel tune grid

registerDoParallel(10)

## Set up preprocessing
prepped_recipe <- setup_train_recipe(train)

## Bake recipe
bake(prepped_recipe, new_data=train)
bake(prepped_recipe, new_data=test)

bayes_model <- parsnip::naive_Bayes(
  Laplace=tune(),
  smoothness=tune()) %>%
  set_engine("naivebayes") %>%
  set_mode("classification")

## Define workflow
bayes_wf <- workflow(prepped_recipe) %>%
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
y_pred <- predict(final_wf, new_data=test, type='prob')
