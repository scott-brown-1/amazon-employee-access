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

#doParallel::registerDoParallel(10)
cl <- makePSOCKcluster(15)
registerDoParallel(cl)

## Set up preprocessing
prepped_recipe <- setup_train_recipe(train, use_pca=T, pca_threshold=0.85)

## Bake recipe
bake(prepped_recipe, new_data=train)
bake(prepped_recipe, new_data=test)

#########################
## Fit Regression Model #
#########################

## Define model
rand_forest_model <- rand_forest(
  mtry = tune(),
  min_n =tune(),
  trees = 750
  ) %>%
  set_engine("ranger") %>%
  set_mode("classification")

## Define workflow
# Transform response to get different cutoff
rand_forest_wf <- workflow(prepped_recipe) %>%
  add_model(rand_forest_model)

## Grid of values to tune over
tuning_grid <- grid_regular(
  mtry(range=c(4,ncol(train))),
  min_n(),
  levels = 5)

## Split data for CV
folds <- vfold_cv(train, v = 3, repeats=1)

## Run the CV
cv_results <- rand_forest_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc))

## Find optimal tuning params
best_params <- cv_results %>%
  select_best("roc_auc")

## Fit workflow
final_wf <- rand_forest_wf %>%
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
vroom::vroom_write(output,'./outputs/rf_batch_predictions.csv',delim=',')

stopCluster(cl)
