#########################
### Imports and setup ###
#########################

library(tidyverse)
library(tidymodels)
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

set.seed(2003)

## parallel tune grid

if(PARALLEL){
  cl <- makePSOCKcluster(10)
  registerDoParallel(cl)
}

## Set up preprocessing
prepped_recipe <- setup_train_recipe(train, smote_K = 5, pca_threshold = 0.8)

## Bake recipe
bake(prepped_recipe, new_data=train)
bake(prepped_recipe, new_data=test)

#########################
## Fit Regression Model #
#########################

## Define model
rand_forest_model <- rand_forest(
  mtry = tune(),
  min_n = tune(),
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
  mtry(range=c(1,5)),#(range=c(4,ncol(train))),
  min_n(),
  levels = 5)

## Split data for CV
folds <- vfold_cv(train, v = 5, repeats=1)

## Run the CV
cv_results <- rand_forest_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc))

## Find optimal tuning params
best_params <- cv_results %>%
  select_best("roc_auc")

tryCatch(
  expr = {
    print(best_params)
  },
  error = function(e){ 
    print('Error caught')
    print(e)
  })

## Fit workflow
final_wf <- rand_forest_wf %>%
  finalize_workflow(best_params) %>%
  fit(data = train)

final_wf <- rand_forest_wf %>%
  fit(data = train)

## Predict new y
output <- predict(final_wf, new_data=test, type='prob') %>%
  bind_cols(., test) %>%
  rename(ACTION=.pred_1) %>%
  select(id, ACTION)

#LS: penalty, then mixture
vroom::vroom_write(output,'./outputs/rand_forest_predictions.csv',delim=',')

if(PARALLEL){
  stopCluster(cl)
}
