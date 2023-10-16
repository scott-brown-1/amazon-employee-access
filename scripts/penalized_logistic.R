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
prepped_recipe <- setup_train_recipe(train)

## Bake recipe
bake(prepped_recipe, new_data=train)
bake(prepped_recipe, new_data=test)

#########################
## Fit Regression Model #
#########################

## Define model
pen_logit_model <- logistic_reg(
  mixture = tune(),
  penalty = tune()) %>%
  set_engine("glmnet")

## Define workflow
# Transform response to get different cutoff
pen_logic_wf <- workflow(prepped_recipe) %>%
  add_model(pen_logit_model)

## Grid of values to tune over10
tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 3)
## Split data for CV
folds <- vfold_cv(train, v = 3, repeats=1)

## Run the CV
cv_results <- pen_logic_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc))

## Find optimal tuning params
best_params <- cv_results %>%
  select_best("roc_auc")

## Fit workflow
final_wf <- pen_logic_wf %>%
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
vroom::vroom_write(output,'./outputs/penalized_logistic_predictions.csv',delim=',')

stopCluster(cl)
