#########################
### Imports and setup ###
#########################

library(tidyverse)
library(tidymodels)
library(doParallel)
library(bonsai)

#setwd('./amazon-employee-access')
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

set.seed(843)

## parallel tune grid

if(PARALLEL){
  cl <- makePSOCKcluster(5)
  registerDoParallel(cl)
}

## Set up preprocessing
prepped_recipe <- setup_train_recipe(train, encode = F, smote_K = 0, pca_threshold = 0)

## Bake recipe
bake(prepped_recipe, new_data=train)
bake(prepped_recipe, new_data=test)

#########################
## Fit Regression Model #
#########################

boost_model <- boost_tree(
  trees = 225, #tune(), #100
  tree_depth = 6, #tune(), #1,
  learn_rate = 0.1,#tune(), #0.1,
  mtry = 2,#tune(), #3,
  min_n = 20, #tune(), #20,
  loss_reduction = 0.1#tune(), #0
  ) %>% 
  set_engine("lightgbm") %>% 
  set_mode("classification")

## Define workflow
# Transform response to get different cutoff
boost_wf <- workflow(prepped_recipe) %>%
  add_model(boost_model)

## Grid of values to tune over
tuning_grid <- grid_regular(
  trees(),
  tree_depth(),
  learn_rate(),
  mtry(range=c(3,ncol(train))),
  min_n(),
  loss_reduction(),
  levels = 5)

## Split data for CV
folds <- vfold_cv(train, v = 5, repeats=1)

# Run the CV
# cv_results <- boost_wf %>%
#   tune_grid(resamples=folds,
#             grid=tuning_grid,
#             metrics=metric_set(roc_auc))

# Find optimal tuning params
# best_params <- cv_results %>%
#   select_best("roc_auc")

# print(best_params)

# Fit workflow
final_wf <- boost_wf %>%
  #finalize_workflow(best_params) %>%
  fit(data = train)

## Predict new y
output <- predict(final_wf, new_data=test, type='prob') %>%
  bind_cols(., test) %>%
  rename(ACTION=.pred_1) %>%
  select(id, ACTION)

#LS: penalty, then mixture
vroom::vroom_write(output,'./outputs/light_gbm_preds.csv',delim=',')

if(PARALLEL){
  stopCluster(cl)
}
