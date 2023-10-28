#########################
### Imports and setup ###
#########################

library(tidyverse)
library(tidymodels)
library(doParallel)
library(stacks)

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

set.seed(2003)

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
## PREPARE BASE MODELS ##
#########################

## Split data for CV
folds <- vfold_cv(train, v = 5, repeats=1)

## Create a control grid
untuned_model <- control_stack_grid() #If tuning over a grid

#### BART ####

## Define model
bart_model <- 
  parsnip::bart(
    trees = tune(),
    prior_terminal_node_coef = 0.75,#tune(), #0.75,
    prior_terminal_node_expo = 1.75#tune()  #1.75,
  ) %>% 
  set_engine("dbarts") %>% 
  set_mode("classification")

## Define workflow
bart_wf <-
  workflow(prepped_recipe) %>%
  add_model(bart_model)

## Grid of values to tune over
bart_grid <- grid_regular(
  trees(),
  levels = 5
)

## Perform parameter tuning
tuned_bart_models <- bart_wf %>%
  tune_grid(
    resamples=folds,
    grid=bart_grid,
    metrics = metric_set(roc_auc),
    control = untuned_model)

#### NAIVE BAYES ####

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
bayes_grid <- grid_regular(
  Laplace(),
  smoothness(),
  levels = 5)

## Perform parameter tuning
tuned_bayes_models <- bayes_wf %>%
  tune_grid(
    resamples=folds,
    grid=bayes_grid,
    metrics = metric_set(roc_auc),
    control = untuned_model)

#########################
# Stack models together #
#########################

# Create meta learner
model_stack <- stacks() %>%
  add_candidates(tuned_bart_models) %>%
  add_candidates(tuned_bayes_models)

# Fit meta learner
fitted_stack <- model_stack %>%
  blend_predictions() %>% # This is a Lasso (L1) penalized reg model
  fit_members()

print(head(as_tibble(fitted_stack)))

## Predict new y
output <- predict(fitted_stack, new_data=test, type='prob') %>%
  bind_cols(., test) %>%
  rename(ACTION=.pred_1) %>%
  select(id, ACTION)

vroom::vroom_write(output,'./outputs/stacked_predictions.csv',delim=',')

if(PARALLEL){
  stopCluster(cl)
}
