#########################
### Imports and setup ###
#########################

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
## Feature Engineering ##data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABIAAAASCAYAAABWzo5XAAAAWElEQVR42mNgGPTAxsZmJsVqQApgmGw1yApwKcQiT7phRBuCzzCSDSHGMKINIeDNmWQlA2IigKJwIssQkHdINgxfmBBtGDEBS3KCxBc7pMQgMYE5c/AXPwAwSX4lV3pTWwAAAABJRU5ErkJggg==
#########################

set.seed(42)

## parallel tune grid
#cl <- makePSOCKcluster(15)
#registerDoParallel(cl)
doParallel::registerDoParallel(10)

## Set up preprocessing
prepped_recipe <- setup_train_recipe(train)

## Bake recipe
bake(prepped_recipe, new_data=train)
bake(prepped_recipe, new_data=test)

#########################
## Fit Regression Model #
#########################

## Define model
logistic_model <- logistic_reg() %>%
  set_engine("glm")

## Define workflow
# Transform response to get different cutoff
logistic_wf <- workflow() %>%
  add_recipe(prepped_recipe) %>%
  add_model(logistic_model)

## Fit workflow
final_wf <- logistic_wf %>%
  fit(data = train)

## Predict new rentals
y_pred <- predict(final_wf, new_data=test, type='prob')

# Create output df in Kaggle format
output <- data.frame(
  Id=test$id,
  Action=y_pred$.pred_1
)

vroom::vroom_write(output,'./outputs/logistic_predictions.csv',delim=',')

#stopCluster(cl)
