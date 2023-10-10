#########################
### Imports and setup ###
#########################

library(tidyverse)
library(tidymodels)

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
CUTOFF <- 0.8
y_pred <- predict(final_wf, new_data=test, type='prob')
y_pred_class <- y_pred$.pred_1 > CUTOFF

# Create output df in Kaggle format
output <- data.frame(
  Id=test$id,
  Action=y_pred$.pred_1
)

vroom::vroom_write(output,'./outputs/logistic_predictions.csv',delim=',')
