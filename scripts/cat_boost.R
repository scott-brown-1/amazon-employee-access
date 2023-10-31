# TESTING ONLY, NOT FUNCTIONAL #

# #########################
# ### Imports and setup ###
# #########################
# 
# library(tidyverse)
# library(tidymodels)
# library(doParallel)
# library(treesnip)
# library(catboost)
# 
# #setwd('..')
# source('./scripts/amazon_analysis.R')
# PARALLEL <- T
# 
# #########################
# ####### Load Data #######
# #########################
# 
# ## Load data
# train <- prep_df(vroom::vroom('./data/train.csv'))
# test <- prep_df(vroom::vroom('./data/test.csv'))
# 
# #########################
# ## Feature Engineering ##
# #########################
# 
# set.seed(843)
# 
# ## parallel tune grid
# 
# if(PARALLEL){
#   cl <- makePSOCKcluster(10)
#   registerDoParallel(cl)
# }
# 
# ## Set up preprocessing
# prepped_recipe <- setup_train_recipe(train, encode=F, smote_K = 0, pca_threshold = 0)
# 
# ## Bake recipe
# bake(prepped_recipe, new_data=train)
# bake(prepped_recipe, new_data=test)
# 
# #########################
# ## Fit Regression Model #
# #########################
# ## Define model
# boost_model <- parsnip::boost_tree(
#   trees = 1,#1000,
#   min_n = 2,#tune(),
#   learn_rate = 0.01,#tune(),
#   tree_depth = 4#tune()
#   ) %>%
#   set_engine('catboost') %>%
#   set_mode('classification') %>%
#   translate()
# 
# boost_wf <- workflow(prepped_recipe) %>%
#   add_model(boost_model)
# 
# train <- train %>% mutate_if(is.numeric,as.factor)
# final_wf <- boost_wf %>%
#   fit(data = train)
# 
# ## Predict new y
# output <- predict(final_wf, new_data=test, type='prob') %>%
#   bind_cols(., test) %>%
#   rename(ACTION=.pred_1) %>%
#   select(id, ACTION)
# 
# #LS: penalty, then mixture
# vroom::vroom_write(output,'./outputs/cat_boost_preds-csv',delim=',')
# 
# if(PARALLEL){
#   stopCluster(cl)
# }
