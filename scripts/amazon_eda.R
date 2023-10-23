#########################
### Imports and setup ###
#########################

# SHOULD HAVE 112 ANSWERS

# Import packages
library(tidyverse)
library(DataExplorer)
library(ggmosaic)

## Load data
amazon <- vroom::vroom('./data/train.csv')

###########################
####### Examine Data ######
###########################

## Exmaine dataframe and check data types; check shape
glimpse(amazon)
View(amazon)

## Fix dtypes: change all cols to factors
## Check number of levels in each factor
for (col in colnames(amazon)){
  amazon[[col]] <- factor(amazon[[col]])
  print(paste0(col,': ',length(unique(amazon[[col]]))))
}

## Check categorical vs discrete vs continuous factors
intro_plot <- plot_intro(amazon)
intro_plot

###########################
### Check Missing Values ##
###########################

# View missing values by feature
plot_missing(amazon)

# Count total missing values
sum(sum(is.na(amazon)))

###########################
## Visually examine data ##
###########################

ggplot(amazon, aes(x=ROLE_FAMILY), stat=count) +
  geom_bar()

###########################
### Examine response var ##
###########################

response_hist <- ggplot(amazon, aes(x=ACTION)) +
  geom_histogram(binwidth=0.5, fill='skyblue') +
  labs(title="Histogram of Action Decision",x="Action", y = "Count of observations")