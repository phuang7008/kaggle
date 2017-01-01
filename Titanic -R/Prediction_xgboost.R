# this script is used to run the prediction for the Titanic Dataset
library(readr)
library(xgboost)
library(dplyr)
library(Matrix)

# need to clean up all the variables from the previous exercises!
rm(list=ls())

# change the directoroy to point to the Titanic directory
setwd("G:/Tutorial/Kaggle Competitions/Titanic")

# read pre-processed data in ==> contains both train and test dataset
all_data = read_csv('final_train.csv')

# remove columns that is not going to be used for our analysis
# 'PassengerId' can not be removed as it is need for the output
removed_columns <- c('Deck', 'Name', 'Ticket', 'Surname')
all_data <- all_data[-c(which(colnames(all_data) %in% removed_columns))]
summary(all_data)

# fill the missing values for Cabin
all_data$Cabin <- substr(all_data$Cabin, 1, 1)
all_data$Cabin[is.na(all_data$Cabin)] <- 'N'

#all_data$Pclass <- as.numeric(all_data$Pclass)

# need to convert the following variables to factor variables
factor_variables <- c('Sex', 'Embarked', 'Title', 'Cabin',
                      'Surname', 'Family', 'FsizeD', 'Child', 'Mother')

all_data[factor_variables] <- lapply(all_data[factor_variables], 
                                     function(x) factor(x))

str(all_data)

# However, convert categorical data into factor is still not good enough
# need to convert them to the numerical data for xgboost using one-hot-encoding (ohe)
# one way is to use dummyVars(), another way is to use sparse.model.matrix()

# however, for any factors with 2 levels, we could use the following:
# make sure it is converted to factor before doing this.
#all_data$Sex <- ifelse(all_data$Sex == 'male', 1, 0)
#all_data$Sex <- as.numeric(all_data$Sex)

#all_data$Child <- ifelse(all_data$Child == 'Adult', 1, 0)
#all_data$Child <- as.numeric(all_data$Child)

#all_data$Mother <- ifelse(all_data$Mother == 'Mother', 1, 0)
#all_data$Mother <- as.numeric(all_data$Mother)

# for any factors with 3 levels, we could also try this to reduce the size of ohe
#all_data$Embarked <- ifelse(all_data$Embarked == 'C', 0, 
                            #ifelse(all_data$Embarked == 'Q', 1, 2))
#all_data$Embarked <- as.numeric(all_data$Embarked)

#all_data$FsizeD <- ifelse(all_data$FsizeD == 'large', 0, 
                          #ifelse(all_data$FsizeD == 'singleton', 1, 2))
#all_data$FsizeD <- as.numeric(all_data$FsizeD)

library(caret)
variables_to_convert <- c('Title', 'Cabin', 'Surname', 'Family', 'Pclass', 'Sex',
                          'Child', 'Mother', 'FsizeD', 'Embarked')

###########################################################################################
# Method 1
"
dmy <- dummyVars( ~ Title + Cabin + Family + Pclass + Sex + Child + Mother + FsizeD + Embarked, 
                  data = all_data)
all_data_ohe <- data.frame(predict(dmy, newdata = all_data))

# remove those columns with ohe
all_data <- all_data[-c(which(colnames(all_data) %in% variables_to_convert))]

# combine both data.frames
all_data = cbind(all_data, all_data_ohe)

# now need to separate the training and test dataset
train <- all_data[1:891,]
test  <- all_data[892:1309,]

trainX <- train[-c(which(colnames(train) %in% c('Survived', 'PassengerId')))]
testX  <- test[-c(which(colnames(train) %in% c('Survived', 'PassengerId')))]
dtrain <- as.matrix(trainX)
dlevel <- as.matrix(as.factor(train$Survived))
dtest  <- as.matrix(testX)
"
###########################################################################################
# Method 2
# now need to separate the training and test dataset
train <- all_data[1:891,]
test  <- all_data[892:1309,]
test$Survived <- 1

dtrain <- sparse.model.matrix(Survived~.-1, data = train)
dtest  <- sparse.model.matrix(Survived~.-1, data = test)

xgb_model <- xgboost(data  = dtrain,
                     #label = dlevel,
                     label = train$Survived,        # use label= train$Survived, for sparse one
                     #missing = NA,
                     eval.metric= 'logloss',        # model minimizes Root Mean Squared Error
                     objective= "binary:logistic",  #regression
                     #tuning parameters
                     max.depth= 10,            #Vary btwn 3-15
                     eta= 0.1,                #Vary btwn 0.1-0.3
                     #nthread = 2,             #Increase this to improve speed
                     subsample= 1,            #Vary btwn 0.8-1
                     colsample_bytree= 0.7,   #Vary btwn 0.3-0.8
                     lambda= 0.8,             #Vary between 0-3
                     alpha= 0.8,              #Vary between 0-3
                     min_child_weight= 1,     #Vary btwn 1-10
                     nround= 500               #Vary btwn 100-3000 based on max.depth, eta, 
                                              #                   subsample and colsample
                     )

# do the prediction
preds <- predict(xgb_model, dtest)
submission <- data.frame(PassengerId=test$PassengerId)
submission$Survived <- ifelse(preds>0.5, 1, 0) 

# save the prediction for submission
write.csv(submission, file="submission_xgb.csv", row.names=F)
