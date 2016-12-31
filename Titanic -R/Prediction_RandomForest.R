# this script is used to run the prediction for the Titanic Dataset
library(readr)
library(randomForest)

# need to clean up all the variables from the previous exercises!
rm(list=ls())

# change the directoroy to point to the Titanic directory
setwd("G:/Tutorial/Kaggle Competitions/Titanic")

# read pre-processed data in ==> contains both train and test dataset
all_data = read_csv('final_train.csv')
summary(all_data)

# need to convert the following variables to factor variables
factor_variables <- c('Pclass', 'PassengerId', 'Sex', 'Embarked', 'Title', 
                      'Surname', 'Family', 'FsizeD', 'Child', 'Mother')

all_data[factor_variables] <- lapply(all_data[factor_variables], 
                                     function(x) as.factor(x))

str(all_data)

# now need to separate the training and test dataset
train <- all_data[1:891,]
test  <- all_data[892:1309,]


# set random seed
set.seed(175)

# build the model. Here I am only going to use those variables that I think are 
# important for the survival
rf_model <- randomForest(factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch +
                           Fare + Embarked + Title + FsizeD + Child + 
                           Mother, 
                         data = train)

# let's find the model error
plot(rf_model, ylim=c(0,0.36))
legend("topright", colnames(rf_model$err.rate), col=1:3, fill=1:3)

# from the graph, we see that the model is much better prediction death than survival
# The average error rate is depicted by the black line

# Let's look at relative variable importance by plotting the mean decrease 
# in Gini calculated across all trees.
importances <- importance(rf_model)
varImportance <- data.frame(Variables = row.names(importances),
                            Importance = round(importances[, 'MeanDecreaseGini']), 2)

# Create a rank variable based on importance
ranked <- varImportance %>%
  mutate(Rank = paste0('#', dense_rank(desc(Importance))))

# Use ggplot2 to visualize the relative importance of variables
g <- ggplot(ranked, aes(x=reorder(Variables, Importance),
                               y=Importance, fill=Importance))
g <- g + geom_bar(stat='identity') + 
  geom_text(aes(x = Variables, y = 0.5, label = Rank),hjust=0, vjust=0.55, size = 4, colour = 'red')
g + labs(x='Variables') + coord_flip() + theme_few()

# do the prediction
preds <- predict(rf_model, test)
prediction <- data.frame(PassengerId=test$PassengerId, Survived=preds)

# save the prediction for submission
write.csv(prediction, file="submission_rf.csv", row.names=F)
