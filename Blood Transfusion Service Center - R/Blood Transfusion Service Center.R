# Load packages
library(readr)
library(caret)
library(dplyr)
library(car)

# Load data
raw.data <- read_csv("data/dd-train.csv")
names(raw.data) <- make.names(names(raw.data))  # Make names more R friendly
colnames(raw.data)[1] <- "id" # rename the column that has the ID
colnames(raw.data)[6] <- "Class"  # rename the column that has our class (to make it fit on the page)

# Split data into train set and validation set (80:20 slit)
set.seed(7)
validationIndex <- createDataPartition(raw.data$Class, p=0.80, list=FALSE)
validation.set <- raw.data[-validationIndex,]
train.set <- raw.data[validationIndex,]
train.set$Total.Volume.Donated..c.c.. <- NULL # remove correlated column

#Feature Engineering
train.set <- train.set %>% mutate(donations.per.month = Number.of.Donations/Months.since.First.Donation)
train.set <- train.set %>% mutate(tenure.ratio = Months.since.Last.Donation/Months.since.First.Donation)

#Recode the class labels to Yes/No (required when using class probs)
required.labels <- train.set['Class']
recoded.labels <- recode(required.labels$Class, "0='No'; 1 = 'Yes'")
train.set$Class <- recoded.labels
train.set$Class  <-as.factor(train.set$Class) # Make the class variable a factor
train.set$id <- NULL  #Remove id column

# Tune Model

preProcess="BoxCox"

# 10-fold cross validation with 3 repeats
trainControl <- trainControl(method="repeatedcv", summaryFunction=mnLogLoss, number=10, repeats=3, 
                             classProbs=TRUE)
metric <- "logLoss"

set.seed(7)
grid <- expand.grid(alpha=c((70:100)/100), lambda = c(0.001, 0.0009, 0.0008,0.0007))
fit.glmnet <- train(Class~., data=train.set, method="glmnet", metric=metric, tuneGrid=grid,
                    preProc=preProcess, trControl=trainControl)

## Check The Validation Set

# Match our training set
validation.set$Total.Volume.Donated..c.c.. <- NULL # remove correlated column

# Redo the feature engineering
validation.set <- validation.set %>% mutate(donations.per.month = Number.of.Donations/Months.since.First.Donation)
validation.set <- validation.set %>% mutate(tenure.ratio = Months.since.Last.Donation/Months.since.First.Donation)

# Recode the class labels to Yes/No (required when using class probs)
required.labels <- validation.set['Class']
recoded.labels <- recode(required.labels$Class, "0='No'; 1 = 'Yes'")
validation.set$Class <- recoded.labels
validation.set$Class  <-as.factor(validation.set$Class) # Make the class variable a factor
set.seed(7)
test.pred <- predict(fit.glmnet, newdata=validation.set[, c(2:4, 6:7)], type  = "prob")

# Function to calculate logLoss
LogLoss <- function(actual, predicted, eps=0.00001) {
  predicted <- pmin(pmax(predicted, eps), 1-eps)
  -1/length(actual)*(sum(actual*log(predicted)+(1-actual)*log(1-predicted)))
}

# In order to use the logLoss function we need to recode the class labels back to to 0 and 1
required.labels <- validation.set['Class']
recoded.labels <- recode(required.labels$Class, "'No'=0; 'Yes'=1")
validation.set$Class <- recoded.labels

# Now get logLoss on the validation set prediction
ll <- LogLoss(as.numeric(as.character(validation.set$Class)), test.pred$Yes)
ll

# Make A Prediction on the Test Set
test.data <- read_csv("data/dd-test.csv")
names(test.data) <- make.names(names(test.data))  # Make names more R friendly
test.data$Total.Volume.Donated..c.c.. <- NULL # remove this column (corrolated with Months.since.Last.Donation)
test.data <- test.data %>% mutate(donations.per.month = Number.of.Donations/Months.since.First.Donation)
test.data <- test.data %>% mutate(tenure.ratio = Months.since.Last.Donation/Months.since.First.Donation)
test.set <- test.data[,-1] # remove id column

# Make prediction
set.seed(7)
predictions <- predict(fit.glmnet, newdata=test.set, type  = "prob")
pred.df <- as.data.frame(predictions$Yes)
pred.df$id <- test.data$NA.
pred.df <- pred.df[c(2,1)]

# Prepare date for writing to csv using thre format from the submission format file
submission_format <- read.csv("data/BloodDonationSubmissionFormat.csv", check.names=FALSE)
colnames(pred.df) <- colnames(submission_format)
write.csv(pred.df, file="submission4.csv", row.names=FALSE )
