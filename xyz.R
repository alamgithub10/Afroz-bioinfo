##08102022##10.1
uscrime <-read.table("E:/COURSES/TEACHERON/Jose/11092022/uscrime.data.txt", stringsAsFactors = FALSE, header = TRUE)
uscrime
View(uscrime)
#set seed 
set.seed(1)
# Load necessary packages
install.packages("randomForest")
install.packages("tree")
install.packages("caret")
install.packages("lattice")
library(randomForest)
library(tree)
library(caret)
# Build regression tree model using tree()
crimeTreeMod <- tree(Crime ~ ., data = uscrime)
crimeTreeMod
summary(crimeTreeMod)
# see how tree was split
crimeTreeMod$frame
par(mar = c(1, 2.1, 2.1, 1))
plot(crimeTreeMod)
text(crimeTreeMod)
title("USCRIME Training Set's Classification Tree")
# Prune the tree
termnodes <- 5
prune.crimeTreeMod <- prune.tree(crimeTreeMod, best = termnodes)
plot(prune.crimeTreeMod)
text(prune.crimeTreeMod)
title("Pruned Tree")
summary(prune.crimeTreeMod)
# Looks like pruning the tree actually increased the residual mean deviance, 
#implying that it's a worse fit. Interesting.
# Look at the deviation and do cross validation
cv.crime <- cv.tree(crimeTreeMod)
prune.tree(crimeTreeMod)$size
prune.tree(crimeTreeMod)$dev
cv.crime$dev
# This seems to indicate that overfitting is a big problem with our model
plot(cv.crime$size, cv.crime$dev, type = "b")
# This indicates that it's best to use the max number of terminal nodes, 7,
#as it has the least amount of error
termnodes2 <- 7
prune.crimeTreeMod2 <- prune.tree(crimeTreeMod, best = termnodes2)
plot(prune.crimeTreeMod2)
text(prune.crimeTreeMod2)
title("Pruned Tree")
summary(prune.crimeTreeMod2)
# Calculate quality of fit for this model
crimeTreePredict <- predict(prune.crimeTreeMod2, data = uscrime.data[,1:15])
RSS <- sum((crimeTreePredict - uscrime[,16])^2)
TSS <- sum((uscrime[,16] - mean(uscrime[,16]))^2)
R2 <- 1 - RSS/TSS
R2
# Create baseline randomForest Model
crime.rf <- randomForest(Crime ~ ., data=uscrime, importance = TRUE, nodesize = 5)
crime.rf
crime.rf.predict <- predict(crime.rf, data=uscrime[,-16])
crime.rf.predict
RSS <- sum((crime.rf.predict - uscrime[,16])^2)
RSS
R2 <- 1 - RSS/TSS
R2
# We get an R2 that's even worse than the previous tree. 
#Perhaps I will try using different values for mtry..
crime.rf2 <- randomForest(Crime ~ ., data=uscrime, importance = TRUE, nodesize = 5, mtry = 10)
crime.rf2
crime.rf.predict2 <- predict(crime.rf2, data=uscrime[,-16])
crime.rf.predict2
RSS <- sum((crime.rf.predict2 - uscrime[,16])^2)
RSS
R2 <- 1 - RSS/TSS
R2
# Let's make a loop to plug in different values of mtry and nodesize to try and 
#find the model with the best R2
result.rf <- data.frame(matrix(nrow=5, ncol=3))
result.rf
colnames(result.rf) <- c("NodeSize", "mtry", "R2")
i = 1
suppressWarnings(for (nodesize in 2:15) {
  for (m in 1:20) {
    model <- randomForest(Crime ~ ., data=uscrime, importance = TRUE, nodesize = nodesize, mtry = m)
    predict <- predict(model, data=uscrime[,-16])
    RSS <- sum((predict - uscrime[,16])^2)
    TSS <- sum((uscrime[,16] - mean(uscrime[,16]))^2)
    R2 <- 1 - RSS/TSS
    result.rf[i,1] <- nodesize
    result.rf[i,2] <- m
    result.rf[i,3] <- R2
    i = i + 1
  }
})
result.rf
headrf <- head(result.rf)
headrf
result.rf[which.max(result.rf[,3]),]
# So it looks like a nodesize of 4 and an mtry value of 3 gives us the 
#best randomForest model.
crime.rf.final <- randomForest(Crime ~ ., data=uscrime, importance = TRUE, nodesize = 4, mtry = 3)
crime.rf.final
# Let's look at the importance of the variables for our final model...
#we can use the importance function
importance(crime.rf.final)
varImpPlot(crime.rf.final)
par(mar = c(0.5, 0.5, 0.5, 0.5))
dev.off()
varImpPlot(crime.rf.final)

#10.3
library(knitr)
library(dplyr)
library(tidyr)
library(reshape2)
library(RColorBrewer)
library(GGally)
library(ggplot2)
library(caret)
library(glmnet)
library(boot)
library(verification)
set.seed(1)
credit <- read.table("E:/COURSES/TEACHERON/Jose/08102022/german.data", header = FALSE)
str(credit)
# See what the responses look like
table(credit$V21)
# Replace 1 and 2 with 0 and 1
credit$V21[credit$V21==1] <- 0
credit$V21[credit$V21==2] <- 1
table(credit$V21)
# We should measure model quality in terms of accuracy 
#(truePositive + trueNegative /(total predictions + actual values))

# Split data into train and validation set
creditPart <- createDataPartition(credit$V21, times = 1, p = 0.7, list=FALSE)
head(creditPart)
creditTrain <- credit[creditPart,] 
creditValid <- credit[-creditPart,]

table(creditTrain$V21)
table(creditValid$V21)
creditLogModel <- glm(V21 ~ ., data = creditTrain, family=binomial(link="logit"))
creditLogModel
# Look at importance of predictors
summary(creditLogModel)
#Let's do a baseline prediction.
creditPredict <- predict(creditLogModel, newdata=creditValid[,-21], type="response")
creditPredict
table(creditValid$V21, round(creditPredict))
# our initial baseline model using all predictors and a simple threshold of 0.5 
#seems to correctly classify a lot of the good borrowers (1), but misclassifies 
#a lot of the bad borrowers. Given that the cost of bad borrowers is 5x 
#that of misclassifying good borrowers, we should look for ways to minimize 
#the misclassifying of bad borrowers (specificity). Let's start by only including 
#variables that had a signficance of at least p <0.1

# Since there are multiple categorical variables within a single column, 
#we need to manually remove the non-significant variables.

creditTrain$V1A14[creditTrain$V1 == "A14"] <- 1
creditTrain$V1A14[creditTrain$V1 != "A14"] <- 0

creditTrain$V3A34[creditTrain$V3 == "A34"] <- 1
creditTrain$V3A34[creditTrain$V3 != "A34"] <- 0

creditTrain$V4A41[creditTrain$V4 == "A41"] <- 1
creditTrain$V4A41[creditTrain$V4 != "A41"] <- 0

creditTrain$V4A43[creditTrain$V4 == "A43"] <- 1
creditTrain$V4A43[creditTrain$V4 != "A43"] <- 0

# Now let's use the validation data and process it in the same way
creditValid$V1A14[creditValid$V1 == "A14"] <- 1
creditValid$V1A14[creditValid$V1 != "A14"] <- 0

creditValid$V3A34[creditValid$V3 == "A34"] <- 1
creditValid$V3A34[creditValid$V3 != "A34"] <- 0

creditValid$V4A41[creditValid$V4 == "A41"] <- 1
creditValid$V4A41[creditValid$V4 != "A41"] <- 0

creditValid$V4A43[creditValid$V4 == "A43"] <- 1
creditValid$V4A43[creditValid$V4 != "A43"] <- 0

creditLogModel2 <- glm(V21 ~ V1A14+V2+V3A34+V4A41+V4A43, data = creditTrain, family=binomial(link="logit"))
summary(creditLogModel2)

# create confusion matrix of predicted vs. observed values on validation set
creditPredict2 <- predict(creditLogModel2, newdata=creditValid[,-21], type="response")
t <- as.matrix(table(round(creditPredict2), creditValid$V21))
names(dimnames(t)) <- c("Predicted", "Observed")
t
# Calculate accuracy and specificity, aiming to maximize specificity given the 
#costs of a false positive
threshold <- 0.7
t2 <- as.matrix(table(round(creditPredict2 > threshold), creditValid$V21))
names(dimnames(t)) <- c("Predicted", "Observed")
names(dimnames(t))
t2
accuracy <- (t2[1,1]+t2[2,2])/(t2[1,1]+t2[1,2]+t2[2,1]+t2[2,2])
accuracy
specificity <- (t2[1,1])/(t2[1,1]+t2[2,1])
specificity

#So, with a threshold of 0.7, we arrive at specificity of 98.5% and overall accuracy 
#of 71%, which for the purposes of this application, is pretty good.