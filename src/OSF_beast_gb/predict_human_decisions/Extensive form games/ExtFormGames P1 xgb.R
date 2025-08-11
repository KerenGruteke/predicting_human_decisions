rm(list = ls())
setwd("C:/Users/plonsky/Dropbox/CPC 2018/NHB23 files/R1 files/Final files for upload/Extensive form games")
library(reshape)
library(xgboost)

# read and prepare esimation data player 1
EstSet1P1 = read.csv("Raw Data Est Set1 Player1.csv", header = T)
EstSet1P1 = melt(EstSet1P1,id=c("game","f1","s1","f2","s2","f3","s3"))
names(EstSet1P1)[8:9] = c("subj","choice")
EstSet1P1$subjID1 = as.numeric(EstSet1P1$subj)
EstSet1P1$subjID2 = 101
EstSet2P1 = read.csv("Raw Data Est set2 Player 1.csv", header = T)
EstSet2P1 = melt(EstSet2P1,id=c("game","f1","s1","f2","s2","f3","s3"))
names(EstSet2P1)[8:9] = c("subj","choice")
EstSet2P1$subjID2 = as.numeric(EstSet2P1$subj)
EstSet2P1$subjID1 = 101
EstSetP1 = rbind(EstSet1P1,EstSet2P1)
EstSetP1$subjID1 = factor(EstSetP1$subjID1)
EstSetP1$subjID2 = factor(EstSetP1$subjID2)
EstSetP1$subj = NULL

data = EstSetP1

ratP2=(1+sign(data$s3-data$s2))/2
avgIn = (data$f2+data$f3)/2
minIn = pmin(data$f2,data$f3)
data$Rational = 0
data$Rational[(ratP2==0 & data$f1==data$f2) | (ratP2==1 & data$f1==data$f3)] = 0.5
data$Rational[(ratP2==0 & data$f1 < data$f2) | (ratP2==1 & data$f1 < data$f3)] = 1
data$Rational[data$f1 < minIn] = 1
data$Rational[ratP2==0.5 & data$f1 < avgIn] = 1
data$Rational[ratP2==0.5 & data$f1 == avgIn] = 0.5
data$Level1 = 0
data$Level1[avgIn > data$f1] = 1
data$Level1[avgIn == data$f1] = 0.5
data$MaxMin = 0
data$MaxMin[minIn > data$f1] = 1
data$MaxMin[minIn == data$f1] = 0.5
data$jointMax = 0
data$jointMax[(data$s1+data$f1) < pmax(data$f2+data$s2,data$f3+data$s3)] = 1
data$jointMax[(data$s1+data$f1) == pmax(data$f2+data$s2,data$f3+data$s3)] = 0.5
data$maxWeak = 0 
data$maxWeak[pmin(data$f1,data$s1) < pmax(pmin(data$f2,data$s2),pmin(data$f3,data$s3))] = 1
data$maxWeak[pmin(data$f1,data$s1) == pmax(pmin(data$f2,data$s2),pmin(data$f3,data$s3))] = 0.5
data$minDiff = 0
data$minDiff[abs(data$f1-data$s1) > pmin(abs(data$f2-data$s2),abs(data$f3-data$s3))] = 1
data$minDiff[abs(data$f1-data$s1) == pmin(abs(data$f2-data$s2),abs(data$f3-data$s3))] = 0.5

# read and prepare competition data player 1
CompSet1P1 = read.csv("Raw Data Pred Set1 Player1.csv", header = T)
CompSet1P1 = melt(CompSet1P1,id=c("game","f1","s1","f2","s2","f3","s3"))
names(CompSet1P1)[8:9] = c("subj","choice")
CompSet1P1$subjID1 = as.numeric(CompSet1P1$subj)
CompSet1P1$subjID2 = 101
CompSet2P1 = read.csv("Raw Data Pred Set2 Player1.csv", header = T)
CompSet2P1$game = CompSet2P1$game + 60
CompSet2P1 = melt(CompSet2P1,id=c("game","f1","s1","f2","s2","f3","s3"))
names(CompSet2P1)[8:9] = c("subj","choice")
CompSet2P1$subjID2 = as.numeric(CompSet2P1$subj)
CompSet2P1$subjID1 = 101
CompSetP1 = rbind(CompSet1P1,CompSet2P1)
CompSetP1$subjID1 = factor(CompSetP1$subjID1)
CompSetP1$subjID2 = factor(CompSetP1$subjID2)
CompSetP1$subj = NULL

test=CompSetP1
ratP2=(1+sign(test$s3-test$s2))/2
avgIn = (test$f2+test$f3)/2
minIn = pmin(test$f2,test$f3)
test$Rational = 0
test$Rational[(ratP2==0 & test$f1==test$f2) | (ratP2==1 & test$f1==test$f3)] = 0.5
test$Rational[(ratP2==0 & test$f1 < test$f2) | (ratP2==1 & test$f1 < test$f3)] = 1
test$Rational[test$f1 < minIn] = 1
test$Rational[ratP2==0.5 & test$f1 < avgIn] = 1
test$Rational[ratP2==0.5 & test$f1 == avgIn] = 0.5
test$Level1 = 0
test$Level1[avgIn > test$f1] = 1
test$Level1[avgIn == test$f1] = 0.5
test$MaxMin = 0
test$MaxMin[minIn > test$f1] = 1
test$MaxMin[minIn == test$f1] = 0.5
test$jointMax = 0
test$jointMax[(test$s1+test$f1) < pmax(test$f2+test$s2,test$f3+test$s3)] = 1
test$jointMax[(test$s1+test$f1) == pmax(test$f2+test$s2,test$f3+test$s3)] = 0.5
test$maxWeak = 0 
test$maxWeak[pmin(test$f1,test$s1) < pmax(pmin(test$f2,test$s2),pmin(test$f3,test$s3))] = 1
test$maxWeak[pmin(test$f1,test$s1) == pmax(pmin(test$f2,test$s2),pmin(test$f3,test$s3))] = 0.5
test$minDiff = 0
test$minDiff[abs(test$f1-test$s1) > pmin(abs(test$f2-test$s2),abs(test$f3-test$s3))] = 1
test$minDiff[abs(test$f1-test$s1) == pmin(abs(test$f2-test$s2),abs(test$f3-test$s3))] = 0.5

estsubj = data
compsubj = test
estsubj$choice = factor(estsubj$choice)
compsubj$choice = factor(compsubj$choice)
data = aggregate(choice ~ ., data = data[,-c(9:10)], mean) 
test = aggregate(choice ~ ., data = test[,-c(9:10)], mean) 

####
data$baselinePred = 0.438*data$Rational + 0.192*data$MaxMin +
  0.193*data$Level1 + 0.075*data$jointMax + 0.075*data$maxWeak + 0.027*data$minDiff

test$baselinePred = 0.438*test$Rational + 0.192*test$MaxMin +
  0.193*test$Level1 + 0.075*test$jointMax + 0.075*test$maxWeak + 0.027*test$minDiff

rm(list=setdiff(ls(), c("data", "test")))

xTrain <- data[, c(2:13,15)]
yTrain <- data$choice
xTest  <- test[, c(2:13,15)]
yTest  <- test$choice

# ####################################
# #### hyper-parameter tuning ####
# ####################################
# 
# library(caret)
# library(doParallel)
# library(parallel)
# library(xgboost)
# 
# 
# 
# cvCtrl <- trainControl(
#   method = "repeatedcv",  # repeated cross-validation
#   number = 10,            # 5-fold CV
#   repeats = 3,           # repeat it twice
#   verboseIter = TRUE,  
#   allowParallel = TRUE    
# )
# 
# xgbGrid <- expand.grid(
#   nrounds = seq(300, 500, by = 50),  # Focus around 400
#   max_depth = c(2, 3, 4),            # Focus on shallow trees
#   eta = c(0.01, 0.015, 0.02),        # Smaller learning rates
#   gamma = c(0, 0.1, 0.2),            # Small gamma values
#   colsample_bytree = c(0.5, 0.6, 0.7), # Slightly lower values
#   min_child_weight = c(1, 2, 3),     # Slightly higher values
#   subsample = c(0.7, 0.8, 0.9)       # Slightly lower values
# )
# 
# num_cores <- detectCores() - 1  
# # Register the parallel backend
# cl <- makePSOCKcluster(num_cores)
# registerDoParallel(cl)
# 
# xgbTune <- train(
#   x = xTrain,
#   y = yTrain,
#   method = "xgbTree",
#   trControl = cvCtrl,
#   tuneGrid = xgbGrid,
#   metric = "RMSE",     
#   verbose = FALSE
# )
# 
# stopCluster(cl)  # stop cluster
# 
# print(xgbTune$bestTune)


#########################################################
###### Train, predict, and evaluate using best hps ######
#########################################################
set.seed(2023)
#### Setting best hps 
xg_params = list(
  'colsample_bytree' = 0.5,
  'gamma'= 0,
  'eta'= 0.015,
  'max_depth'= 2,
  'subsample'= 0.7,
  'min_child_weight' = 1 
)

### get data
xy_train = data[,2:15]
x_train = model.matrix(choice~., data= xy_train)[,-1]
xy_test = test[,2:15]
x_test = model.matrix(choice~.,data=xy_test)[,-1]

#### standardize features
trainMean = apply(x_train,2,mean)
trainSd = apply(x_train,2,sd)
scaled_x_train = sweep(sweep(x_train, 2, trainMean), 2, trainSd, "/")
scaled_x_test <- sweep(sweep(x_test, 2, trainMean), 2, trainSd, "/")

#### train BEAST-GB and generate predictions
ddd = xgb.DMatrix(x_train,label=yTrain)
seven_strat_GB = xgboost(params = xg_params, data = ddd, nrounds =400, verbose = 0)

#### model output and scores
pred = predict(seven_strat_GB, x_test)
jointPredObs = data.frame(xy_test,pred)
mse_seven_strat_GB = mean((jointPredObs$choice - jointPredObs$pred)^2)
print(mse_seven_strat_GB)

