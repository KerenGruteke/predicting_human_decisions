rm(list = ls())
setwd("C:/Users/plonsky/Dropbox/CPC 2018/NHB23 files/R1 files/Final files for upload/Extensive form games")
library(reshape)
library(xgboost)

####################
# read and prepare esimation data player 2
EstSet1P2 = read.csv("Raw Data Est Set1 Player2.csv", header = T)
EstSet1P2 = melt(EstSet1P2,id=c("game","f1","s1","f2","s2","f3","s3"))
names(EstSet1P2)[8:9] = c("subj","choice")
EstSet1P2$set = 1
EstSet2P2 = read.csv("Raw Data Est Set2 Player2.csv", header = T)
EstSet2P2 = melt(EstSet2P2,id=c("game","f1","s1","f2","s2","f3","s3"))
names(EstSet2P2)[8:9] = c("subj","choice")
EstSet2P2$set = 2
EstSetP2 = rbind(EstSet1P2,EstSet2P2)
EstSetP2$subjID = EstSetP2$set*100 + as.numeric(EstSetP2$subj)
EstSetP2$subjID = factor(EstSetP2$subjID)
EstSetP2$subj = NULL
EstSetP2$set = NULL
# EstSetP2$choice = factor(EstSetP2$choice)
EstSetP2Agg = aggregate(choice ~ ., data = EstSetP2[,-9], mean)

data = EstSetP2Agg
data$Rational=(1+sign(data$s3-data$s2))/2
maxp1 = (1+sign(data$f3-data$f2))/2
data$NiceRat = data$Rational
data$NiceRat[data$Rational==0.5] = maxp1[data$Rational==0.5]
jmRL = data$s3 + data$f3 - data$s2 - data$f2
data$jointMax = (1+sign(jmRL))/2
data$maxWeak = 0 
data$maxWeak[pmin(data$f2,data$s2) < pmin(data$f3,data$s3)] = 1
data$maxWeak[pmin(data$f2,data$s2) == pmin(data$f3,data$s3)] = 0.5
data$minDiff = 0
data$minDiff[abs(data$f2-data$s2) > abs(data$f3-data$s3)] = 1
data$minDiff[abs(data$f2-data$s2) == abs(data$f3-data$s3)] = 0.5

# read and prepare competition data player 2
CompSet1P2 = read.csv("Raw Data Pred Set1 Player2.csv", header = T)
CompSet1P2 = melt(CompSet1P2,id=c("game","f1","s1","f2","s2","f3","s3"))
names(CompSet1P2)[8:9] = c("subj","choice")
CompSet1P2$set = 1
CompSet2P2 = read.csv("Raw Data Pred Set2 Player2.csv", header = T)
CompSet2P2 = melt(CompSet2P2,id=c("game","f1","s1","f2","s2","f3","s3"))
names(CompSet2P2)[8:9] = c("subj","choice")
CompSet2P2$set = 2
CompSetP2 = rbind(CompSet1P2,CompSet2P2)
CompSetP2$subjID = CompSetP2$set*100 + as.numeric(CompSetP2$subj)
CompSetP2$subjID = factor(CompSetP2$subjID)
CompSetP2$subj = NULL
CompSetP2$set = NULL
CompSetP2Agg = aggregate(choice ~ ., data = CompSetP2[,-9], mean)

test = CompSetP2Agg
test$Rational=(1+sign(test$s3-test$s2))/2
maxp1 = (1+sign(test$f3-test$f2))/2
test$NiceRat = test$Rational
test$NiceRat[test$Rational==0.5] = maxp1[test$Rational==0.5]
jmRL = test$s3 + test$f3 - test$s2 - test$f2
test$jointMax = (1+sign(jmRL))/2
test$maxWeak = 0 
test$maxWeak[pmin(test$f2,test$s2) < pmin(test$f3,test$s3)] = 1
test$maxWeak[pmin(test$f2,test$s2) == pmin(test$f3,test$s3)] = 0.5
test$minDiff = 0
test$minDiff[abs(test$f2-test$s2) > abs(test$f3-test$s3)] = 1
test$minDiff[abs(test$f2-test$s2) == abs(test$f3-test$s3)] = 0.5

####
data$baselinePred = 0.506*data$Rational + 0.354*data$NiceRat +
  0.059*data$jointMax + 0.044*data$maxWeak + 0.037*data$minDiff

test$baselinePred = 0.506*test$Rational + 0.354*test$NiceRat +
  0.059*test$jointMax + 0.044*test$maxWeak + 0.037*test$minDiff

rm(list=setdiff(ls(), c("data", "test")))

xTrain <- data[, c(2:7,9:14)]  
yTrain <- data$choice
xTest  <- test[, c(2:7,9:14)]
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
#   eta = c(0.015, 0.02, 0.025),        # Smaller learning rates
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
#   verbose = TRUE
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
  'eta'= 0.02,
  'max_depth'= 2,
  'subsample'= 0.7,
  'min_child_weight' = 1 
)

### get data
xy_train = data[,c(2:14)]
x_train = model.matrix(choice~., data= xy_train)[,-1]
xy_test = test[,c(2:14)]
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

