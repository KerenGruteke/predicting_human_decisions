rm(list = ls())
library(xgboost)
library(SHAPforxgboost)

setwd("C:/Users/plonsky/Dropbox/CPC 2018/NHB23 files/R1 files/Final files for upload/CPC18")

######
# get train and test data
load("PF_TrainSetFull.RData")
load("PF_TestSetFull.RData")

################# replication of the original model submitted to the cpc

# BEAST-GB hyperparamteres as submitted to the competition
xg_params = list(
  'booster' = 'gbtree',
  'colsample_bytree' = 0.9911994607412087,
  'gamma'= 0.012241954019987821,
  'learning_rate'= 0.010878922437398755,
  'max_depth'= 3,
  'nrounds'= 978,
  'reg_alpha'= 0.04306269451141776,
  'reg_lambda'= 2.9053833404234397,
  'subsample'= 0.5079640412046551,
  'n_jobs'= -1
)

preprocess_for_xgb <- function(train, test, y_name) {
  formula_str <- paste(y_name, "~ .+0")
  xtrain = model.matrix(as.formula(formula_str), data = train)
  trainMean <- apply(xtrain,2,mean)
  trainSd <- apply(xtrain,2,sd)
  ytrain = train[[y_name]]
  scaled_x_train <- sweep(sweep(xtrain, 2L, trainMean), 2, trainSd, "/")
  xtest = model.matrix(as.formula(formula_str),data = test)
  scaled_x_test <- sweep(sweep(xtest, 2L, trainMean), 2, trainSd, "/")
  ytest = test[[y_name]]
  
  ddd <- xgb.DMatrix(scaled_x_train,label=ytrain)
  to_return = list("xgb.DMatrix" = ddd, "x_test" = scaled_x_test)
  return(to_return)
}

# pre-processing for XGB
TrainData$block = as.numeric(TrainData$block)
TrainData = TrainData[,-1]
testData$block = as.numeric(testData$block)
testData = testData[,-1]

# run original model
out = preprocess_for_xgb(TrainData, testData, "B_rate")
ddd = out$xgb.DMatrix
scaled_x_test = out$x_test

set.seed(2023)
# Model
xgb_model = xgboost(params = xg_params, data = ddd,nrounds =978,verbose = 0)

# model output and scores
pred = predict(xgb_model,scaled_x_test)
jointPredObs = data.frame(testData,pred)
print(cor(testData$B_rate, pred))
mse_xgb = mean((jointPredObs$B_rate - jointPredObs$pred)^2)
print(mse_xgb)

# Compute and plot SHAP for CPC18
tmp = shap.prep(xgb_model = xgb_model, X_train = scaled_x_test)
shap.plot.summary(tmp, kind="bar")
mean_shap = unique(tmp[,c(2,6)])
mean_shap = mean_shap[order(-mean_shap$mean_value)]
# save(mean_shap, file = "SHAP CPC18.RData")

# For computing completeness 
raw_d = read.csv('rawData_comp_All.csv')
raw_d_subj = aggregate(B ~ SubjID + GameID+Ha+pHa+La+LotShapeA+LotNumA+Hb+pHb+Lb+LotShapeB+LotNumB+Amb+Corr+block, 
                       data=raw_d, mean)
aggMeans = aggregate(B ~ GameID+Ha+pHa+La+LotShapeA+LotNumA+Hb+pHb+Lb+LotShapeB+LotNumB+Amb+Corr+block, 
                     data=raw_d_subj, 
                     FUN = function(x) c(mean = mean(x), std = sd(x), n = length(x), SE2 = (sd(x)^2)/length(x)))
naive_err = mean((aggMeans$B[, "mean"] - 0.5)^2)
irreducible = mean(aggMeans$B[,"SE2"])
completeness_BEAST_GB = (naive_err - mse_xgb)/(naive_err- irreducible)
print(completeness_BEAST_GB)

####################################
# REMOVAL of FEATURE SETS #######
#################################

obj_feats = names(TrainData)[c(1:12,30,31)]
naive_feats = names(TrainData)[13:16]
psych_feats = names(TrainData)[c(15,17:28)]
foresight = "BEASTpred"

# run without naive
train_no_naive = TrainData[ , !(names(TrainData) %in% naive_feats)]
test_no_naive = testData[ , !(names(testData) %in% naive_feats)]
out = preprocess_for_xgb(train_no_naive, test_no_naive, "B_rate")
ddd = out$xgb.DMatrix
scaled_x_test = out$x_test
set.seed(2023)
xgb_model = xgboost(params = xg_params, data = ddd,nrounds =978,verbose = 0)
pred = predict(xgb_model,scaled_x_test)
jointPredObs = data.frame(testData,pred)
print(cor(testData$B_rate, pred))
mse_xgb = mean((jointPredObs$B_rate - jointPredObs$pred)^2)
print(mse_xgb)
completeness = (naive_err - mse_xgb)/(naive_err- irreducible)
print(completeness)

# run without psych
train_no_psych = TrainData[ , !(names(TrainData) %in% psych_feats)]
test_no_psych = testData[ , !(names(testData) %in% psych_feats)]
out = preprocess_for_xgb(train_no_psych, test_no_psych, "B_rate")
ddd = out$xgb.DMatrix
scaled_x_test = out$x_test
set.seed(2023)
xgb_model = xgboost(params = xg_params, data = ddd,nrounds =978,verbose = 0)
pred = predict(xgb_model,scaled_x_test)
jointPredObs = data.frame(testData,pred)
print(cor(testData$B_rate, pred))
mse_xgb = mean((jointPredObs$B_rate - jointPredObs$pred)^2)
print(mse_xgb)
completeness = (naive_err - mse_xgb)/(naive_err- irreducible)
print(completeness)

# run without foresight
train_no_foresight = TrainData[ , !(names(TrainData) %in% foresight)]
test_no_foresight = testData[ , !(names(testData) %in% foresight)]
out = preprocess_for_xgb(train_no_foresight, test_no_foresight, "B_rate")
ddd = out$xgb.DMatrix
scaled_x_test = out$x_test
set.seed(2023)
xgb_model = xgboost(params = xg_params, data = ddd,nrounds =978,verbose = 0)
pred = predict(xgb_model,scaled_x_test)
jointPredObs = data.frame(testData,pred)
print(cor(testData$B_rate, pred))
mse_xgb = mean((jointPredObs$B_rate - jointPredObs$pred)^2)
print(mse_xgb)
completeness = (naive_err - mse_xgb)/(naive_err- irreducible)
print(completeness)

# run without objective
train_no_obj = TrainData[ , !(names(TrainData) %in% obj_feats)]
test_no_obj = testData[ , !(names(testData) %in% obj_feats)]
out = preprocess_for_xgb(train_no_obj, test_no_obj, "B_rate")
ddd = out$xgb.DMatrix
scaled_x_test = out$x_test
set.seed(2023)
xgb_model = xgboost(params = xg_params, data = ddd,nrounds =978,verbose = 0)
pred = predict(xgb_model,scaled_x_test)
jointPredObs = data.frame(testData,pred)
print(cor(testData$B_rate, pred))
mse_xgb = mean((jointPredObs$B_rate - jointPredObs$pred)^2)
print(mse_xgb)
completeness = (naive_err - mse_xgb)/(naive_err- irreducible)
print(completeness)
