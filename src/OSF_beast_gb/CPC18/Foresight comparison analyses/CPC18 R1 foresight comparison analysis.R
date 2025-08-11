rm(list = ls())
library(xgboost)
library(caret)
library(doParallel)
# library(tictoc)

setwd("C:/Users/plonsky/Dropbox/CPC 2018/NHB23 files/R1 files/Final files for upload/CPC18/Foresight comparison analyses")

######
# read all (fitted) model predictions
bst = read.csv("beast_predRisk.csv")
DbS = read.csv("DbS_predRisk.csv")
PH = read.csv("PH_predRisk.csv")
SCPT = read.csv("SCPT_predRisk.csv")
dCPT = read.csv("DetCPT_predRisk.csv")

####
feats = c("Ha","pHa","La","LotShapeA","LotNumA",
           "Hb","pHb","Lb","LotShapeB","LotNumB",
           "Amb","Corr","pred", "B")

ytrain = bst[bst$GameID < 211,"B"]
ytest = bst[bst$GameID >= 211,"B"]

x.train.bst = bst[bst$GameID < 211,feats]
x.test.bst = bst[bst$GameID >= 211,feats]

x.train.dbs = DbS[DbS$GameID < 211,feats]
x.test.dbs = DbS[DbS$GameID >= 211,feats]

x.train.ph = PH[PH$GameID < 211,feats]
x.test.ph = PH[PH$GameID >= 211,feats]

x.train.cpt = SCPT[SCPT$GameID < 211,feats]
x.test.cpt = SCPT[SCPT$GameID >= 211,feats]

x.train.dcpt = dCPT[dCPT$GameID < 211,feats]
x.test.dcpt = dCPT[dCPT$GameID >= 211,feats]


###### XGB BEAST tuning
# 
# tune.grid <- expand.grid(
#   nrounds =seq(900, 1350, 150),
#   max_depth = seq(2, 5, 1),
#   eta = seq(0.0025, 0.015, 0.0025),
#   gamma = seq(0.01, 0.03, 0.01),
#   colsample_bytree =  seq(0.7, 1, 0.1),
#   min_child_weight = c(1),
#   subsample = seq(0.45, 0.85, 0.2)
# )
# tuneGrid_refined <- expand.grid(
#   nrounds = seq(400, 1000, 100),            
#   max_depth = seq(8, 13, 1),                     
#   eta = seq(0.01, 0.03, 0.01),                
#   gamma = 0,                               
#   colsample_bytree = 1,                    
#   min_child_weight = 1,         
#   subsample = seq(0.3, 0.5, 0.05)                  
# )
# 
# cl <- makePSOCKcluster(6)
# registerDoParallel(cl)
# 
# ctrl <- caret::trainControl(method = "repeatedcv",    # Cross-validation
#                             number = 5,       # Number of folds
#                             search = "grid",  # Perform grid search
#                             repeats = 5,
#                             verboseIter = TRUE) # Print training log
# 
# 
# xtrain = model.matrix(B~.+0, data = x.train.bst)
# trainMean <- apply(xtrain,2,mean)
# trainSd <- apply(xtrain,2,sd)
# scaled_x_train <- sweep(sweep(xtrain, 2L, trainMean), 2, trainSd, "/")
# xtest = model.matrix(B~.+0,data = x.test.bst)
# scaled_x_test <- sweep(sweep(xtest, 2L, trainMean), 2, trainSd, "/")
# 
# tic()
# xgb_tune <- caret::train(
#   x = scaled_x_train,
#   y = ytrain,
#   trControl = ctrl,#train_control,
#   tuneGrid = tuneGrid_refined,#tune.grid,
#   method = "xgbTree",
#   verbose = FALSE #TRUE
# )
# toc()
# bb= xgb_tune$results
# bb=bb[order(bb$RMSE),]
# 
# stopCluster(cl)

##### Run tuned XGB BEAST

xg_params = list(
  'booster' = 'gbtree',
  'colsample_bytree' = 1,
  'gamma'= 0,
  'learning_rate'= 0.02,
  'max_depth'= 12,
  'nrounds'= 600,
  'subsample'= 0.45,
  'min_child_weight' = 1
)

xtrain = model.matrix(B~.+0, data = x.train.bst)
trainMean <- apply(xtrain,2,mean)
trainSd <- apply(xtrain,2,sd)
scaled_x_train <- sweep(sweep(xtrain, 2L, trainMean), 2, trainSd, "/")
xtest = model.matrix(B~.+0,data = x.test.bst)
scaled_x_test <- sweep(sweep(xtest, 2L, trainMean), 2, trainSd, "/")

ddd <- xgb.DMatrix(scaled_x_train,label=ytrain)

set.seed(2023)
# Model
xgb_model = xgboost(params = xg_params, data = ddd,nrounds =600,verbose = 0)

# model output and scores
pred = predict(xgb_model,scaled_x_test)
mse_bst = mean((pred-ytest)^2)
print(mse_bst)

mse_bst_own =  mean((x.test.bst$pred - ytest)^2)
print(mse_bst_own)

####################### tune XGB CPT
# xtrain = model.matrix(B~.+0, data = x.train.cpt)
# trainMean <- apply(xtrain,2,mean)
# trainSd <- apply(xtrain,2,sd)
# scaled_x_train <- sweep(sweep(xtrain, 2L, trainMean), 2, trainSd, "/")
# xtest = model.matrix(B~.+0,data = x.test.cpt)
# scaled_x_test <- sweep(sweep(xtest, 2L, trainMean), 2, trainSd, "/")
# 
# 
# tuneGrid_refined <- expand.grid(
#   nrounds = seq(1200, 2300, 100),            
#   max_depth = seq(10, 14, 1),                     
#   eta = seq(0.005, 0.025, 0.005),                
#   gamma = 0,                               
#   colsample_bytree = 1,                    
#   min_child_weight = 2,#c(1,2,3),         
#   subsample = 0.45 #seq(0.3, 0.5, 0.05)                  
# )
# 
# cl <- makePSOCKcluster(6)
# registerDoParallel(cl)
# 
# tic()
# xgb_tune <- caret::train(
#   x = scaled_x_train,
#   y = ytrain,
#   trControl = ctrl,#train_control,
#   tuneGrid = tuneGrid_refined,#tune.grid,
#   method = "xgbTree",
#   verbose = FALSE #TRUE
# )
# toc()
# aa= xgb_tune$results
# aa=aa[order(aa$RMSE),]
# 
# stopCluster(cl)


##### Run tuned XGB SCPT

xg_params = list(
  'booster' = 'gbtree',
  'colsample_bytree' = 1,
  'gamma'= 0,
  'learning_rate'= 0.02,
  'max_depth'= 13,
  'nrounds'= 2100,
  'subsample'= 0.45,
  'min_child_weight' = 2
)
xtrain = model.matrix(B~.+0, data = x.train.cpt)
trainMean <- apply(xtrain,2,mean)
trainSd <- apply(xtrain,2,sd)
scaled_x_train <- sweep(sweep(xtrain, 2L, trainMean), 2, trainSd, "/")
xtest = model.matrix(B~.+0,data = x.test.cpt)
scaled_x_test <- sweep(sweep(xtest, 2L, trainMean), 2, trainSd, "/")

ddd <- xgb.DMatrix(scaled_x_train,label=ytrain)

set.seed(2023)
# Model
xgb_model = xgboost(params = xg_params, data = ddd,nrounds =2100,verbose = 0)

# model output and scores
pred = predict(xgb_model,scaled_x_test)
mse_cpt = mean((pred-ytest)^2)
print(mse_cpt)

mse_cpt_own =  mean((x.test.cpt$pred - x.test.cpt$B)^2)
print(mse_cpt_own)

####################### tune XGB DCPT
# xtrain = model.matrix(B~.+0, data = x.train.dcpt)
# trainMean <- apply(xtrain,2,mean)
# trainSd <- apply(xtrain,2,sd)
# scaled_x_train <- sweep(sweep(xtrain, 2L, trainMean), 2, trainSd, "/")
# xtest = model.matrix(B~.+0,data = x.test.dcpt)
# scaled_x_test <- sweep(sweep(xtest, 2L, trainMean), 2, trainSd, "/")
# 
# 
# tuneGrid <- expand.grid(
#   nrounds = seq(200, 1100, 300), 
#   max_depth = c(3, 5, 7, 9), 
#   eta = seq(0.01, 0.13, 0.06), 
#   gamma = c(0, 1, 5), 
#   colsample_bytree = c(0.5, 0.7, 1), 
#   min_child_weight = c(1, 3, 5), 
#   subsample = seq(0.4, 1, 0.3)
# )
# 
# tuneGrid_refined <- expand.grid(
#   nrounds = seq(400, 1000, 100), 
#   max_depth = seq(4, 10, 1), 
#   eta = seq(0.01, 0.04, 0.01), 
#   gamma = 0, 
#   colsample_bytree = 1, 
#   min_child_weight = 1, 
#   subsample = seq(0.5,0.7, 0.05)             
# )
# 
# cl <- makePSOCKcluster(6)
# registerDoParallel(cl)
# 
# tic()
# xgb_tune <- caret::train(
#   x = scaled_x_train,
#   y = ytrain,
#   trControl = ctrl,#train_control,
#   tuneGrid = tuneGrid_refined,
#   method = "xgbTree",
#   verbose = FALSE #TRUE
# )
# toc()
# aa= xgb_tune$results
# aa=aa[order(aa$RMSE),]
# 
# stopCluster(cl)

##### Run tuned XGB dCPT

xg_params = list(
  'booster' = 'gbtree',
  'colsample_bytree' = 1,
  'gamma'= 0,
  'learning_rate'= 0.01,
  'max_depth'= 8,
  'nrounds'= 500,
  'subsample'= 0.5,
  'min_child_weight' = 1
)
xtrain = model.matrix(B~.+0, data = x.train.dcpt)
trainMean <- apply(xtrain,2,mean)
trainSd <- apply(xtrain,2,sd)
scaled_x_train <- sweep(sweep(xtrain, 2L, trainMean), 2, trainSd, "/")
xtest = model.matrix(B~.+0,data = x.test.dcpt)
scaled_x_test <- sweep(sweep(xtest, 2L, trainMean), 2, trainSd, "/")
ddd <- xgb.DMatrix(scaled_x_train,label=ytrain)

set.seed(2023)
# Model
xgb_model = xgboost(params = xg_params, data = ddd,nrounds =500,verbose = 0)

# model output and scores
pred = predict(xgb_model,scaled_x_test)
mse_dcpt = mean((pred-ytest)^2)
print(mse_dcpt)

mse_dcpt_own =  mean((x.test.dcpt$pred - x.test.dcpt$B)^2)
print(mse_dcpt_own)

####################### tune XGB DbS
# xtrain = model.matrix(B~.+0, data = x.train.dbs)
# trainMean <- apply(xtrain,2,mean)
# trainSd <- apply(xtrain,2,sd)
# scaled_x_train <- sweep(sweep(xtrain, 2L, trainMean), 2, trainSd, "/")
# xtest = model.matrix(B~.+0,data = x.test.dbs)
# scaled_x_test <- sweep(sweep(xtest, 2L, trainMean), 2, trainSd, "/")
# 
# 
# tuneGrid <- expand.grid(
#   nrounds = seq(200, 1100, 300), 
#   max_depth = c(3, 5, 7, 9), 
#   eta = seq(0.01, 0.13, 0.06), 
#   gamma = c(0, 1, 5), 
#   colsample_bytree = c(0.5, 0.7, 1), 
#   min_child_weight = c(1, 3, 5), 
#   subsample = seq(0.4, 1, 0.3)
# )
# 
# tuneGrid_refined <- expand.grid(
#   nrounds = seq(2500, 4000, 100), 
#   max_depth = 5, #c(4,5), 
#   eta = 0.02, #seq(0.02, 0.03, 0.01), 
#   gamma = 0, 
#   colsample_bytree = 1, 
#   min_child_weight = 1,  
#   subsample = c(0.35)          
# )
# 
# cl <- makePSOCKcluster(6)
# registerDoParallel(cl)
# 
# tic()
# xgb_tune <- caret::train(
#   x = scaled_x_train,
#   y = ytrain,
#   trControl = ctrl,#train_control,
#   tuneGrid = tuneGrid_refined,
#   method = "xgbTree",
#   verbose = FALSE #TRUE
# )
# toc()
# aa= xgb_tune$results
# aa=aa[order(aa$RMSE),]
# 
# stopCluster(cl)

##### Run tuned XGB DbS
xg_params = list(
  'booster' = 'gbtree',
  'colsample_bytree' = 1,
  'gamma'= 0,
  'learning_rate'= 0.02,
  'max_depth'= 5,
  'nrounds'= 3500,
  'subsample'= 0.35,
  'min_child_weight' = 1
)
xtrain = model.matrix(B~.+0, data = x.train.dbs)
trainMean <- apply(xtrain,2,mean)
trainSd <- apply(xtrain,2,sd)
scaled_x_train <- sweep(sweep(xtrain, 2L, trainMean), 2, trainSd, "/")
xtest = model.matrix(B~.+0,data = x.test.dbs)
scaled_x_test <- sweep(sweep(xtest, 2L, trainMean), 2, trainSd, "/")


ddd <- xgb.DMatrix(scaled_x_train,label=ytrain)

set.seed(2023)
# Model
xgb_model = xgboost(params = xg_params, data = ddd,nrounds =3500,verbose = 0)

# model output and scores
pred = predict(xgb_model,scaled_x_test)
mse_DbS = mean((pred-ytest)^2)
print(mse_DbS)

mse_dbs_own =  mean((x.test.dbs$pred - ytest)^2)
print(mse_dbs_own)

####################### tune XGB PH
# xtrain = model.matrix(B~.+0, data = x.train.ph)
# trainMean <- apply(xtrain,2,mean)
# trainSd <- apply(xtrain,2,sd)
# scaled_x_train <- sweep(sweep(xtrain, 2L, trainMean), 2, trainSd, "/")
# xtest = model.matrix(B~.+0,data = x.test.ph)
# scaled_x_test <- sweep(sweep(xtest, 2L, trainMean), 2, trainSd, "/")
# 
# 
# tuneGrid <- expand.grid(
#   nrounds = seq(200, 1100, 300), 
#   max_depth = c(3, 5, 7, 9), 
#   eta = seq(0.01, 0.13, 0.06), 
#   gamma = c(0, 1, 5), 
#   colsample_bytree = c(0.5, 0.7, 1), 
#   min_child_weight = c(1, 3, 5), 
#   subsample = seq(0.4, 1, 0.3)
# )
# 
# tuneGrid_refined <- expand.grid(
#   nrounds = seq(1000, 2000, 200), 
#   max_depth = seq(5,7, 1), 
#   eta = seq(0.01, 0.05, 0.01), 
#   gamma = 0, 
#   colsample_bytree = 0.9, 
#   min_child_weight = c(4, 5),   
#   subsample = seq(0.55, 0.7, 0.05)          
# )
# 
# cl <- makePSOCKcluster(6)
# registerDoParallel(cl)
# 
# tic()
# xgb_tune <- caret::train(
#   x = scaled_x_train,
#   y = ytrain,
#   trControl = ctrl,#train_control,
#   tuneGrid = tuneGrid_refined,
#   method = "xgbTree",
#   verbose = FALSE #TRUE
# )
# toc()
# aa= xgb_tune$results
# aa=aa[order(aa$RMSE),]
# 
# stopCluster(cl)

##### Run tuned XGB PH
xg_params = list(
  'booster' = 'gbtree',
  'colsample_bytree' = 0.9,
  'gamma'= 0,
  'learning_rate'= 0.03,
  'max_depth'= 6,
  'nrounds'= 1400,
  'subsample'= 0.6,
  'min_child_weight' = 4
)

xtrain = model.matrix(B~.+0, data = x.train.ph)
trainMean <- apply(xtrain,2,mean)
trainSd <- apply(xtrain,2,sd)
scaled_x_train <- sweep(sweep(xtrain, 2L, trainMean), 2, trainSd, "/")
xtest = model.matrix(B~.+0,data = x.test.ph)
scaled_x_test <- sweep(sweep(xtest, 2L, trainMean), 2, trainSd, "/")

ddd <- xgb.DMatrix(scaled_x_train,label=ytrain)

set.seed(2023)
# Model
xgb_model = xgboost(params = xg_params, data = ddd,nrounds =1400,verbose = 0)

# model output and scores
pred = predict(xgb_model,scaled_x_test)
mse_ph = mean((pred-ytest)^2)
print(mse_ph)

mse_ph_own =  mean((x.test.ph$pred - ytest)^2)
print(mse_ph_own)


##### prepare ensemble of foresights

names(bst)[names(bst)=="pred"] = "beast_pred"
names(SCPT)[names(SCPT)=="pred"] = "SCPT_pred"
names(DbS)[names(DbS)=="pred"] = "dbs_pred"
names(PH)[names(PH)=="pred"] = "ph_pred"
names(dCPT)[names(dCPT)=="pred"] = "dCPT_pred"

tmp = merge(bst,SCPT)
tmp = merge(tmp, DbS)
tmp = merge(tmp, PH)
tmp = merge(tmp, dCPT)
tmp = tmp[order(tmp$GameID),]

feats = c("Ha","pHa","La","LotShapeA","LotNumA",
          "Hb","pHb","Lb","LotShapeB","LotNumB",
          "Amb","Corr","beast_pred","SCPT_pred",
          "dCPT_pred","dbs_pred","ph_pred","B")

ytrain = tmp[tmp$GameID < 211,"B"]
ytest = tmp[tmp$GameID >= 211,"B"]

x.train.all = tmp[tmp$GameID < 211,feats]
x.test.all = tmp[tmp$GameID >= 211,feats]

##### tune ensemble of foresights

# tune.grid <- expand.grid(
#   nrounds =seq(600, 1500, 300),
#   max_depth = seq(2, 8, 2),
#   eta = seq(0.001, 0.005, 0.002),
#   gamma = seq(0, 0.05, 0.025),
#   colsample_bytree =  seq(0.6, 1, 0.2),
#   min_child_weight = c(1, 2, 3),
#   subsample = seq(0.4, 1, 0.2)
# )
# cl <- makePSOCKcluster(6)
# registerDoParallel(cl)
# 
# ctrl <- caret::trainControl(method = "repeatedcv",    # Cross-validation
#                             number = 5,       # Number of folds
#                             search = "grid",  # Perform grid search
#                             repeats = 5,
#                             verboseIter = TRUE) # Print training log
# 
# 
# xtrain = model.matrix(B~.+0, data = x.train.all)
# trainMean <- apply(xtrain,2,mean)
# trainSd <- apply(xtrain,2,sd)
# scaled_x_train <- sweep(sweep(xtrain, 2L, trainMean), 2, trainSd, "/")
# xtest = model.matrix(B~.+0,data = x.test.all)
# scaled_x_test <- sweep(sweep(xtest, 2L, trainMean), 2, trainSd, "/")
# 
# # tic()
# xgb_tune <- caret::train(
#   x = scaled_x_train,
#   y = ytrain,
#   trControl = ctrl,#train_control,
#   tuneGrid = tune.grid,
#   method = "xgbTree",
#   verbose = FALSE #TRUE
# )
# save.image("output ensemble.RData")
# # toc()
# bb= xgb_tune$results
# bb=bb[order(bb$RMSE),]
# 
# stopCluster(cl)

##### Run tuned ensemble of foresights

xg_params = list(
  'booster' = 'gbtree',
  'colsample_bytree' = 1,
  'gamma'= 0,
  'learning_rate'= 0.015,
  'max_depth'= 4,
  'nrounds'= 2700,
  'subsample'= 0.4,
  'min_child_weight' = 1
)

xtrain = model.matrix(B~.+0, data = x.train.all)
trainMean <- apply(xtrain,2,mean)
trainSd <- apply(xtrain,2,sd)
scaled_x_train <- sweep(sweep(xtrain, 2L, trainMean), 2, trainSd, "/")
xtest = model.matrix(B~.+0,data = x.test.all)
scaled_x_test <- sweep(sweep(xtest, 2L, trainMean), 2, trainSd, "/")

ddd <- xgb.DMatrix(scaled_x_train,label=ytrain)

set.seed(2023)
# Model
xgb_model = xgboost(params = xg_params, data = ddd,nrounds =2700,verbose = 0)

# model output and scores
pred = predict(xgb_model,scaled_x_test)
mse_ensemble = mean((pred-ytest)^2)
print(mse_ensemble)

