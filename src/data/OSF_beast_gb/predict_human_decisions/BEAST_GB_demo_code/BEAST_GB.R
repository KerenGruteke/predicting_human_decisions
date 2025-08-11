# rm(list = ls())

library(xgboost)
library(dplyr)
setwd("C:/Users/plonsky/Dropbox (Technion Dropbox)/CPC 2018/NHB23 files/Final files for upload/BEAST_GB sample code")
source('funcs for feature engineering.R')

################################
###### Feature extraction ######
################################
# load tasks to to engineer features for
all_data = read.csv("CPC18_data.csv", header = T)

feats <- data.frame(matrix(ncol = 31, nrow = 0))
for (prob in 1:nrow(all_data)) {
  
  # In CPC18 train data, there are two instances of problems with the same
  # set of parameters (and thus features) - see Erev et al. 2017 for origins of this
  if (prob == 13 | prob == 20) {  
    next  
  }
  
  #read problem's parameters
  Ha = all_data$Ha[prob]
  pHa = all_data$pHa[prob]
  La = all_data$La[prob]
  LotShapeA = all_data$LotShapeA[prob]
  LotNumA = all_data$LotNumA[prob]
  Hb = all_data$Hb[prob]
  pHb = all_data$pHb[prob]
  Lb = all_data$Lb[prob]
  LotShapeB = all_data$LotShapeB[prob]
  LotNumB = all_data$LotNumB[prob]
  Amb = all_data$Amb[prob]
  Corr = all_data$Corr[prob]
  
  feats_tmp = c(Ha, pHa, La, LotShapeA, LotNumA, 
                              Hb, pHb, Lb, LotShapeB, LotNumB, 
                              Amb, Corr,
                              nBlocks = 5, nTrials = 25, firstFeedbackBlock = 2)
  
  feats = rbind(feats, feats_tmp)
  if (prob == 1) {
    colnames(feats) <- names(feats_tmp)
  }
  
  if (prob %% 5 == 0){
    print(prob) # for verbose progression
  }
    
}

###### merge back features
all_data_long = reshape(all_data, direction = "long", timevar = "block", 
                        v.names = "B_rate", varying = c("B.1", "B.2", "B.3", "B.4", "B.5"))
all_data_long$id = NULL
data_with_features = left_join(all_data_long, feats)

# it is advised that the extracted features will be saved before the next steps.
# save(data_with_features,file = "feats_cpc18.RData") 

#################################
###### Data pre-processing ######
#################################

#### separate to train and test sets
xy_train = data_with_features[data_with_features$Train_Test == "Train",]
xy_test = data_with_features[data_with_features$Train_Test == "Test",]

#### remove constant columns
constant_train_columns = sapply(xy_train, function(x) length(unique(x)) == 1)
xy_train = xy_train[, !constant_train_columns]
xy_test = xy_test[, !constant_train_columns]

#### dummy coding categorical variables
x_train = model.matrix(B_rate~., data= xy_train)[,-1]
x_test = model.matrix(B_rate~.,data=xy_test)[,-1]

#### standardize features
trainMean = apply(x_train,2,mean)
trainSd = apply(x_train,2,sd)
scaled_x_train = sweep(sweep(x_train, 2, trainMean), 2, trainSd, "/")
scaled_x_test <- sweep(sweep(x_test, 2, trainMean), 2, trainSd, "/")

#### get labels
y_train = xy_train$B_rate
y_test = xy_test$B_rate

#### Setting hyperparameters 
# (note the values hereafter are those submitted by the winners of CPC18 and 
# are suitable for this data. For different data, a tuning process is required)
xg_params = list(
  'booster' = 'gbtree',
  'colsample_bytree' = 0.9911994607412087,
  'gamma'= 0.012241954019987821,
  'learning_rate'= 0.010878922437398755,
  'max_depth'= 3,
  'reg_alpha'= 0.04306269451141776,
  'reg_lambda'= 2.9053833404234397,
  'subsample'= 0.5079640412046551,
  'n_jobs'= -1
)

##########################################
###### Train, predict, and evaluate ######
##########################################

#### train BEAST-GB and generate predictions
ddd = xgb.DMatrix(scaled_x_train,label=y_train)
BEAST_GB = xgboost(params = xg_params, data = ddd, nrounds =978, verbose = 0)

#### model output and scores
pred = predict(BEAST_GB, scaled_x_test)
jointPredObs = data.frame(xy_test,pred)
mse_BEAST_GB = mean((jointPredObs$B_rate - jointPredObs$pred)^2)
print(mse_BEAST_GB)
