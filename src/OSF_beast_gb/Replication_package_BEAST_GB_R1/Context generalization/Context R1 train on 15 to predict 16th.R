rm(list = ls())
library(xgboost)
library(dplyr)
library(ggplot2)
library(tictoc)

setwd("C:/Users/plonsky/Dropbox/CPC 2018/NHB23 files/R1 files/Final files for upload")

####
load('HAB22/HAB22 all data with Stew1C_Uni.RData')

all_tasks = unique(HAB22[,c(1:9,73:90)])

########## get context generalization predictions of BEAST-GB for 16th dataset
xg_params = list(
  'colsample_bytree' = 0.55,
  'gamma'= 0.01,
  'learning_rate'= 0.01,
  'max_depth'= 5,
  'subsample'= 0.25,
  'min_child_weight'= 3
)
dss = unique(all_tasks$dataset)
dataset_mses = rep(NA, length(dss))
naive_irreduc = array(dimnames = list(dss, c("naive", "irreducible")), dim = c(length(dss), 2))
feats <- c("Ha", "pHa", "La", "Hb", "pHb", "Lb", 
           "diffEV", "diffSDs", "diffMins", "diffMaxs", 
           "diffUV", "RatioMin", "SignMax", "pBbet_Unbiased1", 
           "pBbet_Uniform", "pBbet_Sign1", "Dom", "diffSignEV", 
           "BEASTpred",
           "B_rate")
set.seed(2023)
for (ds in 1:length(dss)){
  train_set = all_tasks[!(all_tasks$dataset == dss[ds]), ]
  test_set = all_tasks[all_tasks$dataset == dss[ds], ]
  subj_test_set = HAB22[HAB22$dataset == dss[ds], ]
  y_test_set = aggregate(choice~task_id, data=subj_test_set, 
                         FUN = function(x) c(mean = mean(x), std = sd(x), n = length(x), SE2 = (sd(x))^2/length(x)))
  test_set = merge(test_set, y_test_set)
  
  xy_train = train_set[,feats]
  cols_zero_var = apply(xy_train, 2, var, na.rm=TRUE)==0
  xy_train = xy_train[,apply(xy_train, 2, var, na.rm=TRUE) != 0] #remove constant columns
  xy_test = test_set[,feats]
  xy_test = xy_test[,!cols_zero_var] #remove constant train columns 
  
  x_train = model.matrix(B_rate~0+., data= xy_train)
  trainMean <- apply(x_train,2,mean)
  trainSd <- apply(x_train,2,sd)
  y_train = xy_train$B_rate
  scaled_x_train <- sweep(sweep(x_train, 2L, trainMean), 2, trainSd, "/")
  
  x_test = model.matrix(B_rate~0+.,data=xy_test)
  scaled_x_test <- sweep(sweep(x_test, 2L, trainMean), 2, trainSd, "/")
  y_test = xy_test$B_rate
  
  mmm <- xgb.DMatrix(scaled_x_train,label=y_train)
  
  xgb_model = xgboost(params = xg_params, data = mmm,nrounds =1800,verbose = 0)
  
  pred = predict(xgb_model,scaled_x_test)
  
  jointPredObs = data.frame(test_set,pred)
  
  if (ds==1){
    pred_all = jointPredObs
  } else {
    pred_all = rbind(pred_all, jointPredObs)
  }
  
  dataset_mses[ds] = mean((jointPredObs$B_rate - jointPredObs$pred)^2)
  naive_err = mean((jointPredObs$B_rate - 0.5)^2)
  irreducible = mean(jointPredObs$choice[,"SE2"])
  naive_irreduc[ds,] = c(naive_err, irreducible)
  print(paste("dataset ",ds, dss[ds]))
  
}
mse = mean((pred_all$B_rate - pred_all$pred)^2)
cat("mean BEAST-GB MSE over all problems: ", mse)
cat("mean BEAST-GB MSE in a dataset: ", mean(dataset_mses))
cat("SD BEAST-GB MSE in a dataset: ", sd(dataset_mses))

completness_xgb <- rep(NA, length(dss))
for (ds in 1:length(dss)) {
  naive_err <- naive_irreduc[ds, 1]
  irreducible <- naive_irreduc[ds, 2]
  val <- dataset_mses[ds]
  completness_xgb[ds] <- (naive_err - val) / (naive_err - irreducible)
}
mean_completness_xgb = mean(completness_xgb)
cat("mean BEAST-GB Completeness in a dataset: ", mean_completness_xgb)
sd(completness_xgb)
