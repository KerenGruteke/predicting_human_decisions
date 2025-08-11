rm(list = ls())
library(xgboost)
library(dplyr)
library(SHAPforxgboost)
library(ggplot2)
library(BayesFactor)

setwd("C:/Users/plonsky/Dropbox/CPC 2018/NHB23 files/R1 files/Final files for upload/Choices13k")

####
load("all features for PF BeastGB c13k.RData") 

# get standard error squared
dd13 = read.csv('c13k_selections.csv')
dd13 = dd13[order(dd13$Problem),]
dd13_clean = dd13[dd13$Feedback==TRUE & dd13$Amb==FALSE,]
dd13_clean$SE2 = (dd13_clean$bRate_std^2)/dd13_clean$n
lot_shape_levels <- c("-", "Symm", "R-skew", "L-skew")
dd13_clean$LotShapeB <- factor(dd13_clean$LotShapeB, levels = 0:3, labels = lot_shape_levels)

dd = merge(dd,dd13_clean)
dd = dd[,c(1:9,11:17,19:20,22:23,26:28, 41)]

####################################### run tuned BEAST-GB

xg_params = list(
  'colsample_bytree' = 0.4,
  'gamma'= 0.04,
  'learning_rate'= 0.01,
  'max_depth'= 6,
  'subsample'= 0.55,
  'min_child_weight' = 3
)

################################# train curve
n_reps = 50

props_train = c(0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 
                0.15, 0.2,  0.25, 0.3, 0.35, 0.4, 0.45, 0.5,  0.55, 0.6,
                0.65, 0.7,  0.75, 0.8,  0.85, 0.9,  0.95, 1)

nProportions = length(props_train)
mses = rep(NA, nProportions)
all_mses = vector("list", n_reps)
irreducible = NA
naive_err = NA
completeness = NA

feats <- c("Ha", "pHa", "La", "Hb", "pHb", "Lb",
           "LotShapeB", "LotNumB", "Corr",
           "diffEV", "diffSDs", "diffMins", "diffMaxs", 
           "diffUV", "RatioMin", "SignMax", "pBbet_UnbiasedFB", 
           "pBbet_Uniform", "pBbet_SignFB", "Dom", "diffSignEV", 
           "BEASTpred",
           "B_rate")
all_preds_df = data.frame()

set.seed(2023)
for (cvid in 1:n_reps){
  dd = dd[sample(nrow(dd)),] # shuffle
  test_ids = sample(1:9831, 983)
  train_ids = rep(TRUE, 9831)
  train_ids[test_ids] = FALSE
  test_set = dd[test_ids,]
  train_set = dd[train_ids,]
  xy_train = train_set[,feats]
  xy_test = test_set[,feats]
  
  for (j in 1:nProportions){
    ttrain_set = xy_train[1:(props_train[j]*nrow(xy_train)),]
    
    x_train = model.matrix(B_rate~0+., data= ttrain_set)
    trainMean <- apply(x_train,2,mean)
    trainSd <- apply(x_train,2,sd)
    constant_columns <- which(trainSd == 0)
    y_train = ttrain_set$B_rate
    scaled_x_train <- sweep(sweep(x_train, 2L, trainMean), 2, trainSd, "/")
    scaled_x_train[, constant_columns] <- trainMean[constant_columns]
    
    x_test = model.matrix(B_rate~0+.,data=xy_test)
    scaled_x_test <- sweep(sweep(x_test, 2L, trainMean), 2, trainSd, "/")
    y_test = test_set$B_rate
    scaled_x_test[, constant_columns] <- trainMean[constant_columns]

    ddd <- xgb.DMatrix(scaled_x_train,label=y_train)
    
    xgb_model = xgboost(params = xg_params, data = ddd,nrounds =1900,verbose = 0)
    
    pred = predict(xgb_model,scaled_x_test)
    jointPredObs = data.frame(xy_test,pred)

    mses[j] = mean((jointPredObs$B_rate - jointPredObs$pred)^2)
    
    print(props_train[j])
  }
  
  jointPredObs$rep <- cvid
  all_preds_df <- rbind(all_preds_df, jointPredObs)
  
  all_mses[[cvid]] = mses
  
  irreducible[cvid] = mean(test_set$SE2)
  naive_err[cvid] = mean((test_set$B_rate - 0.5)^2)
  completeness[cvid] = 
    (naive_err[cvid] - all_mses[[cvid]][nProportions])/(naive_err[cvid] - irreducible[cvid])
  
  tmp = shap.prep(xgb_model = xgb_model, X_train = scaled_x_test)

  if (cvid!=1){
    shap_long = rbind(shap_long, tmp)
  } else {
    shap_long = tmp
  }
  
  cat("finished repetition ", cvid, "\n")
}

all_mses_matrix = do.call(rbind, all_mses)
mean_mse_with_training_data = colMeans(all_mses_matrix)
plot_data = data.frame(props_train, mean_mse_with_training_data)
cat("Mean completeness over 50 repetitions is ", mean(completeness))
cat("Mean MSE across 50 repetitions is ", 
    plot_data$mean_mse_with_training_data[plot_data$props_train==1])
print(plot_data$mean_mse_with_training_data[plot_data$props_train==0.02])

nfeats = ncol(x_test)
ntest_obs = 983*n_reps
shap_long_long = shap_long[order(shap_long$variable),]
shap_long_long$ID = rep(1:ntest_obs,nfeats)
for (var in 1:nfeats){
  shap_long_long$mean_value[(((var-1)*ntest_obs)+1):(ntest_obs*var)] =
    mean(abs(shap_long_long$value[(((var-1)*ntest_obs)+1):(ntest_obs*var)]))
}
ordering <- order(-shap_long_long$mean_value)
ordered_levels <- unique(shap_long_long$variable[ordering])
shap_long_long$variable <- factor(shap_long_long$variable, levels = ordered_levels)
shap.plot.summary(shap_long_long, kind="bar")
mean_shap = unique(tmp[,c(2,6)])
mean_shap = mean_shap[order(-mean_shap$mean_value)]

################################# train curve no BEAST
mses_noBEAST = rep(NA, nProportions)
all_mses_noBEAST = vector("list", n_reps)
feats_no_BEAST <- c("Ha", "pHa", "La", "Hb", "pHb", "Lb",
                    "LotShapeB", "LotNumB", "Corr",           
                    "diffEV", "diffSDs", "diffMins", "diffMaxs", 
                    "diffUV", "RatioMin", "SignMax", "pBbet_UnbiasedFB", 
                    "pBbet_Uniform", "pBbet_SignFB", "Dom", "diffSignEV", 
                    "B_rate")
set.seed(2023)
for (cvid in 1:n_reps){
  dd = dd[sample(nrow(dd)),]
  test_ids = sample(1:9831, 983)
  train_ids = rep(TRUE, 9831)
  train_ids[test_ids] = FALSE
  test_set = dd[test_ids,]
  train_set = dd[train_ids,]
  xy_train = train_set[,feats_no_BEAST]
  xy_test = test_set[,feats_no_BEAST]
  
  for (j in 1:nProportions){
    ttrain_set = xy_train[1:(props_train[j]*nrow(xy_train)),]
    
    x_train = model.matrix(B_rate~0+., data= ttrain_set)
    trainMean <- apply(x_train,2,mean)
    trainSd <- apply(x_train,2,sd)
    constant_columns <- which(trainSd == 0)
    y_train = ttrain_set$B_rate
    scaled_x_train <- sweep(sweep(x_train, 2L, trainMean), 2, trainSd, "/")
    scaled_x_train[, constant_columns] <- trainMean[constant_columns]
    
    x_test = model.matrix(B_rate~0+.,data=xy_test)
    scaled_x_test <- sweep(sweep(x_test, 2L, trainMean), 2, trainSd, "/")
    y_test = test_set$B_rate
    scaled_x_test[, constant_columns] <- trainMean[constant_columns]

    ddd <- xgb.DMatrix(scaled_x_train,label=y_train)
    
    xgb_model = xgboost(params = xg_params, data = ddd,nrounds =1900,verbose = 0)
    
    pred = predict(xgb_model,scaled_x_test)
    jointPredObs = data.frame(xy_test,pred)

    mses_noBEAST[j] = mean((jointPredObs$B_rate - jointPredObs$pred)^2)
  }
  all_mses_noBEAST[[cvid]] = mses_noBEAST

  print(cvid)
  
}
all_mses_matrix_noBEAST = do.call(rbind, all_mses_noBEAST)
mean_mse_with_training_data_noBEAST = colMeans(all_mses_matrix_noBEAST)
plot_data = data.frame(props_train, mean_mse_with_training_data_noBEAST)


################################# train curve no Psych
mses_noPsyc = rep(NA, nProportions)
all_mses_noPsych = vector("list", n_reps)
feats_no_psych <- c("Ha", "pHa", "La", "Hb", "pHb", "Lb",
                    "LotShapeB", "LotNumB", "Corr",           
                    "diffEV", "diffSDs", "diffMins", "diffMaxs", 
                    "BEASTpred",
                    "B_rate")
set.seed(2023)
for (cvid in 1:n_reps){
  dd = dd[sample(nrow(dd)),]
  test_ids = sample(1:9831, 983)
  train_ids = rep(TRUE, 9831)
  train_ids[test_ids] = FALSE
  test_set = dd[test_ids,]
  train_set = dd[train_ids,]
  xy_train = train_set[,feats_no_psych]
  xy_test = test_set[,feats_no_psych]
  
  for (j in 1:nProportions){
    ttrain_set = xy_train[1:(props_train[j]*nrow(xy_train)),]
    
    x_train = model.matrix(B_rate~0+., data= ttrain_set)
    trainMean <- apply(x_train,2,mean)
    trainSd <- apply(x_train,2,sd)
    constant_columns <- which(trainSd == 0)
    y_train = ttrain_set$B_rate
    scaled_x_train <- sweep(sweep(x_train, 2L, trainMean), 2, trainSd, "/")
    scaled_x_train[, constant_columns] <- trainMean[constant_columns]
    
    x_test = model.matrix(B_rate~0+.,data=xy_test)
    scaled_x_test <- sweep(sweep(x_test, 2L, trainMean), 2, trainSd, "/")
    y_test = test_set$B_rate
    scaled_x_test[, constant_columns] <- trainMean[constant_columns]

    ddd <- xgb.DMatrix(scaled_x_train,label=y_train)
    
    xgb_model = xgboost(params = xg_params, data = ddd,nrounds =1900,verbose = 0)
    
    pred = predict(xgb_model,scaled_x_test)
    jointPredObs = data.frame(xy_test,pred)

    mses_noPsyc[j] = mean((jointPredObs$B_rate - jointPredObs$pred)^2)
  }
  all_mses_noPsych[[cvid]] = mses_noPsyc
  
  print(cvid)
  
}
all_mses_matrix_noPsych = do.call(rbind, all_mses_noPsych)
mean_mse_with_training_data_noPsych = colMeans(all_mses_matrix_noPsych)
plot_data = data.frame(props_train, mean_mse_with_training_data_noPsych)


################################# train curve no Psych or BEAST
mses_noPsycBEAST = rep(NA, nProportions)
all_mses_noPsychBEAST = vector("list", n_reps)
feats_no_behavioral <- c("Ha", "pHa", "La", "Hb", "pHb", "Lb",
                         "LotShapeB", "LotNumB", "Corr",
                         "diffEV", "diffSDs", "diffMins", "diffMaxs", 
                         "B_rate")
set.seed(2023)
for (cvid in 1:n_reps){
  dd = dd[sample(nrow(dd)),]
  test_ids = sample(1:9831, 983)
  train_ids = rep(TRUE, 9831)
  train_ids[test_ids] = FALSE
  test_set = dd[test_ids,]
  train_set = dd[train_ids,]
  xy_train = train_set[,feats_no_behavioral]
  xy_test = test_set[,feats_no_behavioral]
  
  for (j in 1:nProportions){
    ttrain_set = xy_train[1:(props_train[j]*nrow(xy_train)),]
    
    x_train = model.matrix(B_rate~0+., data= ttrain_set)
    trainMean <- apply(x_train,2,mean)
    trainSd <- apply(x_train,2,sd)
    constant_columns <- which(trainSd == 0)
    y_train = ttrain_set$B_rate
    scaled_x_train <- sweep(sweep(x_train, 2L, trainMean), 2, trainSd, "/")
    scaled_x_train[, constant_columns] <- trainMean[constant_columns]
    
    x_test = model.matrix(B_rate~0+.,data=xy_test)
    scaled_x_test <- sweep(sweep(x_test, 2L, trainMean), 2, trainSd, "/")
    y_test = test_set$B_rate
    scaled_x_test[, constant_columns] <- trainMean[constant_columns]

    ddd <- xgb.DMatrix(scaled_x_train,label=y_train)
    
    xgb_model = xgboost(params = xg_params, data = ddd,nrounds =1900,verbose = 0)
    
    pred = predict(xgb_model,scaled_x_test)
    jointPredObs = data.frame(xy_test,pred)

    mses_noPsycBEAST[j] = mean((jointPredObs$B_rate - jointPredObs$pred)^2)
  }
  all_mses_noPsychBEAST[[cvid]] = mses_noPsycBEAST
 
  print(cvid)
  
}
all_mses_matrix_noPsychBEAST = do.call(rbind, all_mses_noPsychBEAST)
mean_mse_with_training_data_noPsychBEAST = colMeans(all_mses_matrix_noPsychBEAST)
plot_data = data.frame(props_train, mean_mse_with_training_data_noPsychBEAST)

################################# train curve no naive
mses_noNaive = rep(NA, nProportions)
all_mses_noNaive = vector("list", n_reps)
feats_no_naive <- c("Ha", "pHa", "La", "Hb", "pHb", "Lb",
                    "LotShapeB", "LotNumB", "Corr",
                    "diffEV", "diffMins", 
                    "diffUV", "RatioMin", "SignMax", "pBbet_UnbiasedFB", 
                    "pBbet_Uniform", "pBbet_SignFB", "Dom", "diffSignEV", 
                    "BEASTpred",
                    "B_rate")
set.seed(2023)
for (cvid in 1:n_reps){
  dd = dd[sample(nrow(dd)),]
  test_ids = sample(1:9831, 983)
  train_ids = rep(TRUE, 9831)
  train_ids[test_ids] = FALSE
  test_set = dd[test_ids,]
  train_set = dd[train_ids,]
  xy_train = train_set[,feats_no_naive]
  xy_test = test_set[,feats_no_naive]
  
  for (j in 1:nProportions){
    ttrain_set = xy_train[1:(props_train[j]*nrow(xy_train)),]
    
    x_train = model.matrix(B_rate~0+., data= ttrain_set)
    trainMean <- apply(x_train,2,mean)
    trainSd <- apply(x_train,2,sd)
    constant_columns <- which(trainSd == 0)
    y_train = ttrain_set$B_rate
    scaled_x_train <- sweep(sweep(x_train, 2L, trainMean), 2, trainSd, "/")
    scaled_x_train[, constant_columns] <- trainMean[constant_columns]
    
    x_test = model.matrix(B_rate~0+.,data=xy_test)
    scaled_x_test <- sweep(sweep(x_test, 2L, trainMean), 2, trainSd, "/")
    y_test = test_set$B_rate
    scaled_x_test[, constant_columns] <- trainMean[constant_columns]

    ddd <- xgb.DMatrix(scaled_x_train,label=y_train)
    
    xgb_model = xgboost(params = xg_params, data = ddd,nrounds =1900,verbose = 0)
    
    pred = predict(xgb_model,scaled_x_test)
    jointPredObs = data.frame(xy_test,pred)

    mses_noNaive[j] = mean((jointPredObs$B_rate - jointPredObs$pred)^2)
  }
  all_mses_noNaive[[cvid]] = mses_noNaive
  
  print(cvid)
  
}
all_mses_matrix_noNaive = do.call(rbind, all_mses_noNaive)
mean_mse_with_training_data_noNaive = colMeans(all_mses_matrix_noNaive)
plot_data = data.frame(props_train, mean_mse_with_training_data_noNaive)

################################# train curve ONLY objective
mses_justObj = rep(NA, nProportions)
all_mses_justObj = vector("list", n_reps)
feats_justObj <- c("Ha", "pHa", "La", "Hb", "pHb", "Lb",
                    "LotShapeB", "LotNumB", "Corr",
                    "B_rate")
set.seed(2023)
for (cvid in 1:n_reps){
  dd = dd[sample(nrow(dd)),]
  test_ids = sample(1:9831, 983)
  train_ids = rep(TRUE, 9831)
  train_ids[test_ids] = FALSE
  test_set = dd[test_ids,]
  train_set = dd[train_ids,]
  xy_train = train_set[,feats_justObj]
  xy_test = test_set[,feats_justObj]
  
  for (j in 1:nProportions){
    ttrain_set = xy_train[1:(props_train[j]*nrow(xy_train)),]
    
    x_train = model.matrix(B_rate~0+., data= ttrain_set)
    trainMean <- apply(x_train,2,mean)
    trainSd <- apply(x_train,2,sd)
    constant_columns <- which(trainSd == 0)
    y_train = ttrain_set$B_rate
    scaled_x_train <- sweep(sweep(x_train, 2L, trainMean), 2, trainSd, "/")
    scaled_x_train[, constant_columns] <- trainMean[constant_columns]
    
    x_test = model.matrix(B_rate~0+.,data=xy_test)
    scaled_x_test <- sweep(sweep(x_test, 2L, trainMean), 2, trainSd, "/")
    y_test = test_set$B_rate
    scaled_x_test[, constant_columns] <- trainMean[constant_columns]
    
    ddd <- xgb.DMatrix(scaled_x_train,label=y_train)
    
    xgb_model = xgboost(params = xg_params, data = ddd,nrounds =1900,verbose = 0)
    
    pred = predict(xgb_model,scaled_x_test)
    jointPredObs = data.frame(xy_test,pred)
    
    mses_justObj[j] = mean((jointPredObs$B_rate - jointPredObs$pred)^2)
  }
  all_mses_justObj[[cvid]] = mses_justObj
  
  print(cvid)
  
}
all_mses_matrix_justObj = do.call(rbind, all_mses_justObj)
mean_mse_with_training_data_justObj = colMeans(all_mses_matrix_justObj)
plot_data_j = data.frame(props_train, mean_mse_with_training_data_justObj)

################################# train curve NO objective
mses_noObj = rep(NA, nProportions)
all_mses_noObj = vector("list", n_reps)
feats_no_noObj <- c("diffEV", "diffSDs", "diffMins", "diffMaxs", 
                    "diffUV", "RatioMin", "SignMax", "pBbet_UnbiasedFB", 
                    "pBbet_Uniform", "pBbet_SignFB", "Dom", "diffSignEV", 
                    "BEASTpred",
                    "B_rate")
set.seed(2023)
for (cvid in 1:n_reps){
  dd = dd[sample(nrow(dd)),]
  test_ids = sample(1:9831, 983)
  train_ids = rep(TRUE, 9831)
  train_ids[test_ids] = FALSE
  test_set = dd[test_ids,]
  train_set = dd[train_ids,]
  xy_train = train_set[,feats_no_noObj]
  xy_test = test_set[,feats_no_noObj]
  
  for (j in 1:nProportions){
    ttrain_set = xy_train[1:(props_train[j]*nrow(xy_train)),]
    
    x_train = model.matrix(B_rate~0+., data= ttrain_set)
    trainMean <- apply(x_train,2,mean)
    trainSd <- apply(x_train,2,sd)
    constant_columns <- which(trainSd == 0)
    y_train = ttrain_set$B_rate
    scaled_x_train <- sweep(sweep(x_train, 2L, trainMean), 2, trainSd, "/")
    scaled_x_train[, constant_columns] <- trainMean[constant_columns]
    
    x_test = model.matrix(B_rate~0+.,data=xy_test)
    scaled_x_test <- sweep(sweep(x_test, 2L, trainMean), 2, trainSd, "/")
    y_test = test_set$B_rate
    scaled_x_test[, constant_columns] <- trainMean[constant_columns]
    
    ddd <- xgb.DMatrix(scaled_x_train,label=y_train)
    
    xgb_model = xgboost(params = xg_params, data = ddd,nrounds =1900,verbose = 0)
    
    pred = predict(xgb_model,scaled_x_test)
    jointPredObs = data.frame(xy_test,pred)
    
    mses_noObj[j] = mean((jointPredObs$B_rate - jointPredObs$pred)^2)
  }
  all_mses_noObj[[cvid]] = mses_noObj
  
  print(cvid)
  
}
all_mses_matrix_noObj = do.call(rbind, all_mses_noObj)
mean_mse_with_training_data_noObj = colMeans(all_mses_matrix_noObj)
plot_data = data.frame(props_train, mean_mse_with_training_data_noObj)

#######

plot_data = data.frame(props_train,
                       mean_mse_with_training_data,
                       mean_mse_with_training_data_noObj,
                       mean_mse_with_training_data_noBEAST,
                       mean_mse_with_training_data_noPsych,
                       mean_mse_with_training_data_noNaive,
                       mean_mse_with_training_data_noPsychBEAST
                       )

# save.image("C13k R1 - all training curves by feature set.RData")

t.test(all_mses_matrix[,28], all_mses_matrix_noBEAST[,28], paired = TRUE)
bf <- ttestBF(
  x = all_mses_matrix[,28],
  y = all_mses_matrix_noBEAST[,28],
  paired = TRUE,
  rscale = "medium"  # Default, corresponds to a Cauchy prior with r = 0.707
)
t.test(all_mses_matrix[,28], all_mses_matrix_noPsych[,28], paired = TRUE)
t.test(all_mses_matrix[,28], all_mses_matrix_noNaive[,28], paired = TRUE)
t.test(all_mses_matrix[,28], all_mses_matrix_noPsychBEAST[,28], paired = TRUE)
t.test(all_mses_matrix[,28], all_mses_matrix_noObj[,28], paired = TRUE)


################################# performance only BEAST
mses_beast = rep(NA,n_reps)
irreducible = NA
naive_err = NA
completeness = NA

set.seed(2023)
for (cvid in 1:n_reps){
  dd = dd[sample(nrow(dd)),]
  test_ids = sample(1:9831, 983)
  train_ids = rep(TRUE, 9831)
  train_ids[test_ids] = FALSE
  test_set = dd[test_ids,]
  train_set = dd[train_ids,]
  xy_train = train_set[,c(1:23)]
  xy_test = test_set[,c(1:23)]
  
  mses_beast[cvid] = mean((xy_test$B_rate - xy_test$BEASTpred)^2)
  
  irreducible[cvid] = mean(test_set$SE2)
  naive_err[cvid] = mean((test_set$B_rate - 0.5)^2)
  completeness[cvid] = (naive_err[cvid] - mses_beast[cvid])/(naive_err[cvid] - irreducible[cvid])
  
  print(cvid)
  
}
print(mean(completeness))
print(mean(mses_beast))
