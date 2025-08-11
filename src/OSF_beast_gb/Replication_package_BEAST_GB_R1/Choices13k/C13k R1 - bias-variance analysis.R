rm(list = ls())
library(xgboost)
library(dplyr)
library(ggplot2)


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
n_reps = 30

props_train = c(0.01, 0.1,  1)

nProportions = length(props_train)

set.seed(123)
dd <- dd[sample(nrow(dd)), ]  # shuffle entire dataset
test_ids <- sample(1:9831, 983)  # 10% as test
train_ids <- setdiff(seq_len(nrow(dd)), test_ids)

test_set  <- dd[test_ids, ]
train_set <- dd[train_ids, ]

feats <- c("Ha", "pHa", "La", "Hb", "pHb", "Lb",
           "LotShapeB", "LotNumB", "Corr",
           "diffEV", "diffSDs", "diffMins", "diffMaxs", 
           "diffUV", "RatioMin", "SignMax", "pBbet_UnbiasedFB", 
           "pBbet_Uniform", "pBbet_SignFB", "Dom", "diffSignEV", 
           "BEASTpred",
           "B_rate")

xy_train = train_set[,feats]
xy_test = test_set[,feats]

x_test <- model.matrix(B_rate ~ ., data = xy_test)[, -1]
y_test <- test_set$B_rate

n_test <- nrow(test_set)
pred_array <- array(NA, dim = c(n_test, nProportions, n_reps))
set.seed(2023)
for (rep in 1:n_reps){
  train_all_shuffled <- xy_train[sample(nrow(xy_train)), ]
  
  for (j in 1:nProportions){
    n_train_j <- floor(props_train[j] * nrow(train_all_shuffled))
    ttrain_set <- train_all_shuffled[seq_len(n_train_j), ]
    
    x_train = model.matrix(B_rate~., data= ttrain_set)[,-1]
    trainMean <- apply(x_train,2,mean)
    trainSd <- apply(x_train,2,sd)
    constant_columns <- which(trainSd == 0)
    y_train = ttrain_set$B_rate
    scaled_x_train <- sweep(sweep(x_train, 2L, trainMean), 2, trainSd, "/")
    scaled_x_train[, constant_columns] <- trainMean[constant_columns]

    # x_test = model.matrix(B_rate~.,data=xy_test)[,-1]
    scaled_x_test <- sweep(sweep(x_test, 2L, trainMean), 2, trainSd, "/")
    # y_test = test_set$B_rate
    scaled_x_test[, constant_columns] <- trainMean[constant_columns]

    ddd <- xgb.DMatrix(scaled_x_train,label=y_train)
    
    xgb_model = xgboost(params = xg_params, data = ddd,nrounds =1900,verbose = 0)
    
    pred = predict(xgb_model,scaled_x_test)
    pred_array[, j, rep] <- pred
    
    print(props_train[j])
    
    
  }
  
  cat("Finished repetition:", rep, "\n")
}

bias_variance_results <- data.frame(
  prop = props_train,
  bias_sq = NA_real_,
  var = NA_real_,
  mse = NA_real_
)

for(j in seq_len(nProportions)) {
  
  # Get the matrix of predictions for this proportion across runs:
  # shape [n_test, n_reps]
  pred_j <- pred_array[, j, ]
  
  # # Some fractions might have missing runs if n_train_j < 1
  # # so skip if it's all NA
  # if(all(is.na(pred_j))) {
  #   next
  # }
  
  # For convenience, remove any columns that are entirely NA
  keep_cols <- which(colSums(is.na(pred_j)) < n_test)
  pred_j <- pred_j[, keep_cols, drop=FALSE]
  Rj <- ncol(pred_j)
  
  obs   <- y_test
  # mean prediction across runs for each item:
  mean_pred <- rowMeans(pred_j, na.rm = TRUE)
  
  # BIAS^2
  bias2 <- mean((mean_pred - obs)^2)
  
  # VARIANCE
  # for each item, measure how spread out the runs' predictions are around mean_pred[i].
  var_item <- numeric(n_test)
  for(i in seq_len(n_test)) {
    # difference from the average
    diff_i <- pred_j[i, ] - mean_pred[i]
    var_item[i] <- mean(diff_i^2)
  }
  variance <- mean(var_item)
  
  # MSE
  mse_runs <- numeric(Rj)
  for(r in seq_len(Rj)) {
    mse_runs[r] <- mean((pred_j[, r] - obs)^2)
  }
  mse_ <- mean(mse_runs)
  
  bias_variance_results$bias_sq[j] <- bias2
  bias_variance_results$var[j]     <- variance
  bias_variance_results$mse[j]     <- mse_
}

bias_variance_results

################################# train curve no BEAST
# set.seed(123)

feats_no_BEAST <- c("Ha", "pHa", "La", "Hb", "pHb", "Lb",
           "LotShapeB", "LotNumB", "Corr",
           "diffEV", "diffSDs", "diffMins", "diffMaxs", 
           "diffUV", "RatioMin", "SignMax", "pBbet_UnbiasedFB", 
           "pBbet_Uniform", "pBbet_SignFB", "Dom", "diffSignEV", 
           "B_rate")

xy_train_noBEAST = train_set[,feats_no_BEAST]
xy_test_noBEAST = test_set[,feats_no_BEAST]

x_test_noBEAST <- model.matrix(B_rate ~ ., data = xy_test_noBEAST)[, -1]
y_test <- test_set$B_rate

pred_array_noBEAST <- array(NA, dim = c(n_test, nProportions, n_reps))
set.seed(2023)
for (rep in 1:n_reps){
  train_all_shuffled <- xy_train_noBEAST[sample(nrow(xy_train_noBEAST)), ]
  
  for (j in 1:nProportions){
    n_train_j <- floor(props_train[j] * nrow(train_all_shuffled))
    ttrain_set <- train_all_shuffled[seq_len(n_train_j), ]
    
    x_train = model.matrix(B_rate~., data= ttrain_set)[,-1]
    trainMean <- apply(x_train,2,mean)
    trainSd <- apply(x_train,2,sd)
    constant_columns <- which(trainSd == 0)
    y_train = ttrain_set$B_rate
    scaled_x_train <- sweep(sweep(x_train, 2L, trainMean), 2, trainSd, "/")
    scaled_x_train[, constant_columns] <- trainMean[constant_columns]
    
    # x_test = model.matrix(B_rate~.,data=xy_test)[,-1]
    scaled_x_test <- sweep(sweep(x_test_noBEAST, 2L, trainMean), 2, trainSd, "/")
    # y_test = test_set$B_rate
    scaled_x_test[, constant_columns] <- trainMean[constant_columns]
    
    ddd <- xgb.DMatrix(scaled_x_train,label=y_train)
    
    xgb_model = xgboost(params = xg_params, data = ddd,nrounds =1900,verbose = 0)
    
    pred = predict(xgb_model,scaled_x_test)
    pred_array_noBEAST[, j, rep] <- pred
    
    print(props_train[j])
    
  }
  
  cat("Finished repetition:", rep, "\n")
}


noBEAST_bias_variance_results <- data.frame(
  prop = props_train,
  bias_sq = NA_real_,
  var = NA_real_,
  mse = NA_real_
)

for(j in seq_len(nProportions)) {
  
  # Get the matrix of predictions for this proportion across runs:
  # shape [n_test, n_reps]
  pred_j <- pred_array_noBEAST[, j, ]
  
  # # Some fractions might have missing runs if n_train_j < 1
  # # so skip if it's all NA
  # if(all(is.na(pred_j))) {
  #   next
  # }
  
  # For convenience, remove any columns that are entirely NA
  keep_cols <- which(colSums(is.na(pred_j)) < n_test)
  pred_j <- pred_j[, keep_cols, drop=FALSE]
  Rj <- ncol(pred_j)
  
  obs   <- y_test
  # mean prediction across runs for each item:
  mean_pred <- rowMeans(pred_j, na.rm = TRUE)
  
  # BIAS^2
  bias2 <- mean((mean_pred - obs)^2)
  
  # VARIANCE
  # for each item, measure how spread out the runs' predictions are around mean_pred[i].
  var_item <- numeric(n_test)
  for(i in seq_len(n_test)) {
    # difference from the average
    diff_i <- pred_j[i, ] - mean_pred[i]
    var_item[i] <- mean(diff_i^2)
  }
  variance <- mean(var_item)
  
  # MSE
  mse_runs <- numeric(Rj)
  for(r in seq_len(Rj)) {
    mse_runs[r] <- mean((pred_j[, r] - obs)^2)
  }
  mse_ <- mean(mse_runs)
  
  noBEAST_bias_variance_results$bias_sq[j] <- bias2
  noBEAST_bias_variance_results$var[j]     <- variance
  noBEAST_bias_variance_results$mse[j]     <- mse_
}

noBEAST_bias_variance_results

noBEAST_bias_variance_results- bias_variance_results