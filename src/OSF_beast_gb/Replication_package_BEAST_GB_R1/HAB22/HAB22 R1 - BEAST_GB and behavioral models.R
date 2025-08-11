rm(list = ls())
library(xgboost)
library(dplyr)
library(SHAPforxgboost)
library(ggplot2)

setwd("C:/Users/plonsky/Dropbox/CPC 2018/NHB23 files/R1 files/Final files for upload/HAB22")

####

load('HAB22 all data with Stew1C_Uni.RData')

# for the main analysis, remove the Stewart15_1C_Uniform dataset
HAB22 = HAB22[HAB22$dataset != "Stewart15_1C_uniform",]

######### create subject folds
subjs_ids = unique(paste(sep="_", HAB22$dataset, HAB22$subject))  
df = unique(HAB22[,c(1,2,11)])
df$fold <- NA
nfold_subjs = 5
set.seed(123)  # for reproducibility
for (i in unique(df$dataset)) {
  group_df <- df %>% filter(dataset == i)
  group_df$fold <- as.numeric(cut(sample(seq_len(nrow(group_df))), 
                                  breaks = nfold_subjs, labels = FALSE))
  df$fold[df$dataset == i & df$subject %in% group_df$subject] <- group_df$fold
}
df = df[order(df$dataset, df$fold, df$subject),]
ddd = merge(HAB22, df)

######### get MSEs for all behavioral models with CV over both problems and subjects
cols_to_mean = names(ddd)[c(13:72,90)]
nfold_tasks = 10
mses_behavioral_models_in_sample = array(dimnames = list(1:nfold_subjs, cols_to_mean, 1:nfold_tasks), 
                                         dim = c(nfold_subjs, 61, nfold_tasks))
mses_behavioral_models = array(dimnames = list(1:nfold_subjs, cols_to_mean, 1:nfold_tasks), 
                               dim = c(nfold_subjs, 61, nfold_tasks))
completeness_behavioral_models = array(dimnames = list(1:nfold_subjs, cols_to_mean, 1:nfold_tasks), 
                                       dim = c(nfold_subjs, 61, nfold_tasks))
naive_irreduc = array(dimnames = list(1:nfold_subjs, c("naive","irreducible"), 1:nfold_tasks), 
                      dim = c(nfold_subjs, 2, nfold_tasks))
set.seed(2023)
for (cvid in 1:nfold_tasks){
  probs_train_set = ddd[ddd$crossValidation_id != cvid,]
  probs_test_set = ddd[ddd$crossValidation_id == cvid,]
  for (cvsubjs in 1:nfold_subjs){
    test_set_in_sample = probs_test_set[probs_test_set$fold != cvsubjs,]
    
    # models' predictions (for both in and out of sample subjects)
    # are the average of predictions for subjects in-sample 
    preds_behavioral_models = test_set_in_sample %>% 
      select(task_id, all_of(cols_to_mean)) %>% 
      group_by(task_id) %>% 
      summarize_all(mean, na.rm = TRUE)
    
    # compute the models' prediction error on in-sample subjects (out of sample tasks)
    y_test_set_in_sample = aggregate(choice~task_id, data=test_set_in_sample, mean) 
    tmp_in_sample = merge(preds_behavioral_models, y_test_set_in_sample)
    tmp = sapply(tmp_in_sample[,cols_to_mean], function(x) mean((x - tmp_in_sample$choice)^2))
    mses_behavioral_models_in_sample[cvsubjs,,cvid] = tmp
    
    # compute the models' prediction error and completeness on out of sample subjects (and tasks)
    test_set_out_of_sample = probs_test_set[probs_test_set$fold == cvsubjs,]
    y_test_set_out_of_sample = aggregate(choice~task_id, data=test_set_out_of_sample,  
                                         FUN = function(x) c(mean = mean(x), std = sd(x), n = length(x), SE2 = (sd(x))^2/length(x)))
    
    tmp_out_sample = merge(preds_behavioral_models, y_test_set_out_of_sample)
    tmp = sapply(tmp_out_sample[,cols_to_mean], function(x) mean((x - tmp_out_sample$choice[,"mean"])^2))
    mses_behavioral_models[cvsubjs,,cvid] = tmp
    naive_err = mean((0.5 - tmp_out_sample$choice[,"mean"])^2)
    irreducible = mean(y_test_set_out_of_sample$choice[,"SE2"])
    completeness_behavioral_models[cvsubjs,,cvid] = (naive_err - tmp)/(naive_err- irreducible)
    naive_irreduc[cvsubjs,,cvid] = c(naive_err, irreducible)
    
  }
  print(cvid)
}
mean_mse = apply(mses_behavioral_models, 2, function(x) mean(x,na.rm=T))
sem_mse = apply(mses_behavioral_models, 2, function(x) sd(x, na.rm = TRUE) / sqrt(sum(!is.na(x))))
mean_completeness = apply(completeness_behavioral_models, 2, function(x) mean(x,na.rm=T))
aa= data.frame(sort(mean_mse))
bb= data.frame(sort(mean_completeness, decreasing = TRUE))
mean_mse_in_sample = apply(mses_behavioral_models_in_sample, 2, function(x) mean(x,na.rm=T))
sem_mse_in_sample = apply(mses_behavioral_models_in_sample, 2, function(x) sd(x, na.rm = TRUE) / sqrt(sum(!is.na(x))))
cc = data.frame(sort(mean_mse_in_sample))

########## BEAST-GB predictions and performance
xg_params = list(
  'colsample_bytree' = 0.55,
  'gamma'= 0.01,
  'learning_rate'= 0.01,
  'max_depth'= 5,
  'subsample'= 0.25,
  'min_child_weight' = 3
)
mse_xgb = matrix(NA, nfold_subjs, nfold_tasks)
mse_xgb_in_sample = matrix(NA, nfold_subjs, nfold_tasks)
feats <- c("dataset", "task_id", "Ha", "pHa", "La", "Hb", "pHb", "Lb", 
           "diffEV", "diffSDs", "diffMins", "diffMaxs", 
           "diffUV", "RatioMin", "SignMax", "pBbet_Unbiased1", 
           "pBbet_Uniform", "pBbet_Sign1", "Dom", "diffSignEV", 
           "BEASTpred")
all_preds_df = data.frame()
set.seed(2023)
for (cvid in 1:nfold_tasks){
  probs_train_set = ddd[ddd$crossValidation_id != cvid,]
  probs_test_set = ddd[ddd$crossValidation_id == cvid,]
  for (cvsubjs in 1:nfold_subjs){
    # prepare train set: choice rates of in-sample subjects 
    # facing in-sample tasks
    train_set = probs_train_set[probs_train_set$fold != cvsubjs,]
    choice_mean <- train_set %>%
      group_by(task_id) %>%
      summarize(choice = mean(choice, na.rm = TRUE)) %>%
      ungroup()
    xy_train <- train_set %>%
      select(all_of(feats)) %>%  # Remove the original 'choice' column
      left_join(choice_mean, by = "task_id") %>%  # Join the mean 'choice' values
      distinct()
    
    # get in-sample test set: choice rates of in-sample subjects 
    # facing out-of-sample tasks
    test_set_in_sample = probs_test_set[probs_test_set$fold != cvsubjs,]
    choice_mean_in_sample = test_set_in_sample %>% 
      group_by(task_id) %>%
      summarize(choice = mean(choice, na.rm = TRUE)) %>%
      ungroup()
    xy_test_in_sample <- test_set_in_sample %>%
      select(all_of(feats)) %>%  # Remove the original 'choice' column
      left_join(choice_mean_in_sample, by = "task_id") %>%  # Join the mean 'choice' values
      distinct()
    
    # get main test set: choice rates of out-of-sample subjects 
    # facing out-of-sample tasks
    test_set = probs_test_set[probs_test_set$fold == cvsubjs,]
    choice_mean <- test_set %>%
      group_by(task_id) %>%
      summarize(choice = mean(choice, na.rm = TRUE)) %>%
      ungroup()
    xy_test <- test_set %>%
      select(all_of(feats)) %>%  # Remove the original 'choice' column
      left_join(choice_mean, by = "task_id") %>%  # Join the mean 'choice' values
      distinct()
    
    x_train = model.matrix(choice~0+., data= xy_train[-2])
    trainMean <- apply(x_train,2,mean)
    trainSd <- apply(x_train,2,sd)
    y_train = xy_train$choice
    scaled_x_train <- sweep(sweep(x_train, 2L, trainMean), 2, trainSd, "/")
    
    x_test_in_sample = model.matrix(choice~0+.,data=xy_test_in_sample[-2])
    scaled_x_test_in_sample <- sweep(sweep(x_test_in_sample, 2L, trainMean), 2, trainSd, "/")
    y_test_in_sample = xy_test_in_sample$choice
    
    x_test = model.matrix(choice~0+.,data=xy_test[-2])
    scaled_x_test <- sweep(sweep(x_test, 2L, trainMean), 2, trainSd, "/")
    y_test = xy_test$choice
    
    xgbm <- xgb.DMatrix(scaled_x_train,label=y_train)
    
    xgb_model = xgboost(params = xg_params, data = xgbm,nrounds =1800,verbose = 0)
    
    BEAST_GB_pred = predict(xgb_model,scaled_x_test)
    jointPredObs = data.frame(xy_test,BEAST_GB_pred)
    mse_xgb[cvsubjs,cvid] = mean((jointPredObs$choice - jointPredObs$BEAST_GB_pred)^2)
    jointPredObs$cvid <- cvid
    jointPredObs$cvsubjs <- cvsubjs
    all_preds_df <- rbind(all_preds_df, jointPredObs)
    
    BEAST_GB_pred_in_sample = predict(xgb_model, scaled_x_test_in_sample)
    mse_xgb_in_sample[cvsubjs,cvid] = mean((xy_test_in_sample$choice - BEAST_GB_pred_in_sample)^2)
    
    tmp_shap = shap.prep(xgb_model = xgb_model, X_train = scaled_x_test)
    
    if (cvid==1 && cvsubjs==1){
      shap_long = tmp_shap
    } else {
      shap_long = rbind(shap_long, tmp_shap)
    }
  }
  print(cvid)
}
cat("the mean MSE of BEAST-GB out of sample is", mean(mean(mse_xgb)))
best_contender <-  "CPT-Prelec" 
j <- which(cols_to_mean == best_contender) 
t.test(c(mses_behavioral_models[,j,]),c(mse_xgb), paired = TRUE)
cat("the mean MSE of BEAST-GB in sample is", mean(mean(mse_xgb_in_sample)))
t.test(c(mse_xgb_in_sample),c(mses_behavioral_models_in_sample[,10,]), paired = TRUE)

nfeats = ncol(x_test)
shap_long_long = shap_long[order(shap_long$variable),]
shap_long_long$ID = rep(1:7825,nfeats)
for (var in 1:nfeats){
  shap_long_long$mean_value[(((var-1)*7825)+1):(7825*var)] =
    mean(abs(shap_long_long$value[(((var-1)*7825)+1):(7825*var)]))
}
ordering <- order(-shap_long_long$mean_value)
ordered_levels <- unique(shap_long_long$variable[ordering])
shap_long_long$variable <- factor(shap_long_long$variable, levels = ordered_levels)
shap.plot.summary(shap_long_long, kind="bar")

completness_xgb <- matrix(0, nrow = nfold_subjs, ncol = nfold_tasks)
for (i in 1:nfold_subjs) {
  for (j in 1:nfold_tasks) {
    naive_err <- naive_irreduc[i, 1, j]
    irreducible <- naive_irreduc[i, 2, j]
    val <- mse_xgb[i, j]
    
    completness_xgb[i, j] <- (naive_err - val) / (naive_err - irreducible)
  }
}
completness_xgb = mean(mean(completness_xgb))
cat("BEAST-GB's completenes: ", completness_xgb)

########### without BEAST prediction as feature.
mse_xgb_no_BEAST = matrix(NA, nfold_subjs, nfold_tasks)
feats_no_BEAST <- c("dataset", "task_id", "Ha", "pHa", "La", "Hb", "pHb", "Lb", 
           "diffEV", "diffSDs", "diffMins", "diffMaxs", 
           "diffUV", "RatioMin", "SignMax", "pBbet_Unbiased1", 
           "pBbet_Uniform", "pBbet_Sign1", "Dom", "diffSignEV")
set.seed(2023)
for (cvid in 1:nfold_tasks){
  probs_train_set = ddd[ddd$crossValidation_id != cvid,]
  probs_test_set = ddd[ddd$crossValidation_id == cvid,]
  for (cvsubjs in 1:nfold_subjs){
    train_set = probs_train_set[probs_train_set$fold != cvsubjs,]
    choice_mean <- train_set %>%
      group_by(task_id) %>%
      summarize(choice = mean(choice, na.rm = TRUE)) %>%
      ungroup()
    xy_train <- train_set %>%
      select(all_of(feats_no_BEAST)) %>%  
      left_join(choice_mean, by = "task_id") %>%  # Join the mean 'choice' values
      distinct()
    
    test_set = probs_test_set[probs_test_set$fold == cvsubjs,]
    choice_mean <- test_set %>%
      group_by(task_id) %>%
      summarize(choice = mean(choice, na.rm = TRUE)) %>%
      ungroup()
    xy_test <- test_set %>%
      select(all_of(feats_no_BEAST)) %>% 
      left_join(choice_mean, by = "task_id") %>%  # Join the mean 'choice' values
      distinct()
    
    x_train = model.matrix(choice~0+., data= xy_train[-2])
    trainMean <- apply(x_train,2,mean)
    trainSd <- apply(x_train,2,sd)
    y_train = xy_train$choice
    scaled_x_train <- sweep(sweep(x_train, 2L, trainMean), 2, trainSd, "/")
    
    x_test = model.matrix(choice~0+.,data=xy_test[-2])
    scaled_x_test <- sweep(sweep(x_test, 2L, trainMean), 2, trainSd, "/")
    y_test = xy_test$choice
    
    xgbm <- xgb.DMatrix(scaled_x_train,label=y_train)
    
    xgb_model = xgboost(params = xg_params, data = xgbm,nrounds =1800,verbose = 0)
    
    BEAST_GB_pred = predict(xgb_model,scaled_x_test)
    jointPredObs = data.frame(xy_test,BEAST_GB_pred)
    mse_xgb_no_BEAST[cvsubjs,cvid] = mean((jointPredObs$choice - jointPredObs$BEAST_GB_pred)^2)
    
    
  }
  print(cvid)
}
cat("MSE without BEAST as feature is: ", mean(mean(mse_xgb_no_BEAST)))
t.test(c(mse_xgb),c(mse_xgb_no_BEAST), paired = TRUE)


completness_xgb_no_BEAST <- matrix(0, nrow = nfold_subjs, ncol = nfold_tasks)
for (i in 1:nfold_subjs) {
  for (j in 1:nfold_tasks) {
    naive_err <- naive_irreduc[i, 1, j]
    irreducible <- naive_irreduc[i, 2, j]
    val <- mse_xgb_no_BEAST[i, j]
    
    completness_xgb_no_BEAST[i, j] <- (naive_err - val) / (naive_err - irreducible)
  }
}
completness_xgb_no_BEAST = mean(mean(completness_xgb_no_BEAST))
completness_xgb_no_BEAST

########### without psych features.
mse_xgb_no_Psych = matrix(NA, nfold_subjs, nfold_tasks)
feats_no_psych <- c("dataset", "task_id", "Ha", "pHa", "La", "Hb", "pHb", "Lb",
                    "diffEV", "diffSDs", "diffMins", "diffMaxs", 
                    "BEASTpred")
set.seed(2023)
for (cvid in 1:nfold_tasks){
  probs_train_set = ddd[ddd$crossValidation_id != cvid,]
  probs_test_set = ddd[ddd$crossValidation_id == cvid,]
  for (cvsubjs in 1:nfold_subjs){
    train_set = probs_train_set[probs_train_set$fold != cvsubjs,]
    choice_mean <- train_set %>%
      group_by(task_id) %>%
      summarize(choice = mean(choice, na.rm = TRUE)) %>%
      ungroup()
    xy_train <- train_set %>%
      select(all_of(feats_no_psych)) %>%  
      left_join(choice_mean, by = "task_id") %>%  # Join the mean 'choice' values
      distinct()
    
    test_set = probs_test_set[probs_test_set$fold == cvsubjs,]
    choice_mean <- test_set %>%
      group_by(task_id) %>%
      summarize(choice = mean(choice, na.rm = TRUE)) %>%
      ungroup()
    xy_test <- test_set %>%
      select(all_of(feats_no_psych)) %>%  # Remove the original 'choice' column
      left_join(choice_mean, by = "task_id") %>%  # Join the mean 'choice' values
      distinct()
    
    x_train = model.matrix(choice~0+., data= xy_train[-2])
    trainMean <- apply(x_train,2,mean)
    trainSd <- apply(x_train,2,sd)
    y_train = xy_train$choice
    scaled_x_train <- sweep(sweep(x_train, 2L, trainMean), 2, trainSd, "/")
    
    x_test = model.matrix(choice~0+.,data=xy_test[-2])
    scaled_x_test <- sweep(sweep(x_test, 2L, trainMean), 2, trainSd, "/")
    y_test = xy_test$choice
    
    xgbm <- xgb.DMatrix(scaled_x_train,label=y_train)
    
    xgb_model = xgboost(params = xg_params, data = xgbm,nrounds =1800,verbose = 0)
    
    BEAST_GB_pred = predict(xgb_model,scaled_x_test)
    jointPredObs = data.frame(xy_test,BEAST_GB_pred)
    mse_xgb_no_Psych[cvsubjs,cvid] = mean((jointPredObs$choice - jointPredObs$BEAST_GB_pred)^2)
    
    
  }
  print(cvid)
}
cat("MSE without psych features is: ", mean(mean(mse_xgb_no_Psych)))

completness_xgb_no_Psych <- matrix(0, nrow = nfold_subjs, ncol = nfold_tasks)
for (i in 1:nfold_subjs) {
  for (j in 1:nfold_tasks) {
    naive_err <- naive_irreduc[i, 1, j]
    irreducible <- naive_irreduc[i, 2, j]
    val <- mse_xgb_no_Psych[i, j]
    
    completness_xgb_no_Psych[i, j] <- (naive_err - val) / (naive_err - irreducible)
  }
}
completness_xgb_no_Psych = mean(mean(completness_xgb_no_Psych))
completness_xgb_no_Psych


########### without psych or BEAST features.
mse_xgb_no_PsychBEAST = matrix(NA, nfold_subjs, nfold_tasks)
feats_no_psychBEAST <- c("dataset", "task_id", "Ha", "pHa", "La", "Hb", "pHb", "Lb", 
                         "diffEV", "diffSDs", "diffMins", "diffMaxs")
set.seed(2023)
for (cvid in 1:nfold_tasks){
  probs_train_set = ddd[ddd$crossValidation_id != cvid,]
  probs_test_set = ddd[ddd$crossValidation_id == cvid,]
  for (cvsubjs in 1:nfold_subjs){
    train_set = probs_train_set[probs_train_set$fold != cvsubjs,]
    choice_mean <- train_set %>%
      group_by(task_id) %>%
      summarize(choice = mean(choice, na.rm = TRUE)) %>%
      ungroup()
    xy_train <- train_set %>%
      select(all_of(feats_no_psychBEAST)) %>%  
      left_join(choice_mean, by = "task_id") %>%  # Join the mean 'choice' values
      distinct()
    
    test_set = probs_test_set[probs_test_set$fold == cvsubjs,]
    choice_mean <- test_set %>%
      group_by(task_id) %>%
      summarize(choice = mean(choice, na.rm = TRUE)) %>%
      ungroup()
    xy_test <- test_set %>%
      select(all_of(feats_no_psychBEAST)) %>%  # Remove the original 'choice' column
      left_join(choice_mean, by = "task_id") %>%  # Join the mean 'choice' values
      distinct()
    
    x_train = model.matrix(choice~0+., data= xy_train[-2])
    trainMean <- apply(x_train,2,mean)
    trainSd <- apply(x_train,2,sd)
    y_train = xy_train$choice
    scaled_x_train <- sweep(sweep(x_train, 2L, trainMean), 2, trainSd, "/")
    
    x_test = model.matrix(choice~0+.,data=xy_test[-2])
    scaled_x_test <- sweep(sweep(x_test, 2L, trainMean), 2, trainSd, "/")
    y_test = xy_test$choice
    
    xgbm <- xgb.DMatrix(scaled_x_train,label=y_train)
    
    xgb_model = xgboost(params = xg_params, data = xgbm,nrounds =1800,verbose = 0)
    
    BEAST_GB_pred = predict(xgb_model,scaled_x_test)
    jointPredObs = data.frame(xy_test,BEAST_GB_pred)
    mse_xgb_no_PsychBEAST[cvsubjs,cvid] = mean((jointPredObs$choice - jointPredObs$BEAST_GB_pred)^2)
    
    
  }
  print(cvid)
}
cat("MSE without either BEAST or psych features is: ", mean(mean(mse_xgb_no_PsychBEAST)))

completness_xgb_no_PsychBEAST <- matrix(0, nrow = nfold_subjs, ncol = nfold_tasks)
for (i in 1:nfold_subjs) {
  for (j in 1:nfold_tasks) {
    naive_err <- naive_irreduc[i, 1, j]
    irreducible <- naive_irreduc[i, 2, j]
    val <- mse_xgb_no_PsychBEAST[i, j]
    
    completness_xgb_no_PsychBEAST[i, j] <- (naive_err - val) / (naive_err - irreducible)
  }
}
completness_xgb_no_PsychBEAST = mean(mean(completness_xgb_no_PsychBEAST))
completness_xgb_no_PsychBEAST

########### without naive features.
mse_xgb_no_naive = matrix(NA, nfold_subjs, nfold_tasks)
feats_no_naive <- c("dataset", "task_id", "Ha", "pHa", "La", "Hb", "pHb", "Lb",
                    "diffBEV0", "diffMins",
                    "diffUV", "RatioMin", "SignMax", "pBbet_Unbiased1", 
                    "pBbet_Uniform", "pBbet_Sign1", "Dom", "diffSignEV", 
                    "BEASTpred")
set.seed(2023)
for (cvid in 1:nfold_tasks){
  probs_train_set = ddd[ddd$crossValidation_id != cvid,]
  probs_test_set = ddd[ddd$crossValidation_id == cvid,]
  for (cvsubjs in 1:nfold_subjs){
    train_set = probs_train_set[probs_train_set$fold != cvsubjs,]
    choice_mean <- train_set %>%
      group_by(task_id) %>%
      summarize(choice = mean(choice, na.rm = TRUE)) %>%
      ungroup()
    xy_train <- train_set %>%
      select(all_of(feats_no_naive)) %>%  
      left_join(choice_mean, by = "task_id") %>%  # Join the mean 'choice' values
      distinct()
    
    test_set = probs_test_set[probs_test_set$fold == cvsubjs,]
    choice_mean <- test_set %>%
      group_by(task_id) %>%
      summarize(choice = mean(choice, na.rm = TRUE)) %>%
      ungroup()
    xy_test <- test_set %>%
      select(all_of(feats_no_naive)) %>%  # Remove the original 'choice' column
      left_join(choice_mean, by = "task_id") %>%  # Join the mean 'choice' values
      distinct()
    
    x_train = model.matrix(choice~0+., data= xy_train[-2])
    trainMean <- apply(x_train,2,mean)
    trainSd <- apply(x_train,2,sd)
    y_train = xy_train$choice
    scaled_x_train <- sweep(sweep(x_train, 2L, trainMean), 2, trainSd, "/")
    
    x_test = model.matrix(choice~0+.,data=xy_test[-2])
    scaled_x_test <- sweep(sweep(x_test, 2L, trainMean), 2, trainSd, "/")
    y_test = xy_test$choice
    
    xgbm <- xgb.DMatrix(scaled_x_train,label=y_train)
    
    xgb_model = xgboost(params = xg_params, data = xgbm,nrounds =1800,verbose = 0)
    
    BEAST_GB_pred = predict(xgb_model,scaled_x_test)
    jointPredObs = data.frame(xy_test,BEAST_GB_pred)
    mse_xgb_no_naive[cvsubjs,cvid] = mean((jointPredObs$choice - jointPredObs$BEAST_GB_pred)^2)
    
    
  }
  print(cvid)
}
cat("MSE without naive features is: ", mean(mean(mse_xgb_no_naive)))

completness_xgb_no_naive <- matrix(0, nrow = nfold_subjs, ncol = nfold_tasks)
for (i in 1:nfold_subjs) {
  for (j in 1:nfold_tasks) {
    naive_err <- naive_irreduc[i, 1, j]
    irreducible <- naive_irreduc[i, 2, j]
    val <- mse_xgb_no_naive[i, j]
    
    completness_xgb_no_naive[i, j] <- (naive_err - val) / (naive_err - irreducible)
  }
}
completness_xgb_no_naive = mean(mean(completness_xgb_no_naive))
completness_xgb_no_naive

########### without objective features.
mse_xgb_no_obj = matrix(NA, nfold_subjs, nfold_tasks)
feats_no_obj <- c("task_id", "diffEV", "diffSDs", "diffMins", "diffMaxs", 
                  "diffUV", "RatioMin", "SignMax", "pBbet_Unbiased1", 
                  "pBbet_Uniform", "pBbet_Sign1", "Dom", "diffSignEV", 
                  "BEASTpred")
set.seed(2023)
for (cvid in 1:nfold_tasks){
  probs_train_set = ddd[ddd$crossValidation_id != cvid,]
  probs_test_set = ddd[ddd$crossValidation_id == cvid,]
  for (cvsubjs in 1:nfold_subjs){
    train_set = probs_train_set[probs_train_set$fold != cvsubjs,]
    choice_mean <- train_set %>%
      group_by(task_id) %>%
      summarize(choice = mean(choice, na.rm = TRUE)) %>%
      ungroup()
    xy_train <- train_set %>%
      select(all_of(feats_no_obj)) %>%  
      left_join(choice_mean, by = "task_id") %>%  # Join the mean 'choice' values
      distinct()
    
    test_set = probs_test_set[probs_test_set$fold == cvsubjs,]
    choice_mean <- test_set %>%
      group_by(task_id) %>%
      summarize(choice = mean(choice, na.rm = TRUE)) %>%
      ungroup()
    xy_test <- test_set %>%
      select(all_of(feats_no_obj)) %>%  # Remove the original 'choice' column
      left_join(choice_mean, by = "task_id") %>%  # Join the mean 'choice' values
      distinct()
    
    x_train = model.matrix(choice~0+., data= xy_train[-1])
    trainMean <- apply(x_train,2,mean)
    trainSd <- apply(x_train,2,sd)
    y_train = xy_train$choice
    scaled_x_train <- sweep(sweep(x_train, 2L, trainMean), 2, trainSd, "/")
    
    x_test = model.matrix(choice~0+.,data=xy_test[-1])
    scaled_x_test <- sweep(sweep(x_test, 2L, trainMean), 2, trainSd, "/")
    y_test = xy_test$choice
    
    xgbm <- xgb.DMatrix(scaled_x_train,label=y_train)
    
    xgb_model = xgboost(params = xg_params, data = xgbm,nrounds =1800,verbose = 0)
    
    BEAST_GB_pred = predict(xgb_model,scaled_x_test)
    jointPredObs = data.frame(xy_test,BEAST_GB_pred)
    mse_xgb_no_obj[cvsubjs,cvid] = mean((jointPredObs$choice - jointPredObs$BEAST_GB_pred)^2)
    
    
  }
  print(cvid)
}
cat("MSE without objective features is: ", mean(mean(mse_xgb_no_obj)))

completness_xgb_no_obj <- matrix(0, nrow = nfold_subjs, ncol = nfold_tasks)
for (i in 1:nfold_subjs) {
  for (j in 1:nfold_tasks) {
    naive_err <- naive_irreduc[i, 1, j]
    irreducible <- naive_irreduc[i, 2, j]
    val <- mse_xgb_no_obj[i, j]
    
    completness_xgb_no_obj[i, j] <- (naive_err - val) / (naive_err - irreducible)
  }
}
completness_xgb_no_obj = mean(mean(completness_xgb_no_obj))
completness_xgb_no_obj

# save.image('HAB22 R1 analyses all.RData')
