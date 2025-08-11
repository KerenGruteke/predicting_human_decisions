rm(list = ls())
library(xgboost)
library(dplyr)
library(ggplot2)
library(keras)
library(tensorflow)

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
    naive_err = tmp[2]
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

#### train NN using keras

# only objective features . 
feats = names(ddd)[ c(1,4:10) ]  

# Prepare storage for MSEs and final predictions
mse_nn = matrix(NA, nfold_subjs, nfold_tasks)
all_preds_df = data.frame()

set.seed(2023)

# Define the neural network architecture
create_nn_model <- function(input_shape) {
  model <- keras_model_sequential() %>%
    layer_dense(units = 128, activation = "relu", 
                kernel_regularizer = regularizer_l2(l = 0.001),
                input_shape = input_shape) %>%  # Define input_shape here
    layer_dropout(rate = 0.2) %>%
    layer_dense(units = 256, activation = "relu",
                kernel_regularizer = regularizer_l2(l = 0.001)) %>%
    layer_dropout(rate = 0.2) %>%
    layer_dense(units = 64, activation = "relu",
                kernel_regularizer = regularizer_l2(l = 0.001)) %>%
    layer_dropout(rate = 0.2) %>%
    layer_dense(units = 1, activation = "linear")  # Output layer for regression

  # Compile the model
  model %>% compile(
    loss = "mean_squared_error",
    optimizer = optimizer_rmsprop(learning_rate = 0.001),
    metrics = list("mean_absolute_error")
  )

  return(model)
}

# Nested cross-validation loops
# for(cvid in seq_len(nfold_tasks)){
for(cvid in 1:10){
  # Partition tasks: 
  probs_train_set = ddd[ddd$crossValidation_id != cvid,]
  probs_test_set  = ddd[ddd$crossValidation_id == cvid,]
  
  # for(cvsubjs in seq_len(nfold_subjs)){
  for(cvsubjs in 1:5){
    
    # Prepare train set
    train_set = probs_train_set[probs_train_set$fold != cvsubjs, ]
    choice_mean <- train_set %>%
      group_by(task_id) %>%
      summarize(choice = mean(choice, na.rm = TRUE)) %>%
      ungroup()
    
    xy_train <- train_set %>%
      select(all_of(feats)) %>%        # only objective features
      left_join(choice_mean, by = "task_id") %>%
      distinct()
    
    # Prepare test set
    test_set = probs_test_set[probs_test_set$fold == cvsubjs, ]
    choice_mean <- test_set %>%
      group_by(task_id) %>%
      summarize(choice = mean(choice, na.rm = TRUE)) %>%
      ungroup()
    
    xy_test <- test_set %>%
      select(all_of(feats)) %>% 
      left_join(choice_mean, by = "task_id") %>%
      distinct()
    
    # Scale the data
    x_train <- model.matrix(choice ~ ., data = xy_train[-2])[, -1]
    trainMean <- apply(x_train, 2, mean)
    trainSd <- apply(x_train, 2, sd)
    y_train <- xy_train$choice
    scaled_x_train <- sweep(sweep(x_train, 2L, trainMean), 2, trainSd, "/")
    
    x_test <- model.matrix(choice ~ ., data = xy_test[-2])[, -1]
    scaled_x_test <- sweep(sweep(x_test, 2L, trainMean), 2, trainSd, "/")
    y_test <- xy_test$choice
    
    
    # Create & train a neural network 
    input_dim <- as.integer(ncol(scaled_x_train))
    nn_model <- create_nn_model(input_shape = input_dim)
    callback_es <- callback_early_stopping(
      monitor = "val_loss",
      patience = 1000,       # stop if val_loss doesn't improve for 10 epochs
      restore_best_weights = TRUE
    )
    history <- nn_model %>% fit(
      scaled_x_train, y_train,
      epochs = 1000,
      batch_size = 256,
      validation_split = 0.2,
      callbacks = list(callback_es),
      verbose = 0
    )
    
    # Predict on test
    nn_pred <- predict(nn_model, scaled_x_test)
    
    # Compute MSE
    this_mse <- mean((y_test - nn_pred)^2)
    mse_nn[cvsubjs, cvid] <- this_mse

    # Store for potential later analysis
    tmp_df <- data.frame(xy_test, NN_pred = nn_pred[,1])
    tmp_df$cvid    <- cvid
    tmp_df$cvsubjs <- cvsubjs
    all_preds_df <- rbind(all_preds_df, tmp_df)
  }
  cat("Finished tasks-fold:", cvid, "\n")
}

# Average MSE across folds
mean_mse <- mean(mse_nn, na.rm = T)
cat("Nested-CV mean MSE (NN, no behavioral features):", mean_mse, "\n")




# Tuning
# #############################
# # Grid for 2-layer networks only
# grid_2layers <- expand.grid(
#   use_third = FALSE,           # always false
#   units1 = c(32, 64, 128, 256),
#   units2 = c(32, 64, 128, 256),
#   units3 = 0,
#   dropout = c(0.1,0.2, 0.3),
#   ll2_reg = c(FALSE, TRUE),
#   learning_rate = c(1e-3, 5e-4, 1e-4),
#   epochs = 600,
#   batch_size = c(32, 64, 128, 256),
#   patience = 40
# )
# 
# # Grid for 3-layer networks
# grid_3layers <- expand.grid(
#   use_third = TRUE,            # always true
#   units1 = c(32, 64, 128, 256),
#   units2 = c(32, 64, 128, 256),
#   units3 = c(32, 64, 128),         # matters only for the 3-layer case
#   dropout = c(0.1,0.2, 0.3),
#   ll2_reg = c(FALSE, TRUE),
#   learning_rate = c(1e-3, 5e-4, 1e-4),
#   epochs = 600,
#   batch_size = c(32, 64, 128, 256),
#   patience = 40
# )
# 
# # Combine the two grids
# hyper_grid <- rbind(grid_2layers, grid_3layers)
# nrow(hyper_grid)
# 
# # 
# # hyper_grid <- expand.grid(
# #   units1         = c(64, 128, 256),      # 1st hidden layer size
# #   units2         = c(64, 128, 256),      # 2nd hidden layer size
# #   use_third      = c(FALSE, TRUE),       # whether to add a 3rd layer
# #   units3         = c(64, 128),           # possible sizes for 3rd layer
# #   dropout        = c(0.1, 0.3),     # dropout rate
# #   l2_reg         = c(FALSE, TRUE),       # L2 regularization or not
# #   learning_rate  = c(1e-3, 3e-4, 1e-4),  # RMSProp/Adam LR
# #   epochs         = c(100, 300, 500),     # max epochs
# #   batch_size     = c(16, 32, 64),        # batch sizes
# #   patience       = 40             # early stopping patience
# # )
# # # Filter out combos where use_third=FALSE but units3 is not "irrelevant"
# # hyper_grid <- hyper_grid[
# #   !(hyper_grid$use_third == FALSE & hyper_grid$units3 > 0),
# # ]
# # 
# # nrow(hyper_grid)
# 
# create_nn_model <- function(input_dim,
#                             units1, units2, use_third, units3,
#                             dropout, l2_reg, learning_rate) {
#   
#   # If we are using L2, define a reg object; else NULL
#   l2_obj <- if (l2_reg) regularizer_l2(0.001) else NULL
#   
#   model <- keras_model_sequential()
#   
#   # 1st hidden layer
#   model %>%
#     layer_dense(
#       units = units1,
#       activation = "relu",
#       input_shape = c(input_dim),
#       kernel_regularizer = l2_obj
#     ) %>%
#     layer_dropout(rate = dropout)
#   
#   # 2nd hidden layer
#   model %>%
#     layer_dense(
#       units = units2,
#       activation = "relu",
#       kernel_regularizer = l2_obj
#     ) %>%
#     layer_dropout(rate = dropout)
#   
#   # Optional 3rd layer
#   if (use_third) {
#     model %>%
#       layer_dense(
#         units = units3,
#         activation = "relu",
#         kernel_regularizer = l2_obj
#       ) %>%
#       layer_dropout(rate = dropout)
#   }
#   
#   # Output layer
#   model %>%
#     layer_dense(units = 1, activation = "linear")
#   
#   # Compile
#   model %>% compile(
#     loss = "mean_squared_error",
#     optimizer = optimizer_rmsprop(learning_rate = learning_rate),
#     metrics = c("mean_absolute_error")
#   )
#   
#   model
# }
# 
# 
# 
# cvid <- 4
# cvsubjs <- 3
# 
# probs_train_set <- ddd[ddd$crossValidation_id != cvid, ]
# probs_test_set  <- ddd[ddd$crossValidation_id == cvid, ]
# 
# train_set <- probs_train_set[probs_train_set$fold != cvsubjs, ]
# test_set  <- probs_test_set[ probs_test_set$fold == cvsubjs, ]
# 
# # Summarize 'choice' in training data
# choice_mean_train <- train_set %>%
#   group_by(task_id) %>%
#   summarize(choice = mean(choice, na.rm = TRUE)) %>%
#   ungroup()
# 
# xy_train <- train_set %>%
#   select(all_of(feats)) %>%    # your objective features
#   left_join(choice_mean_train, by = "task_id") %>%
#   distinct()
# 
# # Summarize 'choice' in test data
# choice_mean_test <- test_set %>%
#   group_by(task_id) %>%
#   summarize(choice = mean(choice, na.rm = TRUE)) %>%
#   ungroup()
# 
# xy_test <- test_set %>%
#   select(all_of(feats)) %>%
#   left_join(choice_mean_test, by = "task_id") %>%
#   distinct()
# 
# # Scale numeric data
# x_train_mat <- model.matrix(choice ~ ., data = xy_train[-2])[, -1]
# y_train_vec <- xy_train$choice
# 
# trainMean <- apply(x_train_mat, 2, mean)
# trainSd   <- apply(x_train_mat, 2, sd)
# 
# scaled_x_train <- sweep(sweep(x_train_mat, 2, trainMean), 2, trainSd, "/")
# 
# x_test_mat <- model.matrix(choice ~ ., data = xy_test[-2])[, -1]
# scaled_x_test <- sweep(sweep(x_test_mat, 2, trainMean), 2, trainSd, "/")
# 
# y_test_vec <- xy_test$choice
# 
# input_dim <- ncol(scaled_x_train)
# 
# results <- list()
# 
# for (i in seq_len(nrow(hyper_grid))) {
#   
#   params <- hyper_grid[i, ]
#   
#   cat("\n=== Starting Model", i, "===\n")
#   print(params)
#   
#   set.seed(2023)
#   
#   # Build model
#   model <- create_nn_model(
#     input_dim      = input_dim,
#     units1         = params$units1,
#     units2         = params$units2,
#     use_third      = params$use_third,
#     units3         = params$units3,
#     dropout        = params$dropout,
#     l2_reg         = params$ll2_reg,
#     learning_rate  = params$learning_rate
#   )
#   
#   # Early stopping
#   cb_es <- callback_early_stopping(
#     monitor = "val_loss",
#     patience = params$patience,
#     restore_best_weights = TRUE
#   )
#   
#   # Fit
#   history <- model %>% fit(
#     scaled_x_train, 
#     y_train_vec,
#     epochs = params$epochs,
#     batch_size = params$batch_size,
#     validation_split = 0.2,
#     callbacks = list(cb_es),
#     verbose = 0
#   )
#   
#   # Predict
#   preds <- predict(model, scaled_x_test)
#   mse_val <- mean((y_test_vec - preds)^2)
#   
#   # Store in results
#   results[[i]] <- list(
#     i = i,
#     params = params,
#     mse = mse_val,
#     final_epoch = length(history$metrics$loss),
#     history = history
#   )
#   
#   cat("... Done. MSE =", mse_val, "\n")
# }
# 
# 
# library(dplyr)
# 
# score_df <- do.call(rbind, lapply(results, function(x) {
#   data.frame(
#     i = x$i,
#     units1 = x$params$units1,
#     units2 = x$params$units2,
#     use_third = x$params$use_third,
#     units3 = x$params$units3,
#     dropout = x$params$dropout,
#     l2_reg = x$params$ll2_reg,
#     learning_rate = x$params$learning_rate,
#     epochs = x$params$epochs,
#     batch_size = x$params$batch_size,
#     patience = x$params$patience,
#     mse = x$mse,
#     final_epoch = x$final_epoch
#   )
# }))
# 
# # Sort by MSE
# score_df <- score_df %>% arrange(mse)
# 
# # Print top 5
# cat("\nTop 5 models:\n")
# head(score_df, 5)
