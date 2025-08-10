rm(list = ls())
library(xgboost)
library(dplyr)
library(ggplot2)
library(tictoc)

setwd("C:/Users/plonsky/Dropbox/CPC 2018/NHB23 files/R1 files/Final files for upload")

####
load('HAB22/HAB22 all data with Stew1C_Uni.RData')

all_tasks = unique(HAB22[,c(1:9,73:90)])


#############################
compare_with_tolerance <- function(x, y, tolerance = .Machine$double.eps^0.5) {
  abs(x - y) < tolerance
}

dups = all_tasks
non_unique_cols <- c("Ha","pHa","La","Hb","pHb","Lb")
temp_df <- dups[non_unique_cols]
temp_df$combined <- apply(temp_df, 1, paste, collapse = "_")
dup_rows <- which(duplicated(temp_df$combined) | duplicated(temp_df$combined, fromLast = TRUE))
temp_df = temp_df[,1:6]
dups$duplicate_serial_number <- NA
# Loop through each unique row in the non-unique subset and assign serial numbers
unique_rows <- unique(temp_df[dup_rows, , drop = FALSE])
for (i in seq_len(nrow(unique_rows))) {
  row_values <- as.list(unique_rows[i,])
  indices <- which(apply(temp_df, 1,
                         function(x) all(mapply(compare_with_tolerance, x, row_values))))
  dups$duplicate_serial_number[indices] <- seq_along(indices)
}
dups = dups[!is.na(dups$duplicate_serial_number),]
dups = dups[order(dups$Ha, dups$La, dups$pHa, dups$Hb, dups$Lb, dups$pHb),]

# Here, dups includes all instances in which a task appears in more than
# once context. That is, there is a row for each time a task that will be 
# in the test set will have also appeared on the train set. 

df = merge(HAB22, dups)

######################
cols_to_mean = names(df)[c(30:90)]

preds_behavioral_models <- df %>%
  select(task_id, duplicate_serial_number, all_of(cols_to_mean)) %>%
  group_by(task_id, duplicate_serial_number) %>%
  summarize(
    count = n(),
    across(all_of(cols_to_mean), \(x) mean(x, na.rm = TRUE)),
    .groups = 'drop' 
  )
# preds_behavioral_models$B_rate = preds_behavioral_models$choice
allpreds = merge(dups, preds_behavioral_models)

# non_unique_cols = 5:10

# Here, rows_x is the instance to be predicted
# rows_not_x is the instances that are part of the train set
compute_squared_diff <- function(rows_x, rows_not_x, squared_differences_list) {
  for (i in seq_len(nrow(rows_x))) {
    row_values <- as.list(rows_x[i, non_unique_cols])
    matching_row_indices <- which(apply(rows_not_x[non_unique_cols], 1, 
                                        function(x) all(mapply(compare_with_tolerance, x, row_values))))
    
    if (length(matching_row_indices) > 0) {
      for (col_name in cols_to_mean) {
        # Compute the weighted average for value_1
        weights <- rows_not_x[matching_row_indices, "count"]
        column_values <- rows_not_x[matching_row_indices, col_name]
        value_1 <- sum(weights * column_values) / sum(weights)
        
        value_2 <- rows_x[i, "B_rate"]
        squared_difference <- (value_1 - value_2)^2
        
        # Update squared_differences_list
        if (is.null(squared_differences_list[[col_name]])) {
          squared_differences_list[[col_name]] <- list(squared_diffs = numeric(), row_values_list = list())
        }
        squared_differences_list[[col_name]]$squared_diffs <- c(squared_differences_list[[col_name]]$squared_diffs, 
                                                                squared_difference)
        squared_differences_list[[col_name]]$row_values_list <- c(squared_differences_list[[col_name]]$row_values_list, 
                                                                  list(as.list(rows_x[i, 1:10])))
      }
    }
  }
  return(squared_differences_list)
}

squared_differences_list <- setNames(vector("list", length(cols_to_mean)), cols_to_mean)
for (x in 1:5) {
  rows_x <- allpreds[allpreds$duplicate_serial_number == x, ]
  rows_not_x <- allpreds[allpreds$duplicate_serial_number != x, ]
  squared_differences_list <- compute_squared_diff(rows_x, rows_not_x, squared_differences_list)
}
mean_squared_differences <- sapply(squared_differences_list, function(x) mean(x$squared_diffs, na.rm = TRUE))
sort(mean_squared_differences)
sem_squared_differences <- sapply(squared_differences_list, function(x) {
  sd(x$squared_diffs, na.rm = TRUE) / sqrt(sum(!is.na(x$squared_diffs)))
})

# ########## get generalization predictions of repeated tasks in 16th dataset

xg_params = list(
  'colsample_bytree' = 0.55,
  'gamma'= 0.01,
  'learning_rate'= 0.01,
  'max_depth'= 5,
  'subsample'= 0.25,
  'min_child_weight'= 3
)
dss = unique(all_tasks$dataset)
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
  test_set = merge(test_set, dups)
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
  
  print(paste("dataset ",ds, dss[ds]))
  
}
pred_all$sq_diff = (pred_all$B_rate - pred_all$pred)^2
mse = mean(pred_all$sq_diff)
mse
naive_err = mean((pred_all$B_rate - 0.5)^2)
irreducible = mean(pred_all$choice[,"SE2"])
completeness = (naive_err - mse)/(naive_err - irreducible)
completeness
diff_from_choice = (mean_squared_differences["choice"] - mse)/mean_squared_differences["choice"]
diff_from_CPT = (mean_squared_differences["CPT-Prelec"] - mse)/mean_squared_differences["CPT-Prelec"]
diff_from_choice

extracted_data <- squared_differences_list[['choice']]
other_exps_data <- do.call(rbind, lapply(extracted_data$row_values_list, function(x) as.data.frame(t(unlist(x)))))
other_exps_data$squared_diffs <- extracted_data$squared_diffs

joint_other_BEAST_GB = merge(pred_all, other_exps_data, by = "task_id")
t.test(joint_other_BEAST_GB$sq_diff, joint_other_BEAST_GB$squared_diffs, paired = TRUE)

# save(mean_squared_differences, sem_squared_differences, pred_all, file = "Context R1 - data for figure.RData")
# save.image("Context R1 analyses.RData")
