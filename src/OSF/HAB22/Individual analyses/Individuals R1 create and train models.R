rm(list = ls())
library(broom)
library(dplyr)
library(ggplot2)
library(tidyr)
library(tibble)
library(randomForest)
library(brms)

setwd("C:/Users/plonsky/Dropbox/CPC 2018/NHB23 files/R1 files/Final files for upload/HAB22")

####
load('Individual analyses/Individuals R1 data.RData')

allsubjs = unique(mmm$subj_id)
nsubjs = length(allsubjs)

# Random Forest predictions
set.seed(2023)
all_subj_preds <- data.frame()
for (cvid in 1:10){
  probs_train_set = mmm[mmm$crossValidation_id != cvid,]
  probs_test_set = mmm[mmm$crossValidation_id == cvid,]
  for (subj in 1:nsubjs){
    sid = allsubjs[subj]
    subj_train = probs_train_set[probs_train_set$subj_id == sid,]
    subj_test = probs_test_set[probs_test_set$subj_id == sid,]
    
    if (length(unique(subj_train$choice)) > 1){
      rf=randomForest(factor(choice) ~ BEAST_GB_pred 
                      + pBbet_Unbiased1 + pBbet_Uniform + pBbet_Sign1 + diffMins
                      + diffEV + Ha + La + Hb + Lb + BEASTpred
                      + pHa + pHb + diffSignEV + diffUV + SignMax + RatioMin 
                      + Dom + diffSDs + diffMaxs
                      , data = subj_train)
      subj_preds = predict(rf, subj_test, type = "prob")[,2]
    } else {
      subj_preds = rep(unique(subj_train$choice), nrow(subj_test))
    }

    subj_test$RFs = subj_preds
    all_subj_preds = rbind(all_subj_preds, subj_test)
  }
  
  cat("Completed CV fold:", cvid, "\n")
  
}
mse_over_choices = mean((all_subj_preds$RFs - all_subj_preds$choice)^2)
cat("overall MSE over choices:", mse_over_choices, "\n")

# compare MSEs over subjects
subject_mse <- all_subj_preds %>%
  group_by(subj_id) %>%
  summarize(
    across(
      .cols = c(13:72, 90:92),  
      .fns  = ~ mean((.x - choice)^2), 
      .names = "MSE_{col}"
    )
  )

t.test(subject_mse$`MSE_CPT-Prelec`, subject_mse$MSE_RFs, paired = TRUE)
t.test(subject_mse$`MSE_CPT-LBW`, subject_mse$MSE_RFs, paired = TRUE)

# model_mse <- subject_mse %>%
#   summarize(
#     across(starts_with("MSE_"), mean)
#   )
# model_mse_vec <- unlist(model_mse)
# model_mse_sorted <- sort(model_mse_vec)
# model_mse_sorted


### Bayesian mixed effects model
# Warning: this code is likely to take a very long time (days) to run

# The model formula
formula <- bf(
  choice ~ I(BEAST_GB_pred - 0.5) + pBbet_Unbiased1 + pBbet_Uniform + pBbet_Sign1 + diffMins + diffEV +
    (I(BEAST_GB_pred - 0.5) + pBbet_Unbiased1 + pBbet_Uniform + pBbet_Sign1 + diffMins + diffEV| subj_id)
)

pred_list <- vector("list", length = 10)
cv_models  <- vector("list", length = 10)

set.seed(2023)

for(cvid in 1:10) {
  
  # Subset training and test data based on the crossValidation_id column
  train_data <- mmm %>% filter(crossValidation_id != cvid)
  test_data  <- mmm %>% filter(crossValidation_id == cvid)
  
  cat("Fitting Bayesian mixed model for CV fold", cvid, "...\n")
  
  # Fit the Bayesian mixed-effects model on the training data using default priors.
  bayes_model_cv <- brm(
    formula = formula,
    data = train_data,
    family = bernoulli(link = "logit"),
    chains = 4,
    cores = parallel::detectCores() - 1,
    iter = 1500,
    warmup = 750,
    control = list(adapt_delta = 0.95),
    silent = TRUE
  )
  
  # Save the fitted model for later analysis
  cv_models[[cvid]] <- bayes_model_cv
  
  # Predict on the test data:
  test_epred <- posterior_epred(bayes_model_cv, newdata = test_data)
  test_data <- test_data %>% mutate(Bayes_reg = colMeans(test_epred))
  
  pred_list[[cvid]] <- test_data
  
  cat("Completed CV fold", cvid, "\n")
  
}

# Combine the predictions from all folds
all_pred <- bind_rows(pred_list)

mse_over_choices = mean((all_pred$Bayes_reg - all_pred$choice)^2)
cat("overall MSE over choices:", mse_over_choices, "\n")
subject_mse_Bayes_reg <- all_pred %>%
  group_by(subj_id) %>%
  summarise(MSE_Bayes_reg = mean((choice - Bayes_reg)^2))

subject_mse = left_join(subject_mse, subject_mse_Bayes_reg)

t.test(subject_mse$`MSE_CPT-Prelec`, subject_mse$MSE_Bayes_reg, paired = TRUE)
t.test(subject_mse$`MSE_CPT-LBW`, subject_mse$MSE_Bayes_reg, paired = TRUE)

model_mse <- subject_mse %>%
  summarize(
    across(starts_with("MSE_"), mean)
  )
model_mse_vec <- unlist(model_mse)
model_mse_sorted <- sort(model_mse_vec)
model_mse_sorted



