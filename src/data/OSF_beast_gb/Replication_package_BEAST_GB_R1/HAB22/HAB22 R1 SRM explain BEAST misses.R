rm(list = ls())
library(dplyr)

setwd('C:/Users/plonsky/Dropbox/CPC 2018/NHB23 files/R1 files/Final files for upload/HAB22')


load("HAB22 R1 preds 5x10cv.RData")

###########
# sort by largest diffs
all_preds$mse_from_beast = (all_preds$BEASTpred - all_preds$pred)^2
all_preds = all_preds[order(-all_preds$mse_from_beast),]

# create regression BEAST preds
all_preds$diff_from_random = all_preds$BEASTpred - 0.5

reg_beast = lm(pred-BEASTpred~
                 dataset*diff_from_random, 
               data=all_preds)
summary(reg_beast)
summary(reg_beast)$r.squared
reg_beast_pred = predict(reg_beast, all_preds)
all_preds$reg_BEAST_pred = reg_beast_pred + all_preds$BEASTpred

# MSE of regression beast
mse_reg_beast = mean((all_preds$reg_BEAST_pred - all_preds$B_rate)^2)
# 0.04001878


all_preds$A_sure_loss = 
  all_preds$Ha < 0 &
  all_preds$La < 0

all_preds$B_sure_loss = 
  all_preds$Hb < 0 &
  all_preds$Lb < 0

all_preds$A_sure_gain = 
  all_preds$La > 0 &
  all_preds$Ha > 0

all_preds$B_sure_gain = 
  all_preds$Lb > 0 &
  all_preds$Hb > 0


reg_beast2 = lm(pred-BEASTpred ~ diff_from_random*dataset
                + pBbet_Uniform*A_sure_loss*B_sure_loss*A_sure_gain*B_sure_gain*dataset
                + pBbet_Sign1*A_sure_loss*B_sure_loss*A_sure_gain*B_sure_gain*dataset
                + pBbet_Unbiased1*A_sure_loss*B_sure_loss*A_sure_gain*B_sure_gain*dataset
                + diffMins*A_sure_loss*B_sure_loss*A_sure_gain*B_sure_gain*dataset
                + diffEV*A_sure_loss*B_sure_loss*A_sure_gain*B_sure_gain*dataset
                , data=all_preds)
# summary(reg_beast2)
summary(reg_beast2)$r.squared
# 0.8099822
reg_beast2_pred = predict(reg_beast2, all_preds)
mean((reg_beast2_pred + all_preds$BEASTpred - all_preds$B_rate)^2)
# 0.03403506
