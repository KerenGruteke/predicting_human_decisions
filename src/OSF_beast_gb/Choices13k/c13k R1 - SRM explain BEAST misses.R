rm(list = ls())
library(dplyr)
library(ggplot2)

setwd('C:/Users/plonsky/Dropbox/CPC 2018/NHB23 files/R1 files/Final files for upload/Choices13k')

### BEAST-GB predictions
load("C13k R1 BEAST-GB preds 5x10cv.RData")

# MSE of BEAST-GB:
mse = mean((all_preds$B_rate - all_preds$pred)^2)
mse
# 0.008426626

# MSE of GB from BEAST
mse_from_beast = mean((all_preds$BEASTpred - all_preds$pred)^2)
# 0.01320463

# sort by largest diffs
all_preds$mse_from_beast = (all_preds$BEASTpred - all_preds$pred)^2
all_preds = all_preds[order(-all_preds$mse_from_beast),]

####### difference from mid-point
all_preds$diff_from_random = all_preds$BEASTpred - 0.5
all_preds$multi = ifelse(all_preds$LotNumB > 1, TRUE, FALSE)

# analyze largest deviations
top100 = all_preds[1:100,]
mean(abs(top100$diff_from_random))
mean(abs(all_preds$diff_from_random))
mean(abs(all_preds$pred[1:100] - 0.5))

mean(top100$multi)
mean(all_preds$multi)

mean(top100$Dom != "0")
mean(all_preds$Dom != "0")

# predict differences from BEAST-GB
reg_beast = lm((pred-BEASTpred)~
                 diff_from_random*multi+
                 diff_from_random*Dom
               , data=all_preds)
summary(reg_beast)
reg_beast_pred = predict(reg_beast, all_preds)
mean((reg_beast_pred+all_preds$BEASTpred - all_preds$B_rate)^2)
# 0.01218917
all_preds$reg_BEAST_pred = reg_beast_pred + all_preds$BEASTpred

####################################

# sort by largest diffs from regression BEAST
all_preds$mse_from_reg_beast = (all_preds$reg_BEAST_pred - all_preds$pred)^2
all_preds = all_preds[order(-all_preds$mse_from_reg_beast),]

##### "reverse loss aversion" 
all_preds$A_loss_B_mixed = 
  all_preds$A_max <= 0 &
  all_preds$B_max > 0 &
  all_preds$A_min > all_preds$B_min
all_preds$B_loss_A_mixed = 
  all_preds$B_max <= 0 &
  all_preds$A_max > 0 &
  all_preds$A_min < all_preds$B_min

top100 = all_preds[1:100,]
mean(top100$A_loss_B_mixed==1 | top100$B_loss_A_mixed==1)
(top100$BEASTpred[top100$A_loss_B_mixed==1]- top100$pred[top100$A_loss_B_mixed==1])>0
(top100$BEASTpred[top100$B_loss_A_mixed==1]- top100$pred[top100$B_loss_A_mixed==1])>0
mean(all_preds$A_loss_B_mixed==1 | all_preds$B_loss_A_mixed==1)

# predict differences from BEAST-GB
reg_beast2 = lm((pred-BEASTpred)~
                  diff_from_random*multi+
                  diff_from_random*Dom+
                  A_loss_B_mixed+
                  B_loss_A_mixed
                , data=all_preds)
summary(reg_beast2)
reg_beast2_pred = predict(reg_beast2, all_preds)
mean((reg_beast2_pred+all_preds$BEASTpred - all_preds$B_rate)^2)
# 0.01093018

all_preds$reg_BEAST2_pred = reg_beast2_pred + all_preds$BEASTpred
# sort by largest diffs 
all_preds$mse_from_reg_beast2 = (all_preds$reg_BEAST2_pred - all_preds$pred)^2
all_preds = all_preds[order(-all_preds$mse_from_reg_beast2),]

##########################
top100 = all_preds[1:100,]

mean(top100$SignMax != "1")
median(top100$RatioMin[top100$SignMax == "1"])
mean(all_preds$SignMax != "1")
median(all_preds$RatioMin[all_preds$SignMax == "1"])

all_preds$no_pess_contingency =
  all_preds$SignMax != 1 |
  all_preds$RatioMin > 0

top100 = all_preds[1:100,]

summary(top100$pBbet_Uniform[top100$no_pess_contingency])
summary(all_preds$pBbet_Uniform[all_preds$no_pess_contingency])

summary(top100$diffUV[top100$no_pess_contingency])
summary(all_preds$diffUV[all_preds$no_pess_contingency])

reg_beast3 = lm(pred-BEASTpred~
                  diff_from_random*multi+
                  diff_from_random*Dom+
                  A_loss_B_mixed+B_loss_A_mixed+
                  no_pess_contingency*pBbet_Uniform+
                  no_pess_contingency*pBbet_SignFB
                ,data=all_preds)
summary(reg_beast3)
reg_beast3_pred = predict(reg_beast3, all_preds)
mean((reg_beast3_pred+all_preds$BEASTpred - all_preds$B_rate)^2)
# 0.01004235

all_preds$reg_beast3_pred = reg_beast3_pred+all_preds$BEASTpred
# sort by largest diffs 
all_preds$mse_from_reg_beast3 = (all_preds$reg_beast3_pred - all_preds$pred)^2
all_preds = all_preds[order(-all_preds$mse_from_reg_beast3),]

#####################

all_preds$B_EV_A_pessimism = 
  all_preds$B_min < all_preds$A_min &
  all_preds$diffEV > abs(all_preds$diffMins) 

all_preds$A_EV_B_pessimism = 
  all_preds$A_min < all_preds$B_min &
  all_preds$diffEV < 0 &
  abs(all_preds$diffEV) > abs(all_preds$diffMins) 

all_preds$B_EV_A_unbiased = 
  all_preds$pBbet_UnbiasedFB < 0 &
  all_preds$diffEV > 0 

all_preds$A_EV_B_unbiased = 
  all_preds$pBbet_UnbiasedFB > 0 &
  all_preds$diffEV < 0 

all_preds$A_certain = 
  all_preds$A_min == all_preds$A_max &
  all_preds$Dom == 0
all_preds$B_certain = 
  all_preds$B_min == all_preds$B_max &
  all_preds$Dom == 0

top100 = all_preds[1:100,]

mean(top100$A_certain | top100$B_certain)
mean(all_preds$A_certain | all_preds$B_certain)

mean(top100$multi)
mean(all_preds$multi)

mean(top100$A_EV_B_unbiased | top100$B_EV_A_unbiased)
mean(all_preds$A_EV_B_unbiased | all_preds$B_EV_A_unbiased)

mean(top100$A_EV_B_pessimism | top100$B_EV_A_pessimism)
mean(all_preds$A_EV_B_pessimism | all_preds$B_EV_A_pessimism)


reg_beast4 = lm(pred-BEASTpred~
                  diff_from_random*multi+
                  diff_from_random*Dom+
                  A_loss_B_mixed+B_loss_A_mixed+
                  no_pess_contingency*pBbet_Uniform+
                  no_pess_contingency*pBbet_SignFB+
                  pBbet_Uniform*A_certain*multi+
                  pBbet_SignFB*A_certain*multi+
                  pBbet_UnbiasedFB*A_certain*multi+
                  diffMins*A_certain*multi+
                  diffEV*A_certain*multi
                ,data=all_preds)
summary(reg_beast4)
reg_beast4_pred = predict(reg_beast4, all_preds)
mean((reg_beast4_pred+all_preds$BEASTpred - all_preds$B_rate)^2)
# 0.009651077

##################### Additional analyses

library(emmeans)
values_to_check <- c(-1,-0.5, 0, 0.5, 1)
emmeans(reg_beast4, ~pBbet_Uniform | A_certain:multi, at=list(pBbet_Uniform = values_to_check, Dom = "0")) 
trends_Uni <- emtrends(reg_beast4, ~ A_certain * multi, var = "pBbet_Uniform",at = list(Dom = "0"))
summary(trends_Uni)
pairs(trends_Uni)
emmeans(reg_beast4, ~pBbet_SignFB | A_certain:multi, at=list(pBbet_SignFB = values_to_check, Dom = "0")) 
trends_Sign <- emtrends(reg_beast4, ~ A_certain * multi, var = "pBbet_SignFB",at = list(Dom = "0"))
summary(trends_Sign)
pairs(trends_Sign)
emmeans(reg_beast4, ~pBbet_UnbiasedFB | A_certain:multi, at=list(pBbet_UnbiasedFB = values_to_check, Dom = "0")) 
trends_Unbiased <- emtrends(reg_beast4, ~ A_certain * multi, var = "pBbet_UnbiasedFB",at = list(Dom = "0"))
summary(trends_Unbiased)
pairs(trends_Unbiased)
emmeans(reg_beast4, ~diffMins | A_certain:multi, at=list(diffMins = c(-50,-25,0,25,50), Dom = "0")) 
trends_pess <- emtrends(reg_beast4, ~ A_certain * multi, var = "diffMins",at = list(Dom = "0"))
summary(trends_pess)
pairs(trends_pess)
emmeans(reg_beast4, ~diffEV | A_certain:multi, at=list(diffEV = c(-8,-4,0,4,8), Dom = "0")) 
trends_EV <- emtrends(reg_beast4, ~ A_certain * multi, var = "diffEV",at = list(Dom = "0"))
summary(trends_EV)
pairs(trends_EV)


