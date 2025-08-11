rm(list = ls())
library(boot)

setwd("C:/Users/plonsky/Dropbox/CPC 2018/NHB23 files/R1 files/Final files for upload/CPC18")

##########
# loading and organizing test data
load("comp set to predict T1.RData")
toPredictWide = toPredictT1
names(toPredictWide)[14:18] = c("observed1","observed2","observed3","observed4","observed5")
toPredictWide = toPredictWide[with(toPredictWide, order(GameID)), ]
toPredict = reshape(toPredictWide, varying = 14:18, timevar = "block", idvar = 1:13, direction = "long", v.names = "B")

###########

# For computing completeness 
raw_d = read.csv('rawData_comp_All.csv')
raw_d_subj = aggregate(B ~ SubjID + GameID+Ha+pHa+La+LotShapeA+LotNumA+Hb+pHb+Lb+LotShapeB+LotNumB+Amb+Corr+block, 
                       data=raw_d, mean)
aggMeans = aggregate(B ~ GameID+Ha+pHa+La+LotShapeA+LotNumA+Hb+pHb+Lb+LotShapeB+LotNumB+Amb+Corr+block, 
                     data=raw_d_subj, 
                     FUN = function(x) c(mean = mean(x), std = sd(x), n = length(x), SE2 = (sd(x)^2)/length(x)))
naive_err = mean((aggMeans$B[, "mean"] - 0.5)^2)
irreducible = mean(aggMeans$B[,"SE2"])


########### get all models performance and compare to winner

evaluatePredictions <- function(y, y_hat,
                                do.MSE = TRUE,
                                do.cor = TRUE,
                                do.plot = TRUE,
                                do.MAE = FALSE,
                                do.completeness = TRUE){
  mse = mean((y-y_hat)^2)
  cor_yy = NA
  if (do.MSE){
    print(paste("MSE is ", 100*mse))
  }
  if (do.cor){
    cor_yy = cor(y,y_hat)
    print(paste("correlation is ", cor_yy))
  }
  if (do.plot){
    plot(y_hat,y)
    lmline = lm(y~y_hat)
    abline(a=lmline$coefficients[1],b=lmline$coefficients[2])
    abline(a=0,b=1,col = "red")
  }
  if (do.MAE){
    mae = mean((abs(y-y_hat)))
    print(paste("MAE is ", 10*mae))
    return(list(mse,mae))
  }
  if (do.completeness){
    completeness = (naive_err - mse)/(naive_err- irreducible)
    print(paste("Completness is ", completeness))
  }
  return(list(mse,cor_yy,completeness))
}



########## compute statistical difference with bootstrap

folder_path = './all Track 1 submissions/'

# this function compares the means of the two columns in data.
# It returns 0 if mean of the 1st column (model) is lower than 
# the mean of the 2nd column (baseline); 0.5 if they are equal;
# and 1 otherwise.
cmpMSD <- function(data,indices){
  d = data[indices,]
  return(ifelse(mean(d[,1]) < mean(d[,2]), yes = 0, ifelse(mean(d[,1]) == mean(d[,2]), no = 1, yes=0.5 )))  
}

# Baseline Winner
winner_file = paste0(folder_path, "BP47.csv")
winner_predictions <- read.csv(winner_file) 
jointObsPreds = cbind(toPredictWide ,winner_predictions)
winner_mses = apply(jointObsPreds[,14:23], 1, function(x) 100*mean((x[1:5]-x[6:10])^2) )

filenames = list.files(path = folder_path, pattern="*.csv")
for (file in filenames){
  file_path <- paste0(folder_path, file)
  print(file)
  predictions <- read.csv(file_path) 
  jointObsPreds = cbind(toPredictWide ,predictions)
  
  predictions = reshape(predictions, varying = 1:5, timevar = "block", idvar = "GameID", direction = "long", v.names = "B")
  tmp = evaluatePredictions(toPredict$B, predictions$B, do.plot = FALSE)
  print('*******************')
  
  modelMSDs = apply(jointObsPreds[,14:23], 1, function(x) 100*mean((x[1:5]-x[6:10])^2) )
  diffMSDs = data.frame(modelMSDs , winner_mses)
  set.seed(42)
  res = boot(data=diffMSDs,statistic=cmpMSD,R=10000)
  print(paste("baseline model is superior to model in", round(10000*mean(res$t))/100, "percent of the samples"))
  print(paste("p-value:", 1-mean(res$t) ))
  print('*******************')
  print('*******************')
  print('*******************')
}
