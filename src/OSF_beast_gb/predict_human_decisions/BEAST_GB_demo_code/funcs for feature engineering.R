distSample = function( Numbers, Probabilities, rndNum ){
  # Sampling a single number from a discrete distribution
  #   The possible Numbers in the distribution with their resective
  #   Probabilities. rndNum is a randomly drawn probability 
  #
  #   Conditions on Input (not checked):
  #   1. Numbers and Probabilites correspond one to one (i.e. first number is
  #   drawn w.p. first probability etc)
  #   2. rndNum is a number between zero and one
  #   3. Probabilites is a probability vector
  
  cumProb = 0
  sampledInt = 0
  while (rndNum > cumProb){
    sampledInt = sampledInt +1
    cumProb = cumProb + Probabilities[sampledInt]
  }
  return(Numbers[sampledInt])
}

CPC18_getDist = function(H,pH,L,LotShape,LotNum){
  # Extract true full distributions of an option in CPC18
  #   input is high outcome (int), its probability (double), low outcome
  #   (int), the shape of the lottery ('-'/'Symm'/'L-skew'/'R-skew' only), and
  #   the number of outcomes in the lottery.
  #   output is a matrix with first column a list of outcomes (sorted
  #   ascending) and the second column their respective probabilities.
  
  if (LotShape=='-'){
    if (pH == 1){
      Dist = cbind(H, pH)}
    else{
      Dist = rbind(c(L,1-pH), c(H,pH))
    }
  } else { # H is multioutcome
    #compute H distribution
    if (LotShape=='Symm') {
      highDist = cbind(rep(NA,LotNum),rep(NA,LotNum))
      k = LotNum - 1
      for (i in 0:k) {
        highDist[i+1,1] = H - k/2 + i
        highDist[i+1,2] = pH*dbinom(i,k,0.5)
      }
    }
    else if ((LotShape=='R-skew') || (LotShape=='L-skew')){
      highDist = cbind(rep(NA,LotNum),rep(NA,LotNum))
      if (LotShape=='R-skew') {
        C = -1-LotNum;
        distsign = 1}
      else{
        C = 1+LotNum;
        distsign = -1
      }
      for (i in 1:LotNum){
        highDist[i,1] = H + C + distsign*2^i
        highDist[i,2] = pH/(2^i)
      }
      highDist[LotNum,2] =highDist[LotNum,2]*2
    }
    
    # incorporate L into the distribution
    Dist = highDist
    locb = match(L,highDist[,1])
    if (!is.na(locb)){
      Dist[locb,2] = Dist[locb,2] + (1-pH)}
    else if (pH < 1){
      Dist=rbind(Dist,c(L,1-pH))
    }
    Dist = Dist[order(Dist[,1]),]     
  }
}


get_pBetter  = function( DistX, DistY, corr, accuracy = 10000 ) {
  #Return probability that a value drawn from DistX is strictly larger than one drawn from DistY
  # Input: 2 discrete distributions which are set as matrices of 1st column
  # as outcome and 2nd its probability; correlation between the distributions;
  # level of accuracy in terms of number of samples to take from distributions 
  # Output: the estimated probability that X generates value strictly larger than Y, and 
  # the probability that Y generates value strictly larger than X
  
  nXbetter = 0
  nYbetter = 0
  for (j in 1:accuracy){
    rndNum = runif(2)
    sampleX = distSample(DistX[,1],DistX[,2],rndNum[1])
    if (corr == 1){
      sampleY = distSample(DistY[,1],DistY[,2],rndNum[1])}
    else if (corr == -1) {
      sampleY = distSample(DistY[,1],DistY[,2],1-rndNum[1])}
    else {
      sampleY = distSample(DistY[,1],DistY[,2],rndNum[2])
    }
    nXbetter = nXbetter + as.numeric(sampleX > sampleY)
    nYbetter = nYbetter + as.numeric(sampleY > sampleX)
  }
  pXbetter = nXbetter/accuracy
  pYbetter = nYbetter/accuracy
  return(list(pXbetter,pYbetter))
}

CPC15_BEASTsimulation = function (DistA, DistB, Amb, Corr, nBlocks=5, nTrials = 25, firstFeedbackBlock = 2){
  # A single simulation run of the original BEAST model
  SIGMA = 7
  KAPA = 3
  BETA = 2.6
  GAMA = 0.5
  PSI = 0.07
  THETA = 1

  # nTrials = 25
  # firstFeedback = 6
  # nBlocks = 5
  blockSize = nTrials/nBlocks
  firstFeedback = blockSize * (firstFeedbackBlock-1) + 1
  
  #draw personal traits
  sigma = SIGMA*runif(1)
  kapa = sample(KAPA,1)
  beta = BETA*runif(1)
  gama = GAMA*runif(1)
  psi = PSI*runif(1)
  theta = THETA*runif(1)
  
  pBias = rep(nTrials - firstFeedback+1,1)
  ObsPay = matrix(0,nrow=nTrials - firstFeedback+1,ncol=2) # observed outcomes in A (col1) and B (col2)
  Decision = matrix(NA,nrow=nTrials,ncol=1)
  simPred = matrix(NA,nrow=1,ncol=nBlocks)
  # Useful variables
  nA = nrow(DistA) # num outcomes in A
  nB = nrow(DistB) # num outcomes in B
  
  if (Amb == 1){
    ambiguous = TRUE}
  else{
    ambiguous = FALSE
  }
  
  nfeed = 0 # "t"; number of outcomes with feedback so far
  pBias[nfeed+1] = beta/(beta+1+nfeed^theta)
  MinA = DistA[1,1]
  MinB = DistB[1,1]
  MaxOutcome = max(DistA[nA,1],DistB[nB,1])
  SignMax = sign(MaxOutcome)
  if (MinA == MinB)
  {RatioMin = 1}
  else if (sign(MinA) == sign(MinB))
  {RatioMin = min(abs(MinA),abs(MinB))/max(abs(MinA),abs(MinB))}
  else
  {RatioMin = 0}
  
  Range = MaxOutcome - min(MinA, MinB)
  trivial = CPC15_isStochasticDom( DistA, DistB )
  BEVa = DistA[,1]%*%DistA[,2]
  if (ambiguous){
    UEVb = DistB[,1]%*%rep(1/nB,nB)
    BEVb = (1-psi)*(UEVb+BEVa)/2 + psi*MinB
    pEstB = rep(nB,1); # estimation of probabilities in Amb
    t_SPminb = (BEVb -mean(DistB[2:nB,1]))/(MinB-mean(DistB[2:nB,1]))
    if (t_SPminb < 0 )
    {pEstB[1] = 0}
    else if (t_SPminb > 1) 
    {pEstB[1] = 1}
    else
    {pEstB[1] = t_SPminb}
    
    pEstB[2:nB] = (1-pEstB[1])/(nB-1)
  }
  else {
    pEstB = DistB[,2]
    BEVb = DistB[,1]%*%pEstB
  }
  
  # simulation of decisions
  for (trial in 1:nTrials){
    STa = 0
    STb = 0
    # mental simulations
    for (s in 1:kapa) {
      rndNum = runif(2)
      if (rndNum[1] > pBias[nfeed+1]){ # Unbiased technique
        if (nfeed == 0) {
          outcomeA = distSample(DistA[,1],DistA[,2],rndNum[2])
          outcomeB = distSample(DistB[,1],pEstB,rndNum[2])
        } else {
          uniprobs = rep(1/nfeed,nfeed)
          outcomeA = distSample(ObsPay[1:nfeed,1],uniprobs,rndNum[2])
          outcomeB = distSample(ObsPay[1:nfeed,2],uniprobs,rndNum[2])
        }
      } else if (rndNum[1] > (2/3)*pBias[nfeed+1]){ #uniform
        outcomeA = distSample(DistA[,1],rep(1/nA,nA),rndNum[2])
        outcomeB = distSample(DistB[,1],rep(1/nB,nB),rndNum[2])
      } else if (rndNum[1] > (1/3)*pBias[nfeed+1]){ #contingent pessimism
        if (SignMax > 0 && RatioMin < gama){
          outcomeA = MinA
          outcomeB = MinB
        } else {
          outcomeA = distSample(DistA[,1],rep(1/nA,nA),rndNum[2])
          outcomeB = distSample(DistB[,1],rep(1/nB,nB),rndNum[2])
        }
      } else { # Sign
        if (nfeed == 0){
          outcomeA = Range * distSample(sign(DistA[,1]),DistA[,2],rndNum[2])
          outcomeB = Range * distSample(sign(DistB[,1]),pEstB,rndNum[2])
        } else {
          uniprobs = rep(1/nfeed,nfeed)
          outcomeA = Range * distSample(sign(ObsPay[1:nfeed,1]),uniprobs,rndNum[2])
          outcomeB = Range * distSample(sign(ObsPay[1:nfeed,2]),uniprobs,rndNum[2])
        }
      }
      STa = STa + outcomeA
      STb = STb + outcomeB
    }
    STa = STa/kapa
    STb = STb/kapa
    
    #error term
    if (trivial$dom) {
      error = 0
    } else {
      error = sigma*rnorm(1) # positive values contribute to attraction to A
    }
      
    # decision
    Decision[trial] = (BEVa - BEVb) + (STa - STb) + error < 0
    if ((BEVa - BEVb) + (STa - STb) + error == 0)
      Decision[trial] = sample(2,1) -1
    
    if (trial >= firstFeedback){ # got feedback
      nfeed = nfeed +1
      pBias[nfeed+1] = beta/(beta+1+nfeed^theta)
      rndNumObs = runif(1)
      ObsPay[nfeed,1] = distSample(DistA[,1],DistA[,2],rndNumObs) # draw outcome from A
      if (Corr == 1) {
        ObsPay[nfeed,2] = distSample(DistB[,1],DistB[,2],rndNumObs)
      } else if (Corr == -1) {
        ObsPay[nfeed,2] = distSample(DistB[,1],DistB[,2],1-rndNumObs)
      } else {
        ObsPay[nfeed,2] = distSample(DistB[,1],DistB[,2],runif(1))
      } # draw outcome from B
      if (ambiguous) {
        BEVb = (1-1/(nTrials-firstFeedback+1))*BEVb + 1/(nTrials-firstFeedback+1)*ObsPay[nfeed,2]
      }
    }
  }
  
  #compute B-rates for this simulation
  for (b in 1:nBlocks){
    simPred[b] = mean(Decision[((b-1)*blockSize+1):(b*blockSize)])
  }
  return(simPred)
}

CPC15_BEASTpred = function(Ha, pHa, La, LotShapeA, LotNumA, 
                           Hb, pHb, Lb, LotShapeB, LotNumB, 
                           Amb, Corr, 
                           nBlocks = 5, nTrials = 25, firstFeedbackBlock = 2 ) {
  # Prediction of (the original) BEAST model for one problem
  
  Prediction = rep(0,nBlocks)
  
  # get both options' distributions
  DistA = CPC18_getDist(Ha, pHa, La, LotShapeA, LotNumA)
  DistB = CPC18_getDist(Hb, pHb, Lb, LotShapeB, LotNumB)
  
  # run model simulation nSims times
  nSims = 6000;
  for (sim in 1:nSims){
    simPred = CPC15_BEASTsimulation(DistA, DistB, Amb, Corr, nBlocks, nTrials, firstFeedbackBlock);    
    Prediction = Prediction + (1/nSims)*simPred;
  }
  return(Prediction)
}

CPC15_isStochasticDom  = function( DistA, DistB ) {
  #Check if one distribution dominates stochastically the other
  #   Input: 2 discrete distributions which are set as matrices of 1st column
  #   as outcome and 2nd its probability. Output: 'is' a logical output,
  #   'which' a char output ('A', 'B', NaN)
  
  na= nrow(DistA)
  nb= nrow(DistB)
  dom =FALSE
  if (identical(DistA,DistB)) {
    dom = FALSE
    which = NaN}
  else {
    tempa = matrix(1,nrow=na,ncol=1)
    tempb = matrix(1,nrow=nb,ncol=1)
    for (i in 1:nb) {
      sumpa = 0#DistA(i,2)
      j = 1;
      sumpb =sum(DistB[1:i,2])
      while ((sumpa != 1) && (j<=na) && (sumpa + DistA[j,2]  <= sumpb)){
        sumpa = sumpa + DistA[j,2]
        if (sumpa == sumpb) {break}
        j = j +1
      }
      if (j > na) {j = na}
      if (DistB[i,1] < DistA[j,1]){  
        tempb[i] = 0;
        break
      } 
    }
    if (all(tempb!=0)){
      dom = TRUE
      which = 'B'}
    else {
      for (i in 1 : na) {
        sumpb = 0#DistA(i,2)
        j = 1
        sumpa =sum(DistA[1:i,2])
        while ((sumpb != 1) && (j<= nb) && (sumpb + DistB[j,2]  <= sumpa)){
          sumpb = sumpb + DistB[j,2]
          if (sumpa == sumpb) {break}
          j = j +1
        }
        if (j > nb ) {j = nb}
        if (DistA[i,1] < DistB[j,1]){
          tempa[i] = 0
          break
        }
      }
      if (all(tempa!=0)) {
        dom = TRUE
        which = 'A'}
      else {
        dom = FALSE
        which = NA
      }
    }
  }
  return(list("dom" = dom, "which" = which))
}


get_PF_Features = function(Ha, pHa, La, LotShapeA, LotNumA, Hb, pHb, Lb, LotShapeB, LotNumB, Amb, Corr, nBlocks = 5, nTrials = 25, firstFeedbackBlock = 2){
  #Finds the values of the engineered features that are part of Psychological Forest/BEAST-GB
  # Gets as input the parameters defining the choice problem in CPC18 and returns 
  # as output a dataframe with this problem's features
  
  # To compute the distribution's standard deviation
  getSD <- function(vals,probs){
    vals = as.numeric(na.omit(vals))
    probs = as.numeric(na.omit(probs))
    m = vals %*% probs
    sqds= (vals-m[1])^2
    var = probs %*% sqds
    return(sqrt(var))
  }
  
  # Compute "naive" and "psychological" features as per Plonsky, Erev, Hazan, and Tennenholtz, 2017
  DistA = CPC18_getDist(Ha, pHa, La, LotShapeA, LotNumA)
  DistB = CPC18_getDist(Hb, pHb, Lb, LotShapeB, LotNumB)
  diffEV = ((DistB[,1]%*%DistB[,2]) - (DistA[,1]%*%DistA[,2]))[1]
  diffSDs = (getSD(DistB[,1],DistB[,2]) - getSD(DistA[,1],DistA[,2]))[1]
  MinA = DistA[1,1]
  MinB = DistB[1,1]
  diffMins = MinB - MinA
  nA = nrow(DistA)
  nB = nrow(DistB)
  MaxA = DistA[nA,1]
  MaxB = DistB[nB,1]
  diffMaxs = MaxB - MaxA
  
  diffUV = ((DistB[,1] %*% (rep(1,nB)/nB)) - (DistA[,1] %*% (rep(1,nA)/nA)))[1]
  ambiguous = ifelse(Amb == 1, TRUE, FALSE)
  MaxOutcome = max(MaxA,MaxB)
  SignMax = sign(MaxOutcome)
  if (MinA == MinB) {
    RatioMin = 1
  } else if (sign(MinA) == sign(MinB)) {
    RatioMin = min(abs(MinA),abs(MinB))/max(abs(MinA),abs(MinB))
  } else {
    RatioMin = 0
  }
  Range = MaxOutcome - min(MinA, MinB)
  diffSignEV = ((Range*(sign(DistB[,1]))%*%DistB[,2]) - (Range*(sign(DistA[,1]))%*%DistA[,2]))[1]
  trivial = CPC15_isStochasticDom( DistA, DistB )
  whchdom = trivial[[2]]
  Dom = 0
  if ((trivial$dom)&(whchdom == "A")) {Dom = -1}
  if ((trivial$dom)&(whchdom == "B")) {Dom = 1}
  BEVa = DistA[,1]%*%DistA[,2]
  if (ambiguous){
    UEVb = DistB[,1]%*%rep(1/nB,nB)
    BEVb = (UEVb+BEVa+MinB)/3 
    pEstB = rep(nB,1) # estimation of probabilties in Amb
    t_SPminb = (BEVb -mean(DistB[2:nB,1]))/(MinB-mean(DistB[2:nB,1]));
    if (t_SPminb < 0 ) {pEstB[1] = 0} else if (t_SPminb > 1) {pEstB[1] = 1} else {pEstB[1] = t_SPminb}
    pEstB[2:nB] = (1-pEstB[1])/(nB-1)
  } else {
    pEstB = DistB[,2]
    BEVb = DistB[,1]%*%pEstB
  }
  diffBEV0 = (BEVb - BEVa)[1]
  BEVfb = (BEVb+(DistB[,1]%*%DistB[,2]))/2
  diffBEVfb = (BEVfb - BEVa)[1]
  
  sampleDistB = cbind(DistB[,1],pEstB)
  probsBetter = get_pBetter(DistA,sampleDistB,corr=1)
  pAbetter = probsBetter[[1]]
  pBbetter = probsBetter[[2]]
  pBbet_Unbiased1 = pBbetter - pAbetter
  
  sampleUniDistA = cbind(DistA[,1],rep(1/nA,nA))
  sampleUniDistB = cbind(DistB[,1],rep(1/nB,nB))
  probsBetterUni = get_pBetter(sampleUniDistA,sampleUniDistB,corr=1)
  pBbet_Uniform = probsBetterUni[[2]] - probsBetterUni[[1]]
  
  sampleSignA = DistA
  sampleSignA[,1] = sign(sampleSignA[,1])
  sampleSignB = cbind(sign(DistB[,1]),pEstB)
  probsBetterSign = get_pBetter(sampleSignA,sampleSignB,corr=1)
  pBbet_Sign1 = probsBetterSign[[2]] - probsBetterSign[[1]]
  sampleSignBFB = cbind(sign(DistB[,1]),DistB[,2])
  if (Corr == 1){
    probsBetter = get_pBetter(DistA,DistB,corr=1)
    probsBetterSign = get_pBetter(sampleSignA,sampleSignBFB,corr=1)
  } else if (Corr == -1){
    probsBetter = get_pBetter(DistA,DistB,corr=-1)
    probsBetterSign = get_pBetter(sampleSignA,sampleSignBFB,corr=-1)
  } else {
    probsBetter = get_pBetter(DistA,DistB,corr=0)
    probsBetterSign = get_pBetter(sampleSignA,sampleSignBFB,corr=0)
  }
  pBbet_UnbiasedFB = probsBetter[[2]] - probsBetter[[1]]
  pBbet_SignFB = probsBetterSign[[2]] - probsBetterSign[[1]]
  
  # create features data frame
  tmpFeats = data.frame(Ha, pHa, La, LotShapeA, LotNumA, Hb, pHb, Lb, LotShapeB, LotNumB, Amb, Corr,
                        diffEV, diffSDs, diffMins, diffMaxs, diffUV, RatioMin, SignMax, pBbet_Unbiased1,
                        pBbet_UnbiasedFB, pBbet_Uniform, pBbet_Sign1, pBbet_SignFB, Dom, diffBEV0, 
                        diffBEVfb, diffSignEV)
  # duplicate features data frame as per number of blocks
  Feats = tmpFeats[rep(seq(nrow(tmpFeats)),nBlocks),]
  
  # get BEAST model prediction as feature
  beastPs = CPC15_BEASTpred(Ha, pHa, La, LotShapeA, LotNumA, Hb, pHb, Lb, LotShapeB, LotNumB, Amb, Corr, nBlocks, nTrials, firstFeedbackBlock)
  Feats$BEASTpred = t(beastPs)
  
  Feats$block = c(1:nBlocks)
  Feats$Feedback = 1
  Feats$Feedback[Feats$block < firstFeedbackBlock ] = 0
  
  return(Feats)
}
