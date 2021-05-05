library(iBreakDown)
library(readr);
setwd("C:/Users/hrucz/Dropbox/WB_Faza2")
covid <- read_csv("200518COVID19MEXICO.csv")

head(covid)
#covid<-covid[-34]
#covid<-covid[-13]
covid=covid[1:150200, c(6,14,15,16,20,21,22,23,24,25,26,27,28,29,30,31)]
covid
ranger_model <- ranger::ranger(RESULTADO~., data = covid, classification = TRUE, probability = TRUE)
#?ranger
custom_predict <- function(X.model, new_data) {
  predict(X.model, new_data)$predictions[,1]
}

covid[1,]
bd <- break_down(ranger_model, new_observation =covid[56377,-16], data = covid, predict_function = custom_predict)
bd
plot(bd)
?break_down


bd <- shap(ranger_model, new_observation = covid[56377,-16], data = covid, predict_function = custom_predict)
plot(bd)
# SEXO, INTUBADO, NEUMONIA, EDAD, DIABETES,  EPOC,  ASMA, INMUSUPR, HIPERTENSION, OTRA_COM, CARDIOVASCULAR, OBESIDAD, RENAL_CRONICA, TABAQUISMO, OTRO_CASO, RESULTADO
# P£EÆ, INTUBACJA, PNEUMONIA, WIEK, CUKRZYCA, POChP, ASTMA, INMUSUPR, NADCIŒNIENIE, INNE_COM, SERCOWO-NACZYNIOWE, OTY£OŒÆ, DZIECKO_CHRONICZNE, PALENIE, INNY PRZYPADEK, WYNIK 

#------------------------------------------------------------------------------------------------------------------------------

library(readr)
library(DALEX)
library(shiny)
library("ranger")
setwd("C:/Users/hrucz/Dropbox/WB_Faza2")
covid <- read_csv("200518COVID19MEXICO.csv")

covid=covid[1:150200, c(6,14,15,16,20,21,22,23,24,25,26,27,28,29,30,31)]
covid10
set.seed(123)
covid10 = covid[sample(nrow(covid),10000),]
ranger_model <- ranger::ranger(RESULTADO~., data = covid10, classification = TRUE, probability = TRUE)
importance(ranger_model)
#wybieramy 2/3 na train 1/3 na test
train.idx <- sample(x = 150200, size = 100000)
covid.train <- covid[train.idx, ]
covid.test <- covid[-train.idx, ]

rf <- ranger(RESULTADO ~ ., data = covid.train,classification = TRUE, write.forest = TRUE)
pred <- predict(rf, data = covid.test)
c_mat <- table(covid.test$RESULTADO, predictions(pred))
accuracy <- (c_mat[1,1]+c_mat[2,2])/(c_mat[1,1]+c_mat[1,2]+c_mat[2,1]+c_mat[2,2])
precision <- c_mat[1,1]/(c_mat[1,1]+c_mat[2,1])
sensitivity <- c_mat[1,1]/(c_mat[1,1]+c_mat[1,2])
c_mat
accuracy
precision
sensitivity




#--------------------
#Caret w tuningu

library(caret)
set.seed(1)
# Training Parameters
CV_folds <- 5 # number of folds
CV_repeats <- 3 # number of repeats
minimum_resampling <- 5 # minimum number of resamples
# trainControl object for standard repeated cross-validation
train_control <- caret::trainControl(method = "repeatedcv", number = CV_folds, repeats = CV_repeats, 
                                     verboseIter = FALSE, returnData = FALSE) 

# trainControl object for repeated cross-validation with grid search
adapt_control_grid <- caret::trainControl(method = "adaptive_cv", number = CV_folds, repeats = CV_repeats, 
                                          adaptive = list(min = minimum_resampling, # minimum number of resamples tested before model is excluded
                                                          alpha = 0.05, # confidence level used to exclude parameter settings
                                                          method = "gls", # generalized least squares
                                                          complete = TRUE), 
                                          search = "grid", # execute grid search
                                          verboseIter = FALSE, returnData = FALSE) 

# Create grid
ranger_Grid <- expand.grid(
  num.trees = c(100,500,1000),
  write.forest=TRUE,
  mtry=c(1,3,5),
  max.depth=c(3,5),
  classification = TRUE,
  min.node.size=c(1),
  splitrule=c("gini","extratrees")
  
) 


rf <- caret::train(RESULTADO ~ ., data = covid, method="ranger", tuneGrid = ranger_Grid, tuneLength=20,verbose=TRUE)
?ranger

#### To mo¿e dla XGBoosta

library(readr)
library(DALEX)
library(shiny)
library("ranger")
setwd("C:/Users/hrucz/Dropbox/WB_Faza2")
covid <- read_csv("200518COVID19MEXICO.csv")

covid=covid[1:150200, c(6,14,15,16,20,21,22,23,24,25,26,27,28,29,30,31)]
set.seed(123)
#ranger_model <- ranger::ranger(RESULTADO~., data = covid10, classification = TRUE, probability = TRUE)
train.idx <- sample(x = 150200, size = 100000)
covid.train <- covid[train.idx, ]
covid.test <- covid[-train.idx, ]

# Training Parameters
CV_folds <- 5 # number of folds
CV_repeats <- 3 # number of repeats
minimum_resampling <- 5 # minimum number of resamples
# trainControl object for standard repeated cross-validation
train_control <- caret::trainControl(method = "repeatedcv", number = CV_folds, repeats = CV_repeats, 
                                     verboseIter = FALSE, returnData = FALSE) 

# trainControl object for repeated cross-validation with grid search
adapt_control_grid <- caret::trainControl(method = "adaptive_cv", number = CV_folds, repeats = CV_repeats, 
                                          adaptive = list(min = minimum_resampling, # minimum number of resamples tested before model is excluded
                                                          alpha = 0.05, # confidence level used to exclude parameter settings
                                                          method = "gls", # generalized least squares
                                                          complete = TRUE), 
                                          search = "grid", # execute grid search
                                          verboseIter = FALSE, returnData = FALSE) 

XGBoost_Linear_grid <- expand.grid(
  nrounds = c(50, 100, 250, 500), # number of boosting iterations
  eta = c(0.01, 0.1, 1),  # learning rate, low value means model is more robust to overfitting
  lambda = c(0.1, 0.5, 1), # L2 Regularization (Ridge Regression)
  alpha =  c(0.1, 0.5, 1) # L1 Regularization (Lasso Regression)
) 
GS_XGBoost_Linear_model <- caret::train(RESULTADO ~., 
                                        data = covid,
                                        method = "xgbLinear",
                                        trControl = adapt_control_grid,
                                        verbose = FALSE, 
                                        #silent = 1,
                                        # tuneLength = 20
                                        tuneGrid = XGBoost_Linear_grid
)

#------------------
#XGBoost

library(readr)
library(DALEX)
library(shiny)
library("ranger")
setwd("C:/Users/hrucz/Dropbox/WB_Faza2")
covid <- read_csv("200518COVID19MEXICO.csv")
covid=covid[1:150200, c(6,14,15,16,20,21,22,23,24,25,26,27,28,29,30,31)]
covid=as.matrix(covid)
set.seed(123)
#wybieramy 2/3 na train 1/3 na test
train.idx <- sample(x = 150200, size = 100000)
covid.train <- covid[train.idx, ]
covid.test <- covid[-train.idx, ]
covid.train[,16]=covid.train[,16]-1
covid.train
covid.train[,16]-1
dtrain <- xgb.DMatrix(data = covid.train[,-16],label=covid.train[,16])
dtest <- xgb.DMatrix(data = covid.test[,-16],label=covid.test[,16])
xgb_model <- xgboost(RESULTADO~., data = dtrain,  max.depth = 8, eta = 0.01,gamma=0.01, nthread = 10,booster="gbtree", nrounds = 100, objective = "binary:logistic",
                     base_score=0.5)
pred <- predict(xgb_model, data = dtrain, newdata=dtest)
pred
prediction <- as.numeric(pred > 0.5)

c_mat <- table(as.numeric(covid.test[,16]), prediction)
accuracy <- (c_mat[1,1]+c_mat[2,2])/(c_mat[1,1]+c_mat[1,2]+c_mat[2,1]+c_mat[2,2])
precision <- c_mat[1,1]/(c_mat[1,1]+c_mat[2,1])
sensitivity <- c_mat[1,1]/(c_mat[1,1]+c_mat[1,2])
c_mat
accuracy
precision
sensitivity

#-----------

#importance(ranger_model)
# wybieramy 2/3 na train 1/3 na test
#train.idx <- sample(x = 150200, size = 100000)
#covid.train <- covid[train.idx, ]
#covid.test <- covid[-train.idx, ]

#rf <- ranger(RESULTADO ~ ., data = covid,classification = TRUE, write.forest = TRUE)
#pred <- predict(rf, data = covid.test)
#table(covid.test$RESULTADO, predictions(pred))
#length(pred)
#length(predictions(pred))
#length(covid.test$RESULTADO)

