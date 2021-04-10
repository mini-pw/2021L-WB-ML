source("libraries.R")

# Preparing data
data1 <- data.frame(read.csv('data/Outcomerea_FR.csv'))
data2 <- data.frame(read.csv('data/St_Antonius_NL.csv'))
data3 <- data.frame(read.csv('data/Tongji_110_CN.csv'))
data4 <- data.frame(read.csv('data/Tongji_375_CN.csv'))

# Simple data exploration raport
#DataExplorer::create_report(data1)
#DataExplorer::create_report(data2)
#DataExplorer::create_report(data3)
#DataExplorer::create_report(data4)


data <- data1 %>% 
  rbind(data2) %>% 
  rbind(data3) %>% 
  rbind(data4)

colnames(data)
#DataExplorer::create_report(data)

data <- data[-(1:3)]

# Preparing training and test set:
set.seed(213)
test_indices <- sample(1:nrow(data), 0.2 * nrow(data))
data_train <- data[-test_indices, ]
x_train <- data_train[colnames(data_train) != "outcome"]
y_train <- data_train[["outcome"]]
data_test <- data[test_indices, ]
x_test <- data_test[colnames(data_test) != "outcome"]
y_test <- data_test[["outcome"]]

# Training model
param <- list(objective = "reg:squarederror", max_depth = 6)
xgb_model <- xgboost(data = as.matrix(x_train), 
                     label = y_train, 
                     params = param,
                     nrounds = 200)

# Assessing model's performance
preds <- ifelse(predict(xgb_model, as.matrix(x_test)) > 0.5, 1, 0) # model predictions
confusionMatrix <- caret::confusionMatrix(factor(preds), factor(y_test))
confusionMatrix
#Accuracy : 0.7979
#Sensitivity : 0.8527          
#Specificity : 0.6875 



# Explaining the model
unified <- xgboost.unify(xgb_model, x_train)
shaps <- treeshap(unified, x_train)
plot_feature_importance(shaps)
# first blood tests are less important for the model, maybe train model using only last blood tests
plot_feature_dependence(shaps, "LDH_last")
plot_feature_dependence(shaps, "lymphocytes_last")
plot_feature_dependence(shaps, "hsCRP_last")


# Same model with only 1 tree
param <- list(objective = "reg:squarederror", max_depth = 1)
xgb_model_single_tree <- xgboost(data = as.matrix(x_train), 
                     label = y_train, 
                     params = param,
                     nrounds = 1)

preds <- ifelse(predict(xgb_model_single_tree, as.matrix(x_test)) > 0.5, 1, 0) # model predictions
confusionMatrix <- caret::confusionMatrix(factor(preds), factor(y_test))
confusionMatrix
#Accuracy : 0.7979 
#Sensitivity : 0.8682          
#Specificity : 0.6562  
# Almost identical results!

# How does the tree look like:
xgboost.unify(xgb_model_single_tree, x_train)$model

# Same model of one tree and max_depth = 4
param <- list(objective = "reg:squarederror", max_depth = 5)
xgb_model_single_tree_shallow <- xgboost(data = as.matrix(x_train), 
                                 label = y_train, 
                                 params = param,
                                 nrounds = 1)

preds <- ifelse(predict(xgb_model_single_tree_shallow, as.matrix(x_test)) > 0.5, 1, 0) # model predictions
confusionMatrix <- caret::confusionMatrix(factor(preds), factor(y_test))
confusionMatrix
#Accuracy : 0.829
#Sensitivity : 0.8605          
#Specificity : 0.7656
# Somehow even better results
# For max_depth of 3, results are not much worse
# For max_depth of 2 too

xgboost.unify(xgb_model_single_tree_shallow, x_train)$model


# Model with only 3 features
colnames(x_train)
param <- list(objective = "reg:squarederror", max_depth = 4)
xgb_model <- xgboost(data = as.matrix(x_train[, c(2, 4, 6)]), 
                     label = y_train, 
                     params = param,
                     nrounds = 200)

# Assessing model's performance
preds <- ifelse(predict(xgb_model, as.matrix(x_test[, c(2, 4, 6)])) > 0.5, 1, 0) # model predictions
confusionMatrix <- caret::confusionMatrix(factor(preds), factor(y_test))
confusionMatrix
#Accuracy : 0.8135               
#Sensitivity : 0.8450          
#Specificity : 0.7500      



# Performance at single dataset

x_data1 <- data1[-(1:4)]
y_data1 <- data1[[4]]

x_data1 <- x_data1[test_indices[test_indices < nrow(data1)], ]
y_data1 <- y_data1[test_indices[test_indices < nrow(data1)]]

param <- list(objective = "reg:squarederror", max_depth = 6)
xgb_model <- xgboost(data = as.matrix(x_train), 
                     label = y_train, 
                     params = param,
                     nrounds = 200)

preds <- ifelse(predict(xgb_model, as.matrix(x_data1)) > 0.5, 1, 0)
confusionMatrix <- caret::confusionMatrix(factor(preds), factor(y_data1))
confusionMatrix

# Reference
# Prediction   0  1
#           0 15  7
#           1  7  8

# Accuracy : 0.6216          
# 95% CI : (0.4476, 0.7754)
# No Information Rate : 0.5946          
# P-Value [Acc > NIR] : 0.4378          
# 
# Kappa : 0.2152          
# 
# Mcnemar's Test P-Value : 1.0000          
#                                           
#             Sensitivity : 0.6818          
#             Specificity : 0.5333          
#          Pos Pred Value : 0.6818          
#          Neg Pred Value : 0.5333          
#              Prevalence : 0.5946          
#          Detection Rate : 0.4054          
#    Detection Prevalence : 0.5946 

