
performances <- function(x_test, y_test, threshold_to_use = NULL, model = xgb_model_3) {
  preds <- predict(model, as.matrix(x_test)) 
  
  roc_curve <- pROC::roc(response = factor(y_test), predictor = preds, levels = c(0, 1))
  auc <- pROC::auc(roc_curve)
  threshold_optimal <- roc_curve$thresholds[which.max(roc_curve$sensitivities + roc_curve$specificities)]
  
  threshold_to_use <- ifelse(is.null(threshold_to_use), threshold_optimal, threshold_to_use)
  
  confm <- caret::confusionMatrix(factor(ifelse(preds > threshold_to_use, 1, 0)), 
                                  factor(y_test))
  return(list(roc = roc_curve, auc = auc, threshold = threshold_optimal, confm = confm))
}


# preparing datasets

d1 <- data.frame(read.csv('data/Outcomerea_FR.csv')) %>% 
  #select(-admission, -discharge, -id) %>% 
  select(outcome, LDH = LDH_last, lymph = lymphocytes_last, hsCRP = hsCRP_last) %>% 
  na.omit() %>% 
  mutate(set = 1)

d2 <- data.frame(read.csv('data/St_Antonius_NL.csv')) %>% 
  #select(-admission, -discharge, -id) %>% 
  select(outcome, LDH = LDH_last, lymph = lymphocytes_last, hsCRP = hsCRP_last) %>% 
  na.omit() %>% 
  mutate(set = 2)

d3 <- data.frame(read.csv('data/Tongji_110_CN.csv')) %>% 
  #select(-admission, -discharge, -id) %>% 
  select(outcome, LDH = LDH_last, lymph = lymphocytes_last, hsCRP = hsCRP_last) %>% 
  na.omit() %>% 
  mutate(set = 3)

d4 <- data.frame(read.csv('data/Tongji_375_CN.csv')) %>% 
  #select(-admission, -discharge, -id) %>% 
  select(outcome, LDH = LDH_last, lymph = lymphocytes_last, hsCRP = hsCRP_last) %>% 
  na.omit() %>% 
  mutate(set = 4)

d5 <- data.frame(read.csv('data/Northwell_US.csv')) %>% 
  #select(-admission, -discharge, -id) %>% 
  select(outcome, LDH = LDH_last, lymph = lymphocytes_last, hsCRP = hsCRP_last) %>% 
  na.omit() %>% 
  mutate(set = 5)

part_test <- 0.25
set.seed(213)

test_indices1 <- sample(1:nrow(d1), part_test * nrow(d1))
test1 <- d1[test_indices1, ]
train1 <- d1[-test_indices1, ]

test_indices2 <- sample(1:nrow(d2), part_test * nrow(d2))
test2 <- d2[test_indices2, ]
train2 <- d2[-test_indices2, ]

test_indices3 <- sample(1:nrow(d3), part_test * nrow(d3))
test3 <- d3[test_indices3, ]
train3 <- d3[-test_indices3, ]

test_indices4 <- sample(1:nrow(d4), part_test * nrow(d4))
test4 <- d4[test_indices4, ]
train4 <- d4[-test_indices4, ]

test_indices5 <- sample(1:nrow(d5), part_test * nrow(d5))
test5 <- d5[test_indices5, ]
train5 <- d5[-test_indices5, ]

train <- train1 %>% 
  rbind(train2) %>% 
  rbind(train3) %>% 
  rbind(train4) %>% 
  rbind(train5)

x_train <- train %>% 
  select(-outcome, -set)
y_train <- train$outcome


test_all <- test1 %>% 
  rbind(test2) %>% 
  rbind(test3) %>% 
  rbind(test4) %>% 
  rbind(test5)

x_test_all <- test_all %>% 
  select(-outcome, -set)
y_test_all <- test_all$outcome

x_test1 <- test1 %>% 
  select(-outcome, -set)
y_test1 <- test1$outcome

x_test2 <- test2 %>% 
  select(-outcome, -set)
y_test2 <- test2$outcome

x_test3 <- test3 %>% 
  select(-outcome, -set)
y_test3 <- test3$outcome

x_test4 <- test4 %>% 
  select(-outcome, -set)
y_test4 <- test4$outcome

x_test5 <- test5 %>% 
  select(-outcome, -set)
y_test5 <- test5$outcome



# SECOND MODEL

xgb_model_2 <- xgboost(data = as.matrix(x_train), 
                       label = y_train, 
                       params = list(objective = "reg:squarederror", max_depth = 6), 
                       nrounds = 200)

# explaining model

shaps_2 <- treeshap(xgboost.unify(xgb_model_2, x_train), x_train)

plot_feature_importance(shaps_2)
plot_feature_dependence(shaps_2, "LDH")
plot_feature_dependence(shaps_2, "hsCRP")
plot_feature_dependence(shaps_2, "lymph") # dla wysokiego lymph ich wartosci shap sie stabilizują

# model performance
preds <- predict(xgb_model_2, as.matrix(x_test_all)) 

roc_curve <- pROC::roc(response = factor(y_test_all), predictor = preds, levels = c(0, 1))
plot(roc_curve)

auc <- pROC::auc(roc_curve)
auc

threshold <- roc_curve$thresholds[which.max(roc_curve$sensitivities + roc_curve$specificities)]
threshold

confm <- caret::confusionMatrix(factor(ifelse(preds > threshold, 1, 0)), 
                                factor(y_test_all))
confm

# THIRD MODEL - SINGLE TREE


xgb_model_3 <- xgboost(data = as.matrix(x_train), 
                       label = y_train, 
                       params = list(objective = "reg:squarederror", max_depth = 4), 
                       nrounds = 1)


# how does the tree look like
unified <- xgboost.unify(xgb_model_3, x_train)

unified$model
xgb.plot.tree(model = xgb_model_3)

# model performance

th <- 0.42

# all sets
perfs <- performances(x_test_all, y_test_all, threshold_to_use = th)
perfs$roc %>% plot()
perfs$auc
perfs$confm
perfs$threshold

# 1
perfs <- performances(x_test1, y_test1, threshold_to_use = th)
perfs$roc %>% plot()
perfs$auc
perfs$confm

# 2
perfs <- performances(x_test2, y_test2, threshold_to_use = th)
perfs$roc %>% plot()
perfs$auc
perfs$confm

# 3
perfs <- performances(x_test3, y_test3, threshold_to_use = th)
perfs$roc %>% plot()
perfs$auc
perfs$confm

# 4
perfs <- performances(x_test4, y_test4, threshold_to_use = th)
perfs$roc %>% plot()
perfs$auc
perfs$confm

# 5
perfs <- performances(x_test5, y_test5, threshold_to_use = th)
perfs$roc %>% plot()
perfs$auc
perfs$confm


# FORTH MODEL - increasing weight of first two sets
weights <- c(rep(2, times = nrow(train1)), 
             rep(2, times = nrow(train2)), 
             rep(1, times = nrow(train3)), 
             rep(1, times = nrow(train4)), 
             rep(1, times = nrow(train5)))


xgbMatrix <- xgb.DMatrix(as.matrix(x_train), 
                         label = y_train, 
                         weight = weights)

xgb_model_3 <- xgboost(data = xgbMatrix, 
                       label = y_train, 
                       params = list(objective = "reg:squarederror", max_depth = 5), 
                       nrounds = 1)


# model performance

th <- 0.47

# all sets
perfs <- performances(x_test_all, y_test_all, threshold_to_use = th)
perfs$roc %>% plot()
perfs$auc
perfs$confm
perfs$threshold

confusion_plot(perfs$confm)

# 1
perfs <- performances(x_test1, y_test1, threshold_to_use = th)
perfs$roc %>% plot()
perfs$auc
perfs$confm
perfs$confm %>% confusion_plot()

# 2
perfs <- performances(x_test2, y_test2, threshold_to_use = th)
perfs$roc %>% plot()
perfs$auc
perfs$confm
perfs$confm %>% confusion_plot()

# 3
perfs <- performances(x_test3, y_test3, threshold_to_use = th)
perfs$roc %>% plot()
perfs$auc
perfs$confm

# 4
perfs <- performances(x_test4, y_test4, threshold_to_use = th)
perfs$roc %>% plot()
perfs$auc
perfs$confm

# 5
perfs <- performances(x_test5, y_test5, threshold_to_use = th)
perfs$roc %>% plot()
perfs$auc
perfs$confm

confusion_plot(perfs$confm)


