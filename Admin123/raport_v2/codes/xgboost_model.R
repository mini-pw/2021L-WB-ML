# preparing datasets

d1 <- data.frame(read.csv('data/Outcomerea_FR.csv')) %>% 
  select(-admission, -discharge, -id) %>% 
  #select(outcome, LDH = LDH_last, lymph = lymphocytes_last, hsCRP = hsCRP_last) %>% 
  na.omit() %>% 
  mutate(set = 1)

d2 <- data.frame(read.csv('data/St_Antonius_NL.csv')) %>% 
  select(-admission, -discharge, -id) %>% 
  #select(outcome, LDH = LDH_last, lymph = lymphocytes_last, hsCRP = hsCRP_last) %>% 
  na.omit() %>% 
  mutate(set = 2)

d3 <- data.frame(read.csv('data/Tongji_110_CN.csv')) %>% 
  select(-admission, -discharge, -id) %>% 
  #select(outcome, LDH = LDH_last, lymph = lymphocytes_last, hsCRP = hsCRP_last) %>% 
  na.omit() %>% 
  mutate(set = 3)

d4 <- data.frame(read.csv('data/Tongji_375_CN.csv')) %>% 
  select(-admission, -discharge, -id) %>% 
  #select(outcome, LDH = LDH_last, lymph = lymphocytes_last, hsCRP = hsCRP_last) %>% 
  na.omit() %>% 
  mutate(set = 4)

d5 <- data.frame(read.csv('data/Northwell_US.csv')) %>% 
  select(-admission, -discharge, -id) %>% 
  #select(outcome, LDH = LDH_last, lymph = lymphocytes_last, hsCRP = hsCRP_last) %>% 
  na.omit() %>% 
  mutate(set = 5)

part_test <- 0.3
set.seed(123)

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



# FIRST MODEL

xgb_model_1 <- xgboost(data = as.matrix(x_train), 
                     label = y_train, 
                     params = list(objective = "reg:squarederror", max_depth = 6), 
                     nrounds = 200)



# explaining model

xgboost::xgb.importance(model = xgb_model_1)

shaps_1 <- treeshap(xgboost.unify(xgb_model_1, x_train), x_train)

plot_feature_importance(shaps_1)
plot_feature_dependence(shaps_1, "LDH_last")
plot_feature_dependence(shaps_1, "hsCRP_last")
plot_feature_dependence(shaps_1, "lymphocytes_last") # dla wysokiego lymph ich wartosci shap sie stabilizują
plot_feature_dependence(shaps_1, "LDH_first") # nic tu nie widac
plot_feature_dependence(shaps_1, "hsCRP_first")
plot_feature_dependence(shaps_1, "lymphocytes_first") # też dla wysokiego lymph ich wartosci shap sie stabilizują

# model performance
preds <- predict(xgb_model_1, as.matrix(x_test_all)) 

roc_curve <- pROC::roc(response = factor(y_test_all), predictor = preds, levels = c(0, 1))
plot(roc_curve)


auc <- pROC::auc(roc_curve)
auc

threshold <- roc_curve$thresholds[which.max(roc_curve$sensitivities + roc_curve$specificities)]
threshold

confm <- caret::confusionMatrix(factor(ifelse(preds > threshold, 1, 0)), 
                                factor(y_test_all))
confm













