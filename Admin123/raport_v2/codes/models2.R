
d1 <- data.frame(read.csv('data/Outcomerea_FR.csv')) %>% 
  #select(outcome, LDH = LDH_last, lymph = lymphocytes_last, hsCRP = hsCRP_last) %>% 
  na.omit() %>% 
  mutate(set = 1)
  
d2 <- data.frame(read.csv('data/St_Antonius_NL.csv')) %>% 
  #select(outcome, LDH = LDH_last, lymph = lymphocytes_last, hsCRP = hsCRP_last) %>% 
  na.omit() %>% 
  mutate(set = 2)

d3 <- data.frame(read.csv('data/Tongji_110_CN.csv')) %>% 
  #select(outcome, LDH = LDH_last, lymph = lymphocytes_last, hsCRP = hsCRP_last) %>% 
  na.omit() %>% 
  mutate(set = 3)

d4 <- data.frame(read.csv('data/Tongji_375_CN.csv')) %>% 
  #select(outcome, LDH = LDH_last, lymph = lymphocytes_last, hsCRP = hsCRP_last) %>% 
  na.omit() %>% 
  mutate(set = 4)

d5 <- data.frame(read.csv('data/Northwell_US.csv')) %>% 
  #select(outcome, LDH = LDH_last, lymph = lymphocytes_last, hsCRP = hsCRP_last) %>% 
  na.omit() %>% 
  mutate(set = 5)

slices <- function(d) {
  list(
    p1 = ggplot(d, aes(x = LDH, y = hsCRP, color = outcome)) +
    geom_point(), 
    p2 = ggplot(d, aes(x = LDH, y = lymph, color = outcome)) +
    geom_point(), 
    p3 = ggplot(d, aes(x = lymph, y = hsCRP, color = outcome)) +
      geom_point()
  )
}

slices_set <- function(d) {
  list(
    p1 = ggplot(d, aes(x = LDH, y = hsCRP, color = set)) +
      geom_point(), 
    p2 = ggplot(d, aes(x = LDH, y = lymph, color = set)) +
      geom_point(), 
    p3 = ggplot(d, aes(x = lymph, y = hsCRP, color = set)) +
      geom_point()
  )
}

slices_cluster <- function(d) {
  list(
    p1 = ggplot(d, aes(x = LDH, y = hsCRP, color = cluster)) +
      geom_point(), 
    p2 = ggplot(d, aes(x = LDH, y = lymph, color = cluster)) +
      geom_point(), 
    p3 = ggplot(d, aes(x = lymph, y = hsCRP, color = cluster)) +
      geom_point()
  )
}

slices(d1)$p1
slices(d1)$p2
slices(d1)$p3

all <- d1 %>% 
  rbind(d2) %>% 
  rbind(d3) %>% 
  rbind(d4) %>% 
  rbind(d5)

all <- all %>% 
  mutate(set = factor(set))


slices(all)$p1
slices(all)$p2
slices(all)$p3


slices_set(all)$p1
slices_set(all)$p2
slices_set(all)$p3



all_log1p <- all %>% 
  mutate(LDH = log(LDH) + 1, lymph = log(lymph) + 1, hsCRP = log(hsCRP) + 1)

slices_set(all_log1p)$p1
slices_set(all_log1p)$p2
slices_set(all_log1p)$p3

slices(all_log1p)$p1
slices(all_log1p)$p2
slices(all_log1p)$p3


d3_log1p <- d3 %>% 
  mutate(LDH = log(LDH) + 1, lymph = log(lymph) + 1, hsCRP = log(hsCRP) + 1)

slices(d3_log1p)$p1
slices(d3_log1p)$p2
slices(d3_log1p)$p3


d4_log1p <- d4 %>% 
  mutate(LDH = log(LDH) + 1, lymph = log(lymph) + 1, hsCRP = log(hsCRP) + 1) %>% 
  # removing 3 outliers
  filter(LDH > 5.75, lymph > 0)

slices(d4_log1p)$p1
slices(d4_log1p)$p2
slices(d4_log1p)$p3

kmeans_d4_log1p <- d4_log1p %>% cbind(cluster = kmeans(d4_log1p, 2)$cluster)

slices_cluster(kmeans_d4_log1p)$p1
slices_cluster(kmeans_d4_log1p)$p2
slices_cluster(kmeans_d4_log1p)$p3

slices(kmeans_d4_log1p)$p1
slices(kmeans_d4_log1p)$p2
slices(kmeans_d4_log1p)$p3

write.csv(kmeans_d4_log1p, "results/kmeans_log1p_tongji_375.csv")

# confusion matrix for clustering
cm_kmeans_d4_log1p <- caret::confusionMatrix(factor(kmeans_d4_log1p$cluster - 1), factor(kmeans_d4_log1p$outcome))
cm_kmeans_d4_log1p
# wykryło wszystkie śmierci, część przeżywających uznało jako umierających




set.seed(213)

test_indices1 <- sample(1:nrow(d1), 0.25 * nrow(d1))
test1 <- d1[test_indices1, ]
train1 <- d1[-test_indices1, ]

test_indices2 <- sample(1:nrow(d2), 0.25 * nrow(d2))
test2 <- d2[test_indices2, ]
train2 <- d2[-test_indices2, ]

test_indices3 <- sample(1:nrow(d3), 0.25 * nrow(d3))
test3 <- d3[test_indices3, ]
train3 <- d3[-test_indices3, ]

test_indices4 <- sample(1:nrow(d4), 0.25 * nrow(d4))
test4 <- d4[test_indices4, ]
train4 <- d4[-test_indices4, ]

test_indices5 <- sample(1:nrow(d5), 0.25 * nrow(d5))
test5 <- d5[test_indices5, ]
train5 <- d5[-test_indices5, ]

train <- train1 %>% 
  rbind(train2) %>% 
  rbind(train3) %>% 
  rbind(train4) %>% 
  rbind(train5)

x_train <- train[, !(colnames(train) %in% c("outcome", "set"))]
y_train <- train$outcome


xgb_model <- xgboost(data = as.matrix(x_train), 
                     label = y_train, 
                     params = list(objective = "reg:squarederror", max_depth = 3), 
                     nrounds = 1)

unified <- xgboost.unify(xgb_model, x_train)
shaps <- treeshap(unified, x_train)

plot_feature_importance(shaps)
plot_feature_dependence(shaps, "LDH")
plot_feature_dependence(shaps, "hsCRP")
plot_feature_dependence(shaps, "lymph") # dla wysokiego lymph model sie stabilizuje


test_all <- test1 %>% 
  rbind(test2) %>% 
  rbind(test3) %>% 
  rbind(test4) %>% 
  rbind(test5)

x_test_all <- test_all %>% 
  select(-outcome, -set)
y_test_all <- test_all$outcome

preds <- predict(xgb_model, as.matrix(x_test_all)) 

roc_curve <- pROC::roc(response = factor(y_test_all), predictor = preds, levels = c(0, 1))

plot(roc_curve)

threshold <- roc_curve$thresholds[which.max(roc_curve$sensitivities + roc_curve$specificities)]
threshold

confm <- caret::confusionMatrix(factor(ifelse(preds > threshold, 1, 0)), 
                                factor(y_test_all))
confm





