set.seed(213)

d1 <- data.frame(read.csv('data/Outcomerea_FR.csv')) %>% 
  select(outcome, LDH = LDH_last, lymph = lymphocytes_last, hsCRP = hsCRP_last) %>% 
  na.omit() %>% 
  mutate(set = 1)

d2 <- data.frame(read.csv('data/St_Antonius_NL.csv')) %>% 
  select(outcome, LDH = LDH_last, lymph = lymphocytes_last, hsCRP = hsCRP_last) %>% 
  na.omit() %>% 
  mutate(set = 2)

d3 <- data.frame(read.csv('data/Tongji_110_CN.csv')) %>% 
  select(outcome, LDH = LDH_last, lymph = lymphocytes_last, hsCRP = hsCRP_last) %>% 
  na.omit() %>% 
  mutate(set = 3)

d4 <- data.frame(read.csv('data/Tongji_375_CN.csv')) %>% 
  select(outcome, LDH = LDH_last, lymph = lymphocytes_last, hsCRP = hsCRP_last) %>% 
  na.omit() %>% 
  mutate(set = 4)

d5 <- data.frame(read.csv('data/yan_reply.csv')) %>% 
  select(outcome, LDH = LDH_last, lymph = lymphocytes_last, hsCRP = hsCRP_last) %>% 
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

#write.csv(kmeans_d4_log1p, "results/kmeans_log1p_tongji_375.csv")

# confusion matrix for clustering
cm_kmeans_d4_log1p <- caret::confusionMatrix(factor(kmeans_d4_log1p$cluster - 1), factor(kmeans_d4_log1p$outcome))
cm_kmeans_d4_log1p
# wykryło wszystkie śmierci, część przeżywających uznało jako umierających
cm_kmeans_d4_log1p %>% confusion_plot()

