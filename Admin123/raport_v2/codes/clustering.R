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

data3cols <- data[c("LDH_last", "hsCRP_last", "lymphocytes_last")]
data3cols <- na.omit(data3cols)

# HDBSCAN
clusters1 <- hdbscan(dist(data3cols), minPts = 13)
visualization_data_1 <- data3cols %>% 
  cbind(data.frame(cluster = clusters1$cluster, outcome = (na.omit(data))$outcome + 1))


ggplot(visualization_data_1, aes(x = LDH_last, y = hsCRP_last, color = cluster)) +
  geom_point()
# ggplot(visualization_data_1, aes(x = LDH_last, y = hsCRP_last, color = outcome)) +
#   geom_point()

ggplot(visualization_data_1, aes(x = LDH_last, y = lymphocytes_last, color = cluster)) +
  geom_point()
# ggplot(visualization_data_1, aes(x = LDH_last, y = lymphocytes_last, color = outcome)) +
#   geom_point()

ggplot(visualization_data_1, aes(x = lymphocytes_last, y = hsCRP_last, color = cluster)) +
  geom_point()
# ggplot(visualization_data_1, aes(x = lymphocytes_last, y = hsCRP_last, color = outcome)) +
#   geom_point()




# KMEANS
clusters2 <- kmeans(data3cols, 2)
visualization_data_2 <- data3cols %>% 
  cbind(data.frame(cluster = clusters2$cluster, outcome = (na.omit(data))$outcome + 1))

ggplot(visualization_data_2, aes(x = LDH_last, y = hsCRP_last, color = cluster)) +
  geom_point()
ggplot(visualization_data_2, aes(x = LDH_last, y = hsCRP_last, color = outcome)) +
  geom_point()

ggplot(visualization_data_2, aes(x = LDH_last, y = lymphocytes_last, color = cluster)) +
  geom_point()
ggplot(visualization_data_2, aes(x = LDH_last, y = lymphocytes_last, color = outcome)) +
  geom_point()

ggplot(visualization_data_2, aes(x = lymphocytes_last, y = hsCRP_last, color = cluster)) +
  geom_point()
ggplot(visualization_data_2, aes(x = lymphocytes_last, y = hsCRP_last, color = outcome)) +
  geom_point()


# KMEANS ON SINGLE SETS

data3b <- na.omit(data3)
data3b <- data3b[c("LDH_last", "hsCRP_last", "lymphocytes_last")]

clusters3 <- kmeans(data3b, 2)
visualization_data_3 <- data3b %>% 
  cbind(data.frame(cluster = clusters3$cluster, outcome = (na.omit(data3))$outcome + 1))

write.csv(visualization_data_3, "kmeans_tongji110.csv")

ggplot(visualization_data_3, aes(x = LDH_last, y = hsCRP_last, color = cluster)) +
  geom_point()
ggplot(visualization_data_3, aes(x = LDH_last, y = hsCRP_last, color = outcome)) +
  geom_point()

ggplot(visualization_data_3, aes(x = LDH_last, y = lymphocytes_last, color = cluster)) +
  geom_point()
ggplot(visualization_data_3, aes(x = LDH_last, y = lymphocytes_last, color = outcome)) +
  geom_point()

ggplot(visualization_data_3, aes(x = lymphocytes_last, y = hsCRP_last, color = cluster)) +
  geom_point()
ggplot(visualization_data_3, aes(x = lymphocytes_last, y = hsCRP_last, color = outcome)) +
  geom_point()

# KMEANS TONGJI 300+cośtam
data4b <- na.omit(data4)
data4b <- data4b[c("LDH_last", "hsCRP_last", "lymphocytes_last")]

clusters4 <- kmeans(data4b, 2)
visualization_data_4 <- data4b %>% 
  cbind(data.frame(cluster = clusters4$cluster, outcome = (na.omit(data4))$outcome + 1))

write.csv(visualization_data_4, "kmeans_tongji375.csv")

ggplot(visualization_data_4, aes(x = LDH_last, y = hsCRP_last, color = cluster)) +
  geom_point()
ggplot(visualization_data_4, aes(x = LDH_last, y = hsCRP_last, color = outcome)) +
  geom_point()

ggplot(visualization_data_4, aes(x = LDH_last, y = lymphocytes_last, color = cluster)) +
  geom_point()
ggplot(visualization_data_4, aes(x = LDH_last, y = lymphocytes_last, color = outcome)) +
  geom_point()

ggplot(visualization_data_4, aes(x = lymphocytes_last, y = hsCRP_last, color = cluster)) +
  geom_point()
ggplot(visualization_data_4, aes(x = lymphocytes_last, y = hsCRP_last, color = outcome)) +
  geom_point()








