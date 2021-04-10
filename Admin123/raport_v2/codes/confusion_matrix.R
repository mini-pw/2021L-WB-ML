# ggplot  confusion matrix visualization

confusion_plot <- function(confusionMatrix) {
  ggplot(confusionMatrix$table %>% as_data_frame(), aes(x=Prediction, y=Reference)) +
    geom_tile(aes(fill = n), alpha = 0.6) +
    geom_text(aes(label = sprintf("%d", n)), fontface = "bold") +
    ggtitle("Confusion matrix for the merged data xgboost model") +
    scale_x_discrete(labels = c("Survived", "Died")) +
    scale_y_discrete(labels = c("Survived", "Died")) +
    scale_fill_distiller(direction = 1) +
    theme_bw() +
    theme(legend.position = "none")
}
