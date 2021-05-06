# Requires running process_raw.R

source("libraries.R")

draw_histogram <- function(filename, which_LDH){
  df = data.frame(read.csv(str_c("data/", filename, ".csv")))
  
  ldh_min = 150
  ldh_max = 1000
  hist_scale_factor = 300
  title = sprintf('Plot of %s in %s', which_LDH, filename)
  
  df %>% 
    mutate(
      died = case_when(
        outcome == 1 ~ "died",
        outcome == 0 ~ "survived"
    )) %>% 
    ggplot() + 
    geom_histogram(aes_string(x = which_LDH, fill = "died"), binwidth=100, alpha=0.3) +
    scale_fill_manual(name='Outcome', labels=c("Died", "Survived"), values=c('blue', 'orange')) + 
    xlim(ldh_min, ldh_max) + 
    scale_y_continuous(name = "Number of patients") +
    ggtitle(title) +
    theme_bw() 
}

draw_histogram("Tongji_375_CN", "LDH_first")
draw_histogram("Outcomerea_FR", "LDH_first")
draw_histogram("Tongji_110_CN", "LDH_first")
draw_histogram("St_Antonius_NL", "LDH_first")
draw_histogram("Tongji_375_CN", "LDH_last")
draw_histogram("Outcomerea_FR", "LDH_last")
draw_histogram("Tongji_110_CN", "LDH_last")
draw_histogram("St_Antonius_NL", "LDH_last")

