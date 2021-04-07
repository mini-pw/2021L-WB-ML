# Requires running process_raw.R

source("libraries.R")

draw_freqpoly <- function(filename, which){
  df = data.frame(read.csv(str_c("data/", filename, ".csv")))
  if(str_starts(which, "LDH")) {
    x_min = -50
    x_max = 1000
  } else if (str_starts(which, "hsCRP")) {
    x_min = -50
    x_max = 500
  } else {
    x_min = -10
    x_max= 110
  }
  bins = 20
  binsize = (x_max-x_min)/bins
  title = sprintf('Plot of %s in %s', which, filename)
  
  df %>% 
    mutate(
      died = case_when(
        outcome == 1 ~ "died",
        outcome == 0 ~ "survived"
    )) %>% 
    ggplot() + 
    geom_freqpoly(aes_string(x = which, color = "died"), binwidth=binsize, size=2, alpha=0.4) +
    scale_color_manual(name='Outcome', labels=c("Died", "Survived"), values=c('orange', 'blue')) + 
    xlim(x_min, x_max) + 
    scale_y_continuous(name = "Number of patients") +
    ggtitle(title) +
    theme_bw()
}

draw_freqpoly("Tongji_375_CN", "LDH_first")
draw_freqpoly("Outcomerea_FR", "LDH_first")
draw_freqpoly("Tongji_110_CN", "LDH_first")
draw_freqpoly("St_Antonius_NL", "LDH_first")

(draw_freqpoly("Tongji_375_CN", "LDH_last") + draw_freqpoly("Outcomerea_FR", "LDH_last")) /
(draw_freqpoly("Tongji_110_CN", "LDH_last") + draw_freqpoly("St_Antonius_NL", "LDH_last"))

(draw_freqpoly("Tongji_375_CN", "hsCRP_last") + draw_freqpoly("Outcomerea_FR", "hsCRP_last")) /
(draw_freqpoly("Tongji_110_CN", "hsCRP_last") + draw_freqpoly("St_Antonius_NL", "hsCRP_last"))

(draw_freqpoly("Tongji_375_CN", "lymphocytes_last") + draw_freqpoly("Outcomerea_FR", "lymphocytes_last") ) /
(draw_freqpoly("Tongji_110_CN", "lymphocytes_last") + draw_freqpoly("St_Antonius_NL", "lymphocytes_last"))

