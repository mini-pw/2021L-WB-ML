# Modified script for creating Fig2-like plots from other data we've collected
# Barish, M., Bolourani, S., Lau, L.F. et al. 
# External validation demonstrates limited clinical utility of the interpretable mortality prediction model for patients with COVID-19. 
# Nat Mach Intell 3, 25–27 (2021). https://doi.org/10.1038/s42256-020-00254-2

# Requires running process_raw.R
source("libraries.R")

recreate_ppv_hist <- function(dataset_name, scale_factor = 100) {
  df = data.frame(read.csv(paste0("data/", dataset_name,".csv")))

  df['Died'] = df['outcome'] == 1
  
  ldh_min = 150
  ldh_max = 1000
  
  threshold_data = array(numeric(), c(0, 2))
  for (ldh_thresh in seq(ldh_min, ldh_max)){
    df_c = df
    df_c$Pred = df_c$LDH_last >= ldh_thresh
    df_pred_pos = df_c[df_c['Pred'] == TRUE, ]
    
    num_tps = nrow(df_pred_pos[df_pred_pos$Died == TRUE, ])
    num_fps = nrow(df_pred_pos[df_pred_pos$Died == FALSE, ])
    ppv = num_tps / (num_tps + num_fps)
    threshold_data = rbind(threshold_data, c(ldh_thresh, ppv))
  }
  
  threshold_data = data.frame(threshold_data)
  colnames(threshold_data) = c('LDH_last', 'PPV')
  
  hist_scale_factor = scale_factor
  
  ggplot() + 
    geom_line(data=threshold_data, aes(x=LDH_last, y=PPV, color='Precision/PPV'), size=1.5) +
    scale_color_manual(values=c('black')) +
    geom_histogram(data=df, aes(x=LDH_last, fill=Died, ..count../hist_scale_factor), binwidth=30, alpha=0.3) +
    scale_fill_manual(name='Outcome', labels=c('Survived', 'Died'), values=c('blue', 'orange')) +
    scale_y_continuous(limits=c(0, 1), sec.axis = sec_axis(~ . *hist_scale_factor, name='Number of Patients')) +
    xlim(ldh_min, ldh_max) +
    geom_vline(xintercept = 365, linetype='longdash') +
    ggtitle(paste0('Histograms and Precision for Predicting Death - ', dataset_name)) +
    xlab('Lactate Dehydrogenase (U/L)t') +
    ylab('Precision/PPV for Death') +
    theme_bw() +
    theme(legend.justification=c(1,1), legend.title=element_blank(), legend.position=c(0.95,0.95), legend.spacing.y = unit(-0.15, "cm"))

}
# recreate_ppv_hist("Tongji_375_CN") + recreate_ppv_hist("St_Antonius_NL")

recreate_ppv_hist("Northwell_US")
