#=
# ---------------------------------------------------------------------------
# DESCRIPTION
# Get data from PNAD Contínua in order to calibrate the model
# Open the Rproject, instead of the script itself
# ---------------------------------------------------------------------------
#=
# Write this in the terminal this to install packages
# renv::restore()     # Answer '1'

rm(list = ls())       # Clear the environment


# Import Packages
library(tidyverse)
library(PNADcIBGE)
library(survey)
library(readxl)



# Prepare Data Frame
dataset = data.frame(matrix(c('F', 'I', 'U', 'Occup', 'Pop', 'h_F', 'h_I',
                              'y_F', 'y_I', 'xi', 'h_I_std', 'resid_by_hh')))



# ---------------------------------------------------------------------------
# Set the period
j = 2025     # year

for (i in 1:4) {   # quarter
  
  
  # Install for the first time
  # df = get_pnadc(year=j, quarter=i, savedir='data/pnad/', deflator=F, reload=F)


  # Read the copied file
  df = pnadc_design(read_pnadc(microdata = paste0('data/pnad/PNADC_0', i, j, '.txt'),
                               input_txt = 'data/pnad/input_PNADC_trimestral.txt'))
  
  
  
  # ---------------------------------------------------------------------------
  # 1. Occupied Population
  formal_idx = c('01', '03', '05', '07', '08')
  inform_idx = c('02', '04', '06', '09', '10')
  
  total_pop    = svytotal(~VD4001, df, na.rm=T) %>% as.matrix() %>% as.numeric()
  occupied_pop = svytotal(~VD4002, df, na.rm=T) %>% as.matrix() %>% as.numeric()
  formal_ocup  = svytotal(~VD4009, df, na.rm=T) %>% as.matrix() %>% as.numeric()
  
  
  formal = sum(formal_ocup[as.numeric(formal_idx)]) / sum(occupied_pop)
  inform = sum(formal_ocup[as.numeric(inform_idx)]) / sum(occupied_pop)
  
  unemployment = occupied_pop[2] / sum(occupied_pop)
  
  
  
  # ---------------------------------------------------------------------------
  # 2. Worked Hours and Income by Formality
  formal_hours = svymean(~VD4031, subset(df, VD4009 %in% formal_idx), na.rm=T)[1]
  inform_hours = svymean(~VD4031, subset(df, VD4009 %in% inform_idx), na.rm=T)[1]
  
  formal_income = svymean(~VD4019, subset(df, VD4009 %in% formal_idx), na.rm=T)[1]
  inform_income = svymean(~VD4019, subset(df, VD4009 %in% inform_idx), na.rm=T)[1]
  
  xi = inform_income / formal_income
  
  
  
  # ---------------------------------------------------------------------------
  # 3. Informal Hours Variance
  hours_std  = sqrt(svyvar(~VD4031, subset(df, VD4009 %in% inform_idx), na.rm=T)) / formal_hours
  
  hours_dist = svyquantile(~VD4031, subset(df, VD4009 %in% inform_idx),
                           quantiles = seq(0, 100, 100/16)/100, ci=F, na.rm=T)[1][1]
  
  
  
  # ---------------------------------------------------------------------------
  # 4. Bolsa Família
  resid_by_hh = svymean(~V2001, df, na.rm=T) %>% as.matrix() %>% as.numeric()
  
  
  
  # ---------------------------------------------------------------------------
  # 5. Combine Information
  dataset = cbind.data.frame(dataset, c(formal, inform, unemployment, sum(occupied_pop),
                                        sum(total_pop), formal_hours, inform_hours,
                                        formal_income, inform_income, xi,
                                        hours_std, resid_by_hh))

}

colnames(dataset) = c('Information', 'Q1', 'Q2', 'Q3', 'Q4')



# ---------------------------------------------------------------------------
# Bolsa Família
df_bf = read.csv('data/visdata3-download.csv', fileEncoding = "latin1") %>%
  `colnames<-`(c('Year', 'HH1', 'HH', 'Value1', 'Value', 'Avg1', 'Avg')) %>%
  apply(2, function(x) as.numeric(gsub(',', '.', x))) %>% as.data.frame()

bf_values = df_bf[df_bf$Year == 2025, c('HH', 'Value')]



# ---------------------------------------------------------------------------
# Results
df_final = cbind(dataset[1], apply(dataset[2:5], 1, mean))


avg_bf_value  = bf_values$Value / bf_values$HH / 12   # R$ 711 / household
avg_bf_person = avg_bf_value / df_final[df_final$Information == "resid_by_hh", 2]

print(paste('T/w =', round(avg_bf_value /
                             df_final[df_final$Information == "y_F", 2], 3)))


print(paste('BF =', round(bf_values$HH *
              df_final[df_final$Information == "resid_by_hh", 2] /
              df_final[df_final$Information == "Pop", 2], 3)))



### Results:  F = 49.99%;      I = 44.16%;        U = 5.85%
### Results:  h_F = 42.5 hrs;  h_I = 36.8 hrs;    hours_std = 0.34
### Results:  w_F = R$ 4,183;  w_I = R$ 2,698;    xi = 0.64
### Results:  T = R$ 711/hh;   T/w = 0.14;        BF = 0.358




