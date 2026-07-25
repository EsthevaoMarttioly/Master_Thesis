#=
# ---------------------------------------------------------------------------
# DESCRIPTION
# Get data from PNAD Continua in order to calibrate the model
# If using RStudio, open the Rproject, instead of the script itself
# ---------------------------------------------------------------------------
#=

# ---- Packages --------------------------------------------------------------
# renv::restore()     # run once to install the locked versions ('1' or 'Y')
library(tidyverse)
library(PNADcIBGE)
library(survey)
library(readxl)
library(ggplot2)


# ---------------------------------------------------------------------------
# Read every PNAD file
pnad_dir  = "data/pnad/raw"
input_txt = "data/pnad/input_PNADC_trimestral.txt"
sel_years = 2025

pnad_files = list.files(pnad_dir, pattern = "^PNADC_\\d{6}\\.txt$", full.names = TRUE)
codes = str_extract(basename(pnad_files), "\\d{6}")
names(pnad_files) = paste0(substr(codes, 3, 6), "Q", substr(codes, 2, 2))
pnad_files = pnad_files[as.integer(substr(codes, 3, 6)) %in% sel_years]

pnad = lapply(pnad_files, function(f) pnadc_design(read_pnadc(microdata = f, input_txt = input_txt)))


dataset = data.frame(matrix(c("F", "I", "U", "Occup", "Pop", "h_F", "h_I",
                              "y_F", "y_I", "xi", "h_I_std", "resid_by_hh")))


# Survey's Statistics
j = 2025   # year of interest

for (q in paste0(j, "Q", 1:4)) {
  df = pnad[[q]]

  # 1. Occupied Population
  formal_idx = c("01", "03", "05", "07", "08")
  inform_idx = c("02", "04", "06", "09", "10")
  total_pop    = svytotal(~VD4001, df, na.rm = TRUE) %>% as.matrix() %>% as.numeric()
  occupied_pop = svytotal(~VD4002, df, na.rm = TRUE) %>% as.matrix() %>% as.numeric()
  formal_ocup  = svytotal(~VD4009, df, na.rm = TRUE) %>% as.matrix() %>% as.numeric()
  formal = sum(formal_ocup[as.numeric(formal_idx)]) / sum(occupied_pop)
  inform = sum(formal_ocup[as.numeric(inform_idx)]) / sum(occupied_pop)
  unemployment = occupied_pop[2] / sum(occupied_pop)

  # 2. Worked Hours and Income by Formality
  formal_hours  = svymean(~VD4031, subset(df, VD4009 %in% formal_idx), na.rm = TRUE)[1]
  inform_hours  = svymean(~VD4031, subset(df, VD4009 %in% inform_idx), na.rm = TRUE)[1]
  formal_income = svymean(~VD4019, subset(df, VD4009 %in% formal_idx), na.rm = TRUE)[1]
  inform_income = svymean(~VD4019, subset(df, VD4009 %in% inform_idx), na.rm = TRUE)[1]
  xi = inform_income / formal_income

  # 3. Informal Hours Variance
  hours_std  = sqrt(svyvar(~VD4031, subset(df, VD4009 %in% inform_idx), na.rm = TRUE)) / formal_hours
  hours_dist = svyquantile(~VD4031, subset(df, VD4009 %in% inform_idx),
                           quantiles = seq(0, 100, 100 / 16) / 100, ci = FALSE, na.rm = TRUE)[1][1]

  # 4. Bolsa Familia
  resid_by_hh = svymean(~V2001, df, na.rm = TRUE) %>% as.matrix() %>% as.numeric()

  # 5. Combine
  dataset = cbind.data.frame(dataset, c(formal, inform, unemployment, sum(occupied_pop),
                                        sum(total_pop), formal_hours, inform_hours,
                                        formal_income, inform_income, xi,
                                        hours_std, resid_by_hh))
}

colnames(dataset) = c("Information", "Q1", "Q2", "Q3", "Q4")


# ---------------------------------------------------------------------------
# Bolsa Familia
df_bf = read.csv("data/visdata3-download.csv", fileEncoding = "latin1") %>%
  `colnames<-`(c("Year", "HH1", "HH", "Value1", "Value", "Avg1", "Avg")) %>%
  apply(2, function(x) as.numeric(gsub(",", ".", x))) %>% as.data.frame()

bf_values = df_bf[df_bf$Year == 2025, c("HH", "Value")]


# ---------------------------------------------------------------------------
# Results
df_final = cbind(dataset[1], apply(dataset[2:5], 1, mean))
avg_bf_value  = bf_values$Value / bf_values$HH / 12
avg_bf_person = avg_bf_value / df_final[df_final$Information == "resid_by_hh", 2]

print(paste("T/w =", round(avg_bf_value / df_final[df_final$Information == "y_F", 2], 3)))
print(paste("BF =", round(bf_values$HH * df_final[df_final$Information == "resid_by_hh", 2] /
                            df_final[df_final$Information == "Pop", 2], 3)))

### F = 49.99%;  I = 44.16%;  U = 5.85%
### h_F = 42.5;  h_I = 36.8;  hours_std = 0.34
### w_F = 4183;  w_I = 2698;  xi = 0.64
### T = 711/hh;  T/w = 0.14;  BF = 0.358


