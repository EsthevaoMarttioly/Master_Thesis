#=
# ---------------------------------------------------------------------------
# DESCRIPTION
# Get data from PNAD Contínua in order to calibrate the model
# Open the Rproject, instead of the script itself
# ---------------------------------------------------------------------------
#=
# Write this in the terminal this to install packages
# renv::restore()     # Answer "1"

rm(list = ls())       # Clear the environment


# Import Packages
library(tidyverse)
library(PNADcIBGE)
library(survey)


# Set the period
i = 4        # quarter
j = 2025     # year


# Install for the first time
# df = get_pnadc(year = j, quarter = i, savedir = "data/")


# Read the copied file
df = pnadc_design(read_pnadc(microdata = paste0("data/PNADC_0", i, j, ".txt"),
                             input_txt = "data/input_PNADC_trimestral.txt"))


# Survey's Tables

occupied_pop = svytotal(~VD4002, df, na.rm = T) %>% as.matrix() %>% as.numeric()
formal_ocup  = svytotal(~VD4009, df, na.rm = T) %>% as.matrix() %>% as.numeric()

formal_hours = svyby(~VD4031, by = ~VD4009, df, svymean, na.rm = T)[2] %>% as.matrix()
formal_income = svyby(~VD4019, by = ~VD4009, df, svymean, na.rm = T)[2] %>% as.matrix()



# Descriptive Statistics

## Average Function
avg_by_ocup = function(df_ocup, df_avg, col_idx) {
  # Calculate the average of df_avg, weighted by the occupancy level.
  tot_ocup  = sum(df_ocup[col_idx])
  
  total_avg = sum(df_ocup[col_idx] * df_avg[col_idx] / tot_ocup)
  
  return(total_avg)
}


## Formal, Informal, and Unemployed Shares
formal_idx   = c(1,3,5,7,8)
informal_idx = c(2,4,6,9,10)

formal   = sum(formal_ocup[formal_idx]) / sum(occupied_pop)    # F = 50.28%
informal = sum(formal_ocup[informal_idx]) / sum(occupied_pop)  # I = 44.65%

unemployment = occupied_pop[2] / sum(occupied_pop)             # U = 5.07%


## Formal and Informal Worked Hours
formal_h = avg_by_ocup(formal_ocup, formal_hours, formal_idx)         # 42.5 hrs
informal_h = avg_by_ocup(formal_ocup, formal_hours, informal_idx)     # 36.8 hrs

formal_inc = avg_by_ocup(formal_ocup, formal_income, formal_idx)      # R$ 4,294
informal_inc = avg_by_ocup(formal_ocup, formal_income, informal_idx)  # R$ 2,793

xi = informal_inc / formal_inc    # xi = 0.65




