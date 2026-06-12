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



# ---------------------------------------------------------------------------
# 1. Occupied Population
formal_idx = c("01", "03", "05", "07", "08")
inform_idx = c("02", "04", "06", "09", "10")

occupied_pop = svytotal(~VD4002, df, na.rm = T) %>% as.matrix() %>% as.numeric()
formal_ocup  = svytotal(~VD4009, df, na.rm = T) %>% as.matrix() %>% as.numeric()


formal = sum(formal_ocup[as.numeric(formal_idx)]) / sum(occupied_pop)
inform = sum(formal_ocup[as.numeric(inform_idx)]) / sum(occupied_pop)

unemployment = occupied_pop[2] / sum(occupied_pop)

### Results:  F = 50.28%;  I = 44.65%;  U = 5.07%



# ---------------------------------------------------------------------------
# 2. Worked Hours and Income by Formality
formal_hours = svymean(~VD4031, subset(df, VD4009 %in% formal_idx), na.rm = T)[1]
inform_hours = svymean(~VD4031, subset(df, VD4009 %in% inform_idx), na.rm = T)[1]

formal_income = svymean(~VD4019, subset(df, VD4009 %in% formal_idx), na.rm = T)[1]
inform_income = svymean(~VD4019, subset(df, VD4009 %in% inform_idx), na.rm = T)[1]

xi = inform_income / formal_income    # xi = 0.66

### Results:  h_F = 42.5 hrs;  h_I = 36.8 hrs;  w_F = R$ 4,294;  w_I = R$ 2,826



# ---------------------------------------------------------------------------
# 3. Informal Hours Variance
hours_std  = sqrt(svyvar(~VD4031, subset(df, VD4009 %in% inform_idx), na.rm = T)) / formal_hours

hours_dist = svyquantile(~VD4031, subset(df, VD4009 %in% inform_idx),
                         quantiles = seq(0, 100, 100/16)/100, ci = F, na.rm = T)[1][1]

### Results: hours_std = 0.33



# ---------------------------------------------------------------------------
# 4. Bolsa Familia

svymean(~V2001, df, na.rm = T) %>% as.matrix() %>% as.numeric()

18.9e6 * 3.3227 / 213.5e6

600 / 4294




