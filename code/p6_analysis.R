#=
# ---------------------------------------------------------------------------
# DESCRIPTION
# PNAD Continua statistics to calibrate the model, plus Bolsa Familia.
# If you are in RStudio, open the Rproject.
# ---------------------------------------------------------------------------
#=

# ---- Packages -------------------------------------------------------------
# renv::restore()     # run once to install the locked versions ('1' or 'Y')
library(tidyverse)
library(PNADcIBGE)
library(survey)
library(readxl)

year       = 2025
quarter    = FALSE     # TRUE to run quarter
transition = FALSE     # TRUE to run every panel


# ---- Config ---------------------------------------------------------------
S          = c("F", "I", "U")     # N = Outside Labor Force
wedges     = seq(0, 8000, 200)
formal_idx = c("01", "03", "05", "07", "08")
inform_idx = c("02", "04", "06", "09", "10")
input_txt  = "data/pnad/input_PNADC_trimestral.txt"

# Quarterly Deflator
deflator = list.files("data/pnad", "^deflator_PNADC.*\\.xls$", full.names = TRUE)[1] %>%
  read_excel() %>% mutate(m = as.integer(str_sub(trim, 1, 2))) %>% filter(m %% 3 == 1) %>%
  summarise(defl = mean(Habitual), .by = c(Ano, m)) %>%
  transmute(q = paste0(Ano, "Q", (m + 2) / 3), defl) %>% deframe()
def_base = names(which.min(deflator))

# Calibration Statistics
statistics = function(d) {
  d$variables$status = with(d$variables, case_when(VD4009 %in% formal_idx ~ "F",
                                                   VD4009 %in% inform_idx ~ "I",
                                                   VD4002 == "2"          ~ "U",
                                                   VD4001 == "2"          ~ "N",
                                                   TRUE                   ~ NA_character_))
  occ = subset(d, status %in% c("F", "I"))

  # Statistics by Formality
  n = coef(svytotal(~factor(status), d, na.rm = TRUE))
  names(n) = sub("factor\\(status\\)", "", names(n))
  m   = svyby(~VD4031 + VD4019, ~status, occ, svymean, na.rm = TRUE)
  v_h = svyby(~VD4031,          ~status, occ, svyvar,  na.rm = TRUE)

  as.numeric(c(n[S] / sum(n[S]), sum(n[S]), sum(n), m$VD4031, m$VD4019,
             m$VD4019[2] / m$VD4019[1], sqrt(coef(v_h))[2] / m$VD4031[1]))
}

info = c("F", "I", "U", "LF", "Pop", "h_F", "h_I", "y_F", "y_I", "xi", "h_I_sd")



# ---------------------------------------------------------------------------
# 1. Quarterly Statistics
if (quarter) {
  library(doParallel)       # Setup parallel back-end to use extra cores
  cl = makeCluster(4)
  registerDoParallel(cl)

  txts        = list.files("data/pnad/raw", "\\.txt$", full.names = TRUE)
  names(txts) = sub(".*_0(\\d)(\\d{4})\\.txt$", "\\2Q\\1", txts)   # -> "2012Q1"
  txts        = txts[order(names(txts))]

  q_stats = function(f, d) {
    gc(verbose = FALSE)
    s = statistics(pnadc_design(read_pnadc(microdata = f, input_txt = input_txt,
                                           c("VD4001", "VD4002", "VD4009", "VD4019", "VD4031"))))
    replace(s, info %in% c("y_F", "y_I"), s[info %in% c("y_F", "y_I")] * d)
  }

  cols = foreach(f = txts, d = deflator[names(txts)],
                 .packages = c("PNADcIBGE", "survey", "dplyr"),
                 .export = c("statistics", "info", "formal_idx", "inform_idx",
                             "S", "input_txt")) %dopar% q_stats(f, d)
  full_dataset = data.frame(Information = info, setNames(cols, names(txts)), check.names = FALSE)
  write.csv(full_dataset, "data/final/pnad_historical.csv", row.names = FALSE)
  stopCluster(cl)
}



# ---------------------------------------------------------------------------
# 2. Annual Statistics, Visita 1
df = get_pnadc(year = year, interview = 1, deflator = FALSE, labels = FALSE,
               reload = FALSE, savedir = "data/pnad/raw/annual/")

df = update(df, bf     = as.integer(as.character(V5002A) == "1"),
                wage   = ifelse(is.na(as.numeric(VD4019)), 0, as.numeric(VD4019)),
                wbin = cut(wage, c(wedges, Inf), include.lowest = TRUE),
                status = case_when(VD4009 %in% formal_idx ~ "F",
                                   VD4009 %in% inform_idx ~ "I",
                                   VD4002 == "2"          ~ "U",
                                   VD4001 == "2"          ~ "N", TRUE ~ NA_character_))

dataset = data.frame(t(statistics(df))) %>% `colnames<-`(info)

df$variables$id_dom = with(df$variables, paste0(UPA, V1008, V1014))
df$variables$bf_hh  = ave(df$variables$bf, df$variables$id_dom,
                          FUN = function(x) as.integer(any(x == 1, na.rm = TRUE)))


# Wage Distribution
cap = as.numeric(coef(svyquantile(~wage, subset(df, wage > 0), 0.99, na.rm = TRUE)))
dfp = subset(df, wage > 0)

ln_par = function(d) c(coef(svymean(~I(log(wage)), d, na.rm = TRUE))[[1]],   # MLE for lognorm
                       sqrt(coef(svyvar(~I(log(wage)), d, na.rm = TRUE))[[1]]))

dens = function(d, g, f = ~wage)
  with(svysmooth(f, d, bandwidth = 200, xlim = c(0, cap))[[1]],
       tibble(group = g, wage = x, y = y))

# Theta
lp_F = ln_par(subset(dfp, status == "F" & wage <= cap))    # c(mu, sigma) of log wage
lp_I = ln_par(subset(dfp, status == "I" & wage <= cap))
theta = c(mu_F = -lp_F[2]^2 / 2,                           # E[theta_F] = 1
          mu_I = -lp_I[2]^2 / 2 + (lp_I[1] - lp_F[1]),     # E[theta_I] = exp(d_mu)
          sd_F = lp_F[2], sd_I = lp_I[2],                  # sd of log wage
          d_mu = lp_I[1] - lp_F[1])
dataset = cbind(dataset, t(theta))

wage_dist = bind_rows(dens(subset(dfp, status == "F"), "F") %>%
                        mutate(lognormal = dlnorm(wage, lp_F[1], lp_F[2])),
                      dens(subset(dfp, status == "I"), "I") %>%
                        mutate(lognormal = dlnorm(wage, lp_I[1], lp_I[2])))



# ---------------------------------------------------------------------------
# 3. Bolsa Familia (PNAD)
subs = list(Total = df,
            Formal = subset(df, status=="F"),
            Informal = subset(df, status=="I"))

bf_size   = coef(svytotal(~bf_hh, df, na.rm = TRUE))
bf_sector = coef(svyby(~bf_hh, ~status, df, svymean, na.rm = TRUE))
bf_share  = coef(svymean(~bf_hh, subset(df, status %in% S), na.rm = TRUE))
bf_value  = coef(svymean(~V5002A2, subset(df, bf == 1), na.rm = TRUE))


# Wage Distribution
cov_dist = imap_dfr(subs, ~dens(.x, .y, bf_hh ~ wage))
cov_wage = imap_dfr(subs, ~as_tibble(svyby(~bf_hh, ~wbin, .x, svymean, na.rm = TRUE)) %>%
  transmute(group = .y, wage = wedges, y = bf_hh)) %>% drop_na()
cov_wage$group = factor(cov_wage$group, levels = names(subs))
cov_dist$group = factor(cov_dist$group, levels = names(subs))

bf_wage_dist = bind_rows(dens(subset(dfp, bf_hh == 1), "Receives BF"),
                         dens(subset(dfp, bf_hh == 0), "No BF"))


# Household-level
df_hh      = subset(df, !duplicated(id_dom))
bf_size_hh = coef(svytotal(~bf_hh, df_hh, na.rm = TRUE))
pphh       = bf_size / bf_size_hh



# ---------------------------------------------------------------------------
# 4. Bolsa Familia (Vis Data)
read_vis = function(file, cols) read.csv(file, fileEncoding = "latin1") %>%
  `colnames<-`(cols) %>% mutate(across(-1, ~ as.numeric(gsub(",", ".", .))))

fam = read_vis("data/visdata3-download.csv", c("Year", "HH1", "HH", "Val1", "Value", "Avg1", "Avg"))
ind = read_vis("data/visdata3-download_individual.csv", c("Year", "Ind1", "Ind"))
siz = read_vis("data/visdata3-download-hhsize.csv", c("Ref", 1:8)) %>%
  filter(str_sub(Ref, -4) == year) %>% summarise(across(-1, mean)) %>% unlist()

bf_size_vis  = fam$HH[fam$Year == year]              # BF Households
bf_ind_vis   = ind$Ind[ind$Year == year]             # BF Individuals
pphh_vis     = sum((1:8) * siz) / sum(siz)           # People per HH (8 = 8+)
bf_value_vis = fam$Value[fam$Year == year] / bf_size_vis / 12



# ---------------------------------------------------------------------------
# 5. Sector Transitions
classify = function(d, v) {
  vd4001 = d[[paste0("vd4001_", v)]]
  vd4002 = d[[paste0("vd4002_", v)]]
  vd4009 = str_pad(d[[paste0("vd4009_", v)]], 2, pad = "0")
  case_when(vd4009 %in% formal_idx ~ "F",
            vd4009 %in% inform_idx ~ "I",
            vd4002 == "2" ~ "U",
            vd4001 == "2" ~ "N", TRUE ~ NA_character_)
}

pair_of = function(d, v)      # Consecutive visits v -> v+1, using origin weight
  tibble(s0 = classify(d, v), s1 = classify(d, v+1),
         w = d[[paste0("v1028_", v)]], y0 = d[[paste0("ano_", v)]])

# Pairwise: the panels with a visit in "year", or all of them if transition
pids = if (transition) 2012:year else (year - 1):year
csvs = sprintf("data/pnad/final/pnadc.microdados.painel.%d.csv",
               as.vector(outer(pids, 1:4, function(y, q) 10 * y + q)))
csvs = csvs[file.exists(csvs)]
cols = c(outer(c("vd4001", "vd4002", "vd4009", "v1028", "ano"), 1:5, paste, sep = "_"))

counts = map_dfr(csvs, function(f) {
  d = read_csv(f, col_select = all_of(cols), show_col_types = FALSE)
  map_dfr(1:4, ~pair_of(d, .x)) %>%
    filter(s0 %in% S, s1 %in% S, !is.na(w)) %>%
    count(y0, s0, s1, wt = w, name = "n")
})

# Transition rates by origin year
trans = counts %>% group_by(y0, s0, s1) %>% summarise(n = sum(n), .groups = "drop") %>%
  complete(y0, s0 = S, s1 = S, fill = list(n = 0)) %>%
  group_by(y0, s0) %>% mutate(p = n / sum(n)) %>% ungroup()

if (transition)
  write.csv(trans, "data/final/pnad_transition_historical.csv", row.names = FALSE)

# Calibration Matrix
P = trans %>% filter(y0 == year) %>%
  pivot_wider(id_cols = s0, names_from = s1, values_from = p) %>%
  column_to_rownames("s0") %>% as.matrix()
P = P[S, S]
P_ss = Re(eigen(t(P))$vectors[, 1])
P_ss = P_ss / sum(P_ss); names(P_ss) = S



# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
xmax = max(wedges)        # plotting window for wage densities
pal  = c("#141827", "#62455b", "#736681", "#c1d9d0", "#fffae3")
col_sector = c(F = pal[1], I = pal[2])
col_state  = c(F = pal[1], I = pal[2], U = pal[4])
col_bf     = c("Receives BF" = pal[2], "No BF" = pal[3])

mytheme = theme(legend.position = "bottom",
                plot.title = element_text(size = 12, face = "bold"),
                plot.subtitle = element_text(size = 10),
                panel.background = element_rect(fill = "transparent", colour = "black",
                                                linewidth = 0.5, linetype = "solid"),
                panel.grid.major.y = element_line(colour = "grey", linewidth = 0.5),
                panel.grid.minor.y = element_line(colour = "grey", linewidth = 0.5),
                panel.grid = element_line(colour = "grey98"),
                panel.grid.major.x = element_line(colour = "transparent"),
                panel.grid.minor.x = element_line(colour = "transparent"),
                axis.text = element_text(colour = "black", size = 9),
                strip.background = element_rect(fill = pal[5], colour = "black"),
                strip.text = element_text(colour = "black", size = 9))

save_fig = function(g, name, h = 5) ggsave(paste0("output/figures/", name, ".png"),
                                           g, width = 8, height = h, dpi = 150)


# ---- Graphics -------------------------------------------------------------
g = ggplot(wage_dist, aes(wage, y, colour = group)) +
  geom_line(linewidth = 1.2) + mytheme +
  geom_line(aes(y = lognormal), linetype = "dashed", linewidth = 1.2) +
  geom_vline(xintercept = 1518, color = "black", linetype = "dashed", linewidth = 0.8) +
  scale_colour_manual(values = col_sector) +
  coord_cartesian(xlim = c(0, xmax)) +
  labs(title = paste("PNAD", year, "- Wage Distribution vs Log-Normal"),
       x = "Monthly Wage (R$)", y = "Density", colour = "")
save_fig(g, "wage_distribution")


g = ggplot(bf_wage_dist, aes(wage, y, colour = group, fill = group)) +
  geom_area(alpha = 0.25, position = "identity") +
  geom_line(linewidth = 1) + mytheme +
  geom_vline(xintercept = 1518, color = "black", linetype = "dashed", linewidth = 0.8) +
  scale_colour_manual(values = col_bf) +
  scale_fill_manual(values = col_bf) +
  coord_cartesian(xlim = c(0, xmax)) +
  labs(title = paste("PNAD", year, "- Wage Distribution by BF"),
       x = "Monthly wage (R$)", y = "Density", colour = "", fill = "")
save_fig(g, "bf_wage")


g = ggplot(cov_wage, aes(wage, y)) +
  geom_col(fill = pal[2], alpha = 0.5) +
  geom_line(data = cov_dist, colour = pal[1], linewidth = 1.2) + mytheme +
  geom_vline(xintercept = 1518, color = "black", linetype = "dashed", linewidth = 0.8) +
  facet_wrap(~group) +
  coord_cartesian(xlim = c(0, xmax)) +
  labs(title = paste("PNAD", year, "- Population receiving BF, by Wage and Status"),
       x = "Monthly Wage (R$)", y = "Share receiving BF")
save_fig(g, "bf_coverage_by_wage")


# ---- Time series ----------------------------------------------------------
th      = read.csv("data/final/pnad_transition_historical.csv")
history = read.csv("data/final/pnad_historical.csv", check.names = FALSE) %>%
  pivot_longer(-Information, names_to = "q", values_to = "value") %>%
  mutate(t = as.numeric(str_sub(q, 1, 4)) + (as.numeric(str_sub(q, 6)) - 1) / 4)
tbreaks = seq(floor(min(history$t)), ceiling(max(history$t)), by = 2)

# 100% stacked area over time
area_ts = function(d, breaks, title, x_lab, y_lab, fill_lab = "", labels = NULL) {
  g = ggplot(d, aes(x, y, fill = group)) +
    geom_area(position = "fill", colour = NA) + mytheme +
    scale_fill_manual(values = col_state) +
    scale_x_continuous(breaks = breaks, expand = c(0, 0)) +
    scale_y_continuous(labels = scales::percent, expand = c(0, 0)) +
    labs(title = title, x = x_lab, y = y_lab, fill = fill_lab)
  if (is.null(d$panel)) g else g + facet_wrap(~panel, labeller = as_labeller(labels))
}

save_fig(area_ts(transmute(th, x = y0, y = p, group = s1, panel = s0),
                 seq(min(th$y0), max(th$y0), by = 3),
                 "Sector Transition Rates over Time", "Origin Year",
                 "Destination Share", "To",
                 c(F = "From Formal", I = "From Informal", U = "From Unemployed")),
         "transition_timeseries")

save_fig(area_ts(history %>% filter(Information %in% S) %>%
                   transmute(x = t, y = value, group = Information),
                 tbreaks, "Sector Shares over Time (PNAD)",
                 "Year", "Share of the Labor Force"),
         "shares_timeseries")


# Calibrated Moments
cal_lab = c(h_F = "Formal, usual weekly hours",
            h_I = "Informal, usual weekly hours",
            y_F = paste0("Formal, monthly wage (R$ of ", def_base, ")"),
            xi  = "Wage Gap: Informal / Formal")

cal = history %>% filter(Information %in% names(cal_lab)) %>%
  mutate(Information = factor(Information, levels = names(cal_lab)),
         sector = case_when(Information %in% c("h_F", "y_F") ~ "F",
                            Information == "h_I" ~ "I", TRUE ~ "X"))
g = ggplot(cal, aes(t, value, colour = sector)) +
  geom_line(linewidth = 1) + mytheme +
  facet_wrap(~Information, ncol = 2, scales = "free_y",
             labeller = as_labeller(cal_lab)) +
  scale_colour_manual(values = c(col_sector, X = pal[3]), guide = "none") +
  scale_x_continuous(breaks = tbreaks) +
  labs(title = "Labor Moments over Time (PNAD)", x = "Year", y = NULL)
save_fig(g, "calibration_timeseries", h = 6)


# ---- Tables ---------------------------------------------------------------
# Statistics
write.csv(dataset, "data/final/pnad_calibration.csv", row.names = FALSE)
write.csv(wage_dist, "data/final/pnad_income_dist.csv", row.names = FALSE)
write.csv(rbind(P, P_ss), "data/final/pnad_transition_matrix.csv")


# Labor Market
print(data.frame(row.names   = c("Formal", "Informal", "Unemployed"),
                 share       = round(c(t(dataset[S])), 4),
                 hours       = round(c(dataset$h_F, dataset$h_I, NA), 1),
                 wage        = round(c(dataset$y_F, dataset$y_I, NA)),
                 `wage/y_F`  = round(c(1, dataset$xi, NA), 3),
                 `theta_mu`  = round(c(dataset$mu_F, dataset$mu_I, NA), 3),
                 `theta_sd`  = round(c(dataset$sd_F, dataset$sd_I, NA), 3),
                 check.names = FALSE))

# Bolsa Familia: Survey vs Administrative Record
print(data.frame(row.names = c("Households (M)", "Individuals (M)", "People/HH",
                               "Value (R$/month)"),
                 PNAD  = round(c(bf_size_hh / 1e6, bf_size / 1e6, pphh, bf_value), 2),
                 VisData = round(c(bf_size_vis / 1e6, bf_ind_vis / 1e6,
                                   pphh_vis, bf_value_vis), 2),
                 ratio = round(c(bf_size_vis / bf_size_hh, bf_ind_vis / bf_size,
                                 pphh_vis / pphh, bf_value_vis / bf_value), 2)))

print(data.frame(row.names = "coverage",
                 share_of_LF = round(bf_share[[1]], 4),
                 in_formal   = round(bf_sector[["F"]], 4),
                 in_informal = round(bf_sector[["I"]], 4),
                 `Tr/y_F`    = round(bf_value_vis / dataset$y_F, 3),
                 check.names = FALSE))

# Quarter-to-quarter Transitions
print(rbind(round(P, 4), `stationary` = round(P_ss, 4), `survey` = round(c(t(dataset[S])), 4)))

