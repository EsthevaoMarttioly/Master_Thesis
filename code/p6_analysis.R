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


# ---- Config ---------------------------------------------------------------
year = c(2025)
S    = c("F", "I", "U")    # N = Outside Labor Force
formal_idx = c("01", "03", "05", "07", "08")
inform_idx = c("02", "04", "06", "09", "10")
input_txt  = "data/pnad/input_PNADC_trimestral.txt"
wedges = seq(0, 7000, 200)


# ---------------------------------------------------------------------------
# 1. Quarterly Statistics
txts        = sprintf("data/pnad/raw/PNADC_0%d%d.txt", 1:4, rep(year, each = 4))
names(txts) = paste0(rep(year, each = 4), "Q", 1:4)
pnad = lapply(txts, function(f) pnadc_design(read_pnadc(microdata = f, input_txt = input_txt)))

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
  v_y = svyby(~VD4019,          ~status, occ, svyvar,  na.rm = TRUE)
  v_h = svyby(~VD4031,          ~status, occ, svyvar,  na.rm = TRUE)

  lf = sum(n[S])
  as.numeric(c(n[S] / lf, lf, sum(n), m$VD4031, m$VD4019, sqrt(coef(v_y)),
             m$VD4019[2] / m$VD4019[1], sqrt(coef(v_h))[2] / m$VD4031[1]))
}

info = c("F", "I", "U", "LF", "Pop", "h_F", "h_I",
         "y_F", "y_I", "sd_F", "sd_I", "xi", "h_I_std")
full_dataset = data.frame(Information = info, map(pnad, statistics), check.names=FALSE)



# ---------------------------------------------------------------------------
# 2. Annual Statistics, Visita 1
df = get_pnadc(year = max(year), interview = 1, deflator = FALSE, labels = FALSE,
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
dfw = subset(df, wage > 0 & wage <= cap)

ln_par = function(d) c(coef(svymean(~I(log(wage)), d, na.rm = TRUE))[[1]],   # MLE for lognorm
                       sqrt(coef(svyvar(~I(log(wage)), d, na.rm = TRUE))[[1]]))

dens = function(d, s) {
  p = ln_par(d)
  k = svysmooth(~wage, d, bandwidth = 200)$wage
  tibble(sector = s, wage = k$x, density = k$y, lognormal = dlnorm(k$x, p[1], p[2]))
}

wage_dist = bind_rows(dens(subset(dfw, status == "F"), "F"),
                      dens(subset(dfw, status == "I"), "I"))



# ---------------------------------------------------------------------------
# 3. Bolsa Familia (PNAD)
bf_size   = coef(svytotal(~bf_hh, df, na.rm = TRUE))
bf_sector = coef(svyby(~bf_hh, ~status, df, svymean, na.rm = TRUE))
bf_value  = coef(svymean(~V5002A2, subset(df, bf == 1), na.rm = TRUE))

subs = list(Total = df, Formal = subset(df, status=="F"), Informal = subset(df, status=="I"))

cov_dist = imap_dfr(subs, function(d, g) {
  s = svysmooth(bf_hh ~ wage, d, bandwidth = 200)[[1]]
  tibble(panel = g, wage = s$x, bf_hh = s$y)
})

cov_wage = imap_dfr(subs, ~as_tibble(svyby(~bf_hh, ~wbin, .x, svymean, na.rm = TRUE)) %>%
  mutate(panel = .y, wage = wedges)) %>% drop_na()
cov_wage$panel = factor(cov_wage$panel, levels = names(subs))
cov_dist$panel = factor(cov_dist$panel, levels = names(subs))

bf_wage_dist = bind_rows(dens(subset(df, bf_hh == 1), "Receives BF"),
                         dens(subset(df, bf_hh == 0), "No BF"))


# Household-level
df_hh      = subset(df, !duplicated(id_dom))
bf_size_hh = coef(svytotal(~bf_hh, df_hh, na.rm = TRUE))
iphh       = bf_size / bf_size_hh



# ---------------------------------------------------------------------------
# 4. Bolsa Familia (Vis Data)
df_vis = read.csv("data/visdata3-download.csv", fileEncoding = "latin1") %>%
  `colnames<-`(c("Year", "HH1", "HH", "Value1", "Value", "Avg1", "Avg")) %>%
  apply(2, function(x) as.numeric(gsub(",", ".", x))) %>% as.data.frame()

df_vis       = bf_vis[df_vis$Year == max(year), c("HH", "Value")]
bf_size_vis  = df_vis$HH
bf_value_vis = bf_vis$Value / df_vis$HH / 12



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

# Download and Pairwise
pids  = 10 * rep(year, each = 4) + 1:4
csvs = file.path(paste0("data/pnad/final/pnadc.microdados.painel.", pids, ".csv"))
csvs = csvs[file.exists(csvs)]
cols  = c(outer(c("vd4001", "vd4002", "vd4009", "v1028", "ano"), 1:5, paste, sep = "_"))

pairs = map_dfr(csvs, function(f) {
  d = read_csv(f, col_select = all_of(cols), show_col_types = FALSE)
  map_dfr(1:4, ~pair_of(d, .x)) %>% filter(!is.na(s0), !is.na(s1), !is.na(w), y0 == min(year))
})


# Transition Matrix
P = pairs %>% count(s0, s1, wt = w, name = "n") %>%
  complete(s0 = S, s1 = S, fill = list(n = 0)) %>%
  pivot_wider(names_from = s1, values_from = n) %>%
  column_to_rownames("s0") %>% as.matrix()
P = P[S, S]
P = P / rowSums(P)
P_ss = Re(eigen(t(P))$vectors[, 1])
P_ss = P_ss / sum(P_ss); names(P_ss) = S



# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
col_sector = c(F = "steelblue", I = "tomato")
col_bf     = c("Receives BF" = "forestgreen", "No BF" = "gray")

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
                strip.background = element_rect(fill = "grey95", colour = "black"),
                strip.text = element_text(colour = "black", size = 9))

save_fig = function(g, name) ggsave(paste0("output/figures/", name, ".png"),
                                    g, width = 8, height = 5, dpi = 150)


# ---- Graphics -------------------------------------------------------------
g = ggplot(wage_dist, aes(wage, density, colour = sector)) +
  geom_line(linewidth = 1.2) + mytheme +
  geom_line(aes(y = lognormal), linetype = "dashed", linewidth = 1.2) +
  geom_vline(xintercept = 1518, color = "black", linetype = "dashed", linewidth = 0.8) +
  scale_colour_manual(values = col_sector) +
  coord_cartesian(xlim = c(0, 7e3)) +
  labs(title = paste("PNAD", max(year), "- Wage Distribution vs Log-Normal"),
       x = "Monthly Wage (R$)", y = "Density", colour = "")
save_fig(g, "wage_distribution")


g = ggplot(bf_wage_dist, aes(wage, density, colour = sector, fill = sector)) +
  geom_area(alpha = 0.25, position = "identity") +
  geom_line(linewidth = 1) + mytheme +
  geom_vline(xintercept = 1518, color = "black", linetype = "dashed", linewidth = 0.8) +
  scale_colour_manual(values = col_bf) +
  scale_fill_manual(values = col_bf) +
  coord_cartesian(xlim = c(0, 7e3)) +
  labs(title = paste("PNAD", max(year), "- Wage Distribution by BF"),
       x = "Monthly wage (R$)", y = "Density", colour = "", fill = "")
save_fig(g, "bf_wage")


g = ggplot(cov_wage, aes(wage, bf_hh)) +
  geom_col(fill = col_bf[1], alpha = 0.5) +
  geom_line(data = cov_dist, colour = col_bf[1], linewidth = 1.2) + mytheme +
  geom_vline(xintercept = 1518, color = "black", linetype = "dashed", linewidth = 0.8) +
  facet_wrap(~panel) +
  coord_cartesian(xlim = c(0, 7e3)) +
  labs(title = paste("PNAD", max(year), "- % Receiving BF, by Wage and Status"),
       x = "Monthly Wage (R$)", y = "Share receiving BF")
save_fig(g, "bf_coverage_by_wage")


# ---- Tables ---------------------------------------------------------------
# Statistics
shares = tibble(state       = c("Formal", "Informal", "Unemployed"),
                share       = c(t(dataset[S])),
                mean_hours  = c(dataset$h_F, dataset$h_I, NA),
                mean_income = c(dataset$y_F, dataset$y_I, NA))
write.csv(shares, "data/pnad_formality_shares.csv", row.names = FALSE)
write.csv(wage_dist, "data/pnad_income_dist.csv", row.names = FALSE)
write.csv(rbind(P, P_ss), "data/pnad_transition_matrix.csv")


# Bolsa Familia
print(paste("BF Size (Individuals):", round(bf_size)))
print(bf_sector)
print(paste("Avg BF Value: R$", round(bf_value), "(PNAD), R$",
                                round(bf_value_vis), "(VIS)"))
print(paste("Tr/w =", round(bf_value_vis / dataset["y_F"], 3)))
print(coverage)
cat(sprintf("BF households: PNAD %.1fM vs Vis %.1fM  (undercount x%.2f)\n",
            bf_size_hh / 1e6, bf_size_vis / 1e6, bf_size_vis / bf_size_hh))
cat(sprintf("BF individuals:  PNAD %.1fM vs Vis-Implied %.1fM  (%.2f people/household)\n",
            bf_size / 1e6, bf_size_vis * iphh / 1e6, iphh))


# Transition
cat(max(year), "Quarter-to-Quarter Transition Matrix:\n"); print(round(P, 4))
cat("\nImplied Stationary Shares:\n"); print(round(P_ss, 4))


### F = 49.99%;  I = 44.16%;  U = 5.85%
### h_F = 42.5;  h_I = 36.8;  hours_std = 0.34
### w_F = 4183;  w_I = 2698;  xi = 0.64
### T = 711/hh;  T/w = 0.14;  BF = 0.358


