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

set.seed(20260415)

year       = 2025
quarter    = FALSE     # TRUE to run quarter
transition = FALSE     # TRUE to run every panel


# ---- Config ---------------------------------------------------------------
mw         = 1518                 # Minimum Wage
bf_line    = c(218, mw / 2, mw)   # Poverty Line and Regra de Protecao
wedges     = seq(0, 8000, 200)
S          = c("F", "I", "U")     # N = Outside Labor Force
formal_idx = c("01", "03", "05", "07", "08")
inform_idx = c("02", "04", "06", "09", "10")
input_txt  = "data/pnad/input_PNADC_trimestral.txt"

# Quarterly Deflator
deflator = list.files("data/pnad", "^deflator_PNADC.*\\.xls$", full.names = TRUE)[1] %>%
  read_excel() %>% mutate(m = as.integer(str_sub(trim, 1, 2))) %>% filter(m %% 3 == 1) %>%
  summarise(defl = mean(Habitual), .by = c(Ano, m)) %>%
  transmute(q = paste0(Ano, "Q", (m + 2) / 3), defl) %>% deframe()
def_base = names(which.min(deflator))

vs    = function(x)    unname(c(coef(x)[1], SE(x)[1]))    # Estimate and its Standard Error
vs_by = function(x, s) unname(c(coef(x)[[s]], SE(x)[x$status == s]))

# Calibration Statistics
statistics = function(d, se = FALSE) {
  z = function(x) as.numeric(ifelse(is.na(x), 0, x))
  d = update(d, status = case_when(VD4009 %in% formal_idx ~ "F",
                                   VD4009 %in% inform_idx ~ "I",
                                   VD4002 == "2"          ~ "U",
                                   VD4001 == "2"          ~ "N",
                                   TRUE                   ~ NA_character_))
  d = update(d, nF = z(status == "F"), nI = z(status == "I"),
                nU = z(status == "U"), nN = z(status == "N"))
  d = update(d, hF = z(VD4031) * nF,  hI = z(VD4031) * nI,  hI2 = z(VD4031)^2 * nI,
                yF = z(VD4019) * nF,  yI = z(VD4019) * nI,
                mhF = z(!is.na(VD4031)) * nF, mhI = z(!is.na(VD4031)) * nI,
                myF = z(!is.na(VD4019)) * nF, myI = z(!is.na(VD4019)) * nI)

  T = svytotal(~nF + nI + nU + nN + hF + hI + hI2 + yF + yI +
                mhF + mhI + myF + myI, d)
  m = lapply(list(F       = quote(nF / (nF + nI + nU)),
                  I       = quote(nI / (nF + nI + nU)),
                  U       = quote(nU / (nF + nI + nU)),
                  LF      = quote(nF + nI + nU),
                  Pop     = quote(nF + nI + nU + nN),
                  h_F     = quote(hF / mhF),
                  h_I     = quote(hI / mhI),
                  y_F     = quote(yF / myF),
                  y_I     = quote(yI / myI),
                  xi      = quote((yI / myI) / (yF / myF)),
                  h_ratio = quote((hI / mhI) / (hF / mhF)),
                  h_I_sd  = quote(sqrt(hI2 / mhI - (hI / mhI)^2) / (hF / mhF))),
             function(e) svycontrast(T, e))    # Delta Method

  v = setNames(sapply(m, coef), names(m))
  if (!se) return(v)
  rbind(est = v, se = setNames(sapply(m, SE), names(m)))
}

info = c("F", "I", "U", "LF", "Pop", "h_F", "h_I",
         "y_F", "y_I", "xi", "h_ratio", "h_I_sd")



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

dataset = data.frame(statistics(df, se = TRUE))

df$variables$id_dom = with(df$variables, paste0(UPA, V1008, V1014))
df$variables$bf_hh  = ave(df$variables$bf, df$variables$id_dom,
                          FUN = function(x) as.integer(any(x == 1, na.rm = TRUE)))


# Wage Distribution
cap = as.numeric(coef(svyquantile(~wage, subset(df, wage > 0), 0.99, na.rm = TRUE)))
dfp = subset(df, wage > 0)

dens = function(d, g, f = ~wage, bw = 200)
  with(svysmooth(f, d, bandwidth = bw, xlim = c(0, cap))[[1]],
       tibble(group = g, wage = x, y = y))


# Theta: Quantiles of log Wage (Net from E[y_F])
qs = c(.10, .25, .50, .75, .90)
lq = function(s) {
  x = svyquantile(~I(log(wage)), subset(dfp, status == s), qs, na.rm = TRUE)
  rbind(as.numeric(coef(x)) - log(dataset["est", "y_F"]), as.numeric(SE(x)))
}
qmat = cbind(lq("F"), lq("I"))     # Shift does not move the SE
colnames(qmat) = paste0("q", 100 * qs, "_", rep(S[1:2], each = length(qs)))
dataset = cbind(dataset, qmat)

wage_dist = bind_rows(dens(subset(dfp, status == "F"), "F"),
                      dens(subset(dfp, status == "I"), "I"))



# ---------------------------------------------------------------------------
# 3. Bolsa Familia (PNAD)
subs = list(Total = df,
            Formal = subset(df, status == "F"),
            Informal = subset(df, status == "I"))

bf_size     = coef(svytotal(~bf_hh, df, na.rm = TRUE))
bf_value    = coef(svymean(~V5002A2, subset(df, bf == 1), na.rm = TRUE))
bf_sector   = svyby(~bf_hh, ~status, df, svymean, na.rm = TRUE)
bf_share_lf = svymean(~bf_hh, subset(df, status %in% S), na.rm = TRUE)
bf_value_lf = svymean(~V5002A2, subset(df, bf == 1 & status %in% S), na.rm = TRUE)
dataset     = cbind(dataset,
                    BF   = vs(bf_share_lf),
                    BF_I = vs_by(bf_sector, "I"),
                    BF_U = vs_by(bf_sector, "U"),
                    Tr_y = vs(bf_value_lf) / dataset["est", "y_F"])


# Wage Distribution
cov_dist = imap_dfr(subs, ~dens(.x, .y, bf_hh ~ wage))
cov_wage = imap_dfr(subs, ~as_tibble(svyby(~bf_hh, ~wbin, .x, svymean, na.rm = TRUE)) %>%
  transmute(group = .y, wage = wedges, y = bf_hh)) %>% drop_na()
cov_wage$group = factor(cov_wage$group, levels = names(subs))
cov_dist$group = factor(cov_dist$group, levels = names(subs))

bf_wage_dist = bind_rows(dens(subset(dfp, bf_hh == 1), "Receives BF"),
                         dens(subset(dfp, bf_hh == 0), "No BF"))


# Household Income per Capita
df$variables$hh_pc = ave(df$variables$wage, df$variables$id_dom, FUN = sum) /
                     as.numeric(df$variables$V2001)

wage_pc_dist = bind_rows(dens(subset(df, bf_hh == 1 & status == "F"), "F", ~hh_pc, 40),
                         dens(subset(df, bf_hh == 1 & status == "I"), "I", ~hh_pc, 40))


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
# 5. Matched Panel: Transitions and Wage Persistence
## The panel follows a household over 5 quarterly visits
classify = function(d, v) {
  vd4001 = d[[paste0("vd4001_", v)]]
  vd4002 = d[[paste0("vd4002_", v)]]
  vd4009 = str_pad(d[[paste0("vd4009_", v)]], 2, pad = "0")
  case_when(vd4009 %in% formal_idx ~ "F",
            vd4009 %in% inform_idx ~ "I",
            vd4002 == "2" ~ "U",
            vd4001 == "2" ~ "N", TRUE ~ NA_character_)
}

wage_of = function(d, v) suppressWarnings(as.numeric(d[[paste0("vd4019_", v)]]))

pair_of = function(d, v, h = 1)     # Visits v -> v+h, using the origin weight
  tibble(s0 = classify(d, v),     s1 = classify(d, v + h),   # sector
         x0 = wage_of(d, v),      x1 = wage_of(d, v + h),    # wage
         w  = d[[paste0("v1028_", v)]], y0 = d[[paste0("ano_", v)]],
         id = d$upa, hh = d$estrato) %>%          # PSU and stratum
  filter(s0 %in% S, s1 %in% S, !is.na(w))


# Stratified Cluster Bootstrap
B    = 200
mult = function(hh, id) {
  u = !duplicated(id);  s = hh[u]
  M = matrix(1, length(s), B + 1)
  for (k in split(seq_along(s), s)) {
    n = length(k)
    if (n > 1) M[k, -1] = 1 + sqrt(n / (n - 1)) * (rmultinom(B, n, rep(1, n)) - 1)
  }
  M[match(id, id[u]), ]
}


# Weighted (co)moments of log Wage for Stayers
lw_mom = function(p, h) {
  p = filter(p, s0 == s1, x0 > 0, x1 > 0)
  M = mult(p$hh, p$id)
  map_dfr(S[1:2], function(s) {
    k = p$s0 == s;  w = p$w[k];  a = log(p$x0[k]);  b = log(p$x1[k])
    Z = cbind(n = w, sa = w * a, sb = w * b,
              saa = w * a^2, sbb = w * b^2, sab = w * a * b)
    as_tibble(crossprod(M[k, , drop = FALSE], Z)) %>%
      mutate(h = h, s = s, r = row_number() - 1L, .before = 1)
  })
}


# Attrition Tilt
tilt = function(P, alpha, tol = 1e-14, maxit = 500) {
  a = alpha / sum(alpha);  cj = rep(1, length(a))
  for (it in 1:maxit) {
    cn = a / as.vector((a / as.vector(P %*% cj)) %*% P)
    cn = cn / cn[1]
    if (max(abs(cn - cj)) < tol) { cj = cn; break }
    cj = cn
  }
  P = sweep(P, 2, cj, "*")
  structure(P / rowSums(P), c = cj, it = it)
}

long_P = function(P, y)             # matrix -> (y0, s0, s1, p), P is column-major
  tibble(y0 = y, s0 = rep(S, times = length(S)),
         s1 = rep(S, each = length(S)), p = as.vector(P))

wide_P = function(d) d %>% pivot_wider(id_cols = s0, names_from = s1, values_from = p) %>%
  column_to_rownames("s0") %>% as.matrix() %>% .[S, S]


# Pairwise the Panels
pids = if (transition) 2012:year else (year - 1):year
csvs = sprintf("data/pnad/final/pnadc.microdados.painel.%d.csv",
               as.vector(outer(pids, 1:4, function(y, q) 10 * y + q)))
csvs = csvs[file.exists(csvs)]
cols = c("upa", "estrato",
         outer(c("vd4001", "vd4002", "vd4009", "vd4019", "v1028", "ano"),
               1:5, paste, sep = "_"))

panel = map(csvs, function(f) {
  d  = read_csv(f, col_select = all_of(cols), show_col_types = FALSE)
  p1 = map_dfr(1:4, ~pair_of(d, .x))          # one quarter apart
  p4 = pair_of(d, 1, 4)                       # one year apart
  list(flow = count(p1, y0, s0, s1, wt = w, name = "n"),
       wage = bind_rows(lw_mom(p1, 1), lw_mom(p4, 4)))
})


# Transition rates by origin year
trans = map_dfr(panel, "flow") %>%
  group_by(y0, s0, s1) %>% summarise(n = sum(n), .groups = "drop") %>%
  complete(y0, s0 = S, s1 = S, fill = list(n = 0)) %>%
  group_by(y0, s0) %>% mutate(p = n / sum(n)) %>% ungroup()


# Wage Persistence: Corr(log w_t, log w_{t+h}) among same-sector stayers.
persist = map_dfr(panel, "wage") %>% group_by(r, h, s) %>%
  summarise(across(n:sab, sum), .groups = "drop") %>%
  transmute(r, h, s, rho = (sab / n - sa * sb / n^2) /
                     sqrt((saa / n - (sa / n)^2) * (sbb / n - (sb / n)^2)))

ac = function(h, s = "F") {
  x = persist$rho[persist$h == h & persist$s == s]
  c(x[1], sd(x[-1]))
}
dataset = cbind(dataset, ac1 = ac(1), ac4 = ac(4))


# Calibration Matrix (tilt the flows with annual stocks)
alpha = c(t(dataset["est", S]));  alpha = alpha / sum(alpha)
P_raw = wide_P(filter(trans, y0 == year))
P     = tilt(P_raw, alpha)
cat(sprintf("Attrition tilt: c = [%s], %d it, max|dP| = %.4f\n",
            paste(round(attr(P, "c"), 3), collapse = ", "),
            attr(P, "it"), max(abs(P - P_raw))))

# Historical Series
if (transition) {
  a_year = read.csv("data/final/pnad_historical.csv", check.names = FALSE) %>%
    filter(Information %in% S) %>%
    pivot_longer(-Information, names_to = "q", values_to = "v") %>%
    mutate(y0 = as.integer(str_sub(q, 1, 4))) %>%
    summarise(v = mean(v), .by = c(y0, Information)) %>%
    pivot_wider(names_from = Information, values_from = v)

  trans = map_dfr(sort(intersect(trans$y0, a_year$y0)), function(y)
    long_P(tilt(wide_P(filter(trans, y0 == y)),
                unlist(a_year[a_year$y0 == y, S])), y))
  write.csv(trans, "data/final/pnad_transition_historical.csv", row.names = FALSE)
}

P_ss = Re(eigen(t(P))$vectors[, 1])
P_ss = P_ss / sum(P_ss); names(P_ss) = S



# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
xmax = max(wedges)        # plotting window for wage densities
pal  = c('#1b325f', '#9cc4e4', '#e9f2f9', '#3a89c9', '#f26c4f', '#a8a3af')
col_sector = c(F = pal[1], I = pal[4])
col_state  = c(F = pal[1], I = pal[4], U = pal[2])
col_bf     = c("Receives BF" = pal[5], "No BF" = pal[1])
col_cal    = c(h_F = pal[1], h_I = pal[4], y_F = pal[1], xi = pal[5])
cal_lab    = c(h_F = "Formal, usual weekly hours",
               h_I = "Informal, usual weekly hours",
               y_F = paste0("Formal, monthly wage (R$ of ", def_base, ")"),
               xi  = "Wage Gap: Informal / Formal")


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
                strip.background = element_rect(fill = pal[6], colour = "black"),
                strip.text = element_text(colour = "black", size = 9))

save_fig = function(g, name, h = 5) ggsave(paste0("output/figures/", name, ".png"),
                                           g, width = 8, height = h, dpi = 150)

# Table for Latex.
esc = function(x) gsub("([$_%&#])", "\\\\\\1", x)
save_tex = function(d, name, caption, label, rows = esc(rownames(d))) {
  rule = "%---------------------------------------------------"
  body = paste(apply(cbind(rows, format(d, trim = TRUE)), 1,
                     paste, collapse = " & "), "\\\\")
  writeLines(c(rule, "\\begin{table}[htbp]", "\\centering",
               paste0("\\caption{", caption, "}"), paste0("\\label{", label, "}"), "\\small",
               paste0("\\begin{tabular}{l", strrep("c", ncol(d)), "}"), "\\toprule",
               paste(paste(c("", esc(colnames(d))), collapse = " & "), "\\\\"), "\\midrule",
               body, "\\bottomrule", "\\end{tabular}", "\\end{table}", rule),
             paste0("output/tables/", name, ".tex"))
}


# ---- Graphics -------------------------------------------------------------
g = ggplot(wage_dist, aes(wage, 100 * y, colour = group)) +
  geom_line(linewidth = 1.2) + mytheme +
  geom_vline(xintercept = mw, color = "black", linetype = "dashed", linewidth = 0.8) +
  scale_colour_manual(values = col_sector) +
  coord_cartesian(xlim = c(0, xmax)) +
  labs(title = paste("PNAD", year, "- Wage Distribution by Sector"),
       x = "Monthly Wage (R$)", y = "Density (%)", colour = "")
save_fig(g, "wage_distribution")


g = ggplot(bf_wage_dist, aes(wage, 100 * y, colour = group, fill = group)) +
  geom_area(alpha = 0.25, position = "identity") +
  geom_line(linewidth = 1) + mytheme +
  geom_vline(xintercept = mw, color = "black", linetype = "dashed", linewidth = 0.8) +
  scale_colour_manual(values = col_bf) +
  scale_fill_manual(values = col_bf) +
  coord_cartesian(xlim = c(0, xmax)) +
  labs(title = paste("PNAD", year, "- Wage Distribution by BF"),
       x = "Monthly Wage (R$)", y = "Density (%)", colour = "", fill = "")
save_fig(g, "bf_wage")


g = ggplot(wage_pc_dist, aes(wage, 100 * y, colour = group)) +
  geom_line(linewidth = 1.2) + mytheme +
  geom_vline(xintercept = bf_line, color = "black", linetype = "dashed", linewidth = 0.8) +
  annotate("text", x = bf_line, y = 0, hjust = -0.1, vjust = -0.5, size = 3,
           label = c(paste0("R$ ", bf_line[1]), paste0("R$ ", bf_line[2]), paste0("R$ ", bf_line[3]))) +
  scale_colour_manual(values = col_sector) +
  coord_cartesian(xlim = c(0, 3000)) +
  labs(title = paste("PNAD", year, "- Household Labor Income per Capita, by Status (Only Beneficiaries)"),
       x = "Monthly Household Income per Capita (R$)", y = "Density (%)", colour = "")
save_fig(g, "bf_eligibility")


g = ggplot(cov_wage, aes(wage, 100 * y)) +
  geom_col(fill = pal[2], alpha = 0.5) +
  geom_line(data = cov_dist, colour = pal[1], linewidth = 1.2) + mytheme +
  geom_vline(xintercept = mw, color = "black", linetype = "dashed", linewidth = 0.8) +
  facet_wrap(~group) +
  coord_cartesian(xlim = c(0, xmax)) +
  labs(title = paste("PNAD", year, "- Population receiving BF, by Wage and Status"),
       x = "Monthly Wage (R$)", y = "Share receiving BF (%)")
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
  if (!"panel" %in% names(d)) g else g + facet_wrap(~panel, labeller = as_labeller(labels))
}

save_fig(area_ts(transmute(th, x = y0, y = p, group = s1, panel = s0),
                 seq(min(th$y0), max(th$y0), by = 3),
                 "Sector Transition Rates over Time", "Origin Year",
                 "Destination Share", "To",
                 c(F = "From Formal", I = "From Informal", U = "From Unemployed")),
         "transition_timeseries")

save_fig(area_ts(history %>% filter(Information %in% S) %>%
                   transmute(x = t, y = value, group = Information),
                 tbreaks, "Sector Shares over Time (PNAD)", "Year", "Share of Labor Force"),
         "shares_timeseries")

cal = history %>% filter(Information %in% names(cal_lab)) %>%
  mutate(Information = factor(Information, levels = names(cal_lab)))

g = ggplot(cal, aes(t, value, colour = Information)) +
  geom_line(linewidth = 1) + mytheme +
  facet_wrap(~Information, ncol = 2, scales = "free_y",
             labeller = as_labeller(cal_lab)) +
  scale_colour_manual(values = col_cal, guide = "none") +
  scale_x_continuous(breaks = tbreaks) +
  labs(title = "Labor Moments over Time (PNAD)", x = "Year", y = NULL)
save_fig(g, "calibration_timeseries", h = 6)


# ---- Tables ---------------------------------------------------------------
# Statistics
write.csv(dataset, "data/final/pnad_calibration.csv", row.names = FALSE)
write.csv(wage_dist, "data/final/pnad_income_dist.csv", row.names = FALSE)
write.csv(rbind(P, P_ss), "data/final/pnad_transition_matrix.csv")


# Labor Market
est = function(...) c(unlist(dataset["est", c(...)]), NA)
print(data.frame(row.names   = c("Formal", "Informal", "Unemployed"),
                 Share       = round(c(t(dataset["est", S])), 4),
                 Hours       = round(est("h_F", "h_I"), 1),
                 Wage        = round(est("y_F", "y_I")),
                 `Wage/y_F`  = round(c(1, dataset["est", "xi"], NA), 3),
                 `log q50`   = round(est("q50_F", "q50_I"), 3),
                 `log IQR`   = round(est("q75_F", "q75_I") - est("q25_F", "q25_I"), 3),
                 `rho 1q`    = round(c(ac(1, "F")[1], ac(1, "I")[1], NA), 3),
                 `rho 1y`    = round(c(ac(4, "F")[1], ac(4, "I")[1], NA), 3),
                 check.names = FALSE))


# Bolsa Familia: Survey vs Administrative Record
bf_size_tab = data.frame(
  row.names = c("Households (M)", "Individuals (M)", "People/HH", "Value (R$/month)"),
  PNAD    = round(c(bf_size_hh / 1e6, bf_size / 1e6, pphh, bf_value), 2),
  VisData = round(c(bf_size_vis / 1e6, bf_ind_vis / 1e6, pphh_vis, bf_value_vis), 2),
  Ratio   = round(c(bf_size_vis / bf_size_hh, bf_ind_vis / bf_size,
                    pphh_vis / pphh, bf_value_vis / bf_value), 2))
print(bf_size_tab)
save_tex(bf_size_tab, "bf_size",
         paste0("Bolsa Fam\\'ilia ", year, ": Statistics"), "tab:bf_size")

bf_cover_tab = data.frame(
  row.names = c("BF / Labor Force", "BF in Formal", "BF in Informal",
                "BF in Unemployed", "BF in Non-Participating", "Tr/y_F"),
  Coverage  = round(c(coef(bf_share_lf), coef(bf_sector)[c("F", "I", "U", "N")],
                      coef(bf_value_lf) / dataset["est", "y_F"]), 3))
print(bf_cover_tab)
save_tex(bf_cover_tab, "bf_coverage",
         paste0("Bolsa Fam\\'ilia ", year, ": Coverage"), "tab:bf_coverage",
         rows = replace(esc(rownames(bf_cover_tab)),
                        rownames(bf_cover_tab) == "Tr/y_F", "$T / \\E(y^F)$"))


# Quarter-to-quarter Transitions
print(rbind(round(P, 4), `Stationary` = round(P_ss, 4), `Survey` = round(alpha, 4)))



