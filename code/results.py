#=
#----------------------------------------------------------------------------
# DESCRIPTION
# Plotting and reporting functions for the main model.
# All functions are self-contained and imported in main.py.
#----------------------------------------------------------------------------
#=

# ---- Packages --------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# Reload results.py to update global namespace (for interactive use)
def rr():
    import importlib, code.results
    importlib.reload(code.results)
    globals().update({k: v for k, v in vars(code.results).items()
                      if not k.startswith('_')})


#----------------------------------------------------------------------------
# Consistent style
#----------------------------------------------------------------------------

COLORS = {
    'Tr'  : 'steelblue',     # Conditional Transfer (BF) shock
    'Z'   : 'forestgreen',   # TFP shock
    'r'   : 'darkorange',    # interest rate
    'F'   : 'steelblue',     # formal employment
    'I'   : 'tomato',          # informal employment
    'U'   : 'mediumpurple',  # unemployed
    'Y'   : 'steelblue',
    'C_GHH'   : 'darkorange',
}


LS = {0: '-', 1: '--', 2: ':', 3: '-.'}


VAR_LABELS = {
    'Y'    : 'Output $Y$',
    'C_GHH': 'Consumption $C_{GHH}$',
    'F'    : 'Formal share $\\alpha_F$',
    'I'    : 'Informal share $\\alpha_I$',
    'U'    : 'Unemployed $U$',
    'BF'   : 'BF recipients $BF$',
    'w'    : 'Real wage $w$',
    'w_I'  : 'Informal wage $w_I$',
    'pi'   : 'Inflation $\\pi$',
    'pi_w' : 'Wage inflation $\\pi^w$',
    'r'    : 'Real rate $r$',
    'Div'  : 'Dividends',
    'A'    : 'Assets $A$',
    'G'    : 'Gov. spending $G$',
}

plt.rcParams.update({
    'font.size'        : 10,
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'figure.dpi'       : 120,
})


def _save_or_show(fig, savepath):
    plt.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, dpi=150, bbox_inches='tight')
    plt.show()


#----------------------------------------------------------------------------
# 1. Steady-State Summary
# ---------------------------------------------------------------------------

def print_ss_summary(ss, var_ss = ['Y', 'C', 'beta_high', 'A', 'B']):
    """Print key steady-state moments and sanity checks."""
    print("\n" + "="*55)
    print("  STEADY STATE   (Model)        (Theory)")
    print("="*55)
    for k in var_ss:
        print(f"  {k:12s} = {ss[k]:.4f}")

    print(f"  {'beta_low':12s} = {ss['beta_high'] - ss['dbeta']:.4f}")
    print(f"  F + I + U  = 1.000,  model = {ss['F'] + ss['I'] + ss['U']:.3f}")
    print("="*55)


# ---------------------------------------------------------------------------
# 2. Consumption Policy Functions
# ---------------------------------------------------------------------------

def plot_consumption_policy(ss, calibration, T_plot_a=20, savepath=None):
    """Plot c(s, theta_median, e_median, a) for all three employment states."""
    a_grid = ss.internals['household']['a_grid']
    c_pol  = ss.internals['household']['c_ghh']

    nT, nE, nA = calibration['nT'], calibration['nE'], calibration['nA']
    e_med = nE // 2            # median productivity index
    t_med = nT // 2            # median productivity index

    # Reshape to (nS=3, nT=nT, nBeta=2, nE, nA)
    c_3d = c_pol.reshape(3, nT, 2, nE, nA)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    for bi, beta_name in enumerate(['Impatient', 'Patient']):
        ax = axes[bi]
        ax.plot(a_grid, c_3d[0, t_med, bi, e_med, :],
                color=COLORS['F'], label='Formal')
        ax.plot(a_grid, c_3d[1, t_med, bi, e_med, :],
                color=COLORS['I'], linestyle='--', label='Informal')
        ax.plot(a_grid, c_3d[2, t_med, bi, e_med, :],
                color=COLORS['U'],   linestyle=':',  label='Unemployed')
        ax.set_xlabel('Assets $a$')
        ax.set_ylabel('Consumption $c(s, \\bar{\\theta}, \\bar{e}, a)$')
        ax.set_title(f'Policy Functions - {beta_name}')
        ax.set_xlim(0, T_plot_a)
        ax.set_ylim(0, 4)
        ax.legend(frameon=False)

    _save_or_show(fig, savepath)
    return fig, ax


# ---------------------------------------------------------------------------
# 3. Wealth Distribution
# ---------------------------------------------------------------------------

def gini_coefficient(values, weights=None):
    # Values must be sorted in ascending order
    idx = np.argsort(values)
    values = values[idx]

    if weights is None:
        weights = np.ones_like(values)
    else:
        weights = weights[idx]

    pop  = np.concatenate([[0], np.cumsum(weights) / np.sum(weights)])
    wlth = np.concatenate([[0], np.cumsum(weights * values) / np.sum(weights * values)])

    # Calculate weighted Gini coefficient
    return 1.0 - np.sum((pop[1:] - pop[:-1]) * (wlth[1:] + wlth[:-1]))


def plot_wealth_distribution(ss, lorenz_data=None, n_bins=60, savepath=None):
    """Two-panel figure: wealth PDF near constraint + Lorenz curve."""
    D      = ss.internals['household']['D']
    a_grid = ss.internals['household']['a_grid']

    a_dist   = D.sum(axis=0)
    htm      = a_dist[0]
    cum_pop  = np.cumsum(a_dist)
    cum_wlth = np.cumsum(a_dist * a_grid) / np.sum(a_dist * a_grid)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: PDF near borrowing constraint
    widths = np.diff(np.append(a_grid[:n_bins], a_grid[n_bins]))
    axes[0].bar(a_grid[:n_bins], a_dist[:n_bins],
                width=widths, color='steelblue', alpha=0.7, align='edge')
    axes[0].axvline(a_grid[0], color='tomato', linestyle='--',
                    label=f'HtM = {htm:.1%}')
    axes[0].set_xlabel('Assets $a$')
    axes[0].set_ylabel('Mass of households')
    axes[0].set_title('Wealth Distribution (near borrowing constraint)')
    axes[0].legend(frameon=False)

    # Right: Lorenz curve   (values = wealth level, weights = population mass)
    axes[1].plot(cum_pop, cum_wlth, color='steelblue',
                 label='Model, Gini = {:.2f}'.format(gini_coefficient(a_grid, weights=a_dist)))
    axes[1].plot([0, 1], [0, 1], color='gray', linestyle=':', label='Perfect equality')
    if lorenz_data is not None:
        pctls = np.arange(501) / 500
        lorenz_emp = np.array([np.interp(p, lorenz_data[:, 0], lorenz_data[:, 1])
                               for p in pctls])
        axes[1].plot(pctls, lorenz_emp, color='forestgreen', linestyle='--',
                     label='SCF 2019, Gini = {:.2f}'.format(gini_coefficient(lorenz_emp)))
    axes[1].set_xlabel('Cumulative population share')
    axes[1].set_ylabel('Cumulative wealth share')
    axes[1].set_title('Lorenz Curve')
    axes[1].legend(frameon=False)

    _save_or_show(fig, savepath)
    return fig, axes


#----------------------------------------------------------------------------
# 4. PE Intertemporal MPC Profiles
# ---------------------------------------------------------------------------

def plot_impc_profiles(G_hh, T_plot=30, savepath=None):
    """Plot PE iMPC for Tr shock (BF conditional transfer).

    G_hh['C_GHH']['Tr'][:T, 0]: response of C_t to a unit Tr shock at t=0,
    holding all prices (r, w, Div) fixed.
    """
    iMPC_Tr = G_hh['C_GHH']['Tr']

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(iMPC_Tr[:T_plot, 0] * 100, marker='o', ms=4,
            color=COLORS['Tr'], linewidth=1.8,
            label='Conditional transfer $Tr$ (BF-targeted)')
    ax.axhline(0, color='gray', linestyle=':')
    ax.set_xlabel('Quarter $t$')
    ax.set_ylabel(r'$\partial C_t / \partial Tr_0 \;(\times 100)$')
    ax.set_title('PE Intertemporal MPC Profile — BF Transfer')
    ax.legend(frameon=False)

    _save_or_show(fig, savepath)
    return fig, ax


# ---------------------------------------------------------------------------
# 5. GE IRFs for a single shock
# ---------------------------------------------------------------------------

def plot_irfs(irf_dict, T_plot=30, title='', savepath=None):
    """Panel IRF plot for a single shock.

    irf_dict : {variable_name: array of length T}  (already in levels)
    """
    variables = list(irf_dict.keys())
    n    = len(variables)
    ncols = max(1, (n + 1) // 2)
    nrows = 2 if n > ncols else 1

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes_flat = np.array(axes).flatten()

    for ax, var in zip(axes_flat, variables):
        ax.plot(irf_dict[var][:T_plot] * 100, color='steelblue', linewidth=1.8)
        ax.axhline(0, color='gray', linestyle=':')
        ax.set_title(VAR_LABELS.get(var, var))
        ax.set_xlabel('Quarters')
        ax.set_xlim(0, T_plot)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(5))

    # Hide unused panels
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    axes_flat[0].set_ylabel('% deviation from SS')
    if title:
        fig.suptitle(title, fontsize=11)

    _save_or_show(fig, savepath)
    return fig, axes_flat


# ---------------------------------------------------------------------------
# 6. GE IRF comparison across fiscal instruments
# ---------------------------------------------------------------------------

def plot_irf_comparison(irfs_dict, variables, T_plot=30, savepath=None):
    """Overlay IRFs from multiple shocks on the same axes.

    irfs_dict : {label_str: irf_dict}
    variables : list of variable names (one panel each)
    """
    n   = len(variables)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    col_cycle = ['steelblue', 'tomato', 'forestgreen', 'darkorange']

    for ax, var in zip(axes, variables):
        for i, (label, irf) in enumerate(irfs_dict.items()):
            ax.plot(irf[var][:T_plot] * 100,
                    color=col_cycle[i % len(col_cycle)],
                    linestyle=LS[i % 4],
                    linewidth=1.8,
                    label=label)
        ax.axhline(0, color='gray', linestyle=':')
        ax.set_title(VAR_LABELS.get(var, var))
        ax.set_xlabel('Quarters')
        ax.set_xlim(0, T_plot)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(5))

    axes[0].set_ylabel('% deviation from SS')
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle('GE IRFs: Fiscal Instrument Comparison', fontsize=11)

    _save_or_show(fig, savepath)
    return fig, axes


# ---------------------------------------------------------------------------
# 7. Fiscal multipliers summary table
# ---------------------------------------------------------------------------

def compute_multipliers(G, shocks, horizons=(0, 3, 7, 19)):
    # Compute output and consumption multipliers at given horizons.
    mults = {}
    for shock, dz in shocks.items():
        dY = G['Y'][shock] @ dz
        dC = G['C_GHH'][shock] @ dz
        sz = dz[0]    # impact size
        mults[shock] = {h: (dY[h] / sz, dC[h] / sz) for h in horizons}
    return mults


def print_multipliers(mults, horizons=(0, 3, 7, 19)):
    H_LABELS = {0: 'Impact', 3: '1 year', 7: '2 years', 19: '5 years'}
    print("\n" + "=" * 60)
    print("  FISCAL MULTIPLIERS   dY/d(shock_0)")
    print("=" * 60)
    header = f"  {'Shock':12s}" + "".join(f"  {H_LABELS.get(h, f't={h}'):>9s}" for h in horizons)
    print(header)
    for shock, hmap in mults.items():
        row = f"  {shock:12s}" + "".join(f"  {hmap[h][0]:>9.3f}" for h in horizons)
        print(row)
    print("=" * 60)


# ---------------------------------------------------------------------------
# 8. Bolsa Família: research-question results
# ---------------------------------------------------------------------------

def _ss_stats(ss):
    """Scalar summary of one steady state (composition + insurance margins)."""
    D = ss.internals['household']['D']
    c = ss.internals['household']['c_ghh']
    V = ss.internals['household']['V']
    a = ss.internals['household']['a_grid']
    a_dist = D.sum(0)
    shF, shI, shU = ss['F'], ss['I'], ss['U']
    return {
        'Formal share'      : shF,
        'Informal share'    : shI,
        'Unemployed share'  : shU,
        'Informality (emp)' : shI / (shF + shI),
        'HtM (a=a_min)'     : a_dist[0],
        'Wealth Gini'       : gini_coefficient(a, weights=a_dist),
        'Consumption Gini'  : gini_coefficient(c.ravel(), weights=D.ravel()),
        'Aggregate C'       : ss['C'],
        'Aggregate Y'       : ss['Y'],
        'BF spending'       : ss['Tr'] * ss['BF'],
        'Welfare E[V]'      : float(np.sum(D * V)),
    }


def compare_bf_steady_states(ss_bf, ss_nobf):
    """Table: economy WITH BF vs the Tr=0 counterfactual.
    Composition channel (informality up) + insurance channel (inequality down)."""
    s1, s0 = _ss_stats(ss_bf), _ss_stats(ss_nobf)
    print("\n" + "=" * 58)
    print(f"  {'':22s}{'with BF':>12s}{'no BF':>12s}{'delta':>10s}")
    print("=" * 58)
    for k in s1:
        print(f"  {k:22s}{s1[k]:>12.4f}{s0[k]:>12.4f}{s1[k]-s0[k]:>+10.4f}")
    print("=" * 58)
    return s1, s0


def consumption_by_state(ss, savepath=None):
    """Mean consumption in F / I / U and the drop on job loss.
    F->U drop is the insurance metric: smaller = BF cushions more."""
    D = ss.internals['household']['D']
    c = ss.internals['household']['c_ghh']
    block = c.shape[0] // 3
    seg = {'Formal': (0, block), 'Informal': (block, 2*block),
           'Unemployed': (2*block, 3*block)}
    means = {n: (D[lo:hi] * c[lo:hi]).sum() / D[lo:hi].sum()
             for n, (lo, hi) in seg.items()}
    drop_FU = 1 - means['Unemployed'] / means['Formal']
    drop_FI = 1 - means['Informal']  / means['Formal']

    print("\n  Mean consumption by state")
    for n, m in means.items():
        print(f"    {n:12s} = {m:.4f}")
    print(f"    F->U drop    = {drop_FU:6.1%}")
    print(f"    F->I drop    = {drop_FI:6.1%}")

    if savepath is not None:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(list(means.keys()), list(means.values()),
               color=[COLORS['F'], COLORS['I'], COLORS['U']])
        ax.set_ylabel('Mean consumption $c$')
        ax.set_title('Consumption by employment state')
        _save_or_show(fig, savepath)
    return means, drop_FU


def plot_bf_sweep(solve_fn, calibration, Tr_grid, savepath=None):
    """Re-solve the steady state across BF generosities Tr and plot the
    composition (informality, unemployment) and insurance (Gini, welfare) margins.
    `solve_fn(calib_dict) -> ss` wraps your GE solver, e.g.
        solve_fn = lambda cal: solve_ss(hank_ss, cal, unknowns, targets)[0]
    """
    keys = ['Informality (emp)', 'Unemployed share', 'Wealth Gini', 'Welfare E[V]']
    series = {k: [] for k in keys}
    for Tr in Tr_grid:
        st = _ss_stats(solve_fn({**calibration, 'Tr': Tr}))
        for k in keys:
            series[k].append(st[k])

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, k in zip(axes.flat, keys):
        ax.plot(Tr_grid, series[k], marker='o', ms=4, color='steelblue', lw=1.8)
        ax.set_xlabel('BF generosity $Tr$')
        ax.set_title(k)
    fig.suptitle('Steady state vs BF generosity', fontsize=11)
    _save_or_show(fig, savepath)
    return series


def formality_by_wealth(ss, n_q=5, savepath=None):
    """Formal share (among employed) and unemployment rate by wealth quantile.
    Shows formality as the fallback of the asset-poor."""
    D = ss.internals['household']['D']
    block = D.shape[0] // 3
    a_dist = D.sum(0)
    cum_left = np.concatenate([[0.0], np.cumsum(a_dist)[:-1]])   # mass strictly below
    qidx = np.minimum((cum_left * n_q).astype(int), n_q - 1)     # gridpoint -> quantile

    formal, inform, unemp = np.full(n_q, np.nan), np.full(n_q, np.nan), np.full(n_q, np.nan)
    for q in range(n_q):
        sel = (qidx == q)
        mF, mI, mU = (D[:block, sel].sum(),
                      D[block:2*block, sel].sum(),
                      D[2*block:, sel].sum())
        if mF + mI + mU > 0:       formal[q] = mF / (mF + mI + mU)
        if mF + mI + mU > 0:       inform[q] = mI / (mF + mI + mU)
        if mF + mI + mU > 0:       unemp[q]  = mU / (mF + mI + mU)

    x = np.arange(n_q)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - 0.2, formal, width=0.2, color=COLORS['F'], label='Formal')
    ax.bar(x + 0.0, inform, width=0.2, color=COLORS['I'], label='Informal')
    ax.bar(x + 0.2, unemp,  width=0.2, color=COLORS['U'], label='Unemployed')
    ax.set_xticks(x); ax.set_xticklabels([f'Q{i+1}' for i in range(n_q)])
    ax.set_xlabel('Wealth quantile (poor -> rich)')
    ax.set_ylabel('Share')
    ax.set_title('Formality and unemployment by wealth')
    ax.legend(frameon=False)
    _save_or_show(fig, savepath)
    return formal, inform, unemp


def welfare_by_type(ss_bf, ss_nobf, calibration, savepath=None):
    """Welfare gain from BF (E[V] with - without) by patience type and by sector.
    Who gains: the impatient / constrained; who pays: via tau_l."""
    nT, nE, nBeta = calibration['nT'], calibration['nE'], 2

    def ev_by(ss):
        D = ss.internals['household']['D']; V = ss.internals['household']['V']
        nA = D.shape[1]
        Dr = D.reshape(3, nT, nBeta, nE, nA); Vr = V.reshape(3, nT, nBeta, nE, nA)
        out = {}
        for b, name in [(0, 'Impatient'), (1, 'Patient')]:
            out[name] = (Dr[:, :, b] * Vr[:, :, b]).sum() / Dr[:, :, b].sum()
        for s, name in [(0, 'Formal'), (1, 'Informal'), (2, 'Unemployed')]:
            out[name] = (Dr[s] * Vr[s]).sum() / Dr[s].sum()
        return out

    e1, e0 = ev_by(ss_bf), ev_by(ss_nobf)
    gains = {k: e1[k] - e0[k] for k in e1}

    print("\n  Welfare gain from BF  (E[V] with - without)")
    for k, g in gains.items():
        print(f"    {k:12s} = {g:>+8.4f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ks = list(gains.keys())
    ax.bar(ks, [gains[k] for k in ks],
           color=['gray', 'gray', COLORS['F'], COLORS['I'], COLORS['U']])
    ax.axhline(0, color='k', linewidth=0.8)
    ax.set_ylabel('Welfare gain $\\Delta E[V]$')
    ax.set_title('Who gains from Bolsa Familia')
    _save_or_show(fig, savepath)
    return gains


# ---------------------------------------------------------------------------
# 9. LaTeX table: with-BF vs no-BF steady state
# ---------------------------------------------------------------------------

_TEX_LABELS = {
    'Formal share'     : r'Formal share $\alpha_F$',
    'Informal share'   : r'Informal share $\alpha_I$',
    'Unemployed share' : r'Unemployment $u$',
    'Informality (emp)': r'Informality (of employed)',
    'HtM (a=a_min)'    : r'Hand-to-mouth',
    'Wealth Gini'      : r'Wealth Gini',
    'Consumption Gini' : r'Consumption Gini',
    'Aggregate C'      : r'Aggregate consumption $C$',
    'Aggregate Y'      : r'Formal output $Y$',
    'BF spending'      : r'BF spending',
    'Welfare E[V]'     : r'Welfare $\mathbb{E}[V]$',
}
_PCT_ROWS = {'Formal share', 'Informal share', 'Unemployed share',
             'Informality (emp)', 'HtM (a=a_min)'}


def save_ss_table(ss_bf, ss_nobf, savepath='output/tables/ss_table.tex', label='tab:bf_ss'):
    """Write the with-BF vs no-BF steady-state comparison as a booktabs LaTeX table.
    Percentage rows in %, gaps in points (pp); levels/welfare in 3 decimals."""
    s1, s0 = _ss_stats(ss_bf), _ss_stats(ss_nobf)

    def cell(k, x, delta=False):
        if k in _PCT_ROWS:
            return f'{x*100:+.1f}' if delta else rf'{x*100:.1f}\%'
        return f'{x:+.3f}' if delta else f'{x:.3f}'

    rows = "\n".join(
        rf'{_TEX_LABELS.get(k, k)} & {cell(k, s1[k])} & {cell(k, s0[k])} '
        rf'& {cell(k, s1[k]-s0[k], delta=True)} \\'
        for k in s1)

    tex = (
        "\\begin{table}[htbp]\\centering\n"
        "\\caption{Steady state: Bolsa Fam\\'ilia vs.\\ No-Transfer Counterfactual}\n"
        f"\\label{{{label}}}\n"
        "\\begin{tabular}{lccc}\n\\toprule\n"
        " & With BF & No BF & $\\Delta$ \\\\\n\\midrule\n"
        f"{rows}\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    with open(savepath, 'w', encoding='utf-8') as fobj:
        fobj.write(tex)
    print(f"LaTeX table written to {savepath}")
    return tex