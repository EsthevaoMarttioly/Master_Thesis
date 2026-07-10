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

LS = {0: '-', 1: '--', 2: ':', 3: '-.'}

COLORS = {
    'Tr'  : 'steelblue',     # Conditional Transfer (BF) shock
    'Z'   : 'forestgreen',   # TFP shock
    'r'   : 'darkorange',    # interest rate
    'F'   : 'steelblue',     # formal employment
    'I'   : 'tomato',        # informal employment
    'U'   : 'mediumpurple',  # unemployed
    'BF'  : 'forestgreen',   # beneficiaries
    'Y'   : 'steelblue',
    'C_GHH'   : 'darkorange',
}

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

_TEX_LABELS = {
    'Formal share'     : r'Formal share $\alpha_F$',
    'Informal share'   : r'Informal share $\alpha_I$',
    'Unemployed share' : r'Unemployment $U$',
    'Informality (emp)': r'Informality (of employed)',
    'HtM (a=a_min)'    : r'Hand-to-Mouth',
    'Wealth Gini'      : r'Wealth Gini',
    'Consumption Gini' : r'Consumption Gini',
    'Aggregate C'      : r'Aggregate Consumption $C$',
    'Aggregate Y'      : r'Formal Output $Y$',
    'BF spending'      : r'BF spending',
    'Welfare E[V]'     : r'Welfare $\mathbb{E}[V]$',
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
    # Print key Steady-State moments and sanity checks.
    print("\n" + "="*55)
    print("  STEADY STATE   (Model)")
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
    # Plot c(s, theta_median, e_median, a) for all three employment states.
    a_grid = ss.internals['household']['a_grid']
    c_pol  = ss.internals['household']['c_ghh']

    nT, nE, nA = calibration['nT'], calibration['nE'], calibration['nA']
    e_med = nE // 2; t_med = nT // 2            # median productivity index

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
    # Two-panel Figure: Wealth PDF near constraint + Lorenz curve.
    D      = ss.internals['household']['D']
    a_grid = ss.internals['household']['a_grid']

    a_dist   = D.sum(axis=0)
    htm      = a_dist[0]
    cum_pop  = np.cumsum(a_dist)
    cum_wlth = np.cumsum(a_dist * a_grid) / np.sum(a_dist * a_grid)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: PDF near Borrowing Constraint
    widths = np.diff(np.append(a_grid[:n_bins], a_grid[n_bins]))
    axes[0].bar(a_grid[:n_bins], a_dist[:n_bins],
                width=widths, color='steelblue', alpha=0.7, align='edge')
    axes[0].axvline(a_grid[0], color='tomato', linestyle='--',
                    label=f'HtM = {htm:.1%}')
    axes[0].set_xlabel('Assets $a$')
    axes[0].set_ylabel('Mass of households')
    axes[0].set_title('Wealth Distribution (near borrowing constraint)')
    axes[0].legend(frameon=False)

    # Right: Lorenz Curve
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



# ---------------------------------------------------------------------------
# 4. BF Steady State Comparison
# ---------------------------------------------------------------------------

def _ss_stats(ss):
    # Scalar Summary of one Steady State.
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
        'HtM (a=a_min)'     : a_dist[0],
        'Wealth Gini'       : gini_coefficient(a, weights=a_dist),
        'Consumption Gini'  : gini_coefficient(c.ravel(), weights=D.ravel()),
        'Aggregate C'       : ss['C'],
        'BF spending'       : ss['Tr'] * ss['BF'],
        'Welfare E[V]'      : float(np.sum(D * V))}


def compare_bf_ss(ss_bf, ss_nobf, savepath=None, label='tab:bf_ss'):
    # Table: Economy WITH BF vs the Tr=0 Counterfactual.
    s1, s0 = _ss_stats(ss_bf), _ss_stats(ss_nobf)
    pct_rows = {'Formal share', 'Informal share', 'Unemployed share', 'HtM (a=a_min)'}

    def cell(k, x, delta=False):
        if k in pct_rows:
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

    if savepath is not None:
        with open(savepath, 'w', encoding='utf-8') as fobj:
            fobj.write(tex)
        print(f"LaTeX Table written to {savepath}")
    else:
        print("\n" + "=" * 58)
        print(f"  {'':22s}{'with BF':>12s}{'no BF':>12s}{'delta':>10s}")
        print("=" * 58)
        for k in s1:
            print(f"  {k:22s}{s1[k]:>12.4f}{s0[k]:>12.4f}{s1[k]-s0[k]:>+10.4f}")
        print("=" * 58)


# ---------------------------------------------------------------------------
# 5. Descriptive Analysis
# ---------------------------------------------------------------------------

def consumption_by_state(ss, savepath=None):
    # Mean Consumption in F / I / U.
    D = ss.internals['household']['D']
    c = ss.internals['household']['c_ghh']
    block = c.shape[0] // 3
    seg = {'Formal': (0, block), 'Informal': (block, 2*block),
           'Unemployed': (2*block, 3*block)}
    means = {n: (D[lo:hi] * c[lo:hi]).sum() / D[lo:hi].sum()
             for n, (lo, hi) in seg.items()}

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(list(means.keys()), list(means.values()),
            color=[COLORS['F'], COLORS['I'], COLORS['U']])
    ax.set_ylabel('Mean Consumption $c$')
    ax.set_title('Consumption by Employment State')
    _save_or_show(fig, savepath)
    return fig, ax


def formality_by_wealth(ss, n_q=5, savepath=None):
    # Formal share and unemployment rate by wealth quantile.
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
    return fig, ax


def welfare_by_type(ss_bf, ss_nobf, calibration, savepath=None):
    """Welfare Gain from BF (E[V] with - without) by patience type and by sector.
        Who gains: the impatient / constrained;   who pays: via tau_l."""
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
        print(f"    {k:18s} = {g:>+8.4f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ks = list(gains.keys())
    ax.bar(ks, [gains[k] for k in ks],
           color=['gray', 'gray', COLORS['F'], COLORS['I'], COLORS['U']])
    ax.axhline(0, color='k', linewidth=0.8)
    ax.set_ylabel('Welfare Gain $\\Delta E[V]$')
    ax.set_title('Who gains from Bolsa Familia')
    _save_or_show(fig, savepath)
    return fig, ax


#----------------------------------------------------------------------------
# 6. PE iMPC
# ---------------------------------------------------------------------------

def plot_impc(G_hh, T_plot=30, savepath=None):
    """Plot Partial Equilibrium iMPC for BF shock:
        response of C_t to a unit Tr shock at t=0, holding all prices (r, w, Div) fixed.
    """
    iMPC_Tr = G_hh['C_GHH']['Tr']

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(iMPC_Tr[:T_plot, 0] * 100, marker='o', ms=4,
            color=COLORS['Tr'], linewidth=1.8, label='Conditional Transfer $Tr$')
    ax.axhline(0, color='gray', linestyle=':')
    ax.set_xlabel('Quarter $t$')
    ax.set_ylabel(r'$\partial C_t / \partial Tr_0 \;(\times 100)$')
    ax.set_title('PE iMPC Profile - BF Transfer')
    ax.legend(frameon=False)

    _save_or_show(fig, savepath)
    return fig, ax


# ---------------------------------------------------------------------------
# 7. GE IRFs
# ---------------------------------------------------------------------------

def plot_irf(irf_dict, variables, T_plot=30, savepath=None):
    # Plot IRFs for multiple shocks
    n = len(variables)
    ncols = max(1, (n + 1) // 2)
    nrows = 2 if n > ncols else 1

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes_flat = np.array(axes).flatten()

    col_cycle = ['steelblue', 'tomato', 'forestgreen', 'darkorange']

    for ax, var in zip(axes_flat, variables):
        for i, (label, irf) in enumerate(irf_dict.items()):
            ax.plot(irf[var][:T_plot] * 100,
                    color=col_cycle[i % len(col_cycle)],
                    linestyle=LS[i % 4], linewidth=1.8, label=label)
        ax.axhline(0, color='gray', linestyle=':')
        ax.set_title(VAR_LABELS.get(var, var))
        ax.set_xlabel('Quarters')
        ax.set_xlim(0, T_plot)
        ax.set_ylabel('% deviation from SS')
        ax.xaxis.set_major_locator(mticker.MultipleLocator(5))

    # Hide unused panels
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    axes_flat[-1].legend(frameon=False, fontsize=8)
    fig.suptitle('GE IRFs: Conditional Transfer Targeted (BF shock)', fontsize=11)

    _save_or_show(fig, savepath)
    return fig, axes_flat


def plot_channel_decomposition(irf_ins, irf_full, variables=('C','I','U','BF'),
                               T_plot=30, savepath=None):
    # Overlay Insurance (Pi frozen) vs Full (Pi moving); shade the composition gap.
    variables = list(variables)
    n = len(variables)
    ncols = max(1, (n + 1) // 2)
    nrows = 2 if n > ncols else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes_flat = np.array(axes).flatten()

    for ax, v in zip(axes_flat, variables):
        ins, full = irf_ins[v][:T_plot]*100, irf_full[v][:T_plot]*100
        x = np.arange(T_plot)
        ax.plot(x, ins,  color='steelblue', lw=1.8, label='Insurance Channel')
        ax.plot(x, full, color='tomato',    lw=1.8, label='Full Channel')
        ax.fill_between(x, ins, full, color='tomato', alpha=0.15, label='Composition Channel')
        ax.axhline(0, color='gray', ls=':')
        ax.set_title(VAR_LABELS.get(v, v)); ax.set_xlabel('Quarters'); ax.set_xlim(0, T_plot)
        ax.set_ylabel('% deviation from SS')
    
    # Hide unused panels
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    axes_flat[-1].legend(frameon=False, fontsize=8)
    _save_or_show(fig, savepath)
    return fig, axes_flat


# ---------------------------------------------------------------------------
# 8. Fiscal Multipliers Summary Table
# ---------------------------------------------------------------------------

def cumulative_response_table(irf_ins, irf_full, var='C', horizons=(4, 8, 20, 100),
                              savepath=None, label='tab:response_consump'):
    # Cumulative response of `var` split into insurance / full / composition.
    ins, full = irf_ins[var], irf_full[var]
    comp = full - ins
    rows = [(h, ins[:h].sum(), full[:h].sum(), comp[:h].sum(),
             100*comp[:h].sum()/full[:h].sum() if full[:h].sum() else np.nan)
            for h in horizons]
    print(f"\nCumulative Response of {var}  (sum of dX_t up to horizon)")
    print(f"{'H':>5}{'Insurance':>12}{'Full':>12}{'Composition':>14}{'Comp%':>8}")
    for h, si, sf, sc, pc in rows:
        print(f"{h:>5}{si:>12.4f}{sf:>12.4f}{sc:>14.4f}{pc:>7.1f}%")
    if savepath is not None:
        with open(savepath, 'w') as f:
            f.write("\\begin{table}[htbp]\\centering\n"
                    "\\caption{Response in Consumption: Different Horizons}\n"
                    f"\\label{{{label}}}\n"
                    "\\begin{tabular}{ccccc}\n\\toprule\n"
                    " Horizon & Insurance & Full & Composition & Comp.\\ (\\%) \\\\\\midrule\n")
            for h, si, sf, sc, pc in rows:
                f.write(f" {h} & {si:.4f} & {sf:.4f} & {sc:.4f} & {pc:.1f} \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    else:
        return rows


# ---------------------------------------------------------------------------
# 9. Sensitivity Analysis
# ---------------------------------------------------------------------------

def plot_bf_sweep(solve_fn, calibration, Tr_grid, savepath=None):
    """Re-Solve the Steady State across BF and plot the
        composition (informality, unemployment) and insurance (Gini, welfare) margins.
    solve_fn(calib) = lambda calib: solve_ss(hank_ss, calib, unknowns, targets)"""
    keys = ['Informal share', 'Unemployed share', 'Wealth Gini', 'Welfare E[V]']
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
    return fig, ax