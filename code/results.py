#=
# Plotting and reporting functions for the main model.
# All functions are self-contained and imported in main.py.
#=

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def rr():
    # Reload results.py into the global namespace (interactive use).
    import importlib, code.results
    importlib.reload(code.results)
    globals().update({k: v for k, v in vars(code.results).items()
                      if not k.startswith('_')})


# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------

BLUE, RED, GREEN, ORANGE, PURPLE, GRAY = (
    'steelblue', 'tomato', 'forestgreen', 'darkorange', 'mediumpurple', 'gray')

LS = {0: '-', 1: '--', 2: ':', 3: '-.'}

COLORS = {
    'Tr'   : BLUE,     # Conditional Transfer (BF) shock
    'Z'    : GREEN,    # TFP shock
    'i'    : ORANGE,   # nominal interest rate
    'r'    : ORANGE,   # real interest rate
    'B'    : PURPLE,   # real debt
    'F'    : BLUE,     # formal employment
    'I'    : RED,      # informal employment
    'U'    : PURPLE,   # unemployed
    'BF'   : GREEN,    # beneficiaries
    'Y'    : BLUE,
    'C'    : ORANGE,
    'ins'  : BLUE,     # insurance channel
    'full' : RED,      # full channel
}

CYCLE = [BLUE, RED, GREEN, ORANGE]   # shock cycle for multi-line panels

VAR_LABELS = {
    'Y'    : 'Output $Y$',
    'C'    : 'Consumption $C$',
    'F'    : 'Formal share $\\alpha_F$',
    'I'    : 'Informal share $\\alpha_I$',
    'U'    : 'Unemployment $\\alpha_U$',
    'BF'   : 'BF recipients $BF$',
    'w'    : 'Real wage $w$',
    'w_I'  : 'Informal wage $w_I$',
    'pi'   : 'Inflation $\\pi$',
    'pi_w' : 'Wage inflation $\\pi^w$',
    'r'    : 'Real rate $r$',
    'i'    : 'Nominal rate $i$',
    'B'    : 'Real Debt $B$',
    'A'    : 'Assets $A$',
    'G'    : 'Gov. spending $G$',
}

_TEX_LABELS = {
    'Formal share'     : r'Formal share $\alpha_F$',
    'Informal share'   : r'Informal share $\alpha_I$',
    'Unemployment'     : r'Unemployment $\alpha_U$',
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


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _hh(ss):
    # Household block of a steady state (keys: D, c, V, a_grid, ...).
    return ss.internals['household']


def _reshape_hh(arr, calibration):
    # (3 states, nT, 2 patience types, nE, nA)
    return arr.reshape(3, calibration['nT'], 2, calibration['nE'], -1)


def _panels(n):
    # Flattened axes grid sized to hold n panels.
    ncols = max(1, (n + 1) // 2)
    nrows = 2 if n > ncols else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    return fig, np.array(axes).flatten()


def _save_or_show(fig, savepath):
    plt.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, dpi=150, bbox_inches='tight')
    plt.show()


# ---------------------------------------------------------------------------
# 1. Steady-state summary
# ---------------------------------------------------------------------------

def print_ss_summary(ss, var_ss=['Y', 'C', 'beta_high', 'A', 'B']):
    print("\n" + "=" * 55)
    print("  STEADY STATE   (Model)")
    print("=" * 55)
    for k in var_ss:
        print(f"  {k:12s} = {ss[k]:.4f}")
    print(f"  {'beta_low':12s} = {ss['beta_high'] - ss['dbeta']:.4f}")
    print(f"  F + I + U  = 1.000,  model = {ss['F'] + ss['I'] + ss['U']:.3f}")
    print("=" * 55)


# ---------------------------------------------------------------------------
# 2. Consumption policy functions
# ---------------------------------------------------------------------------

def plot_consumption_policy(ss, calibration, T_plot_a=20, savepath=None):
    # c(s, theta_median, e_median, a) for the three employment states.
    h = _hh(ss)
    a_grid = h['a_grid']
    c = _reshape_hh(h['c'], calibration)
    nT, nE = calibration['nT'], calibration['nE']

    sectors = [(0, 'F', '-', 'Formal'),
               (1, 'I', '--', 'Informal'),
               (2, 'U', ':', 'Unemployed')]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for bi, beta_name in enumerate(['Impatient', 'Patient']):
        ax = axes[bi]
        for s, key, ls, lab in sectors:
            ax.plot(a_grid, c[s, nT // 2, bi, nE // 2, :],
                    color=COLORS[key], ls=ls, lw=1.8, label=lab)
        ax.set_xlabel('Assets $a$')
        ax.set_ylabel('Consumption $c(s, \\bar{\\theta}, \\bar{e}, a)$')
        ax.set_title(f'Policy Functions - {beta_name}')
        ax.set_xlim(0, T_plot_a)
        ax.set_ylim(0, 3)
        ax.legend(frameon=False)

    _save_or_show(fig, savepath)
    return fig, ax


# ---------------------------------------------------------------------------
# 3. Wealth distribution
# ---------------------------------------------------------------------------

def gini_coefficient(values, weights=None):
    idx = np.argsort(values)
    values = values[idx]
    weights = np.ones_like(values) if weights is None else weights[idx]

    pop  = np.concatenate([[0], np.cumsum(weights) / np.sum(weights)])
    wlth = np.concatenate([[0], np.cumsum(weights * values) / np.sum(weights * values)])
    return 1.0 - np.sum((pop[1:] - pop[:-1]) * (wlth[1:] + wlth[:-1]))


def gini_from_lorenz(pop, share):
    # Gini from a Lorenz curve: 1 - 2 * (area under the curve).
    # `pop` and `share` run 0 -> 1, sorted by population share.
    return 1.0 - np.sum(np.diff(pop) * (share[1:] + share[:-1]))


def plot_wealth_distribution(ss, lorenz_data=None, income_lorenz_data=None,
                             n_bins=40, savepath=None):
    # Row 1: wealth PDF near the constraint + Lorenz curve.
    # Row 2: income PDF + Lorenz curve (empirical PNAD).
    h = _hh(ss)
    D, a_grid = h['D'], h['a_grid']

    a_dist   = D.sum(axis=0)
    htm      = a_dist[0]
    cum_pop  = np.cumsum(a_dist)
    cum_wlth = np.cumsum(a_dist * a_grid) / np.sum(a_dist * a_grid)

    y = h['y']
    y = y[:, 0] if y.ndim == 2 else np.asarray(y)   # labor income per state
    m = D.sum(axis=1)                               # mass per state
    order = np.argsort(y)
    cum_pop_y = np.cumsum(m[order]) / m.sum()
    cum_inc_y = np.cumsum((m * y)[order]) / np.sum(m * y)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Wealth PDF near borrowing constraint
    widths = np.diff(np.append(a_grid[:n_bins], a_grid[n_bins]))
    axes[0, 0].bar(a_grid[:n_bins], a_dist[:n_bins],
                   width=widths, color=BLUE, alpha=0.7, align='edge')
    axes[0, 0].axvline(a_grid[0], color=RED, ls='--', label=f'HtM = {htm:.1%}')
    axes[0, 0].set_xlabel('Assets $a$')
    axes[0, 0].set_ylabel('Mass of households')
    axes[0, 0].set_title('Wealth Distribution (near borrowing constraint)')
    axes[0, 0].legend(frameon=False)

    # Wealth Lorenz
    axes[0, 1].plot(cum_pop, cum_wlth, color=BLUE, lw=1.6,
                    label='Model, Gini = {:.2f}'.format(gini_coefficient(a_grid, weights=a_dist)))
    axes[0, 1].plot([0, 1], [0, 1], color=GRAY, ls=':', lw=1.8, label='Perfect equality')
    if lorenz_data is not None:
        pctls = np.arange(501) / 500
        lorenz_emp = np.array([np.interp(p, lorenz_data[:, 0], lorenz_data[:, 1])
                               for p in pctls])
        axes[0, 1].plot(pctls, lorenz_emp, color=GREEN, ls='--', lw=1.6,
                        label='SCF 2019, Gini = {:.2f}'.format(gini_from_lorenz(pctls, lorenz_emp)))
    axes[0, 1].set_xlabel('Cumulative population share')
    axes[0, 1].set_ylabel('Cumulative wealth share')
    axes[0, 1].set_title('Lorenz Curve (wealth)')
    axes[0, 1].legend(frameon=False)

    # Income PDF
    edges = np.linspace(y.min(), y.max(), 41)
    hist, _ = np.histogram(y, bins=edges, weights=m)
    axes[1, 0].bar(edges[:-1], hist, width=np.diff(edges),
                   color=BLUE, alpha=0.7, align='edge')
    axes[1, 0].axvline(np.sum(m * y) / m.sum(), color=RED, ls='--', label='Mean income')
    axes[1, 0].set_xlabel('Income $y$')
    axes[1, 0].set_ylabel('Mass of households')
    axes[1, 0].set_title('Income Distribution')
    axes[1, 0].legend(frameon=False)

    # Income Lorenz
    axes[1, 1].plot(cum_pop_y, cum_inc_y, color=BLUE,
                    label='Model, Gini = {:.2f}'.format(gini_coefficient(y, weights=m)))
    axes[1, 1].plot([0, 1], [0, 1], color=GRAY, ls=':', lw=1.8, label='Perfect equality')
    if income_lorenz_data is not None:
        pctls = np.arange(501) / 500
        inc_emp = np.array([np.interp(p, income_lorenz_data[:, 0], income_lorenz_data[:, 1])
                            for p in pctls])
        axes[1, 1].plot(pctls, inc_emp, color=GREEN, ls='--', lw=1.8,
                        label='PNAD, Gini = {:.2f}'.format(gini_from_lorenz(pctls, inc_emp)))
    axes[1, 1].set_xlabel('Cumulative population share')
    axes[1, 1].set_ylabel('Cumulative income share')
    axes[1, 1].set_title('Lorenz Curve (income)')
    axes[1, 1].legend(frameon=False)

    _save_or_show(fig, savepath)
    return fig, axes


# ---------------------------------------------------------------------------
# 4. BF steady-state comparison
# ---------------------------------------------------------------------------

def _ss_stats(ss):
    # Scalar summary of one steady state.
    h = _hh(ss)
    D, c, V, a = h['D'], h['c'], h['V'], h['a_grid']
    a_dist = D.sum(0)
    return {
        'Formal share'     : ss['F'],
        'Informal share'   : ss['I'],
        'Unemployed share' : ss['U'],
        'HtM (a=a_min)'    : a_dist[0],
        'Wealth Gini'      : gini_coefficient(a, weights=a_dist),
        'Consumption Gini' : gini_coefficient(c.ravel(), weights=D.ravel()),
        'Aggregate C'      : ss['C'],
        'BF spending'      : ss['Tr'] * ss['BF'],
        'Welfare E[V]'     : float(np.sum(D * V)),
    }


def compare_bf_ss(ss_bf, ss_nobf, savepath=None, label='tab:bf_ss'):
    # Table: economy WITH BF vs the Tr=0 counterfactual.
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
# 5. Descriptive analysis
# ---------------------------------------------------------------------------

STATES  = ['Formal', 'Informal', 'Unemployed']
STATE_C = [COLORS['F'], COLORS['I'], COLORS['U']]


def _by_state(D, x, block):
    # Mean of x within each of the 3 employment segments (x broadcasts over assets).
    seg = lambda i: slice(i * block, (i + 1) * block)
    xi  = lambda i: x[seg(i)] if x.shape[0] == D.shape[0] else x
    return [(D[seg(i)] * xi(i)).sum() / D[seg(i)].sum() for i in range(3)]


def _welfare_gains(ss_bf, ss_nobf, calibration):
    # E[V] gain from BF for [Impatient, Patient, Formal, Informal, Unemployed].
    def ev(ss):
        h = _hh(ss)
        D, V = _reshape_hh(h['D'], calibration), _reshape_hh(h['V'], calibration)
        pat = [(D[:, :, b] * V[:, :, b]).sum() / D[:, :, b].sum() for b in (0, 1)]
        sec = [(D[s] * V[s]).sum() / D[s].sum() for s in (0, 1, 2)]
        return np.array(pat + sec)
    return ev(ss_bf) - ev(ss_nobf)


def plot_descriptives(ss, ss_nobf, calibration, n_q=5, savepath=None):
    # Four steady-state descriptives: consumption and wealth by employment state,
    # formality by wealth quantile, and welfare gain from BF by group.
    h = _hh(ss)
    D, c, a_grid = h['D'], h['c'], h['a_grid']
    block = D.shape[0] // 3

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].bar(STATES, _by_state(D, c, block), color=STATE_C)
    axes[0, 0].set_ylabel('Mean Consumption $c$')
    axes[0, 0].set_title('Consumption by Employment State')

    axes[0, 1].bar(STATES, _by_state(D, a_grid[None, :], block), color=STATE_C)
    axes[0, 1].set_ylabel('Mean Assets $a$')
    axes[0, 1].set_title('Wealth by Employment State')

    # Formality and unemployment by wealth quantile
    cum_left = np.concatenate([[0.0], np.cumsum(D.sum(0))[:-1]])
    qidx = np.minimum((cum_left * n_q).astype(int), n_q - 1)
    shares = np.full((3, n_q), np.nan)
    for q in range(n_q):
        mass = np.array([D[i * block:(i + 1) * block, qidx == q].sum() for i in range(3)])
        if mass.sum() > 0:
            shares[:, q] = mass / mass.sum()
    x = np.arange(n_q)
    for i, (lab, col) in enumerate(zip(STATES, STATE_C)):
        axes[1, 0].bar(x + (i - 1) * 0.2, shares[i], width=0.2, color=col, label=lab)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels([f'Q{i+1}' for i in range(n_q)])
    axes[1, 0].set_xlabel('Wealth quantile')
    axes[1, 0].set_ylabel('Share')
    axes[1, 0].set_title('Formality and unemployment by wealth')
    axes[1, 0].legend(frameon=False)

    # Welfare gain from BF by patience type and by sector
    labels = ['Impatient', 'Patient'] + STATES
    axes[1, 1].bar(labels, _welfare_gains(ss, ss_nobf, calibration),
                   color=[GRAY, GRAY] + STATE_C)
    axes[1, 1].axhline(0, color='k', linewidth=0.8)
    axes[1, 1].set_ylabel('Welfare Gain $\\Delta E[V]$')
    axes[1, 1].set_title('Who gains from Bolsa Familia')

    _save_or_show(fig, savepath)
    return fig, axes


# ---------------------------------------------------------------------------
# 6. PE iMPC
# ---------------------------------------------------------------------------

def plot_impc(G_hh, T_plot=16, savepath=None):
    # PE iMPC: response of C_t to a unit Tr shock at t=0 (prices fixed).
    iMPC_Tr = G_hh['C_GHH']['Tr']

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(iMPC_Tr[:T_plot, 0] * 100, marker='o', ms=4,
            color=COLORS['Tr'], lw=2.0, label='Conditional Transfer $Tr$')
    ax.axhline(0, color=GRAY, linestyle=':')
    ax.set_xlabel('Quarter $t$')
    ax.set_ylabel(r'$\partial C_t / \partial Tr_0 \;(\times 100)$')
    ax.set_title('PE iMPC Profile - BF Transfer')
    ax.legend(frameon=False)

    _save_or_show(fig, savepath)
    return fig, ax


# ---------------------------------------------------------------------------
# 7. GE IRFs
# ---------------------------------------------------------------------------

def plot_irf(irf_dict, variables=('C','I','U','BF','pi','w'), T_plot=30, savepath=None):
    n = len(variables)
    fig, axes = _panels(n)

    for ax, var in zip(axes, variables):
        for i, (label, irf) in enumerate(irf_dict.items()):
            ax.plot(irf[var][:T_plot] * 100, color=CYCLE[i % len(CYCLE)],
                    ls=LS[i % 4], lw=2.2, label=label)
        ax.axhline(0, color=GRAY, ls=':')
        ax.set_title(VAR_LABELS.get(var, var))
        ax.set_xlabel('Quarters')
        ax.set_xlim(0, T_plot)
        ax.set_ylabel('% deviation from SS')
        ax.xaxis.set_major_locator(mticker.MultipleLocator(5))

    for ax in axes[n:]:
        ax.set_visible(False)

    axes[-1].legend(frameon=False)
    fig.suptitle('GE IRFs: Conditional Transfer Targeted (BF shock)', fontsize=11)
    _save_or_show(fig, savepath)
    return fig, axes


def plot_irf_decomposition(irf_ins, irf_full, irf_pe=None,
                               variables=('C','I','U','BF','pi','w'), T_plot=30, savepath=None):
    # Insurance (Pi frozen) vs Full (Pi moving); shade the composition gap.
    variables = list(variables)
    n = len(variables)
    fig, axes = _panels(n)
    x = np.arange(T_plot)

    for ax, v in zip(axes, variables):
        ins, full = irf_ins[v][:T_plot] * 100, irf_full[v][:T_plot] * 100

        if irf_pe is not None and v in irf_pe:
            ax.plot(x, irf_pe[v][:T_plot]*100, color='gray',
                    lw=1.6, ls='--', label='Partial Eq. (prices fixed)')

        ax.plot(x, ins,  color=COLORS['ins'],  lw=2.2, label='Insurance Channel')
        ax.plot(x, full, color=COLORS['full'], lw=2.2, label='Full Channel')
        ax.fill_between(x, ins, full, color=COLORS['full'], alpha=0.15, label='Composition Channel')
        ax.axhline(0, color=GRAY, ls=':')
        ax.set_title(VAR_LABELS.get(v, v))
        ax.set_xlabel('Quarters')
        ax.set_xlim(0, T_plot)
        ax.set_ylabel('% deviation from SS')

    for ax in axes[n:]:
        ax.set_visible(False)

    axes[0].legend(frameon=False)
    _save_or_show(fig, savepath)
    return fig, axes


def plot_irf_financing(irf_tax, irf_debt, variables=('C','Y','pi','w','r','B'),
                              T_plot=30, savepath=None):
    variables = list(variables); n = len(variables)
    ncols = max(1, (n + 1)//2); nrows = 2 if n > ncols else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
    axf = np.array(axes).flatten(); x = np.arange(T_plot)
    for ax, v in zip(axf, variables):
        if v in irf_tax:
            ax.plot(x, irf_tax[v][:T_plot]*100,  color='steelblue', lw=1.8, label='Tax-financed')
        if v in irf_debt:
            ax.plot(x, irf_debt[v][:T_plot]*100, color='tomato', lw=1.8, ls='--', label='Debt-financed')
        ax.axhline(0, color='gray', ls=':')
        ax.set_title(VAR_LABELS.get(v, v)); ax.set_xlabel('Quarters')
        ax.set_xlim(0, T_plot); ax.set_ylabel('% deviation from SS')
    for ax in axf[n:]:
        ax.set_visible(False)
        
    axf[0].legend(frameon=False)
    _save_or_show(fig, savepath)
    return fig, axf


# ---------------------------------------------------------------------------
# 8. Fiscal multipliers summary table
# ---------------------------------------------------------------------------

def cumulative_response_table(irf_ins, irf_full, var='C', horizons=(4, 8, 20, 100),
                              savepath=None, label='tab:response_consump'):
    # Cumulative response of `var` split into insurance / full / composition.
    ins, full = irf_ins[var], irf_full[var]
    comp = full - ins
    rows = [(h, ins[:h].sum(), full[:h].sum(), comp[:h].sum(),
             100 * comp[:h].sum() / full[:h].sum() if full[:h].sum() else np.nan)
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
# 9. Sensitivity analysis
# ---------------------------------------------------------------------------

def plot_bf_sweep(solve_fn, calibration, ss_base=None, span=0.2, nT=5, savepath=None):
    # Re-solve the steady state across BF generosity and plot key margins.
    keys = ['Informal share', 'Unemployed share', 'Wealth Gini', 'Welfare E[V]']
    series = {k: [] for k in keys}
    Tr0 = calibration['Tr']
    Tr_grid = np.linspace(Tr0 - span, Tr0 + span, nT)
    for Tr in Tr_grid:
        if ss_base is not None and np.isclose(Tr, Tr0):
            st = _ss_stats(ss_base)
        else:
            st = _ss_stats(solve_fn({**calibration, 'Tr': Tr}))
        for k in keys:
            series[k].append(st[k])

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, k in zip(axes.flat, keys):
        ax.plot(Tr_grid, series[k], marker='o', ms=4, color=BLUE, lw=2.0)
        ax.axvline(Tr0, color='gray', ls='--', lw=1)
        ax.set_xticks(Tr_grid)
        ax.set_xlabel('BF generosity $Tr$')
        ax.set_title(k)
    fig.suptitle('Steady state vs BF generosity', fontsize=11)
    _save_or_show(fig, savepath)
    return fig, axes