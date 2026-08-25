#=
# ---------------------------------------------------------------------------
# Graphics and Tables in functions for main.py.
# ---------------------------------------------------------------------------
#=

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from code.p5_calibration import (pnad, alpha, qs, gini_coefficient,
                                 gini_from_lorenz, top_share, _wquantile)

def rr():
    # Reload results.py into the global namespace (interactive use).
    import importlib, code.p7_results
    importlib.reload(code.p7_results)
    globals().update({k: v for k, v in vars(code.p7_results).items()
                      if not k.startswith('_')})


# ---- Config ---------------------------------------------------------------
PAL = ('#1b325f', '#9cc4e4', '#e9f2f9',
       '#3a89c9', '#f26c4f', '#a8a3af')
STATES = {'F': 'Formal', 'I': 'Informal', 'U': 'Unemployed'}
DIGITS = str.maketrans(dict(zip('0123456789', ('Zero', 'One', 'Two', 'Three',
          'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine'))))
COL1, COL2, COL3, COL4, COL5, GRAY = PAL

LS = {0: '-', 1: '--', 2: ':', 3: '-.'}
CYCLE = [PAL[i] for i in (0, 3, 1, 4, 2)]

COLORS = {
    'Tr'   : COL1,        # Conditional Transfer
    'Z'    : COL5,        # TFP shock
    'i'    : COL5,        # nominal interest rate
    'r'    : COL5,        # real interest rate
    'B'    : COL5,        # real debt
    'F'    : COL1,        # formal employment
    'I'    : COL4,        # informal employment
    'U'    : COL2,        # unemployed
    'BF'   : COL5,        # beneficiaries
    'Y'    : COL4,
    'C'    : COL4,
    'full' : COL1,        # full channel
    'ins'  : COL4,        # insurance channel
}

LABELS = {
    'Y'    : 'Output $Y$',
    'C'    : 'Consumption $C$',
    'F'    : 'Formal share $\\alpha_F$',
    'I'    : 'Informal share $\\alpha_I$',
    'U'    : 'Unemployment $\\alpha_U$',
    'BF'   : 'BF recipients $BF$',
    'w'    : 'Real wage $w$',
    'pi'   : 'Inflation $\\pi$',
    'pi_w' : 'Wage inflation $\\pi^w$',
    'r'    : 'Real rate $r$',
    'i'    : 'Nominal rate $i$',
    'B'    : 'Real Debt $B$',
    'A'    : 'Assets $A$',
    'G'    : 'Gov. spending $G$',
    'Formal Share'     : r'Formal Share $\alpha_F$',
    'Informal Share'   : r'Informal Share $\alpha_I$',
    'Unemployed Share' : r'Unemployment $\alpha_U$',
    'HtM (a=a_min)'    : r'Hand-to-Mouth',
    'Wealth Gini'      : r'Wealth Gini',
    'Consumption Gini' : r'Consumption Gini',
    'Aggregate C'      : r'Aggregate Consumption $C$',
    'BF Spending'      : r'BF Spending',
    'Welfare E[V]'     : r'Welfare $\mathbb{E}[V]$',
}

_hh = lambda ss: ss.internals['household']
_reshape_hh = lambda arr, calib: arr.reshape(3, calib['nT'], 2, calib['nE'], -1)

plt.rcParams.update({'font.size'        : 10,
                     'axes.spines.top'  : False,
                     'axes.spines.right': False,
                     'figure.dpi'       : 120})


# ---- Shared Helpers -------------------------------------------------------
def _panels(n):         # Flattened axes grid sized to hold n panels.
    ncols = max(1, (n + 1) // 2)
    nrows = 2 if n > ncols else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    return fig, np.array(axes).flatten()


def _save_or_show(fig, savepath):
    plt.tight_layout()
    if savepath is None:
        plt.show()
    else:
        fig.savefig(savepath, dpi=150, bbox_inches='tight')
        plt.close(fig)


def _tex_table(head, rows, caption, label, colspec, size=r'\small', savepath=None):
    # Return the LaTeX table when savepath is None; otherwise write it silently.
    line = lambda r: r if r.startswith('\\') else r + r' \\'
    tex = ("%---------------------------------------------------\n"
           "\\begin{table}[htbp]\n\\centering\n"
           f"\\caption{{{caption}}}\n\\label{{{label}}}\n{size}\n"
           f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n"
           + "\n".join(map(line, head)) + "\n\\midrule\n"
           + "\n".join(map(line, rows)) + "\n"
           "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
           "%---------------------------------------------------\n")
    if savepath is None:
        return tex
    with open(savepath, 'w', encoding='utf-8') as fobj:
        fobj.write(tex)


def _irf_panels(variables, T_plot, draw, legend_ax=0,
                suptitle=None, xtick=None, titles=None, savepath=None):
    # Shared IRF grid: `draw(ax, var)` plots the lines; the rest is scaffolding.
    variables = list(variables)
    titles = titles or {}
    n = len(variables)
    fig, axes = _panels(n)
    for ax, v in zip(axes, variables):
        draw(ax, v)
        ax.axhline(0, color=GRAY, ls=':')
        ax.set_title(titles.get(v, LABELS.get(v, v)))
        ax.set_xlabel('Quarters')
        ax.set_ylabel('% deviation from SS')
        ax.set_xlim(0, T_plot)
        if xtick:
            ax.xaxis.set_major_locator(mticker.MultipleLocator(xtick))
    for ax in axes[n:]:
        ax.set_visible(False)
    axes[legend_ax].legend(frameon=False)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    _save_or_show(fig, savepath)
    return fig, axes



# ---------------------------------------------------------------------------
# 1. Steady-State Summary
# ---------------------------------------------------------------------------

vars_ss = ['Y', 'Y_I', 'C_GHH', 'C', 'beta_high', 'A', 'B', 'psi', 'w',
           'Z', 'BF', 'L', 'Div', 'tau', 'asset_mkt',
           'goods_mkt', 'labor_mkt', 'wage_nkpc']

def print_ss_summary(ss, var_ss=vars_ss):
    print("\n" + "=" * 55)
    print("  STEADY STATE   (Model)")
    print("=" * 55)
    for k in var_ss:
        print(f"  {k:12s} = {ss[k]:.4f}")
        if k == 'beta_high':
            print(f"  {'beta_low':12s} = {ss['beta_high'] - ss['dbeta']:.4f}")
    print("=" * 55)


# ---- Transition Matrix ----------------------------------------------------
def transition_table(ss, savepath=None, label='tab:transitions'):
    # Model vs PNAD quarterly transition matrix
    P    = _hh(ss)['P']
    d    = pd.read_csv('data/final/pnad_transition_matrix.csv', index_col=0)
    d_ss = pd.read_csv('data/final/pnad_calibration.csv').loc[0]
    Pd, shd = d.loc[list(STATES), list(STATES)].to_numpy(), d_ss[list(STATES)].to_numpy()
    sh = np.array([ss[s] for s in STATES])

    head = [r' & \multicolumn{3}{c}{\textbf{Model}} & \multicolumn{3}{c}{\textbf{PNAD-C}}',
            r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}',
            r'Origin & $F$ & $I$ & $U$ & $F$ & $I$ & $U$']
    rows = [f'{n} & ' + " & ".join(f'{100*x:.2f}' for x in np.r_[P[i], Pd[i]])
            for i, n in enumerate(STATES.values())]
    rows += ['\\midrule', 'Stationary Share & '
             + " & ".join(f'{100*x:.2f}' for x in np.r_[sh, shd])]

    return _tex_table(head, rows, r'Quarterly Labor-Market Transitions (\%)',
                      label, 'lcccccc', savepath=savepath)


# ---- Compare Tr with no-Tr ------------------------------------------------
def _ss_stats(ss):
    h = _hh(ss)
    D, c, V, a = h['D'], h['c'], h['V'], h['a_grid']
    a_dist = D.sum(0)
    return {'Formal Share'     : ss['F'],
            'Informal Share'   : ss['I'],
            'Unemployed Share' : ss['U'],
            'HtM (a=a_min)'    : a_dist[0],
            'Wealth Gini'      : gini_coefficient(a, weights=a_dist),
            'Consumption Gini' : gini_coefficient(c.ravel(), weights=D.ravel()),
            'Aggregate C'      : ss['C'],
            'BF Spending'      : ss['Tr'] * ss['BF'],
            'Welfare E[V]'     : float(np.sum(D * V))}


def compare_bf_ss(ss_bf, ss_nobf, savepath=None, label='tab:bf_ss'):
    # Table: Economy with BF vs the Tr=0 counterfactual.
    s1, s0 = _ss_stats(ss_bf), _ss_stats(ss_nobf)
    pct_rows = {'Formal Share', 'Informal Share', 'Unemployed Share', 'HtM (a=a_min)'}

    def cell(k, x, delta=False):
        if k in pct_rows:
            return f'{x*100:+.1f}' if delta else rf'{x*100:.1f}\%'
        return f'{x:+.3f}' if delta else f'{x:.3f}'

    rows = [rf'{LABELS.get(k, k)} & {cell(k, s1[k])} & {cell(k, s0[k])} '
            rf'& {cell(k, s1[k]-s0[k], delta=True)}' for k in s1]

    return _tex_table([r' & With BF & No BF & $\Delta$'], rows,
                      r"Steady State: Bolsa Fam\'ilia vs.\ No-Transfer Counterfactual",
                      label, 'lccc', savepath=savepath)



# ---------------------------------------------------------------------------
# 2. Consumption Policy Functions
# ---------------------------------------------------------------------------

def plot_consumption_policy(ss, calibration, T_plot_a=10, savepath=None):
    # c(s, theta_median, e_median, a) for the three employment states.
    h = _hh(ss)
    a_grid = h['a_grid']
    c = _reshape_hh(h['c'], calibration)
    nT, nE = calibration['nT'], calibration['nE']

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for bi, beta_name in enumerate(['Impatient', 'Patient']):
        ax = axes[bi]
        for s, key in enumerate(STATES):
            ax.plot(a_grid, c[s, nT // 2, bi, nE // 2, :],
                    color=COLORS[key], ls=LS[s], lw=1.8, label=STATES[key])
        ax.set_xlabel('Assets $a$')
        ax.set_ylabel('Consumption $c(s, \\bar{\\theta}, \\bar{e}, a)$')
        ax.set_title(f'Policy Functions - {beta_name}')
        ax.set_xlim(0, T_plot_a)
        ax.set_ylim(0.5, 3)
        ax.legend(frameon=False)

    _save_or_show(fig, savepath)
    return fig, ax



# ---------------------------------------------------------------------------
# 3. Wealth and Income Distribution
# ---------------------------------------------------------------------------

def pnad_lorenz(n=2001):
    # Lorenz Curve of Labor Earnings (PNAD).
    d = pd.read_csv('data/final/pnad_income_dist.csv')
    shares = alpha[:2] / alpha[:2].sum()          # F, I over the employed
    x = np.linspace(0, d.wage.max(), n)
    f = sum(s * np.interp(x, g.wage, g.y, left=0, right=0)
            for s, (_, g) in zip(shares, d.groupby('group')))
    pop, inc = np.cumsum(f), np.cumsum(f * x)
    return np.column_stack([pop / pop[-1], inc / inc[-1]])


def _bar_panel(ax, left, height, width, xlabel, title, vlines=(), xlim=None):
    # PDF Panel: Weighted Histogram + Labelled Vertical Markers.
    ax.bar(left, 100 * height, width=width, color=COL1, alpha=0.7, align='edge')
    for xv, lab, col, ls in vlines:
        ax.axvline(xv, color=col, ls=ls, label=lab)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Households (%)')
    if xlim is not None:
        ax.set_xlim(0, xlim)
    ax.set_title(title)
    if vlines:
        ax.legend(frameon=False)


def _lorenz_panel(ax, pop, share, emp, emp_label, kind):
    # Lorenz Panel: Model vs Data, Gini in the legend, Top Shares in a box.
    ax.plot(pop, share, color=COL1, lw=1.8,
            label=f'Model, Gini = {gini_from_lorenz(pop, share):.2f}')
    ax.plot([0, 1], [0, 1], color=COL2, ls=':', lw=1.8, label='Perfect Equality')
    p = np.linspace(0, 1, 501)
    L = np.interp(p, emp[:, 0], emp[:, 1])
    ax.plot(p, L, color=COL4, ls='--', lw=1.6,
            label=f'{emp_label}, Gini = {gini_from_lorenz(p, L):.2f}')

    box = (f'Top 10% = {top_share(pop, share, 0.10):.0%}\n'
           f'Top 1%  = {top_share(pop, share, 0.01):.0%}')
    ax.text(0.97, 0.03, box, transform=ax.transAxes, ha='right', va='bottom',
            fontsize=8, bbox=dict(boxstyle='round', fc=COL3, ec=COL3, alpha=0.6))

    ax.set_xlabel('Cumulative Population Share')
    ax.set_ylabel(f'Cumulative {kind} Share')
    ax.set_title(f'Lorenz Curve ({kind})')
    ax.legend(frameon=False, loc='upper left')


def plot_income_distribution(ss, bins=51, lim=3.0, savepath=None):
    # Model vs PNAD earnings density, by sector.
    h = _hh(ss)
    earn = float(ss['w']) * (h['n_f'] + h['n_i'])[:, 0]     # gross, a-invariant
    mass = h['D'].sum(1)
    blk  = earn.size // 3
    ref  = np.average(earn[:blk], weights=mass[:blk])       # mean formal wage

    d = pd.read_csv('data/final/pnad_income_dist.csv')
    edges = np.linspace(-lim, lim, bins)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    for ax, s, sl, col in zip(axes, STATES, [slice(0, blk), slice(blk, 2*blk)],
                              [COL1, COL4]):
        ly = np.log(earn[sl] / ref)
        mod, _ = np.histogram(ly, bins=edges, weights=mass[sl])
        g   = d[(d.group == s) & (d.wage > 0)]
        dat, _ = np.histogram(np.log(g.wage / pnad['y_F']), bins=edges,
                              weights=g.y * np.gradient(g.wage))
        c = 0.5 * (edges[:-1] + edges[1:])
        ax.bar(c, 100 * mod / mod.sum(), width=np.diff(edges), color=col,
               alpha=0.55, label='Model')
        ax.plot(c, 100 * dat / dat.sum(), color='black', lw=1.6, ls='--', label='PNAD')
        # Targeted Quantiles
        ax.plot(_wquantile(ly, mass[sl], qs), np.zeros(len(qs)), '|', color=col,
                ms=14, mew=2)
        ax.plot([pnad[f'q{q}_{s}'] for q in qs], np.zeros(len(qs)), '|',
                color='black', ms=14, mew=2)
        ax.axvline(0, color=GRAY, lw=0.8)
        ax.set_xlabel(r'$\log(y / \bar{y}^F)$')
        ax.set_ylabel('Density (%)')
        ax.set_title(f'{STATES[s]} Earnings')
        ax.legend(frameon=False)

    _save_or_show(fig, savepath)


def plot_wealth_distribution(ss, n_bins=30, savepath=None):
    # 2x2: Wealth and Income, each as a PDF and a Lorenz curve.
    h = _hh(ss)
    D, a_grid = h['D'], h['a_grid']

    a_dist  = D.sum(0)
    aL_pop  = np.concatenate([[0], np.cumsum(a_dist)])
    aL_share = np.concatenate([[0], np.cumsum(a_dist * a_grid) / np.sum(a_dist * a_grid)])

    y = h['y']
    y = y[:, 0] if y.ndim == 2 else np.asarray(y)       # labor income per state
    m = D.sum(1)
    o = np.argsort(y); ys, ms = y[o], m[o]
    cy = np.cumsum(ms) / ms.sum()
    yL_pop  = np.concatenate([[0], cy])
    yL_share = np.concatenate([[0], np.cumsum(ms * ys) / np.sum(ms * ys)])

    edges = np.linspace(y.min(), y.max(), 231)
    y_hist, _ = np.histogram(y, bins=edges, weights=m)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    _bar_panel(axes[0, 0], a_grid[:n_bins], a_dist[:n_bins],
               np.diff(np.append(a_grid[:n_bins], a_grid[n_bins])),
               'Assets $a$', 'Wealth Distribution (Near Constraint)',
               vlines=[(a_grid[1], f'HtM = {a_dist[0]:.1%}', COL1, '--')])
    _lorenz_panel(axes[0, 1], aL_pop, aL_share,
                  np.loadtxt('data/lorenz_nw_scf_2019.raw', delimiter=','),
                  'US SCF 2019 (Proxy)', 'Wealth')

    _bar_panel(axes[1, 0], edges[:-1], y_hist, np.diff(edges),
               'Income $y$', 'Income Distribution (Near Constraint)', xlim=10)
    _lorenz_panel(axes[1, 1], yL_pop, yL_share, pnad_lorenz(),
                  'PNAD', 'Income')

    _save_or_show(fig, savepath)
    return fig, axes



# ---------------------------------------------------------------------------
# 4. Descriptive Analysis
# ---------------------------------------------------------------------------

def _by_state(D, x, block):
    # Mean of x within each of the 3 Employment Segments (x broadcasts over assets).
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
    # Consumption and Wealth by State, Formality by Wealth, Welfare gain by Group.
    h = _hh(ss)
    D, c, a_grid = h['D'], h['c'], h['a_grid']
    block = D.shape[0] // 3

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].bar(list(STATES.values()), _by_state(D, c, block),
                   color=[COLORS[s] for s in STATES])
    axes[0, 0].set_ylabel('Mean Consumption $c$')
    axes[0, 0].set_title('Consumption by Employment State')

    axes[0, 1].bar(list(STATES.values()), _by_state(D, a_grid[None, :], block),
                   color=[COLORS[s] for s in STATES])
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
    for i, (key, lab) in enumerate(STATES.items()):
        axes[1, 0].bar(x + (i - 1) * 0.2, shares[i], width=0.2, color=COLORS[key], label=lab)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels([f'Q{i+1}' for i in range(n_q)])
    axes[1, 0].set_xlabel('Wealth Quantile')
    axes[1, 0].set_ylabel('Share')
    axes[1, 0].set_title('Formality and Unemployment by Wealth')
    axes[1, 0].legend(frameon=False)

    # Welfare gain from BF by patience type and by sector
    axes[1, 1].bar(['Impatient', 'Patient'] + list(STATES.values()),
                   _welfare_gains(ss, ss_nobf, calibration),
                   color=[GRAY, GRAY] + [COLORS[s] for s in STATES])
    axes[1, 1].axhline(0, color='k', linewidth=0.8)
    axes[1, 1].set_ylabel('Welfare Gain $\\Delta E[V]$')
    axes[1, 1].set_title('Who gains from Bolsa Familia')

    _save_or_show(fig, savepath)
    return fig, axes


# ---- Sensitivity Analysis -------------------------------------------------
def plot_bf_sweep(solve_fn, calibration, ss_base=None, ss_nobf=None, savepath=None):
    # Re-solve the Steady State over Tr in {0, .5, 1, 1.5, 2} x Tr0
    keys = ['Informal Share', 'Unemployed Share', 'Wealth Gini', 'Welfare E[V]']
    series = {k: [] for k in keys}
    Tr0 = calibration['Tr']
    Tr_grid = Tr0 * np.array([0, 0.5, 1, 1.5, 2])
    reuse = [(0.0, ss_nobf), (Tr0, ss_base)]
    for Tr in Tr_grid:
        base = next((s for t, s in reuse if s is not None and np.isclose(Tr, t)), None)
        st = _ss_stats(base if base is not None else solve_fn({**calibration, 'Tr': Tr}))
        for k in keys:
            series[k].append(st[k])

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, k in zip(axes.flat, keys):
        ax.plot(Tr_grid, series[k], marker='o', ms=4, color=COL4, lw=2.2)
        ax.axvline(Tr0, color=GRAY, ls='--', lw=1)
        ax.set_xticks(Tr_grid, [f'{t:.2f}' for t in Tr_grid])
        ax.set_xlabel('BF Size $Tr$')
        ax.set_title(k)
    fig.suptitle('Steady State vs BF Transfer', fontsize=11)
    _save_or_show(fig, savepath)
    return fig, axes



# ---------------------------------------------------------------------------
# 5. PE iMPC and GE IRFs
# ---------------------------------------------------------------------------

def plot_impc(G_hh, h_ant=4, T_plot=20, key='C_GHH', savepath=None):
    # Intertemporal MPC: consumption path, holding prices (r, w, Div) fixed.
    M   = G_hh[key]['Tr']
    m   = M[:T_plot, 0]
    m_a = M[:T_plot, h_ant]
    cum = np.cumsum(m)
    t   = np.arange(T_plot)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(t, m * 100, color=COLORS['Tr'], alpha=0.85, label='Immediate $t=0$')
    ax.plot(t, cum * 100, color=COL5, marker='o', ms=3, lw=1.8, label='Cumulative')
    ax.plot(t, m_a * 100, color=COL4, marker='s', ms=3, lw=1.8,
            label=f'Anticipated $t={h_ant}$')
    ax.axvline(h_ant, color=COL4, ls=':', lw=1, alpha=0.6)
    ax.axhline(0, color=GRAY, ls=':')
    ax.set_xlabel('Quarters after Transfer $t$')
    ax.set_ylabel('Consumption Response')
    ax.set_title(f'PE iMPC - Instantaneous vs Anticipated at $t={h_ant}$')
    ax.legend(frameon=False)

    _save_or_show(fig, savepath)
    return fig, ax


def plot_irf(irf_dict, variables=('C', 'I', 'U', 'BF', 'pi', 'w'),
             title='CCT Shock (T)', T_plot=30, savepath=None):
    def draw(ax, v):
        for i, (label, irf) in enumerate(irf_dict.items()):
            ax.plot(irf[v][:T_plot] * 100, color=CYCLE[i % len(CYCLE)],
                    ls=LS[i % 4], lw=2.2, label=label)
    return _irf_panels(variables, T_plot, draw, legend_ax=0,
                       suptitle=f'GE IRFs: {title}', xtick=5, savepath=savepath)


def plot_irf_decomposition(irf_ins, irf_full, irf_pe=None,
                           variables=('C', 'I', 'U', 'BF', 'pi', 'w'),
                           T_plot=30, savepath=None):
    # Insurance (Pi frozen) vs Full (Pi moving); shade the composition gap.
    x = np.arange(T_plot)
    def draw(ax, v):
        ins, full = irf_ins[v][:T_plot] * 100, irf_full[v][:T_plot] * 100
        if irf_pe is not None and v in irf_pe:
            ax.plot(x, irf_pe[v][:T_plot] * 100, color=GRAY, lw=1.6, ls='--',
                    label='Partial Eq. (Prices Fixed)')
        ax.plot(x, ins,  color=COLORS['ins'],  lw=2.2, label='Insurance Channel')
        ax.plot(x, full, color=COLORS['full'], lw=2.2, label='Full Channel')
        ax.fill_between(x, ins, full, color=COLORS['full'], alpha=0.15,
                        label='Composition Channel')
    return _irf_panels(variables, T_plot, draw, savepath=savepath)


def plot_irf_financing(irf_tax, irf_debt, variables=('C', 'Y', 'pi', 'w', 'r', 'B'),
                       T_plot=30, savepath=None):
    x = np.arange(T_plot)
    def draw(ax, v):
        if v in irf_tax:
            ax.plot(x, irf_tax[v][:T_plot] * 100, color=COLORS['ins'],
                    lw=1.8, label='Tax-Financed')
        if v in irf_debt:
            ax.plot(x, irf_debt[v][:T_plot] * 100, color=COLORS['full'],
                    lw=1.8, ls='--', label='Debt-Financed')
        if v == 'B':         # overlay tau on the same axis
            for irf, ls, lab in [(irf_tax, '-', r'$\tau$ (Tax)'),
                                 (irf_debt, '--', r'$\tau$ (Debt)')]:
                if 'tau' in irf:
                    ax.plot(x, irf['tau'][:T_plot] * 100, color=COL5,
                            ls=ls, lw=1.8, label=lab)
            ax.legend(frameon=False)
    return _irf_panels(variables, T_plot, draw,
                       titles={'B': r'Debt $B$ x Transfers $\tau$'}, savepath=savepath)



# ---------------------------------------------------------------------------
# 6. Fiscal Multipliers
# ---------------------------------------------------------------------------

def cumulative_response_table(irf_ins, irf_full, variables=('C', 'U', 'pi', 'w'),
                              horizons=(1, 4, 20, 100), shock='T', savepath=None,
                              label='tab:cumulative_response'):
    # Cumulative Response (sum of dX_t up to horizon, %)
    if isinstance(variables, str):
        variables = [variables]

    rows = []
    for var in variables:
        cins  = (np.asarray(irf_ins[var])  * 100).cumsum()
        cfull = (np.asarray(irf_full[var]) * 100).cumsum()
        blocks = []
        for h in horizons:
            i = min(h, len(cfull)) - 1                      # guard h > T
            blocks.append((h, cins[i], cfull[i], cfull[i] - cins[i]))
        if rows:
            rows.append('\\midrule')
        lab = LABELS.get(var, var)
        rows += [f' {lab if j == 0 else ""} & {h} & {si:.3f} & {sf:.3f} & {sc:.3f}'
                 for j, (h, si, sf, sc) in enumerate(blocks)]

    return _tex_table(['Variable & Horizon & Insurance & Full & Composition'], rows,
                      f'Cumulative Response (\\%), Shock in ${shock}$', label, 'lccccc',
                      savepath=savepath)



# ---------------------------------------------------------------------------
# 9. Calibration
# ---------------------------------------------------------------------------

MACRO_FMT = {k: '.1%' for k in ['F', 'I', 'U', 'BF', 'rstar']}
MACRO_FMT |= {f'{p}_{a}{b}': '.1%' for p in ('mod', 'dat') for a in STATES for b in STATES}
MACRO_FMT |= {f'dat_{s}': '.1%' for s in STATES}

# Estimated from PNAD or calibrated inside the model: report 3 decimals.
ROUND3 = ['delta_F', 'delta_I', 'mu_I', 'sigma_F', 'sigma_I', 'sd_e', 'h_ratio',
          'pi_F', 'pi_I', 'pi_UF', 'pi_UI', 'psi', 'varphi', 'beta_high', 'beta_low',
          'xi', 'Z', 'tau', *[f'q{q}_{s}' for s in STATES for q in qs]]
MACRO_FMT |= {k: '.3f' for k in ROUND3}
MACRO_FMT |= {f'dat_{k}': '.3f' for k in ROUND3}

# Normalizations print as 1.0, not 1; mean hours as one decimal.
MACRO_FMT |= {k: '.1f' for k in ['h_F', 'Y', 'dat_h_F', 'dat_h_I']}


def tex_macros(ss, calibration, savepath=None, prefix='m'):
    # Return the \newcommand macros when savepath is None; otherwise write them.
    P = _hh(ss)['P']
    d = pd.read_csv('data/final/pnad_transition_matrix.csv', index_col=0)

    ctx  = {k: float(v) for k, v in calibration.items() if np.size(v) == 1}
    ctx |= {k: float(ss[k]) for k in ss.keys() if np.size(ss[k]) == 1}
    ctx |= {f'mod_{a}{b}': P[i, j] for i, a in enumerate(STATES) for j, b in enumerate(STATES)}
    ctx |= {f'dat_{a}{b}': d.loc[a, b] for a in STATES for b in STATES}
    ctx |= {f'dat_{k}': float(v) for k, v in     # PNAD cross-section: F, I, U, xi, ...
            pd.read_csv('data/final/pnad_calibration.csv').loc[0].items()}
    ctx |= dict(B_Y=ctx['B'] / ctx['Y'], Tr_w=ctx['Tr'] / ctx['w'],
                beta_low=ctx['beta_high'] - ctx['dbeta'])

    # A control sequence takes letters only, so LOG2_Y_F -> \mLOGTwoYF.
    name = lambda k: (prefix + ''.join(w[0].upper() + w[1:]
                                       for w in k.split('_'))).translate(DIGITS)
    val  = lambda k: format(ctx[k], MACRO_FMT.get(k, '.4g')).replace('%', r'\%')

    # Keys differing only in case ('I' vs 'i') map to one macro: keep the first.
    seen, rows = {}, []
    for k in sorted(ctx):
        if name(k) in seen:
            continue
        seen[name(k)] = k
        rows.append(rf'\newcommand{{\{name(k)}}}{{{val(k)}}}')

    text = "% Generated by p4_results.tex_macros() - do not edit.\n" + "\n".join(rows) + "\n"
    if savepath is None:
        return text
    with open(savepath, 'w', encoding='utf-8') as fobj:
        fobj.write(text)
