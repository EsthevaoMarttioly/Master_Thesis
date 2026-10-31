#=
# ---------------------------------------------------------------------------
# DESCRIPTION
# Graphics and Tables in functions for main.py.
# ---------------------------------------------------------------------------
#=

# ---- Packages -------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from code.p5_calibration import *


# ---- Config ---------------------------------------------------------------
cov_path = 'data/final/pnad_bf_coverage.csv'
trs_path = 'data/final/pnad_transition_matrix.csv'
cal_path = 'data/final/pnad_calibration.csv'
inc_path = 'data/final/pnad_income_dist.csv'

PAL    = ('#1b325f', '#9cc4e4', '#e9f2f9', '#3a89c9', '#f26c4f', '#a8a3af')
STATES = {'F': 'Formal', 'I': 'Informal', 'U': 'Unemployed'}
LS     = {0: '-', 1: '--', 2: ':', 3: '-.'}
CYCLE  = [PAL[i] for i in (0, 3, 1, 4, 2)]
COL1, COL2, COL3, COL4, COL5, GRAY = PAL
DIGITS = str.maketrans(dict(zip('0123456789', ('Zero', 'One', 'Two', 'Three', 'Four',
                                               'Five', 'Six', 'Seven', 'Eight', 'Nine'))))

COLORS = {
    'Tr'   : COL1,        # Conditional Transfer
    'F'    : COL1,        # Formal Employment
    'I'    : COL4,        # Informal Employment
    'U'    : COL2,        # Unemployed
    'full' : COL1,        # Full Channel
    'ins'  : COL4,        # Insurance Channel
}

LABELS = {
    'Y'    : 'Output $Y$',
    'Y_I'  : 'Informal Output $Y_I$',
    'C'    : 'Consumption $C$',
    'N_F'  : 'Formal Labor Offer $N_F$',
    'N_I'  : 'Informal Labor Offer $N_I$',
    'F'    : 'Formal Share $\\alpha_F$',
    'I'    : 'Informal Share $\\alpha_I$',
    'U'    : 'Unemployment $\\alpha_U$',
    'BF'   : 'BF Recipients $BF$',
    'w'    : 'Real Wage $w$',
    'w_I'  : 'Informal Wage $w_I$',
    'pi'   : 'Inflation $\\pi$',
    'pi_w' : 'Wage Inflation $\\pi^w$',
    'pi_wI': 'Informal Wage Inflation $\\pi^{w_I}$',
    'r'    : 'Real Rate $r$',
    'i'    : 'Nominal Rate $i$',
    'B'    : 'Real Debt $B$',
    'A'    : 'Assets $A$',
    'G'    : 'Gov. Spending $G$',
    'tau_c': 'Consumption Tax $\\tau_c$',
    'Formal Share'     : r'Formal Share $\alpha_F$',
    'Informal Share'   : r'Informal Share $\alpha_I$',
    'Unemployed Share' : r'Unemployment $\alpha_U$',
    'xi'               : r'$\xi, \E[y_F]/\E[y_I]$',
    'C / Y'            : r'$C/Y$',
    'A / Y'            : r'$A/Y$',
    'HtM'              : r'Hand-to-Mouth',
    'Wealth Gini'      : r'Wealth Gini',
    'Bottom 50%'       : r'Bottom 50\%',
    'Top 10%'          : r'Top 10\%',
    'BF Spending / GDP': r'BF Spending / GDP',
    'Welfare E[V]'     : r'Welfare, CEV',
    'Tr / y_F'         : r'$T/y^F$',
}

# The channel names, in one place: rename here and every figure and table follows.
CHANNEL = {'ins' : 'Insurance Channel', 'full': 'Full Channel',
           'comp': 'Composition Channel', 'pft': 'Perfect CCT'}

SHORT = {'Informal Share': r'$\alpha_I$', 'Unemployed Share': r'$\alpha_U$',
         'xi': r'$\E[y_F]/\E[y_I]$', 'HtM': 'HtM', 'Wealth Gini': 'Gini'}

plt.rcParams.update({'font.size'        : 10,
                     'axes.spines.top'  : False,
                     'axes.spines.right': False,
                     'figure.dpi'       : 120})



# ---------------------------------------------------------------------------
# 0. Shared Helpers
# ---------------------------------------------------------------------------
_hh = lambda ss: ss.internals['household']
_reshape_hh = lambda arr, calib: arr.reshape(3, 2 * calib['nT'], 2, calib['nE'], -1)
_pct = lambda x, d=1, sign='': format(100 * x, f'{sign}.{d}f')

def _cell(x, delta=False, d=1):
    # Add a percent in ratio statistics, and sign + with `delta`.
    if x is None or not np.isfinite(x): return '--'
    return _pct(x, d, '+' if delta else '')


def _axis(ax, xlabel=None, ylabel=None, title=None,
          xlim=None, ylim=None, legend=False, **kw):
    # Axis scaffolding: labels, title, (0, x) limits, frameless legend.
    if xlabel is not None: ax.set_xlabel(xlabel)
    if ylabel is not None: ax.set_ylabel(ylabel)
    if title  is not None: ax.set_title(title)
    if xlim   is not None: ax.set_xlim(0, xlim)
    if ylim   is not None: ax.set_ylim(0, ylim)
    if legend: ax.legend(frameon=False, **kw)


def _save_or_show(fig, savepath):
    plt.tight_layout()
    if savepath is None: plt.show()
    else: fig.savefig(savepath, dpi=150, bbox_inches='tight'); plt.close(fig)


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
    if savepath is None: return tex
    with open(savepath, 'w', encoding='utf-8') as fobj:
        fobj.write(tex)



# ---------------------------------------------------------------------------
# 1. Steady State
# ---------------------------------------------------------------------------
vars_ss = ['Y', 'Y_I', 'C', 'F', 'L', 'psi',
           'beta_high', 'A', 'B', 'p_e', 'Div', 'BF', 'asset_mkt',
           'goods_mkt', 'labor_mkt', 'wage_nkpc', 'wage_nkpc_I', 'informal']

def print_ss_summary(ss, var_ss=vars_ss):
    print("\n" + "=" * 55)
    print("  STEADY STATE   (Model)")
    print("=" * 55)
    for k in var_ss:
        print(f"  {k:12s} = {ss[k]:.4f}")
        if k == 'beta_high':
            print(f"  {'beta_low':12s} = {ss['beta_high'] - ss['dbeta']:.4f}")
        if k == 'BF':
            print(f"  {'BF_F':12s} = {ss['BF_F_LF'] / ss['F']:.4f}")
            print(f"  {'BF_I':12s} = {ss['BF_I_LF'] / ss['I']:.4f}")
            print(f"  {'BF_U':12s} = {ss['BF_U_LF'] / ss['U']:.4f}")
    print("=" * 55)


def transition_table(ss, savepath=None, label='tab:transitions'):
    # Model vs PNAD quarterly transition matrix
    P    = _hh(ss)['P']
    d    = pd.read_csv(trs_path, index_col=0)
    d_ss = pd.read_csv(cal_path).loc[0]
    Pd, shd = d.loc[list(STATES), list(STATES)].to_numpy(), d_ss[list(STATES)].to_numpy()
    sh = np.array([ss[s] for s in STATES])

    head = [r' & \multicolumn{3}{c}{\textbf{Model}} & \multicolumn{3}{c}{\textbf{PNAD-C}}',
            r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}',
            r'Origin & $F$ & $I$ & $U$ & $F$ & $I$ & $U$']
    rows = [f'{n} & ' + " & ".join(_pct(x, 2) for x in np.r_[P[i], Pd[i]])
            for i, n in enumerate(STATES.values())]
    rows += ['\\midrule', 'Stationary Share & '
             + " & ".join(_pct(x, 2) for x in np.r_[sh, shd])]

    return _tex_table(head, rows, r'Quarterly Labor-Market Transitions (\%)',
                      label, 'lcccccc', savepath=savepath)



# ---------------------------------------------------------------------------
# 2. Distributions and Policy Functions
# ---------------------------------------------------------------------------
def _dens(x, w, grid):
    # Density of a discrete (value, mass) distribution: the atoms binned on `grid`,
    # read as bin centres. Not smoothed, so the steps of the state space stay visible.
    d = np.diff(grid)
    e = np.r_[grid[0] - d[0] / 2, grid[:-1] + d / 2, grid[-1] + d[-1] / 2]
    h = np.histogram(x, bins=e, weights=w / w.sum())[0]
    return h / np.diff(e)


def _lorenz(ss):
    # Cumulative population and wealth shares over the asset grid.
    a_d, a = _hh(ss)['D'].sum(0), _hh(ss)['a_grid']
    return (np.r_[0, np.cumsum(a_d)], np.r_[0, np.cumsum(a_d * a) / (a_d @ a)])


def _state_bar(ax, values, ylabel, title, names=(), colors=()):
    # Bar over the 3 employment states, optionally prefixed by extra categories.
    ax.bar(list(names) + list(STATES.values()), values,
           color=list(colors) + [COLORS[s] for s in STATES])
    _axis(ax, ylabel=ylabel, title=title)


def _by_state(D, x, block):
    # Mean of x within each of the 3 Employment Segments (x broadcasts over assets).
    seg = lambda i: slice(i * block, (i + 1) * block)
    xi  = lambda i: x[seg(i)] if x.shape[0] == D.shape[0] else x
    return [(D[seg(i)] * xi(i)).sum() / D[seg(i)].sum() for i in range(3)]


def _by_wealth(D, qidx, n_q, block):
    # Sector shares and BF coverage within each wealth quantile.
    sh, mass = np.full((3, n_q), np.nan), np.full(n_q, np.nan)
    for q in range(n_q):
        sel = qidx == q
        m   = np.array([D[i * block:(i + 1) * block, sel].sum() for i in range(3)])
        if m.sum() > 0:
            sh[:, q], mass[q] = m / m.sum(), m.sum()
    return sh, mass


def _wealth_bins(D, n_q):
    # Map each asset node to a quantile of the overall wealth distribution.
    cum = np.concatenate([[0.0], np.cumsum(D.sum(0))[:-1]])
    return np.minimum((cum * n_q).astype(int), n_q - 1)


def _bar_panel(ax, left, height, width, xlabel, title, vlines=(), xlim=None):
    # PDF Panel: Weighted Histogram + Labelled Vertical Markers.
    ax.bar(left, 100 * height, width=width, color=COL1, alpha=0.7, align='edge')
    for xv, lab, col, ls in vlines:
        ax.axvline(xv, color=col, ls=ls, label=lab)
    _axis(ax, xlabel=xlabel, ylabel='Households (%)', title=title, xlim=xlim, legend=bool(vlines))


# ---- Graphs ---------------------------------------------------------------
def _lorenz_panel(ax, pop, share, emp, emp_label, kind, ref=None):
    # Lorenz Panel: Model vs Data, Gini in the legend, Top Shares in a box.
    # `ref`: {'top10','top1','gini'}
    ax.plot(pop, share, color=COL1, lw=1.8,
            label=f'Model, Gini = {gini_from_lorenz(pop, share):.2f}')
    ax.plot([0, 1], [0, 1], color=COL2, ls=':', lw=1.8, label='Perfect Equality')
    p = np.linspace(0, 1, 501)
    L = np.interp(p, emp[:, 0], emp[:, 1])
    ax.plot(p, L, color=COL4, ls='--', lw=1.6,
            label=f'{emp_label}, Gini = {gini_from_lorenz(p, L):.2f}')

    ref = ref or {'top10': top_share(p, L, 0.10), 'top1': top_share(p, L, 0.01)}
    box = (f'Top 10% = {top_share(pop, share, 0.10):.0%}  (Data = {ref["top10"]:.0%})\n'
           f'Top 1%  = {top_share(pop, share, 0.01):.0%}  (Data = {ref["top1"]:.0%})')
    if 'gini' in ref:
        box += (f'\nGini    = {gini_from_lorenz(pop, share):.2f}  (Data = {ref["gini"]:.2f})')
    ax.text(0.97, 0.03, box, transform=ax.transAxes, ha='right', va='bottom',
            fontsize=8, bbox=dict(boxstyle='round', fc=COL3, ec=COL3, alpha=0.6))

    _axis(ax, xlabel='Cumulative Population Share', ylabel=f'Cumulative {kind} Share',
          title=f'Lorenz Curve ({kind})', legend=True, loc='upper left')


def plot_consumption_policy(ss, calibration, T_plot_a=10, savepath=None):
    # c(s, theta_median, e_median, a) for the three employment states, BF status b = 0.
    h      = _hh(ss)
    a_grid = h['a_grid']
    nT, nE = calibration['nT'], calibration['nE']
    c      = _reshape_hh(h['c'], calibration)[:, :nT]   # axis 1 is (b, theta): b = 0 block

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for bi, beta_name in enumerate(['Impatient', 'Patient']):
        for s, key in enumerate(STATES):
            axes[bi].plot(a_grid, c[s, nT // 2, bi, nE // 2, :],
                          color=COLORS[key], ls=LS[s], lw=1.8, label=STATES[key])
        _axis(axes[bi], xlabel='Assets $a$',
              ylabel='Consumption $c(s, \\bar{\\theta}, \\bar{e}, a)$',
              title=f'Policy Functions, Non-Recipients - {beta_name}',
              xlim=T_plot_a, ylim=4, legend=True)

    _save_or_show(fig, savepath)


def plot_income_distribution(ss, bins=51, lim=3.0, savepath=None):
    # Model vs PNAD earnings density, by sector.
    h = _hh(ss)
    ly_s = (h['log_y_f'] + h['log_y_i'])[:, 0]              # gross, a-invariant
    mass = h['D'].sum(1)
    blk  = ly_s.size // 3
    ly_s = ly_s - np.r_[np.full(blk, np.log(ss['h_F'])),    # hourly
                        np.full(blk, np.log(ss['h_I'])), np.zeros(ly_s.size - 2 * blk)]
    ref  = np.log(mass[:blk] @ np.exp(ly_s[:blk]) / mass[:blk].sum())

    d = pd.read_csv(inc_path)
    edges = np.linspace(-lim, lim, bins)
    grid  = 0.5 * (edges[:-1] + edges[1:])          # the same bins as the PNAD histogram
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    for ax, s, sl, col in zip(axes, STATES, [slice(0, blk), slice(blk, 2*blk)],
                              [COL1, COL4]):
        ly  = ly_s[sl] - ref
        mod = _dens(ly, mass[sl], grid)                 # the states are atoms
        g   = d[(d.group == s) & (d.wage > 0)]
        dat, _ = np.histogram(np.log(g.wage / pnad['w_F']), bins=edges,
                              weights=g.y * np.gradient(g.wage))
        c, bw = 0.5 * (edges[:-1] + edges[1:]), edges[1] - edges[0]
        ax.fill_between(grid, 100 * mod, color=col, alpha=0.35, step='mid')
        ax.plot(grid, 100 * mod, color=col, lw=1.6, marker='o', ms=3, label='Model')
        ax.plot(c, 100 * dat / dat.sum() / bw, color='black', lw=1.6, ls='--', label='PNAD')
        ax.axvline(0, color=GRAY, lw=0.8)
        _axis(ax, xlabel=r'$\log(w / \bar{w}^F)$', ylabel='Density (%)',
              title=f'{STATES[s]} Hourly Wage', legend=True)

    _save_or_show(fig, savepath)


def plot_wealth_distribution(ss, n_bins=30, savepath=None):
    # 2x2: Wealth and Income, each as a PDF and a Lorenz curve.
    h = _hh(ss)
    D, a_grid = h['D'], h['a_grid']

    a_dist = D.sum(0)
    aL_pop, aL_share = _lorenz(ss)

    # Earnings + Transfers, as in PNAD
    y = np.broadcast_to(h['y'][:, None], D.shape).ravel()
    m = D.ravel()
    o = np.argsort(y); ys, ms = y[o], m[o]
    yL_pop   = np.concatenate([[0], np.cumsum(ms) / ms.sum()])
    yL_share = np.concatenate([[0], np.cumsum(ms * ys) / np.sum(ms * ys)])

    edges = np.linspace(y.min(), y.max(), 101)
    y_hist, _ = np.histogram(y, bins=edges, weights=m)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # PNAD Lorenz
    d = pd.read_csv(inc_path)
    shares = alpha[:2] / alpha[:2].sum()
    x = np.linspace(0, d.wage.max(), 2001)
    f = sum(s * np.interp(x, g.wage, g.y, left=0, right=0)
            for s, (_, g) in zip(shares, d.groupby('group')))
    pop, inc = np.cumsum(f), np.cumsum(f * x)
    pnad_lorenz = np.column_stack([pop / pop[-1], inc / inc[-1]])

    # HtM: liquid wealth under a month of mean formal earnings
    thr = float(ss['N_F']) / float(ss['F']) / 3
    htm = float(a_dist[a_grid <= thr].sum())
    n_b = max(int(np.searchsorted(a_grid, 2.5 * thr)) + 1, n_bins)
    _bar_panel(axes[0, 0], a_grid[:n_b], a_dist[:n_b],
               np.diff(np.append(a_grid[:n_b], a_grid[n_b])),
               'Assets $a$', 'Wealth Distribution (Near Constraint)',
               vlines=[(a_grid[1], f'At Constraint = {a_dist[0]:.1%}', COL1, '--'),
                       (thr, f'HtM = {htm:.1%}', COL4, '--')])
    _lorenz_panel(axes[0, 1], aL_pop, aL_share,
                  np.loadtxt('data/lorenz_nw_scf_2019.raw', delimiter=','),
                  'US SCF 2019 (Proxy)', 'Wealth', ref=wid)   # shares: WID / UBS Brazil
    _bar_panel(axes[1, 0], edges[:-1], y_hist, np.diff(edges),
               'Income $y$', 'Income Distribution', xlim=10)
    _lorenz_panel(axes[1, 1], yL_pop, yL_share, pnad_lorenz, 'PNAD', 'Income')

    _save_or_show(fig, savepath)


def plot_bf_targeting(ss, lim=3.0, savepath=None):
    # Who the means test catches: tested income against the threshold, by sector.
    h = _hh(ss)
    D, a = h['D'], h['a_grid']
    blk  = D.shape[0] // 3

    y = np.exp((h['log_y_f'] + h['log_y_i'])[:, 0])        # labour income, a-invariant
    y[2 * blk:] = 0.0                                      # y^U is out of the means test
    yt = (y[:, None] + float(ss['r']) * a[None, :]) / float(ss['ybar'])

    grid = np.linspace(0, lim, 41)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, s_, sl, col in zip(axes, STATES, [slice(0, blk), slice(blk, 2 * blk)], [COL1, COL4]):
        x, w, b  = yt[sl].ravel(), D[sl].ravel(), h['bf'][sl].ravel()
        share    = (w * b).sum() / w.sum()                 # _dens renormalises: put it back
        tot, got = _dens(x, w, grid), share * _dens(x, w * b, grid)
        ax.fill_between(grid, 100 * tot, color=col, alpha=0.30, step='mid', label='All')
        ax.plot(grid, 100 * got, color=col, lw=2.0, marker='o', ms=3, label='Receives BF')
        ax.axvline(1.0, color=GRAY, ls='--', lw=1.2)
        _axis(ax, xlabel=r'Tested income $(y + ra) \,/\, \bar{y}$',
              title=f'{STATES[s_]}  ({100 * share:.1f}% covered)', legend=True)
        axes[2].plot(grid, np.where(tot > 0.01 * tot.max(), got / np.maximum(tot, 1e-12),
                                    np.nan), color=col, lw=2.0, marker='o', ms=3,
                     label=STATES[s_])
    if os.path.exists(cov_path):                               # PNAD: P(BF | wage, s)
        cov = pd.read_csv(cov_path)
        cov['x'] = cov['wage'] / (pnad['y_F'] * pnad['yrule'])  # Same unit: 1 is the rule
        for g, col in zip(('Formal', 'Informal'), [COL1, COL4]):
            e = cov[(cov['group'] == g) & (cov['x'] <= lim)]
            axes[2].plot(e['x'], e['y'], color=col, ls=':', lw=1.6, marker='o', ms=3.5)
        axes[2].plot([], [], color=GRAY, ls=':', lw=1.6, marker='o', ms=3.5, label='PNAD')
    axes[0].set_ylabel('Density (%)')
    axes[2].axvline(1.0, color=GRAY, ls='--', lw=1.2)
    _axis(axes[2], xlabel=r'Tested income $(y + ra) \,/\, \bar{y}$',
          ylabel=r'$P(\mathrm{BF})$', title='Coverage by tested income', legend=True)
    _save_or_show(fig, savepath)


def plot_descriptives(ss, n_q=5, savepath=None):
    # Consumption and Wealth by State, Formality by Wealth, and by Patience.
    h = _hh(ss)
    D, c, a_grid = h['D'], h['c'], h['a_grid']
    block = D.shape[0] // 3
    qidx  = _wealth_bins(D, n_q)
    x     = np.arange(n_q)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    _state_bar(axes[0, 0], _by_state(D, c, block),
               'Mean Consumption $c$', 'Consumption by Employment State')
    _state_bar(axes[0, 1], _by_state(D, a_grid[None, :], block),
               'Mean Assets $a$', 'Wealth by Employment State')

    # Formality and Unemployment by wealth quantile
    shares, mass = _by_wealth(D, qidx, n_q, block)
    bf_q = np.array([(D[:, qidx == q] * h['bf'][:, qidx == q]).sum() / mass[q]
                     for q in range(n_q)])
    for i, key in enumerate(STATES):
        axes[1, 0].bar(x + (i - 1) * 0.2, shares[i], width=0.2,
                       color=COLORS[key], label=STATES[key])
    axes[1, 0].plot(x, bf_q, color='black', lw=1.8, ls='--', marker='o', ms=5, label='Receives BF')
    _axis(axes[1, 0], ylabel='Share', title='Formality and Unemployment by Wealth', legend=True)

    # Formality by wealth quantile and patience
    D5 = _reshape_hh(D, ss)                       # (3, nF x nT, 2, nE, nA)
    for b, (lab, ls, mk) in enumerate([('Impatient', (0, (4, 2)), 's'),
                                       ('Patient', '-', 'o')]):
        Db = D5[:, :, b].sum((1, 2))              # (3, nA): one row per sector
        sh, _ = _by_wealth(Db, qidx, n_q, 1)
        for i, key in enumerate(('F', 'I')):
            axes[1, 1].plot(x, sh[i], color=COLORS[key], ls=ls, lw=2.0, marker=mk,
                            ms=6, mfc='white' if b == 0 else COLORS[key],
                            label=f'{STATES[key]}, {lab}')      # hollow square = impatient
    _axis(axes[1, 1], ylabel='Share', title='Formality by Wealth and Patience', legend=True, fontsize=8)

    for ax in (axes[1, 0], axes[1, 1]):
        ax.set_xticks(np.arange(n_q), [f'Q{i+1}' for i in range(n_q)])
        ax.set_xlabel('Wealth Quantile')

    _save_or_show(fig, savepath)



# ---------------------------------------------------------------------------
# 3. Welfare
# ---------------------------------------------------------------------------
def _horizon(ss):
    # E[sum beta^t] by state: Lam = 1 + beta Pi Lam. The log-CEV normalizer.
    b = _hh(ss)['beta']
    return np.linalg.solve(np.eye(b.size) - b[:, None] * _hh(ss)['Pi'], np.ones(b.size))


def _cev(v1, v0, eis, vc1, lam=None):
    # Consumption Equivalent Variation: gain in welfare.
    if eis == 1.0: return np.exp((v0 - v1) / lam) - 1    # log: (1+g) shifts u, not scales
    return ((v0 + vc1 - v1) / vc1) ** (1 / (1 - 1/eis)) - 1


def _welfare_gains(ss_bf, ss_nobf):
    # CEV from BF for [Impatient, Patient, Formal, Informal, Unemployed].
    # The ratio nets out the 1/(1-beta) scale: raw dE[V] is not comparable across types.
    eis = float(ss_bf['eis'])
    def ev(ss, key):
        h = _hh(ss)
        D, V = _reshape_hh(h['D'], ss), _reshape_hh(h[key], ss)
        pat = [(D[:, :, b] * V[:, :, b]).sum() / D[:, :, b].sum() for b in (0, 1)]
        sec = [(D[s] * V[s]).sum() / D[s].sum() for s in (0, 1, 2)]
        return np.array(pat + sec)
    return _cev(ev(ss_nobf, 'V'), ev(ss_bf, 'V'), eis, ev(ss_nobf, 'Vc'))


def plot_bf_comparison(ss, ss_nobf, n_q=5, savepath=None):
    # Who gains from BF, and how the two steady states differ.
    block = _hh(ss)['D'].shape[0] // 3
    x = np.arange(n_q)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Welfare gain by patience type and by sector
    _state_bar(axes[0, 0], 100 * _welfare_gains(ss, ss_nobf),
               'Welfare Gain (CEV, %)', 'Who gains from Bolsa Familia',
               names=('Impatient', 'Patient'), colors=(GRAY, GRAY))

    # Welfare gain by wealth: the distributional cut the table cannot show
    eis  = float(ss['eis'])
    qidx = _wealth_bins(_hh(ss_nobf)['D'], n_q)
    ev = lambda ss, key, sel: ((_hh(ss)['D'][:, sel] * _hh(ss)[key][:, sel]).sum()
                               / _hh(ss)['D'][:, sel].sum())
    D, lam = _hh(ss_nobf)['D'], _horizon(ss_nobf)
    lm  = lambda sel: float(D[:, sel].sum(1) @ lam) / D[:, sel].sum()
    cev_by_wealth = [_cev(ev(ss_nobf, 'V', qidx == q), ev(ss, 'V', qidx == q), eis,
                          ev(ss_nobf, 'Vc', qidx == q), lm(qidx == q)) for q in range(n_q)]

    axes[0, 1].bar(x, 100 * np.array(cev_by_wealth), color=COL4)
    _axis(axes[0, 1], ylabel='Welfare Gain (CEV, %)', title='Welfare Gain by Wealth')

    # Wealth concentration, both economies
    for lab, sx, col, ls in [('With BF', ss, COL1, '-'), ('No BF', ss_nobf, COL4, '--')]:
        pop, share = _lorenz(sx)
        axes[1, 0].plot(pop, share, color=col, ls=ls, lw=1.8,
                        label=f'{lab}, Gini = {gini_from_lorenz(pop, share):.2f}')
    axes[1, 0].plot([0, 1], [0, 1], color=COL2, ls=':', lw=1.6)
    _axis(axes[1, 0], xlabel='Cumulative Population Share',
          ylabel='Cumulative Wealth Share', title='Wealth Concentration', legend=True, loc='upper left')

    # Consumption by state, both economies
    for i, (lab, sx, col) in enumerate([('With BF', ss, COL1), ('No BF', ss_nobf, COL4)]):
        h = _hh(sx)
        axes[1, 1].bar(np.arange(3) + (i - 0.5) * 0.35,
                       _by_state(h['D'], h['c'], block), width=0.35, color=col, label=lab)
    axes[1, 1].set_xticks(np.arange(3), list(STATES.values()))
    _axis(axes[1, 1], ylabel='Mean Consumption $c$',
          title='Consumption by Employment State', legend=True)

    for ax in (axes[0, 0], axes[0, 1], axes[1, 1]):
        ax.axhline(0, color='k', lw=0.8)
    axes[0, 1].set_xticks(np.arange(n_q), [f'Q{i+1}' for i in range(n_q)])
    axes[0, 1].set_xlabel('Wealth Quantile')

    _save_or_show(fig, savepath)



# ---------------------------------------------------------------------------
# 4. PE iMPC and GE IRFs
# ---------------------------------------------------------------------------
def _irf_panels(variables, T_plot, draw, legend_ax=0,
                suptitle=None, xtick=None, titles=None, savepath=None):
    # Shared IRF grid: `draw(ax, var)` plots the lines; the rest is scaffolding.
    variables = list(variables)
    titles = titles or {}

    # Flattened axes grid: 1 row to 2 panels, 2 rows to 8, then 3.
    n = len(variables)
    nrows = 1 if n <= 2 else (2 if n <= 8 else 3)
    ncols = -(-n // nrows)
    fig, axes_ = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes       = np.array(axes_).flatten()

    for ax, v in zip(axes, variables):
        draw(ax, v)
        ax.axhline(0, color=GRAY, ls=':')
        _axis(ax, xlabel='Quarters', ylabel='% deviation from SS',
              title=titles.get(v, LABELS.get(v, v)), xlim=T_plot)
        if xtick:
            ax.xaxis.set_major_locator(mticker.MultipleLocator(xtick))
    for ax in axes[n:]:
        ax.set_visible(False)
    axes[legend_ax].legend(frameon=False)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    _save_or_show(fig, savepath)


def plot_impc(G_hh, h_ant=4, T_plot=20, key='C', savepath=None):
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
    _axis(ax, xlabel='Quarters after Transfer $t$', ylabel='Consumption Response',
          title=f'PE iMPC - Instantaneous vs Anticipated at $t={h_ant}$', legend=True)

    _save_or_show(fig, savepath)


def plot_irf(irf_dict, variables=('C', 'Y', 'Y_I', 'I', 'BF', 'pi', 'w', 'w_I'),
             title='CCT Shock', shock=None, T_plot=40, savepath=None):
    # One line per economy in `irf_dict`, one panel per variable.
    if shock is not None:
        dZ = shock[0]; pk = float(np.asarray(dZ)[np.argmax(np.abs(dZ))])
        msg = f'{pk * 40000:+.0f} bps' if shock[1] == 'rstar' else f'{pk:+.0%}'
        title = f'{title} ({msg})'
    def draw(ax, v):
        for i, (label, irf) in enumerate(irf_dict.items()):
            ax.plot(irf[v][:T_plot] * 100, color=CYCLE[i % len(CYCLE)],
                    ls=LS[i % 4], lw=2.2, label=label)
    _irf_panels(variables, T_plot, draw, legend_ax=0,
                suptitle=f'GE IRFs: {title}', xtick=5, savepath=savepath)


def plot_irf_decomposition(irf_ins, irf_full, irf_pe=None, irf_pft=None,
                           variables=('C', 'Y', 'Y_I', 'I', 'BF', 'pi', 'w', 'w_I'),
                           T_plot=40, savepath=None):
    # Insurance (Pi frozen) vs Full (Pi moving); shade the composition gap.
    x = np.arange(T_plot)
    def draw(ax, v):
        ins, full = irf_ins[v][:T_plot] * 100, irf_full[v][:T_plot] * 100
        if irf_pe is not None and v in irf_pe:
            ax.plot(x, irf_pe[v][:T_plot] * 100, color=GRAY, lw=1.6, ls='--',
                    label='Partial Eq. (Prices Fixed)')
        ax.plot(x, ins,  color=COLORS['ins'],  lw=2.2, label=CHANNEL['ins'])
        ax.plot(x, full, color=COLORS['full'], lw=2.2, label=CHANNEL['full'])
        ax.fill_between(x, ins, full, color=COLORS['full'], alpha=0.15, label=CHANNEL['comp'])
        if irf_pft is not None and v in irf_pft:          # the frictionless means test
            ax.plot(x, irf_pft[v][:T_plot] * 100, color=COL5, lw=2.0, ls=':',
                    label=CHANNEL['pft'])
    _irf_panels(variables, T_plot, draw, savepath=savepath)


def plot_irf_financing(irf_a, irf_b, labels=('Revenue-Neutral', 'Debt-Financed'),
                       variables=('C', 'Y', 'pi', 'w', 'r', 'B'), T_plot=40, savepath=None):
    # Two financing paths of the same shock. `labels` names them: the caller knows which.
    x    = np.arange(T_plot)
    both = [(irf_a, labels[0], COLORS['ins'], '-'), (irf_b, labels[1], COLORS['full'], '--')]
    def draw(ax, v):
        for irf, lab, col, ls in both:
            if v in irf:
                ax.plot(x, irf[v][:T_plot] * 100, color=col, ls=ls, lw=2.2, label=lab)
        if v == 'B':         # overlay tau_c on the same axis
            for irf, lab, _, ls in both:
                if 'tau_c' in irf:
                    ax.plot(x, irf['tau_c'][:T_plot] * 100, color=COL5, ls=ls,
                            lw=2.2, label=rf'$\tau_c$ ({lab})')
            ax.legend(frameon=False)
    _irf_panels(variables, T_plot, draw,
                titles={'B': r'Debt $B$ x Consumption Tax $\tau_c$'}, savepath=savepath)



# ---------------------------------------------------------------------------
# 5. Tables
# ---------------------------------------------------------------------------
def _ss_stats(ss):
    h = _hh(ss)
    D, a = h['D'], h['a_grid']
    a_dist = D.sum(0)
    pop, share = _lorenz(ss)
    gdp = float(ss['Y'] + ss['Y_I'])
    return {'Formal Share'       : ss['F'],
            'Informal Share'     : ss['I'],
            'Unemployed Share'   : ss['U'],
            'C / Y'              : ss['C'] / gdp,
            'A / Y'              : ss['A'] / (4 * gdp),             # annual GDP
            'xi'                 : (ss['w_I'] * ss['N_I'] / ss['I'] / ss['h_I'])
                                   / (ss['w'] * ss['N_F'] / ss['F'] / ss['h_F']),
            'MPC'                : ss['MPC_Y'] / ss['INC_L'],       # income-weighted
            'HtM'                : a_dist[a <= ss['N_F'] / ss['F'] / 3].sum(),
            'Wealth Gini'        : gini_coefficient(a, weights=a_dist),
            'Bottom 50%'         : 1 - top_share(pop, share, 0.50),
            'Top 10%'            : top_share(pop, share, 0.10),
            'BF / GDP'           : ss['Tr'] * ss['BF'] / gdp,
            'Tr / y_F'           : ss['Tr'] * ss['F'] / (ss['w'] * ss['N_F']),
            'Welfare E[V]'       : float(np.sum(D * h['V'])),
            '_Vc'                : float(np.sum(D * h['Vc'])),
            '_Lam'               : float(D.sum(1) @ _horizon(ss)),
            **{f'_{k}': float(ss[k]) for k in pi_calib}}


def compare_ss(economies, savepath=None, label='tab:counterfactuals'):
    # Table: one column per steady state in `economies`.
    SHORT = {'Perfect CCT': 'Pft CCT'}        # Delta header
    names = list(economies)
    stats = {n: _ss_stats(economies[n]) for n in names}
    ref, cf = stats[names[0]], names[1:]

    rows = [' & '.join([LABELS.get(q, q)] + [_cell(stats[n][q]) for n in names]
                       + [_cell(stats[n][q] - ref[q], delta=True) for n in cf])
            for q in ref if not q.startswith(('Welfare', '_'))]

    cev = [_cev(ref['Welfare E[V]'], stats[n]['Welfare E[V]'],
                float(economies[names[0]]['eis']), ref['_Vc'], ref['_Lam']) for n in cf]
    rows += [' & '.join([LABELS['Welfare E[V]']] + ['--'] * len(names)   # a level of V
                        + [_cell(c, delta=True) for c in cev])]       # says nothing

    cols = names + ([rf'$\Delta$ {SHORT.get(n, n)}' for n in cf] if len(cf) > 1 else [r'$\Delta$'])
    cap = (rf"Steady State: {names[1]} vs.\ {names[0]} (\%)" if len(cf) == 1 else
           rf"Steady State: {names[0]} and Counterfactuals (\%)")
    return _tex_table([' & ' + ' & '.join(cols)], rows, cap, label,
                      'l' + 'c' * len(cols), savepath=savepath)


def compare_bf_ss(ss_bf, ss_nobf, labels=('No BF', 'With BF'), savepath=None, label='tab:bf_ss'):
    # Two-economy case: the Tr = 0 counterfactual is the reference, Delta = With BF - No BF.
    return compare_ss({labels[0]: ss_nobf, labels[1]: ss_bf}, savepath=savepath, label=label)


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

    return _tex_table([f"Variable & Horizon & {CHANNEL['ins'].split()[0]}"
                       f" & {CHANNEL['full'].split()[0]} & {CHANNEL['comp'].split()[0]}"], rows,
                      f'Cumulative Response (\\%), Shock in ${shock}$', label, 'lccccc',
                      savepath=savepath)



# ---------------------------------------------------------------------------
# 6. LaTeX Macros
# ---------------------------------------------------------------------------
# Macro formats. Default 3 decimals.
PCT     = ['F', 'I', 'U', 'BF', 'BF_F', 'BF_I', 'BF_U',              # %, 1 decimal
           'BF_w', 'Tr_yF', 'top10', 'top1', 'rstar']
PCTZERO = ['tau_l', 'tau_d', 'tau_c', 'B_gdp', 'y_u', 'pe_gdp']      # %, 0 decimal
ZERO    = ['amin', 'amax', 'nT', 'nE', 'nA', 'T_BFF', 'y_F', 'y_I']  # Zero decimal
ONE     = ['h_F', 'h_I', 'w', 'Y', 'phi', 'eis', 'varphi', 'sig']    # One decimal
TWO     = ['lambda_BF', 'ybar', 'sig_BF', 'rho_F', 'rho_I',          # Two Decimals
           'kappa', 'kappa_w', 'kappa_wI', 'mu', 'mu_w', 'gini',
           'q', 'mpc', 'htm', 'alpha_I']

MACRO_FMT  = ({k: '.0%' for k in PCTZERO} | {k: '.1%' for k in PCT}
              | {k: '.0f' for k in ZERO} | {k: '.1f' for k in ONE} | {k: '.2f' for k in TWO})
MACRO_FMT |= {f'{p}_{k}': v for p in ('mod', 'dat') for k, v in tuple(MACRO_FMT.items())}
MACRO_FMT |= {f'{p}_{a}{b}': '.1%' for p in ('mod', 'dat') for a in STATES for b in STATES}

def tex_macros(ss, calibration, savepath=None, prefix='m'):
    # Create macros.tex, to use in the latex.
    P = _hh(ss)['P']
    d = pd.read_csv(trs_path, index_col=0)

    ctx  = {k: float(v) for k, v in calibration.items() if np.size(v) == 1}
    ctx |= {k: float(ss[k]) for k in ss.keys() if np.size(ss[k]) == 1}
    ctx |= {f'mod_{a}{b}': P[i, j] for i, a in enumerate(STATES) for j, b in enumerate(STATES)}
    ctx |= {f'dat_{a}{b}': d.loc[a, b] for a in STATES for b in STATES}
    ctx |= {f'dat_{k}': float(v) for k, v in pd.read_csv(cal_path).loc[0].items()}   # PNAD
    ctx |= {f'dat_{k}': float(v) for k, v in mom_data.items()}                       # targets, WID
    ctx |= {f'mod_{k}': float(v) for k, v in model_moments(ss).items()}              # model moments
    ctx |= dict(beta_low=ctx['beta_high'] - ctx['dbeta'])

    # A control sequence takes letters only, so LOG2_Y_F -> \mLOGTwoYF.
    alias = {'phi': 'phi_Taylor'}        # `phi` and `Phi` would both give \mPhi
    name = lambda k: (prefix + ''.join(w[0].upper() + w[1:]
                                       for w in alias.get(k, k).split('_'))).translate(DIGITS)
    val  = lambda k: format(ctx[k], MACRO_FMT.get(k, '.3f')).replace('%', r'\%')

    # Keys differing only in case ('I' vs 'i') map to one macro: keep the first.
    seen, rows = {}, []
    for k in sorted(ctx):
        if name(k) in seen: continue
        seen[name(k)] = k
        rows.append(rf'\newcommand{{\{name(k)}}}{{{val(k)}}}')

    text = "% Generated by p8_results.tex_macros() - do not edit.\n" + "\n".join(rows) + "\n"
    if savepath is None: return text
    with open(savepath, 'w', encoding='utf-8') as fobj:
        fobj.write(text)

