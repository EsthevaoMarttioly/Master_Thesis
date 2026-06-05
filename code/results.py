#=
#----------------------------------------------------------------------------
# DESCRIPTION
# Plotting and reporting functions for the main model.
# All functions are self-contained and imported in main.py.
#----------------------------------------------------------------------------
#=

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

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
    'C'   : 'darkorange',
}


LS = {0: '-', 1: '--', 2: ':', 3: '-.'}


VAR_LABELS = {
    'Y'    : 'Output $Y$',
    'C'    : 'Consumption $C$',
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

def print_ss_summary(ss, calibration, var_ss = ['Y', 'C', 'beta_high', 'A', 'B']):
    """Print key steady-state moments and sanity checks."""
    w_theory = ss['Z'] / calibration['mu']

    print("\n" + "="*55)
    print("  STEADY STATE")
    print("="*55)
    for k in var_ss:
        print(f"  {k:12s} = {ss[k]:.4f}")
    print(f"  {'beta_low':12s} = {ss['beta_high'] - ss['dbeta']:.4f}")
    print(f"  w*  = Z/mu = {w_theory:.3f},  model = {ss['w']:.3f}")
    print(f"  F + I + U  = 1.000,  model = {ss['F'] + ss['I'] + ss['U']:.3f}")
    print("="*55)


# ---------------------------------------------------------------------------
# 2. Consumption Policy Functions
# ---------------------------------------------------------------------------

def plot_consumption_policy(ss, calibration, T_plot_a=20, savepath=None):
    """Plot c(s, e_median, a) for all three employment states."""
    a_grid = ss.internals['household']['a_grid']
    c_pol  = ss.internals['household']['c']

    nE = calibration['nE']
    nA = calibration['nA']
    e_med = nE // 2            # median productivity index

    # Reshape to (nS=3, nBeta=2, nE, nA)
    c_3d = c_pol.reshape(3, 2, nE, nA)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    for bi, beta_name in enumerate(['Impatient', 'Patient']):
        ax = axes[bi]
        ax.plot(a_grid, c_3d[0, bi, e_med, :],
                color=COLORS['F'], label='Formal')
        ax.plot(a_grid, c_3d[1, bi, e_med, :],
                color=COLORS['I'], linestyle='--', label='Informal')
        ax.plot(a_grid, c_3d[2, bi, e_med, :],
                color=COLORS['U'],   linestyle=':',  label='Unemployed')
        ax.set_xlabel('Assets $a$')
        ax.set_ylabel('Consumption $c(s, \\bar{e}, a)$')
        ax.set_title(f'Policy Functions - {beta_name}')
        ax.set_xlim(0, T_plot_a)
        ax.set_ylim(0, 5)
        ax.legend(frameon=False)

    _save_or_show(fig, savepath)
    return fig, ax


# ---------------------------------------------------------------------------
# 3. Wealth Distribution
# ---------------------------------------------------------------------------

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
 
    # Right: Lorenz curve
    axes[1].plot(cum_pop, cum_wlth, color='steelblue', label='Model')
    axes[1].plot([0, 1], [0, 1], color='gray', linestyle=':', label='Perfect equality')
    if lorenz_data is not None:
        pctls = np.arange(501) / 500
        lorenz_emp = np.array([np.interp(p, lorenz_data[:, 0], lorenz_data[:, 1])
                               for p in pctls])
        axes[1].plot(pctls, lorenz_emp, color='forestgreen',
                     linestyle='--', label='SCF 2019')
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
 
    G_hh['C']['Tr'][:T, 0]: response of C_t to a unit Tr shock at t=0,
    holding all prices (r, w, Div) fixed.
    """
    iMPC_Tr = G_hh['C']['Tr']
 
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
        dC = G['C'][shock] @ dz
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


