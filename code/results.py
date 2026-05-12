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
    'b'      : 'steelblue',    # unemployment benefit (U-targeted)
    'Tr'     : 'tomato',       # safety-net transfer (V-targeted)
    'Z'      : 'forestgreen',  # TFP shock
    'r'      : 'darkorange',   # interest rate
    'E'      : 'steelblue',    # employed HH
    'U'      : 'tomato',       # unemployed (UI)
    'V'      : 'forestgreen',  # needy (no UI)
}

LS = {           # linestyles for comparison plots
    0: '-',
    1: '--',
    2: ':',
    3: '-.',
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

def print_ss_summary(ss, calibration):
    """Print key steady-state moments and sanity checks."""
    f_, s_, Tb_ = calibration['f'], calibration['s'], calibration['Tb']
    U_theory = (f_ * Tb_ / (f_ * Tb_ + 1)) * s_ / (s_ + f_)
    V_theory = (1 / (f_ * Tb_ + 1)) * s_ / (s_ + f_)
    w_theory = ss['Z'] / calibration['mu']

    print("\n" + "="*55)
    print("  STEADY STATE")
    print("="*55)
    for k in ['Y', 'C', 'beta_high', 'A', 'B', 'w',
              'U', 'V', 'L', 'Div', 'asset_mkt', 'goods_mkt', 'Z']:
        print(f"  {k:12s} = {ss[k]:.4f}")

    print(f"\n  U*  theory = {100*U_theory:.2f}%,  model = {100*ss['U']:.2f}%")
    print(f"  V*  theory = {100*V_theory:.2f}%,  model = {100*ss['V']:.2f}%")
    print(f"  w*  = Z/mu = {w_theory:.4f},  model = {ss['w']:.4f}")

    print(f"\n  Govt budget residual:  {ss['deficit']:.2e}")
    print(f"  Labor mkt residual:    {(1-ss['U']-ss['V'])-ss['L']:.2e}")
    print(f"  NKPC residual:         {ss['nkpc']:.2e}")
    print(f"  NKWPC residual:        {ss['nkwpc']:.2e}")
    print("="*55)


# ---------------------------------------------------------------------------
# 2. Consumption Policy Functions
# ---------------------------------------------------------------------------

def plot_consumption_policy(ss, calibration, patient_type = 'P', T_plot_a=20, savepath=None):
    """Plot c(s, e_median, a) for all three employment states."""
    a_grid = ss.internals['household']['a_grid']
    c_pol  = ss.internals['household']['c']

    nE = calibration['nE']
    nA = calibration['nA']
    e_med = nE // 2            # median productivity index

    if patient_type == 'P':
        beta_idx = 1
        beta_name = 'Patient'
    elif patient_type == 'I':
        beta_idx = 0
        beta_name = 'Impatient'
    else:
        raise ValueError("Invalid patient_type. Use 'P' for patient or 'I' for impatient.")

    # Reshape to (nS=3, nBeta=2, nE, nA)
    c_3d = c_pol.reshape(3, 2, nE, nA)

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(a_grid, c_3d[0, beta_idx, e_med, :],
            color=COLORS['E'], label='Employed')
    ax.plot(a_grid, c_3d[1, beta_idx, e_med, :],
            color=COLORS['U'], linestyle='--', label='Unemployed (UI)')
    ax.plot(a_grid, c_3d[2, beta_idx, e_med, :],
            color=COLORS['V'], linestyle=':', label='Needy (no UI)')

    ax.set_xlabel('Assets $a$')
    ax.set_ylabel('Consumption $c(s, \\bar{e}, a)$')
    ax.set_title(f'Steady-State Consumption Policy Functions for {beta_name}')
    ax.set_xlim(0, T_plot_a)
    ax.set_ylim(0, 3)
    ax.legend(frameon=False)

    _save_or_show(fig, savepath)
    return fig, ax


# ---------------------------------------------------------------------------
# 3. Wealth Distribution
# ---------------------------------------------------------------------------

def plot_wealth_distribution(ss, lorenz_scf_raw=None, n_bins=60, savepath=None):
    """Two-panel figure: wealth PDF near constraint + Lorenz curve."""
    D      = ss.internals['household']['D']
    a_grid = ss.internals['household']['a_grid']

    a_dist = D.sum(axis=0)           # marginal distribution over assets
    htm    = a_dist[0]
    cum_pop  = np.cumsum(a_dist)
    cum_wlth = np.cumsum(a_dist * a_grid) / np.sum(a_dist * a_grid)

    ## SCF Lorenz Curve
    percentiles = np.arange(501) / 500
    lorenz_scf  = np.array([np.interp(pctl, lorenz_scf_raw[:, 0], lorenz_scf_raw[:, 1]) for pctl in percentiles])


    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: PDF near constraint
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
    axes[1].plot(percentiles, lorenz_scf, label='SCF 2019', linestyle='--', color='green')
    axes[1].set_xlabel('Cumulative population share')
    axes[1].set_ylabel('Cumulative wealth share')
    axes[1].set_title('Lorenz Curve - Wealth Distribution')
    axes[1].legend(frameon=False)

    _save_or_show(fig, savepath)
    return fig, axes


#----------------------------------------------------------------------------
# 4. PE Intertemporal MPC Profiles
# ---------------------------------------------------------------------------

def plot_impc_profiles(G_hh, T_plot=30, savepath=None):
    """Plot PE impact-MPC for b and Tr shocks.

    The iMPC profile G_hh['C']['x'][:T, 0] gives the response of C_t
    to a unit shock to x at t=0, holding all prices fixed.
    """
    iMPC_b  = G_hh['C']['b']
    iMPC_Tr = G_hh['C']['Tr']

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(iMPC_b[:T_plot, 0]  * 100, marker='s', ms=4,
            color=COLORS['b'],  label='Unemployment benefit $b$ (U-targeted)')
    ax.plot(iMPC_Tr[:T_plot, 0] * 100, marker='o', ms=4,
            color=COLORS['Tr'], linestyle='--',
            label='Safety-net transfer $Tr$ (V-targeted)')
    ax.axhline(0, color='gray', linestyle=':')

    ax.set_xlabel('Quarter $t$')
    ax.set_ylabel(r'$\partial C_t / \partial \mathrm{shock}_0 \;(\times 100)$')
    ax.set_title('PE Intertemporal MPC Profiles')
    ax.legend(frameon=False)

    _save_or_show(fig, savepath)
    return fig, ax


# ---------------------------------------------------------------------------
# 5. GE IRFs for a single shock
# ---------------------------------------------------------------------------

def plot_irfs(irf_dict, T_plot=30, title='', savepath=None):
    """Panel IRF plot for a single shock.

    irf_dict : dict  {variable_name: array of length T}
                     values already in levels (multiply by 100 for %-dev)
    T_plot   : int   number of quarters to display
    """
    variables = list(irf_dict.keys())
    n = len(variables)
    fig, axes = plt.subplots(2, n//2 + n%2, figsize=(4*(n//2 + n%2), 4*2))
    if n == 1:
        axes = [axes]
        
    axes_flat = axes.flatten()  

    VAR_LABELS = {
        'Y': 'Output $Y$', 'C': 'Consumption $C$',
        'U': 'Unemp. (UI) $U$', 'V': 'Needy $V$',
        'w': 'Real wage $w$', 'pi': 'Inflation $\\pi$',
        'pi_w': 'Wage inflation $\\pi^w$',
    }

    for ax, var in zip(axes_flat, variables):
        ax.plot(irf_dict[var][:T_plot] * 100, color='steelblue', linewidth=1.8)
        ax.axhline(0, color='gray', linestyle=':')
        ax.set_title(VAR_LABELS.get(var, var))
        ax.set_xlabel('Quarters')
        ax.set_xlim(0, T_plot)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(5))

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

    irfs_dict : dict  {label_str: irf_dict}
                      e.g. {'b shock': irf_b, 'Tr shock': irf_Tr}
    variables : list  variables to plot (one panel each)
    """
    VAR_LABELS = {
        'Y': 'Output $Y$', 'C': 'Consumption $C$',
        'U': 'Unemp. (UI) $U$', 'V': 'Needy $V$',
        'w': 'Real wage $w$', 'pi': 'Inflation $\\pi$',
    }

    n   = len(variables)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    if n == 1:
        axes = [axes]

    labels = list(irfs_dict.keys())
    col_cycle = [COLORS['b'], COLORS['Tr'], COLORS['Z'], COLORS['r']]

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
    mults = {}
    for shock, dz in shocks.items():
        dY = G['Y'][shock] @ dz
        dC = G['C'][shock] @ dz
        sz = dz[0]                  # shock size at impact
        mults[shock] = {h: (dY[h]/sz, dC[h]/sz) for h in horizons}
    return mults


def print_multipliers(mults, horizons=(0, 3, 7, 19)):
    """Print multiplier table."""
    H_LABELS = {0: 'Impact', 3: '1 year', 7: '2 years', 19: '5 years'}
    print("\n" + "="*55)
    print("  OUTPUT MULTIPLIERS  (dY/d shock_0)")
    print("="*55)
    header = f"  {'Shock':10s}" + "".join(f"  {H_LABELS.get(h, f't={h}'):>8s}" for h in horizons)
    print(header)
    for shock, hmap in mults.items():
        row = f"  {shock:10s}" + "".join(f"  {hmap[h][0]:>8.3f}" for h in horizons)
        print(row)
    print("="*55)


