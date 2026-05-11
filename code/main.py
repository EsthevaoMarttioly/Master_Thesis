#### Thesis - Targeted vs Universal Fiscal Policy ####
## Author: Esthevao Marttioly
## Model: HANK two-assets with Unemployment
## Date: 2024-04-21

#=
#---------------------------------------------------------------------------
# DESCRIPTION
# This program solves a one-asset HANK model with unemployment using the
# sequence-space jacobian (SSJ) method.
# The model is a standard Aiyagari infinite horizon with NK sticky prices,
# unemployment and a rule for monetary policy.
#---------------------------------------------------------------------------
#=
# Reference to SSJ:
# https://github.com/shade-econ/sequence-jacobian
#
# Write this in the terminal this to install packages
# pip install -r requirements.txt


# Import Packages
import random
import time
import numpy as np
import matplotlib.pyplot as plt

from sequence_jacobian import create_model


# Set a seed for future replications
random.seed(20260415)


# Import parameters
from code.parameters import calibration, unknowns_ss


# Import blocks
from code.household_block import hh
from code.other_blocks import firm, price_nkpc, wage_nkpc, dividends, monetary, fiscal, mkt_clearing



#---------------------------------------------------------------------------
# Assemble Model

blocks = [hh, firm, price_nkpc, wage_nkpc, dividends, monetary, fiscal, mkt_clearing]
hank   = create_model(blocks, name="HANK with Unemployment")

print(f"\nModel inputs:  {hank.inputs}")
print(f"Model outputs: {hank.outputs}")


# Steady State - let labor market untargeted, to robustly check the result
targets_ss  = {'asset_mkt': 0,
               #'goods_mkt': 0,
               'nkpc': 0}

start = time.time()
ss = hank.solve_steady_state(calibration, unknowns = unknowns_ss,
                             targets = targets_ss, solver = 'hybr')
print("Elapsed = %s seconds" % (time.time() - start))    # 6.0 seconds on my laptop


#---------------------------------------------------------------------------
## Display results
print("Steady-state:")
for k in ['Y', 'C', 'beta_high', 'A', 'B',
          'U', 'L', 'V', 'asset_mkt', 'goods_mkt', 'div']:
    print(f"  {k:15s} = {ss[k]:.4f}")



# Analytical targets
UV_theory = calibration['s'] / (calibration['s'] + calibration['f'])
U_theory = (calibration['f'] * calibration['Tb'] /
            (1 + calibration['f'] * calibration['Tb'])) * UV_theory
V_theory = (1 / (1 + calibration['f'] * calibration['Tb'])) * UV_theory
w_theory = calibration['Z'] / calibration['mu']
 
print(f"\n  U* theory (general) = {U_theory:.4f},  model = {ss['U']:.4f}")
print(f"  V* theory (general) = {V_theory:.4f},  model = {ss['V']:.4f}")
print(f"  w* = Z/mu           = {w_theory:.4f},  model = {ss['w']:.4f}")
print(f"\n  Govt budget residual : {ss['deficit']:.2e}")
print(f"  Labor mkt residual   : {(1-ss['U']-ss['V'])-ss['L']:.2e}")
print(f"  NKPC residual        : {ss['nkpc']:.2e}")
print(f"  Asset mkt residual   : {ss['asset_mkt']:.2e}")
print(f"  Goods mkt residual   : {ss['goods_mkt']:.2e}")



#---------------------------------------------------------------------------
# Wealth Distribution

## SCF Lorenz Curve
lorenz_scf_raw = np.loadtxt('data/lorenz_nw_scf_2019.raw', delimiter=',')

percentiles = np.arange(501) / 500
lorenz_scf = np.array([np.interp(pctl, lorenz_scf_raw[:, 0], lorenz_scf_raw[:, 1]) for pctl in percentiles])


## Model Lorenz Curve
D      = ss.internals['household']['D']
a_grid = ss.internals['household']['a_grid']

nE   = calibration['nE']
nA   = calibration['nA']


# Sum over beta and e within each s-block
n_s      = 2 * nE                                 # nBeta * nE per s-block
a_dist_E = D[:n_s].sum(axis=0)                    # employed
a_dist_U = D[n_s:2*n_s].sum(axis=0)               # unemployed (UI)
a_dist_V = D[2*n_s:].sum(axis=0)                  # needy
a_dist   = a_dist_E + a_dist_U + a_dist_V         # total

htm_share = a_dist[0]
print(f"\nHtM share (a = amin): {htm_share:.4f}  ({htm_share*100:.1f}%)")

cum_pop  = np.cumsum(a_dist)
cum_wlth = np.cumsum(a_dist * a_grid) / np.sum(a_dist * a_grid)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(a_grid[:60], a_dist[:60],
            width=np.diff(np.append(a_grid[:60], a_grid[60])),
            color='steelblue', alpha=0.7, align='edge')
axes[0].set_xlabel('Assets $a$'); axes[0].set_ylabel('Mass of households')
axes[0].set_title('Wealth Distribution (near constraint)')
axes[0].axvline(a_grid[0], color='tomato', linestyle='--',
                label=f'HtM = {htm_share:.1%}')
axes[0].legend()
 
axes[1].plot(cum_pop, cum_wlth, color='steelblue', label='Model')
axes[1].plot([0, 1], [0, 1], color='gray', linestyle=':', label='Equality')
axes[1].plot(percentiles, lorenz_scf, label='SCF 2019', linestyle='dashed', color='green')
axes[1].set_xlabel('Cumulative population share')
axes[1].set_ylabel('Cumulative wealth share')
axes[1].set_title('Lorenz Curve — Wealth Distribution')
axes[1].legend()
 
plt.tight_layout()
plt.savefig('output/wealth_distribution.png', dpi=150)
plt.show()



#---------------------------------------------------------------------------
# Consumption Policy Function
c_pol  = ss.internals['household']['c'].reshape(3, 2, calibration['nE'], calibration['nA'])

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(a_grid, c_pol[0, 1, nE//2, :], color='steelblue',
        label='Employed (median $e$, patient)')
ax.plot(a_grid, c_pol[1, 1, nE//2, :], color='tomato', linestyle='--',
        label='Unemployed — UI (median $e$, patient)')
ax.plot(a_grid, c_pol[2, 1, nE//2, :], color='forestgreen', linestyle=':',
        label='Needy — no UI (median $e$, patient)')
ax.set_xlabel('Assets $a$')
ax.set_ylabel('Consumption $c(s,e,a)$')
ax.set_title('Steady-State Consumption Policy Function')
ax.set_xlim(0,30)
ax.set_ylim(0,5)
ax.legend()
plt.tight_layout()
plt.savefig('output/consumption_policy.png', dpi=150)
plt.show()



# ---------------------------------------------------------------------------
# General Equilibrium Jacobians
# Unknowns: {Y, pi}    Targets: {goods_mkt, nkpc}
# (Add 'w' / 'nkwpc' once wage PC is fully activated)
# ---------------------------------------------------------------------------
T = 100

unknowns_dyn = ['Y', 'pi']
targets_dyn  = ['goods_mkt', 'nkpc']
inputs_dyn   = ['b', 'Tr', 'Z']          # fiscal shocks + TFP

G = hank.solve_jacobian(ss, unknowns_dyn, targets_dyn, inputs_dyn, T=T)



# ---------------------------------------------------------------------------
# IRFs: Unemployment-Benefit Shock (employment-targeted)
# ---------------------------------------------------------------------------
rho_sh = 0.40
db     = 0.01 * rho_sh ** np.arange(T)

irf_b = {v: G[v]['b'] @ db for v in ['Y', 'C', 'U', 'V']}

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
labels = {'Y': 'Output $Y$', 'C': 'Consumption $C$',
          'U': 'Unemp. (UI) $U$', 'V': 'Needy $V$'}

for i, var in enumerate(['Y', 'C', 'U', 'V']):
    axes[i].plot(irf_b[var][:40] * 100, color='steelblue',
                 label='Benefit $b$ shock')
    axes[i].axhline(0, color='gray', linestyle=':')
    axes[i].set_title(labels[var])
    axes[i].set_xlabel('Quarters')
    axes[i].set_xlim(0, 30)

axes[0].set_ylabel('% deviation from SS')
plt.suptitle('GE IRFs: Employment-Targeted (benefit shock)', fontsize=11)
plt.tight_layout()
plt.legend()
plt.savefig('output/irf_targeted.png', dpi=150)
plt.show()



# ---------------------------------------------------------------------------
# IRFs: Universal Transfer Shock (Tr shock to all V-type / or add T_univ)
# ---------------------------------------------------------------------------
# A truly universal transfer would be added as a separate input to
# labor_income (e.g. 'T_univ' paid to ALL households regardless of status).
# For now, Tr targets only V-type (needy) households.
dTr = 0.01 * rho_sh ** np.arange(T)
irf_Tr = {v: G[v]['Tr'] @ dTr for v in ['Y', 'C']}

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for i, var in enumerate(['Y', 'C']):
    axes[i].plot(irf_b[var][:40]*100, color='steelblue', label='$b$ (employment-targeted)')
    axes[i].plot(irf_Tr[var][:40]*100, color='tomato', linestyle='--', label='$Tr$ (needy-targeted)')
    axes[i].axhline(0, color='gray', linestyle=':')
    axes[i].set_title(labels[var])
    axes[i].set_xlabel('Quarters')
    axes[i].set_xlim(0, 30)

axes[0].set_ylabel('% deviation from SS')
plt.suptitle('GE IRFs: Targeted vs. Needy-Targeted Transfers', fontsize=11)
plt.tight_layout()
plt.legend()
plt.savefig('output/irf_comparison.png', dpi=150)
plt.show()



# ---------------------------------------------------------------------------
# PE Intertemporal MPC profiles
# ---------------------------------------------------------------------------
G_hh  = hh.jacobian(ss, inputs=['Tr', 'b', 'r', 'w'], T=T)
iMPC_b  = G_hh['C']['b']
iMPC_Tr = G_hh['C']['Tr']

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(iMPC_b[:30, 0]  * 100, marker='s', ms=4, color='steelblue',
        label='UI benefit $b$ (U-targeted)')
ax.plot(iMPC_Tr[:30, 0] * 100, marker='o', ms=4, color='tomato',
        label='Safety-net $Tr$ (V-targeted)')
ax.axhline(0, color='gray', linestyle='--')
ax.set_xlabel('Quarter $t$')
ax.set_ylabel(r'$\partial C_t / \partial \mathrm{shock}_0\ (\times 100)$')
ax.set_title('PE Intertemporal MPC Profiles')
ax.legend()
plt.tight_layout()
plt.savefig('output/impc_profiles.png', dpi=150)
plt.show()



# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
print(f"\n=== Key moments ===")
print(f"  HtM share:                    {htm_share:.3f}  ({htm_share*100:.1f}%)")
print(f"  Unemployment rate U (UI):     {ss['U']:.3f}  ({ss['U']*100:.1f}%)")
print(f"  Needy rate V (no UI):         {ss['V']:.3f}  ({ss['V']*100:.1f}%)")
print(f"  Total unemployment U+V:       {ss['U']+ss['V']:.3f}")
print(f"  Firm dividends D = Y - wL:    {ss['div']:.4f}")
print(f"  PE impact MPC — b:            {iMPC_b[0,0]:.4f}")
print(f"  PE impact MPC — Tr:           {iMPC_Tr[0,0]:.4f}")
print(f"  GE output mult. — b (impact): {G['Y']['b'][0,0]:.4f}")
print(f"  GE output mult. — Tr(impact): {G['Y']['Tr'][0,0]:.4f}")