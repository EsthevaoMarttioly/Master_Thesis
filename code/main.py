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
from code.other_blocks import firm, pricing, monetary, fiscal, mkt_clearing



#---------------------------------------------------------------------------
# Assemble Model

blocks = [hh, firm, pricing, monetary, fiscal, mkt_clearing]
hank   = create_model(blocks, name="One-Asset HANK with Unemployment")

print(f"\nModel inputs:  {hank.inputs}")
print(f"Model outputs: {hank.outputs}")


# Steady State
targets_ss  = {'asset_mkt': 0,
               #'goods_mkt': 0,
               'nkpc': 0}   # let labor market untargeted, to robustly check the result

start = time.time()
ss = hank.solve_steady_state(calibration, unknowns = unknowns_ss,
                             targets = targets_ss, solver = 'hybr')
print("Elapsed = %s seconds" % (time.time() - start))    # 3.7 seconds on my laptop



#---------------------------------------------------------------------------
## Display results
print("Steady-state:")
for k in ['Y', 'C', 'beta_high', 'w', 'r', 'pi',
          'L', 'U', 'Tr', 'mu']:
    print(f"  {k:15s} = {ss[k]:.4f}")



# Sanity checks
u_theory = calibration['s'] / (calibration['s'] + calibration['f'])
w_theory = calibration['Z'] / calibration['mu']
print(f"  U*:  theory={u_theory:.4f},  model={ss['U']:.4f}")             # ok
print(f"  w*:  theory={w_theory:.4f},  model={ss['w']:.4f}  (Z/mu)")     # ok
print(f"  Govt budget residual: {ss['Tr']+ss['BenefCost']+ss['r']*ss['B']-ss['Tax']:.2e}")    # ok
print(f"  Labor mkt residual:   {(1-ss['U'])-ss['L']:.2e}")              # ok

print(f"NKPC: {ss['nkpc']:.2e}")                          # ok
print(f"Asset market clearing: {ss['asset_mkt']:.2e}")    # ok
print(f"Goods market clearing: {ss['goods_mkt']:.2e}")    # ok



#---------------------------------------------------------------------------
# Consumption Policy Function
a_grid = ss.internals['household']['a_grid']
c_pol  = ss.internals['household']['c'].reshape(2, calibration['nE'], calibration['nA'])


fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(a_grid, c_pol[0, 3, :], color='steelblue',
        label='Employed (median $e$)')
ax.plot(a_grid, c_pol[1, 3, :], color='tomato', linestyle='--',
        label='Unemployed (median $e$)')
ax.set_xlabel('Assets $a$')
ax.set_ylabel('Consumption $c(s,e,a)$')
ax.set_title('Steady-State Consumption Policy Function')
ax.set_xlim(0,30)
ax.set_ylim(0,5)
ax.legend()
plt.tight_layout()
plt.savefig('output/consumption_policy.png', dpi=150)
plt.show()



#---------------------------------------------------------------------------
# Wealth Distribution
# HtM share and wealth concentration - motivation for targeting.

D = ss.internals['household']['D']
a_dist_employ = D[0].sum(axis=0)
a_dist_unempl = D[1].sum(axis=0)
a_dist = a_dist_employ + a_dist_unempl


# HtM = households at the borrowing constraint
htm_share = a_dist[0] + a_grid[1]
print(f"\nHtM share (a = amin): {htm_share:.3f}  ({htm_share*100:.1f}%)")  # 2.8%

 
# Wealth shares: bottom 50%, middle 40%, top 10%
cum_pop  = np.cumsum(a_dist)
cum_wlth = np.cumsum(a_dist * a_grid) / np.sum(a_dist * a_grid)
 
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
 
# Left: wealth PDF (zoom near constraint)
axes[0].bar(a_grid[:60], a_dist[:60], width=np.diff(np.append(a_grid[:60], a_grid[60])),
            color='steelblue', alpha=0.7, align='edge')
axes[0].set_xlabel('Assets $a$')
axes[0].set_ylabel('Mass of households')
axes[0].set_title('Wealth Distribution (near constraint)')
axes[0].axvline(a_grid[1], color='tomato', linestyle='--', label=f'HtM = {a_grid[1]:.1%}')
axes[0].legend()
 
# Right: Lorenz curve
axes[1].plot(np.cumsum(a_dist), cum_wlth, color='steelblue', label='Model')
axes[1].plot([0, 1], [0, 1], color='gray', linestyle=':', label='Perfect equality')
axes[1].set_xlabel('Cumulative population share')
axes[1].set_ylabel('Cumulative wealth share')
axes[1].set_title('Lorenz Curve — Wealth Distribution')
axes[1].legend()
 
plt.tight_layout()
plt.savefig('output/wealth_distribution.png', dpi=150)
plt.show()



#---------------------------------------------------------------------------
# General Equilibrium Jacobians
# {Y, pi} adjust to satisfy {goods_mkt=0, nkpc=0}
# Inputs: fiscal shocks {b, T} and real shock {Z}

T = 100

unknowns = ['Y', 'pi']            # variables that adjust
targets = ['goods_mkt', 'nkpc']   # equilibrium conditions
inputs   = ['b', 'T', 'Z']        # exogenous variables

G = hank.solve_jacobian(ss, unknowns, targets, inputs, T = T)



#---------------------------------------------------------------------------
# IRFs: Emploment-Targeted vs Universal Transfer
#   db: persistent 1% rise in b

rho_sh = 0.40
db = 0.01 * rho_sh ** np.arange(T)   # benefit shock with AR(1) persistence

irf_b = {v: G[v]['b'] @ db for v in ['Y', 'C', 'U']}

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
labels = {'Y': 'Output $Y$', 'C': 'Consumption $C$', 'U': 'Unemployment $U$'}

for i, var in enumerate(['Y', 'C', 'U']):
    axes[i].plot(irf_b[var][:T] * 100, color='steelblue',
                 label='Employment-targeted ($b$ increase)')
    axes[i].axhline(0, color='gray', linestyle=':')
    axes[i].set_title(labels[var])
    axes[i].set_xlabel('Quarters')
    axes[i].set_xlim(0, 30)
    axes[i].legend(fontsize=8)
 
axes[0].set_ylabel('% deviation from SS')
plt.suptitle('GE IRFs: Employment-Targeted', fontsize=11)
plt.tight_layout()
plt.savefig('output/irf_targeting.png', dpi=150)
plt.show()
 


#---------------------------------------------------------------------------
# 12. PE iMPC PROFILES
# Partial-equilibrium Jacobians: hold r, w fixed, shock only the benefit.
# These isolate the DIRECT household response before GE amplification.

G_hh = hh.jacobian(ss, inputs=['T', 'b', 'r', 'w'], T=T)

iMPC_b = G_hh['C']['b']


fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(iMPC_b[:30, 0] * 100, marker='s',
        ms=4, color='steelblue', label='Unemployment benefit $b$')
ax.axhline(0, color='gray', linestyle='--')
ax.set_xlabel('Quarter $t$')
ax.set_ylabel(r'$\partial C_t / \partial \mathrm{shock}_0\ (\times 100)$')
ax.set_title('PE Intertemporal MPC Profiles')
ax.legend()
plt.tight_layout()
plt.savefig('output/impc_profiles.png', dpi=150)
plt.show()


print(f"  HtM share:                    {htm_share:.3f}  ({htm_share*100:.1f}%)")
print(f"  SS unemployment rate:         {ss['U']:.3f}  ({ss['U']*100:.1f}%)")
print(f"  PE impact MPC — b:            {iMPC_b[0,0]:.3f}")
print(f"  GE output mult. - b (impact): {G['Y']['b'][0,0]:.3f}")

# 2.8% HtM Share, 20% Unemployment, and PE Impact = GE Impact = 0.073

