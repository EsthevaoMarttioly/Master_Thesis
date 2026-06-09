#### Thesis - Conditional Cash Transfers: A HANK Approach ####
## Author:  Esthevao Marttioly  |  EESP-FGV  |  2026
## Advisor: Bernardo Guimarães
#=
# ---------------------------------------------------------------------------
# DESCRIPTION
# This program solves an one-asset HANK model with 3-state formality-status.
#
# Solution method: Sequence-Space Jacobian (Auclert et al. 2021).
# https://github.com/shade-econ/sequence-jacobian
# ---------------------------------------------------------------------------
#=
# Write this in the terminal to install packages
# pip install -r requirements.txt


# Import Packages
import random
import time
import numpy as np

from sequence_jacobian import create_model


# Set a seed for future replications
random.seed(20260415)


# Import parameters
from code.parameters import calibration


# Import blocks
from code.household_block import hh
from code.other_blocks import *
from code.results import *


# ---------------------------------------------------------------------------
# Assemble Model

blocks_ss = [hh, firm_formal, firm_informal, nkpc_ss, monetary, fiscal, mkt_clearing]
hank_ss   = create_model(blocks_ss, name="HANK - Steady State")

print(f"\nModel inputs (SS):  {hank_ss.inputs}")
print(f"Model outputs (SS): {hank_ss.outputs}")


# ---------------------------------------------------------------------------
# Steady State
# 1. Unknown values to be estimated
unknowns_ss = {k: calibration[k] for k in
               ['beta_high', 'Z', 'G']}


# 2. Target asset market + goods market clearing
targets_ss = {
    'asset_mkt'  : 0,    # adjust beta_high to A = B
    'labor_mkt'  : 0,    # adjust Z to L = Y/Z = N_F
    'gov_budget' : 0,    # adjust G to balance govt budget
    # 'goods_mkt'  : 0,    # untargeted - Walras' Law
}

# 3. Steady State
start = time.time()
ss0 = hank_ss.solve_steady_state(calibration, unknowns = unknowns_ss,
                                 targets = targets_ss, solver = 'hybr')
print(f"Steady State solved in {time.time()-start:.1f}s")    # 27.0 seconds on my laptop



# ---------------------------------------------------------------------------
# Dynamics

blocks = [hh, firm_formal, firm_informal,
          phillips_curve, monetary, fiscal, mkt_clearing]
hank = create_model(blocks, name="HANK - Dynamics")

print(f"\nModel inputs (Dynamics):  {hank.inputs}")
print(f"Model outputs (Dynamics): {hank.outputs}")


# Verify ss0 is also a valid Steady State
ss = hank.steady_state(ss0)
for k in ss0.keys():
    assert np.all(np.isclose(ss[k], ss0[k])), f"SS mismatch at key {k}"
print("Steady State reached in dynamics DAG.")



# ---------------------------------------------------------------------------
# Steady State Diagnostics
var_ss_summary = ['Y', 'Y_I', 'C', 'beta_high', 'A', 'B',
                  'w', 'w_I', 'xi', 'Z', 'F', 'I', 'U', 'BF', 'Div',
                  'G', 'asset_mkt', 'goods_mkt', 'labor_mkt']

print_ss_summary(ss, calibration, var_ss_summary)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
# 1. Steady State Distribution and Policy Functions
lorenz_scf_raw = np.loadtxt('data/lorenz_nw_scf_2019.raw', delimiter=',')

plot_consumption_policy(ss, calibration, savepath='output/consumption_policy.png')
plot_wealth_distribution(ss, lorenz_scf_raw, savepath='output/wealth_distribution.png')



# 2. General Equilibrium Jacobians
T = 100
unknowns_dyn = ['Y', 'pi', 'w']
targets_dyn  = ['goods_mkt', 'nkpc', 'wage_nkpc']
inputs_dyn   = ['Tr']

G = hank.solve_jacobian(ss, unknowns_dyn, targets_dyn, inputs_dyn, T=T)


# 3. Partial Equilibrium Jacobians
G_hh = hh.jacobian(ss, inputs=['Tr', 'r'], T=T)     # hold r, w, Div fixed

plot_impc_profiles(G_hh, T_plot=30, savepath='output/impc_profiles.png')



# 4. Impulse Response Functions
# AR(1) shock with rho = 0.4, size = 1%
k = 4              # 1 year antecipation horizon
dTr0   = 0.01
rho_sh = 0.40

dTr_mit  = dTr0 * rho_sh ** np.arange(T)                # MIT Shock
dTr_ant  = np.concatenate([np.zeros(k), dTr_mit[:-k]])  # Antecipated Shock: Perfect Foresight
dTr_perm = dTr0 * np.ones(T)                            # Permanent Shock

variables = ['Y', 'C', 'I', 'U', 'pi', 'w']
irf_mit  = {v: G[v]['Tr'] @ dTr_mit  for v in variables}
irf_ant  = {v: G[v]['Tr'] @ dTr_ant  for v in variables}
irf_perm = {v: G[v]['Tr'] @ dTr_perm for v in variables}


plot_irfs(irf_mit, T_plot=30,
          title='GE IRFs: Conditional Transfer Targeted (Tr shock)',
          savepath='output/irf_Tr.png')

plot_irf_comparison({'MIT shock (surprise)'      : irf_mit,
                     f'Anticipated ({k}q ahead)' : irf_ant},
                     variables=['Y', 'C', 'I', 'pi'],
                     T_plot=30, savepath='output/irf_comparison.png')



# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
# Fiscal multipliers summary table
mults = compute_multipliers(G, {'Tr': dTr_mit})
print_multipliers(mults)



# ==========================================================================
# Estimate dbeta with a third target (HtM share)
# ==========================================================================
# Use scipy.optimize.root to compuet arbitrary distribution moments.

# def evaluate_calibration(x, calibration, htm_target = 0.20):
#     beta_high, dbeta = x[0], x[1]
#     ss_ = hh.steady_state(calibration | dict(beta_high=beta_high, dbeta=dbeta))

#     # --- Target 1: asset market (A = B)
#     res_A = ss_['A'] - calibration['B']

#     # --- Target 2: HtM share (match data)
#     D_     = ss_.internals['household']['D']
#     a_dist = D_.sum(axis=0)
#     htm    = a_dist[0]
#     res_HtM = htm - htm_target

#     return [res_A, res_HtM]

# # Fix Z from labor market first (or jointly)

# res = optimize.root(lambda x: evaluate_calibration(x, calibration), [unknowns_ss['beta_high'], 0.06])
# assert res.success, f"dbeta estimation failed: {res.message}"

# print(f"  Estimated beta_high = {res.x[0]:.4f}, dbeta = {res.x[1]:.4f}")

