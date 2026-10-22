#=
# ---------------------------------------------------------------------------
# DESCRIPTION
# Counterfactual policies and robustness: re-solve the Steady State under another
# Bolsa Familia rule, or at alternative external parameters.
# ---------------------------------------------------------------------------
#=

# ---- Packages -------------------------------------------------------------
import os
import pickle
import numpy as np

from code.p5_calibration import calibration, update
from code.p8_results import _ss_stats, _cev, _tex_table


# ---- Config ---------------------------------------------------------------
rob_path = 'output/smm/robustness.pkl'     # cache a variant

cf_grid  = {r"No Bolsa Fam\'ilia"  : {'lambda_BF': 0.0, 'BF_w': 0.0},
            'Perfect Means Test'  : {'lambda_BF': 1.0, 'rho_F': 1.0, 'rho_I': 1.0}}
cf_keep  = [*update, 'h_F', 'w_I', 'tau_c', 'Pi', 'Qb']   # counterfactual=True moves these

rob_grid = {'eis'     : (0.30, 0.70),      # EIS: log and 2.0 are run on their own
            'varphi'  : (0.25, 1.00),      # Frisch elasticity
            'alpha_I' : (0.75, 0.95),      # Informal returns to scale
            'sig'     : (0.30, 0.55),      # Taste dispersion
            'y_u'     : (0.00, 0.04)}      # Unemployment income

rob_cols = ['Informal Share', 'Unemployed Share', 'HtM', 'Wealth Gini',
            'Top 10%', 'Aggregate C', 'BF Spending / GDP']

rob_head = {'Informal Share': r'$\alpha_I$', 'Unemployed Share': r'$\alpha_U$',
            'HtM': 'HtM', 'Wealth Gini': 'Gini', 'Top 10%': r'Top 10\%',
            'Aggregate C': '$C$', 'BF Spending / GDP': 'BF/GDP'}

rob_pct = {'Informal Share', 'Unemployed Share', 'HtM', 'Top 10%',
           'Wealth Gini', 'BF Spending / GDP'}

rob_sym = {'eis': r'\sigma^{-1}', 'varphi': r'\varphi', 'alpha_I': r'\alpha_I',
           'sig': r'\sigma_S', 'y_u': 'y^U',
           'BF_w': r'\mathcal{T}/wN'}


# ---------------------------------------------------------------------------
# 1. Counterfactual Policies
# ---------------------------------------------------------------------------
def counterfactuals(solve_fn, calib=None, grid=cf_grid):
    # One steady state per rule in `grid`, each with the calibration it converged to.
    # solve_fn(cal, **kw) -> ss.  BF_w is held, so Tr rebalances: same budget, new rule.
    calib, out = dict(calib or calibration), {}
    for name, rule in grid.items():
        cal = {**calib, **rule}
        ss  = solve_fn(cal, counterfactual=True)
        cal.update({k: ss[k] for k in cf_keep})
        out[name] = (ss, cal)
        print(f"  {name:22s} done", flush=True)
    return out



# ---------------------------------------------------------------------------
# 2. Robustness Tests
# ---------------------------------------------------------------------------
def _stats(solve_fn, calib, nobf):
    # Statistics of one economy, plus the CEV of removing Bolsa Familia.
    s1 = _ss_stats(solve_fn(calib))
    if not nobf: return s1
    s0 = _ss_stats(solve_fn({**calib, 'lambda_BF': 0.0, 'BF_w': 0.0}, counterfactual=True))
    s1['CEV'] = _cev(s0['Welfare E[V]'], s1['Welfare E[V]'], float(calib['eis']), s0['_Vc'])
    return s1


def robustness(solve_fn, calib=None, grid=rob_grid, nobf=True, path=rob_path):
    """Solve every variant and cache it.  solve_fn(cal, **kw) -> ss.
    Re-run to fill in only what a previous call could not solve."""
    calib = dict(calib or calibration)
    out   = pickle.load(open(path, 'rb')) if os.path.exists(path) else {}
    jobs  = [(None, None)] + [(k, v) for k, vs in grid.items() for v in vs]

    for k, v in jobs:
        key = 'base' if k is None else f'{k}={v:g}'
        if key in out: continue
        try:
            out[key] = _stats(solve_fn, {**calib, **({} if k is None else {k: v})}, nobf)
            print(f"  {key:16s} done", flush=True)
        except (RuntimeError, ValueError) as err:
            print(f"  {key:16s} no steady state: {err}", flush=True)
        with open(path, 'wb') as fobj: pickle.dump(out, fobj)
    return out


def bf_sizes(solve_fn, calib=None, mult=(0, 0.5, 1, 1.5, 2), path=rob_path):
    # Re-solve over BF_w: calibrate_ss inverts Tr out of the ratio every iteration.
    calib = dict(calib or calibration)
    out   = pickle.load(open(path, 'rb')) if os.path.exists(path) else {}
    base  = float(calib['BF_w'])

    for m in mult:
        key = f'BF_w={m * base:.4f}'
        if key in out: continue
        cal = {**calib, 'BF_w': m * base, **({'lambda_BF': 0.0} if m == 0 else {})}
        try:
            out[key] = _ss_stats(solve_fn(cal, counterfactual=True))
            print(f"  {key:16s} done", flush=True)
        except (RuntimeError, ValueError) as err:
            print(f"  {key:16s} no steady state: {err}", flush=True)
        with open(path, 'wb') as fobj: pickle.dump(out, fobj)

    v0 = out.get(f'BF_w={0.0:.4f}')                        # CEV against no transfer
    for m in mult:
        s = out.get(f'BF_w={m * base:.4f}')
        if s and v0: s['CEV'] = _cev(v0['Welfare E[V]'], s['Welfare E[V]'],
                                     float(calib['eis']), v0['_Vc'])
    with open(path, 'wb') as fobj: pickle.dump(out, fobj)
    return out


# ---------------------------------------------------------------------------
# 3. Tables
# ---------------------------------------------------------------------------
def _cell(k, x):
    if x is None or not np.isfinite(x): return '--'
    return f'{100 * x:.1f}' if k in rob_pct else f'{x:.3f}'


def _body(rows, cols, cev):
    # One line per economy: the label is already LaTeX.
    out = []
    for name, s in rows:
        cells = [_cell(c, None if s is None else s.get(c)) for c in cols]
        if cev: cells.append('--' if s is None or 'CEV' not in s
                             else rf'{100 * s["CEV"]:+.2f}')
        out.append(' & '.join([name] + cells))
    return out


def robustness_table(rob, grid=rob_grid, cols=rob_cols, cev=True,
                     savepath=None, label='tab:robustness'):
    # One line per parameter value, the baseline first.
    rows = [('Baseline', rob.get('base'))]
    for k, vs in grid.items():
        rows += [(rf'$ {rob_sym.get(k, k)} = {v:g}$', rob.get(f'{k}={v:g}')) for v in vs]

    head = ['Parameter & ' + ' & '.join(rob_head.get(c, c) for c in cols)
            + (r' & CEV (\%)' if cev else '')]
    return _tex_table(head, _body(rows, cols, cev),
                      'Robustness: Steady State under Alternative Parameters',
                      label, 'l' + 'c' * (len(cols) + cev), savepath=savepath)


def bf_size_table(rob, calib=None, mult=(0, 0.5, 1, 1.5, 2), cols=rob_cols,
                  savepath=None, label='tab:robust_bf'):
    # The size of the program, in place of the sweep figure.
    base = float((calib or calibration)['BF_w'])
    rows = [(rf'${m * base:.4f}$' + (r' \,(baseline)' if m == 1 else
                                     r' \,(no BF)' if m == 0 else ''),
             rob.get(f'BF_w={m * base:.4f}')) for m in mult]

    head = [r'$\mathcal{T} / wN$ & ' + ' & '.join(rob_head.get(c, c) for c in cols)
            + r' & CEV (\%)']
    return _tex_table(head, _body(rows, cols, True),
                      r"Steady State over the Size of Bolsa Fam\'ilia",
                      label, 'l' + 'c' * (len(cols) + 1), savepath=savepath)


# ---------------------------------------------------------------------------
# 4. Dynamics
# ---------------------------------------------------------------------------
# kappa_wI, kappa_w, and phi move only Dynamics
dyn_grid = {'kappa_wI': (0.10, 0.60),      # Informal wage rigidity
            'kappa_w' : (0.05, 0.20),      # Formal wage rigidity
            'phi'     : (1.05, 2.00)}      # Taylor rule


def dyn_robustness(build_fn, shock, dZ, grid=dyn_grid, var=('C', 'Y', 'I', 'BF'),
                   horizon=8, savepath=None, label='tab:dyn_robustness'):
    """Cumulative response over `horizon` quarters under alternative parameters.
    build_fn(calib) -> the irf_builder closure of p6_solve."""
    rows, base = [], None
    for k, vs in [(None, (None,))] + list(grid.items()):
        for v in vs:
            name = 'Baseline' if k is None else rf'$ {rob_sym.get(k, k)} = {v:g}$'
            try:
                irf = build_fn({} if k is None else {k: v})(shock, dZ, split=False)
                cum = [float(np.sum(irf[x][:horizon])) for x in var]
            except (RuntimeError, ValueError) as err:
                print(f"  {name} failed: {err}");  cum = [np.nan] * len(var)
            if base is None: base = cum
            rows.append(' & '.join([name] + [f'{c:+.3f}' for c in cum]
                                   + [f'{100 * (c / b - 1):+.1f}' if np.isfinite(c) and b
                                      else '--' for c, b in zip(cum[:1], base[:1])]))

    head = [r'Parameter & ' + ' & '.join(f'$\\sum {x}$' for x in var)
            + r' & $\Delta C$ (\%)']
    return _tex_table(head, rows,
                      f'Robustness: Cumulative Response to a {shock} Shock '
                      f'({horizon} Quarters)', label, 'l' + 'c' * (len(var) + 1),
                      savepath=savepath)
