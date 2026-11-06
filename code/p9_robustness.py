#=
# ---------------------------------------------------------------------------
# DESCRIPTION
# Counterfactual policies and Robustness: re-solve the Steady State
# under another Bolsa Familia rule, or at alternative external parameters.
# ---------------------------------------------------------------------------
#=

# ---- Packages -------------------------------------------------------------
import os
import time
import pickle
import numpy as np

from code.p5_calibration import calibration, cf_keep
from code.p8_results import _ss_stats, _cev, _cell, _tex_table, plot_irf, LABELS, SHORT, COL1, COL2, COL5, GRAY


# ---- Config ---------------------------------------------------------------
rob_path = 'output/smm/robustness.pkl'     # cache a variant
dyn_path = 'output/smm/dyn_%s.pkl'         # one IRF per variant, 'moving' or 'fixed'

rob_lab = {'rstar': lambda v: rf'${100 * ((1 + v) ** 4 - 1):.0f}\%$'}
rob_sym = {'eis': r'\gamma', 'varphi': r'\varphi',
           'sig': r'\sigma_S', 'y_u': 'y^U', 'rstar': 'r^*', 'BF_w': r'T/wN',
           'kappa_w': r'\kappa_w', 'kappa_wI': r'\kappa_{w}^{I}', 'phi': r'\phi', 'pi': r'\pi'}


# ---------------------------------------------------------------------------
# 1. Counterfactual Policies
# ---------------------------------------------------------------------------
cf_grid  = {'No BF'      : {'lambda_BF': 0.0, 'BF_w': 0.0},
            'Perfect CCT': {'lambda_BF': 1.0, 'rho_F': 1.0, 'rho_I': 1.0}}

def counterfactuals(solve_fn, calib=None, grid=cf_grid):
    # One steady state per `grid`, the calibration it converged to.
    calib, out = dict(calib or calibration), {}
    for name, rule in grid.items():
        start = time.time()
        cal = {**calib, **rule}
        ss  = solve_fn(cal, counterfactual=True)
        cal.update({k: ss[k] for k in (*cf_keep, 'Pi', 'Qb')})
        out[name] = (ss, cal)
        print(f"  {name:15s} done  ({(time.time() - start)/60:.1f}min)", flush=True)
    return out



# ---------------------------------------------------------------------------
# 2. Robustness Tests
# ---------------------------------------------------------------------------
rob_grid = {'rstar'   : (1.04 ** 0.25 - 1, 1.06 ** 0.25 - 1),
            'eis'     : (0.70, 1.00),           # Risk Aversion: 1.4 and log
            'varphi'  : (0.25, 0.75),           # Frisch Elasticity
            'sig'     : (0.50, 0.90),           # Taste Dispersion
            'y_u'     : (0.01, 0.04)}           # Unemployment Income


def _stats(solve_fn, calib, nobf, s1=None):
    # Statistics of each economy, plus the CEV of removing BF. `s1` skips the solved economy.
    s1 = s1 if s1 and '_cal' in s1 else _ss_stats(solve_fn(calib))
    if not nobf: return s1
    s0 = _ss_stats(solve_fn({**calib, **s1['_cal'], 'lambda_BF': 0.0, 'BF_w': 0.0}, counterfactual=True))
    s1['CEV'] = _cev(s0['Welfare E[V]'], s1['Welfare E[V]'],
                     float(calib['eis']), s0['_Vc'], s0['_Lam'])
    return s1


def robustness(solve_fn, calib=None, grid=rob_grid, nobf=True, warm_ss=None,
               path=rob_path, damps=(0.0, 0.3, 0.5, 0.7), maxit=200, solve=True):
    # Solve every variant and cache it.  solve_fn(cal, **kw) -> ss.
    calib = dict(calib or calibration)
    if warm_ss is not None: calib |= {q: warm_ss[q] for q in ('Pi', 'Qb')}   # start from baseline
    out   = pickle.load(open(path, 'rb')) if os.path.exists(path) else {}
    if not solve: return out
    jobs  = [(None, None)] + [(k, v) for k, vs in grid.items() for v in vs]

    for k, v in jobs:
        start = time.time()
        key = 'base' if k is None else f'{k}={v:g}'
        if key in out and (not nobf or 'CEV' in out[key]): continue     # a missing CEV is only the no-BF solve
        cal = {**calib, **({} if k is None else {k: v})}
        for d in damps:                    # if didn't converge, try another 'damp'
            try:
                out[key] = _stats(lambda c, **kw: solve_fn(c, damp=d, maxit=maxit, warm=False, **kw), cal, nobf, out.get(key))
                print(f"  {key:16s} done  ({(time.time() - start)/60:.1f}min)" +
                      ('' if d == damps[0] else f'   (damp={d})'), flush=True)
                break
            except (RuntimeError, ValueError) as err:
                last = err
        else:
            print(f"  {key:16s} {last}", flush=True)
        with open(path, 'wb') as fobj: pickle.dump(out, fobj)
    return out


def bf_sizes(solve_fn, calib=None, mult=(0, 0.5, 1, 1.5, 2), path=rob_path, solve=True):
    # Re-solve over BF_w: calibrate_ss inverts Tr out of the ratio every iteration.
    calib = dict(calib or calibration)
    out   = pickle.load(open(path, 'rb')) if os.path.exists(path) else {}
    base  = float(calib['BF_w'])
    if not solve: return out

    for m in mult:
        start = time.time()
        key = f'BF_w={m * base:.4f}'
        if key in out: continue
        cal = {**calib, 'BF_w': m * base, **({'lambda_BF': 0.0} if m == 0 else {})}
        try:
            out[key] = _ss_stats(solve_fn(cal, counterfactual=True, warm=False))
            print(f"  {key:16s} done  ({(time.time() - start)/60:.1f}min)", flush=True)
        except (RuntimeError, ValueError) as err:
            print(f"  {key:16s} {err}", flush=True)
        with open(path, 'wb') as fobj: pickle.dump(out, fobj)

    v0 = out.get(f'BF_w={base:.4f}')               # CEV against today's size
    for m in mult:
        s = out.get(f'BF_w={m * base:.4f}')
        if s and v0: s['CEV'] = _cev(v0['Welfare E[V]'], s['Welfare E[V]'],
                                     float(calib['eis']), v0['_Vc'], v0['_Lam'])
    with open(path, 'wb') as fobj: pickle.dump(out, fobj)
    return out


# ---------------------------------------------------------------------------
# 3. Tables
# ---------------------------------------------------------------------------
rob_cols = ['Informal Share', 'Unemployed Share', 'C / Y', 'A / Y',
            'xi', 'HtM', 'Wealth Gini', 'MPC', 'Top 10%']


def _grid_table(data, rows, grid, caption, label, d=1, cev=True, rlab=None, note=None,
                savepath=None, size=r'\small\setlength{\tabcolsep}{5pt}'):
    # One row per statistic, one column per economy.
    rlab = rlab or (lambda q: LABELS.get(q, q))
    keys = ['base'] + [f'{k}={v:g}' for k, vs in grid.items() for v in vs]
    get  = lambda j, q: _cell(data.get(j, {}).get(q), delta=(q == 'CEV'), d=d)
    _val = lambda k, v: rob_lab.get(k, lambda x: f'${float(x)}$')(v)

    size_ = [len(vs) for vs in grid.values()]
    ends  = np.cumsum([3, *size_])                  # the label and Baseline come first
    head  = [' & Baseline & ' + ' & '.join(
                 rf'\multicolumn{{{n}}}{{c}}{{${rob_sym.get(k, k)}$}}' for k, n in zip(grid, size_)),
             ''.join(rf'\cmidrule(lr){{{a}-{b - 1}}}'      # a rule under each parameter
                     for a, b in zip(ends[:-1], ends[1:])),
             ' & & ' + ' & '.join(_val(k, v) for k, vs in grid.items() for v in vs)]
    body = [' & '.join([rlab(q)] + [get(j, q) for j in keys]) for q in rows]
    if cev: body += [' & '.join([r'CEV'] + [get(j, 'CEV') for j in keys])]
    return _tex_table(head, body, caption, label,
                      'lc' + '|'.join('c' * n for n in size_), size=size, savepath=savepath, note=note)


def robustness_table(rob, grid=rob_grid, rows=rob_cols,
                     cev=True, savepath=None, label='tab:robustness'):
    # Changing values of parameters, the baseline in the first column.
    return _grid_table(rob, rows, grid,
                       r'Steady State under Alternative Parameters (\%)',
                       label, cev=cev, size=r'\footnotesize\setlength{\tabcolsep}{4pt}',
                       savepath=savepath, rlab=lambda q: SHORT.get(q, LABELS.get(q, q)),
                       note='CEV in comparison to No-BF Counterfactual' if cev else None)


def bf_size_table(rob, calib=None, mult=(0, 0.5, 1, 1.5, 2),
                  rows=('Tr / y_F', *rob_cols), savepath=None, label='tab:robust_bf'):
    # One column per size of the program, today's size ruled off on both sides.
    base = float((calib or calibration)['BF_w'])
    keys = [f'BF_w={m * base:.4f}' for m in mult]
    i    = list(mult).index(1)
    get  = lambda j, q: '--' if (q == 'CEV' and j == keys[i]) else \
                        _cell(rob.get(j, {}).get(q), delta=(q == 'CEV'))

    head = [r'$T/wN$ (\%) & ' + ' & '.join(rf'${100 * m * base:.1f}$' for m in mult)]
    body = [' & '.join([LABELS.get(q, q)] + [get(j, q) for j in keys]) for q in rows]
    body += [' & '.join([r'CEV (\%)'] + [get(j, 'CEV') for j in keys])]
    return _tex_table(head, body, r"Steady State over the Size of Bolsa Fam\'ilia (\%)",
                      label, 'l' + 'c' * i + '|c|' + 'c' * (len(mult) - i - 1), savepath=savepath)


# ---------------------------------------------------------------------------
# 4. Dynamics
# ---------------------------------------------------------------------------
dyn_grid = {'kappa_wI': (calibration['kappa_wI'] / 2, 2 * calibration['kappa_wI']),
            'kappa_w' : (calibration['kappa_w'] / 2, 2 * calibration['kappa_w']),
            'phi'     : (1.05, 2.00)}      # Taylor Rule
dyn_col  = {'kappa_wI': COL5, 'kappa_w': COL2, 'phi': GRAY}


def dyn_robustness(build_fn, shock, dZ, grid=dyn_grid, moving=True, path=None, solve=True, **kw):
    # One IRF per variant, cached, for both the table and the figure.
    path = path or dyn_path % ('moving' if moving else 'fixed')
    out  = pickle.load(open(path, 'rb')) if os.path.exists(path) else {}
    if not solve: return out

    for k, v in [(None, None)] + [(k, v) for k, vs in grid.items() for v in vs]:
        key = 'base' if k is None else f'{k}={v:g}'
        if out.get(key) is not None: continue      # a failed variant (None) is kept, and retried
        try:
            out[key] = build_fn({} if k is None else {k: v})(shock, dZ, split=False, moving=moving, **kw)
            print(f"  {key:16s} done", flush=True)
        except (RuntimeError, ValueError) as err:
            out[key] = None
            print(f"  {key:16s} failed: {err}", flush=True)
        with open(path, 'wb') as fobj: pickle.dump(out, fobj)
    return out


def dyn_rob_table(dyn, grid=dyn_grid, rows=('C', 'Y', 'I', 'w', 'pi', 'BF'),
                  horizon=8, shock='i', savepath=None, label='tab:dyn_robustness'):
    # Cumulative response over `horizon` quarters, in % of the ss.
    cum = {j: ({x: float(np.sum(d[x][:horizon])) for x in rows} if d is not None else {}) for j, d in dyn.items()}
    return _grid_table(cum, rows, grid, f'Cumulative Response to a {shock} Shock '
                       rf'({horizon} Quarters, \% of SS)', label, d=3, cev=False,
                       rlab=lambda q: rf'$\sum {rob_sym.get(q, q)}$', savepath=savepath)


def plot_dyn_robustness(dyn, grid=dyn_grid, title='Monetary Policy Shock (i): Dynamic Robustness',
                        savepath=None, **kw):
    # IRF Robustness: one line per parameter and value.
    lines, style = {}, {}
    for k, vs in grid.items():
        for v, ls in zip(vs, ('-', '--')):
            if dyn.get(f'{k}={v:g}') is not None:
                lab = rf'${rob_sym.get(k, k)} = {v:g}$'
                lines[lab], style[lab] = dyn[f'{k}={v:g}'], (dyn_col[k], ls)
    lines['Baseline'], style['Baseline'] = dyn['base'], (COL1, '-')      # last, so it overlaps

    def fig_legend(fig, axes):      # baseline on top, then one column per parameter
        hs = [h for h in axes[0].get_lines() if not h.get_label().startswith('_')]
        kw = dict(frameon=False, loc='upper center')
        fig.legend(hs[-1:], [hs[-1].get_label()], bbox_to_anchor=(0.5, 0.0), **kw)
        fig.legend(hs[:-1], [h.get_label() for h in hs[:-1]], ncol=max(len(grid), 2),
                   bbox_to_anchor=(0.5, -0.04), **kw)

    return plot_irf(lines, variables=('C', 'Y', 'Y_I', 'N_I', 'pi', 'r', 'w', 'tau_c'),
                    title=title, savepath=savepath, style=style, fig_legend=fig_legend, **kw)
