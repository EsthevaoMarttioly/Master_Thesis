#=
#----------------------------------------------------------------------------
# DESCRIPTION
# The SMM Estimator: the objective, the search, identification and inference.
# ---------------------------------------------------------------------------
#=

# ---- Packages -------------------------------------------------------------
seed = 20260415

import os
import time
import pickle
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from code.p5_calibration import *
from code.p6_solve import solve_ss
from code.p8_results import _tex_table   # the only import from above


# ---------------------------------------------------------------------------
# 1. Transforms and Helperss
# ---------------------------------------------------------------------------
smm_grid = dict(nA=50, nE=11)     # SMM default
mom_flow = ['FI', 'IF']

def hank_model():
    # The steady-state model, rebuilt inside a worker process.
    from sequence_jacobian import create_model
    from code.p1_household import hh_ss
    from code.p2_other_blocks import (firm_formal, firm_informal, nkpc_ss, union_ss,
                                      equity_ss, monetary, fiscal, mkt_clearing, calibrate_ss)
    return create_model([hh_ss, firm_formal, firm_informal, nkpc_ss, union_ss,
                         equity_ss, monetary, fiscal, mkt_clearing, calibrate_ss])


def _to_x(p, space=smm_space):
    # Transform a parameter into the transformation
    x = []
    for k, (lo, hi, tr) in space.items():
        if   tr == 'log':   x.append(np.log(p[k]))
        elif tr == 'logit': x.append(np.log((p[k] - lo) / (hi - p[k])))
        else:               x.append(p[k])
    return np.array(x)


def _to_p(x, space=smm_space):
    # Invert the transformation back to the parameter
    p = {}
    for xi, (k, (lo, hi, tr)) in zip(x, space.items()):
        if   tr == 'log':   p[k] = float(np.clip(np.exp(xi), lo, hi))
        elif tr == 'logit': p[k] = float(lo + (hi - lo) / (1 + np.exp(-xi)))
        else:               p[k] = float(np.clip(xi, lo, hi))
    return p


def _simplex(x0, space=smm_space, frac=0.10):
    # NM's default edge is 5% of x: that is ~0 for a logit near a bound, or for any x = 0
    d = []
    for xi, (lo, hi, tr) in zip(x0, space.values()):
        if   tr == 'log':   g = np.exp(xi)                       # dp/dx
        elif tr == 'logit': p = lo + (hi - lo) / (1 + np.exp(-xi)); g = (p-lo) * (hi-p) / (hi-lo)
        else:               g = 1.0
        d.append(frac * (hi - lo) / max(g, 1e-8))
    return np.vstack([x0, x0 + np.diag(d)])


def _nm(obj, x0, maxiter, frac=0.10):
    # One simplex. Interruption (Ctrl-C) keeps the best point reached so far.
    try:
        x0 = minimize(obj, x0, args=(True,), method='Nelder-Mead',
                      options=dict(maxiter=maxiter, xatol=1e-3, fatol=1e-2, adaptive=True,
                                   initial_simplex=_simplex(x0, obj.space, frac))).x
    except KeyboardInterrupt:
        print(f"\n--- Interrupted at {obj.n} solves", flush=True)
    return obj.best              # NM can end on a vertex worse than one it already saw


def _weights(se=mom_se, floor=0.05):
    # Var(gap) = sampling + specification error (floor). W = S^-1, so it is efficient.
    S = np.diag([se[k] ** 2 + mom_fl.get(k, floor) ** 2 for k in mom_smm])
    return S, np.diag(1.0 / np.diag(S))


def _fj(J):           # J's notation:  59.3 | 223 | 1.0e+06
    return f"{J:.1f}" if J < 100 else (f"{J:.0f}" if J < 1000 else f"{J:.1e}")


def save_res(res, stage='f'):
    # `ss` does not pickle, so report() and inference() reload the rest.
    pd.Series(res['params'], name='value').to_csv(smm_path[stage], index_label='param')
    with open(smm_path[stage].replace('.csv', '.pkl'), 'wb') as fobj:
        pickle.dump({k: v for k, v in res.items() if k != 'ss'}, fobj)


def load_res(stage='f'):
    # Reload a finished run to report() or inference() in another session.
    with open(smm_path[stage].replace('.csv', '.pkl'), 'rb') as fobj:
        return pickle.load(fobj)



# ---------------------------------------------------------------------------
# 2. The Objective
# ---------------------------------------------------------------------------
class SMM:
    # Objective with caching and an evaluation log.
    def __init__(self, model, calib=None, space=smm_space, se=mom_se, coarse=None,
                 verbose=False, floor=0.05, logpath=None, every=25, tag='', warm=None):
        self.model, self.tag = model, tag
        self.cal0  = {k: v for k, v in (calibration if calib is None else calib).items()
                      if k not in ('Pi', 'Qb')}      # a stale Pi starts every solve far away
        self.space = space
        self.S, self.W = _weights(se, floor)
        self.coarse  = coarse or {}                 # e.g. smm_grid; {} is the full grid
        self.logpath, self.every = logpath, every   # checkpoint, a run takes hours
        self.verbose, self.log, self.n = verbose, [], 0
        self._warm = {k: v for k, v in (warm or {}).items() if k in (*update, 'Pi', 'Qb')}
        self._cache, self._best = {}, (np.inf, None)

    def gaps(self, p, coarse=True):
        # Run the model and get moments
        cal = {**self.cal0, **self._warm, **p, **(self.coarse if coarse else {})}
        ss  = solve_ss(self.model, cal, flows=flows, strict=False, damp=0.5,
                       maxit=300 if coarse else 400,               # damp=0.5 needs ~2.5x the its
                       tol=5e-5 if coarse else 1e-6, stol=1e-3 if coarse else 1e-4,
                       bmaxit=1000 if coarse else 5000)            # a slow point is a bad point
        cal.update({k: float(ss[k]) for k in update})
        self._warm = {**{k: cal[k] for k in update}, 'Pi': ss['Pi'], 'Qb': ss['Qb']}
        mod = model_moments(ss)
        return np.array([mod[k] - mom_data[k] for k in mom_smm]), mod, ss, cal

    def __call__(self, x, coarse=True):
        # Turn the moment vector g into g'Wg (minimization objective)
        key = tuple(np.round(x, 10))
        if key in self._cache: return self._cache[key]
        self.n += 1; t0 = time.time()
        p, why = _to_p(x, self.space), ''
        try:
            g, mod, *_ = self.gaps(p, coarse)
            J = float(g @ self.W @ g)
            if not np.isfinite(J): J, why = 1e6, 'non-finite'
        except Exception as e:
            mod, why = {}, str(e)
            J = 1e6             # Stalled solve
        self._cache[key] = J
        self.log.append({**p, **{f'm_{k}': mod.get(k, np.nan) for k in mom_smm},
                         'J': J, 'secs': time.time() - t0})
        if J < self._best[0]: self._best = (J, dict(p))
        if self.logpath and self.n % self.every == 0: self.history().to_csv(self.logpath)
        if self.verbose:
            print(f"  {self.tag} [{self.n:4d}] J={_fj(J):>7} best={_fj(self._best[0]):>7}  " +
                  " ".join(f"{k}={v:6.3f}" for k, v in p.items() if k in self.space) +
                  f"  ({time.time()-t0:3.0f}s) {why}")
        return J

    @property
    def best(self): return self._best[1]
    def history(self): return pd.DataFrame(self.log)



# ---------------------------------------------------------------------------
# 3. Search
# ---------------------------------------------------------------------------
def sobol_starts(n, space=smm_space, seed=seed):
    # n quasi-random starts, uniform inside the parameter box
    from scipy.stats import qmc
    lo, hi = np.array([[s[0], s[1]] for s in space.values()]).T
    u = 0.05 + 0.90 * qmc.Sobol(len(space), scramble=True, seed=seed).random(n)
    return [_to_x(dict(zip(space, lo + ui * (hi - lo))), space) for ui in u]


def screen(obj, n=4, pool=8):
    # Score `pool` Sobol draws on the coarse grid and keep the `n` best
    X  = sobol_starts(pool, obj.space)
    i0 = len(obj.log)                                  # this screen's rows, not the whole log
    J  = np.array([obj(x) for x in X])
    fi = pd.DataFrame(obj.log[i0:])['m_FI'].values     # a clipped flow is not a candidate
    out = np.abs(fi - mom_data['FI']) > mom_fl['FI']
    keep = np.lexsort((J, out))[:n]                    # feasible flows first, then J
    tbl  = pd.DataFrame([{**_to_p(X[k], obj.space), 'FI': fi[k], 'J': J[k]} for k in keep])
    if out[keep].any():
        print(f"  !! {int(out[keep].sum())} of {n} starts cannot match FI")
    tbl.to_csv(smm_path['s'], index_label='start')
    print("=== Starts ===\n", tbl.round(4).to_string())
    return [X[k] for k in keep]


def multistart(k=None, n=4, workers=4, calib=None, coarse=None, maxiter=150, frac=0.10):
    # One simplex per screened start, each in its own process, writing smm_nm{k}.csv.
    # k=None runs every start;  k=0,...,n-1 runs just that one.
    if k is None:
        os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '1'   # 1 per worker
        from functools import partial
        from concurrent.futures import ProcessPoolExecutor
        run = partial(multistart, n=n, calib=calib, coarse=coarse,
                      maxiter=maxiter, frac=frac)
        with ProcessPoolExecutor(workers) as ex:
            return list(ex.map(run, range(n)))

    t0  = time.time()
    obj = SMM(hank_model(), calib=calib, coarse=coarse, verbose=True,
              tag=f'{k}', logpath=smm_dir + f'smm_log{k}.csv')
    x0  = _to_x(pd.read_csv(smm_path['s'], index_col=0).iloc[k].to_dict(), obj.space)
    print(f"--- Start {k}/{n}: " +
          "  ".join(f"{a}={v:.3f}" for a, v in _to_p(x0, obj.space).items()), flush=True)
    _nm(obj, x0, maxiter, frac)
    pd.Series({**obj.best, 'J': obj._best[0]}, name='value').to_csv(
        smm_dir + f'smm_nm{k}.csv', index_label='param')
    obj.history().to_csv(obj.logpath)
    print(f"Start {k} done in {(time.time()-t0)/3600:.1f}hrs"
          f"   J = {_fj(obj._best[0])}   {obj.n} solves")
    return obj.best


def collect(obj, n=4, k=None):
    # Re-score every start on the full grid: the coarse J can rank a phantom first.
    # k=None scores and prints them;  k=0,...,n-1 saves that one as the start of polish.
    path = smm_path['m'].replace('.csv', '_all.pkl')
    out  = pickle.load(open(path, 'rb')) if os.path.exists(path) else {}
    par  = {j: {q: float(pd.read_csv(smm_dir + f'smm_nm{j}.csv', index_col=0)['value'][q])
                for q in obj.space}
            for j in range(n) if os.path.exists(smm_dir + f'smm_nm{j}.csv')}

    for j, pj in par.items():
        if out.get(j, {}).get('params') == pj: continue      # already scored on this point
        try:
            g, mod, ss, cal = obj.gaps(pj, coarse=False)
            out[j] = dict(params=pj, x=_to_x(pj, obj.space), J=float(g @ obj.W @ g),
                          g=g, moments=mod, ss=ss, calibration=cal)
        except RuntimeError as e:
            print(f"  start {j} does not solve on the full grid: {e}")
    with open(path, 'wb') as fobj:
        pickle.dump({j: {a: b for a, b in r.items() if a != 'ss'} for j, r in out.items()}, fobj)

    print("=== Multi-Start ===\n", pd.DataFrame(
        {j: {**r['params'], 'J_full': r['J'],                 # pi_max = 1.0: a flow clipped
             'pi_max': max(r['calibration'][q] for q in pi_calib)}
         for j, r in out.items()}).round(4).to_string())
    if k is None: return None                                 # pick with collect(obj, k=...)
    save_res(out[k], 'm')
    return out[k]


def polish(obj, p0, blocks=pre_block, coarse=None, maxiter=60, frac=0.03, pool=8):
    # Pre-SMM:  one solve per block of `blocks`, on the coarse grid, with `pool` random draws.
    # Polish:   one solve on the whole space, on the full grid, to estimate G, std errors and J.
    t0, keep, p = time.time(), obj.coarse, dict(p0)

    for name, b in (blocks or {}).items():
        sub = SMM(obj.model, calib={**obj.cal0, **p}, space={k: obj.space[k] for k in b},
                  coarse=smm_grid, verbose=obj.verbose, tag=name, warm=obj._warm)
        sub.S, sub.W = obj.S, obj.W                  # one J across every pass
        x = _to_x(p, sub.space)
        if pool: x = min([x, *sobol_starts(pool, sub.space)], key=sub)
        p.update(_nm(sub, x, maxiter))
        g, *_ = sub.gaps({q: p[q] for q in b})       # re-score: the warm start is the
        obj._warm = sub._warm                        # chosen point, not the last vertex
        print(f"--- Block [{name}]   J = {_fj(g @ obj.W @ g)}   {sub.n} solves"
              f"   ({(time.time()-t0)/60:.0f}min)\n    "
              + "  ".join(f"{k}={p[k]:.4f}" for k in b), flush=True)

    if blocks is not None: save_res(dict(params=p), 'm')
    print("\n--- Joint polish from:",
          " ".join(f"{k}={p[k]:.3f}" for k in obj.space), flush=True)
    obj.coarse = dict(coarse) if coarse else {}      # the full grid, unless asked otherwise
    obj._cache.clear(); obj._best = (np.inf, None)
    p = _nm(obj, _to_x(p, obj.space), maxiter, frac)

    g, mod, ss, cal = obj.gaps(p, coarse=False)      # score the estimate on the full grid
    res = dict(params=p, x=_to_x(p, obj.space), J=float(g @ obj.W @ g), g=g,
               moments=mod, ss=ss, calibration=cal)
    print(f"\nPolish done in {(time.time()-t0)/3600:.1f}hrs   "
          f"J = {_fj(res['J'])}   {obj.n} solves")
    save_res(res, 'f')
    obj.coarse = keep
    return res



# ---------------------------------------------------------------------------
# 4. Identification
# ---------------------------------------------------------------------------
MID = [i for i, k in enumerate(mom_smm) if k not in mom_flow]   # Flows match by construction
MOM = [mom_smm[i] for i in MID]
_drop_flows = lambda G, W: (G[MID], W[np.ix_(MID, MID)])        # Remove G rows for flows


def jacobian(obj, p_hat, step=0.05, coarse=False, tries=3):
    # G = dg/dtheta, shape (nMoments, nParams).
    at = lambda k, x: obj.gaps({**p_hat, k: x}, coarse)[0]
    g0, G = obj.gaps(p_hat, coarse)[0], []
    for k, (lo, hi, _) in obj.space.items():
        t0 = time.time()
        for i in range(tries):                     # shrink before giving up a side
            h  = step * (hi - lo) / 2 * 0.5 ** i
            pt = []
            for x in (max(p_hat[k] - h, lo), min(p_hat[k] + h, hi)):
                try: pt.append((x, at(k, x)))
                except RuntimeError: pt.append((p_hat[k], g0))    # one-sided: the centre
            if all(x != p_hat[k] for x, _ in pt): break
        (x0, g_0), (x1, g_1) = pt
        if x1 == x0:
            raise RuntimeError(f"\n     No steady state either side of {k}={p_hat[k]:.4f}")
        G.append((g_1 - g_0) / (x1 - x0))
        print(f"  [G] {k:8s} h={h:.4f}  {sum(x != p_hat[k] for x, _ in pt)}-sided"
              f"  ({(time.time()-t0)/60:.1f}min)", flush=True)
    return np.array(G).T


def sensitivity(G, W):
    # Andrews-Gentzkow-Shapiro: how far theta_j moves per unit of moment m.
    G, W = _drop_flows(G, W)
    S = -np.linalg.inv(G.T @ W @ G) @ G.T @ W
    print("=== Sensitivity ===\n",
          pd.DataFrame(S, index=list(smm_space), columns=MOM).round(3).to_string())


def identification(G, W):
    # Weighted Jacobian: a large condition number = a direction not identified.
    G, W = _drop_flows(G, W)
    _, s, vt = np.linalg.svd(np.sqrt(W) @ G)
    print(f"=== Identification ===   Condition Number {s[0] / s[-1]:.1f}\n",
          pd.DataFrame({'sing. value': s, 'weakest dir.': vt[-1]},
                       index=list(smm_space)).round(4).to_string())



# ---------------------------------------------------------------------------
# 5. Inference
# ---------------------------------------------------------------------------
def std_errors(G, W, S, n_obs=1):
    # Sandwich SEs. S = Var(g_data), in the same absolute units as g.
    GWG = np.linalg.inv(G.T @ W @ G)
    V = GWG @ (G.T @ W @ S @ W @ G) @ GWG / n_obs
    return np.sqrt(np.diag(V)), V


def jtest(g, G, W, S, n_obs=1):
    # Hansen over-identification test.
    from scipy.stats import chi2
    M  = np.eye(len(g)) - G @ np.linalg.inv(G.T @ W @ G) @ G.T @ W
    Om = M @ S @ M.T
    stat, df = n_obs * float(g @ np.linalg.pinv(Om) @ g), len(g) - G.shape[1]
    return stat, df, (1 - chi2.cdf(stat, df) if df > 0 else np.nan)


PARAM = {'mu_I': r'$\mu_I$', 'sigma_F': r'$\sigma_F$', 'sigma_I': r'$\sigma_I$',
         'rho_e': r'$\rho_e$', 'sd_e': r'$\sigma_e$', 'dbeta': r'$\Delta\beta$',
         'omega_I': r'$\omega_I$', 'sig': r'$\sigma_S$'}

def inference(res, obj, G=None, step=0.05, coarse=False, scale=True,
              savepath=None, label='tab:smm_se'):
    # Classical Minimum Distance: S is already Var(moment), so n_obs = 1. G on the full grid.
    from scipy.stats import norm
    from code.p8_results import _tex_table
    if G is None: G = jacobian(obj, res['params'], step=step, coarse=coarse)
    sensitivity(G, obj.W)                         # which moment moves which parameter
    i = np.ix_(MID, MID)
    se = std_errors(G[MID], obj.W[i], obj.S[i])[0]
    stat, df, pval = jtest(res['g'][MID], G[MID], obj.W[i], obj.S[i])
    k  = np.sqrt(stat / df) if scale else 1.0     # inflate the SE by the misspecification

    tbl = pd.DataFrame({'estimate': res['params'], 's.e.': dict(zip(obj.space, k * se)),
                        'lower': {k_: v[0] for k_, v in obj.space.items()},
                        'upper': {k_: v[1] for k_, v in obj.space.items()}})
    tbl['t'] = tbl['estimate'] / tbl['s.e.']
    tbl['p'] = 2 * norm.sf(np.abs(tbl['t']))
    tbl = tbl[['estimate', 's.e.', 't', 'p', 'lower', 'upper']]
    print(tbl.round(4).to_string())
    print(f"Hansen J = {stat:.3f}   df = {df}   p = {pval:.3f}   SE scale = {k:.2f}")

    rows = [f'{PARAM.get(n, n)} & {r.estimate:.3f} & {r["s.e."]:.3f} & {r.t:.2f} & {r.p:.3f}'
            f' & {r.lower:.2f} & {r.upper:.3f}' for n, r in tbl.iterrows()]
    rows += [r'\midrule', rf'Hansen $J$ & \multicolumn{{6}}{{c}}'
             rf'{{{stat:.2f}  (df {df}, $p$ = {pval:.3f})}}']
    _tex_table([r'Parameter & Estimate & S.E. & $t$ & $p$ & Lower & Upper'], rows,
               'SMM Estimates and Over-identification Test', label, 'lcccccc', savepath=savepath)



# ---------------------------------------------------------------------------
# Report
MOMENT = {'xi': r'$\xi$, hourly wage gap', 'sd_F': r'sd$(\log y_F)$',
          'sd_I': r'sd$(\log y_I)$', 'ac4': r'$\mathrm{Corr}_4(\log y)$',
          'dw_FI': r'$\mathbb{E}[\Delta \log y \mid F \to I]$',
          'dw_IF': r'$\mathbb{E}[\Delta \log y \mid I \to F]$',
          'gini': 'Wealth Gini', 'mpc': 'Aggregate MPC', 'htm': 'Hand-to-Mouth',
          'bot50': r'Wealth, Bottom 50\%', 'top10': r'Wealth, Top 10\%',
          'top1': r'Wealth, Top 1\%', 'ac1': r'$\mathrm{Corr}_1(\log y)$',
          'h_ratio': r'$h^I / h^F$', 'Tr_yF': r'$T / \mathbb{E}[y^F]$',
          'cut_FI': r'$P(\Delta \log y < 0 \mid F \to I)$',
          'cut_IF': r'$P(\Delta \log y < 0 \mid I \to F)$'}


def report(res, savepath=None, label='tab:moments'):
    # Print report and save to latex.
    _mname = lambda k: MOMENT.get(k) or\
    (rf'$q_{{{k[1:-2]}}}^{{{k[-1]}}}$' if k[0] == 'q' else k.replace('_', r'\_'))
    
    mod  = res['moments']
    keys = sorted((k for k in mom_data if k not in (*mom_fix, *mom_flow)),
                  key=lambda k: k not in mom_smm)       # stable sort: targeted first
    gap  = lambda k: (mom_data[k], mod[k], mod[k] - mom_data[k])

    jk = dict(zip(mom_smm, res['g'] ** 2 * np.diag(_weights()[1])))   # W is diagonal
    jk = {k: 100 * v / sum(jk.values()) for k, v in jk.items()}       # share of J, %
    share = lambda k: jk.get(k, np.nan)

    print("=== Moments (* = Untargeted) ===\n", pd.DataFrame(
        [(k if k in mom_smm else k + '*', *gap(k), share(k)) for k in keys],
        columns=['Moment', 'Data', 'Model', 'Gap', '%J']).set_index('Moment')
        .round(3).to_string(na_rep=''))

    n = sum(k in mom_smm for k in keys)              # the targeted block ends here
    rows = [rf'{_mname(k)} & {d:.3f} & {m:.3f} & {g:+.2f} & '
            + (f'{share(k):.0f}' if k in mom_smm else '')
            for k in keys for d, m, g in [gap(k)]]
    return _tex_table([r'Moment & Data & Model & Gap & \% $J$'],
                      rows[:n] + [r'\midrule'] + rows[n:],
                      'Targeted and Untargeted Moments', label, 'lcccc',
                      size=r'\footnotesize', savepath=savepath)



# ---------------------------------------------------------------------------
# 6. Entry Point
# ---------------------------------------------------------------------------
# One stage per call, in this order.
#   .\.venv\Scripts\Activate.ps1                  if you are not in venv
#   python -m code.p7_smm screen                  Sobol starts         -> smm_starts.csv
#   python -m code.p7_smm multistart              Nelder-Mead          -> smm_nm{k}.csv
#   python -m code.p7_smm collect [k]             re-score, pick k     -> smm_multi.csv
#   python -m code.p7_smm polish                  blocks, then joint   -> smm_final.csv
#   python -m code.p7_smm inference               G, J and the s.e.    -> smm_se.tex
if __name__ == '__main__':
    import sys
    stage, args = sys.argv[1], sys.argv[2:]
    obj = SMM(hank_model(), verbose=True, coarse=smm_grid if stage == 'screen' else None)

    if   stage == 'screen':     screen(obj)
    elif stage == 'multistart': multistart(coarse=smm_grid)
    elif stage == 'collect':    collect(obj, k=int(args[0]) if args else None)
    elif stage == 'polish':     report(polish(obj, load_res('m')['params']),
                                       savepath='output/tables/moments.tex')
    elif stage == 'inference':  inference(load_res('f'), obj, step=0.02,
                                          savepath='output/tables/smm_se.tex')
    else: raise SystemExit(f"Unknown stage: {stage}")

