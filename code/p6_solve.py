#=
#----------------------------------------------------------------------------
# DESCRIPTION
# Solve the model: endogenous sector transition, steady state, dynamics,
# and the SMM estimator.
# ---------------------------------------------------------------------------
#=

# ---- Packages -------------------------------------------------------------
import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from sequence_jacobian.classes import SteadyStateDict

from code.p1_household import make_egrid, make_bgrid, nS, nB, _HH_WARM
from code.p5_calibration import *


# ---------------------------------------------------------------------------
# 1. Endogenous Sector Transition
# ---------------------------------------------------------------------------
def _softmax(Vals, sig):
    # Turn "pick the best option" into smooth probabilities (sig -> 0 = hard max).
    V = np.stack(Vals, 0)
    Probs = np.exp((V - V.max(0)) / np.maximum(sig, 1e-10))
    return Probs / Probs.sum(0)


def _status_probs(Vst, Vaj, p, probF, probI):
    # Transition Matrix Pi_{s,theta} 3*nT x 3*nT for each (beta, e, a).
    nS, nT, nA = Vst.shape
    sig = p['sig'] * Vaj.reshape(nS*nT, nA)      # taste shock in consumption units
    piF = np.repeat([p['pi_F'], p['pi_F'], p['pi_UF']], nT)[:, None]
    piI = np.repeat([p['pi_I'], p['pi_I'], p['pi_UI']], nT)[:, None]
    delta = np.repeat([p['delta_F'], p['delta_I'], 0.0], nT)
    keep = 1 - delta

    EV_F = probF @ Vst[F]     # Expected Value of Formal Sector
    EV_I = probI @ Vst[I]     # Expected Value of Informal Sector

    Vstay = Vst.reshape(nS*nT, nA)
    EVF = np.broadcast_to(EV_F, (nS*nT, nA))
    EVI = np.broadcast_to(EV_I, (nS*nT, nA))

    cF = _softmax([Vstay, EVF], sig)              # Only a Formal Offer
    cI = _softmax([Vstay, EVI], sig)              # Only an Informal Offer
    cB = _softmax([Vstay, EVF, EVI], sig)         # Both Offers

    a_stay = (1-piF) * (1-piI) + piF * (1-piI) * cF[0] +\
                (1-piF) * piI * cI[0] + piF * piI * cB[0]
    a_F    = piF * (1-piI) * cF[1] + piF * piI * cB[1]
    a_I    = (1-piF) * piI * cI[1] + piF * piI * cB[2]

    ar = np.arange(nS*nT)
    P = np.zeros((nS*nT, nS*nT, nA))

    P[ar, ar, :]             += keep[:, None] * a_stay
    P[:, F*nT:(F+1)*nT, :]   += keep[:, None,None] * a_F[:,None,:] * probF[None,:,None]
    P[:, I*nT:(I+1)*nT, :]   += keep[:, None,None] * a_I[:,None,:] * probI[None,:,None]
    P[ar, U*nT + ar % nT, :] += delta[:, None]

    return P


def build_Pi(V, Va, D, p, Pi_b, Pi_e, probF, probI):
    # Assemble the full transition matrix:  s (x) theta (x) beta (x) e.
    # `V` is (nS, nT, nBeta, nE, nA):  the value of each sector at (beta,e,a).
    nT, nE, nA = p['nT'], Pi_e.shape[0], V.shape[1]
    Vr  = V.reshape(nS*nT, nB, nE, nA)
    Var = Va.reshape(nS*nT, nB, nE, nA)
    Dr  = D.reshape(nS*nT, nB, nE, nA)

    Pstat = np.empty((nB, nE, nS*nT, nS*nT))    # Pstat[beta, e, from, to]
    flow  = np.zeros((nS*nT, nS*nT))            # mass flowing from -> to

    for _beta in range(nB):
        for _e in range(nE):
            P = _status_probs(Vr[:, _beta, _e, :].reshape(nS, nT, nA),
                              Var[:, _beta, _e, :], p, probF, probI)
            w = Dr[:, _beta, _e, :]
            flow += np.einsum('mna,ma->mn', P, w)              # Weight by mass
            tot = w.sum(1, keepdims=True)
            w = np.where(tot > 1e-14, w / np.where(tot > 1e-14, tot, 1.0), 1.0 / nA)
            Pstat[_beta, _e] = np.einsum('mna,ma->mn', P, w)   # Average over assets

    # Order:   s (x) theta (x) beta (x) e
    # Pi[(s,t,b,e),(s',t',b',e')] = Pstat[b,e,s,s'] * Pi_b[b,b'] * Pi_e[e,e']
    Pi = np.einsum('beMN,bB,eE->MbeNBE', Pstat, Pi_b, Pi_e)
    Pi = Pi.reshape(nS * nT * nB * nE, nS * nT * nB * nE)

    # Sector Transition F/I/U, collapsing theta
    flow = flow.reshape(nS, nT, nS, nT).sum((1, 3))
    return Pi, flow / flow.sum(1)[:, None]


# ---------------------------------------------------------------------------
# 2. Steady State
# ---------------------------------------------------------------------------
# Initial Guess
unknowns = dict(beta_high = 0.98, psi = 0.2, L = 0.5, tau = 0.1, Tr = 0.2, B = 3.2)

def solve_ss(hank_block, calib, flows=None, verbose=False, bgain=0.005, gain=1.0,
             smax=0.3, tol=1e-8, stol=1e-3, atol=1e-6, maxit=400):
    # Solve the Steady-State by iterating the value function and the transition matrix.
    # Clear Markets by analytical forms, but beta_high clears A = B by secant.
    start = time.time()
    # Import Grids
    c = {**unknowns, **calib}
    c['tau_ss'], c['B_ss'] = c['tau'], c['B']
    nE, nA, nT = c['nE'], c['nA'], c['nT']

    _, Pi_e, _, _, probF, _, probI =\
        make_egrid(c['rho_e'], c['sd_e'], nE, c['amin'],
                   c['amax'], nA, c['sigma_F'],
                   c['mu_I'], c['sigma_I'], nT)

    _, Pi_b = make_bgrid(c['beta_high'], c['dbeta'],
                         c['omega_I'], c['q'], nE, nT)

    # Initial guess for Pi.
    diff, dpi, dA, dbf = 0.1, 0.0, 0.0, 0.0; x0 = f0 = xb = fb = None
    Pi, _ = build_Pi(np.zeros((nS*nT*nB*nE, nA)), np.ones((nS*nT*nB*nE, nA)),
                     np.ones((nS*nT*nB*nE, nA)), c, Pi_b, Pi_e, probF, probI)

    for it in range(maxit):
        # Guess Pi -> Solve -> Read V -> Rebuild Pi -> Repeat until Pi converges.
        c['Pi'] = Pi
        ss = SteadyStateDict(c); ss.update(hank_block.steady_state(c))

        # Market Clearing: analytical forms + solver to asset_mkt (secant)
        if 'L_hat' in ss:
            c['L'], c['psi'], c['Tr'] = float(ss['L_hat']), float(ss['psi_hat']), float(ss['Tr_hat'])
            c['tau'] = c['tau_ss'] = float(ss['tau_hat'])
            c['B']   = c['B_ss']   = float(ss['B_hat'])
            xn, fn = c['beta_high'], float(ss['asset_mkt'])  # A - B, dA/dbeta > 0
            if xb is not None and abs(fn - fb) > 1e-12:
                g = (xn - xb) / (fn - fb)     # dbeta/dA > 0; if not, Pi moved A
                if g > 0: bgain = float(np.clip(g, 1e-4, 0.05))
            xb, fb, dA = xn, fn, abs(fn)
            c['beta_high'] = float(np.clip(xn - np.clip(bgain*fn, -0.02, 0.02), 0.5, 0.999))

        hhi = ss.internals['household']
        Pi_new, P_s = build_Pi(hhi['V'], hhi['Va'], hhi['D'], c, Pi_b, Pi_e, probF, probI)
        diff = np.max(np.abs(Pi_new - Pi)); Pi = Pi_new

        # Calibrate the arrival rates (by sector flows)
        if flows is not None:
            x   = np.log([c[k] for k in pi_calib])
            f   = np.log([max(P_s[i], 1e-12) / t
                          for i, t in zip(pi_calib.values(), flows.values())])
            dpi = np.max(np.abs(f))
            # Freeze Pi_s once close enough, then let Pi_s settle
            if dpi > stol:
                if x0 is not None:
                    gain = np.clip(np.where(np.abs(f - f0) > 1e-12,
                                            (x - x0) / (f - f0), gain), 0.1, 20.0)
                x0, f0 = x, f
                xn = np.clip(x - np.clip(gain * f, -smax, smax),
                             np.log(1e-4), np.log(0.99))
                for k, val in zip(pi_calib, np.exp(xn)): c[k] = float(val)
            # Calibrate the BF Take-up
            dbf = 0.0
            for k, (agg, sh) in bf_calib.items():
                r = mom_data[agg[:4]] / max(float(ss[agg]) / float(ss[sh]), 1e-12)
                dbf = max(dbf, abs(np.log(r)))
                c[k] = float(np.clip(c[k] * r, 1e-4, 1.0))
        # Iterate until converges
        if verbose: print(f"[Pi loop] it {it:3d}  |dPi|={diff:.1e}  |dpi|={dpi:.1e}")
        if np.isfinite(hhi['Va']).all():
            _HH_WARM[(hhi['Va'].shape[0], hhi['Va'].shape[1])] = (hhi['Va'].copy(), hhi['V'].copy())
        if diff < tol and dpi < stol and dbf < stol and dA < atol * c['B']:
            hhi['P'] = P_s                  # F/I/U Transition
            for k in (*pi_calib, *bf_calib, *unknowns, 'tau_ss', 'B_ss'): ss.toplevel[k] = c[k]
            tdiff = time.time() - start
            if verbose:
                print(f"Steady State solved in {tdiff:.1f}s ({tdiff/60:.1f}min),  " +
                      "  ".join(f"{k}={c[k]:.4f}" for k in pi_calib))
            return ss
    # Name the criterion that stalled, they fail for very different reasons
    stuck = [f'{n}={v:.1e}>{t:.1e}' for n, v, t in
             (('|dPi|', diff, tol), ('|dpi|', dpi, stol), ('|dbf|', dbf, stol),
              ('|A-B|', dA, atol * c['B'])) if v >= t]
    raise RuntimeError(f"\n     Steady state stalled in {maxit} it: " + ", ".join(stuck))


# ---------------------------------------------------------------------------
# 3. Dynamics
# ---------------------------------------------------------------------------
def irf_partial(G, inp, dZ, var):
    # Partial Equilibrium: the Household Jacobian against the shock path.
    return {v: G[v][inp] @ dZ for v in var if v in G.outputs}


def ar1(size, rho, T, delay=0):
    # AR(1) shock path, announced `delay` quarters in advance.
    dZ = size * rho ** np.arange(T)
    return np.r_[np.zeros(delay), dZ[:T-delay]]


def hike(pps, n, rho, T, delay=0):
    # Copom cycle: `pps` a year each quarter for n quarters, then AR(1) decay.
    dZ = np.zeros(T)
    dZ[:n] = pps / 400 * np.arange(1, n + 1)      # annual bps -> quarterly rate
    for t in range(n, T): dZ[t] = rho * dZ[t-1]
    return np.r_[np.zeros(delay), dZ[:T-delay]]


def solve_dyn(hank, ss, shock, dZ, unknowns, targets, calib, var,
              moving=True, tol=1e-6, maxit=50, verbose=False):
    # `shock` is the exogenous input name ('Tr', 'rstar', ...); `dZ` its path.
    start = time.time()
    T = len(dZ)

    if not moving:
        td = hank.solve_impulse_nonlinear(ss, unknowns, targets, {shock: dZ},
                                          internals=['household'], verbose=False)
        return {v: td[v] for v in var}

    _, Pi_e, _, _, probF, _, probI =\
        make_egrid(calib['rho_e'], calib['sd_e'], calib['nE'], calib['amin'],
                   calib['amax'], calib['nA'], calib['sigma_F'],
                   calib['mu_I'], calib['sigma_I'], calib['nT'])

    _, Pi_b = make_bgrid(calib['beta_high'], calib['dbeta'], calib['omega_I'],
                         calib['q'], calib['nE'], calib['nT'])
    
    Pi_ss = ss['Pi']; dPi = np.zeros((T,) + Pi_ss.shape)
    V_ss  = ss.internals['household']['V']
    Va_ss = ss.internals['household']['Va']
    D_ss  = ss.internals['household']['D']

    for it in range(maxit):
        # GE Transition given the Current Pi Path + shock
        td = hank.solve_impulse_nonlinear(ss, unknowns, targets, {shock: dZ, 'Pi': dPi},
                                          internals=['household'], verbose=False)

        # MOVING: Rebuild Pi_t from the period-t value/dist, iterate to consistency
        V  = V_ss[None]  + td.internals['household']['V']   # Levels + Deviations
        Va = Va_ss[None] + td.internals['household']['Va']
        D  = D_ss[None]  + td.internals['household']['D']
        Pi_new = np.stack([build_Pi(V[t], Va[t], D[t], calib, Pi_b, Pi_e, probF, probI)[0]
                           for t in range(T)])
        dPi_new = Pi_new - Pi_ss
        diff = np.max(np.abs(dPi_new - dPi)); dPi = dPi_new
        if verbose: print(f"[Pi loop] it {it:3d}  |dPi|={diff:.1e}")
        if diff < tol:
            tdiff = time.time() - start
            print(f"Dynamics solved in {tdiff:.1f}s ({tdiff/60:.1f}min).")
            return {v: td[v] for v in var}
    raise RuntimeError("Pi path did not converge in {maxit} iterations.")


def irf_builder(hank, ss, calib, unknowns, targets, var):
    # IRF after a shock: `insu`, `full` and `comp` effects.
    def build(shock, dZ, unk=unknowns, targ=targets, v=var, split=True, verbose=False):
        run = lambda mv: solve_dyn(hank, ss, shock, dZ, unk, targ, calib, v,
                                   moving=mv, verbose=verbose)
        if not split: return run(True)
        insu, full = run(False), run(True)
        return dict(insu=insu, full=full, comp={x: full[x] - insu[x] for x in v})
    return build


# ---------------------------------------------------------------------------
# 4. SMM
# ---------------------------------------------------------------------------
# 1. Helpers
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


def _bounds_x(space=smm_space):
    b = []
    for lo, hi, tr in space.values():
        if   tr == 'log':   b.append((np.log(max(lo, 1e-8)), np.log(hi)))
        elif tr == 'logit': b.append((-8.0, 8.0))
        else:               b.append((lo, hi))
    return b


# 2. SMM Object 
class SMM:
    # Objective with caching and an evaluation log.
    def __init__(self, model, calib=None, space=smm_space, se=mom_se, coarse=None,
                 verbose=False, floor=0.02, logpath='output/smm_log.csv', every=25):
        self.model = model
        self.cal0  = dict(calibration if calib is None else calib)
        self.space = space
        self.keys  = mom_smm
        # Var(gap) = sampling + specification error (floor). W = S^-1, so it is efficient.
        self.S = np.diag([se[k] ** 2 + floor ** 2 for k in mom_smm])
        self.W = np.diag(1.0 / np.diag(self.S))
        self.coarse  = coarse or {}                 # e.g. {'nA': 60, 'nE': 7, 'nT': 3}
        self.logpath, self.every = logpath, every   # checkpoint, a run takes hours
        self.verbose, self.log, self.n = verbose, [], 0
        self._cache, self._best = {}, (np.inf, None)

    def gaps(self, p, coarse=True):
        # Run the model and get moments
        cal = {**self.cal0, **p, **(self.coarse if coarse else {})}
        ss  = solve_ss(self.model, cal, flows=flows,      # a slow point is a bad point
                       maxit=150 if coarse else 400)
        cal.update({k: float(ss[k]) for k in (*pi_calib, *bf_calib, *unknowns)})
        mod = model_moments(ss)
        return np.array([mod[k] - mom_data[k] for k in self.keys]), mod, ss, cal

    def __call__(self, x, coarse=True):
        # Turn the moment vector g into g'Wg (minimization objective)
        key = tuple(np.round(x, 10))
        if key in self._cache: return self._cache[key]
        self.n += 1; t0 = time.time()
        p = _to_p(x, self.space)
        why = ''
        try:
            g, mod, _, _ = self.gaps(p, coarse)
            J = float(g @ self.W @ g)
            if not np.isfinite(J): J, why = 1e6, 'non-finite'
        except Exception as e:
            mod, J, why = {}, 1e6, str(e)
        self._cache[key] = J
        self.log.append({**p, **{f'm_{k}': mod.get(k, np.nan) for k in self.keys},
                         'J': J, 'secs': time.time() - t0})
        if J < self._best[0]: self._best = (J, dict(p))
        if self.logpath and self.n % self.every == 0: self.history().to_csv(self.logpath)
        if self.verbose:
            print(f"  [{self.n:4d}] J={J:9.1e} best={self._best[0]:9.1e}  " +
                  " ".join(f"{k}={v:6.3f}" for k, v in p.items() if k in self.space) +
                  f"  ({time.time()-t0:3.0f}s) {why}")
        return J

    @property
    def best(self): return self._best[1]
    def history(self): return pd.DataFrame(self.log)


# 3. Estimate Parameters
def estimate(obj, p0=None, global_stage=True, local_stage=True, polish=None,
             popsize=8, maxiter_g=25, maxiter_l=200, seed=20260415):
    # Global Search on obj.coarse, then a local polish on `polish` (None keeps it).
    t0 = time.time()
    x0 = _to_x(p0 or {k: guess[k] for k in obj.space}, obj.space)
    if global_stage:
        print("\n--- Stage 1: Global Search ---")
        x0 = differential_evolution(obj, _bounds_x(obj.space), args=(True,),
                                    popsize=popsize, maxiter=maxiter_g, tol=1e-3,
                                    seed=seed, polish=False, init='sobol').x
    if local_stage:
        # The stages score on different grids, so `best` restarts with the polish
        print("\n--- Stage 2: Local Polish ---")
        print("  from:", " ".join(f"{k}={v:.3f}" for k, v in _to_p(x0, obj.space).items()))
        if polish is not None: obj.coarse = polish
        obj._cache.clear(); obj._best = (np.inf, None)
        x0 = minimize(obj, x0, args=(True,), method='Nelder-Mead',
                      options=dict(maxiter=maxiter_l, xatol=1e-4, fatol=1e-8, adaptive=True)).x

    p_hat = _to_p(x0, obj.space)
    g, mod, ss, cal = obj.gaps(p_hat, coarse=False)     # score on the full grid
    J = float(g @ obj.W @ g)
    print(f"\nSMM done in {(time.time()-t0)/60:.0f}min ({(time.time()-t0)/3600:.1f}hrs)",
             f"   J = {J:.4e}   {obj.n} solves")
    pd.Series(p_hat, name='value').to_csv(smm_paths['l' if local_stage else 'g'],
                                          index_label='param')
    return dict(params=p_hat, x=x0, J=J, g=g, moments=mod, ss=ss, calibration=cal)


# ---------------------------------------------------------------------------
# 5. Identification and Inference
# ---------------------------------------------------------------------------
def jacobian(obj, p_hat, step=0.02, coarse=False):
    # G = dg/dtheta by central differences, shape (nMoments, nParams).
    x, ks = _to_x(p_hat, obj.space), list(obj.space)
    G = []
    for j in range(x.size):
        t0 = time.time()
        xp, xm = x.copy(), x.copy(); xp[j] += step; xm[j] -= step
        gp, *_ = obj.gaps(_to_p(xp, obj.space), coarse)
        gm, *_ = obj.gaps(_to_p(xm, obj.space), coarse)
        dth = _to_p(xp, obj.space)[ks[j]] - _to_p(xm, obj.space)[ks[j]]
        G.append((gp - gm) / dth)
        print(f"  [G] {j+1}/{x.size} {ks[j]:8s} d={dth:+.3f}  ({(time.time()-t0)/60:.1f}min)")
    return np.array(G).T


def sensitivity(G, W):
    # Andrews-Gentzkow-Shapiro: how far theta_j moves per unit of moment m.
    return -np.linalg.inv(G.T @ W @ G) @ G.T @ W


def identification(G, W, names=None):
    # Weighted Jacobian: a large condition number = a direction not identified.
    s = np.linalg.svd(np.sqrt(W) @ G, compute_uv=False)
    v = np.linalg.svd(np.sqrt(W) @ G)[2]
    return pd.Series(s, name='sing. value'), s[0] / s[-1], \
           pd.Series(v[-1], index=names or list(smm_space), name='weakest dir.')


def std_errors(G, W, S, n_obs):
    # Sandwich SEs. S = Var(g_data), in the same absolute units as g.
    GWG = np.linalg.inv(G.T @ W @ G)
    V = GWG @ (G.T @ W @ S @ W @ G) @ GWG / n_obs
    return np.sqrt(np.diag(V)), V


def jtest(g, G, W, S, n_obs):
    # Hansen over-identification test.
    from scipy.stats import chi2
    M  = np.eye(len(g)) - G @ np.linalg.inv(G.T @ W @ G) @ G.T @ W
    Om = M @ S @ M.T
    stat, df = n_obs * float(g @ np.linalg.pinv(Om) @ g), len(g) - G.shape[1]
    return stat, df, (1 - chi2.cdf(stat, df) if df > 0 else np.nan)



# ---------------------------------------------------------------------------
def inference(res, obj, step=0.02, coarse=False, savepath=None, label='tab:smm_se'):
    # Classical Minimum Distance: S is already Var(moment), so n_obs = 1.
    from code.p7_results import _tex_table
    G  = jacobian(obj, res['params'], step=step, coarse=coarse)
    se = std_errors(G, obj.W, obj.S, 1)[0]
    stat, df, pval = jtest(res['g'], G, obj.W, obj.S, 1)

    tbl = pd.DataFrame({'estimate': res['params'], 's.e.': dict(zip(obj.space, se))})
    tbl['t'] = tbl['estimate'] / tbl['s.e.']
    print(tbl.round(4).to_string())
    print(f"Hansen J = {stat:.3f}   df = {df}   p = {pval:.3f}")

    esc  = lambda k: k.replace('_', r'\_')
    rows = [f'{esc(k)} & {r.estimate:.4f} & {r["s.e."]:.4f} & {r.t:.2f}'
            for k, r in tbl.iterrows()]
    rows += [r'\midrule', rf'Hansen $J$ & \multicolumn{{3}}{{c}}'
             rf'{{{stat:.2f}  (df {df}, $p$ = {pval:.3f})}}']
    _tex_table([r'Parameter & Estimate & S.E. & $t$'], rows,
               'SMM Estimates and Over-identification Test', label, 'lccc', savepath=savepath)
    return tbl, (stat, df, pval)


def report(res):
    mod  = res['moments']
    keys = [k for k in mom_data if k not in mom_fix]      # drop the by-construction ones
    name = lambda k: k if k in mom_smm else k + '*'
    tbl = pd.DataFrame([(name(k), mom_data[k], mod[k], mod[k] - mom_data[k]) for k in keys],
                       columns=['Moment', 'Data', 'Model', 'Gap']).set_index('Moment').round(4)
    print("=== Moments (* = Untargeted) ===\n", tbl.to_string())



# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from sequence_jacobian import create_model
    from code.p1_household import hh
    from code.p2_other_blocks import *

    hank_ss = create_model([hh, firm_formal, firm_informal, wages, nkpc_ss,
                            union_ss, monetary, fiscal, mkt_clearing, calibrate_ss])

    # 1. First Run
    p0  = {k: guess[k] for k in smm_space}
    obj = SMM(hank_ss, coarse=dict(nA=60, nE=9, nT=5), verbose=True)

    # 2. Identification and Sensitivity
    G = jacobian(obj, p0, coarse=True); s, cond, weak = identification(G, obj.W)
    print(f"Condition Number {cond:.1f}\n", s.round(4).to_string(),
          "\n", weak.round(3).to_string())
    print(pd.DataFrame(sensitivity(G, obj.W), index=list(smm_space),
                       columns=obj.keys).round(3).to_string())

    # 3. Global Search
    # res = estimate(obj, local_stage=False)
    # report(res)
    # print(pd.DataFrame({'estimate': res['params'],
    #                     'lower': {k: v[0] for k, v in smm_space.items()},
    #                     'upper': {k: v[1] for k, v in smm_space.items()}}).round(4).to_string())

    # 4. Polish (Mid Grid)
    # res = estimate(obj, p0=res['params'], global_stage=False,
    #                polish=dict(nA=100, nE=11, nT=5), maxiter_l=80)
    # report(res)

    # 5. Standard Errors and Over-identification
    # inference(res, obj, savepath="output/tables/smm_se.tex")

