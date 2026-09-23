#=
#----------------------------------------------------------------------------
# DESCRIPTION
# Solve the model: Endogenous Sector Transition, Steady State, Dynamics,
# and the SMM Estimator.
# ---------------------------------------------------------------------------
#=

# ---- Packages -------------------------------------------------------------
import time
import pickle
import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from sequence_jacobian.classes import SteadyStateDict
from sequence_jacobian.classes.impulse_dict import ImpulseDict
from sequence_jacobian.classes.jacobian_dict import FactoredJacobianDict

from code.p1_household import make_egrid, make_bgrid, bf_income, bf_test, nS, nB, nF, _HH_WARM
from code.p5_calibration import *


# ---------------------------------------------------------------------------
# 1. Endogenous Sector Transition
# ---------------------------------------------------------------------------
CH = 7        # Channels: cF0 cF1 cI0 cI1 cB0 cB1 cB2

def _softmax(Vals, sig):
    # Smooth max{Vstay, VF, VI, VU, ...} (sig -> 0 = hard max).
    V = np.stack(Vals, 0)
    Probs = np.exp((V - V.max(0)) / np.maximum(sig, 1e-10))
    return Probs / Probs.sum(0)


def _test(c, D, e_grid, thetaF, thetaI, a_grid):
    # Pooled BF Test P[b' = 1 | state], averaged over own assets
    q = bf_test(bf_income(e_grid, thetaF, thetaI, a_grid, c['rstar'] / (c['w'] * c['h_F'])),
                c['ybar'], c['sig_BF'], c['lambda_BF'], c['rho_F'], c['rho_I'])
    m = D.sum(1)
    return np.where(m > 1e-14, (D * q).sum(1) / np.maximum(m, 1e-14), q.mean(1))


def _pool(x, D, qb, nT, nE):
    # For each b in {0,1}, average over (e, theta) assets to give Asset's Income (r*a)
    M, nA = nS * nF * nT, D.shape[-1]
    w   = D.reshape(M, nB, nE, nA).transpose(1, 2, 0, 3)
    tot = w.sum(3, keepdims=True)
    y   = np.einsum('be...ma,bema->be...m', x,
                    np.where(tot > 1e-14, w / np.maximum(tot, 1e-14), 1.0 / nA))
    q   = qb.reshape(M, nB, nE).transpose(1, 2, 0).reshape(nB, nE, *[1] * (y.ndim - 3), M)
    at  = lambda g: np.broadcast_to(y.reshape(*y.shape[:-1], nS, nF, nT)[..., g:g+1, :],
                                    y.shape[:-1] + (nS, nF, nT)).reshape(y.shape)
    return [(1 - q) * at(0), q * at(1)], tot[..., 0]


def _choice(V, Va, sig, nT, nE, probF, probI):
    # Acceptance Probabilities per Channel, (nB, nE, CH, nS*nT, nA).
    # Given `sig`, compare `V`s to get the probability of each channel being chosen.
    M, nA = nS * nF * nT, V.shape[1]
    Vr  = V.reshape(M, nB, nE, nA).transpose(1, 2, 0, 3)
    Var = Va.reshape(M, nB, nE, nA).transpose(1, 2, 0, 3)
    Vs  = Vr.reshape(nB, nE, nS, nF, nT, nA)

    # An offer keeps the BF status: E_theta V(s', b, theta')
    ev  = lambda p, s: np.broadcast_to(np.einsum('t,bekta->beka', p, Vs[:, :, s])\
                                       [:, :, None, :, None], Vs.shape).reshape(Vr.shape)
    EVF, EVI = ev(probF, F), ev(probI, I)

    cF = _softmax([Vr, EVF], sig * Var)           # Only a Formal Offer
    cI = _softmax([Vr, EVI], sig * Var)           # Only an Informal Offer
    cB = _softmax([Vr, EVF, EVI], sig * Var)      # Both Offers
    return np.stack([cF[0], cF[1], cI[0], cI[1], cB[0], cB[1], cB[2]], 2)


def _rates(C, D, qb, tgt, p, it=200, tol=1e-13):
    # Given the flows, calculate the offer probabilities, by inverting the acceptance.
    A, tot = _pool(C, D, qb, p['nT'], p['nE'])                     # (nB, nE, CH, M)
    m = tot.sum((0, 1)).reshape(nS, -1).sum(1)                     # mass by sector
    S = np.einsum('bekm,bem->km', sum(A), tot).reshape(CH, nS, -1).sum(2) / np.maximum(m, 1e-16)

    def check(x, name, flow, tag):
        if not 0 < x <= 1:
            raise RuntimeError(f"\n     Flow {tag} = {flow:.3f} infeasible: "
                               f"{name} > 1 (sig too small)")
        return x

    kF, kI = 1 - p['delta_F'], 1 - p['delta_I']    # layoffs are exogenous
    a, b, c, d = S[1], S[5], S[3], S[6]            # cF1 cB1 cI1 cB2, by origin

    # Unemployed: a 2 x 2 block, independent of the rest
    uF, uI = p['pi_UF'], p['pi_UI']
    for _ in range(it):
        nF = check(tgt['UF'] / ((1-uI)*a[U] + uI*b[U]), 'pi_UF', tgt['UF'], 'U -> F')
        nI = check(tgt['UI'] / ((1-uF)*c[U] + uF*d[U]), 'pi_UI', tgt['UI'], 'U -> I')
        if max(abs(nF-uF), abs(nI-uI)) < tol: break
        uF, uI = nF, nI

    # Employed: pi_F and pi_I, a 2 x 2 block net of the layoffs
    eF, eI = p['pi_F'], p['pi_I']
    for _ in range(it):
        nF = check(tgt['IF'] / (kI * ((1-eI)*a[I] + eI*b[I])), 'pi_F', tgt['IF'], 'I -> F')
        nI = check(tgt['FI'] / (kF * ((1-nF)*c[F] + nF*d[F])), 'pi_I', tgt['FI'], 'F -> I')
        if max(abs(nF-eF), abs(nI-eI)) < tol: break
        eF, eI = nF, nI

    return dict(pi_F=eF, pi_I=eI, pi_UF=uF, pi_UI=uI)


def _assemble(C, D, r, qb, nT, nE, Pi_b, Pi_e, probF, probI):
    # Pi and the Sector Flows, contracting over assets without ever forming P.
    M = nS * nF * nT

    piF  = np.repeat([r['pi_F'], r['pi_F'], r['pi_UF']], nF * nT)[:, None]
    piI  = np.repeat([r['pi_I'], r['pi_I'], r['pi_UI']], nF * nT)[:, None]
    dlt  = np.repeat([r['delta_F'], r['delta_I'], 0.0], nF * nT)
    keep = 1 - dlt

    stay = ((1-piF)*(1-piI) + piF*(1-piI)*C[:, :, 0]
            + (1-piF)*piI*C[:, :, 2] + piF*piI*C[:, :, 4])
    a_F  = piF * ((1-piI)*C[:, :, 1] + piI*C[:, :, 5])
    a_I  = piI * ((1-piF)*C[:, :, 3] + piF*C[:, :, 6])

    # BF Test, then sector moves, averaged over its assets
    ar = np.arange(M)
    s_, t_ = ar // (nF * nT), ar % nT
    X  = np.zeros((nB, nE, M, M))
    (st, aF, aI, q), tot = zip(*(_pool(x, D, qb, nT, nE) for x in (stay, a_F, a_I, 0 * stay + 1)))
    for g in range(nF):
        X[:, :, ar, (s_*nF + g)*nT + t_]          += keep * st[g]
        X[:, :, :, (F*nF + g)*nT + np.arange(nT)] += (keep * aF[g])[..., None] * probF
        X[:, :, :, (I*nF + g)*nT + np.arange(nT)] += (keep * aI[g])[..., None] * probI
        X[:, :, ar, (U*nF + g)*nT + t_]           += dlt * q[g]       # layoffs

    # Order:   s (x) b (x) theta (x) beta (x) e
    # Pi[(s,b,t,b,e),(s',b',t',b',e')] = X[b,e,(s,b,t),(s',b',t')] * Pi_beta[beta,beta'] * Pi_e[e,e']
    Pi = np.einsum('beMN,bB,eE->MbeNBE', X, Pi_b, Pi_e).reshape(M*nB*nE, M*nB*nE)
    flow = np.einsum('bem,bemN->mN', tot[0], X).reshape(nS, nF*nT, nS, nF*nT).sum((1, 3))
    return Pi, flow / flow.sum(1)[:, None]


# ---------------------------------------------------------------------------
# 2. Steady State
# ---------------------------------------------------------------------------
def solve_ss(hank_block, calib, flows=None, counterfactual=False, verbose=False,
             damp=0.75, tol=1e-6, stol=1e-3, atol=1e-5, maxit=400, bmaxit=5000):
    """Solve the Steady-State by iterating the value function and the transition matrix.
    1) Household Solve:       expensive step, slow;
    2) Market Clear:          closed form, invert exactly, very fast;
    3) Asset Market:          calibrates beta_high to A = B + p_e, secant method;
    4) Pi_s and BF:           internally calibrate Pi_s and BF.

    flows           :  dict,      {'FI': value, ...} to be matched
    counterfactual  :  bool,      True, so B clears asset_market and h_F clear wage_nkpc
    damp            :  [0,1],     damp < 1 to avoid stuck, but converges slower
    tol, stol       :  value,     tolerance for 'Pi', and 'Pi_s' and 'BF'
    atol            :  value,     tolerance for 'asset_mkt / B'."""

    start = time.time()
    # Import Grids
    c = {**unknowns, **calib}
    c['tau_ss'], c['B_ss'] = c['tau'], c['B']
    nE, nA, nT = c['nE'], c['nA'], c['nT']

    e_grid, Pi_e, a_grid, thetaF, probF, thetaI, probI =\
        make_egrid(c['rho_e'], c['sd_e'], nE, c['amin'], c['amax'], nA,
                   c['sigma_F'], c['mu_I'], c['sigma_I'], nT)

    _, Pi_b = make_bgrid(c['beta_high'], c['dbeta'], c['omega_I'], c['q'], nE, nT)
    grid  = (nT, nE, Pi_b, Pi_e, probF, probI)
    tgrid = (e_grid, thetaF, thetaI, a_grid)

    # Start from a given Pi and BF test
    N = nS * nF * nT * nB * nE
    if np.shape(c.get('Pi')) == (N, N) and np.shape(c.get('Qb')) == (N,):
        Pi, qb = c['Pi'], c['Qb']
    else:
        qb  = _test(c, np.ones((N, nA)), *tgrid)
        one = np.ones((N, nA))
        Pi, _ = _assemble(_choice(0*one, one, c['sig'], nT, nE, probF, probI), one, c, qb, *grid)

    dPi = dpi = dbf = dA = np.inf; xb = fb = None; bgain = 0.005

    # Guess Pi -> Solve -> Read V -> Rebuild Pi -> Repeat until Pi converges.
    for it in range(maxit):
        c['Pi'], c['Qb'] = Pi, qb
        ss = SteadyStateDict(c)
        ss.update(hank_block.steady_state(c, options={'household': dict(backward_maxit=bmaxit)}))
        hhi = ss.internals['household']

        # Solving unknowns to clear markets analytically
        if 'L_hat' in ss:
            c['L'], c['Tr'] = float(ss['L_hat']), float(ss['Tr_hat'])
            c['tau'] = c['tau_ss'] = float(ss['tau_hat'])
            dA = abs(float(ss['asset_mkt']))
            if counterfactual:
                c['h_F'] = np.sqrt(c['h_F'] * float(ss['h_F_hat']))   # psi structural; half-step, or it cycles
                c['B'] = c['B_ss'] = float(ss['A'] - ss['p_e'])       # debt absorbs the savings
            else:
                c['psi'] = float(ss['psi_hat'])                  # hours normalized: psi backed out
                c['B'] = c['B_ss'] = float(ss['B_hat'])          # debt pinned to B / GDP
                xn, fn = c['beta_high'], float(ss['asset_mkt'])  # dA/dbeta > 0
                if xb is not None and abs(fn - fb) > 1e-12:
                    g = (xn - xb) / (fn - fb)     # dbeta/dA > 0; if not, Pi moved A
                    if g > 0: bgain = float(np.clip(g, 1e-4, 0.05))
                xb, fb = xn, fn
                c['beta_high'] = float(np.clip(xn - np.clip(bgain*fn, -0.02, 0.02), 0.5, 1.0))

        # Acceptance rate from `V` and rates `pi_s`
        C  = _choice(hhi['V'], hhi['Va'], c['sig'], nT, nE, probF, probI)
        qb = _test(c, hhi['D'], *tgrid)
        if flows is not None:
            c.update(_rates(C, hhi['D'], qb, flows, c))
            # Calibrate the BF Rule
            dbf = 0.0
            for k, (agg, sh, sgn) in bf_calib.items():
                b = (mom_data[agg[:4]] / max(float(ss[agg]) / float(ss[sh]), 1e-12)) ** (sgn / 2)
                x = float(np.clip(c[k] * b, 1e-4, 1.0))           # probabilities
                dbf, c[k] = max(dbf, abs(np.log(x / c[k]))), x    # at a bound: the closest fit
        else:
            dpi = dbf = 0.0

        Pi_new, P_s = _assemble(C, hhi['D'], c, qb, *grid)
        dPi = np.max(np.abs(Pi_new - Pi)); Pi = damp * Pi + (1-damp) * Pi_new    # damp to avoid stucking
        if flows is not None:
            dpi = max(abs(np.log(max(P_s[i, j], 1e-12) / flows[n]))    # zero, unless a rate has clipped
                      for n, (i, j) in sectors.items())

        # Iterate until converges
        if verbose: print(f"[Pi loop] it {it:3d}     |dPi|={dPi:.1e}")
        if np.isfinite(hhi['Va']).all():
            _HH_WARM[(hhi['Va'].shape[0], hhi['Va'].shape[1])] = (hhi['Va'].copy(), hhi['V'].copy())
        
        # Converged: one last solve
        if dPi < tol and dpi < stol and dbf < stol and dA < atol * abs(c['B']):
            c['Pi'], c['Qb'] = Pi, qb
            ss = SteadyStateDict(c)
            ss.update(hank_block.steady_state(c, options={'household': dict(backward_maxit=bmaxit)}))
            ss.internals['household']['P'] = P_s
            tdiff = time.time() - start
            if verbose:
                print(f"Steady State solved in {tdiff:.1f}s ({tdiff/60:.1f}min),  " +
                      "  ".join(f"{k}={c[k]:.4f}" for k in (*pi_calib, *bf_calib) if flows is not None))
            return ss
        
    # If didn't converge, print why
    stuck = [f'{n}={v:.1e}>{t:.1e}' for n, v, t in
             (('|dPi|', dPi, tol), ('|dpi|', dpi, stol), ('|dbf|', dbf, stol),
              ('|A-B|', dA, atol * abs(c['B']))) if v >= t]
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


def _dyn_jacobian(hank, ss, unknowns, targets, inp, T, cache):
    # Cache Jacobians at Steady State to avoid recomputing in Dyn.
    key = (tuple(unknowns), tuple(targets), tuple(sorted(inp)), T)
    if key not in cache:
        Js = hank.partial_jacobians(ss, set(unknowns) | set(inp), set(targets), T)
        cache[key] = Js, FactoredJacobianDict(hank.jacobian(ss, unknowns, targets, T, Js), T)
    return cache[key]


def solve_dyn(hank, ss, shock, dZ, unknowns, targets, calib, var, moving=True,
              damp=1.0, gpi=0.75, tol=1e-6, maxit=100, verbose=False, jac=None):
    """Solve the Dynamics by iteration the transition matrix.

    shock           :  string,    the exogenous variable input name ('Tr', 'rstar')
    dZ              :  list,      the shock's path, normally an AR(1)
    unknowns        :  list,      variables that moves, solving `targets`
    moving          :  bool,      allow Pi_s to move, endogeneously
    damp, gpi       :  [0,1],     damp < 1 to avoid stuck, but converges slower"""

    start, T = time.time(), len(dZ)
    inp = {shock: dZ}
    if moving:
        inp['Pi'] = np.zeros((T,) + ss['Pi'].shape)     # Change in Pi_s
    Js, HU = _dyn_jacobian(hank, ss, unknowns, targets, inp, T, {} if jac is None else jac)

    if moving:
        e_grid, Pi_e, a_grid, thetaF, probF, thetaI, probI =\
            make_egrid(calib['rho_e'], calib['sd_e'], calib['nE'], calib['amin'],
                       calib['amax'], calib['nA'], calib['sigma_F'],
                       calib['mu_I'], calib['sigma_I'], calib['nT'])

        _, Pi_b = make_bgrid(calib['beta_high'], calib['dbeta'], calib['omega_I'],
                             calib['q'], calib['nE'], calib['nT'])
        hhi = ss.internals['household']
        V_ss, Va_ss, D_ss = hhi['V'], hhi['Va'], hhi['D']
        tgrid = (e_grid, thetaF, thetaI, a_grid)

    U, dpi = ImpulseDict({k: np.zeros(T) for k in unknowns}), 0.0
    for it in range(maxit):
        # One Newton step on the unknowns, given the current Pi path
        td  = hank.impulse_nonlinear(ss, ImpulseDict({**inp, **U}), [*var, *targets],
                                     internals={'household': ['V', 'Va', 'D']}
                                     if moving else {}, Js=Js)
        err = float(np.max([np.max(np.abs(td[k])) for k in targets]))
        if not np.isfinite(err):
            raise RuntimeError(f"\n     Dynamics blew up at it {it}: lower `damp`")
        U  += damp * HU.apply(td)

        # MOVING: Rebuild Pi_t from the period-t value/dist, iterate to consistency
        if moving:
            h, dpi = td.internals['household'], 0.0
            for t in range(T):
                D  = D_ss + h['D'][t]
                C  = _choice(V_ss + h['V'][t], Va_ss + h['Va'][t], calib['sig'],
                             calib['nT'], calib['nE'], probF, probI)
                dP = _assemble(C, D, calib, _test(calib, D, *tgrid), calib['nT'], calib['nE'],
                               Pi_b, Pi_e, probF, probI)[0] - ss['Pi'] - inp['Pi'][t]
                dpi = max(dpi, np.max(np.abs(dP)))
                inp['Pi'][t] += gpi * dP      # Damp to avoid cycling

        tdiff = time.time() - start
        if verbose:
            print(f"[Dyn loop] it {it:3d}  |err|={err:.1e}  |dPi|={dpi:.1e}"
                  f"  ({tdiff:.0f}s, {tdiff/60:.1f}min)")
        if err < tol and dpi < tol:
            print(f"Dynamics solved in {tdiff:.1f}s ({tdiff/60:.1f}min), {it+1} it.")
            return {v: td[v] for v in var}
    raise RuntimeError(f"\n     Dynamics stalled in {maxit} it: "
                       f"|err|={err:.1e}>{tol:.1e}, |dPi|={dpi:.1e}>{tol:.1e}")


def irf_builder(hank, ss, calib, unknowns, targets, var):
    # IRF after a shock: `insu`, `full` and `comp` effects.
    jac = {}
    def build(shock, dZ, unk=unknowns, targ=targets, v=var, split=True, verbose=False):
        run = lambda mv: solve_dyn(hank, ss, shock, dZ, unk, targ, calib, v,
                                   moving=mv, verbose=verbose, jac=jac)
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
        self._warm = {}                             # last converged rates, as a start
        self._cache, self._best = {}, (np.inf, None)

    def gaps(self, p, coarse=True):
        # Run the model and get moments
        cal = {**self.cal0, **self._warm, **p, **(self.coarse if coarse else {})}
        ss  = solve_ss(self.model, cal, flows=flows,      # a slow point is a bad point
                       maxit=150 if coarse else 400,
                       tol=1e-5 if coarse else 1e-6,
                       bmaxit=1000 if coarse else 5000)
        cal.update({k: float(ss[k]) for k in update})
        self._warm = {**{k: cal[k] for k in update}, 'Pi': ss['Pi'], 'Qb': ss['Qb']}
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
             popsize=8, maxiter_g=25, maxiter_l=80, seed=20260415):
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
    res  = dict(params=p_hat, x=x0, J=J, g=g, moments=mod, ss=ss, calibration=cal)
    path = smm_paths['l' if local_stage else 'g']
    pd.Series(p_hat, name='value').to_csv(path, index_label='param')
    save_res(res, path)
    return res


# ---------------------------------------------------------------------------
# 5. Identification
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


# ---------------------------------------------------------------------------
# 6. Inference
# ---------------------------------------------------------------------------
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


def inference(res, obj, step=0.02, coarse=False, scale=True, savepath=None, label='tab:smm_se'):
    # Classical Minimum Distance: S is already Var(moment), so n_obs = 1.
    from scipy.stats import norm
    from code.p7_results import _tex_table
    G  = jacobian(obj, res['params'], step=step, coarse=coarse)
    se = std_errors(G, obj.W, obj.S, 1)[0]
    stat, df, pval = jtest(res['g'], G, obj.W, obj.S, 1)
    k  = np.sqrt(stat / df) if scale else 1.0     # inflate the SE by the misspecification

    tbl = pd.DataFrame({'estimate': res['params'], 's.e.': dict(zip(obj.space, k * se)),
                        'lower': {k_: v[0] for k_, v in obj.space.items()},
                        'upper': {k_: v[1] for k_, v in obj.space.items()}})
    tbl['t'] = tbl['estimate'] / tbl['s.e.']
    tbl['p'] = 2 * norm.sf(np.abs(tbl['t']))
    tbl = tbl[['estimate', 's.e.', 't', 'p', 'lower', 'upper']]
    print(tbl.round(4).to_string())
    print(f"Hansen J = {stat:.3f}   df = {df}   p = {pval:.3f}   SE scale = {k:.2f}")

    esc  = lambda x: x.replace('_', r'\_')
    rows = [f'{esc(n)} & {r.estimate:.3f} & {r["s.e."]:.3f} & {r.t:.2f} & {r.p:.3f}'
            f' & {r.lower:.2f} & {r.upper:.3f}' for n, r in tbl.iterrows()]
    rows += [r'\midrule', rf'Hansen $J$ & \multicolumn{{6}}{{c}}'
             rf'{{{stat:.2f}  (df {df}, $p$ = {pval:.3f})}}']
    _tex_table([r'Parameter & Estimate & S.E. & $t$ & $p$ & Lower & Upper'], rows,
               'SMM Estimates and Over-identification Test', label, 'lcccccc', savepath=savepath)


# ---------------------------------------------------------------------------
def report(res):
    mod  = res['moments']
    keys = [k for k in mom_data if k not in mom_fix]      # drop the by-construction ones
    name = lambda k: k if k in mom_smm else k + '*'
    tbl = pd.DataFrame([(name(k), mom_data[k], mod[k], mod[k] - mom_data[k]) for k in keys],
                       columns=['Moment', 'Data', 'Model', 'Gap']).set_index('Moment').round(4)
    print("=== Moments (* = Untargeted) ===\n", tbl.to_string())


def save_res(res, path):
    # `ss` does not pickle, so report() and inference() reload the rest.
    with open(path.replace('.csv', '.pkl'), 'wb') as fobj:
        pickle.dump({k: v for k, v in res.items() if k != 'ss'}, fobj)


def load_res(stage='l'):
    # Reload a finished run to report() or inference() in another session.
    with open(smm_paths[stage].replace('.csv', '.pkl'), 'rb') as fobj:
        return pickle.load(fobj)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from sequence_jacobian import create_model
    from code.p1_household import hh_ss
    from code.p2_other_blocks import *

    hank_ss = create_model([hh_ss, firm_formal, firm_informal, nkpc_ss, union_ss,
                            equity_ss, monetary, fiscal, mkt_clearing, calibrate_ss])

    # 1. First Run
    p0  = {k: guess[k] for k in smm_space}
    obj = SMM(hank_ss, coarse=dict(nA=40, nE=7), verbose=True)

    # 2. Identification and Sensitivity
    G = jacobian(obj, p0, coarse=True); s, cond, weak = identification(G, obj.W)
    print(f"Condition Number {cond:.1f}\n", s.round(4).to_string(),
          "\n", weak.round(3).to_string())
    print(pd.DataFrame(sensitivity(G, obj.W), index=list(smm_space),
                       columns=obj.keys).round(3).to_string())

    # 3. Global Search
    # res = estimate(obj, local_stage=False, popsize=2)
    # report(res)

    # 4. Polish (Mid Grid)
    # res = load_res('g')       # Load Global Search
    # res = estimate(obj, p0=res['params'], global_stage=False, polish=dict(nA=150, nE=11))
    # report(res)

    # 5. Identification and Inference
    # res = load_res('l')       # Load Local Polish
    # inference(res, obj, savepath="output/tables/smm_se.tex")

