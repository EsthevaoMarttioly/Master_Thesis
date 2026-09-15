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

from code.p1_household import make_egrid, make_bgrid, nS, nB, _HH_WARM
from code.p5_calibration import *


# ---------------------------------------------------------------------------
# 1. Endogenous Sector Transition
# ---------------------------------------------------------------------------
CH = 7        # Channels: cF0 cF1 cI0 cI1 cB0 cB1 cB2
PI_INT = {'household': ['V', 'Va', 'D', 'work']}    # all Pi needs, a T-path each


def _softmax(Vals, sig):
    # Turn "pick the best option" into smooth probabilities (sig -> 0 = hard max).
    V = np.stack(Vals, 0)
    Probs = np.exp((V - V.max(0)) / np.maximum(sig, 1e-10))
    return Probs / Probs.sum(0)


def _choice(V, Va, sig, nT, nE, probF, probI):
    # Acceptance Probabilities per Channel, (nB, nE, CH, nS*nT, nA).
    # Given `sig`, compare `V`s to get the probability of each channel being chosen.
    M, nA = nS * nT, V.shape[1]
    Vr  = V.reshape(M, nB, nE, nA).transpose(1, 2, 0, 3)
    Var = Va.reshape(M, nB, nE, nA).transpose(1, 2, 0, 3)
    Vs  = Vr.reshape(nB, nE, nS, nT, nA)

    EVF = np.broadcast_to(np.einsum('t,beta->bea', probF, Vs[:, :, F])[:, :, None], Vr.shape)
    EVI = np.broadcast_to(np.einsum('t,beta->bea', probI, Vs[:, :, I])[:, :, None], Vr.shape)

    cF = _softmax([Vr, EVF], sig * Var)           # Only a Formal Offer
    cI = _softmax([Vr, EVI], sig * Var)           # Only an Informal Offer
    cB = _softmax([Vr, EVF, EVI], sig * Var)      # Both Offers
    return np.stack([cF[0], cF[1], cI[0], cI[1], cB[0], cB[1], cB[2]], 2)


def _coeffs(C, D, work, nT, nE):
    # Mass-Weighted Acceptance by origin sector, and the Quit Weights of the Formals.
    M, nA = nS * nT, C.shape[-1]
    w = D.reshape(M, nB, nE, nA).transpose(1, 2, 0, 3)
    m = np.einsum('bema->m', w).reshape(nS, nT).sum(1)                  # mass by sector
    S = np.einsum('bekma,bema->km', C, w).reshape(CH, nS, nT).sum(2) / np.maximum(m, 1e-16)

    q, wF, CF = 1 - work, w[:, :, :nT], C[:, :, :, :nT]                 # F block only
    Q = np.r_[np.einsum('beta,te->', wF, q),
              np.einsum('bekta,beta,te->k', CF, wF, q)] / max(m[F], 1e-16)
    return S, Q                                   # Q = [quit, cF0*quit, cF1*quit, ...]


def _rates(S, Q, tgt, p, hard=True, it=200, tol=1e-13):
    # Given the flows, calculate the offer probabilities, by inverting the acceptance.
    # `hard` off over the burn-in: psi is transient, so is the quit set it implies.
    def check(x, name, flow, tag):
        if hard and not 0 < x <= 1:
            raise RuntimeError(f"\n     Flow {tag} = {flow:.3f} infeasible: "
                               f"{name} > 1 (sig too small)")
        return min(max(x, 1e-6), 1.0)

    dI = tgt['IU'];  kI = 1 - dI                   # Informals never Quit
    a, b, c, d = S[1], S[5], S[3], S[6]            # cF1 cB1 cI1 cB2, by origin

    # Unemployed: a 2 x 2 block, independent of the rest
    uF, uI = p['pi_UF'], p['pi_UI']
    for _ in range(it):
        nF = check(tgt['UF'] / ((1-uI)*a[U] + uI*b[U]), 'pi_UF', tgt['UF'], 'U -> F')
        nI = check(tgt['UI'] / ((1-uF)*c[U] + uF*d[U]), 'pi_UI', tgt['UI'], 'U -> I')
        if max(abs(nF-uF), abs(nI-uI)) < tol: break
        uF, uI = nF, nI

    # Employed: pi_F, pi_I and delta_F, coupled through keep_F and the quits
    eF, eI, dF = p['pi_F'], p['pi_I'], p['delta_F']
    for _ in range(it):
        nF = check(tgt['IF'] / (kI * ((1-eI)*a[I] + eI*b[I])), 'pi_F', tgt['IF'], 'I -> F')
        nI = check(tgt['FI'] / ((1-dF) * ((1-nF)*c[F] + nF*d[F])), 'pi_I', tgt['FI'], 'F -> I')
        qt = ((1-nF)*(1-nI)*Q[0] + nF*(1-nI)*Q[1]
              + (1-nF)*nI*Q[3] + nF*nI*Q[5])       # E[a_stay * quit] over the Formals
        if hard and qt >= tgt['FU']:
            raise RuntimeError(f"\n     Voluntary quits ({qt:.3f}) > "
                               f"F->U target ({tgt['FU']:.3f})")
        nD = max((tgt['FU'] - qt) / (1 - qt), 0.0)  # Layoffs take the Residual
        if max(abs(nF-eF), abs(nI-eI), abs(nD-dF)) < tol: break
        eF, eI, dF = nF, nI, nD

    return dict(pi_F=eF, pi_I=eI, pi_UF=uF, pi_UI=uI, delta_F=dF, delta_I=dI)


def _assemble(C, D, work, r, nT, nE, Pi_b, Pi_e, probF, probI):
    # Pi and the Sector Flows, contracting over assets without ever forming P.
    M, nA = nS * nT, C.shape[-1]
    w   = D.reshape(M, nB, nE, nA).transpose(1, 2, 0, 3)
    tot = w.sum(3, keepdims=True)
    wn  = np.where(tot > 1e-14, w / np.where(tot > 1e-14, tot, 1.0), 1.0 / nA)

    piF  = np.repeat([r['pi_F'], r['pi_F'], r['pi_UF']], nT)[:, None]
    piI  = np.repeat([r['pi_I'], r['pi_I'], r['pi_UI']], nT)[:, None]
    dlt  = np.repeat([r['delta_F'], r['delta_I'], 0.0], nT)
    keep = 1 - dlt

    qt = np.zeros((1, nE, M, 1)); qt[0, :, :nT, 0] = (1 - work).T   # Formal below Effort Cost
    stay = ((1-piF)*(1-piI) + piF*(1-piI)*C[:, :, 0]
            + (1-piF)*piI*C[:, :, 2] + piF*piI*C[:, :, 4])
    a_F  = piF * ((1-piI)*C[:, :, 1] + piI*C[:, :, 5])
    a_I  = piI * ((1-piF)*C[:, :, 3] + piF*C[:, :, 6])
    toU  = dlt[:, None] + keep[:, None] * stay * qt
    stay = stay * (1 - qt)

    # Averaged over Assets for Pi, Weighted by mass for the Flows
    ar, out = np.arange(M), []
    for ww in (wn, w):
        st, aF, aI, tu = (np.einsum('bema,bema->bem', x, ww)
                          for x in (stay, a_F, a_I, toU))
        X = np.zeros((nB, nE, M, M))
        X[:, :, ar, ar]             += keep * st
        X[:, :, :, :nT]             += (keep * aF)[..., None] * probF
        X[:, :, :, nT:2*nT]         += (keep * aI)[..., None] * probI
        X[:, :, ar, 2*nT + ar % nT] += tu
        out.append(X)

    # Order:   s (x) theta (x) beta (x) e
    # Pi[(s,t,b,e),(s',t',b',e')] = X[b,e,s,s'] * Pi_b[b,b'] * Pi_e[e,e']
    Pi = np.einsum('beMN,bB,eE->MbeNBE', out[0], Pi_b, Pi_e).reshape(M*nB*nE, M*nB*nE)
    flow = out[1].sum((0, 1)).reshape(nS, nT, nS, nT).sum((1, 3))
    return Pi, flow / flow.sum(1)[:, None]


def build_Pi(V, Va, D, p, Pi_b, Pi_e, probF, probI, work):
    # Pi at one date, taking the rates in `p` as given.
    nT, nE = p['nT'], Pi_e.shape[0]
    C = _choice(V, Va, p['sig'], nT, nE, probF, probI)
    return _assemble(C, D, work, p, nT, nE, Pi_b, Pi_e, probF, probI)


# ---------------------------------------------------------------------------
# 2. Steady State
# ---------------------------------------------------------------------------
# Initial Guess
unknowns = dict(beta_high = 0.98, psi = 0.7, L = 0.7, tau = 0.1, Tr = 0.2, B = 4.0)

def solve_ss(hank_block, calib, flows=None, counterfactual=False, verbose=False,
             bgain=0.005, burn=5, tol=1e-6, stol=1e-3, atol=1e-6, maxit=400, bmaxit=5000):
    """
    Solve the Steady-State by iterating the value function and the transition matrix.
    --- The household solve is the only expensive step:
    --- Markets Clear in closed form, The six arrival rates invert exactly,
    --- and only beta_high needs a secant on A = B.
    `counterfactual`: debt clears A = B (not beta) and hours clear the union FOC (not psi).
    """
    start = time.time()
    # Import Grids
    c = {**unknowns, **calib}
    c['tau_ss'], c['B_ss'] = c['tau'], c['B']
    nE, nA, nT = c['nE'], c['nA'], c['nT']

    _, Pi_e, _, _, probF, _, probI =\
        make_egrid(c['rho_e'], c['sd_e'], nE, c['amin'], c['amax'], nA,
                   c['sigma_F'], c['mu_I'], c['sigma_I'], nT)

    _, Pi_b = make_bgrid(c['beta_high'], c['dbeta'], c['omega_I'], c['q'], nE, nT)
    grid = (nT, nE, Pi_b, Pi_e, probF, probI)

    # Flat Start: with no value to compare, every offer is a coin flip.
    one = np.ones((nS*nT*nB*nE, nA))
    Pi, _ = _assemble(_choice(0*one, one, c['sig'], nT, nE, probF, probI),
                      one, np.ones((nT, nE)), c, *grid)

    dPi = dpi = dbf = dA = np.inf; xb = fb = None
    for it in range(maxit):
        # Guess Pi -> Solve -> Read V -> Rebuild Pi -> Repeat until Pi converges.
        c['Pi'] = Pi
        ss = SteadyStateDict(c)
        ss.update(hank_block.steady_state(c, options={'household': dict(backward_maxit=bmaxit)}))
        hhi = ss.internals['household']

        # Market Clearing: Analytical Forms + Secant on beta_high to asset_mkt
        if 'L_hat' in ss:
            c['L'], c['Tr'] = float(ss['L_hat']), float(ss['Tr_hat'])
            c['tau'] = c['tau_ss'] = float(ss['tau_hat'])
            dA = abs(float(ss['asset_mkt']))                  # A - B
            if counterfactual:
                c['h_F'] = float(ss['h_F_hat'])               # psi is structural: hours adjust
                c['B'] = c['B_ss'] = float(ss['A'])           # debt absorbs the savings
            else:
                c['psi'] = float(ss['psi_hat'])               # hours normalized: psi backed out
                c['B'] = c['B_ss'] = float(ss['B_hat'])       # debt pinned to B / GDP
                xn, fn = c['beta_high'], float(ss['asset_mkt'])  # dA/dbeta > 0
                if xb is not None and abs(fn - fb) > 1e-12:
                    g = (xn - xb) / (fn - fb)     # dbeta/dA > 0; if not, Pi moved A
                    if g > 0: bgain = float(np.clip(g, 1e-4, 0.05))
                xb, fb = xn, fn
                c['beta_high'] = float(np.clip(xn - np.clip(bgain*fn, -0.02, 0.02), 0.5, 0.999))

        # Acceptance, then the rates that make it hit the flow targets exactly
        C = _choice(hhi['V'], hhi['Va'], c['sig'], nT, nE, probF, probI)
        if flows is not None:
            c.update(_rates(*_coeffs(C, hhi['D'], hhi['work'], nT, nE), flows, c,
                            hard=it >= burn))
            # Calibrate the BF Take-up: coverage is linear in phi, so the ratio is the step
            dbf = 0.0
            for k, (agg, sh) in bf_calib.items():
                b = mom_data[agg[:4]] / max(float(ss[agg]) / float(ss[sh]), 1e-12)
                dbf = max(dbf, abs(np.log(b)))
                c[k] = float(np.clip(c[k] * b, 1e-4, 1.0))
        else:
            dpi = dbf = 0.0

        Pi_new, P_s = _assemble(C, hhi['D'], hhi['work'], c, *grid)
        dPi = np.max(np.abs(Pi_new - Pi)); Pi = Pi_new
        if flows is not None:                  # zero unless a rate had to be clipped
            dpi = max(abs(np.log(max(P_s[i], 1e-12) / t))
                      for i, t in zip(pi_calib.values(), flows.values()))

        # Iterate until converges
        if verbose: print(f"[Pi loop] it {it:3d}    |dPi|={dPi:.1e}")
        if np.isfinite(hhi['Va']).all():
            _HH_WARM[(hhi['Va'].shape[0], hhi['Va'].shape[1])] = (hhi['Va'].copy(), hhi['V'].copy())
        if dPi < tol and dpi < stol and dbf < stol and dA < atol * c['B']:
            hhi['P'] = P_s                  # F/I/U Transition
            for k in (*pi_calib, *bf_calib, *unknowns, 'h_F', 'tau_ss', 'B_ss'):
                ss.toplevel[k] = c[k]
            tdiff = time.time() - start
            if verbose:
                print(f"Steady State solved in {tdiff:.1f}s ({tdiff/60:.1f}min),  " +
                      "  ".join(f"{k}={c[k]:.4f}" for k in pi_calib))
            return ss
    # Name the Criterion that Stalled
    stuck = [f'{n}={v:.1e}>{t:.1e}' for n, v, t in
             (('|dPi|', dPi, tol), ('|dpi|', dpi, stol), ('|dbf|', dbf, stol),
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


def _dyn_jacobian(hank, ss, unknowns, targets, inp, T, cache):
    # H_U and Partial Jacobians are taken at the Steady State, so they are
    # constant within the dynamics loop. Cache them to avoid recomputing.
    key = (tuple(unknowns), tuple(targets), tuple(sorted(inp)), T)
    if key not in cache:
        Js = hank.partial_jacobians(ss, set(unknowns) | set(inp), set(targets), T)
        cache[key] = Js, FactoredJacobianDict(hank.jacobian(ss, unknowns, targets, T, Js), T)
    return cache[key]


def solve_dyn(hank, ss, shock, dZ, unknowns, targets, calib, var, moving=True,
              damp=1.0, tol=1e-6, maxit=100, verbose=False, jac=None):
    # `shock` is the exogenous input name ('Tr', 'rstar', ...); `dZ` its path.
    # One Newton loop with Pi rebuilt inside it: nesting a full nonlinear solve
    # within the Pi loop costs the product of both iteration counts.
    start, T = time.time(), len(dZ)
    inp = {shock: dZ}
    if moving: inp['Pi'] = np.zeros((T,) + ss['Pi'].shape)   # Pi path is endogenous
    Js, HU = _dyn_jacobian(hank, ss, unknowns, targets, inp, T,
                           {} if jac is None else jac)

    if moving:
        _, Pi_e, _, _, probF, _, probI =\
            make_egrid(calib['rho_e'], calib['sd_e'], calib['nE'], calib['amin'],
                       calib['amax'], calib['nA'], calib['sigma_F'],
                       calib['mu_I'], calib['sigma_I'], calib['nT'])

        _, Pi_b = make_bgrid(calib['beta_high'], calib['dbeta'], calib['omega_I'],
                             calib['q'], calib['nE'], calib['nT'])
        hhi = ss.internals['household']
        V_ss, Va_ss, D_ss, W_ss = hhi['V'], hhi['Va'], hhi['D'], hhi['work']

    U, dpi = ImpulseDict({k: np.zeros(T) for k in unknowns}), 0.0
    for it in range(maxit):
        # One Newton step on the unknowns, given the current Pi path
        td  = hank.impulse_nonlinear(ss, ImpulseDict({**inp, **U}), [*var, *targets],
                                     internals=PI_INT if moving else {}, Js=Js)
        err = float(np.max([np.max(np.abs(td[k])) for k in targets]))
        if not np.isfinite(err):
            raise RuntimeError(f"\n     Dynamics blew up at it {it}: lower `damp`")
        U  += damp * HU.apply(td)

        # MOVING: Rebuild Pi_t from the period-t value/dist, iterate to consistency
        if moving:
            h = td.internals['household']
            Pi_new = np.stack([build_Pi(V_ss + h['V'][t], Va_ss + h['Va'][t],
                                        D_ss + h['D'][t], calib, Pi_b, Pi_e, probF,
                                        probI, W_ss + h['work'][t])[0] for t in range(T)])
            dpi = np.max(np.abs(Pi_new - ss['Pi'] - inp['Pi']))
            inp['Pi'] = Pi_new - ss['Pi']

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
    jac = {}                    # H_U is the same for every shock and iteration
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
        cal.update({k: float(ss[k]) for k in (*pi_calib, *bf_calib, *unknowns)})
        self._warm = {k: cal[k] for k in (*pi_calib, *bf_calib, *unknowns)}
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
    from code.p1_household import hh
    from code.p2_other_blocks import *

    hank_ss = create_model([hh, firm_formal, firm_informal, wages, nkpc_ss,
                            union_ss, monetary, fiscal, mkt_clearing, calibrate_ss])

    # 1. First Run
    p0  = {k: guess[k] for k in smm_space}
    obj = SMM(hank_ss, coarse=dict(nA=60, nE=9), verbose=True)

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
    res = load_res('g')       # Load Global Search
    res = estimate(obj, p0=res['params'], global_stage=False, polish=dict(nA=150, nE=11))
    report(res)

    # 5. Identification and Inference
    # res = load_res('l')       # Load Local Polish
    inference(res, obj, savepath="output/tables/smm_se.tex")

