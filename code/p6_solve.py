#=
#----------------------------------------------------------------------------
# DESCRIPTION
# Solve the model: Endogenous Sector Transition, Steady State, Dynamics,
# and the SMM Estimator.
# ---------------------------------------------------------------------------
#=

# ---- Packages -------------------------------------------------------------
seed = 20260415

import time
import pickle
import numpy as np
import pandas as pd
from scipy.optimize import minimize
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
    V = np.stack(np.broadcast_arrays(*Vals), 0)
    Probs = np.exp((V - V.max(0)) / np.maximum(sig, 1e-10))
    return Probs / Probs.sum(0)


def _test(c, D, e_grid, thetaF, thetaI, a_grid):
    # Pooled BF Test P[b' = 1 | state], averaged over own assets
    yF = c['w'] * c['h_F']
    q  = bf_test(bf_income(e_grid, thetaF, thetaI, a_grid, c['w_I'] * c['h_I'] / yF,
                           c.get('r', c['rstar']) / yF),
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

    # An offer keeps the BF status: V(s', b, theta') for each draw theta'
    fo = (np.arange(M) // nT) % nF                              # BF status of the origin
    WF = Vs[:, :, F][:, :, fo].transpose(0, 1, 3, 2, 4)         # (nB, nE, nT, M, nA)
    WI = Vs[:, :, I][:, :, fo].transpose(0, 1, 3, 2, 4)

    V1, S1 = Vr[:, :, None], (sig * Var)[:, :, None]
    cF = _softmax([V1, WF], S1)                                 # Only a Formal Offer
    cI = _softmax([V1, WI], S1)                                 # Only an Informal Offer
    cB = _softmax([V1[:, :, None], WF[:, :, :, None],           # Both Offers: two draws
                   WI[:, :, None, :]], S1[:, :, None])

    # Integrate the draws out, channel by channel
    pF, pI = probF[:, None, None], probI[:, None, None]
    bF, bI = np.einsum('u,betuma->betma', probI, cB[1]), np.einsum('t,betuma->beuma', probF, cB[2])
    mean   = lambda p, x: np.einsum('t,betma->bema', p, x)
    C = np.stack([mean(probF, cF[0]), mean(probF, cF[1]),
                  mean(probI, cI[0]), mean(probI, cI[1]),
                  np.einsum('t,u,betuma->bema', probF, probI, cB[0]),
                  mean(probF, bF), mean(probI, bI)], 2)
    return C, [pF * cF[1], pF * bF, pI * cI[1], pI * bI]


def _rates(C, D, qb, tgt, p, it=200, tol=1e-13):
    # Given the flows, calculate the offer probabilities, by inverting the acceptance.
    A, tot = _pool(C, D, qb, p['nT'], p['nE'])                     # (nB, nE, CH, M)
    m = tot.sum((0, 1)).reshape(nS, -1).sum(1)                     # mass by sector
    S = np.einsum('bekm,bem->km', sum(A), tot).reshape(CH, nS, -1).sum(2) / np.maximum(m, 1e-16)

    check = lambda x, *_: float(np.clip(x, 1e-6, 1.0))   # the flows are targets, not identities

    kF, kI = 1 - p['delta_F'], 1 - p['delta_I']    # layoffs are exogenous
    a, b, c, d = S[1], S[5], S[3], S[6]            # cF1 cB1 cI1 cB2, by origin

    # Unemployed: a 2 x 2 block, independent of the rest
    uF, uI = p['pi_UF'], p['pi_UI']
    for _ in range(it):
        xF = check(tgt['UF'] / ((1-uI)*a[U] + uI*b[U]), 'pi_UF', tgt['UF'], 'U -> F')
        xI = check(tgt['UI'] / ((1-uF)*c[U] + uF*d[U]), 'pi_UI', tgt['UI'], 'U -> I')
        if max(abs(xF-uF), abs(xI-uI)) < tol: break
        uF, uI = xF, xI

    # Employed: pi_F and pi_I, a 2 x 2 block net of the layoffs
    eF, eI = p['pi_F'], p['pi_I']
    for _ in range(it):
        xF = check(tgt['IF'] / (kI * ((1-eI)*a[I] + eI*b[I])), 'pi_F', tgt['IF'], 'I -> F')
        xI = check(tgt['FI'] / (kF * ((1-xF)*c[F] + xF*d[F])), 'pi_I', tgt['FI'], 'F -> I')
        if max(abs(xF-eF), abs(xI-eI)) < tol: break
        eF, eI = xF, xI

    return dict(pi_F=eF, pi_I=eI, pi_UF=uF, pi_UI=uI)


def _assemble(C, A, D, pr, qb, nT, nE, Pi_b, Pi_e):
    # Pi and the Sector Flows, contracting over assets without ever forming P.
    M = nS * nF * nT

    piF  = np.repeat([pr['pi_F'], pr['pi_F'], pr['pi_UF']], nF * nT)[:, None]
    piI  = np.repeat([pr['pi_I'], pr['pi_I'], pr['pi_UI']], nF * nT)[:, None]
    dlt  = np.repeat([pr['delta_F'], pr['delta_I'], 0.0], nF * nT)
    keep = 1 - dlt

    stay = ((1-piF)*(1-piI) + piF*(1-piI)*C[:, :, 0]
            + (1-piF)*piI*C[:, :, 2] + piF*piI*C[:, :, 4])
    a_F  = piF * ((1-piI)*A[0] + piI*A[1])        # by destination draw theta'
    a_I  = piI * ((1-piF)*A[2] + piF*A[3])

    # BF Test, then sector moves, averaged over its assets
    ar = np.arange(M)
    s_, t_ = ar // (nF * nT), ar % nT
    X  = np.zeros((nB, nE, M, M))
    (st, aF, aI, q), tot = zip(*(_pool(x, D, qb, nT, nE) for x in (stay, a_F, a_I, 0 * stay + 1)))
    for g in range(nF):
        X[:, :, ar, (s_*nF + g)*nT + t_]          += keep * st[g]
        X[:, :, :, (F*nF + g)*nT + np.arange(nT)] += np.moveaxis(keep * aF[g], 2, -1)
        X[:, :, :, (I*nF + g)*nT + np.arange(nT)] += np.moveaxis(keep * aI[g], 2, -1)
        X[:, :, ar, (U*nF + g)*nT + t_]           += dlt * q[g]       # layoffs

    # Order:   s (x) b (x) theta (x) beta (x) e
    # Pi[(s,b,t,b,e),(s',b',t',b',e')] = X[b,e,(s,b,t),(s',b',t')] * Pi_beta[beta,beta'] * Pi_e[e,e']
    Pi = np.einsum('beMN,bB,eE->MbeNBE', X, Pi_b, Pi_e).reshape(M*nB*nE, M*nB*nE)
    flow = np.einsum('bem,bemN->mN', tot[0], X).reshape(nS, nF*nT, nS, nF*nT).sum((1, 3))
    return Pi, flow / flow.sum(1)[:, None]


# ---------------------------------------------------------------------------
# 2. Steady State
# ---------------------------------------------------------------------------
def solve_ss(hank_block, calib, flows=None, counterfactual=False, verbose=False, strict=True,
             damp=0.4, tol=1e-6, stol=1e-4, atol=1e-5, maxit=500, bmaxit=5000):
    """Solve the Steady-State by iterating the value function and the transition matrix.
    1) Household Solve:       expensive step, slow;
    2) Market Clear:          closed form, invert exactly, very fast;
    3) Asset Market:          calibrates beta_high to A = B + p_e, secant method;
    4) Pi_s and BF:           internally calibrate Pi_s and BF.

    flows           :  dict,      {'FI': value, ...} to be matched
    counterfactual  :  bool,      True to B clears asset_market and tau_l the budget
    strict          :  bool,      False returns the last iterate instead of raising
    damp            :  [0,1],     damp < 1 to avoid stuck, but converges slower
    tol, stol       :  value,     tolerance for 'Pi' and for the 'BF' rule
    atol            :  value,     tolerance for 'asset_mkt / B'."""

    start = time.time()

    # Import Grids
    c = {**unknowns, **calib}
    c['tau_ss'], c['B_ss'] = c['tau_l'], c['B']
    nE, nA, nT = c['nE'], c['nA'], c['nT']

    e_grid, Pi_e, a_grid, thetaF, probF, thetaI, probI =\
        make_egrid(c['rho_e'], c['sd_e'], nE, c['amin'], c['amax'], nA,
                   c['sigma_F'], c['mu_I'], c['sigma_I'], nT)

    _, Pi_b = make_bgrid(c['beta_high'], c['dbeta'], c['omega_I'], c['q'], nE, nT)
    grid  = (nT, nE, Pi_b, Pi_e)
    tgrid = (e_grid, thetaF, thetaI, a_grid)

    # Start from a given Pi and BF test
    N = nS * nF * nT * nB * nE
    if np.shape(c.get('Pi')) == (N, N) and np.shape(c.get('Qb')) == (N,):
        Pi, qb = c['Pi'], c['Qb']
    else:
        qb  = _test(c, np.ones((N, nA)), *tgrid)
        one = np.ones((N, nA))
        Pi, _ = _assemble(*_choice(0*one, one, c['sig'], nT, nE, probF, probI), one, c, qb, *grid)

    dPi = dbf = dA = np.inf; xb = fb = None; bgain = 0.005

    # Guess Pi -> Solve -> Read V -> Rebuild Pi -> Repeat until Pi converges.
    for it in range(maxit):
        c['Pi'], c['Qb'] = Pi, qb
        ss = SteadyStateDict(c)
        ss.update(hank_block.steady_state(c, options={'household': dict(backward_maxit=bmaxit)}))
        hhi = ss.internals['household']

        # Solving unknowns to clear markets analytically
        if 'L_hat' in ss:
            c['L'], c['Tr'] = float(ss['L_hat']), float(ss['Tr_hat'])
            c['h_I'] = np.sqrt(c['h_I'] * float(ss['h_I_hat']))   # informal union: half-step to avoid stuck
            dA = abs(float(ss['asset_mkt']))
            if counterfactual:
                # psi, beta and Z_I structural; hours, w_I, B and tau_l adjust
                half = lambda x, y: float(np.sqrt(x * y))            # half-step, or it cycles
                c['h_F'], c['w_I'] = half(c['h_F'], ss['h_F_hat']), half(c['w_I'], ss['w_I_hat'])
                c['B'] = c['B_ss'] = 0.5 * c['B'] + 0.5 * float(ss['A'] - ss['p_e'])
                c['tau_l'] = c['tau_ss'] = float(np.clip(            # G fixed, tax residual
                    0.5 * (c['tau_l'] + float(ss['tau_l_hat'])), 0.0, 0.9))
            else:
                c['G'] = float(ss['G_hat'])                      # G residual at the calibration
                c['Z_I'] = float(ss['Z_I_hat'])                  # w_I = w in the SS
                c['Phi'] = float(ss['Phi_hat'])                  # p_e pinned to A / GDP
                c['psi'] = float(ss['psi_hat'])                  # hours normalized: psi backed out
                c['B'] = c['B_ss'] = float(ss['B_hat'])          # debt pinned to B / GDP
                xn, fn = c['beta_high'], float(ss['asset_mkt'])  # dA/dbeta > 0
                if xb is not None and abs(fn - fb) > 1e-12:
                    g = (xn - xb) / (fn - fb)                    # dbeta/dA > 0; if not, Pi moved A
                    if g > 0: bgain = float(np.clip(g, 1e-4, 0.05))
                xb, fb = xn, fn
                c['beta_high'] = float(np.clip(xn - np.clip(bgain*fn, -0.02, 0.02), 0.5, 1.0))

        # Acceptance rate from `V` and rates `pi_s`
        C, A = _choice(hhi['V'], hhi['Va'], c['sig'], nT, nE, probF, probI)
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
            dbf = 0.0

        Pi_new, P_s = _assemble(C, A, hhi['D'], c, qb, *grid)
        dPi = np.max(np.abs(Pi_new - Pi)); Pi = damp * Pi + (1-damp) * Pi_new    # damp to avoid stucking

        # Iterate until converges
        if verbose:
            print(f"[Pi loop] it {it:3d}     |dPi|={dPi:.1e}" + (
                  f"  tau_l={c['tau_l']:.4f}  B={c['B']:.3f}  w_I={c['w_I']:.4f}"
                  f"  h_F={c['h_F']:.4f}  I={float(ss['I']):.4f}" if counterfactual else ""))
        if np.isfinite(hhi['Va']).all():
            _HH_WARM[(hhi['Va'].shape[0], hhi['Va'].shape[1])] = \
                (hhi['Va'].copy(), hhi['V'].copy(), hhi['Vc'].copy())
        
        # Converged: one last solve
        if dPi < tol and dbf < stol and dA < atol * abs(c['B']):
            c['Pi'], c['Qb'] = Pi, qb
            ss = SteadyStateDict(c)
            ss.update(hank_block.steady_state(c, options={'household': dict(backward_maxit=bmaxit)}))
            ss.internals['household']['P'] = P_s
            tdiff = time.time() - start
            if verbose:
                print(f"Steady State solved in {tdiff:.1f}s ({tdiff/60:.1f}min),  " +
                      "  ".join(f"{k}={c[k]:.4f}" for k in (*pi_calib, *bf_calib) if flows is not None))
            return ss
        
    # If didn't converge, say why. `strict=False` hands back a near-converged iterate
    crit  = (('|dPi|', dPi, tol), ('|dbf|', dbf, stol), ('|A-B|', dA, atol * abs(c['B'])))
    stuck = [f'{n}={v:.1e}>{t:.1e}' for n, v, t in crit if v >= t]
    msg   = f"     Steady state stalled in {maxit} it: " + ", ".join(stuck)
    if strict or max(v / t for _, v, t in crit) > 1e3:
        raise RuntimeError(msg)
    print(msg, flush=True)
    ss.internals['household']['P'] = P_s
    return ss


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


def hike(pps, rho, T, n, delay=0):
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


PRC = ('w', 'w_I', 'h_F', 'h_I')      # Prices the BF means test is evaluated at

def solve_dyn(hank, ss, shock, dZ, unknowns, targets, calib, var, moving=True,
              damp=0.5, gpi=0.75, tol=1e-6, ptol=1e-5, maxit=150, verbose=False, jac=None):
    """Solve the Dynamics by iteration the transition matrix.

    shock           :  string,    the exogenous variable input name ('Tr', 'rstar')
    dZ              :  list,      the shock's path, normally an AR(1)
    unknowns        :  list,      variables that moves, solving `targets`
    moving          :  bool,      allow Pi_s to move, endogeneously
    damp, gpi       :  [0,1],     damp < 1 to avoid stuck, but converges slower
    tol, ptol       :  value,     tolerance for the targets and for the Pi path"""

    start, T = time.time(), len(dZ)
    prc = [k for k in PRC if k in unknowns]
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

    U, dpi, worst = ImpulseDict({k: np.zeros(T) for k in unknowns}), 0.0, ''
    for it in range(maxit):
        # One Newton step on the unknowns, given the current Pi path
        td  = hank.impulse_nonlinear(ss, ImpulseDict({**inp, **U}), [*var, *targets],
                                     internals={'household': ['V', 'Va', 'D']}
                                     if moving else {}, Js=Js)
        res = np.array([np.max(np.abs(td[k])) for k in targets])
        err, worst = float(res.max()), list(targets)[int(res.argmax())]
        if not np.isfinite(err):
            raise RuntimeError(f"\n     Dynamics blew up at it {it} on `{worst}`: lower `damp`")

        # MOVING: Pi_t is chosen at the end of t over the t+1 values, at the t prices
        if moving:
            h, dpi = td.internals['household'], 0.0
            dr = np.asarray(td['r']) if 'r' in td else np.zeros(T)
            for t in range(T):
                ct = {**calib, 'r': calib['rstar'] + float(dr[t]),
                      **{k: calib[k] + float(U[k][t]) for k in prc}}
                D  = D_ss + h['D'][t]
                s  = min(t + 1, T - 1)          # the choice at t is over the t+1 values
                C, A = _choice(V_ss + h['V'][s], Va_ss + h['Va'][s], calib['sig'],
                               calib['nT'], calib['nE'], probF, probI)
                dP = _assemble(C, A, D, calib, _test(ct, D, *tgrid), calib['nT'],
                               calib['nE'], Pi_b, Pi_e)[0] - ss['Pi'] - inp['Pi'][t]
                dpi = max(dpi, np.max(np.abs(dP)))
                inp['Pi'][t] += gpi * dP        # Damp to avoid cycling

        U += damp * HU.apply(td)

        tdiff = time.time() - start
        if verbose:
            print(f"[Dyn loop] it {it:3d}  |err|={err:.1e}  |dPi|={dpi:.1e}"
                  f"  ({tdiff:.0f}s, {tdiff/60:.1f}min)")
        if err < tol and dpi < ptol:
            print(f"Dynamics solved in {tdiff:.1f}s ({tdiff/60:.1f}min), {it+1} it.")
            return {v: td[v] for v in var}
    raise RuntimeError(f"\n     Dynamics stalled in {maxit} it on `{worst}`: "
                       f"|err|={err:.1e}>{tol:.1e}, |dPi|={dpi:.1e}>{ptol:.1e}")


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
# 4.1. Transforms, the Simplex, and the Result Files
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


def _fj(J):
    # 59.3 | 223 | 1.0e+06
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


# 4.2. The Objective
class SMM:
    # Objective with caching and an evaluation log.
    def __init__(self, model, calib=None, space=smm_space, se=mom_se, coarse=None,
                 verbose=False, floor=0.05, logpath=None, every=25, tag='', warm=None):
        self.model, self.tag = model, tag
        self.cal0  = {k: v for k, v in (calibration if calib is None else calib).items()
                      if k not in ('Pi', 'Qb')}      # a stale Pi starts every solve far away
        self.space = space
        # Var(gap) = sampling + specification error (floor). W = S^-1, so it is efficient.
        self.S = np.diag([se[k] ** 2 + mom_fl.get(k, floor) ** 2 for k in mom_smm])
        self.W = np.diag(1.0 / np.diag(self.S))
        self.coarse  = coarse or {}                 # e.g. smm_grid; {} is the full grid
        self.logpath, self.every = logpath, every   # checkpoint, a run takes hours
        self.verbose, self.log, self.n = verbose, [], 0
        self._warm = {k: v for k, v in (warm or {}).items() if k in (*update, 'Pi', 'Qb')}
        self._cache, self._best = {}, (np.inf, None)

    def gaps(self, p, coarse=True):
        # Run the model and get moments
        cal = {**self.cal0, **self._warm, **p, **(self.coarse if coarse else {})}
        ss  = solve_ss(self.model, cal, flows=flows, strict=False,
                       maxit=150 if coarse else 400,               # the coarse grid is itself approximate
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


# 4.3. Search
def sobol_starts(n, space=smm_space, seed=seed):
    # n quasi-random starts, uniform inside the parameter box
    from scipy.stats import qmc
    lo, hi = np.array([[s[0], s[1]] for s in space.values()]).T
    u = 0.05 + 0.90 * qmc.Sobol(len(space), scramble=True, seed=seed).random(n)
    return [_to_x(dict(zip(space, lo + ui * (hi - lo))), space) for ui in u]


def screen(obj, n=4, pool=8):
    # Score `pool` Sobol draws on the coarse grid and keep the `n` best. A start that
    # stalls or clips a flow cannot be polished, so it drops to the back of the queue.
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
    # Nelder-Mead from the `n` screened starts, each written to smm_nm{k}.csv.
    # k=None runs all of them in `workers` processes; k=0..n-1 runs just that start.
    # coarse=None: each worker builds its own SMM, so it takes smm_grid like everything else.
    if k is None:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(workers) as ex:
            return list(ex.map(multistart, range(n),
                               *[[v] * n for v in (n, workers, calib, coarse, maxiter, frac)]))

    from sequence_jacobian import create_model
    from code.p1_household import hh_ss
    from code.p2_other_blocks import (firm_formal, firm_informal, nkpc_ss, union_ss,
                                      equity_ss, monetary, fiscal, mkt_clearing, calibrate_ss)
    t0  = time.time()
    obj = SMM(create_model([hh_ss, firm_formal, firm_informal, nkpc_ss, union_ss,
                            equity_ss, monetary, fiscal, mkt_clearing, calibrate_ss]),
              calib=calib, coarse=coarse, verbose=True, tag=f'{k}', logpath=smm_dir + f'smm_log{k}.csv')
    x0 = _to_x(pd.read_csv(smm_path['s'], index_col=0).iloc[k].to_dict(), obj.space)
    print(f"--- Start {k}/{n}: ",
          " ".join(f"{a}={v:.3f}" for a, v in _to_p(x0, obj.space).items()), flush=True)
    try:
        res = minimize(obj, x0, args=(True,), method='Nelder-Mead',
                       options=dict(maxiter=maxiter, xatol=1e-3, fatol=1e-2, adaptive=True,
                                    initial_simplex=_simplex(x0, obj.space, frac)))
        p_hat, J = _to_p(res.x, obj.space), float(res.fun)
    except KeyboardInterrupt:              # Ctrl-C still writes the best point so far
        p_hat, J = obj.best, obj._best[0]
        print(f"\n--- Start {k} interrupted at {obj.n} solves", flush=True)

    pd.Series({**p_hat, 'J': J}, name='value').to_csv(smm_dir + f'smm_nm{k}.csv', index_label='param')
    obj.history().to_csv(obj.logpath)
    print(f"Start {k} done in {(time.time()-t0)/3600:.1f}hrs   J = {_fj(J)}   {obj.n} solves")
    return p_hat


def collect(obj, n=4, k=None, coarse=False):
    # Re-score the starts on the full grid: the coarse J can rank a phantom first.
    # k=None scores them all and saves them;  k=0..n-1 keeps one, re-read from the file.
    path = smm_path['m'].replace('.csv', '_all.pkl')
    tbl  = pd.DataFrame({j: pd.read_csv(smm_dir + f'smm_nm{j}.csv', index_col=0)['value']
                         for j in range(n) if os.path.exists(smm_dir + f'smm_nm{j}.csv')})
    par  = {j: {q: float(tbl[j][q]) for q in obj.space} for j in tbl}   # never a stale column

    if k is not None and os.path.exists(path):             # already scored: no solve needed
        with open(path, 'rb') as fobj: out = pickle.load(fobj)
        if k in out and out[k]['params'] == par[k]:
            print(f"=== Start {k} ===   J = {_fj(out[k]['J'])}   (from {path})")
            save_res(out[k], 'm')
            return out[k]

    out = {}
    for j in (list(tbl) if k is None else [k]):
        try:
            g, mod, ss, cal = obj.gaps(par[j], coarse=coarse)
            tbl.loc['J_full', j] = float(g @ obj.W @ g)
            tbl.loc['dFI', j]    = mod['FI'] - mom_data['FI']     # zero unless a rate clipped
            tbl.loc['pi_max', j] = max(cal[r] for r in pi_calib)  # 1.0 = clipped
            out[j] = dict(params=par[j], x=_to_x(par[j], obj.space), J=float(g @ obj.W @ g),
                          g=g, moments=mod, ss=ss, calibration=cal)
        except RuntimeError as e:
            tbl.loc['J_full', j] = np.inf
            print(f"  start {j} does not solve on the full grid: {e}")

    print("=== Multi-Start ===\n", tbl.round(4).to_string())
    if k is None:                                  # keep them all: pick with collect(obj, k=...)
        with open(path, 'wb') as fobj:
            pickle.dump({j: {a: b for a, b in r.items() if a != 'ss'} for j, r in out.items()}, fobj)
        print(f"  scored starts written to {path}")
        return None                                # pick one with collect(obj, k=...)
    if k not in out:
        raise RuntimeError(f"\n     Start {k} does not re-score")
    save_res(out[k], 'm')
    return out[k]


def pre_smm(model, res, blocks=pre_block, calib=None, coarse=None, maxiter=120):
    # One block at a time, each scored on the whole moment set, with a warm starter.
    p, warm = dict(res.get('params', res)), res.get('calibration')
    for b in [[k for k in b if k in smm_space] for b in blocks]:
        if not b: continue                           # a parameter held fixed is just skipped
        obj = SMM(model, calib={**(calib or calibration), **p}, coarse=coarse, verbose=True,
                  space={k: smm_space[k] for k in b}, tag='pre', warm=warm)
        x0, t0 = _to_x(p, obj.space), time.time()
        try:
            minimize(obj, x0, args=(True,), method='Nelder-Mead',
                     options=dict(maxiter=maxiter, xatol=1e-3, fatol=1e-2, adaptive=True,
                                  initial_simplex=_simplex(x0, obj.space)))
        except KeyboardInterrupt:
            print("\n--- Interrupted", flush=True)
        p.update(obj.best or {})
        warm = obj._warm                             # hand the converged rates to the next block
        print(f"--- Pre SMM [{', '.join(b)}]   J = {_fj(obj._best[0])}   {obj.n} solves"
              f"   ({(time.time()-t0)/60:.0f}min)\n    "
              + "  ".join(f"{k}={p[k]:.4f}" for k in b), flush=True)
    return p


def polish(obj, p0=None, coarse=None, maxiter=60, frac=0.03):
    t0, keep_c = time.time(), obj.coarse
    x0 = _to_x(p0 or {k: guess[k] for k in obj.space}, obj.space)
    print("\n--- Local Polish from:",
          " ".join(f"{k}={v:.3f}" for k, v in _to_p(x0, obj.space).items()), flush=True)
    obj.coarse = dict(coarse) if coarse else {}     # the full grid, unless asked otherwise
    obj._cache.clear(); obj._best = (np.inf, None)
    try:
        x0 = minimize(obj, x0, args=(True,), method='Nelder-Mead',
                      options=dict(maxiter=maxiter, xatol=1e-3, fatol=1e-2, adaptive=True,
                                   initial_simplex=_simplex(x0, obj.space, frac))).x
    except KeyboardInterrupt:              # Ctrl-C scores the best point so far
        x0 = _to_x(obj.best, obj.space)
        print("\n--- Interrupted at", obj.n, "solves", flush=True)

    p_hat = _to_p(x0, obj.space)
    g, mod, ss, cal = obj.gaps(p_hat, coarse=False)     # score on the full grid
    J = float(g @ obj.W @ g)
    print(f"\nPolish done in {(time.time()-t0)/60:.0f}min "
          f"({(time.time()-t0)/3600:.1f}hrs)   J = {_fj(J)}   {obj.n} solves")
    res = dict(params=p_hat, x=x0, J=J, g=g, moments=mod, ss=ss, calibration=cal)
    save_res(res, 'f')
    obj.coarse = keep_c
    return res


# 4.4. Identification
MID = [i for i, k in enumerate(mom_smm) if k not in mom_flow]   # Flows match by construction
MOM = [mom_smm[i] for i in MID]
_drop_flows = lambda G, W: (G[MID], W[np.ix_(MID, MID)])        # Remove G rows for flows


def jacobian(obj, p_hat, step=0.05, coarse=False, tries=3):
    # G = dg/dtheta, shape (nMoments, nParams). A perturbed point can have no steady
    # state: halve the step, and fall back to one-sided when only one side solves.
    g0, *_ = obj.gaps(p_hat, coarse)
    G = []
    for k, (lo, hi, _) in obj.space.items():
        t0, one = time.time(), None
        for i in range(tries):                     # shrink first: one-sided is a last resort
            h = step * (hi - lo) / 2 * 0.5 ** i
            d = {}
            for x in (min(p_hat[k] + h, hi), max(p_hat[k] - h, lo)):
                try: d[x] = obj.gaps({**p_hat, k: x}, coarse)[0]
                except RuntimeError: pass
            if len(d) == 2: break
            if d: one = next(iter(d.items()))      # keep the smallest side that solved
        if len(d) == 2:
            (x0, g_0), (x1, g_1) = sorted(d.items())
            G.append((g_1 - g_0) / (x1 - x0))
        elif one is not None:
            G.append((one[1] - g0) / (one[0] - p_hat[k]))
        else:
            raise RuntimeError(f"\n     No steady state either side of {k}={p_hat[k]:.4f}")
        print(f"  [G] {k:8s} h={h:.4f}  {len(d) if len(d) == 2 else 1}-sided"
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


# 4.5. Inference
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


def inference(res, obj, G=None, step=0.05, coarse=False, scale=True,
              savepath=None, label='tab:smm_se'):
    # Classical Minimum Distance: S is already Var(moment), so n_obs = 1. G on the full grid.
    from scipy.stats import norm
    from code.p7_results import _tex_table
    if G is None: G = jacobian(obj, res['params'], step=step, coarse=coarse)
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

    esc  = lambda x: x.replace('_', r'\_')
    rows = [f'{esc(n)} & {r.estimate:.3f} & {r["s.e."]:.3f} & {r.t:.2f} & {r.p:.3f}'
            f' & {r.lower:.2f} & {r.upper:.3f}' for n, r in tbl.iterrows()]
    rows += [r'\midrule', rf'Hansen $J$ & \multicolumn{{6}}{{c}}'
             rf'{{{stat:.2f}  (df {df}, $p$ = {pval:.3f})}}']
    _tex_table([r'Parameter & Estimate & S.E. & $t$ & $p$ & Lower & Upper'], rows,
               'SMM Estimates and Over-identification Test', label, 'lcccccc', savepath=savepath)


def report(res):
    mod  = res['moments']
    keys = [k for k in mom_data if k not in mom_fix]      # drop the by-construction ones
    name = lambda k: k if k in mom_smm else k + '*'
    tbl = pd.DataFrame([(name(k), mom_data[k], mod[k], mod[k] - mom_data[k]) for k in keys],
                       columns=['Moment', 'Data', 'Model', 'Gap']).set_index('Moment').round(4)
    print("=== Moments (* = Untargeted) ===\n", tbl.to_string())

    cap = [r for r in pi_calib if res['calibration'][r] >= 1.0 - 1e-9]
    bad = {k: mod[k] - mom_data[k] for k in mom_flow if abs(mod[k] - mom_data[k]) > 1e-6}
    if bad or cap:                                   # the flows are identities, not targets
        print("  !Flows not matched: " + "  ".join(f"{k}={v:+.4f}" for k, v in bad.items())
              + (f"     capped: {', '.join(cap)}" if cap else ""))


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from sequence_jacobian import create_model
    from code.p1_household import hh_ss
    from code.p2_other_blocks import *

    hank_ss = create_model([hh_ss, firm_formal, firm_informal, nkpc_ss, union_ss,
                            equity_ss, monetary, fiscal, mkt_clearing, calibrate_ss])
    p0  = {k: guess[k] for k in smm_space}
    obj = SMM(hank_ss, coarse=smm_grid, verbose=True)

    # 0. Identification and Sensibility
    # G = jacobian(obj); identification(G, obj.W); sensitivity(G, obj.W)

    # 1. Multi-Start:   python -m code.p6_solve
    # screen(obj, n=4, pool=16)
    # multistart(n=4, workers=4)
    # collect(obj, n=4)
    # res = collect(obj, k=3); report(res)        # pick the start k

    # 2. Polish
    # res = load_res('m')
    # p   = pre_smm(hank_ss, res)
    # res = polish(obj, p0=res['params'], coarse=dict(nA=50))
    # report(res)

    # 3. Identification and Inference
    res = load_res()
    p = pre_smm(hank_ss, res, blocks=[['omega_I', 'dbeta']], coarse=smm_grid)
    G = jacobian(obj, res['params'], step=0.02); identification(G, obj.W); sensitivity(G, obj.W)
    inference(res, obj, G, savepath="output/tables/smm_se.tex")

