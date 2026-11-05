#=
#----------------------------------------------------------------------------
# DESCRIPTION
# Solve the model: Endogenous Sector Transition, Steady State and Dynamics.
# ---------------------------------------------------------------------------
#=

# ---- Packages -------------------------------------------------------------
import os, time, pickle
import numpy as np
from sequence_jacobian.classes import SteadyStateDict
from sequence_jacobian.classes.impulse_dict import ImpulseDict
from sequence_jacobian.classes.jacobian_dict import FactoredJacobianDict

from code.p1_household import make_egrid, make_bgrid, bf_income, bf_test, nS, nB, nF, _HH_WARM
from code.p5_calibration import *


# ---------------------------------------------------------------------------
# 1. Endogenous Sector Transition
# ---------------------------------------------------------------------------
CH = 7        # Channels by offer (F, I, Both) and destination (0, 1, 2)

def _softmax(Vals, sig):
    # Smooth max{Vstay, VF, VI, VU, ...} (sig -> 0 = hard max).
    V = np.stack(np.broadcast_arrays(*Vals), 0)
    Probs = np.exp((V - V.max(0)) / np.maximum(sig, 1e-10))
    return Probs / Probs.sum(0)


def _test(c, D, e_grid, thetaF, thetaI, a_grid):
    # Pooled BF Test P[b' = 1 | state], averaged over own assets
    yF = c['w'] * c['h_F']
    q  = bf_test(bf_income(e_grid, thetaF, thetaI, a_grid,
                           c['w_I'] * c['h_I'] / yF, c['rstar'] / yF),
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

    check = lambda x: float(np.clip(x, 1e-6, 1.0))   # the flows are targets, not identities

    kF, kI = 1 - p['delta_F'], 1 - p['delta_I']      # layoffs are exogenous
    a, b, c, d = S[1], S[5], S[3], S[6]              # cF1 cB1 cI1 cB2, by origin

    # Unemployed: a 2 x 2 block, independent of the rest
    uF, uI = p['pi_UF'], p['pi_UI']
    for _ in range(it):
        xF = check(tgt['UF'] / ((1-uI)*a[U] + uI*b[U]))
        xI = check(tgt['UI'] / ((1-uF)*c[U] + uF*d[U]))
        if max(abs(xF-uF), abs(xI-uI)) < tol: break
        uF, uI = xF, xI

    # Employed: pi_F and pi_I, a 2 x 2 block net of the layoffs
    eF, eI = p['pi_F'], p['pi_I']
    for _ in range(it):
        xF = check(tgt['IF'] / (kI * ((1-eI)*a[I] + eI*b[I])))
        xI = check(tgt['FI'] / (kF * ((1-xF)*c[F] + xF*d[F])))
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
             damp=0.0, tol=1e-6, stol=1e-4, atol=1e-5, maxit=500, bmaxit=5000, warm=True):
    """Solve the Steady-State by iterating the value function and the transition matrix.
    1) Household Solve:       expensive step, slow;
    2) Market Clear:          closed form, invert exactly, very fast;
    3) Asset Market:          calibrates beta_high to A = B + p_e, secant method;
    4) Pi_s and BF:           internally calibrate Pi_s and BF.

    flows           :  dict,      {'FI': value, ...} to be matched; ignored in a counterfactual
    counterfactual  :  bool,      True holds the calibrated parameters (offer rates, BF rule, psi, beta,
                                  w_I); hours, B and tau_c adjust, and sector choice reallocates
    strict          :  bool,      False returns the last iterate instead of raising
    damp            :  [0,1],     damp > 0 to avoid cycling (but converges slower)
    tol, stol       :  value,     tolerance for 'Pi' and for the 'BF' rule
    atol            :  value,     tolerance for 'asset_mkt / B'."""

    start = time.time()

    # Import Grids
    c = {**unknowns, **calib}
    c['tau_css'], c['B_ss'] = c['tau_c'], c['B']
    if counterfactual: flows = None                    # only choices adjust, the offer rates stay
    else: c['w_I'] = c['w']                            # w_I = w in the SS; a counterfactual keeps it
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
            c['h_I'] = np.sqrt(c['h_I'] * float(ss['h_I_hat']))  # half-step to avoid stuck
            dA = abs(float(ss['asset_mkt']))
            if counterfactual:
                # psi, beta and w_I structural; hours, B and tau_c adjust
                c['h_F'] = float(np.sqrt(c['h_F'] * ss['h_F_hat']))   # half-step, or it cycles
                c['B'] = c['B_ss'] = 0.5 * c['B'] + 0.5 * float(ss['A'] - ss['p_e'])
                c['tau_c'] = c['tau_css'] = float(np.clip(       # G fixed, tax residual
                    0.5 * (c['tau_c'] + float(ss['tau_c_hat'])), 0.0, 0.9))
            else:
                c['G'] = float(ss['G_hat'])                      # G residual at the calibration
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
                if c[k] <= 0: continue               # no-BF counterfactual: nothing to calibrate
                b = (mom_data[agg[:4]] / max(float(ss[agg]) / float(ss[sh]), 1e-12)) ** (sgn / 2)
                x = float(np.clip(c[k] * b, 1e-4, 1.0))           # probabilities
                dbf, c[k] = max(dbf, abs(np.log(x / c[k]))), x    # at a bound: the closest fit
        else:
            dbf = 0.0

        Pi_new, P_s = _assemble(C, A, hhi['D'], c, qb, *grid)
        gap = np.abs(Pi_new - Pi).max(1)                 # by origin state
        mass = hhi['D'].sum(1)
        dPi = gap[mass > 1e-10].max()                    # states without mass cannot matter
        Pi = damp * Pi + (1-damp) * Pi_new           # damp to avoid stucking

        # Iterate until converges
        if verbose:
            print(f"[Pi loop] it {it:3d}     |dPi|={dPi:.1e}" + (
                  f"  tau_c={c['tau_c']:.4f}  B={c['B']:.3f}"
                  f"  h_F={c['h_F']:.4f}  I={float(ss['I']):.4f}" if counterfactual else ""))
        if warm and np.isfinite(hhi['Va']).all():
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

# ---- Helpers --------------------------------------------------------------
PRC = ('w', 'h_F', 'h_I')             # Prices the BF means test is evaluated at

def ar1(size, rho, T, delay=0):
    # AR(1) shock path, announced `delay` quarters in advance.
    dZ = size * rho ** np.arange(T)
    return np.r_[np.zeros(delay), dZ[:T-delay]]


def _dyn_jacobian(hank, ss, unknowns, targets, inp, T, cache):
    # Cache Jacobians at Steady State to avoid recomputing in Dyn.
    key = (tuple(unknowns), tuple(targets), tuple(sorted(inp)), T)
    if key not in cache:
        Js = hank.partial_jacobians(ss, set(unknowns) | set(inp), set(targets), T)
        cache[key] = Js, FactoredJacobianDict(hank.jacobian(ss, unknowns, targets, T, Js), T)
    return cache[key]


# ---- Solver ---------------------------------------------------------------
def _anderson(X, F, m):
    # Anderson solver: next point of x -> x + f(x) from the last m differences.
    if len(F) < 2: return X[-1] + F[-1]
    dF, dX = np.diff(F[-m-1:], axis=0).T, np.diff(X[-m-1:], axis=0).T
    g = np.linalg.lstsq(dF, F[-1], rcond=None)[0]
    return X[-1] + F[-1] - (dX + dF) @ g


def solve_dyn(hank, ss, shock, dZ, unknowns, targets, calib, var, moving=True, jac=None,
              strict=True, gpi=0.15, kpi=40, mU=5, damp=1.0, wdw=None, tol=1e-4, ptol=1e-3,
              utol=1e-3, max_pi=None, maxit=300, verbose=False, adapt=True, patience=None):
    """Solve the Dynamics by iterating the transition matrix Pi.
    Pi stops once it has settled, then we polish the targets U.

    shock           :  string,    the exogenous variable input name ('Tr', 'rstar')
    dZ              :  list,      the shock's path, normally an AR(1)
    unknowns        :  list,      variables that moves, solving `targets`
    moving          :  bool,      allow Pi_s to move, endogeneously
    damp            :  [0,1],     damp < 1 makes jacobian lower. damp = 1 is full Newton
    gpi             :  [0,1],     Pi step size at the first step; `adapt` then moves it
    adapt           :  bool,      False keeps `gpi` fixed; True grows it 10% while dU falls
    patience        :  int,       Pi steps without a 2% fall of dU before it stops
    kpi             :  int,       MAX steps of U per Pi step (Pi waits for err < tol until then)
    mU              :  int,       Anderson depth on U (0 = plain Newton)
    max_pi          :  int,       stop after this many Pi steps, None runs until converged
    wdw             :  int,       quarters that must converge, None is all of T
    tol, ptol, utol :  value,     target residuals; tol and ptol scale with the shock; dU is the path
                                  move of U per unit of Pi step (what a full step would move)
    strict          :  bool,      False hands back the last iterate instead of raising"""

    t0, T = time.time(), len(dZ)
    sh = float(np.max(np.abs(dZ)))
    tol, ptol = tol * sh, ptol * sh
    win = T if wdw is None else wdw
    prc = [k for k in PRC if k in unknowns]
    inp = {shock: dZ}
    if moving: inp['Pi'] = np.zeros((T,) + ss['Pi'].shape)     # dP = Change in Pi_s
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
        tgrid, mass = (e_grid, thetaF, thetaI, a_grid), D_ss.sum(1)

    flat = lambda d: np.concatenate([np.asarray(d[k]) for k in unknowns])
    U   = ImpulseDict({k: np.zeros(T) for k in unknowns})
    U_p = {k: np.zeros(T) for k in unknowns}
    X, Fh = [], []
    best, bstep, pstep, bvar = np.inf, 0, 0, None
    err, nt, npi, dpi, dU = np.inf, 0, 0, 0.0, (np.inf if moving else 0.0)
    
    for it in range(maxit):
        settled = dpi < ptol and dU < utol
        upd = moving and not settled and (err < tol or nt >= kpi)
        td  = hank.impulse_nonlinear(ss, ImpulseDict({**inp, **U}), [*var, *targets],
                                     internals={'household': ['V', 'Va', 'D']} if upd else {}, Js=Js)
        res = np.array([np.max(np.abs(td[k])) for k in targets])
        err, worst = float(res.max()), list(targets)[int(res.argmax())]
        if not np.isfinite(err): raise RuntimeError(f"blew up at it {it} on `{worst}`")
        tag = ''

        # MOVING: Pi_t is chosen at the end of t over the t+1 values, at the t prices.
        if upd:
            h, sc = td.internals['household'], np.zeros(T)
            dPs = np.zeros((win,) + ss['Pi'].shape)         # the gaps in the window
            d = np.array([np.abs(U[k][:win] - U_p[k][:win]) / max(float(np.max(np.abs(U[k][:win]))), 1e-12)
                          for k in unknowns])               # U moved by the last Pi step
            dn = float(d.max()) / gpi if npi else float(d.max())    # per unit of Pi step: a small gain cannot hide it
            if adapt and npi > 1: gpi = float(np.clip(gpi * (0.5 if dn > 1.05 * dU else 1.1), 0.01, 0.5))
            dr = np.asarray(td['r']) if 'r' in td else np.zeros(T)
            for t in range(T):
                ct = {**calib, 'r': calib['rstar'] + float(dr[t]),
                      **{k: calib[k] + float(U[k][t]) for k in prc}}
                D  = D_ss + h['D'][t]
                s1 = min(t + 1, T - 1)          # the choice at t is over the t+1 values
                C, A = _choice(V_ss + h['V'][s1], Va_ss + h['Va'][s1], calib['sig'],
                                 calib['nT'], calib['nE'], probF, probI)
                dP = _assemble(C, A, D, calib, _test(ct, D, *tgrid), calib['nT'],
                                 calib['nE'], Pi_b, Pi_e)[0] - ss['Pi'] - inp['Pi'][t]
                if t < win: dPs[t] = dP
                else: inp['Pi'][t] += gpi * dP
                sc[t] = (np.abs(dP) * mass[:, None]).max()
            inp['Pi'][:win] += gpi * dPs
            dpi, dU, U_p = float(sc[:win].max()), dn, {k: np.array(U[k]) for k in unknowns}
            ku, tu = np.unravel_index(d.argmax(), d.shape)
            npi += 1
            tag = (f"  Pi step {npi}: |dPi|={dpi:.1e} @t{int(sc[:win].argmax())}"
                   f"  |dU/g|={dU:.1e} on `{list(unknowns)[ku]}` @t{tu}  gain={gpi:.2f}")
        if verbose:
            print(f"  it {it:3d}  |err|={err:.1e} on `{worst}`{tag}  ({(time.time()-t0)/60:.1f}min)", flush=True)

        if upd:                              # keep the best iterate; stop once dU stops falling
            if dU < best:
                if dU < 0.98 * best: pstep = npi             # a fall of 2% or more resets the patience
                best, bstep, bvar = dU, npi, {v: np.array(td[v]) for v in var}
            if patience and npi - pstep >= patience:
                print(f"Dynamics stagnated: lowest dU={best:.1e} at Pi step {bstep} ({(time.time()-t0)/60:.1f}min).")
                return bvar
        ok = err < tol and dpi < ptol and dU < utol
        if ok or (upd and max_pi and npi >= max_pi):
            t_total = time.time() - t0
            print(f"Dynamics {'solved' if ok else 'stopped at max_pi'} in {t_total:.1f}s "
                  f"({t_total/60:.1f}min), {it+1} it." + (f" Lowest dU={min(best, dU):.1e}." if moving else ""))
            return {v: td[v] for v in var} if ok or bvar is None else bvar
        if upd: err, nt, X, Fh = np.inf, 0, [], []
        else:
            step = HU.apply(td)
            if mU:
                X, Fh = X[-mU-1:] + [flat(U)], Fh[-mU-1:] + [damp * flat(step)]
                x = _anderson(X, Fh, mU)
                U = ImpulseDict({k: x[i*T:(i+1)*T] for i, k in enumerate(unknowns)})
            else: U += damp * step
            nt += 1

    if bvar is not None:                     # unconverged: the lowest dU is the result
        print(f"Dynamics stopped at {maxit} it: lowest dU={best:.1e} at Pi step {bstep}, returned ({(time.time()-t0)/60:.1f}min).")
        return bvar
    bad = [f"|err|={err:.1e}>{tol:.1e}"] * (err >= tol) + \
          [f"|dPi|={dpi:.1e}>{ptol:.1e}"] * (dpi >= ptol) + \
          [f"|dU|={dU:.1e}>{utol:.1e}"] * (dU >= utol)
    msg = f"\n     Dynamics stalled in {maxit} it on `{worst}`: " + ", ".join(bad)
    if strict: raise RuntimeError(msg)
    print(msg, flush=True)
    return {v: td[v] for v in var}


def irf_builder(hank, ss, calib, unknowns, targets, var, wdw=None):
    # IRF after a shock: `benef`, `full` and `realloc`. `Td` keeps the first quarters.
    jac = {}
    def build(shock, dZ, unk=unknowns, targ=targets, v=var,
              split=True, moving=True, verbose=False, **kw):
        run = lambda mv: {x: p for x, p in
                          solve_dyn(hank, ss, shock, dZ, unk, targ, calib, v, moving=mv,
                                    wdw=wdw, verbose=verbose, jac=jac, **kw).items()}
        if not split: return run(moving)
        benef, full = run(False), run(True)
        return dict(benef=benef, full=full, realloc={x: full[x] - benef[x] for x in v})
    return build


def solve_cases(cases, path=None, **kw):
    # Every IRF in one call. cases = {name: (build, shock, dZ, options)}.
    out = pickle.load(open(path, 'rb')) if path and os.path.exists(path) else {}
    for name, (build, shock, dZ, opt) in cases.items():
        if name in out: continue
        print(f"== {name}", flush=True)
        out[name] = build(shock, dZ, **opt, **kw)
        if path:
            with open(path, 'wb') as fobj: pickle.dump(out, fobj)
    return out
