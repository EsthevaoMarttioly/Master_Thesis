#=
#----------------------------------------------------------------------------
# DESCRIPTION
# Define the household block of the model, which includes the EGM problem,
# the grid, and transition matrices for income, labor productivity, and sectoral status.
# ---------------------------------------------------------------------------
#=

# ---- Packages --------------------------------------------------------------
import numpy as np
import time, random
from sequence_jacobian import het, interpolate, grids

random.seed(20260415)


# ---------------------------------------------------------------------------
# 1. Utility, Grid, and Income

# 1.1. Utility Functions
def u(c, eis):
    c = np.maximum(c, 1e-12)
    return np.log(c) if eis == 1 else c ** (1-1/eis) / (1-1/eis)

def v(h, psi, varphi):
    return psi * np.maximum(h, 1e-8) ** (1+1/varphi) / (1+1/varphi)

def discretize_normal(mu, sigma, n):
    # theta_s ~ N(mu_s, sigma_s^2) as Gauss-Hermite quadrature with nT nodes.
    z, w = np.polynomial.hermite.hermgauss(n)
    theta = mu + np.sqrt(2) * sigma * z
    prob  = w / np.sqrt(np.pi)
    return theta, prob


# 1.2. Exogenous Transition States Grid
nB = 2     # beta_grid size
nS = 3     # labor_grid size
def make_egrid(rho_e, sd_e, nE, amin, amax, nA,
               mu_F, sigma_F, mu_I, sigma_I, nT):
    # Productivity Grids.
    e_grid, pi_e_e, Pi_e = grids.markov_rouwenhorst(rho=rho_e, sigma=sd_e, N=nE)
    e_grid = e_grid / np.sum(pi_e_e * e_grid)
    thetaF, probF = discretize_normal(mu_F, sigma_F, nT)
    thetaI, probI = discretize_normal(mu_I, sigma_I, nT)

    # Asset Grid
    a_grid = grids.asset_grid(amin=amin, amax=amax, n=nA)
    return e_grid, Pi_e, a_grid, thetaF, probF, thetaI, probI


def make_bgrid(beta_high, dbeta, omega_I, q, nE, nT):
    # Build the beta grid for discount factors.
    beta_low = beta_high - dbeta
    b_grid   = np.array([beta_low, beta_high])
    pi_b = np.array([omega_I, 1-omega_I])
    Pi_b = (1 - q) * np.eye(nB) + q * np.outer(np.ones(nB), pi_b)
    beta = np.tile(np.repeat(b_grid, nE), nS * nT)
    return beta, Pi_b


# 1.3. Labor Income Function
def labor_income(w, w_I, h_F, Div, Tr, e_grid, nE, nT,
                 thetaF, thetaI, y_bar, tau_l, psi, varphi):
    # Dividend Income and Informal Hours
    div_i = np.tile(Div * e_grid, nB*nS*nT)

    e_F = np.exp(thetaF[:, None]) * e_grid[None, :]
    e_I = np.exp(thetaI[:, None]) * e_grid[None, :]

    h_I = (w_I * e_I / np.maximum(psi, 1e-8)) ** (varphi)
    elig = (w_I * e_I * h_I < y_bar).astype(float)     # Elegibility

    y_F = (1 - tau_l) * w * e_F * h_F
    y_I = 1/(1+varphi) * w_I * e_I * h_I + Tr * elig
    y_U = np.full((nT, nE), Tr)

    # Expand the income into beta grid.
    expand = lambda x: np.repeat(x[:, None, :], nB, axis=1).reshape(-1)
    y = np.r_[expand(y_F), expand(y_I), expand(y_U)] + div_i
    return y, h_I, e_F, e_I, elig


# ---------------------------------------------------------------------------
# 2. Endogenous Grid Method (EGM)
_HH_WARM = {}                     # cache in household_block.py
def hh_init(a_grid, y, r, eis):
    key = (y.shape[0], a_grid.shape[0])
    if key in _HH_WARM:           # reuse last converged guess to speed up
        return _HH_WARM[key]
    coh = (1 + r) * a_grid + y[:, None]
    Va  = (1 + r) * (0.1 * coh) ** (-1 / eis)
    V   = u(0.1 * coh, eis) / (1 - 0.96)
    return Va, V


@het(exogenous=['Pi'], policy='a', backward=['Va', 'V'], backward_init=hh_init)
def household(Va_p, V_p, a_grid, y, r, beta, eis):
    c_nextgrid = (beta[:, None] * Va_p) ** (-eis)
    coh = (1 + r) * a_grid + y[:, None]
    a = interpolate.interpolate_y(c_nextgrid + a_grid, coh, a_grid)
    a = np.maximum(a, a_grid[0])
    c_ghh = coh - a
    Va = (1 + r) * c_ghh ** (-1 / eis)
    V = u(c_ghh, eis) + beta[:, None] * V_p
    return Va, V, a, c_ghh


# ---------------------------------------------------------------------------
# 3. Hetoutputs
def sector_shares(c_ghh, e_F, e_I, h_F, h_I, elig):
    block = c_ghh.shape[0] // nS     # nT * nBeta * nE
    f, n_f = np.zeros_like(c_ghh), np.zeros_like(c_ghh)
    i, n_i = np.zeros_like(c_ghh), np.zeros_like(c_ghh)
    u, bf  = np.zeros_like(c_ghh), np.zeros_like(c_ghh)
    
    # Sector Indicator, F=0, I=1, U=2
    f[:block]        = 1.0
    i[block:2*block] = 1.0
    u[2*block:]      = 1.0
    bf[2*block:]     = 1.0

    # Labor Supply = theta * e * h
    expand = lambda x: np.repeat(x[:, None, :], nB, 1).reshape(-1)
    n_f[:block]        = expand(e_F * h_F)[:, None]
    n_i[block:2*block] = expand(e_I * h_I)[:, None]
    bf[block:2*block]  = expand(elig)[:, None]

    return f, i, u, n_f, n_i, bf


# ---------------------------------------------------------------------------
# 4. Endogenous Sector Transition
F, I, U = 0, 1, 2

def _softmax(Vals, sig):
    # Turn "pick the best option" into smooth probabilities (sig -> 0 = hard max).
    V = np.stack(Vals, 0)
    if sig == 0:
        return (V == V.max(0)).astype(float)
    Probs = np.exp((V - V.max(0)) / sig)
    return Probs / Probs.sum(0)


def _status_probs(Vst, p, probF, probI):
    # Transition Matrix Pi_{s,theta} 3*nT x 3*nT for each (beta, e, a).
    nS, nT, nA = Vst.shape
    piF, piI, sig  = p['pi_F'], p['pi_I'], p['sig']
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


def build_Pi(V, D, p, Pi_b, Pi_e, probF, probI):
    # Assemble the full transition matrix:  s (x) theta (x) beta (x) e.
    # `V` is (nS, nT, nBeta, nE, nA):  the value of each sector at (beta,e,a).
    nT, nE, nA = p['nT'], Pi_e.shape[0], V.shape[1]
    Vr = V.reshape(nS*nT, nB, nE, nA)
    Dr = D.reshape(nS*nT, nB, nE, nA)

    Pstat = np.empty((nB, nE, nS*nT, nS*nT))    # Pstat[beta, e, from, to]

    for _beta in range(nB):
        for _e in range(nE):
            P = _status_probs(Vr[:, _beta, _e, :].reshape(nS, nT, nA), p, probF, probI)
            w = Dr[:, _beta, _e, :]
            tot = w.sum(1, keepdims=True)
            w = np.where(tot > 1e-14, w / np.where(tot > 1e-14, tot, 1.0), 1.0 / nA)
            Pstat[_beta, _e] = np.einsum('mna,ma->mn', P, w)    # Average over assets
    
    # Order:   s (x) theta (x) beta (x) e
    # Pi[(s,t,b,e),(s',t',b',e')] = Pstat[b,e,s,s'] * Pi_b[b,b'] * Pi_e[e,e']
    Pi = np.einsum('beMN,bB,eE->MbeNBE', Pstat, Pi_b, Pi_e)
    Pi = Pi.reshape(nS * nT * nB * nE, nS * nT * nB * nE)
    return Pi


# ---------------------------------------------------------------------------
# 5. Household Block

hh = household.add_hetinputs([make_egrid, make_bgrid, labor_income])
hh = hh.add_hetoutputs([sector_shares])


print(f'Inputs: {hh.inputs}')
print(f'Macro outputs: {hh.outputs}')


# ---------------------------------------------------------------------------
# Solve the Transition in Steady State
def solve_ss(hank_block, calib, unknowns=None, targets=None,
             tol=1e-8, maxit=200, verbose=False):
    start = time.time()
    # Import Grids
    nE, nA, nT = calib['nE'], calib['nA'], calib['nT']

    _, Pi_e, _, _, probF, _, probI =\
        make_egrid(calib['rho_e'], calib['sd_e'], nE, calib['amin'],
                   calib['amax'], nA, calib['mu_F'], calib['sigma_F'],
                   calib['mu_I'], calib['sigma_I'], nT)

    _, Pi_b = make_bgrid(calib['beta_high'], calib['dbeta'],
                         calib['omega_I'], calib['q'], nE, nT)

    # Initial guess for Pi.
    Pi = build_Pi(np.zeros((nS*nT*nB*nE, nA)),
                  np.ones((nS*nT*nB*nE, nA)), calib, Pi_b, Pi_e, probF, probI)
    c = dict(calib); diff = 1

    for it in range(maxit):
        # Guess Pi -> Solve -> Read V -> Rebuild Pi -> Repeat until Pi converges.
        c['Pi'] = Pi; gtol = diff
        if unknowns == None or targets == None:     # Equivalent to steady_steate() in SSJ
            ss = hank_block.steady_state(c)
        else:                                       # Equivalent to solve_steady_steate()
            ss = hank_block.solve_steady_state(c, unknowns, targets, solver='hybr',
                                               ttol=gtol, ctol=gtol)
            for k in unknowns:
                c[k] = float(ss[k])
        hhi = ss.internals['household']
        Pi_new = build_Pi(hhi['V'], hhi['D'], calib, Pi_b, Pi_e, probF, probI)
        diff = np.max(np.abs(Pi_new - Pi)); Pi = Pi_new
        if verbose: print(f"[Pi loop] it {it:3d}  |dPi|={diff:.1e}")
        _HH_WARM[(hhi['Va'].shape[0], hhi['Va'].shape[1])] = (hhi['Va'].copy(), hhi['V'].copy())
        if diff < tol:
            print(f"Steady State solved in {\
                time.time()-start:.1f}s ({(time.time()-start)/60:.1f}min).")
            return ss
    raise RuntimeError(f"Pi did not converge in {maxit} iterations.")


# ---------------------------------------------------------------------------
# Run it
if __name__ == "__main__":
    from code.parameters import *
    from code.other_blocks import *
    from sequence_jacobian import create_model

    # Solve the household block in isolation
    calibration_hh = calibration | dict(
        w = 2.3, w_I = 0.64 * 2.3, Div = 0.9, r = calibration['rstar'])

    ss1 = solve_ss(hh, calibration_hh, verbose=True)

    # Solve the full model in Steady State
    hank_ss = create_model([hh, firm_formal, firm_informal, informal_wage,
                            nkpc_ss, union_ss, monetary, fiscal, mkt_clearing])

    ss0 = solve_ss(hank_ss, calibration, unknowns, targets, verbose=True)

    for k in ['A', 'C', 'beta_high', 'psi', 'w', 'w_I', 'Z', 'F', 'I', 'U', 'BF']:
        print(f"  {k:12s} = {ss0[k]:.4f}")


# ---------------------------------------------------------------------------
# Dynamics
def solve_dyn(hank, ss, unknowns, targets, dTr, calib, var,
               moving=True, tol=1e-6, maxit=50, verbose=False):
    # Import grid
    start = time.time()
    T = len(dTr)

    if not moving:
        G = hank.solve_jacobian(ss, unknowns, targets, ['Tr'], T=T)
        return {v: G[v]['Tr'] @ dTr for v in var}

    _, Pi_e, _, _, probF, _, probI =\
        make_egrid(calib['rho_e'], calib['sd_e'], calib['nE'], calib['amin'],
                   calib['amax'], calib['nA'], calib['mu_F'], calib['sigma_F'],
                   calib['mu_I'], calib['sigma_I'], calib['nT'])

    _, Pi_b = make_bgrid(calib['beta_high'], calib['dbeta'], calib['omega_I'],
                         calib['q'], calib['nE'], calib['nT'])
    
    Pi_ss = ss['Pi']; dPi = np.zeros((T,) + Pi_ss.shape)
    V_ss  = ss.internals['household']['V']
    D_ss  = ss.internals['household']['D']

    for it in range(maxit):
        # GE Transition given the Current Pi Path + shock
        td = hank.solve_impulse_nonlinear(ss, unknowns, targets, {'Tr': dTr, 'Pi': dPi},
                                          internals=['household'], verbose=False)

        # MOVING: Rebuild Pi_t from the period-t value/dist, iterate to consistency
        V = V_ss[None] + td.internals['household']['V']   # Levels + Deviations
        D = D_ss[None] + td.internals['household']['D']
        Pi_new = np.stack([build_Pi(V[t], D[t], calib, Pi_b, Pi_e, probF, probI)
                           for t in range(T)])
        dPi_new = Pi_new - Pi_ss
        diff = np.max(np.abs(dPi_new - dPi)); dPi = dPi_new
        if verbose: print(f"[Pi loop] it {it:3d}  |dPi|={diff:.2e}")
        if diff < tol:
            print(f"Dynamics solved in {\
                time.time()-start:.1f}s ({(time.time()-start)/60:.1f}min).")
            return {v: td[v] for v in var}
    raise RuntimeError("Pi path did not converge in {maxit} iterations.")



