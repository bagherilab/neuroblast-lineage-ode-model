"""NB cell-cycle duration (CCD = 1/k_NB) reconstructed from model solutions."""
import numpy as np

from .params import (
    n, V_thresh_NB, V_floor_NB, k_max_NB, V_thresh_min, V_thresh_base, delta_thresh,
    k_star, K_self,
)


def _vavg(N, V):  # safe V/N
    return np.divide(V, N, out=np.zeros_like(V), where=(N>0))


def _satpow_core(v_avg, v_thresh, n):
    v_avg = np.asarray(v_avg, dtype=float)
    if np.isscalar(v_thresh):
        # scalar threshold → broadcast safely
        ratio = np.where(v_thresh > 0.0, v_avg / v_thresh, 0.0)
        return np.minimum(1.0, ratio**n)
    else:
        # array threshold (e.g., dynamic Vth in Model 2/4)
        v_thresh = np.asarray(v_thresh, dtype=float)
        core = np.zeros_like(v_avg, dtype=float)
        m = (v_thresh > 0.0)
        core[m] = (v_avg[m] / v_thresh[m])**n
        return np.minimum(1.0, core)


def _ccd_from_k(k):  # 1/k with NaN where k<=0
    return np.where(k > 0, 1.0/k, np.nan)


def ccd_M1(sol):
    N, V = sol.y[0], sol.y[1]
    Vavg  = _vavg(N, V)
    core  = _satpow_core(Vavg, V_thresh_NB, n)
    k_nb  = np.where(Vavg >= V_floor_NB, k_max_NB * core, 0.0)
    return _ccd_from_k(k_nb)


def ccd_M2(sol):
    N, V, Vth_eff = sol.y[0], sol.y[1], sol.y[8]
    Vavg   = _vavg(N, V)
    Vth    = np.maximum(Vth_eff, V_thresh_min)
    core   = _satpow_core(Vavg, Vth, n)
    k_nb   = np.where(Vavg >= V_floor_NB, k_max_NB * core, 0.0)
    return _ccd_from_k(k_nb)


def ccd_M3(sol):
    N, V = sol.y[0], sol.y[1]
    Vavg  = _vavg(N, V)
    core  = _satpow_core(Vavg, V_thresh_base, n)
    k_nb  = np.where(Vavg >= V_floor_NB, k_max_NB * core, 0.0)
    return _ccd_from_k(k_nb)


def ccd_M4(sol):
    N, V, S_sym = sol.y[0], sol.y[1], sol.y[8]
    Vavg   = _vavg(N, V)
    Vth    = np.maximum(V_thresh_base - delta_thresh*np.maximum(S_sym,0.0), V_thresh_min)
    core   = _satpow_core(Vavg, Vth, n)
    k_nb   = np.where(Vavg >= V_floor_NB, k_max_NB * core, 0.0)
    return _ccd_from_k(k_nb)


def ccd_M5(sol, beta):
    N = np.maximum(sol.y[0], 0.0)
    k_eff = k_star * (K_self**beta) / (K_self**beta + N**beta)
    return _ccd_from_k(k_eff)
