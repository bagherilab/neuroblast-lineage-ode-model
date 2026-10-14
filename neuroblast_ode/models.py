"""ODE right-hand sides for the base model and Models 1–5."""
import numpy as np


def satpow_rate(Vbar, Vth, kmax, n):
    if Vbar <= 0 or Vth <= 0:
        return 0.0
    r = (Vbar / Vth) ** n
    return kmax if r >= 1 else kmax * r


def extract_counts_from_solution(sol):
    """
    Returns (t, NB, GMC, Im, Mat) from a solve_ivp result with y shaped like:
      - 9 vars: [NB, V_NB, GMC, V_GMC, Im, V_Im, Mat, V_Mat, ...] (counts at 0,2,4,6)
      - 8 vars: same pattern without the extra state at the end
      - 4 vars: [NB, GMC, Im, Mat] counts only
    """
    t = sol.t
    Y = sol.y
    r = Y.shape[0]
    if r >= 8:  # includes Model 2 (9) and vols models (8)
        NB  = Y[0]; GMC = Y[2]; Im = Y[4]; Mat = Y[6]
        return t, NB, GMC, Im, Mat
    if r >= 4:
        return t, Y[0], Y[1], Y[2], Y[3]
    # fallback (shouldn't happen)
    z = np.zeros_like(Y[0])
    return t, Y[0], z, z, z


def base_model(t, y, params):
    """
    Base model

    State vector (y):
        N_NB        : Neuroblast (NB) count
        N_GMC       : GMC count
        N_ImNeuron  : Immature neuron count
        N_MatNeuron : Mature neuron count

    Parameters (params):
        k_NB     : NB division rate (1/hour)
        k_GMC    : GMC division rate (1/hour)
        k_Neuron : Neuron maturation rate (1/hour)
        sym_frac : Fraction of NB divisions that are symmetric
    """
    N_NB, N_GMC, N_ImNeuron, N_MatNeuron = y
    k_NB, k_GMC, k_Neuron, sym_frac = params  # Division and maturation rates

    # Neuroblast divisions
    sym_divs = sym_frac * k_NB * N_NB
    asym_divs = (1 - sym_frac) * k_NB * N_NB

    # ODEs
    dN_NB = sym_divs
    dN_GMC = asym_divs - k_GMC * N_GMC
    dN_ImNeuron = 2 * k_GMC * N_GMC - k_Neuron * N_ImNeuron
    dN_MatNeuron = k_Neuron * N_ImNeuron

    return [dN_NB, dN_GMC, dN_ImNeuron, dN_MatNeuron]


def neuroblast_model_no_feedback_satpow(t, y, params):
    """
    Model 1

    State vector (y):
        N_NB   : Neuroblast (NB) count
        V_NB   : Total NB volume
        N_GMC  : GMC count
        V_GMC  : Total GMC volume
        N_Im   : Immature neuron count
        V_Im   : Total immature neuron volume
        N_Mat  : Mature neuron count
        V_Mat  : Total mature neuron volume

    Parameters (params):
        g_NB        : NB growth rate (volume/hour)
        g_GMC       : GMC growth rate (volume/hour)
        k_Neuron    : Neuron maturation rate (1/hour)
        sym_frac    : Fraction of NB divisions that are symmetric
        V_thresh_NB : NB division threshold volume (uM³)
        V_thresh_GMC: GMC division threshold volume (uM³)
        k_max_NB    : Max NB division rate (1/hour)
        k_max_GMC   : Max GMC division rate (1/hour)
        n           : Hill/sat-power exponent for division scaling
        V_floor_NB  : Minimum NB volume allowed for division (uM³)
        V_floor_GMC : Minimum GMC volume allowed for division (uM³)
    """
    (N_NB, V_NB, N_GMC, V_GMC, N_Im, V_Im, N_Mat, V_Mat) = y
    (g_NB, g_GMC, k_Neuron, sym_frac,
     V_thresh_NB, V_thresh_GMC,
     k_max_NB, k_max_GMC, n,
     V_floor_NB, V_floor_GMC) = params

    Vavg_NB  = V_NB / N_NB if N_NB > 0 else 0.0
    Vavg_GMC = V_GMC / N_GMC if N_GMC > 0 else 0.0
    Vavg_Im  = V_Im  / N_Im  if N_Im  > 0 else 0.0

    k_NB  = satpow_rate(Vavg_NB,  V_thresh_NB,  k_max_NB,  n) if (N_NB>0 and Vavg_NB >= V_floor_NB)  else 0.0
    k_GMC = satpow_rate(Vavg_GMC, V_thresh_GMC, k_max_GMC, n) if (N_GMC>0 and Vavg_GMC>= V_floor_GMC) else 0.0

    sym_divs, asym_divs = sym_frac*k_NB*N_NB, (1-sym_frac)*k_NB*N_NB
    dN_NB, dV_NB = sym_divs, g_NB*N_NB - 0.2*asym_divs*Vavg_NB
    dN_GMC, dV_GMC = asym_divs - k_GMC*N_GMC, g_GMC*N_GMC + 0.2*asym_divs*Vavg_NB - k_GMC*N_GMC*Vavg_GMC
    dN_Im, dV_Im = 2*k_GMC*N_GMC - k_Neuron*N_Im, k_GMC*N_GMC*Vavg_GMC - k_Neuron*N_Im*Vavg_Im
    dN_Mat, dV_Mat = k_Neuron*N_Im, k_Neuron*N_Im*Vavg_Im
    return [dN_NB,dV_NB,dN_GMC,dV_GMC,dN_Im,dV_Im,dN_Mat,dV_Mat]


def neuroblast_model_SymDivPullsThresh_NoRecovery_NoNBself(t, y, params):
    """
    Model 2

    State vector (y):
        N_NB     : NB count
        V_NB     : Total NB volume
        N_GMC    : GMC count
        V_GMC    : Total GMC volume
        N_Im     : Immature neuron count
        V_Im     : Total immature neuron volume
        N_Mat    : Mature neuron count
        V_Mat    : Total mature neuron volume
        Vth_eff  : Effective NB division threshold

    Parameters (params):
        g_NB_base     : Base NB growth rate (volume/hour)
        g_GMC         : GMC growth rate (volume/hour)
        k_Neuron      : Neuron maturation rate (1/hour)
        sym_frac      : Fraction of NB divisions that are symmetric
        V_thresh_base : Baseline NB division threshold volume (uM³)
        V_thresh_GMC  : GMC division threshold volume (uM³)
        k_max_NB      : Max NB division rate (1/hour)
        k_max_GMC     : Max GMC division rate (1/hour)
        n             : Hill/sat-power exponent
        V_floor_NB    : NB division floor volume (uM³)
        V_floor_GMC   : GMC division floor volume (uM³)
        delta_thresh  : Threshold decrement per symmetric NB division (uM³)
        V_thresh_min  : Minimum NB threshold volume (lower bound) (uM³)
    """
    (N_NB, V_NB, N_GMC, V_GMC, N_Im, V_Im, N_Mat, V_Mat, Vth_eff) = y
    (g_NB_base, g_GMC, k_Neuron, sym_frac,
     V_thresh_base, V_thresh_GMC, k_max_NB, k_max_GMC, n,
     V_floor_NB, V_floor_GMC, delta_thresh, V_thresh_min) = params

    Vavg_NB  = V_NB / N_NB if N_NB  > 0 else 0.0
    Vavg_GMC = V_GMC / N_GMC if N_GMC > 0 else 0.0
    Vavg_Im  = V_Im  / N_Im  if N_Im  > 0 else 0.0

    Vth_used = max(Vth_eff, V_thresh_min)
    k_NB  = satpow_rate(Vavg_NB,  Vth_used,      k_max_NB,  n) if (N_NB>0  and Vavg_NB  >= V_floor_NB)  else 0.0
    k_GMC = satpow_rate(Vavg_GMC, V_thresh_GMC,  k_max_GMC, n) if (N_GMC>0 and Vavg_GMC >= V_floor_GMC) else 0.0

    sym_divs  = sym_frac * k_NB * N_NB
    asym_divs = (1.0 - sym_frac) * k_NB * N_NB

    dN_NB = sym_divs
    dV_NB = g_NB_base * N_NB - 0.2 * asym_divs * (Vavg_NB if N_NB > 0 else 0.0)

    dN_GMC = asym_divs - k_GMC * N_GMC
    dV_GMC = g_GMC * N_GMC + 0.2 * asym_divs * (Vavg_NB if N_NB > 0 else 0.0) - k_GMC * N_GMC * (Vavg_GMC if N_GMC > 0 else 0.0)

    dN_Im  = 2.0 * k_GMC * N_GMC - k_Neuron * N_Im
    dV_Im  = k_GMC * N_GMC * (Vavg_GMC if N_GMC > 0 else 0.0) - k_Neuron * N_Im * (Vavg_Im if N_Im  > 0 else 0.0)

    dN_Mat = k_Neuron * N_Im
    dV_Mat = k_Neuron * N_Im * (Vavg_Im if N_Im  > 0 else 0.0)

    dVth_eff = - delta_thresh * sym_divs  # no recovery

    return [dN_NB, dV_NB, dN_GMC, dV_GMC, dN_Im, dV_Im, dN_Mat, dV_Mat, dVth_eff]


def neuroblast_model_vol_scaled_growth_only(t, y, params):
    """
    Model 3

    State vector (y):
        N_NB   : NB count
        V_NB   : Total NB volume
        N_GMC  : GMC count
        V_GMC  : Total GMC volume
        N_Im   : Immature neuron count
        V_Im   : Total immature neuron volume
        N_Mat  : Mature neuron count
        V_Mat  : Total mature neuron volume

    Parameters (params):
        g_NB_base     : Base NB growth rate (volume/hour)
        g_GMC         : GMC growth rate (volume/hour)
        k_Neuron      : Neuron maturation rate (1/hour)
        sym_frac      : Fraction of NB divisions that are symmetric
        V_thresh_base : NB division threshold volume (uM³)
        V_thresh_GMC  : GMC division threshold volume (uM³)
        k_max_NB      : Max NB division rate (1/hour)
        k_max_GMC     : Max GMC division rate (1/hour)
        n             : Hill/sat-power exponent
        V_floor_NB    : NB division floor volume (uM³)
        V_ref         : Reference NB volume for growth scaling (uM³)
        alpha_growth  : Exponent controlling NB growth volume-sensitivity
        V_floor_GMC   : GMC division floor volume (uM³)
    """
    (N_NB, V_NB, N_GMC, V_GMC, N_Im, V_Im, N_Mat, V_Mat) = y
    (g_NB_base, g_GMC_, k_Neuron, sym_frac,
     V_thresh_base, V_thresh_GMC,
     k_max_NB, k_max_GMC, n,
     V_floor_NB, V_ref, alpha_growth, V_floor_GMC) = params

    Vavg_NB  = V_NB / N_NB if N_NB  > 0 else 0.0
    Vavg_GMC = V_GMC / N_GMC if N_GMC > 0 else 0.0
    Vavg_Im  = V_Im  / N_Im  if N_Im  > 0 else 0.0

    g_NB_eff = g_NB_base * ((Vavg_NB / V_ref) ** alpha_growth if V_ref > 0 else 1.0)

    k_NB  = satpow_rate(Vavg_NB,  V_thresh_base, k_max_NB,  n) if (N_NB>0  and Vavg_NB  >= V_floor_NB)  else 0.0
    k_GMC = satpow_rate(Vavg_GMC, V_thresh_GMC, k_max_GMC, n) if (N_GMC>0 and Vavg_GMC >= V_floor_GMC) else 0.0

    sym_divs  = sym_frac * k_NB * N_NB
    asym_divs = (1.0 - sym_frac) * k_NB * N_NB

    dN_NB = sym_divs
    dV_NB = g_NB_eff * N_NB - 0.2 * asym_divs * (Vavg_NB if N_NB > 0 else 0.0)
    dN_GMC = asym_divs - k_GMC * N_GMC
    dV_GMC = g_GMC_ * N_GMC + 0.2 * asym_divs * (Vavg_NB if N_NB > 0 else 0.0) \
             - k_GMC * N_GMC * (Vavg_GMC if N_GMC > 0 else 0.0)
    dN_Im  = 2.0 * k_GMC * N_GMC - k_Neuron * N_Im
    dV_Im  = k_GMC * N_GMC * (Vavg_GMC if N_GMC > 0 else 0.0) - k_Neuron * N_Im * (Vavg_Im if N_Im > 0 else 0.0)
    dN_Mat = k_Neuron * N_Im
    dV_Mat = k_Neuron * N_Im * (Vavg_Im if N_Im > 0 else 0.0)
    return [dN_NB, dV_NB, dN_GMC, dV_GMC, dN_Im, dV_Im, dN_Mat, dV_Mat]


def neuroblast_model_vol_scaled_growth_and_symdiv_thresh(t, y, params):
    """
    Model 4

    State vector (y):
        N_NB   : NB count
        V_NB   : Total NB volume
        N_GMC  : GMC count
        V_GMC  : Total GMC volume
        N_Im   : Immature neuron count
        V_Im   : Total immature neuron volume
        N_Mat  : Mature neuron count
        V_Mat  : Total mature neuron volume
        S_sym  : Cumulative symmetric NB divisions (bookkeeping)

    Parameters (params):
        g_NB_base     : Base NB growth rate (volume/hour)
        g_GMC         : GMC growth rate (volume/hour)
        k_Neuron      : Neuron maturation rate (1/hour)
        sym_frac      : Fraction of NB divisions that are symmetric
        V_thresh_base : Baseline NB division threshold (uM³)
        V_thresh_GMC  : GMC division threshold (uM³)
        k_max_NB      : Max NB division rate (1/hour)
        k_max_GMC     : Max GMC division rate (1/hour)
        n             : Hill/sat-power exponent
        V_floor_NB    : NB division floor volume (uM³)
        V_ref         : Reference NB volume for growth scaling (uM³)
        alpha_growth  : Exponent controlling NB growth volume-sensitivity
        V_floor_GMC   : GMC division floor volume (uM³)
        delta_thresh  : Threshold decrement per symmetric division (uM³)
        V_thresh_min  : Minimum NB threshold volume (lower bound) (uM³)
    """
    (N_NB, V_NB, N_GMC, V_GMC, N_Im, V_Im, N_Mat, V_Mat, S_sym) = y
    (g_NB_base, g_GMC, k_Neuron, sym_frac,
     V_thresh_base, V_thresh_GMC,
     k_max_NB, k_max_GMC, n,
     V_floor_NB, V_ref, alpha_growth, V_floor_GMC,
     delta_thresh, V_thresh_min) = params

    # Averages with safe divide
    Vavg_NB  = V_NB / N_NB if N_NB > 0 else 0.0
    Vavg_GMC = V_GMC / N_GMC if N_GMC > 0 else 0.0
    Vavg_Im  = V_Im  / N_Im  if N_Im  > 0 else 0.0

    # Volume-scaled NB growth
    ratio    = (Vavg_NB / V_ref) if V_ref > 0 else 1.0
    g_NB_eff = g_NB_base * (ratio ** alpha_growth)

    # Dynamic threshold pulled by cumulative symmetric divisions
    V_thresh_eff = V_thresh_base - delta_thresh * np.maximum(S_sym, 0.0)
    if V_thresh_min is not None:
        V_thresh_eff = max(V_thresh_eff, V_thresh_min)

    k_NB  = satpow_rate(Vavg_NB,  V_thresh_eff, k_max_NB,  n) if (N_NB>0  and Vavg_NB  >= V_floor_NB)  else 0.0
    k_GMC = satpow_rate(Vavg_GMC, V_thresh_GMC, k_max_GMC, n) if (N_GMC>0 and Vavg_GMC >= V_floor_GMC) else 0.0

    # Fluxes
    sym_divs  = sym_frac * k_NB * N_NB
    asym_divs = (1.0 - sym_frac) * k_NB * N_NB

    # ODEs (same book-keeping as Model 1/3)
    dN_NB = sym_divs
    dV_NB = g_NB_eff * N_NB - 0.2 * asym_divs * (Vavg_NB if N_NB > 0 else 0.0)

    dN_GMC = asym_divs - k_GMC * N_GMC
    dV_GMC = (
        g_GMC * N_GMC
        + 0.2 * asym_divs * (Vavg_NB if N_NB > 0 else 0.0)
        - k_GMC * N_GMC * (Vavg_GMC if N_GMC > 0 else 0.0)
    )

    dN_Im  = 2.0 * k_GMC * N_GMC - k_Neuron * N_Im
    dV_Im  = k_GMC * N_GMC * (Vavg_GMC if N_GMC > 0 else 0.0) - k_Neuron * N_Im * (Vavg_Im if N_Im > 0 else 0.0)

    dN_Mat = k_Neuron * N_Im
    dV_Mat = k_Neuron * N_Im * (Vavg_Im if N_Im > 0 else 0.0)

    # Cumulative symmetric-division “pull” state
    dS_sym = sym_divs

    return [dN_NB, dV_NB, dN_GMC, dV_GMC, dN_Im, dV_Im, dN_Mat, dV_Mat, dS_sym]


def neuroblast_model_NB_self_repression(t, y, params):
    """
    Model 5

    State vector (y):
        N_NB   : NB count
        N_GMC  : GMC count
        N_Im   : Immature neuron count
        N_Mat  : Mature neuron count

    Parameters (params):
        k_NB_max : Maximum NB division rate (1/hour)
        k_GMC    : GMC division rate (1/hour)
        k_Neuron : Neuron maturation rate (1/hour)
        K        : Half-max NB count for repression (effective population scale)
        n_nb     : Hill exponent for repression steepness
        sym      : Fraction of NB divisions that are symmetric
    """
    # y=[N_NB, N_GMC, N_Im, N_Mat]
    N_NB, N_GMC, N_Im, N_Mat = y
    k_NB_max, k_GMC, k_Neuron, K, n_nb, sym = params
    Np=max(N_NB,0.0); K=max(K,1e-12); n_nb=max(n_nb,1e-12)
    k_NB_eff = k_NB_max * (K**n_nb) / (K**n_nb + Np**n_nb)
    sym_divs  = sym * k_NB_eff * N_NB
    asym_divs = (1.0 - sym) * k_NB_eff * N_NB
    dN_NB  = sym_divs
    dN_GMC = asym_divs - k_GMC * N_GMC
    dN_Im  = 2.0 * k_GMC * N_GMC - k_Neuron * N_Im
    dN_Mat = k_Neuron * N_Im
    return [dN_NB, dN_GMC, dN_Im, dN_Mat]
