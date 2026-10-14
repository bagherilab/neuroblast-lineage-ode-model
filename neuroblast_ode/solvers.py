"""solve_ivp wrappers for each model."""
from scipy.integrate import solve_ivp

from .models import (
    base_model,
    neuroblast_model_no_feedback_satpow,
    neuroblast_model_SymDivPullsThresh_NoRecovery_NoNBself,
    neuroblast_model_vol_scaled_growth_only,
    neuroblast_model_vol_scaled_growth_and_symdiv_thresh,
    neuroblast_model_NB_self_repression,
)
from .params import (
    t_span, t_eval, V0, n, k_max_GMC, V_thresh_NB, V_thresh_GMC, V_floor_NB, V_floor_GMC,
    k_Neuron, g_GMC, k_star, k_max_NB, g_NB_WT,
    V_thresh_base, delta_thresh, V_thresh_min, g_NB_base_WT, K_self,
    base_k_NB, base_k_GMC, base_k_Neuron, base_y0,
)


def solve_base_model(sym_frac):
    params = [base_k_NB, base_k_GMC, base_k_Neuron, sym_frac]
    return solve_ivp(base_model, t_span, base_y0, t_eval=t_eval, args=(params,))


def solve_model1(*, sym_frac=None, g_nb_base_scale=None, y0=None,
                 t_eval=t_eval, rtol=1e-6, atol=1e-9, max_step=0.25):
    if y0 is None: y0 = [1.0, V0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    if g_nb_base_scale is None: g_nb_base_scale = 1.0
    return solve_ivp(
        neuroblast_model_no_feedback_satpow, t_span, y0, t_eval=t_eval,
        args=([g_NB_WT * g_nb_base_scale, g_GMC, k_Neuron, sym_frac,
               V_thresh_NB, V_thresh_GMC, k_max_NB, k_max_GMC, n,
               V_floor_NB, V_floor_GMC],),
        rtol=rtol, atol=atol, max_step=max_step, method="LSODA"
    )


def solve_model2(sym_frac, g_nb_base_scale=1.0, y0=None,
                 t_eval=t_eval, rtol=1e-6, atol=1e-9, max_step=0.25):
    if y0 is None: y0 = [1.0, V0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, V_thresh_base]
    return solve_ivp(
        neuroblast_model_SymDivPullsThresh_NoRecovery_NoNBself, t_span, y0, t_eval=t_eval,
        args=([g_NB_base_WT * g_nb_base_scale, g_GMC, k_Neuron, sym_frac,
               V_thresh_base, V_thresh_GMC, k_max_NB, k_max_GMC, n,
               V_floor_NB, V_floor_GMC, delta_thresh, V_thresh_min],),
        rtol=rtol, atol=atol, max_step=max_step, method="LSODA"
    )


def solve_model3(alpha_growth, sym_frac, g_nb_base_scale, y0=None,
                 t_eval=t_eval, rtol=1e-6, atol=1e-9, max_step=0.25):
    if y0 is None: y0 = [1.0, V0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    return solve_ivp(
        neuroblast_model_vol_scaled_growth_only, t_span, y0, t_eval=t_eval,
        args=([g_NB_WT * g_nb_base_scale, g_GMC, k_Neuron, sym_frac,
               V_thresh_NB, V_thresh_GMC, k_max_NB, k_max_GMC, n,
               V_floor_NB, V0, alpha_growth, V_floor_GMC],),
        rtol=rtol, atol=atol, max_step=max_step, method="LSODA"
    )


def solve_model4(alpha_growth, sym_frac, g_nb_base_scale=1.0, y0=None,
                 t_eval=t_eval, rtol=1e-6, atol=1e-9, max_step=0.25):
    # 9th state is S_sym (cumulative symmetric divisions)
    if y0 is None:
        y0 = [1.0, V0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    return solve_ivp(
        neuroblast_model_vol_scaled_growth_and_symdiv_thresh, t_span, y0, t_eval=t_eval,
        args=([g_NB_WT * g_nb_base_scale, g_GMC, k_Neuron, sym_frac,
               V_thresh_NB, V_thresh_GMC, k_max_NB, k_max_GMC, n,
               V_floor_NB, V0, alpha_growth, V_floor_GMC,
               delta_thresh, V_thresh_min],),
        rtol=rtol, atol=atol, max_step=max_step, method="LSODA"
    )


def model5_k_NB_max(p_div):
    """Max NB division rate for Model 5, scaled so a single NB divides at k_star (CCD = 1.5 h)."""
    return k_star * ((K_self ** p_div + 1.0) / (K_self ** p_div))


def solve_model5(p_div, sym_frac, y0=None,
                 t_eval=t_eval, rtol=1e-6, atol=1e-9, max_step=0.5):
    if y0 is None: y0 = [1.0, 0.0, 0.0, 0.0]
    k_NB_max = model5_k_NB_max(p_div)
    return solve_ivp(
        neuroblast_model_NB_self_repression, t_span, y0, t_eval=t_eval,
        args=([k_NB_max, 1.0/8.0, k_Neuron, K_self, p_div, sym_frac],),
        rtol=rtol, atol=atol, max_step=max_step, method="LSODA"
    )
