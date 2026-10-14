"""Sanity checks on model behavior: WT stability, calibration, positivity and division bookkeeping."""
import numpy as np
import pytest

from neuroblast_ode.ccd import ccd_M1, ccd_M2, ccd_M3, ccd_M4, ccd_M5
from neuroblast_ode.models import extract_counts_from_solution
from neuroblast_ode.params import GENOS, alpha_value, beta_value
from neuroblast_ode.solvers import (
    solve_base_model, solve_model1, solve_model2, solve_model3, solve_model4, solve_model5,
)

WT_CCD = 1.5  # calibrated WT NB cell-cycle duration (hours)

# Model name -> function that simulates that model for one GENOS entry (label, sym_frac, g_nb_base_scale)
RUNNERS = {
    "base":    lambda g: solve_base_model(g[1]),
    "model 1": lambda g: solve_model1(sym_frac=g[1], g_nb_base_scale=g[2]),
    "model 2": lambda g: solve_model2(sym_frac=g[1], g_nb_base_scale=g[2]),
    "model 3": lambda g: solve_model3(alpha_growth=alpha_value, sym_frac=g[1], g_nb_base_scale=g[2]),
    "model 4": lambda g: solve_model4(alpha_growth=alpha_value, sym_frac=g[1], g_nb_base_scale=g[2]),
    "model 5": lambda g: solve_model5(p_div=beta_value, sym_frac=g[1]),
}

# Model name -> function that computes NB cell-cycle duration from that model's solution
# (the base model is omitted: its CCD is a constant, not computed from the solution)
CCD_FUNCTIONS = {
    "model 1": ccd_M1,
    "model 2": ccd_M2,
    "model 3": ccd_M3,
    "model 4": ccd_M4,
    "model 5": lambda sol: ccd_M5(sol, beta_value),
}


@pytest.fixture(scope="module")
def solutions():
    """Solve every model for every genotype once and share the results across tests.

    Returns a dict mapping (model, genotype label) to (t, NB, GMC, Im, Mat, sym_frac, sol).
    """
    out = {}
    for model, run in RUNNERS.items():
        for geno in GENOS:
            sol = run(geno)
            assert sol.success, f"{model} / {geno[0]} failed to integrate: {sol.message}"
            out[(model, geno[0])] = (*extract_counts_from_solution(sol), geno[1], sol)
    return out


@pytest.mark.parametrize("model", RUNNERS)
def test_wt_nb_count_constant(solutions, model):
    """WT NB count stays at exactly 1 for the whole simulation.

    With no symmetric divisions, the single neuroblast can only self-renew.
    """
    _, NB, *_ = solutions[(model, "WT")]
    np.testing.assert_allclose(NB, 1.0, rtol=0, atol=1e-12)


@pytest.mark.parametrize("model", RUNNERS)
def test_counts_non_negative(solutions, model):
    """No cell population goes negative, in any genotype."""
    for geno in GENOS:
        _, NB, GMC, Im, Mat, _, _ = solutions[(model, geno[0])]
        for name, arr in (("NB", NB), ("GMC", GMC), ("Im", Im), ("Mat", Mat)):
            assert arr.min() >= -1e-9, f"{model} / {geno[0]}: {name} goes negative ({arr.min():.3g})"


@pytest.mark.parametrize("model", RUNNERS)
def test_division_bookkeeping(solutions, model):
    """GMC + (Im + Mat)/2 equals the cumulative number of asymmetric NB divisions.

    Each asymmetric NB division makes one GMC, and each GMC division makes two
    neurons. In mutants, symmetric divisions have added NB - 1 neuroblasts, and
    asymmetric divisions are (1 - s)/s times as frequent. In WT, the single NB
    divides once every 1.5 h.
    """
    for geno in GENOS:
        t, NB, GMC, Im, Mat, s, _ = solutions[(model, geno[0])]
        made_by_asym = GMC + (Im + Mat) / 2
        expected = t / WT_CCD if s == 0 else (1 - s) / s * (NB - 1)
        np.testing.assert_allclose(made_by_asym, expected, rtol=1e-9, atol=1e-9,
                                   err_msg=f"{model} / {geno[0]}")


@pytest.mark.parametrize("model", CCD_FUNCTIONS)
def test_wt_ccd_matches_calibration(solutions, model):
    """WT NB cell-cycle duration equals the calibrated 1.5 h for the whole simulation."""
    sol = solutions[(model, "WT")][-1]
    np.testing.assert_allclose(CCD_FUNCTIONS[model](sol), WT_CCD, rtol=1e-6)


@pytest.mark.parametrize("model", RUNNERS)
def test_mud_mutant_expands_nb_pool(solutions, model):
    """The mud mutant ends with more NBs than WT."""
    NB_wt = solutions[(model, "WT")][1]
    NB_mud = solutions[(model, "mud mutant")][1]
    assert NB_mud[-1] > NB_wt[-1]
