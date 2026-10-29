"""
tests/test_mortar_damage.py
---------------------------
Unit and verification test suite for bulk mortar isotropic tensile damage constitutive model.

Tested Behaviors:
1. Elastic response and tensile initiation strain.
2. Softening stress curve and damage evolution.
3. Irreversibility and state management (commit, rollback, unloading/reloading).
4. Damage-cap activation strain kappa_cap < eps_u and residual stiffness.
5. Exact ideal fracture energy integration (G_f / h_e).
6. Stabilized softening energy discrepancy measurement (< 0.02%).
7. Residual energy integration over specified finite intervals.
8. Kinematic strain decomposition (total, elastic, crack-band, crack opening).
9. Element characteristic length calculation (CST approximation).
10. Equivalent tensile strain (Rankine 1D and 2D plane stress).
11. Parameter validation and invalid input rejection.
"""

import pytest
import math
import numpy as np

from core.mortar_damage import (
    MortarDamageParameters,
    compute_element_characteristic_length,
    compute_equivalent_strain,
    compute_mortar_damage,
    evaluate_analytical_piecewise_stress,
    evaluate_1d_secant_stress,
    decompose_strain,
    plane_stress_elastic_matrix,
    compute_secant_stiffness_matrix,
    MortarElementState,
    compute_ideal_softening_energy,
    integrate_softening_energy,
    integrate_post_ultimate_residual_energy
)


@pytest.fixture
def baseline_mortar():
    """Standard benchmark parameters for mortar."""
    return MortarDamageParameters(
        E_m=25000.0,   # MPa
        nu_m=0.20,
        f_t=3.0,       # MPa
        G_f=0.080,     # N/mm
        h_e=2.50,      # mm
        alpha_res=1e-6
    )


# ==============================================================================
# 1. Parameter Properties & Validation
# ==============================================================================

def test_mortar_parameters_properties(baseline_mortar):
    p = baseline_mortar
    assert math.isclose(p.eps_0, 3.0 / 25000.0, rel_tol=1e-12)
    assert math.isclose(p.w_c, 2.0 * 0.080 / 3.0, rel_tol=1e-12)
    assert math.isclose(p.eps_cr_u, (2.0 * 0.080 / 3.0) / 2.50, rel_tol=1e-12)
    assert math.isclose(p.eps_u, p.eps_0 + p.eps_cr_u, rel_tol=1e-12)
    assert math.isclose(p.d_max, 1.0 - 1e-6, rel_tol=1e-12)
    assert math.isclose(p.E_res, 1e-6 * 25000.0, rel_tol=1e-12)
    assert math.isclose(p.g_f_ideal, 0.080 / 2.50, rel_tol=1e-12)


@pytest.mark.parametrize("bad_kwargs", [
    {"E_m": -25000.0},
    {"E_m": 0.0},
    {"nu_m": -0.1},
    {"nu_m": 0.5},
    {"nu_m": 0.6},
    {"f_t": -3.0},
    {"f_t": 0.0},
    {"G_f": -0.08},
    {"G_f": 0.0},
    {"h_e": -2.5},
    {"h_e": 0.0},
    {"alpha_res": 0.0},
    {"alpha_res": 1.0},
    {"alpha_res": -1e-6},
])
def test_mortar_parameters_invalid_inputs(bad_kwargs):
    valid_args = {
        "E_m": 25000.0,
        "nu_m": 0.20,
        "f_t": 3.0,
        "G_f": 0.080,
        "h_e": 2.50,
        "alpha_res": 1e-6
    }
    valid_args.update(bad_kwargs)
    with pytest.raises(ValueError):
        MortarDamageParameters(**valid_args)


# ==============================================================================
# 2. Geometric Characteristic Length
# ==============================================================================

def test_characteristic_length_cst():
    area = 3.125  # mm^2
    expected_he = math.sqrt(2.0 * 3.125)  # sqrt(6.25) = 2.5 mm
    assert math.isclose(compute_element_characteristic_length(area), expected_he, rel_tol=1e-12)


def test_characteristic_length_invalid_area():
    with pytest.raises(ValueError):
        compute_element_characteristic_length(0.0)
    with pytest.raises(ValueError):
        compute_element_characteristic_length(-2.5)


# ==============================================================================
# 3. Equivalent Tensile Strain (1D and 2D Plane Stress)
# ==============================================================================

def test_equivalent_strain_1d():
    assert compute_equivalent_strain(0.001) == 0.001
    assert compute_equivalent_strain(0.0) == 0.0
    assert compute_equivalent_strain(-0.002) == 0.0  # pure compression has zero equivalent tensile strain


def test_equivalent_strain_2d_plane_stress():
    # Pure uniaxial tension along x
    eps_x = np.array([0.0005, -0.0001, 0.0])
    assert math.isclose(compute_equivalent_strain(eps_x), 0.0005, rel_tol=1e-12)

    # Pure biaxial compression
    eps_comp = np.array([-0.001, -0.002, 0.0])
    assert compute_equivalent_strain(eps_comp) == 0.0

    # Pure shear
    eps_shear = np.array([0.0, 0.0, 0.0004])
    # Principal strains: center = 0, radius = 0.5 * 0.0004 = 0.0002
    assert math.isclose(compute_equivalent_strain(eps_shear), 0.0002, rel_tol=1e-12)

    with pytest.raises(ValueError):
        compute_equivalent_strain(np.array([1.0, 2.0]))  # invalid shape


# ==============================================================================
# 4. Elastic Response & Tensile Initiation
# ==============================================================================

def test_elastic_response_before_initiation(baseline_mortar):
    p = baseline_mortar
    strains = [0.0, 0.25 * p.eps_0, 0.5 * p.eps_0, 0.75 * p.eps_0, p.eps_0]

    for eps in strains:
        d = compute_mortar_damage(eps, p)
        assert d == 0.0, f"Expected zero damage at eps={eps}"
        sigma = evaluate_1d_secant_stress(eps, eps, p)
        assert math.isclose(sigma, p.E_m * eps, rel_tol=1e-12)

    # Exactly at initiation: stress equals tensile strength f_t
    sigma_0 = evaluate_1d_secant_stress(p.eps_0, p.eps_0, p)
    assert math.isclose(sigma_0, p.f_t, rel_tol=1e-12)


# ==============================================================================
# 5. Softening Stress & Damage Evolution
# ==============================================================================

def test_damage_evolution_monotonic(baseline_mortar):
    p = baseline_mortar
    strains = np.linspace(p.eps_0, p.kappa_cap, 50)
    damages = [compute_mortar_damage(k, p) for k in strains]

    # Monotonically increasing damage
    for i in range(len(damages) - 1):
        assert damages[i + 1] >= damages[i]

    # At initiation, damage is 0
    assert damages[0] == 0.0

    # At kappa_cap, damage reaches d_max exactly
    d_cap = compute_mortar_damage(p.kappa_cap, p)
    assert math.isclose(d_cap, p.d_max, rel_tol=1e-12)


def test_softening_stress_matches_linear_ideal(baseline_mortar):
    p = baseline_mortar
    # Between eps_0 and kappa_cap, stress should follow linear softening f_t * (eps_u - k) / (eps_u - eps_0)
    strains = np.linspace(p.eps_0 + 1e-6, p.kappa_cap - 1e-6, 30)
    for k in strains:
        d = compute_mortar_damage(k, p)
        sigma_secant = (1.0 - d) * p.E_m * k
        sigma_linear = p.f_t * (p.eps_u - k) / (p.eps_u - p.eps_0)
        assert math.isclose(sigma_secant, sigma_linear, rel_tol=1e-10)


# ==============================================================================
# 6. Damage-Cap Activation & Piecewise Match
# ==============================================================================

def test_damage_cap_activation(baseline_mortar):
    p = baseline_mortar

    # 1. kappa_cap is strictly less than eps_u
    assert p.kappa_cap < p.eps_u
    diff_strain = p.eps_u - p.kappa_cap
    assert diff_strain > 0.0

    # 2. Damage cap threshold
    assert math.isclose(compute_mortar_damage(p.kappa_cap, p), 1.0 - p.alpha_res, rel_tol=1e-12)
    assert math.isclose(compute_mortar_damage(p.eps_u, p), 1.0 - p.alpha_res, rel_tol=1e-12)
    assert math.isclose(compute_mortar_damage(2.0 * p.eps_u, p), 1.0 - p.alpha_res, rel_tol=1e-12)

    # 3. Secant stress matches analytical 4-branch law across full domain
    eval_strains = np.linspace(0.0, 2.5 * p.eps_u, 200)
    for k in eval_strains:
        s_sec = evaluate_1d_secant_stress(k, k, p)
        s_ana = evaluate_analytical_piecewise_stress(k, p)
        assert math.isclose(s_sec, s_ana, abs_tol=1e-11)


# ==============================================================================
# 7. Kinematic Strain Decomposition
# ==============================================================================

def test_strain_decomposition(baseline_mortar):
    p = baseline_mortar

    # A. At initiation
    decomp_0 = decompose_strain(p.eps_0, p.eps_0, p)
    assert math.isclose(decomp_0['elastic_strain'], p.eps_0, rel_tol=1e-12)
    assert math.isclose(decomp_0['crack_band_strain'], 0.0, abs_tol=1e-15)
    assert math.isclose(decomp_0['crack_opening'], 0.0, abs_tol=1e-15)
    assert math.isclose(decomp_0['stress'], p.f_t, rel_tol=1e-12)

    # B. Mid-softening
    eps_mid = 0.5 * (p.eps_0 + p.eps_u)
    decomp_mid = decompose_strain(eps_mid, eps_mid, p)
    assert math.isclose(
        decomp_mid['total_strain'],
        decomp_mid['elastic_strain'] + decomp_mid['crack_band_strain'],
        rel_tol=1e-12
    )
    assert math.isclose(
        decomp_mid['crack_opening'],
        decomp_mid['crack_band_strain'] * p.h_e,
        rel_tol=1e-12
    )
    assert math.isclose(
        decomp_mid['elastic_strain'],
        decomp_mid['stress'] / p.E_m,
        rel_tol=1e-12
    )

    # C. At ultimate strain
    decomp_u = decompose_strain(p.eps_u, p.eps_u, p)
    # Total strain is split almost entirely into crack-band strain
    assert math.isclose(
        decomp_u['total_strain'],
        decomp_u['elastic_strain'] + decomp_u['crack_band_strain'],
        rel_tol=1e-12
    )
    # Total crack opening at eps_u includes the released elastic deformation (eps_0 * h_e)
    # matching eps_u * h_e = w_c + eps_0 * h_e
    expected_opening = p.w_c + p.eps_0 * p.h_e
    assert math.isclose(decomp_u['crack_opening'], expected_opening, rel_tol=1e-3)
    assert math.isclose(decomp_u['crack_opening'], p.w_c, rel_tol=0.01)  # within 0.6% of w_c
    assert math.isclose(decomp_u['crack_band_strain'], p.eps_u, rel_tol=1e-3)


# ==============================================================================
# 8. Fracture Energy Integration & Discrepancy
# ==============================================================================

def test_ideal_fracture_energy_integration(baseline_mortar):
    p = baseline_mortar
    res = integrate_softening_energy(p)

    # Ideal analytical softening energy density equals G_f / h_e
    expected_gf_density = p.G_f / p.h_e  # 0.032 MPa
    assert math.isclose(res['g_f_ideal_analytical'], expected_gf_density, rel_tol=1e-12)
    assert math.isclose(res['g_f_ideal_numerical'], expected_gf_density, rel_tol=1e-10)


def test_stabilized_softening_energy_discrepancy(baseline_mortar):
    p = baseline_mortar
    res = integrate_softening_energy(p)

    # Absolute discrepancy is on the order of 1e-9 MPa
    assert res['absolute_difference'] < 1e-8

    # Relative discrepancy must be well below 0.02% (2e-4)
    # Actual discrepancy is ~3.20e-8 (0.0000032%)
    assert res['relative_difference'] < 2e-4
    assert res['relative_difference'] < 1e-6


def test_residual_energy_over_specified_interval(baseline_mortar):
    p = baseline_mortar
    interval_end = 2.0 * p.eps_u
    res = integrate_post_ultimate_residual_energy(p, interval_end)

    assert res['interval_start'] == p.eps_u
    assert res['interval_end'] == interval_end
    # Analytical and numerical post-ultimate energies match
    assert math.isclose(res['g_post_analytical'], res['g_post_numerical'], rel_tol=1e-10)
    # Ratio to G_f is finite and small (~0.054%)
    assert 0.0005 < res['ratio_to_ideal_gf'] < 0.0006

    with pytest.raises(ValueError):
        integrate_post_ultimate_residual_energy(p, p.eps_u - 1e-4)


# ==============================================================================
# 9. Irreversibility, Unloading/Reloading & State Management
# ==============================================================================

def test_mortar_element_state_irreversibility(baseline_mortar):
    p = baseline_mortar
    elem = MortarElementState(element_id=1, params=p)

    # Initial state is uncracked
    assert elem.damage == 0.0
    assert elem.kappa == p.eps_0
    assert not elem.is_damaged

    # Trial loading past eps_0
    target_strain = 0.005
    d_trial, stress_trial, C_sec_trial = elem.evaluate_trial(target_strain)
    assert d_trial > 0.0
    assert elem.trial_is_damaged
    # Committed state has NOT changed yet
    assert elem.damage == 0.0
    assert elem.kappa == p.eps_0

    # Commit trial state
    elem.commit()
    assert elem.damage == d_trial
    assert elem.kappa == target_strain
    assert elem.is_damaged

    # Elastic unloading to lower strain (0.001 < target_strain)
    d_unload, stress_unload, _ = elem.evaluate_trial(0.001)
    # Damage cannot decrease! (dot{d} >= 0)
    assert math.isclose(d_unload, d_trial, rel_tol=1e-12)
    # Stress unloads along secant slope (1 - d) * E_m * eps
    expected_stress = (1.0 - d_trial) * p.E_m * 0.001
    assert math.isclose(stress_unload[0], expected_stress, rel_tol=1e-12)

    # Rollback of trial state
    elem.rollback()
    assert elem.trial_damage == elem.damage
    assert elem.trial_kappa == elem.kappa


# ==============================================================================
# 10. 2D Plane-Stress Elastic & Secant Matrices
# ==============================================================================

def test_plane_stress_matrices(baseline_mortar):
    p = baseline_mortar
    C_0 = plane_stress_elastic_matrix(p.E_m, p.nu_m)

    assert C_0.shape == (3, 3)
    # Symmetric
    assert np.allclose(C_0, C_0.T)
    # Positive definite
    eigenvals = np.linalg.eigvalsh(C_0)
    assert np.all(eigenvals > 0.0)

    # Secant matrix at d = 0.5
    C_sec = compute_secant_stiffness_matrix(0.5, C_0)
    assert np.allclose(C_sec, 0.5 * C_0)
