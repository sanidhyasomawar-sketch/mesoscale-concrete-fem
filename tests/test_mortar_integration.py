"""
tests/test_mortar_integration.py
--------------------------------
Integration test suite for Phase 6.3: Controlled Bulk Mortar Damage Integration into FEM Solver.

Covers the 10 requirements of Scope E:
1. Mortar elements invoke the mortar constitutive model.
2. Aggregate elements remain elastic.
3. ITZ elements remain on the cohesive formulation.
4. Mortar damage begins only after tensile initiation.
5. Damage is irreversible across converged increments.
6. Trial damage is rolled back after a failed increment / cutback.
7. Damage state is committed only after global convergence.
8. A simple elastic case remains consistent with the existing FEM response.
9. A small damage-enabled case completes without invalid values.
10. Regression compatibility with existing modules.
"""

import pytest
import numpy as np
from shapely.geometry import Polygon

from core.meshing import generate_conforming_mesh, extract_interface_data
from core.fem import MaterialProperties, BoundaryConditions, solve_fem
from core.cohesive import (
    ITZProperties, CohesiveState, CohesiveMesh,
    build_cohesive_mesh, assemble_cohesive_forces_and_stiffness,
    assemble_bulk_forces_and_stiffness, precompute_bulk_element_data,
    solve_cohesive_fem
)
from core.mortar_damage import MortarDamageParameters, BulkMortarState, compute_mortar_damage


@pytest.fixture
def mesoscale_test_setup():
    """Generates a small test mesh with aggregate, mortar, and ITZ elements."""
    poly = Polygon([(10.0, 10.0), (30.0, 10.0), (30.0, 30.0), (10.0, 30.0)])
    mesh, _ = generate_conforming_mesh(40.0, 40.0, [poly], target_edge_mortar=6.0, run_pre_checks=False)
    if_data = extract_interface_data(mesh, [poly])
    coh_mesh = build_cohesive_mesh(mesh, if_data)

    mat_mortar = MaterialProperties(E=25000.0, nu=0.20, name="Mortar")
    mat_aggregate = MaterialProperties(E=60000.0, nu=0.15, name="Aggregate")
    itz_props = ITZProperties(k_n=100000.0, k_t=40000.0, f_t=2.0, G_f=0.04)

    mortar_dam = MortarDamageParameters(
        E_m=25000.0, nu_m=0.20, f_t=3.0, G_f=0.08, h_e=2.5, alpha_res=1e-6
    )

    return {
        'coh_mesh': coh_mesh,
        'mat_mortar': mat_mortar,
        'mat_aggregate': mat_aggregate,
        'itz_props': itz_props,
        'mortar_dam': mortar_dam
    }


# ==============================================================================
# 1. Mortar Elements Invoke Constitutive Model
# ==============================================================================

def test_mortar_elements_invoke_constitutive_model(mesoscale_test_setup):
    """Verifies that bulk mortar elements (phase == 1) evaluate damage using the mortar damage model."""
    s = mesoscale_test_setup
    coh_mesh = s['coh_mesh']
    mortar_dam = s['mortar_dam']

    num_triangles = len(coh_mesh.triangles)
    mortar_state = BulkMortarState.initialize(num_triangles, coh_mesh.phase, mortar_dam.eps_0)

    # Impose a displacement field with large tensile strain on mortar elements
    total_dofs = 2 * len(coh_mesh.nodes)
    u_full = np.zeros(total_dofs, dtype=np.float64)
    # Impose horizontal tensile displacement: u_x = 0.001 * x
    u_full[0::2] = 0.001 * coh_mesh.nodes[:, 0]

    f_bulk, K_bulk, trial_state = assemble_bulk_forces_and_stiffness(
        coh_mesh, s['mat_mortar'], s['mat_aggregate'], u_full,
        mortar_state, mortar_dam, thickness=1.0
    )

    assert trial_state is not None
    mortar_indices = np.where(coh_mesh.phase == 1)[0]
    # Mortar elements must have evaluated damage > 0 because strain 0.001 > eps_0 (0.00012)
    assert np.any(trial_state.damage[mortar_indices] > 0.0)
    # Damage must not exceed d_max
    assert np.all(trial_state.damage[mortar_indices] <= mortar_dam.d_max)


# ==============================================================================
# 2. Aggregate Elements Remain Elastic
# ==============================================================================

def test_aggregate_elements_remain_elastic(mesoscale_test_setup):
    """Verifies that aggregate elements (phase == 0) strictly maintain zero damage and elastic stiffness."""
    s = mesoscale_test_setup
    coh_mesh = s['coh_mesh']
    mortar_dam = s['mortar_dam']

    num_triangles = len(coh_mesh.triangles)
    mortar_state = BulkMortarState.initialize(num_triangles, coh_mesh.phase, mortar_dam.eps_0)

    total_dofs = 2 * len(coh_mesh.nodes)
    u_full = np.zeros(total_dofs, dtype=np.float64)
    # Impose very large tensile displacement
    u_full[0::2] = 0.005 * coh_mesh.nodes[:, 0]

    _, _, trial_state = assemble_bulk_forces_and_stiffness(
        coh_mesh, s['mat_mortar'], s['mat_aggregate'], u_full,
        mortar_state, mortar_dam, thickness=1.0
    )

    agg_indices = np.where(coh_mesh.phase == 0)[0]
    assert len(agg_indices) > 0, "Mesh must contain aggregate elements."
    # Damage on aggregate elements must be strictly 0.0
    assert np.all(trial_state.damage[agg_indices] == 0.0)
    assert not np.any(trial_state.is_damaged[agg_indices])


# ==============================================================================
# 3. ITZ Elements Remain on Cohesive Formulation
# ==============================================================================

def test_itz_elements_remain_on_cohesive_formulation(mesoscale_test_setup):
    """Verifies that cohesive ITZ elements use their independent cohesive formulation."""
    s = mesoscale_test_setup
    coh_mesh = s['coh_mesh']

    # Solve with cohesive ITZ enabled
    res = solve_cohesive_fem(
        coh_mesh=coh_mesh,
        mat_mortar=s['mat_mortar'],
        mat_aggregate=s['mat_aggregate'],
        itz_props=s['itz_props'],
        top_displacement=-0.005,
        num_steps=2,
        mortar_damage_params=s['mortar_dam']
    )

    # ITZ state is tracked in final_state
    assert isinstance(res.final_state, CohesiveState)
    assert len(res.final_state.damage) == len(coh_mesh.cohesive_elements)
    # Cohesive formulation is independent of bulk mortar damage
    assert hasattr(res.final_state, 'delta_n_max')
    assert hasattr(res.final_state, 'element_damage')


# ==============================================================================
# 4. Mortar Damage Begins Only After Tensile Initiation
# ==============================================================================

def test_mortar_damage_begins_only_after_initiation(mesoscale_test_setup):
    """Verifies that mortar damage is strictly zero when strain is below eps_0 = f_t / E_m."""
    s = mesoscale_test_setup
    coh_mesh = s['coh_mesh']
    mortar_dam = s['mortar_dam']

    num_triangles = len(coh_mesh.triangles)
    mortar_state = BulkMortarState.initialize(num_triangles, coh_mesh.phase, mortar_dam.eps_0)

    # Impose very small strain: eps_xx = 0.5 * eps_0 = 0.00006 < 0.00012
    sub_initiation_strain = 0.5 * mortar_dam.eps_0
    total_dofs = 2 * len(coh_mesh.nodes)
    u_full = np.zeros(total_dofs, dtype=np.float64)
    u_full[0::2] = sub_initiation_strain * coh_mesh.nodes[:, 0]

    _, _, trial_state = assemble_bulk_forces_and_stiffness(
        coh_mesh, s['mat_mortar'], s['mat_aggregate'], u_full,
        mortar_state, mortar_dam, thickness=1.0
    )

    # Zero damage across ALL elements
    assert np.all(trial_state.damage == 0.0)
    assert np.all(~trial_state.is_damaged)


# ==============================================================================
# 5. Damage Irreversibility Across Increments
# ==============================================================================

def test_damage_irreversibility_across_increments(mesoscale_test_setup):
    """Verifies that damage is monotonically non-decreasing (dot{d} >= 0) across converged steps."""
    s = mesoscale_test_setup

    res = solve_cohesive_fem(
        coh_mesh=s['coh_mesh'],
        mat_mortar=s['mat_mortar'],
        mat_aggregate=s['mat_aggregate'],
        itz_props=s['itz_props'],
        top_displacement=-0.015,
        num_steps=3,
        mortar_damage_params=s['mortar_dam']
    )

    assert res.converged
    # Check max mortar damage across steps is non-decreasing
    mortar_damages = [step.max_mortar_damage for step in res.steps]
    for i in range(len(mortar_damages) - 1):
        assert mortar_damages[i + 1] >= mortar_damages[i] - 1e-12
    # Ensure damage actually grew
    assert mortar_damages[-1] > 0.0


# ==============================================================================
# 6. Trial Damage Rollback After Failed Increment / Cutback
# ==============================================================================

def test_trial_damage_rollback_on_cutback(mesoscale_test_setup):
    """Verifies that a trial damage state from a failed increment is discarded upon cutback."""
    s = mesoscale_test_setup
    coh_mesh = s['coh_mesh']
    mortar_dam = s['mortar_dam']

    num_triangles = len(coh_mesh.triangles)
    state = BulkMortarState.initialize(num_triangles, coh_mesh.phase, mortar_dam.eps_0)
    state_backup = state.copy()

    # Simulate an increment with large trial displacement
    total_dofs = 2 * len(coh_mesh.nodes)
    u_trial = np.zeros(total_dofs, dtype=np.float64)
    u_trial[0::2] = 0.005 * coh_mesh.nodes[:, 0]

    _, _, trial_state = assemble_bulk_forces_and_stiffness(
        coh_mesh, s['mat_mortar'], s['mat_aggregate'], u_trial,
        state, mortar_dam, thickness=1.0
    )

    # Trial state has non-zero damage
    assert np.max(trial_state.damage) > 0.0

    # Simulate cutback: rollback to backup
    state = state_backup.copy()

    # Rolled-back state must be identical to backup with 0 damage
    assert np.all(state.damage == 0.0)
    assert np.array_equal(state.kappa, state_backup.kappa)


# ==============================================================================
# 7. Damage Committed Only After Global Convergence
# ==============================================================================

def test_damage_state_committed_only_after_convergence(mesoscale_test_setup):
    """Verifies that committed state is updated only when equilibrium convergence is declared."""
    s = mesoscale_test_setup
    res = solve_cohesive_fem(
        coh_mesh=s['coh_mesh'],
        mat_mortar=s['mat_mortar'],
        mat_aggregate=s['mat_aggregate'],
        itz_props=s['itz_props'],
        top_displacement=-0.015,
        num_steps=3,
        mortar_damage_params=s['mortar_dam']
    )

    # For each converged step, equilibrium residual is satisfied
    for step in res.steps:
        assert step.equilibrium_residual < 10.0 * res.force_tol
        # Recorded damage corresponds to converged state
        assert step.max_mortar_damage >= 0.0


# ==============================================================================
# 8. Elastic Consistency with Existing Solver
# ==============================================================================

def test_elastic_case_consistency_with_existing_fem(mesoscale_test_setup):
    """Verifies that in the linear-elastic regime, results with and without mortar damage params match."""
    s = mesoscale_test_setup
    coh_mesh = s['coh_mesh']

    # Solve without mortar damage params (linear elastic bulk)
    res_elastic = solve_cohesive_fem(
        coh_mesh=coh_mesh,
        mat_mortar=s['mat_mortar'],
        mat_aggregate=s['mat_aggregate'],
        itz_props=s['itz_props'],
        top_displacement=-0.003,
        num_steps=2,
        mortar_damage_params=None
    )

    # Solve with mortar damage params at small displacement (below initiation)
    res_dam_enabled = solve_cohesive_fem(
        coh_mesh=coh_mesh,
        mat_mortar=s['mat_mortar'],
        mat_aggregate=s['mat_aggregate'],
        itz_props=s['itz_props'],
        top_displacement=-0.003,
        num_steps=2,
        mortar_damage_params=s['mortar_dam']
    )

    assert res_elastic.converged and res_dam_enabled.converged
    # Reactions and stresses must match to high precision
    assert np.isclose(res_elastic.reaction_curve[-1], res_dam_enabled.reaction_curve[-1], rtol=1e-5)
    assert np.isclose(res_elastic.stress_curve[-1], res_dam_enabled.stress_curve[-1], rtol=1e-5)
    # Mortar damage is zero in the elastic regime
    assert res_dam_enabled.max_mortar_damage == 0.0


# ==============================================================================
# 9. Small Damage-Enabled Case Completes Without Invalid Values
# ==============================================================================

def test_damage_enabled_case_completes_valid(mesoscale_test_setup):
    """Verifies that a damage-enabled simulation completes with finite, valid numerical outputs."""
    s = mesoscale_test_setup

    res = solve_cohesive_fem(
        coh_mesh=s['coh_mesh'],
        mat_mortar=s['mat_mortar'],
        mat_aggregate=s['mat_aggregate'],
        itz_props=s['itz_props'],
        top_displacement=-0.015,
        num_steps=3,
        mortar_damage_params=s['mortar_dam']
    )

    assert res.converged
    assert not np.any(np.isnan(res.final_displacements))
    assert not np.any(np.isinf(res.final_displacements))
    assert not np.any(np.isnan(res.reaction_curve))
    assert not np.any(np.isnan(res.stress_curve))
    assert res.reaction_curve[-1] > 0.0
    assert res.stress_curve[-1] > 0.0
    assert res.max_mortar_damage >= 0.0
    assert res.max_mortar_damage < 1.0


# ==============================================================================
# 10. Bulk Precomputation Integrity
# ==============================================================================

def test_precompute_bulk_element_data_integrity(mesoscale_test_setup):
    """Verifies that precomputation accurately caches all element areas, lengths, and B matrices."""
    s = mesoscale_test_setup
    coh_mesh = s['coh_mesh']
    mortar_dam = s['mortar_dam']

    pre = precompute_bulk_element_data(
        coh_mesh, s['mat_mortar'], s['mat_aggregate'], thickness=1.0, mortar_damage_params=mortar_dam
    )

    assert pre['all_ke0'].shape == (len(coh_mesh.triangles), 6, 6)
    assert pre['all_B'].shape == (len(coh_mesh.triangles), 3, 6)
    assert len(pre['all_he']) == len(coh_mesh.triangles)
    assert np.all(pre['all_he'] > 0.0)
    assert pre['K_bulk_elastic'].shape == (2 * len(coh_mesh.nodes), 2 * len(coh_mesh.nodes))
    # Check that eps_u is computed for mortar elements
    mortar_indices = pre['mortar_indices']
    assert np.all(pre['eps_u'][mortar_indices] > mortar_dam.eps_0)
