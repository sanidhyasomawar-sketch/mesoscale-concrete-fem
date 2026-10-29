"""
tests/test_mortar_ui.py
-----------------------
Unit and integration test suite for Phase 6.4: Bulk Mortar Damage UI & Visualization.

Covers the 6 requirements of Scope E:
1. Mortar damage result data is passed correctly to the UI structures.
2. Aggregate elements remain visually separate from mortar damage.
3. ITZ damage remains separate from mortar damage in visualization and metrics.
4. Damage visualization verifies simulation mesh fingerprint.
5. Missing damage data produces a clear fallback rather than an exception.
6. Existing simulation result summaries and metrics remain intact.
"""

import pytest
import numpy as np
from shapely.geometry import Polygon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from core.meshing import generate_conforming_mesh, extract_interface_data
from core.fem import MaterialProperties
from core.cohesive import (
    ITZProperties, CohesiveState, CohesiveMesh,
    build_cohesive_mesh, solve_cohesive_fem, NonlinearFEMResult
)
from core.mortar_damage import (
    MortarDamageParameters, BulkMortarState, plot_mortar_damage_distribution
)


@pytest.fixture
def ui_test_setup():
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
# 1. Mortar Damage Result Data Passed Correctly to UI
# ==============================================================================

def test_mortar_damage_result_data_passed_to_ui(ui_test_setup):
    """Verifies that solve_cohesive_fem returns mortar metrics and state for UI consumption."""
    s = ui_test_setup

    res = solve_cohesive_fem(
        coh_mesh=s['coh_mesh'],
        mat_mortar=s['mat_mortar'],
        mat_aggregate=s['mat_aggregate'],
        itz_props=s['itz_props'],
        top_displacement=-0.015,
        num_steps=3,
        mortar_damage_params=s['mortar_dam']
    )

    # UI fields must be present and correctly populated
    assert hasattr(res, 'max_mortar_damage')
    assert hasattr(res, 'damaged_mortar_count')
    assert hasattr(res, 'final_mortar_state')
    assert hasattr(res, 'mortar_damage_curve')

    assert isinstance(res.final_mortar_state, BulkMortarState)
    assert res.max_mortar_damage >= 0.0
    assert res.max_mortar_damage < 1.0
    assert res.damaged_mortar_count >= 0

    # Step-level metrics for convergence history table
    for step in res.steps:
        assert hasattr(step, 'max_mortar_damage')
        assert hasattr(step, 'mean_mortar_damage')
        assert hasattr(step, 'damaged_mortar_count')


# ==============================================================================
# 2. Aggregate Elements Remain Visually Separate
# ==============================================================================

def test_aggregate_elements_remain_visually_separate(ui_test_setup):
    """Verifies that plot_mortar_damage_distribution renders aggregate elements in distinct elastic color."""
    s = ui_test_setup
    coh_mesh = s['coh_mesh']

    res = solve_cohesive_fem(
        coh_mesh=coh_mesh,
        mat_mortar=s['mat_mortar'],
        mat_aggregate=s['mat_aggregate'],
        itz_props=s['itz_props'],
        top_displacement=-0.015,
        num_steps=3,
        mortar_damage_params=s['mortar_dam']
    )

    fig, ax, has_data = plot_mortar_damage_distribution(coh_mesh, res)
    assert has_data
    assert fig is not None and ax is not None

    # Verify that collections were created: one for aggregates, one for mortar
    collections = ax.collections
    assert len(collections) >= 2, "Expected at least aggregate PatchCollection and mortar PatchCollection."

    # Verify legend contains label for elastic aggregate
    legend = ax.get_legend()
    assert legend is not None
    legend_texts = [t.get_text() for t in legend.get_texts()]
    assert any("Elastic Aggregate" in text for text in legend_texts)

    plt.close(fig)


# ==============================================================================
# 3. ITZ Damage Remains Separate from Mortar Damage
# ==============================================================================

def test_itz_damage_remains_separate_from_mortar_damage(ui_test_setup):
    """Verifies that ITZ damage and mortar damage are tracked and displayed independently."""
    s = ui_test_setup
    coh_mesh = s['coh_mesh']

    res = solve_cohesive_fem(
        coh_mesh=coh_mesh,
        mat_mortar=s['mat_mortar'],
        mat_aggregate=s['mat_aggregate'],
        itz_props=s['itz_props'],
        top_displacement=-0.015,
        num_steps=3,
        mortar_damage_params=s['mortar_dam']
    )

    # ITZ damage is in final_state.damage (length = num_itz_elements)
    itz_damages = res.final_state.damage
    assert len(itz_damages) == len(coh_mesh.cohesive_elements)

    # Mortar damage is in final_mortar_state.damage (length = num_bulk_elements)
    mortar_damages = res.final_mortar_state.damage
    assert len(mortar_damages) == len(coh_mesh.triangles)

    # They are distinct arrays of different lengths and semantics
    assert itz_damages.shape != mortar_damages.shape

    # On cohesive interface elements, mortar damage is not applied
    agg_indices = np.where(coh_mesh.phase == 0)[0]
    assert np.all(mortar_damages[agg_indices] == 0.0)


# ==============================================================================
# 4. Damage Visualization Uses Simulation Mesh Fingerprint
# ==============================================================================

def test_visualization_verifies_mesh_fingerprint(ui_test_setup):
    """Verifies that plot_mortar_damage_distribution rejects mismatched mesh fingerprints."""
    s = ui_test_setup
    coh_mesh = s['coh_mesh']

    res = solve_cohesive_fem(
        coh_mesh=coh_mesh,
        mat_mortar=s['mat_mortar'],
        mat_aggregate=s['mat_aggregate'],
        itz_props=s['itz_props'],
        top_displacement=-0.005,
        num_steps=2,
        mortar_damage_params=s['mortar_dam']
    )

    # Matching fingerprints succeeds
    fig, ax, has_data = plot_mortar_damage_distribution(coh_mesh, res)
    assert has_data
    plt.close(fig)

    # Create a dummy result with mismatched fingerprint
    res.geometry_fingerprint = "mismatched_fingerprint_123"
    with pytest.raises(ValueError, match="Mesh fingerprint mismatch"):
        plot_mortar_damage_distribution(coh_mesh, res)


# ==============================================================================
# 5. Missing Damage Data Produces Safe Fallback
# ==============================================================================

def test_missing_mortar_damage_data_produces_safe_fallback(ui_test_setup):
    """Verifies that missing mortar damage data returns (None, None, False) instead of raising an exception."""
    s = ui_test_setup
    coh_mesh = s['coh_mesh']

    # Solve without mortar damage params (linear elastic bulk)
    res_elastic = solve_cohesive_fem(
        coh_mesh=coh_mesh,
        mat_mortar=s['mat_mortar'],
        mat_aggregate=s['mat_aggregate'],
        itz_props=s['itz_props'],
        top_displacement=-0.005,
        num_steps=2,
        mortar_damage_params=None
    )

    assert res_elastic.final_mortar_state is None
    fig, ax, has_data = plot_mortar_damage_distribution(coh_mesh, res_elastic)
    # Must produce a clean fallback
    assert not has_data
    assert fig is None
    assert ax is None


# ==============================================================================
# 6. Existing Simulation Result Summaries Intact
# ==============================================================================

def test_existing_result_summaries_intact(ui_test_setup):
    """Verifies that all Phase 5 nonlinear metrics, curves, and convergence data are preserved."""
    s = ui_test_setup

    res = solve_cohesive_fem(
        coh_mesh=s['coh_mesh'],
        mat_mortar=s['mat_mortar'],
        mat_aggregate=s['mat_aggregate'],
        itz_props=s['itz_props'],
        top_displacement=-0.015,
        num_steps=3,
        mortar_damage_params=s['mortar_dam']
    )

    # Phase 5 result fields preserved
    assert hasattr(res, 'converged')
    assert hasattr(res, 'status_message')
    assert hasattr(res, 'steps')
    assert hasattr(res, 'displacement_curve')
    assert hasattr(res, 'reaction_curve')
    assert hasattr(res, 'stress_curve')
    assert hasattr(res, 'strain_curve')
    assert hasattr(res, 'force_tol')
    assert hasattr(res, 'disp_tol')
    assert hasattr(res, 'relaxed_force_tol')
    assert hasattr(res, 'last_converged_load_factor')
    assert hasattr(res, 'actual_displacement_reached')

    # Step fields preserved
    step0 = res.steps[0]
    assert hasattr(step0, 'step_index')
    assert hasattr(step0, 'load_factor')
    assert hasattr(step0, 'displacement_top')
    assert hasattr(step0, 'reaction_force_top')
    assert hasattr(step0, 'reaction_force_bottom')
    assert hasattr(step0, 'nominal_compressive_stress')
    assert hasattr(step0, 'nominal_vertical_strain')
    assert hasattr(step0, 'num_iterations')
    assert hasattr(step0, 'equilibrium_residual')
    assert hasattr(step0, 'max_damage')
    assert hasattr(step0, 'damaged_element_count')
    assert hasattr(step0, 'convergence_method')
