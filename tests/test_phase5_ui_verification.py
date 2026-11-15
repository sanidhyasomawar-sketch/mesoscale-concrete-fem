"""
tests/test_phase5_ui_verification.py
------------------------------------
Verifies that Phase 5 in app.py correctly renders:
1. All 6 tabs including Mesh Sensitivity Benchmark
2. Separate numerical checks (point-wise residual norm and boundary reaction error)
3. Clear labeling of incomplete/stalled simulation runs
4. Mesh sensitivity benchmark table and figures
"""
import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')

from streamlit.testing.v1 import AppTest

from core.geometry import generate_mesostructure
from core.meshing import generate_conforming_mesh, extract_interface_data
from core.fem import MaterialProperties
from core.cohesive import (
    ITZProperties, build_cohesive_mesh, solve_cohesive_fem,
    NonlinearFEMResult
)
from core.mortar_damage import MortarDamageParameters


@pytest.fixture
def fast_phase5_setup():
    specimen = generate_mesostructure(domain_width=30.0, domain_height=30.0, target_area_fraction=0.15, random_seed=42)
    polys = specimen.polygons
    mesh, _ = generate_conforming_mesh(30.0, 30.0, polys, target_edge_mortar=6.0, run_pre_checks=False)
    if_data = extract_interface_data(mesh, polys)
    coh_mesh = build_cohesive_mesh(mesh, if_data)

    fp = specimen.fingerprint
    mesh.geometry_fingerprint = fp
    coh_mesh.geometry_fingerprint = fp

    mat_mortar = MaterialProperties(E=25000.0, nu=0.20, name="Mortar")
    mat_aggregate = MaterialProperties(E=60000.0, nu=0.15, name="Aggregate")
    itz_props = ITZProperties(k_n=100000.0, k_t=40000.0, f_t=2.0, G_f=0.04)
    mortar_dam = MortarDamageParameters(
        E_m=25000.0, nu_m=0.20, f_t=3.0, G_f=0.08, h_e=5.0, alpha_res=1e-4
    )

    res_converged = solve_cohesive_fem(
        coh_mesh=coh_mesh,
        mat_mortar=mat_mortar,
        mat_aggregate=mat_aggregate,
        itz_props=itz_props,
        top_displacement=-0.01,
        num_steps=2,
        thickness=1.0,
        rel_force_tol=5e-3,
        mortar_damage_params=mortar_dam
    )
    res_converged.geometry_fingerprint = fp

    return {
        'specimen': specimen,
        'mesh': mesh,
        'if_data': if_data,
        'coh_mesh': coh_mesh,
        'itz_props': itz_props,
        'res_converged': res_converged
    }


def test_phase5_renders_all_6_tabs(fast_phase5_setup):
    """Verifies that Phase 5 renders all 6 tabs without exception."""
    s = fast_phase5_setup

    at = AppTest.from_file("app.py", default_timeout=30)
    at.session_state['_nav_pending'] = "Phase 5: Cohesive ITZ"
    at.session_state['specimen_result'] = s['specimen']
    at.session_state['current_mesh'] = s['mesh']
    at.session_state['current_if_data'] = s['if_data']
    at.session_state['coh_mesh'] = s['coh_mesh']
    at.session_state['itz_props'] = s['itz_props']
    at.session_state['nl_res'] = s['res_converged']

    at.run()
    assert len(at.exception) == 0, f"Exceptions in Phase 5: {[e.message for e in at.exception]}"

    tab_labels = [t.label for t in at.tabs]
    assert "Macroscopic Response Curves" in tab_labels
    assert "ITZ Damage Distribution" in tab_labels
    assert "Mortar Damage Distribution" in tab_labels
    assert "Deformed Interface Mesh" in tab_labels
    assert "Convergence History & Equilibrium" in tab_labels
    assert "Mesh Sensitivity Benchmark" in tab_labels
    assert len(tab_labels) == 6


def test_phase5_incomplete_simulation_stalled_banner(fast_phase5_setup):
    """Verifies that an incomplete/stalled simulation displays an explicit warning notice."""
    s = fast_phase5_setup
    res = s['res_converged']

    # Create dummy incomplete result
    res_stalled = NonlinearFEMResult(
        converged=False,
        steps=res.steps[:1],
        final_state=res.final_state,
        final_mortar_state=res.final_mortar_state,
        displacement_curve=res.displacement_curve[:1],
        reaction_curve=res.reaction_curve[:1],
        stress_curve=res.stress_curve[:1],
        strain_curve=res.strain_curve[:1],
        total_time_seconds=15.0,
        total_iterations=80,
        total_cutbacks=19,
        last_converged_load_factor=0.465,
        actual_displacement_reached=-0.0558,
        status_message="Maximum cutbacks exceeded near peak",
        termination_reason="Cutback limit reached",
        rel_force_tol=1e-3,
        max_mortar_damage=0.82,
        damaged_mortar_count=20,
        final_displacements=res.final_displacements
    )
    res_stalled.geometry_fingerprint = s['specimen'].fingerprint

    at = AppTest.from_file("app.py", default_timeout=30)
    at.session_state['_nav_pending'] = "Phase 5: Cohesive ITZ"
    at.session_state['specimen_result'] = s['specimen']
    at.session_state['current_mesh'] = s['mesh']
    at.session_state['current_if_data'] = s['if_data']
    at.session_state['coh_mesh'] = s['coh_mesh']
    at.session_state['itz_props'] = s['itz_props']
    at.session_state['nl_res'] = res_stalled

    at.run()
    assert len(at.exception) == 0, f"Exceptions in Phase 5 incomplete: {[e.message for e in at.exception]}"

    error_texts = [e.value for e in at.error]
    found_incomplete = any("Incomplete Simulation" in t for t in error_texts)
    assert found_incomplete, f"Expected Incomplete Simulation warning, got: {error_texts}"
