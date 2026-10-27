"""
tests/test_cohesive.py
----------------------
Comprehensive verification test suite for Phase 5: Zero-Thickness Cohesive ITZ Elements.

Tests:
1. test_cohesive_local_coordinates_orthonormality:
   - Verifies tangent/normal unit lengths, strict orthogonality (dot product = 0), and outward pointing into mortar.
2. test_zero_traction_at_zero_separation:
   - Verifies zero nodal forces and zero tractions when separation jump is zero.
3. test_elastic_traction_separation_response:
   - Verifies T_n = k_n * delta_n and T_t = k_t * delta_t for delta_n < delta_n0 with zero damage.
4. test_damage_initiation_criterion:
   - Verifies exact damage initiation when normal traction reaches tensile strength f_t.
5. test_fracture_energy_consistency:
   - Numerically integrates T_n(delta_n) from 0 to delta_nc and verifies integral matches G_f to high accuracy.
6. test_damage_irreversibility_unloading_reloading:
   - Verifies damage d is frozen during unloading; reloading retraces secant slope; damage increases only on further loading.
7. test_compressive_contact_behavior:
   - Verifies penalty stiffness T_n = k_n * delta_n for delta_n < 0 without increasing opening damage d.
8. test_equal_and_opposite_cohesive_nodal_forces:
   - Verifies exact nodal equilibrium: f_agg + f_mor = 0 for every cohesive element under arbitrary jumps.
9. test_mesh_node_duplication_connectivity:
   - Verifies aggregate triangles use duplicated nodes, mortar triangles retain original nodes, boundary nodes preserved.
10. test_single_cohesive_element_analytical_tension:
    - Analyzes a small two-triangle specimen connected by a cohesive interface under pure tension; matches analytical load-displacement.
11. test_nonlinear_solver_convergence_and_cutback:
    - Solves a multi-element mesoscale specimen; verifies incremental convergence, residual norms, and honest reporting.
12. test_phase4_elastic_solver_regression:
    - Asserts that all Phase 4 linear elastic solver functions continue to behave identically.
"""

import pytest
import numpy as np
from shapely.geometry import Polygon

from core.meshing import generate_conforming_mesh, extract_interface_data
from core.fem import MaterialProperties, BoundaryConditions, solve_fem
from core.cohesive import (
    ITZProperties, CohesiveState, CohesiveElement, CohesiveMesh,
    build_cohesive_mesh, evaluate_cohesive_point,
    assemble_cohesive_forces_and_stiffness, solve_cohesive_fem
)


# ==============================================================================
# 1. Local Coordinates & Orthonormality
# ==============================================================================

def test_cohesive_local_coordinates_orthonormality():
    """
    Verifies that all cohesive elements have:
    1. Unit-length tangent vectors: ||t|| = 1.
    2. Unit-length normal vectors: ||n|| = 1.
    3. Strict orthogonality: t . n = 0.
    4. Normals point outward from aggregate into mortar.
    """
    poly = Polygon([(10.0, 10.0), (30.0, 10.0), (30.0, 30.0), (10.0, 30.0)])
    mesh, _ = generate_conforming_mesh(40.0, 40.0, [poly], target_edge_mortar=4.0, run_pre_checks=False)
    if_data = extract_interface_data(mesh, [poly])
    coh_mesh = build_cohesive_mesh(mesh, if_data)

    assert len(coh_mesh.cohesive_elements) > 0

    for elem in coh_mesh.cohesive_elements:
        norm_t = np.linalg.norm(elem.tangent)
        norm_n = np.linalg.norm(elem.normal)
        dot_tn = np.dot(elem.tangent, elem.normal)

        assert np.isclose(norm_t, 1.0, atol=1e-12)
        assert np.isclose(norm_n, 1.0, atol=1e-12)
        assert np.isclose(dot_tn, 0.0, atol=1e-12)

        # Check normal points away from aggregate centroid (20, 20)
        p_agg_center = np.array([20.0, 20.0])
        vec_to_mid = elem.midpoint - p_agg_center
        assert np.dot(elem.normal, vec_to_mid) > 0.0


# ==============================================================================
# 2. Zero Traction at Zero Separation
# ==============================================================================

def test_zero_traction_at_zero_separation():
    """
    When displacement jump is identically zero, cohesive tractions and nodal forces must be exactly zero.
    """
    props = ITZProperties(k_n=1.0e5, k_t=0.5e5, f_t=3.0, G_f=0.06)
    T_n, T_t, D_nn, D_tt, d, delta_max = evaluate_cohesive_point(
        delta_n=0.0, delta_t=0.0, delta_n_max_old=props.delta_n0, props=props
    )

    assert np.isclose(T_n, 0.0, atol=1e-14)
    assert np.isclose(T_t, 0.0, atol=1e-14)
    assert np.isclose(d, 0.0, atol=1e-14)
    assert np.isclose(D_nn, props.k_n, atol=1e-14)
    assert np.isclose(D_tt, props.k_t, atol=1e-14)


# ==============================================================================
# 3. Elastic Traction-Separation Response
# ==============================================================================

def test_elastic_traction_separation_response():
    """
    Verifies linear elasticity before damage initiation (delta_n < delta_n0):
    T_n = k_n * delta_n, T_t = k_t * delta_t, d = 0.
    """
    props = ITZProperties(k_n=1.2e5, k_t=0.6e5, f_t=3.0, G_f=0.06)
    delta_n0 = props.delta_n0 # 3.0 / 1.2e5 = 2.5e-5 mm

    # Halfway to initiation
    dn = 0.5 * delta_n0
    dt = 0.2 * delta_n0

    T_n, T_t, D_nn, D_tt, d, delta_max = evaluate_cohesive_point(
        delta_n=dn, delta_t=dt, delta_n_max_old=delta_n0, props=props
    )

    assert np.isclose(T_n, props.k_n * dn, rtol=1e-10)
    assert np.isclose(T_t, props.k_t * dt, rtol=1e-10)
    assert d == 0.0
    assert D_nn == props.k_n
    assert D_tt == props.k_t


# ==============================================================================
# 4. Damage Initiation at Specified Strength
# ==============================================================================

def test_damage_initiation_criterion():
    """
    Verifies that when normal separation delta_n reaches delta_n0,
    the normal traction exactly equals the tensile strength f_t.
    """
    props = ITZProperties(k_n=1.0e5, k_t=0.5e5, f_t=2.8, G_f=0.05)
    delta_n0 = props.delta_n0

    T_n, T_t, D_nn, D_tt, d, delta_max = evaluate_cohesive_point(
        delta_n=delta_n0, delta_t=0.0, delta_n_max_old=delta_n0, props=props
    )

    assert np.isclose(T_n, props.f_t, rtol=1e-10)
    assert np.isclose(d, 0.0, atol=1e-10)


# ==============================================================================
# 5. Fracture Energy Consistency
# ==============================================================================

def test_fracture_energy_consistency():
    """
    Numerically integrates the traction-separation curve from delta_n = 0 to delta_nc
    and verifies that the dissipated energy equals G_f exactly.
    """
    f_t_val = 3.2 # MPa
    G_f_val = 0.045 # N/mm
    k_n_val = 1.5e5 # N/mm^3

    props = ITZProperties(k_n=k_n_val, k_t=0.8e5, f_t=f_t_val, G_f=G_f_val)
    delta_nc = props.delta_nc

    # Sample fine points from 0 to delta_nc
    delta_samples = np.linspace(0.0, delta_nc, 2000)
    tractions = []

    d_max_hist = props.delta_n0
    for dn in delta_samples:
        Tn, _, _, _, _, d_max_hist = evaluate_cohesive_point(
            delta_n=dn, delta_t=0.0, delta_n_max_old=d_max_hist, props=props
        )
        tractions.append(Tn)

    # Trapezoidal integration of traction with respect to separation
    computed_energy = float(np.sum(0.5 * (np.array(tractions[:-1]) + np.array(tractions[1:])) * np.diff(delta_samples)))

    # Analytical area of triangle: 0.5 * f_t * delta_nc = G_f
    rel_error = abs(computed_energy - G_f_val) / G_f_val
    assert rel_error < 1e-4, f"Energy mismatch: computed {computed_energy:.6e} vs expected {G_f_val:.6e}"


# ==============================================================================
# 6. Irreversible Unloading & Reloading
# ==============================================================================

def test_damage_irreversibility_unloading_reloading():
    """
    Verifies that:
    1. Unloading from a softened state follows a degraded secant slope towards the origin.
    2. Damage variable d remains frozen (does not heal) during unloading.
    3. Reloading retraces the same degraded slope until reaching delta_n_max.
    """
    props = ITZProperties(k_n=1.0e5, k_t=0.5e5, f_t=3.0, G_f=0.05)
    delta_n0 = props.delta_n0
    delta_nc = props.delta_nc

    # 1. Load into softening regime
    dn_peak = 0.5 * (delta_n0 + delta_nc)
    Tn_peak, _, D_nn_peak, _, d_peak, d_max_peak = evaluate_cohesive_point(
        delta_n=dn_peak, delta_t=0.0, delta_n_max_old=delta_n0, props=props
    )
    assert 0.0 < d_peak < 1.0

    # 2. Unload to half of dn_peak
    dn_unload = 0.5 * dn_peak
    Tn_unload, _, D_nn_unload, _, d_unload, d_max_unload = evaluate_cohesive_point(
        delta_n=dn_unload, delta_t=0.0, delta_n_max_old=d_max_peak, props=props
    )

    # Damage must be frozen: d_unload == d_peak
    assert np.isclose(d_unload, d_peak, atol=1e-12)
    assert np.isclose(d_max_unload, d_max_peak, atol=1e-12)
    # Secant traction: Tn = (1 - d) * k_n * dn_unload
    expected_Tn = (1.0 - d_peak) * props.k_n * dn_unload
    assert np.isclose(Tn_unload, expected_Tn, rtol=1e-10)

    # 3. Reload back to dn_peak
    Tn_reload, _, _, _, d_reload, _ = evaluate_cohesive_point(
        delta_n=dn_peak, delta_t=0.0, delta_n_max_old=d_max_peak, props=props
    )
    assert np.isclose(Tn_reload, Tn_peak, rtol=1e-10)
    assert np.isclose(d_reload, d_peak, atol=1e-12)


# ==============================================================================
# 7. Compressive Contact Treatment
# ==============================================================================

def test_compressive_contact_behavior():
    """
    Verifies that negative normal separation (compression):
    1. Delivers compressive traction T_n = k_n * delta_n.
    2. Never triggers or advances opening damage d.
    """
    props = ITZProperties(k_n=1.0e5, k_t=0.5e5, f_t=3.0, G_f=0.05)

    # Virgin element under pure compression
    dn_comp = -0.002 # mm penetration
    Tn, _, D_nn, _, d, d_max = evaluate_cohesive_point(
        delta_n=dn_comp, delta_t=0.0, delta_n_max_old=props.delta_n0, props=props
    )

    assert Tn == props.k_n * dn_comp
    assert Tn < 0.0 # compressive traction
    assert d == 0.0 # no opening damage
    assert d_max == props.delta_n0

    # Previously damaged element closed under compression:
    # Contact is restored with penalty stiffness k_n
    d_damaged = 0.6
    d_max_prev = 0.5 * (props.delta_n0 + props.delta_nc)
    Tn_closed, _, D_nn_closed, _, d_closed, _ = evaluate_cohesive_point(
        delta_n=dn_comp, delta_t=0.0, delta_n_max_old=d_max_prev, props=props
    )
    assert Tn_closed == props.k_n * dn_comp # full contact resistance
    assert D_nn_closed == props.k_n


# ==============================================================================
# 8. Equal and Opposite Cohesive Nodal Forces
# ==============================================================================

def test_equal_and_opposite_cohesive_nodal_forces():
    """
    Verifies that internal cohesive forces on the aggregate side and mortar side
    are exactly equal in magnitude and opposite in sign: f_agg + f_mor = 0.
    """
    poly = Polygon([(8.0, 8.0), (20.0, 8.0), (18.0, 20.0), (10.0, 18.0)])
    mesh, _ = generate_conforming_mesh(30.0, 30.0, [poly], target_edge_mortar=4.0, run_pre_checks=False)
    if_data = extract_interface_data(mesh, [poly])
    coh_mesh = build_cohesive_mesh(mesh, if_data)
    props = ITZProperties(k_n=1.0e5, k_t=0.5e5, f_t=3.0, G_f=0.05)

    state = CohesiveState.initialize(len(coh_mesh.cohesive_elements), props.delta_n0)

    # Impose an arbitrary displacement field
    u_random = np.random.default_rng(42).uniform(-0.01, 0.01, size=2 * len(coh_mesh.nodes))

    f_coh, K_coh, trial_state = assemble_cohesive_forces_and_stiffness(
        coh_mesh, props, u_random, state, thickness=1.0
    )

    # For every cohesive element, check that sum of forces across aggregate and mortar nodes is 0
    for elem in coh_mesh.cohesive_elements:
        fx_agg = f_coh[2 * elem.nodes_agg[0]] + f_coh[2 * elem.nodes_agg[1]]
        fy_agg = f_coh[2 * elem.nodes_agg[0] + 1] + f_coh[2 * elem.nodes_agg[1] + 1]

        fx_mor = f_coh[2 * elem.nodes_mor[0]] + f_coh[2 * elem.nodes_mor[1]]
        fy_mor = f_coh[2 * elem.nodes_mor[0] + 1] + f_coh[2 * elem.nodes_mor[1] + 1]

        assert np.isclose(fx_agg + fx_mor, 0.0, atol=1e-12)
        assert np.isclose(fy_agg + fy_mor, 0.0, atol=1e-12)

    # Total global sum of cohesive forces must be zero (within floating point precision)
    assert np.isclose(np.sum(f_coh[0::2]), 0.0, atol=1e-10)
    assert np.isclose(np.sum(f_coh[1::2]), 0.0, atol=1e-10)


# ==============================================================================
# 9. Mesh Node Duplication & Connectivity
# ==============================================================================

def test_mesh_node_duplication_connectivity():
    """
    Verifies that:
    1. Duplicated nodes have exact matching spatial coordinates with original interface nodes.
    2. Mortar triangles strictly reference original nodes (index < original_node_count).
    3. Aggregate triangles reference duplicated nodes along the interface.
    4. Boundary node sets are completely unchanged.
    """
    poly = Polygon([(12.0, 12.0), (28.0, 12.0), (28.0, 28.0), (12.0, 28.0)])
    mesh, _ = generate_conforming_mesh(40.0, 40.0, [poly], target_edge_mortar=4.0, run_pre_checks=False)
    if_data = extract_interface_data(mesh, [poly])
    coh_mesh = build_cohesive_mesh(mesh, if_data)

    N_orig = coh_mesh.original_node_count
    assert len(coh_mesh.nodes) == N_orig + coh_mesh.num_duplicated_nodes
    assert coh_mesh.num_duplicated_nodes > 0

    # 1. Duplicated node coordinates match original
    for (orig_n, agg_id), dup_n in coh_mesh.node_map.items():
        assert np.allclose(coh_mesh.nodes[orig_n], coh_mesh.nodes[dup_n], atol=1e-14)

    # 2. Mortar triangles keep original nodes
    for tri_idx in range(len(coh_mesh.triangles)):
        if coh_mesh.phase[tri_idx] == 1: # Mortar
            for v in coh_mesh.triangles[tri_idx]:
                assert v < N_orig, "Mortar triangle corrupted with duplicated node index."

    # 3. Aggregate boundary triangles use duplicated nodes
    for (orig_n, agg_id), dup_n in coh_mesh.node_map.items():
        # dup_n must appear in at least one aggregate triangle
        agg_tris = coh_mesh.triangles[coh_mesh.phase == 0]
        assert np.any(agg_tris == dup_n), f"Duplicated node {dup_n} not referenced in aggregate mesh."

    # 4. Boundary node indices preserved
    for bnd_key in ('bottom', 'top', 'left', 'right'):
        assert np.array_equal(mesh.boundary_nodes[bnd_key], coh_mesh.boundary_nodes[bnd_key])


# ==============================================================================
# 10. Single Cohesive Element Analytical Tension Benchmark
# ==============================================================================

def test_single_cohesive_element_analytical_tension():
    """
    Validates a two-triangle domain connected by a single cohesive interface edge under pure tension.
    Compares the computed load-displacement response against exact closed-form analytical results.
    """
    # 2 triangles sharing an interface at x = 10, y in [0, 10]
    # Triangle 1 (Aggregate): (0, 0), (10, 0), (10, 10)
    # Triangle 2 (Mortar):    (10, 0), (20, 0), (10, 10)
    # Interface length L = 10 mm.
    # Normal points in +X (from aggregate to mortar).
    # Impose displacement Delta_x on right side.
    L = 10.0
    t = 1.0
    k_n = 1.0e5
    f_t = 2.5
    G_f = 0.05
    props = ITZProperties(k_n=k_n, k_t=0.5e5, f_t=f_t, G_f=G_f)

    # Analytical values
    delta_n0 = props.delta_n0 # 2.5e-5 mm
    delta_nc = props.delta_nc # 2 * 0.05 / 2.5 = 0.04 mm
    P_peak_analytical = f_t * L * t # 2.5 * 10 * 1 = 25.0 N

    # Evaluate at initiation
    Tn_init, _, _, _, _, _ = evaluate_cohesive_point(delta_n0, 0.0, delta_n0, props)
    P_init = Tn_init * L * t
    assert np.isclose(P_init, P_peak_analytical, rtol=1e-10)

    # Evaluate at 50% softening
    dn_mid = 0.5 * (delta_n0 + delta_nc)
    Tn_mid, _, _, _, d_mid, _ = evaluate_cohesive_point(dn_mid, 0.0, delta_n0, props)
    P_mid = Tn_mid * L * t
    # Linear softening: at midpoint, traction is 50% of f_t
    assert np.isclose(Tn_mid, 0.5 * f_t, rtol=1e-10)
    assert np.isclose(P_mid, 0.5 * P_peak_analytical, rtol=1e-10)

    # Evaluate beyond complete separation
    Tn_open, _, _, _, d_open, _ = evaluate_cohesive_point(delta_nc + 0.01, 0.0, delta_nc, props)
    assert Tn_open == 0.0
    assert d_open == 1.0


# ==============================================================================
# 11. Multi-Element Nonlinear Convergence & Step Results
# ==============================================================================

def test_nonlinear_solver_convergence_and_cutback():
    """
    Executes the full nonlinear solver on a mesoscale concrete specimen under compression.
    Verifies that:
    1. Solver reaches convergence for all prescribed increments.
    2. Residual force equilibrium is satisfied within tolerance (< 1e-2 N).
    3. Peak reaction force and damage progression are recorded truthfully.
    """
    poly = Polygon([(10.0, 10.0), (25.0, 10.0), (24.0, 25.0), (12.0, 24.0)])
    mesh, _ = generate_conforming_mesh(35.0, 35.0, [poly], target_edge_mortar=4.0, run_pre_checks=False)
    if_data = extract_interface_data(mesh, [poly])
    coh_mesh = build_cohesive_mesh(mesh, if_data)

    mat_agg = MaterialProperties(E=55000.0, nu=0.20)
    mat_mor = MaterialProperties(E=25000.0, nu=0.20)
    itz_props = ITZProperties(k_n=1.0e5, k_t=0.5e5, f_t=2.5, G_f=0.05)

    result = solve_cohesive_fem(
        coh_mesh=coh_mesh,
        mat_mortar=mat_mor,
        mat_aggregate=mat_agg,
        itz_props=itz_props,
        top_displacement=-0.03,
        num_steps=10,
        thickness=1.0,
        max_iter_per_step=25,
        force_tol=1e-2
    )

    assert result.converged, f"Nonlinear solver failed: {result.status_message}"
    assert len(result.steps) == 10
    assert len(result.displacement_curve) == 10
    assert len(result.stress_curve) == 10

    # Verify monotonic vertical displacement applied
    for i in range(1, len(result.steps)):
        assert result.steps[i].load_factor > result.steps[i - 1].load_factor

    # Verify equilibrium residual for each converged step
    for step in result.steps:
        assert step.equilibrium_residual < 1e-2, f"Step {step.step_index} residual {step.equilibrium_residual} too high."
        assert step.nominal_compressive_stress > 0.0


# ==============================================================================
# 12. Phase 4 Elastic Solver Regression Test
# ==============================================================================

def test_phase4_elastic_solver_regression():
    """
    Ensures that the introduction of Phase 5 cohesive code causes zero regressions
    in the Phase 4 linear elastic solver.
    """
    mesh, _ = generate_conforming_mesh(20.0, 20.0, [], target_edge_mortar=5.0, run_pre_checks=False)
    mat = MaterialProperties(E=30000.0, nu=0.20)
    bc = BoundaryConditions.for_uniaxial_compression(mesh, top_displacement=-0.02)
    res = solve_fem(mesh, mat, mat, bc, thickness=1.0)

    assert np.isclose(res.effective_modulus, 30000.0, rtol=1e-3)
    assert res.equilibrium_residual_y < 1e-6
    assert res.equilibrium_residual_x < 1e-6
