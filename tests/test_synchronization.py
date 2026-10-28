"""
tests/test_synchronization.py
-----------------------------
Comprehensive test suite verifying geometry-mesh-result synchronization across
Phases 2, 3, 4, and 5 of the Mesoscale Concrete FEM application.

Covers the 10 required test cases:
1. Matching geometry A and mesh A produces is_synchronized=True.
2. Generating geometry B after A invalidates downstream state (clears mesh, FEM, cohesive results).
3. Phase 5 simulation blocked with geometry B and mesh A (fingerprint mismatch).
4. Mesh B from geometry B allows Phase 5 simulation.
5. Changing geometry dimensions (W, H) invalidates old mesh.
6. Changing aggregate coordinates invalidates old mesh even if particle count is unchanged.
7. Identical geometry produces consistent deterministic fingerprint.
8. Phase 4 and Phase 5 results from old mesh cannot be presented as current results (provenance check).
9. Missing mesh produces actionable warning/status, not an unhandled exception.
10. Regenerating mesh or invalidating downstream state does not erase Phase 1 mix-design data in session state.
"""

import numpy as np
import pytest
from shapely.geometry import Polygon, box

from core.geometry import (
    PackingResult, generate_mesostructure, compute_geometry_fingerprint
)
from core.meshing import (
    ConcreteMesh, InterfaceData, generate_conforming_mesh, extract_interface_data
)
from core.fem import (
    MaterialProperties, BoundaryConditions, FEMResult, solve_fem
)
from core.cohesive import (
    ITZProperties, CohesiveState, CohesiveMesh, NonlinearFEMResult,
    build_cohesive_mesh, solve_cohesive_fem
)
from core.synchronization import (
    SynchronizationStatus, check_mesh_synchronization,
    invalidate_downstream_state, check_result_provenance
)


# ==============================================================================
# Helper fixtures / builders
# ==============================================================================

def create_synthetic_specimen(w=50.0, h=50.0, shift=0.0):
    """Create a synthetic PackingResult with two square aggregate polygons."""
    poly1 = Polygon([
        (10.0 + shift, 10.0 + shift),
        (22.0 + shift, 10.0 + shift),
        (22.0 + shift, 22.0 + shift),
        (10.0 + shift, 22.0 + shift)
    ])
    poly2 = Polygon([
        (28.0 + shift, 28.0 + shift),
        (40.0 + shift, 28.0 + shift),
        (40.0 + shift, 40.0 + shift),
        (28.0 + shift, 40.0 + shift)
    ])
    polygons = [poly1, poly2]
    fp = compute_geometry_fingerprint(w, h, polygons)
    return PackingResult(
        target_area_fraction=0.115,
        achieved_area_fraction=0.115,
        area_fraction_error=0.0,
        total_particles_sampled=2,
        total_particles_placed=2,
        total_particles_unplaced=0,
        packing_status="target_reached",
        stop_reason="target_reached",
        random_seed=42,
        domain_width=w,
        domain_height=h,
        shape_class="angular",
        min_spacing=1.0,
        conversion_method="walraven",
        grading_source="IS 383:2016",
        placed_aggregates=[],
        polygons=polygons,
        bin_reports=[],
        sieve_sizes=[],
        target_percent_passing=[],
        achieved_percent_passing=[],
        geometry_fingerprint=fp
    )


# ==============================================================================
# Test Cases
# ==============================================================================

def test_1_matching_geometry_and_mesh():
    """Test 1: Matching geometry A and mesh A produces is_synchronized=True and matching fingerprints."""
    specimen_a = create_synthetic_specimen(50.0, 50.0, shift=0.0)
    mesh_a, pre_rep = generate_conforming_mesh(
        domain_width=specimen_a.domain_width,
        domain_height=specimen_a.domain_height,
        polygons=specimen_a.polygons,
        target_edge_mortar=5.0,
        target_edge_aggregate=5.0,
        run_pre_checks=False,
        geometry_fingerprint=specimen_a.geometry_fingerprint
    )

    sync = check_mesh_synchronization(specimen_a, mesh_a)

    assert sync.is_synchronized is True
    assert sync.reason == "synchronized"
    assert sync.status_code == "READY"
    assert sync.geometry_fingerprint == specimen_a.geometry_fingerprint
    assert sync.mesh_fingerprint == specimen_a.geometry_fingerprint
    assert "synchronized" in sync.message.lower()


def test_2_geometry_b_invalidates_downstream_state():
    """Test 2: Generating geometry B after A invalidates downstream state (clears mesh, FEM, cohesive results)."""
    specimen_a = create_synthetic_specimen(50.0, 50.0, shift=0.0)
    mesh_a, _ = generate_conforming_mesh(
        domain_width=specimen_a.domain_width,
        domain_height=specimen_a.domain_height,
        polygons=specimen_a.polygons,
        target_edge_mortar=5.0,
        target_edge_aggregate=5.0,
        run_pre_checks=False,
        geometry_fingerprint=specimen_a.geometry_fingerprint
    )

    # Populate session state with Phase 1, Phase 2, Phase 3, Phase 4, Phase 5 artifacts
    session_state = {
        # Phase 1 Mix Design
        "vol_ca_fraction_of_concrete": 0.68,
        "target_mean_strength": 38.25,
        "mix_data": {"water": 186.0, "cement": 380.0},
        # Phase 2 Geometry
        "specimen_result": specimen_a,
        # Phase 3 Mesh
        "current_mesh": mesh_a,
        "current_if_data": "mock_interface_data",
        "mesh_report": "mock_report",
        # Phase 4 FEM
        "fem_result": "mock_fem_result",
        "fem_params": {"disp": -0.05},
        # Phase 5 Cohesive
        "coh_mesh": "mock_coh_mesh",
        "nl_res": "mock_nl_res",
        "itz_props": "mock_itz_props"
    }

    # Invalidate downstream state from "geometry"
    invalidate_downstream_state(session_state, source="geometry")

    # Downstream keys must be removed
    for key in ["current_mesh", "current_if_data", "mesh_report",
                    "fem_result", "fem_params", "coh_mesh", "nl_res", "itz_props"]:
        assert key not in session_state, f"Key {key} should have been cleared from session state"

    # Upstream Mix Design and current specimen must be preserved
    assert session_state["vol_ca_fraction_of_concrete"] == 0.68
    assert session_state["target_mean_strength"] == 38.25
    assert "mix_data" in session_state
    assert session_state["specimen_result"] is specimen_a


def test_3_phase5_blocked_with_geometry_b_and_mesh_a():
    """Test 3: Phase 5 simulation blocked with geometry B and mesh A (fingerprint mismatch)."""
    specimen_a = create_synthetic_specimen(50.0, 50.0, shift=0.0)
    specimen_b = create_synthetic_specimen(50.0, 50.0, shift=5.0)

    mesh_a, _ = generate_conforming_mesh(
        domain_width=specimen_a.domain_width,
        domain_height=specimen_a.domain_height,
        polygons=specimen_a.polygons,
        target_edge_mortar=5.0,
        target_edge_aggregate=5.0,
        run_pre_checks=False,
        geometry_fingerprint=specimen_a.geometry_fingerprint
    )

    # Check synchronization between specimen B and mesh A
    sync = check_mesh_synchronization(specimen_b, mesh_a)

    assert sync.is_synchronized is False
    assert sync.reason == "fingerprint_mismatch"
    assert sync.status_code == "STALE_MESH"
    assert sync.geometry_fingerprint == specimen_b.geometry_fingerprint
    assert sync.mesh_fingerprint == mesh_a.geometry_fingerprint
    assert sync.geometry_fingerprint != sync.mesh_fingerprint
    assert "mismatch" in sync.message.lower() or "regenerate" in sync.message.lower() or "changed" in sync.message.lower()


def test_4_mesh_b_from_geometry_b_allows_phase5():
    """Test 4: Mesh B from geometry B restores synchronization and allows Phase 5 simulation."""
    specimen_b = create_synthetic_specimen(50.0, 50.0, shift=3.0)
    mesh_b, _ = generate_conforming_mesh(
        domain_width=specimen_b.domain_width,
        domain_height=specimen_b.domain_height,
        polygons=specimen_b.polygons,
        target_edge_mortar=5.0,
        target_edge_aggregate=5.0,
        run_pre_checks=False,
        geometry_fingerprint=specimen_b.geometry_fingerprint
    )

    sync_b = check_mesh_synchronization(specimen_b, mesh_b)
    assert sync_b.is_synchronized is True
    assert sync_b.reason == "synchronized"

    # Verify cohesive mesh also inherits the fingerprint
    if_data_b = extract_interface_data(mesh_b, specimen_b.polygons)
    coh_mesh_b = build_cohesive_mesh(mesh_b, if_data_b)
    assert coh_mesh_b.geometry_fingerprint == specimen_b.geometry_fingerprint


def test_5_changing_geometry_dimensions_invalidates_mesh():
    """Test 5: Changing geometry dimensions (W, H) invalidates old mesh."""
    polys = [box(10.0, 10.0, 30.0, 30.0)]
    fp_50x50 = compute_geometry_fingerprint(50.0, 50.0, polys)
    fp_60x50 = compute_geometry_fingerprint(60.0, 50.0, polys)
    fp_50x60 = compute_geometry_fingerprint(50.0, 60.0, polys)

    assert fp_50x50 != fp_60x50
    assert fp_50x50 != fp_50x60

    spec_50 = PackingResult(
        target_area_fraction=0.16,
        achieved_area_fraction=0.16,
        area_fraction_error=0.0,
        total_particles_sampled=1,
        total_particles_placed=1,
        total_particles_unplaced=0,
        packing_status="target_reached",
        stop_reason="target_reached",
        random_seed=42,
        domain_width=50.0,
        domain_height=50.0,
        shape_class="angular",
        min_spacing=1.0,
        conversion_method="walraven",
        grading_source="IS 383:2016",
        placed_aggregates=[],
        polygons=polys,
        bin_reports=[],
        sieve_sizes=[],
        target_percent_passing=[],
        achieved_percent_passing=[],
        geometry_fingerprint=fp_50x50
    )
    spec_60 = PackingResult(
        target_area_fraction=0.133,
        achieved_area_fraction=0.133,
        area_fraction_error=0.0,
        total_particles_sampled=1,
        total_particles_placed=1,
        total_particles_unplaced=0,
        packing_status="target_reached",
        stop_reason="target_reached",
        random_seed=42,
        domain_width=60.0,
        domain_height=50.0,
        shape_class="angular",
        min_spacing=1.0,
        conversion_method="walraven",
        grading_source="IS 383:2016",
        placed_aggregates=[],
        polygons=polys,
        bin_reports=[],
        sieve_sizes=[],
        target_percent_passing=[],
        achieved_percent_passing=[],
        geometry_fingerprint=fp_60x50
    )

    mesh_50, _ = generate_conforming_mesh(
        50.0, 50.0, polys, target_edge_mortar=5.0, run_pre_checks=False,
        geometry_fingerprint=spec_50.geometry_fingerprint
    )

    sync = check_mesh_synchronization(spec_60, mesh_50)
    assert sync.is_synchronized is False
    assert sync.reason == "fingerprint_mismatch"
    assert sync.status_code == "STALE_MESH"


def test_6_changing_coordinates_with_same_particle_count():
    """Test 6: Changing aggregate coordinates invalidates old mesh even if particle count is unchanged."""
    spec_a = create_synthetic_specimen(50.0, 50.0, shift=0.0)
    # Move particles by merely 0.05 mm - count is identical (2 particles)
    spec_b = create_synthetic_specimen(50.0, 50.0, shift=0.05)

    assert len(spec_a.polygons) == len(spec_b.polygons) == 2
    assert spec_a.geometry_fingerprint != spec_b.geometry_fingerprint

    mesh_a, _ = generate_conforming_mesh(
        spec_a.domain_width, spec_a.domain_height, spec_a.polygons,
        target_edge_mortar=5.0, run_pre_checks=False,
        geometry_fingerprint=spec_a.geometry_fingerprint
    )

    sync = check_mesh_synchronization(spec_b, mesh_a)
    assert sync.is_synchronized is False
    assert sync.reason == "fingerprint_mismatch"


def test_7_identical_geometry_deterministic_fingerprint():
    """Test 7: Identical geometry produces consistent deterministic fingerprint."""
    poly1 = box(15.0, 15.0, 35.0, 35.0)
    poly2 = box(45.0, 45.0, 65.0, 65.0)
    polys = [poly1, poly2]

    fp1 = compute_geometry_fingerprint(100.0, 100.0, polys)
    fp2 = compute_geometry_fingerprint(100.0, 100.0, polys)
    assert fp1 == fp2
    assert len(fp1) == 16
    assert all(c in "0123456789abcdef" for c in fp1)

    # Also test identical coordinates constructed separately
    polys_dup = [box(15.0, 15.0, 35.0, 35.0), box(45.0, 45.0, 65.0, 65.0)]
    fp3 = compute_geometry_fingerprint(100.0, 100.0, polys_dup)
    assert fp1 == fp3


def test_8_phase4_and_phase5_provenance_check():
    """Test 8: Phase 4 and Phase 5 results from old mesh cannot be presented as current results."""
    spec_a = create_synthetic_specimen(50.0, 50.0, shift=0.0)
    spec_b = create_synthetic_specimen(50.0, 50.0, shift=4.0)

    mesh_a, _ = generate_conforming_mesh(
        spec_a.domain_width, spec_a.domain_height, spec_a.polygons,
        target_edge_mortar=5.0, run_pre_checks=False,
        geometry_fingerprint=spec_a.geometry_fingerprint
    )
    mesh_b, _ = generate_conforming_mesh(
        spec_b.domain_width, spec_b.domain_height, spec_b.polygons,
        target_edge_mortar=5.0, run_pre_checks=False,
        geometry_fingerprint=spec_b.geometry_fingerprint
    )

    # Solve Phase 4 on mesh A
    mat_agg = MaterialProperties(E=50000.0, nu=0.2)
    mat_mor = MaterialProperties(E=25000.0, nu=0.2)
    bc_a = BoundaryConditions.for_uniaxial_compression(mesh_a, top_displacement=-0.01)
    res_a = solve_fem(mesh_a, mat_mor, mat_agg, bc_a)

    assert res_a.geometry_fingerprint == spec_a.geometry_fingerprint

    # res_a matches spec_a and mesh_a
    assert check_result_provenance(spec_a, mesh_a, res_a) is True

    # res_a MUST NOT match spec_b and mesh_b
    assert check_result_provenance(spec_b, mesh_b, res_a) is False
    assert check_result_provenance(spec_b, mesh_a, res_a) is False
    assert check_result_provenance(spec_a, mesh_b, res_a) is False

    # Simulate cohesive result with fingerprint A
    dummy_nl_res = NonlinearFEMResult(
        converged=True,
        status_message="OK",
        steps=[],
        final_displacements=np.zeros((len(mesh_a.nodes), 2)),
        final_state=CohesiveState.initialize(1, 1e-4),
        total_time_seconds=1.0,
        displacement_curve=np.array([0.0]),
        reaction_curve=np.array([0.0]),
        stress_curve=np.array([0.0]),
        strain_curve=np.array([0.0]),
        geometry_fingerprint=spec_a.geometry_fingerprint
    )
    assert check_result_provenance(spec_a, mesh_a, dummy_nl_res) is True
    assert check_result_provenance(spec_b, mesh_b, dummy_nl_res) is False


def test_9_missing_mesh_or_geometry_handled_gracefully():
    """Test 9: Missing mesh or geometry produces actionable status, not an unhandled exception."""
    spec_a = create_synthetic_specimen(50.0, 50.0, shift=0.0)

    # 1. Mesh is None
    sync_no_mesh = check_mesh_synchronization(spec_a, None)
    assert sync_no_mesh.is_synchronized is False
    assert sync_no_mesh.reason == "mesh_missing"
    assert sync_no_mesh.status_code == "MISSING_MESH"
    assert sync_no_mesh.mesh_fingerprint == ""
    assert sync_no_mesh.geometry_fingerprint == spec_a.geometry_fingerprint
    assert "no conforming mesh" in sync_no_mesh.message.lower()

    # 2. Specimen is None
    mesh_a, _ = generate_conforming_mesh(
        spec_a.domain_width, spec_a.domain_height, spec_a.polygons,
        target_edge_mortar=5.0, run_pre_checks=False,
        geometry_fingerprint=spec_a.geometry_fingerprint
    )
    sync_no_geom = check_mesh_synchronization(None, mesh_a)
    assert sync_no_geom.is_synchronized is False
    assert sync_no_geom.reason == "geometry_missing"
    assert sync_no_geom.status_code == "MISSING_GEOMETRY"
    assert sync_no_geom.geometry_fingerprint == ""

    # 3. Both are None
    sync_both_none = check_mesh_synchronization(None, None)
    assert sync_both_none.is_synchronized is False

    # 4. Provenance check with None
    assert check_result_provenance(spec_a, None, None) is False
    assert check_result_provenance(None, mesh_a, None) is False


def test_10_phase1_mix_design_preserved_on_invalidation():
    """Test 10: Regenerating mesh or geometry does not erase Phase 1 mix-design data in session state."""
    session_state = {
        # Phase 1 Mix Design parameters
        "grade": "M30",
        "max_size": 20.0,
        "slump": 100.0,
        "target_mean_strength": 38.25,
        "vol_ca_fraction_of_concrete": 0.62,
        "mix_data": {
            "water": 186.0,
            "cement": 372.0,
            "fine_aggregate": 710.0,
            "coarse_aggregate": 1150.0
        },
        # Phase 2 Specimen
        "specimen_result": "active_specimen",
        # Phase 3 Mesh
        "current_mesh": "active_mesh",
        "current_if_data": "active_if_data",
        # Phase 4
        "fem_result": "fem_res",
        # Phase 5
        "nl_res": "nl_res"
    }

    # Invalidate when mesh is regenerated (source="mesh")
    invalidate_downstream_state(session_state, source="mesh")

    # Phase 4 and Phase 5 are cleared, but Phase 1 AND Phase 2 are preserved
    assert "fem_result" not in session_state
    assert "nl_res" not in session_state
    assert session_state["specimen_result"] == "active_specimen"
    assert session_state["grade"] == "M30"
    assert session_state["vol_ca_fraction_of_concrete"] == 0.62
    assert session_state["mix_data"]["cement"] == 372.0

    # Invalidate when geometry is regenerated (source="geometry")
    invalidate_downstream_state(session_state, source="geometry")

    # Phase 3, 4, 5 are cleared, but all Phase 1 mix design data remains intact
    assert "current_mesh" not in session_state
    assert "current_if_data" not in session_state
    assert session_state["grade"] == "M30"
    assert session_state["max_size"] == 20.0
    assert session_state["vol_ca_fraction_of_concrete"] == 0.62
    assert session_state["mix_data"]["water"] == 186.0
