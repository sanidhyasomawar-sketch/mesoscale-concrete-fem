"""
tests/test_mesh.py
------------------
Comprehensive unit and validation tests for Phase 3:
1. Synthetic benchmark geometries (square in square, sharp corner, narrow gap).
2. Real concrete mesostructures (rounded, angular, elongated) across multiple seeds.
3. Negative failure tests (overlaps, sub-threshold gaps, inverted elements, tag mismatch).
4. Determinism, .npz round-trip serialization, and element count cap tests.
"""

import os
import tempfile
import numpy as np
import pytest
from shapely.geometry import Polygon, box

from core.aggregates import get_default_grading_curve
from core.geometry import generate_mesostructure
from core.meshing import (
    ConcreteMesh, InterfaceData, generate_conforming_mesh,
    extract_interface_data, save_mesh_npz, load_mesh_npz,
    build_edge_to_triangles
)
from core.mesh_checks import (
    check_pre_mesh_geometry, validate_mesh, PreMeshCheckReport, MeshValidationReport
)

# ==============================================================================
# 1. Synthetic Benchmark Tests
# ==============================================================================

def test_synthetic_square_aggregate():
    """Verify conforming triangulation on a square aggregate centered in a square domain."""
    W, H = 100.0, 100.0
    # Center a 40x40 square aggregate at (50, 50) -> [30, 70] x [30, 70]
    sq_poly = box(30.0, 30.0, 70.0, 70.0)
    polygons = [sq_poly]
    
    mesh, pre_report = generate_conforming_mesh(
        domain_width=W, domain_height=H, polygons=polygons,
        target_edge_mortar=4.0, target_edge_aggregate=5.0, min_angle=25.0
    )
    
    assert pre_report.passed
    assert len(mesh.nodes) > 10
    assert len(mesh.triangles) > 10
    
    # Run full mesh validation
    val_report = validate_mesh(
        mesh.nodes, mesh.triangles, mesh.aggregate_id, polygons, W, H,
        angle_fail_threshold=10.0
    )
    assert val_report.passed, f"Validation errors: {val_report.errors}"
    assert np.isclose(np.sum(mesh.triangle_areas), W * H, atol=1e-4)
    
    # Aggregate area should be 40 * 40 = 1600 mm^2
    agg_elems = np.where(mesh.aggregate_id == 0)[0]
    agg_area = np.sum(mesh.triangle_areas[agg_elems])
    assert np.isclose(agg_area, 1600.0, atol=1e-4)
    
    # Interface extraction
    if_data = extract_interface_data(mesh, polygons)
    assert len(if_data.edges) > 0
    # Perimeter of 40x40 square = 160 mm
    assert np.isclose(np.sum(if_data.edge_lengths), 160.0, atol=1e-3)
    
    # Boundary nodes
    assert len(mesh.boundary_nodes['bottom']) >= 2
    assert len(mesh.boundary_nodes['top']) >= 2
    assert len(mesh.boundary_nodes['left']) >= 2
    assert len(mesh.boundary_nodes['right']) >= 2


def test_synthetic_sharp_corner():
    """Verify meshing of a polygon with an acute corner (30 degrees)."""
    W, H = 100.0, 100.0
    # Triangle with 30 degree corner at (50, 20)
    p_sharp = Polygon([(50.0, 20.0), (70.0, 60.0), (30.0, 60.0)])
    polygons = [p_sharp]
    
    mesh, _ = generate_conforming_mesh(
        domain_width=W, domain_height=H, polygons=polygons,
        target_edge_mortar=5.0, target_edge_aggregate=5.0, min_angle=20.0
    )
    
    val_report = validate_mesh(
        mesh.nodes, mesh.triangles, mesh.aggregate_id, polygons, W, H,
        angle_fail_threshold=5.0
    )
    assert val_report.passed


def test_synthetic_narrow_gap():
    """Verify geometry with gap above failure threshold but below warning threshold."""
    W, H = 100.0, 100.0
    # Two rectangular aggregates separated by 0.35 mm gap
    # Above fail threshold (0.15 mm) but below warning threshold (0.50 mm)
    p1 = box(20.0, 30.0, 49.80, 70.0)
    p2 = box(50.15, 30.0, 80.0, 70.0)
    polygons = [p1, p2]
    
    pre_report = check_pre_mesh_geometry(polygons, W, H, gap_fail_threshold=0.15, gap_warn_threshold=0.50)
    assert pre_report.passed
    assert any("Narrow gap" in w for w in pre_report.warnings)

# ==============================================================================
# 2. Real Concrete Mesostructure Validation Across Shapes & Seeds
# ==============================================================================

@pytest.mark.parametrize("shape_class", ["rounded", "angular", "elongated"])
def test_real_specimens_meshing(shape_class):
    """Verify full meshing and validation on real RSP specimens for each shape class."""
    curve = get_default_grading_curve(nominal_max_size=20, curve_type="midpoint")
    W, H = 100.0, 100.0
    
    # Generate real RSP specimen (0.25 area fraction to guarantee robust placement)
    specimen = generate_mesostructure(
        domain_width=W, domain_height=H, target_area_fraction=0.25,
        grading=curve, shape_class=shape_class, min_spacing=1.5,
        random_seed=42, max_attempts_per_particle=800
    )
    assert len(specimen.polygons) > 0
    
    # Generate conforming mesh
    mesh, pre_report = generate_conforming_mesh(
        domain_width=W, domain_height=H, polygons=specimen.polygons,
        target_edge_mortar=2.5, target_edge_aggregate=4.0, min_angle=20.0,
        gap_fail_threshold=0.15
    )
    assert pre_report.passed
    
    # Run post-mesh checks
    val_report = validate_mesh(
        mesh.nodes, mesh.triangles, mesh.aggregate_id, specimen.polygons, W, H,
        angle_fail_threshold=8.0
    )
    assert val_report.passed, f"Validation failed for {shape_class}: {val_report.errors}"
    
    # Extract interface topology
    if_data = extract_interface_data(mesh, specimen.polygons)
    assert len(if_data.edges) > 0
    
    # Total interface perimeter check
    total_polygon_perimeter = sum(p.length for p in specimen.polygons)
    total_interface_length = np.sum(if_data.edge_lengths)
    assert np.isclose(total_interface_length, total_polygon_perimeter, atol=1e-2)

# ==============================================================================
# 3. Negative Tests (Proving Checks Fail Loudly)
# ==============================================================================

def test_negative_overlapping_polygons():
    """Verify that overlapping polygons are rejected immediately by pre-mesh checks."""
    W, H = 100.0, 100.0
    p1 = box(20.0, 20.0, 60.0, 60.0)
    p2 = box(50.0, 50.0, 80.0, 80.0)  # Overlaps p1
    
    with pytest.raises(ValueError, match="Pre-mesh geometry checks failed"):
        generate_conforming_mesh(W, H, [p1, p2], run_pre_checks=True)


def test_negative_too_narrow_gap():
    """Verify that inter-aggregate gap below failure threshold is rejected."""
    W, H = 100.0, 100.0
    p1 = box(20.0, 20.0, 49.95, 60.0)
    p2 = box(50.00, 20.0, 80.0, 60.0)  # Gap is 0.05 mm < 0.15 mm
    
    with pytest.raises(ValueError, match="Pre-mesh geometry checks failed"):
        generate_conforming_mesh(W, H, [p1, p2], gap_fail_threshold=0.15, run_pre_checks=True)


def test_negative_flipped_material_tag():
    """Verify that a deliberately corrupted material tag is caught by validation."""
    W, H = 100.0, 100.0
    sq_poly = box(30.0, 30.0, 70.0, 70.0)
    mesh, _ = generate_conforming_mesh(W, H, [sq_poly], target_edge_mortar=5.0)
    
    # Deliberately flip one mortar triangle to aggregate
    mortar_idx = np.where(mesh.aggregate_id == -1)[0][0]
    corrupted_tags = mesh.aggregate_id.copy()
    corrupted_tags[mortar_idx] = 0  # Flipped!
    
    val_report = validate_mesh(mesh.nodes, mesh.triangles, corrupted_tags, [sq_poly], W, H)
    assert not val_report.passed
    assert any("Independent centroid point-in-polygon verification failed" in e for e in val_report.errors)


def test_negative_inverted_triangle():
    """Verify that a triangle with inverted orientation (CW) is caught."""
    W, H = 100.0, 100.0
    sq_poly = box(30.0, 30.0, 70.0, 70.0)
    mesh, _ = generate_conforming_mesh(W, H, [sq_poly], target_edge_mortar=5.0)
    
    # Deliberately invert vertex order of element 0
    corrupted_tris = mesh.triangles.copy()
    corrupted_tris[0, [1, 2]] = corrupted_tris[0, [2, 1]]
    
    val_report = validate_mesh(mesh.nodes, corrupted_tris, mesh.aggregate_id, [sq_poly], W, H)
    assert not val_report.passed
    assert any("inverted or degenerate" in e for e in val_report.errors)

# ==============================================================================
# 4. Determinism, Serialization & Safety Limit Tests
# ==============================================================================

def test_mesh_determinism():
    """Verify identical inputs and geometry produce exact identical nodes and triangles."""
    W, H = 100.0, 100.0
    p1 = box(20.0, 20.0, 45.0, 50.0)
    p2 = box(55.0, 40.0, 80.0, 75.0)
    polys = [p1, p2]
    
    mesh1, _ = generate_conforming_mesh(W, H, polys, target_edge_mortar=3.0, target_edge_aggregate=4.0)
    mesh2, _ = generate_conforming_mesh(W, H, polys, target_edge_mortar=3.0, target_edge_aggregate=4.0)
    
    assert np.array_equal(mesh1.nodes, mesh2.nodes)
    assert np.array_equal(mesh1.triangles, mesh2.triangles)
    assert np.array_equal(mesh1.aggregate_id, mesh2.aggregate_id)


def test_save_load_npz_round_trip():
    """Verify serialization to .npz and loading preserves all arrays and attributes."""
    W, H = 100.0, 100.0
    sq_poly = box(30.0, 30.0, 70.0, 70.0)
    mesh, _ = generate_conforming_mesh(W, H, [sq_poly], target_edge_mortar=4.0)
    if_data = extract_interface_data(mesh, [sq_poly])
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_filepath = os.path.join(tmpdir, "test_mesh.npz")
        save_mesh_npz(mesh, tmp_filepath, interface=if_data)
        
        loaded_mesh, loaded_if = load_mesh_npz(tmp_filepath)
        
        assert np.allclose(mesh.nodes, loaded_mesh.nodes)
        assert np.array_equal(mesh.triangles, loaded_mesh.triangles)
        assert np.array_equal(mesh.aggregate_id, loaded_mesh.aggregate_id)
        assert np.array_equal(mesh.phase, loaded_mesh.phase)
        assert np.allclose(mesh.triangle_areas, loaded_mesh.triangle_areas)
        assert np.array_equal(mesh.boundary_nodes['bottom'], loaded_mesh.boundary_nodes['bottom'])
        assert np.array_equal(mesh.boundary_nodes['top'], loaded_mesh.boundary_nodes['top'])
        assert np.isclose(mesh.domain_width, loaded_mesh.domain_width)
        assert np.isclose(mesh.domain_height, loaded_mesh.domain_height)
        
        assert loaded_if is not None
        assert np.array_equal(if_data.edges, loaded_if.edges)
        assert np.allclose(if_data.edge_lengths, loaded_if.edge_lengths)
        assert np.allclose(if_data.unit_normals, loaded_if.unit_normals)


def test_element_count_cap_guard():
    """Verify that requesting an ultra-fine mesh exceeding max_triangles raises RuntimeError."""
    W, H = 100.0, 100.0
    sq_poly = box(30.0, 30.0, 70.0, 70.0)
    
    # Requesting 0.1 mm target edge would produce > 1,000,000 triangles
    with pytest.raises(RuntimeError, match="maximum allowed element cap"):
        generate_conforming_mesh(
            W, H, [sq_poly], target_edge_mortar=0.2, target_edge_aggregate=0.2,
            max_triangles=2000
        )
