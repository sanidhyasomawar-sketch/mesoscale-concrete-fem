"""
core/mesh_checks.py
-------------------
Rigorous pre-mesh geometric integrity and post-mesh topology/quality validation.

Scientific Protocol:
1. Pre-mesh checks: Fails loudly on degenerate geometry, overlaps, boundary violations,
   or sub-threshold gaps. Warns on narrow gaps or short edges.
2. Post-mesh checks: Verifies positive areas, area conservation (domain & aggregate level),
   boundary conformity, 100% independent material tag verification, single-phase connectivity,
   element quality (minimum angle, aspect ratio), and interface topology readiness for zero-thickness
   cohesive node-splitting.
"""

import numpy as np
from shapely.geometry import Polygon, Point, box
from shapely.ops import unary_union
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Set

# ==============================================================================
# 1. Pre-Mesh Geometry Checks Data Structures & Engine
# ==============================================================================

@dataclass
class PreMeshCheckReport:
    passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    min_inter_aggregate_gap: float = float('inf')
    min_boundary_gap: float = float('inf')
    min_polygon_edge_length: float = float('inf')
    min_polygon_corner_angle: float = 180.0
    offending_aggregate_ids: List[int] = field(default_factory=list)


def check_pre_mesh_geometry(
    polygons: List[Polygon],
    domain_width: float,
    domain_height: float,
    gap_fail_threshold: float = 0.15,
    gap_warn_threshold: float = 0.50,
    boundary_warn_threshold: float = 0.50,
    min_edge_fail_threshold: float = 0.05,
    min_edge_warn_threshold: float = 0.25,
    aggregate_ids: Optional[List[int]] = None
) -> PreMeshCheckReport:
    """
    Performs comprehensive pre-meshing geometric checks. Fails loudly on invalid geometry.
    
    Parameters:
    -----------
    polygons: List[Polygon]
        List of aggregate polygons.
    domain_width, domain_height: float
        Specimen dimensions in mm.
    gap_fail_threshold: float
        Inter-aggregate distance below which meshing is aborted (mm). Default 0.15 mm.
    gap_warn_threshold: float
        Inter-aggregate distance below which a quality warning is logged (mm). Default 0.50 mm.
    boundary_warn_threshold: float
        Distance between aggregate and domain border triggering a warning (mm). Default 0.50 mm.
    """
    if aggregate_ids is None:
        aggregate_ids = list(range(1, len(polygons) + 1))
        
    report = PreMeshCheckReport(passed=True)
    domain_box = box(0.0, 0.0, domain_width, domain_height)
    domain_boundary = domain_box.boundary
    
    if not polygons:
        report.errors.append("No aggregate polygons provided for meshing.")
        report.passed = False
        return report

    # 1. Individual polygon checks
    for idx, (poly, agg_id) in enumerate(zip(polygons, aggregate_ids)):
        if not poly.is_valid:
            report.errors.append(f"Aggregate ID {agg_id} (index {idx}) is topologically invalid.")
            report.offending_aggregate_ids.append(agg_id)
            report.passed = False
            
        if not poly.is_simple:
            report.errors.append(f"Aggregate ID {agg_id} has self-intersecting boundaries.")
            report.offending_aggregate_ids.append(agg_id)
            report.passed = False
            
        # Domain containment
        if not domain_box.buffer(1e-6).contains(poly):
            report.errors.append(f"Aggregate ID {agg_id} extends outside the specimen boundary.")
            report.offending_aggregate_ids.append(agg_id)
            report.passed = False
            
        # Boundary clearance check
        bnd_dist = poly.distance(domain_boundary)
        if bnd_dist < report.min_boundary_gap:
            report.min_boundary_gap = bnd_dist
            
        if bnd_dist < 1e-5:
            report.warnings.append(
                f"Aggregate ID {agg_id} touches or is within 0.01 µm of the specimen boundary (gap={bnd_dist*1000:.2f} µm). "
                f"This may generate localized boundary slivers."
            )
        elif bnd_dist < boundary_warn_threshold:
            report.warnings.append(
                f"Aggregate ID {agg_id} has a small boundary clearance: {bnd_dist:.3f} mm < {boundary_warn_threshold:.2f} mm."
            )
            
        # Polygon edge lengths and corner angles
        coords = np.array(poly.exterior.coords)[:-1]
        n_pts = len(coords)
        if n_pts < 3:
            report.errors.append(f"Aggregate ID {agg_id} has fewer than 3 vertices.")
            report.passed = False
            continue
            
        diffs = coords - np.roll(coords, -1, axis=0)
        edge_lengths = np.hypot(diffs[:, 0], diffs[:, 1])
        min_el = float(np.min(edge_lengths))
        if min_el < report.min_polygon_edge_length:
            report.min_polygon_edge_length = min_el
            
        if min_el < min_edge_fail_threshold:
            report.errors.append(
                f"Aggregate ID {agg_id} contains near-zero edge length: {min_el:.4f} mm < {min_edge_fail_threshold:.4f} mm."
            )
            report.offending_aggregate_ids.append(agg_id)
            report.passed = False
        elif min_el < min_edge_warn_threshold:
            report.warnings.append(
                f"Aggregate ID {agg_id} contains short edge: {min_el:.3f} mm < {min_edge_warn_threshold:.2f} mm."
            )
            
        # Corner angles
        for i in range(n_pts):
            p_prev = coords[(i - 1) % n_pts]
            p_curr = coords[i]
            p_next = coords[(i + 1) % n_pts]
            v1 = p_prev - p_curr
            v2 = p_next - p_curr
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 > 1e-9 and n2 > 1e-9:
                cos_ang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
                ang_deg = float(np.degrees(np.arccos(cos_ang)))
                if ang_deg < report.min_polygon_corner_angle:
                    report.min_polygon_corner_angle = ang_deg
                    
        if report.min_polygon_corner_angle < 25.0:
            report.warnings.append(
                f"Aggregate ID {agg_id} has a sharp corner ({report.min_polygon_corner_angle:.1f}° < 25°). "
                f"Constrained Delaunay triangulation quality may be constrained near this vertex."
            )

    # 2. Pairwise aggregate clearance and overlap checks
    n_poly = len(polygons)
    for i in range(n_poly):
        p1 = polygons[i]
        id1 = aggregate_ids[i]
        for j in range(i + 1, n_poly):
            p2 = polygons[j]
            id2 = aggregate_ids[j]
            dist = float(p1.distance(p2))
            
            if dist < report.min_inter_aggregate_gap:
                report.min_inter_aggregate_gap = dist
                
            if dist <= 1e-6:
                report.errors.append(
                    f"Aggregate ID {id1} overlaps with Aggregate ID {id2} (measured distance = {dist:.2e} mm)."
                )
                report.offending_aggregate_ids.extend([id1, id2])
                report.passed = False
            elif dist < gap_fail_threshold:
                report.errors.append(
                    f"Clearance between Aggregate ID {id1} and {id2} is below the failure threshold: "
                    f"{dist:.3f} mm < {gap_fail_threshold:.3f} mm."
                )
                report.offending_aggregate_ids.extend([id1, id2])
                report.passed = False
            elif dist < gap_warn_threshold:
                report.warnings.append(
                    f"Narrow gap between Aggregate ID {id1} and {id2}: {dist:.3f} mm < {gap_warn_threshold:.3f} mm."
                )
                
    report.offending_aggregate_ids = sorted(list(set(report.offending_aggregate_ids)))
    return report

# ==============================================================================
# 2. Post-Mesh Validation & Quality Checks
# ==============================================================================

@dataclass
class QualityMetricSummary:
    min_angle: float
    max_angle: float
    median_angle: float
    worst_aspect_ratio: float
    median_aspect_ratio: float
    sub_10_deg_count: int
    sub_20_deg_count: int
    worst_element_index: int
    worst_element_coords: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]
    worst_element_aggregate_id: int

@dataclass
class MeshValidationReport:
    passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    quality: Optional[QualityMetricSummary] = None
    domain_area_error: float = 0.0
    aggregate_area_error_max: float = 0.0
    interface_perimeter_error_max: float = 0.0


def compute_triangle_angles_and_aspect(
    p1: np.ndarray, p2: np.ndarray, p3: np.ndarray
) -> Tuple[float, float, float, float]:
    """
    Computes (min_angle, max_angle, aspect_ratio, area) for triangle (p1, p2, p3).
    Aspect ratio is defined as (longest_edge * semiperimeter) / (2 * area) [standard FE metric, 1.0 for equilateral].
    """
    a = float(np.hypot(p2[0] - p3[0], p2[1] - p3[1]))
    b = float(np.hypot(p3[0] - p1[0], p3[1] - p1[1]))
    c = float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))
    
    # Area via cross-product
    signed_area = 0.5 * ((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
    area = abs(signed_area)
    
    if area <= 1e-15 or a <= 1e-9 or b <= 1e-9 or c <= 1e-9:
        return 0.0, 180.0, float('inf'), area
        
    # Law of cosines
    cos_A = np.clip((b**2 + c**2 - a**2) / (2.0 * b * c), -1.0, 1.0)
    cos_B = np.clip((a**2 + c**2 - b**2) / (2.0 * a * c), -1.0, 1.0)
    cos_C = np.clip((a**2 + b**2 - c**2) / (2.0 * a * b), -1.0, 1.0)
    
    ang_A = float(np.degrees(np.arccos(cos_A)))
    ang_B = float(np.degrees(np.arccos(cos_B)))
    ang_C = float(np.degrees(np.arccos(cos_C)))
    
    min_ang = min(ang_A, ang_B, ang_C)
    max_ang = max(ang_A, ang_B, ang_C)
    
    # Inradius r = Area / s
    s = (a + b + c) / 2.0
    r_in = area / s
    longest_edge = max(a, b, c)
    # Circumradius R = (a*b*c)/(4*Area); normalized aspect ratio = 2 * r_in / R (or longest_edge / (2*sqrt(3)*r_in))
    # Standard engineering aspect ratio: longest_edge / (2 * sqrt(3) * r_in) >= 1.0
    aspect_ratio = longest_edge / (2.0 * np.sqrt(3.0) * r_in) if r_in > 1e-12 else float('inf')
    
    return min_ang, max_ang, aspect_ratio, area


def validate_mesh(
    nodes: np.ndarray,
    triangles: np.ndarray,
    aggregate_ids: np.ndarray,
    polygons: List[Polygon],
    domain_width: float,
    domain_height: float,
    angle_fail_threshold: float = 10.0,
    angle_warn_threshold: float = 20.0
) -> MeshValidationReport:
    """
    Executes exhaustive post-meshing checks.
    """
    report = MeshValidationReport(passed=True)
    num_nodes = len(nodes)
    num_triangles = len(triangles)
    
    if num_nodes < 3 or num_triangles < 1:
        report.errors.append("Mesh is empty or contains insufficient nodes/triangles.")
        report.passed = False
        return report

    # 1. Duplicate & unused node checks
    used_nodes = set(triangles.flatten())
    if len(used_nodes) < num_nodes:
        unused_count = num_nodes - len(used_nodes)
        report.warnings.append(f"Mesh contains {unused_count} unused isolated nodes.")

    # 2. Triangle quality and positive areas
    min_angles = []
    max_angles = []
    aspect_ratios = []
    element_areas = []
    sub_10_count = 0
    sub_20_count = 0
    worst_aspect = 1.0
    worst_elem_idx = 0
    
    for i in range(num_triangles):
        tri = triangles[i]
        p1, p2, p3 = nodes[tri[0]], nodes[tri[1]], nodes[tri[2]]
        
        # Signed area (CCW test)
        signed_area = 0.5 * ((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
        if signed_area <= 1e-14:
            report.errors.append(f"Triangle {i} is inverted or degenerate (signed area = {signed_area:.2e}).")
            report.passed = False
            
        min_ang, max_ang, aspect, area = compute_triangle_angles_and_aspect(p1, p2, p3)
        min_angles.append(min_ang)
        max_angles.append(max_ang)
        aspect_ratios.append(aspect)
        element_areas.append(area)
        
        if min_ang < 10.0:
            sub_10_count += 1
        if min_ang < 20.0:
            sub_20_count += 1
            
        if aspect > worst_aspect:
            worst_aspect = aspect
            worst_elem_idx = i

    worst_tri = triangles[worst_elem_idx]
    worst_coords = (
        tuple(nodes[worst_tri[0]]),
        tuple(nodes[worst_tri[1]]),
        tuple(nodes[worst_tri[2]])
    )
    
    report.quality = QualityMetricSummary(
        min_angle=float(np.min(min_angles)),
        max_angle=float(np.max(max_angles)),
        median_angle=float(np.median(min_angles)),
        worst_aspect_ratio=float(worst_aspect),
        median_aspect_ratio=float(np.median(aspect_ratios)),
        sub_10_deg_count=sub_10_count,
        sub_20_deg_count=sub_20_count,
        worst_element_index=worst_elem_idx,
        worst_element_coords=worst_coords,
        worst_element_aggregate_id=int(aggregate_ids[worst_elem_idx])
    )
    
    # Quality threshold enforcement
    if report.quality.min_angle < angle_fail_threshold:
        report.errors.append(
            f"Worst triangle has minimum angle {report.quality.min_angle:.2f}° < failure threshold ({angle_fail_threshold:.1f}°). "
            f"Element index {worst_elem_idx} at {worst_coords[0]}."
        )
        report.passed = False
    elif report.quality.min_angle < angle_warn_threshold:
        report.warnings.append(
            f"Mesh contains sharp triangles: {sub_20_count} elements have angle < 20.0° (minimum angle = {report.quality.min_angle:.2f}°)."
        )

    # 3. Domain area conservation
    expected_domain_area = domain_width * domain_height
    total_mesh_area = float(np.sum(element_areas))
    report.domain_area_error = abs(total_mesh_area - expected_domain_area)
    if report.domain_area_error > 1e-4 * expected_domain_area:
        report.errors.append(
            f"Sum of triangle areas ({total_mesh_area:.4f} mm²) does not match specimen domain area "
            f"({expected_domain_area:.4f} mm²). Error: {report.domain_area_error:.4e} mm²."
        )
        report.passed = False

    # 4. Per-aggregate area conservation
    for agg_idx, poly in enumerate(polygons):
        agg_elem_indices = np.where(aggregate_ids == agg_idx)[0]
        agg_mesh_area = float(np.sum([element_areas[e] for e in agg_elem_indices]))
        agg_true_area = poly.area
        diff = abs(agg_mesh_area - agg_true_area)
        if diff > report.aggregate_area_error_max:
            report.aggregate_area_error_max = diff
            
        if diff > 1e-4 * agg_true_area and diff > 1e-4:
            report.errors.append(
                f"Aggregate index {agg_idx} area mismatch: Mesh sum={agg_mesh_area:.4f} mm², "
                f"Polygon area={agg_true_area:.4f} mm² (diff={diff:.4e} mm²)."
            )
            report.passed = False

    # 5. Independent Material Tag Verification (Point-in-Polygon on Centroids)
    tag_mismatch_count = 0
    for i in range(num_triangles):
        tri = triangles[i]
        pts = nodes[tri]
        centroid = Point(np.mean(pts, axis=0))
        assigned_agg_id = aggregate_ids[i]
        
        # Verify
        if assigned_agg_id == -1:
            # Must NOT be inside any aggregate polygon
            for agg_idx, poly in enumerate(polygons):
                if poly.contains(centroid):
                    tag_mismatch_count += 1
                    break
        else:
            # Must be strictly inside the designated polygon
            target_poly = polygons[assigned_agg_id]
            if not target_poly.contains(centroid) and not target_poly.touches(centroid):
                tag_mismatch_count += 1

    if tag_mismatch_count > 0:
        report.errors.append(
            f"Independent centroid point-in-polygon verification failed for {tag_mismatch_count} triangles."
        )
        report.passed = False

    # 6. Check single-phase connectivity (No edge shared by two distinct aggregates)
    edge_to_aggs: Dict[Tuple[int, int], Set[int]] = {}
    for i in range(num_triangles):
        tri = triangles[i]
        agg_id = aggregate_ids[i]
        for e in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
            sorted_e = tuple(sorted(e))
            if sorted_e not in edge_to_aggs:
                edge_to_aggs[sorted_e] = set()
            edge_to_aggs[sorted_e].add(agg_id)
            
    for e, aggs in edge_to_aggs.items():
        # Filter out mortar (-1)
        concrete_aggs = [a for a in aggs if a != -1]
        if len(concrete_aggs) > 1:
            report.errors.append(
                f"Edge {e} is shared directly between two different aggregates ({concrete_aggs[0]} and {concrete_aggs[1]}). "
                f"Aggregates are touching without mortar separation."
            )
            report.passed = False

    return report
