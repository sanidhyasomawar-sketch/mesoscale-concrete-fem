"""
core/meshing.py
---------------
Conforming triangular mesh generation, material tagging, and interface topology extraction.

Meshing Engine:
- Uses Jonathan Shewchuk's Triangle via the `triangle` Python package.
- Enforces boundary-conforming Constrained Delaunay Triangulation (CDT).
- Imposes regional area limits for mortar and aggregate phases.
- Validates 100% of material tags independently via centroid point-in-polygon tests.
- Extracts interface edge topology without node duplication (preparation for Phase 5 cohesive ITZ).
"""

import os
import json
from core.geometry import compute_geometry_fingerprint
import numpy as np
import triangle as tr
from shapely.geometry import Polygon, Point, box
from shapely.ops import unary_union
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Set

from core.mesh_checks import (
    check_pre_mesh_geometry, validate_mesh, PreMeshCheckReport, MeshValidationReport
)

# ==============================================================================
# 1. Mesh & Interface Data Structures
# ==============================================================================

@dataclass
class ConcreteMesh:
    nodes: np.ndarray             # (N, 2) float64 nodal coordinates in mm
    triangles: np.ndarray         # (M, 3) int32 vertex indices (counter-clockwise)
    aggregate_id: np.ndarray      # (M,) int32 (-1 = mortar, 0..K-1 = aggregate index)
    phase: np.ndarray             # (M,) int32 (0 = aggregate, 1 = mortar)
    boundary_nodes: Dict[str, np.ndarray] # {'bottom', 'top', 'left', 'right'} node index arrays
    triangle_areas: np.ndarray    # (M,) float64 element areas in mm^2
    domain_width: float           # Width of specimen in mm
    domain_height: float          # Height of specimen in mm
    mesh_params: Dict[str, Any]   # Target sizes, angles, max areas, and limits
    mesher_info: str = "Shewchuk Triangle (via python-triangle)"
    geometry_fingerprint: str = ""
    mesh_fingerprint: str = ""


def compute_mesh_fingerprint(
    nodes: np.ndarray,
    triangles: np.ndarray,
    phase: Optional[np.ndarray] = None,
    num_cohesive_elements: int = 0
) -> str:
    """
    Computes a deterministic SHA-256 fingerprint for a spatial mesh discretization.
    Incorporates node count, element count, cohesive element count, node coordinates,
    triangle connectivity, and phase assignments.
    """
    import hashlib
    hasher = hashlib.sha256()
    hasher.update(f"NODES:{len(nodes)};TRIS:{len(triangles)};ITZ:{num_cohesive_elements};".encode("utf-8"))
    hasher.update(np.ascontiguousarray(np.round(nodes, 4)).tobytes())
    hasher.update(np.ascontiguousarray(triangles).tobytes())
    if phase is not None:
        hasher.update(np.ascontiguousarray(phase).tobytes())
    return hasher.hexdigest()[:16]


@dataclass
class InterfaceData:
    edges: np.ndarray             # (K, 2) int32 node pairs oriented with aggregate on left
    aggregate_id: np.ndarray      # (K,) int32 index of aggregate owning the interface edge
    tri_aggregate: np.ndarray     # (K,) int32 index of adjacent aggregate triangle
    tri_mortar: np.ndarray        # (K,) int32 index of adjacent mortar triangle
    edge_lengths: np.ndarray      # (K,) float64 edge segment lengths in mm
    unit_normals: np.ndarray      # (K, 2) float64 unit normal pointing from aggregate into mortar
    edge_to_triangles: Dict[Tuple[int, int], List[int]] # Mapping undirected edge to triangle indices


# ==============================================================================
# 2. Adjacency and Topology Utilities
# ==============================================================================

def build_edge_to_triangles(triangles: np.ndarray) -> Dict[Tuple[int, int], List[int]]:
    """
    Builds a reusable mapping from undirected edge tuple (min_node, max_node)
    to a list of incident triangle indices.
    """
    edge_to_tris: Dict[Tuple[int, int], List[int]] = {}
    for tri_idx, tri in enumerate(triangles):
        for e in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
            sorted_e = tuple(sorted(e))
            if sorted_e not in edge_to_tris:
                edge_to_tris[sorted_e] = []
            edge_to_tris[sorted_e].append(tri_idx)
    return edge_to_tris


# ==============================================================================
# 3. Conforming Mesh Generation Engine
# ==============================================================================

def target_edge_to_max_area(edge_length: float) -> float:
    """
    Converts a target edge length h (mm) to Shewchuk Triangle's maximum area convention.
    Formula: An equilateral triangle of side h has area A = (sqrt(3) / 4) * h^2 ≈ 0.4330 * h^2.
    """
    return (np.sqrt(3.0) / 4.0) * (edge_length ** 2)


def generate_conforming_mesh(
    domain_width: float,
    domain_height: float,
    polygons: List[Polygon],
    target_edge_mortar: float = 2.0,
    target_edge_aggregate: float = 3.5,
    min_angle: float = 25.0,
    max_triangles: int = 60000,
    run_pre_checks: bool = True,
    gap_fail_threshold: float = 0.15,
    aggregate_ids: Optional[List[int]] = None,
    geometry_fingerprint: Optional[str] = None
) -> Tuple[ConcreteMesh, PreMeshCheckReport]:
    """
    Generates a 2D boundary-conforming Constrained Delaunay Triangulation (CDT)
    using Shewchuk's Triangle.
    
    Parameters:
    -----------
    domain_width, domain_height: float
        Specimen dimensions in mm.
    polygons: List[Polygon]
        List of non-overlapping aggregate polygons.
    target_edge_mortar: float
        Target edge length for the mortar matrix (mm).
    target_edge_aggregate: float
        Target edge length for aggregate interiors (mm).
    min_angle: float
        Requested minimum interior angle (degrees) for quality refinement.
    max_triangles: int
        Safety guard cap to prevent runaway refinement in tight gaps.
    run_pre_checks: bool
        If True, validates geometry before passing to Triangle and fails loudly if invalid.
    """
    if aggregate_ids is None:
        aggregate_ids = list(range(1, len(polygons) + 1))
        
    # 1. Pre-mesh checks
    if run_pre_checks:
        pre_report = check_pre_mesh_geometry(
            polygons, domain_width, domain_height,
            gap_fail_threshold=gap_fail_threshold,
            aggregate_ids=aggregate_ids
        )
        if not pre_report.passed:
            err_msg = "Pre-mesh geometry checks failed:\n" + "\n".join(pre_report.errors)
            raise ValueError(err_msg)
    else:
        pre_report = PreMeshCheckReport(passed=True)

    # 2. Build Planar Straight Line Graph (PSLG)
    all_vertices: List[List[float]] = []
    all_segments: List[List[int]] = []
    
    # Outer specimen domain boundary (4 vertices & 4 segments)
    domain_corners = [
        [0.0, 0.0],
        [domain_width, 0.0],
        [domain_width, domain_height],
        [0.0, domain_height]
    ]
    all_vertices.extend(domain_corners)
    all_segments.extend([[0, 1], [1, 2], [2, 3], [3, 0]])
    
    # Aggregate polygon boundaries
    for poly in polygons:
        coords = np.array(poly.exterior.coords)[:-1]  # drop closing duplicate
        start_v_idx = len(all_vertices)
        n_v = len(coords)
        for pt in coords:
            all_vertices.append([float(pt[0]), float(pt[1])])
        for i in range(n_v):
            all_segments.append([start_v_idx + i, start_v_idx + ((i + 1) % n_v)])
            
    # 3. Regional Attributes and Area Constraints
    # Calculate area limits
    max_area_mortar = target_edge_to_max_area(target_edge_mortar)
    max_area_aggregate = target_edge_to_max_area(target_edge_aggregate)
    
    # Regional seed points: [x, y, regional_attribute, max_area]
    regions: List[List[float]] = []
    
    # Mortar seed point
    domain_box = box(0.0, 0.0, domain_width, domain_height)
    if polygons:
        mortar_poly = domain_box.difference(unary_union(polygons))
        mortar_seed = mortar_poly.representative_point()
    else:
        mortar_seed = Point(domain_width / 2.0, domain_height / 2.0)
        
    regions.append([float(mortar_seed.x), float(mortar_seed.y), -1.0, float(max_area_mortar)])
    
    # Aggregate seed points
    for agg_idx, poly in enumerate(polygons):
        seed_pt = poly.representative_point()
        regions.append([float(seed_pt.x), float(seed_pt.y), float(agg_idx), float(max_area_aggregate)])
        
    pslg_dict = {
        'vertices': np.array(all_vertices, dtype=np.float64),
        'segments': np.array(all_segments, dtype=np.int32),
        'regions': np.array(regions, dtype=np.float64)
    }
    
    # 4. Invoke Shewchuk's Triangle
    # Switches:
    # 'p': Triangulate PSLG
    # 'q{min_angle}': Quality mesh generation
    # 'a': Regional area constraints
    # 'A': Regional attributes
    switches = f"pq{min_angle:.1f}aA"
    mesh_output = tr.triangulate(pslg_dict, switches)
    
    out_nodes = np.array(mesh_output['vertices'], dtype=np.float64)
    out_triangles = np.array(mesh_output['triangles'], dtype=np.int32)
    
    num_triangles = len(out_triangles)
    if num_triangles > max_triangles:
        raise RuntimeError(
            f"Triangulation exceeded maximum allowed element cap: {num_triangles} > {max_triangles}. "
            f"Refinement runaway likely triggered by microscopic geometric clearance."
        )

    # 5. Extract and Validate Triangle Attributes
    # Regional attributes are returned in 'triangle_attributes'
    raw_attrs = mesh_output.get('triangle_attributes', None)
    if raw_attrs is None or len(raw_attrs) != num_triangles:
        raise RuntimeError("Triangle failed to produce regional attributes for material tagging.")
        
    raw_attrs = np.array(raw_attrs).flatten()
    # Regional attribute: -1.0 = mortar, >= 0.0 = aggregate index
    assigned_agg_id = np.round(raw_attrs).astype(np.int32)
    assigned_phase = np.where(assigned_agg_id == -1, 1, 0).astype(np.int32)

    # 6. Ensure Strict Counter-Clockwise (CCW) Orientation & Positive Area
    triangle_areas = np.zeros(num_triangles, dtype=np.float64)
    for i in range(num_triangles):
        tri = out_triangles[i]
        p1, p2, p3 = out_nodes[tri[0]], out_nodes[tri[1]], out_nodes[tri[2]]
        signed_area = 0.5 * ((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
        
        if signed_area < 0:
            # Swap nodes 1 and 2 to restore CCW orientation
            out_triangles[i, [1, 2]] = out_triangles[i, [2, 1]]
            signed_area = -signed_area
            
        triangle_areas[i] = signed_area

    # 7. Independent Material Tag Verification (Point-in-Polygon on Centroids)
    for i in range(num_triangles):
        tri = out_triangles[i]
        centroid = Point(np.mean(out_nodes[tri], axis=0))
        tag = assigned_agg_id[i]
        
        if tag == -1:
            for p_idx, poly in enumerate(polygons):
                if poly.contains(centroid):
                    raise RuntimeError(
                        f"Material tagging mismatch: Triangle {i} centroid is inside Aggregate {p_idx} "
                        f"but tagged as Mortar (-1)."
                    )
        else:
            if not polygons[tag].contains(centroid) and not polygons[tag].touches(centroid):
                raise RuntimeError(
                    f"Material tagging mismatch: Triangle {i} tagged as Aggregate {tag} "
                    f"but centroid lies outside polygon {tag}."
                )

    # 8. Extract Boundary Nodes
    tol = 1e-5
    x = out_nodes[:, 0]
    y = out_nodes[:, 1]
    
    bnd_nodes = {
        'bottom': np.where(np.abs(y - 0.0) <= tol)[0].astype(np.int32),
        'top': np.where(np.abs(y - domain_height) <= tol)[0].astype(np.int32),
        'left': np.where(np.abs(x - 0.0) <= tol)[0].astype(np.int32),
        'right': np.where(np.abs(x - domain_width) <= tol)[0].astype(np.int32)
    }

    mesh_fp = compute_mesh_fingerprint(out_nodes, out_triangles, assigned_phase, 0)
    mesh_params = {
        'target_edge_mortar': target_edge_mortar,
        'target_edge_aggregate': target_edge_aggregate,
        'min_angle': min_angle,
        'max_triangles': max_triangles,
        'max_area_mortar': max_area_mortar,
        'max_area_aggregate': max_area_aggregate,
        'switches': switches,
        'geometry_fingerprint': geometry_fingerprint if geometry_fingerprint else compute_geometry_fingerprint(domain_width, domain_height, polygons),
        'mesh_fingerprint': mesh_fp
    }

    mesh = ConcreteMesh(
        nodes=out_nodes,
        triangles=out_triangles,
        aggregate_id=assigned_agg_id,
        phase=assigned_phase,
        boundary_nodes=bnd_nodes,
        triangle_areas=triangle_areas,
        domain_width=domain_width,
        domain_height=domain_height,
        mesh_params=mesh_params,
        mesher_info="Shewchuk Triangle (via python-triangle 20250106)",
        geometry_fingerprint=geometry_fingerprint if geometry_fingerprint else compute_geometry_fingerprint(domain_width, domain_height, polygons),
        mesh_fingerprint=mesh_fp
    )
    
    return mesh, pre_report

# ==============================================================================
# 4. Interface Topology Extraction (Zero-Thickness Preparation)
# ==============================================================================

def extract_interface_data(
    mesh: ConcreteMesh, 
    polygons: List[Polygon]
) -> InterfaceData:
    """
    Identifies every mesh edge shared by an aggregate triangle and a mortar triangle.
    Orients each edge [n1, n2] so the aggregate triangle lies on the left,
    and computes the outward unit normal pointing into the mortar.
    
    No node duplication is performed at this stage.
    """
    edge_to_tris = build_edge_to_triangles(mesh.triangles)
    
    interface_edges: List[Tuple[int, int]] = []
    interface_agg_ids: List[int] = []
    tri_agg_list: List[int] = []
    tri_mor_list: List[int] = []
    edge_lengths: List[float] = []
    unit_normals: List[Tuple[float, float]] = []

    for sorted_edge, incident_tris in edge_to_tris.items():
        if len(incident_tris) == 2:
            t1, t2 = incident_tris[0], incident_tris[1]
            agg1, agg2 = mesh.aggregate_id[t1], mesh.aggregate_id[t2]
            
            # Check if this edge is on an aggregate-mortar interface
            is_t1_agg = (agg1 != -1)
            is_t2_agg = (agg2 != -1)
            
            if is_t1_agg != is_t2_agg:
                t_agg = t1 if is_t1_agg else t2
                t_mor = t2 if is_t1_agg else t1
                agg_id = agg1 if is_t1_agg else agg2
                
                # Orient edge [n1, n2] so aggregate triangle is on the left
                tri_nodes = list(mesh.triangles[t_agg])
                n_a, n_b = sorted_edge
                
                # Look for directed edge in CCW triangle: n_a -> n_b or n_b -> n_a
                i_a = tri_nodes.index(n_a)
                if tri_nodes[(i_a + 1) % 3] == n_b:
                    n1, n2 = n_a, n_b
                else:
                    n1, n2 = n_b, n_a
                    
                p1 = mesh.nodes[n1]
                p2 = mesh.nodes[n2]
                
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                L = float(np.hypot(dx, dy))
                
                # Outward normal pointing into mortar (to the right of n1 -> n2)
                # Left is (-dy, dx), Right is (+dy, -dx)
                nx = dy / L
                ny = -dx / L
                
                # Verify normal points from aggregate toward mortar centroid
                p_mid = 0.5 * (p1 + p2)
                mortar_centroid = np.mean(mesh.nodes[mesh.triangles[t_mor]], axis=0)
                dot_test = nx * (mortar_centroid[0] - p_mid[0]) + ny * (mortar_centroid[1] - p_mid[1])
                if dot_test < 0:
                    # Invert normal if numerical sign mismatch
                    nx, ny = -nx, -ny
                    
                interface_edges.append((n1, n2))
                interface_agg_ids.append(agg_id)
                tri_agg_list.append(t_agg)
                tri_mor_list.append(t_mor)
                edge_lengths.append(L)
                unit_normals.append((nx, ny))

    return InterfaceData(
        edges=np.array(interface_edges, dtype=np.int32),
        aggregate_id=np.array(interface_agg_ids, dtype=np.int32),
        tri_aggregate=np.array(tri_agg_list, dtype=np.int32),
        tri_mortar=np.array(tri_mor_list, dtype=np.int32),
        edge_lengths=np.array(edge_lengths, dtype=np.float64),
        unit_normals=np.array(unit_normals, dtype=np.float64),
        edge_to_triangles=edge_to_tris
    )

# ==============================================================================
# 5. Serialization & Round-Trip Persistence (.npz)
# ==============================================================================

def save_mesh_npz(
    mesh: ConcreteMesh, 
    filepath: str, 
    interface: Optional[InterfaceData] = None
) -> None:
    """
    Saves ConcreteMesh (and optional InterfaceData) to a compressed NumPy .npz file.
    """
    data_dict: Dict[str, Any] = {
        'nodes': mesh.nodes,
        'triangles': mesh.triangles,
        'aggregate_id': mesh.aggregate_id,
        'phase': mesh.phase,
        'triangle_areas': mesh.triangle_areas,
        'bnd_bottom': mesh.boundary_nodes['bottom'],
        'bnd_top': mesh.boundary_nodes['top'],
        'bnd_left': mesh.boundary_nodes['left'],
        'bnd_right': mesh.boundary_nodes['right'],
        'domain_width': np.array([mesh.domain_width], dtype=np.float64),
        'domain_height': np.array([mesh.domain_height], dtype=np.float64),
        'mesh_params_json': np.array([json.dumps(mesh.mesh_params)])
    }
    
    if interface is not None:
        data_dict['if_edges'] = interface.edges
        data_dict['if_agg_id'] = interface.aggregate_id
        data_dict['if_tri_agg'] = interface.tri_aggregate
        data_dict['if_tri_mor'] = interface.tri_mortar
        data_dict['if_lengths'] = interface.edge_lengths
        data_dict['if_normals'] = interface.unit_normals
        
    np.savez_compressed(filepath, **data_dict)


def load_mesh_npz(filepath: str) -> Tuple[ConcreteMesh, Optional[InterfaceData]]:
    """
    Loads ConcreteMesh and InterfaceData from a compressed NumPy .npz file.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Mesh file not found at {filepath}")
        
    with np.load(filepath, allow_pickle=False) as npz:
        nodes = npz['nodes']
        triangles = npz['triangles']
        aggregate_id = npz['aggregate_id']
        phase = npz['phase']
        triangle_areas = npz['triangle_areas']
        
        bnd_nodes = {
            'bottom': npz['bnd_bottom'],
            'top': npz['bnd_top'],
            'left': npz['bnd_left'],
            'right': npz['bnd_right']
        }
        domain_width = float(npz['domain_width'][0])
        domain_height = float(npz['domain_height'][0])
        mesh_params = json.loads(str(npz['mesh_params_json'][0]))
        
        mesh = ConcreteMesh(
            nodes=nodes,
            triangles=triangles,
            aggregate_id=aggregate_id,
            phase=phase,
            boundary_nodes=bnd_nodes,
            triangle_areas=triangle_areas,
            domain_width=domain_width,
            domain_height=domain_height,
            mesh_params=mesh_params,
            mesher_info="Shewchuk Triangle (restored from .npz)",
            geometry_fingerprint=mesh_params.get('geometry_fingerprint', '')
        )
        
        interface = None
        if 'if_edges' in npz:
            if_edges = npz['if_edges']
            if_agg_id = npz['if_agg_id']
            if_tri_agg = npz['if_tri_agg']
            if_tri_mor = npz['if_tri_mor']
            if_lengths = npz['if_lengths']
            if_normals = npz['if_normals']
            edge_to_tris = build_edge_to_triangles(triangles)
            
            interface = InterfaceData(
                edges=if_edges,
                aggregate_id=if_agg_id,
                tri_aggregate=if_tri_agg,
                tri_mortar=if_tri_mor,
                edge_lengths=if_lengths,
                unit_normals=if_normals,
                edge_to_triangles=edge_to_tris
            )
            
    return mesh, interface
