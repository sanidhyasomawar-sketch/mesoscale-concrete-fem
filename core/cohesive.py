"""
core/cohesive.py
----------------
Zero-thickness cohesive interface zone (ITZ) modeling for 2D mesoscale concrete FEM.

Formulation & Methodology:
- Topology: 4-node zero-thickness cohesive elements along aggregate-mortar interfaces.
- Duplication: Mortar retains original node indices; aggregate side is assigned duplicate nodes.
- Quadrature: 2-point Lobatto (nodal) integration at element vertices to prevent spatial traction oscillations.
- Constitutive Law: Bilinear traction-separation law with Mode I linear softening, irreversible damage history,
  and contact penalty in compression (compression never induces opening damage).
- Solver: Incremental displacement-controlled Newton-Raphson nonlinear solver with adaptive step cutbacks,
  trial state isolation, and exact force-residual equilibrium checks.

Units:
- Geometry: mm
- Bulk moduli and stresses: MPa (N/mm^2)
- Interface stiffness (k_n, k_t): N/mm^3
- Fracture energy (G_f): N/mm (mJ/mm^2)
- Forces: N
"""

import time
import copy
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any

from core.meshing import ConcreteMesh, InterfaceData, compute_mesh_fingerprint
from core.fem import MaterialProperties, cst_element_stiffness
from core.mortar_damage import MortarDamageParameters, BulkMortarState


# ==============================================================================
# 1. ITZ Material Properties & Cohesive State
# ==============================================================================

@dataclass(frozen=True)
class ITZProperties:
    """
    Constitutive parameters for the Interfacial Transition Zone (ITZ) cohesive law.
    
    Attributes:
    -----------
    k_n: float
        Initial normal interface stiffness in N/mm^3 (penalty stiffness before damage).
    k_t: float
        Initial tangential (shear) interface stiffness in N/mm^3.
    f_t: float
        Tensile strength of the ITZ in MPa (N/mm^2).
    G_f: float
        Mode I fracture energy of the ITZ in N/mm.
    name: str
        Informative label.
    """
    k_n: float = 1.0e5      # N/mm^3 (e.g. 100 GPa/mm)
    k_t: float = 0.5e5      # N/mm^3
    f_t: float = 2.5        # MPa (typical ITZ tensile strength is 40-70% of mortar strength)
    G_f: float = 0.05       # N/mm (typical concrete/mortar ITZ fracture energy: 0.02 - 0.08 N/mm)
    name: str = "Bilinear_ITZ"

    def __post_init__(self):
        if self.k_n <= 0.0:
            raise ValueError(f"Normal stiffness k_n must be strictly positive, got {self.k_n} N/mm^3.")
        if self.k_t <= 0.0:
            raise ValueError(f"Tangential stiffness k_t must be strictly positive, got {self.k_t} N/mm^3.")
        if self.f_t <= 0.0:
            raise ValueError(f"Tensile strength f_t must be strictly positive, got {self.f_t} MPa.")
        if self.G_f <= 0.0:
            raise ValueError(f"Fracture energy G_f must be strictly positive, got {self.G_f} N/mm.")
            
        # Physical consistency check: G_f must exceed elastic strain energy to initiation
        # E_elastic = 0.5 * f_t * delta_n0 = 0.5 * f_t^2 / k_n
        elastic_energy = 0.5 * (self.f_t ** 2) / self.k_n
        if self.G_f <= elastic_energy:
            raise ValueError(
                f"Fracture energy G_f ({self.G_f:.4e} N/mm) must exceed initial elastic energy "
                f"({elastic_energy:.4e} N/mm). Increase G_f or increase k_n."
            )

    @property
    def delta_n0(self) -> float:
        """Normal separation at damage initiation in mm: delta_n0 = f_t / k_n."""
        return self.f_t / self.k_n

    @property
    def delta_nc(self) -> float:
        """Critical normal separation at complete decohesion in mm: delta_nc = 2 * G_f / f_t."""
        return 2.0 * self.G_f / self.f_t


@dataclass
class CohesiveState:
    """
    Historical state variables for cohesive interface elements.
    
    Attributes:
    -----------
    delta_n_max: np.ndarray
        (K, 2) maximum normal separation experienced at integration points (node 1 & 2).
    damage: np.ndarray
        (K, 2) current damage variable d in [0, 1] at integration points.
    element_damage: np.ndarray
        (K,) average damage variable for each cohesive element.
    """
    delta_n_max: np.ndarray
    damage: np.ndarray
    element_damage: np.ndarray

    @classmethod
    def initialize(cls, num_elements: int, delta_n0: float) -> 'CohesiveState':
        return cls(
            delta_n_max=np.full((num_elements, 2), delta_n0, dtype=np.float64),
            damage=np.zeros((num_elements, 2), dtype=np.float64),
            element_damage=np.zeros(num_elements, dtype=np.float64)
        )

    def copy(self) -> 'CohesiveState':
        return CohesiveState(
            delta_n_max=self.delta_n_max.copy(),
            damage=self.damage.copy(),
            element_damage=self.element_damage.copy()
        )


# ==============================================================================
# 2. Cohesive Element & Mesh Data Structures
# ==============================================================================

@dataclass
class CohesiveElement:
    """
    4-node zero-thickness cohesive interface element.
    
    Nodes:
    ------
    nodes_agg: Tuple[int, int]
        (n1_agg, n2_agg) nodes on the aggregate side.
    nodes_mor: Tuple[int, int]
        (n1_mor, n2_mor) nodes on the mortar side (coincident with n1_agg, n2_agg).
    length: float
        Length of interface segment in mm.
    tangent: np.ndarray
        (2,) unit vector pointing from node 1 to node 2.
    normal: np.ndarray
        (2,) unit vector pointing from aggregate into mortar.
    midpoint: np.ndarray
        (2,) spatial coordinates of segment center.
    aggregate_id: int
        ID of the aggregate owning this interface segment.
    """
    element_id: int
    nodes_agg: Tuple[int, int]
    nodes_mor: Tuple[int, int]
    length: float
    tangent: np.ndarray
    normal: np.ndarray
    midpoint: np.ndarray
    aggregate_id: int


@dataclass
class CohesiveMesh:
    """
    Mesoscale concrete mesh after node duplication for cohesive interfaces.
    """
    nodes: np.ndarray                     # (N_total, 2) coordinates in mm
    triangles: np.ndarray                 # (M, 3) triangle connectivity with aggregate nodes updated
    phase: np.ndarray                     # (M,) 0 = aggregate, 1 = mortar
    aggregate_id: np.ndarray              # (M,) aggregate ID (-1 = mortar)
    cohesive_elements: List[CohesiveElement] # List of 4-node zero-thickness elements
    boundary_nodes: Dict[str, np.ndarray] # {'bottom', 'top', 'left', 'right'} node sets
    triangle_areas: np.ndarray            # (M,) triangle areas in mm^2
    domain_width: float                   # mm
    domain_height: float                  # mm
    original_node_count: int
    num_duplicated_nodes: int
    node_map: Dict[Tuple[int, int], int]  # (orig_node, agg_id) -> dup_node
    geometry_fingerprint: str = ""
    mesh_fingerprint: str = ""


# ==============================================================================
# 3. Interface Topology & Node Duplication Engine
# ==============================================================================

def build_cohesive_mesh(
    mesh: ConcreteMesh,
    interface: InterfaceData
) -> CohesiveMesh:
    """
    Constructs a cohesive mesh by duplicating nodes along aggregate-mortar interfaces.
    
    Guarantees:
    1. Mortar phase retains all original node indices.
    2. External boundary nodes remain completely unaltered.
    3. Aggregate triangles are re-indexed to use newly duplicated nodes.
    4. 4-node zero-thickness cohesive elements connect coincident node pairs with
       an orthonormal local coordinate system (t = along edge, n = into mortar).
    """
    if len(interface.edges) == 0:
        raise ValueError("Cannot build cohesive mesh: InterfaceData contains zero interface edges.")

    new_nodes_list = list(mesh.nodes.copy())
    orig_count = len(mesh.nodes)
    
    # Map (original_node_index, aggregate_id) -> duplicated_node_index
    node_map: Dict[Tuple[int, int], int] = {}

    for k in range(len(interface.edges)):
        n1, n2 = int(interface.edges[k, 0]), int(interface.edges[k, 1])
        agg_id = int(interface.aggregate_id[k])
        
        for n in (n1, n2):
            key = (n, agg_id)
            if key not in node_map:
                dup_idx = len(new_nodes_list)
                new_nodes_list.append(mesh.nodes[n].copy())
                node_map[key] = dup_idx

    # Duplicate nodes array
    total_nodes = np.array(new_nodes_list, dtype=np.float64)
    num_duplicated = len(total_nodes) - orig_count

    # Update aggregate triangles connectivity
    updated_triangles = mesh.triangles.copy()
    num_triangles = len(updated_triangles)

    for m in range(num_triangles):
        if mesh.phase[m] == 0:  # Aggregate triangle
            agg_id = int(mesh.aggregate_id[m])
            for j in range(3):
                orig_v = int(updated_triangles[m, j])
                key = (orig_v, agg_id)
                if key in node_map:
                    updated_triangles[m, j] = node_map[key]

    # Build 4-node zero-thickness cohesive elements
    cohesive_elements: List[CohesiveElement] = []
    
    for k in range(len(interface.edges)):
        n1_mor = int(interface.edges[k, 0])
        n2_mor = int(interface.edges[k, 1])
        agg_id = int(interface.aggregate_id[k])

        n1_agg = node_map[(n1_mor, agg_id)]
        n2_agg = node_map[(n2_mor, agg_id)]

        p1 = mesh.nodes[n1_mor]
        p2 = mesh.nodes[n2_mor]

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        L = float(np.hypot(dx, dy))

        if L <= 1e-12:
            raise ValueError(f"Zero-length interface edge detected at index {k}: nodes ({n1_mor}, {n2_mor}).")

        tangent = np.array([dx / L, dy / L], dtype=np.float64)
        normal = np.array(interface.unit_normals[k], dtype=np.float64)
        normal = normal / np.linalg.norm(normal)

        # Confirm orthonormality
        dot_tn = float(np.dot(tangent, normal))
        if abs(dot_tn) > 1e-4:
            raise ValueError(f"Interface normal and tangent at edge {k} are not orthogonal: dot = {dot_tn:.3e}")

        midpoint = 0.5 * (p1 + p2)

        elem = CohesiveElement(
            element_id=k,
            nodes_agg=(n1_agg, n2_agg),
            nodes_mor=(n1_mor, n2_mor),
            length=L,
            tangent=tangent,
            normal=normal,
            midpoint=midpoint,
            aggregate_id=agg_id
        )
        cohesive_elements.append(elem)

    return CohesiveMesh(
        nodes=total_nodes,
        triangles=updated_triangles,
        phase=mesh.phase.copy(),
        aggregate_id=mesh.aggregate_id.copy(),
        cohesive_elements=cohesive_elements,
        boundary_nodes=copy.deepcopy(mesh.boundary_nodes),
        triangle_areas=mesh.triangle_areas.copy(),
        domain_width=mesh.domain_width,
        domain_height=mesh.domain_height,
        original_node_count=orig_count,
        num_duplicated_nodes=num_duplicated,
        node_map=node_map,
        geometry_fingerprint=getattr(mesh, 'geometry_fingerprint', ''),
        mesh_fingerprint=compute_mesh_fingerprint(total_nodes, updated_triangles, mesh.phase, len(cohesive_elements))
    )


# ==============================================================================
# 4. Cohesive Constitutive Evaluation
# ==============================================================================

def evaluate_cohesive_point(
    delta_n: float,
    delta_t: float,
    delta_n_max_old: float,
    props: ITZProperties
) -> Tuple[float, float, float, float, float, float]:
    """
    Evaluates the bilinear traction-separation law at a single integration point.
    
    Parameters:
    -----------
    delta_n: float
        Current normal separation (mm). Positive = opening (tension), Negative = contact (compression).
    delta_t: float
        Current tangential separation (mm).
    delta_n_max_old: float
        Historical maximum opening separation from previous converged state.
    props: ITZProperties
        Material properties.
        
    Returns:
    --------
    T_n: float
        Normal traction in MPa.
    T_t: float
        Tangential traction in MPa.
    D_nn: float
        Secant/tangent normal stiffness in N/mm^3.
    D_tt: float
        Secant/tangent shear stiffness in N/mm^3.
    d: float
        Current damage variable in [0, 1].
    delta_n_max_new: float
        Updated maximum normal opening separation in mm.
    """
    delta_n0 = props.delta_n0
    delta_nc = props.delta_nc

    if delta_n <= 0.0:
        # 1. Compressive contact: full penalty stiffness, compression never induces opening damage
        delta_n_max_new = delta_n_max_old
        d = 0.0 if delta_n_max_new <= delta_n0 else (
            (delta_nc * (delta_n_max_new - delta_n0)) / (delta_n_max_new * (delta_nc - delta_n0))
        )
        d = min(1.0, max(0.0, d))
        T_n = props.k_n * delta_n
        T_t = (1.0 - d) * props.k_t * delta_t
        D_nn = props.k_n
        D_tt = (1.0 - d) * props.k_t
        return T_n, T_t, D_nn, D_tt, d, delta_n_max_new

    # 2. Tensile opening: advance historical maximum separation
    delta_n_max_new = max(delta_n_max_old, delta_n)

    if delta_n_max_new <= delta_n0:
        # Elastic tension
        d = 0.0
        T_n = props.k_n * delta_n
        T_t = props.k_t * delta_t
        D_nn = props.k_n
        D_tt = props.k_t
        return T_n, T_t, D_nn, D_tt, d, delta_n_max_new

    if delta_n_max_new >= delta_nc:
        # Complete decohesion (fully opened crack)
        d = 1.0
        T_n = 0.0
        T_t = 0.0
        # Maintain small positive residual stiffness for numerical solvability
        D_nn = 1e-8 * props.k_n
        D_tt = 1e-8 * props.k_t
        return T_n, T_t, D_nn, D_tt, d, delta_n_max_new

    # Softening regime: delta_n0 < delta_n_max_new < delta_nc
    d = (delta_nc * (delta_n_max_new - delta_n0)) / (delta_n_max_new * (delta_nc - delta_n0))
    d = min(1.0, max(0.0, d))

    # Secant unloading/reloading path to origin:
    T_n = (1.0 - d) * props.k_n * delta_n
    T_t = (1.0 - d) * props.k_t * delta_t
    D_nn = max(1e-8 * props.k_n, (1.0 - d) * props.k_n)
    D_tt = max(1e-8 * props.k_t, (1.0 - d) * props.k_t)

    return T_n, T_t, D_nn, D_tt, d, delta_n_max_new


# ==============================================================================
# 5. Cohesive Forces and Stiffness Matrix Assembly
# ==============================================================================

def assemble_cohesive_forces_and_stiffness(
    coh_mesh: CohesiveMesh,
    props: ITZProperties,
    u_full: np.ndarray,
    state_converged: CohesiveState,
    thickness: float = 1.0
) -> Tuple[np.ndarray, sp.csr_matrix, CohesiveState]:
    """
    Computes global cohesive internal forces, tangent stiffness matrix, and trial state.
    Uses 2-point Lobatto (nodal) quadrature.
    """
    total_dofs = 2 * len(coh_mesh.nodes)
    f_coh = np.zeros(total_dofs, dtype=np.float64)

    trial_delta_n_max = state_converged.delta_n_max.copy()
    trial_damage = np.zeros_like(state_converged.damage)

    # 4 nodes per cohesive element => 8 DOFs per element
    # 2 integration points, each creates a 4x4 block (16 entries), total 32 entries per element
    num_elems = len(coh_mesh.cohesive_elements)
    entries_per_elem = 32
    total_entries = num_elems * entries_per_elem

    row_indices = np.zeros(total_entries, dtype=np.int32)
    col_indices = np.zeros(total_entries, dtype=np.int32)
    matrix_values = np.zeros(total_entries, dtype=np.float64)

    entry_ptr = 0

    for elem_idx, elem in enumerate(coh_mesh.cohesive_elements):
        t = elem.tangent
        n = elem.normal
        tx, ty = t[0], t[1]
        nx, ny = n[0], n[1]

        # 2-point Lobatto quadrature: node 1 and node 2 each have weight = L / 2
        A_ip = (elem.length / 2.0) * thickness

        node_pairs = [
            (elem.nodes_agg[0], elem.nodes_mor[0]),
            (elem.nodes_agg[1], elem.nodes_mor[1])
        ]

        for ip_idx, (n_agg, n_mor) in enumerate(node_pairs):
            u_agg_x = u_full[2 * n_agg]
            u_agg_y = u_full[2 * n_agg + 1]
            u_mor_x = u_full[2 * n_mor]
            u_mor_y = u_full[2 * n_mor + 1]

            # Displacement jump: delta = u_mor - u_agg
            jump_x = u_mor_x - u_agg_x
            jump_y = u_mor_y - u_agg_y

            delta_t = tx * jump_x + ty * jump_y
            delta_n = nx * jump_x + ny * jump_y

            delta_n_max_old = state_converged.delta_n_max[elem_idx, ip_idx]
            T_n, T_t, D_nn, D_tt, d, delta_n_max_new = evaluate_cohesive_point(
                delta_n, delta_t, delta_n_max_old, props
            )

            trial_delta_n_max[elem_idx, ip_idx] = delta_n_max_new
            trial_damage[elem_idx, ip_idx] = d

            # Global traction vector
            t_glob_x = tx * T_t + nx * T_n
            t_glob_y = ty * T_t + ny * T_n

            # Internal nodal force vector: equal and opposite on aggregate and mortar
            fx = A_ip * t_glob_x
            fy = A_ip * t_glob_y

            f_coh[2 * n_agg] -= fx
            f_coh[2 * n_agg + 1] -= fy
            f_coh[2 * n_mor] += fx
            f_coh[2 * n_mor + 1] += fy

            # Stiffness block in global coordinates: K_glob = Q @ D @ Q^T
            # Q = [[tx, nx], [ty, ny]]
            # K_glob_00 = D_tt*tx^2 + D_nn*nx^2
            # K_glob_01 = D_tt*tx*ty + D_nn*nx*ny
            # K_glob_11 = D_tt*ty^2 + D_nn*ny^2
            k00 = A_ip * (D_tt * (tx ** 2) + D_nn * (nx ** 2))
            k01 = A_ip * (D_tt * tx * ty + D_nn * nx * ny)
            k11 = A_ip * (D_tt * (ty ** 2) + D_nn * (ny ** 2))

            K_block = np.array([
                [k00, k01],
                [k01, k11]
            ], dtype=np.float64)

            # Node pair DOFs: [dof_agg_x, dof_agg_y, dof_mor_x, dof_mor_y]
            pair_dofs = np.array([2 * n_agg, 2 * n_agg + 1, 2 * n_mor, 2 * n_mor + 1], dtype=np.int32)
            
            # 4x4 matrix:
            # [[+K, -K],
            #  [-K, +K]]
            k4x4 = np.zeros((4, 4), dtype=np.float64)
            k4x4[0:2, 0:2] = K_block
            k4x4[0:2, 2:4] = -K_block
            k4x4[2:4, 0:2] = -K_block
            k4x4[2:4, 2:4] = K_block

            row_indices[entry_ptr:entry_ptr + 16] = np.repeat(pair_dofs, 4)
            col_indices[entry_ptr:entry_ptr + 16] = np.tile(pair_dofs, 4)
            matrix_values[entry_ptr:entry_ptr + 16] = k4x4.flatten()
            entry_ptr += 16

    K_coh = sp.coo_matrix(
        (matrix_values, (row_indices, col_indices)),
        shape=(total_dofs, total_dofs),
        dtype=np.float64
    ).tocsr()

    trial_state = CohesiveState(
        delta_n_max=trial_delta_n_max,
        damage=trial_damage,
        element_damage=np.mean(trial_damage, axis=1)
    )

    return f_coh, K_coh, trial_state


def precompute_bulk_element_data(
    coh_mesh: CohesiveMesh,
    mat_mortar: MaterialProperties,
    mat_aggregate: MaterialProperties,
    thickness: float = 1.0,
    mortar_damage_params: Optional[MortarDamageParameters] = None
) -> Dict[str, Any]:
    """
    Precomputes constant geometric and linear-elastic quantities for all CST bulk elements in CohesiveMesh.
    """
    total_dofs = 2 * len(coh_mesh.nodes)
    num_triangles = len(coh_mesh.triangles)
    entries_per_element = 36
    total_entries = num_triangles * entries_per_element

    D_mortar = mat_mortar.plane_stress_D()
    D_aggregate = mat_aggregate.plane_stress_D()

    all_ke0 = np.zeros((num_triangles, 6, 6), dtype=np.float64)
    all_B = np.zeros((num_triangles, 3, 6), dtype=np.float64)
    all_A = np.zeros(num_triangles, dtype=np.float64)
    all_he = np.zeros(num_triangles, dtype=np.float64)
    all_dofs = np.zeros((num_triangles, 6), dtype=np.int32)

    row_indices = np.zeros(total_entries, dtype=np.int32)
    col_indices = np.zeros(total_entries, dtype=np.int32)

    for e in range(num_triangles):
        tri = coh_mesh.triangles[e]
        phase = coh_mesh.phase[e]
        D_elem = D_aggregate if phase == 0 else D_mortar
        coords = coh_mesh.nodes[tri]
        ke, B, A = cst_element_stiffness(coords, D_elem, thickness)

        all_ke0[e] = ke
        all_B[e] = B
        all_A[e] = A
        all_he[e] = np.sqrt(2.0 * A) if A > 0 else 0.0

        dofs = np.array([
            2 * tri[0], 2 * tri[0] + 1,
            2 * tri[1], 2 * tri[1] + 1,
            2 * tri[2], 2 * tri[2] + 1
        ], dtype=np.int32)
        all_dofs[e] = dofs

        start_idx = e * entries_per_element
        end_idx = start_idx + entries_per_element
        row_indices[start_idx:end_idx] = np.repeat(dofs, 6)
        col_indices[start_idx:end_idx] = np.tile(dofs, 6)

    K_bulk_elastic = sp.coo_matrix(
        (all_ke0.flatten(), (row_indices, col_indices)),
        shape=(total_dofs, total_dofs),
        dtype=np.float64
    ).tocsr()

    mortar_indices = np.where(coh_mesh.phase == 1)[0]
    agg_indices = np.where(coh_mesh.phase == 0)[0]

    eps_0 = 0.0
    eps_u = np.zeros(num_triangles, dtype=np.float64)
    d_max = 1.0
    if mortar_damage_params is not None:
        eps_0 = mortar_damage_params.eps_0
        d_max = mortar_damage_params.d_max
        w_c = mortar_damage_params.w_c
        for idx in mortar_indices:
            he = all_he[idx]
            if he > 0:
                eps_u[idx] = eps_0 + (w_c / he)
            else:
                eps_u[idx] = eps_0 + 1e-2

    return {
        'total_dofs': total_dofs,
        'num_triangles': num_triangles,
        'all_ke0': all_ke0,
        'all_B': all_B,
        'all_A': all_A,
        'all_he': all_he,
        'all_dofs': all_dofs,
        'row_indices': row_indices,
        'col_indices': col_indices,
        'K_bulk_elastic': K_bulk_elastic,
        'mortar_indices': mortar_indices,
        'agg_indices': agg_indices,
        'eps_0': eps_0,
        'eps_u': eps_u,
        'd_max': d_max
    }


def assemble_bulk_stiffness_matrix(
    coh_mesh: CohesiveMesh,
    mat_mortar: MaterialProperties,
    mat_aggregate: MaterialProperties,
    thickness: float = 1.0
) -> sp.csr_matrix:
    """
    Assembles the constant linear-elastic stiffness matrix for all triangular bulk elements.
    Preserved for backward compatibility.
    """
    pre = precompute_bulk_element_data(coh_mesh, mat_mortar, mat_aggregate, thickness)
    return pre['K_bulk_elastic']


def assemble_bulk_forces_and_stiffness(
    coh_mesh: CohesiveMesh,
    mat_mortar: MaterialProperties,
    mat_aggregate: MaterialProperties,
    u_full: np.ndarray,
    mortar_state: Optional[BulkMortarState] = None,
    mortar_damage_params: Optional[MortarDamageParameters] = None,
    thickness: float = 1.0,
    precomputed: Optional[Dict[str, Any]] = None
) -> Tuple[np.ndarray, sp.csr_matrix, Optional[BulkMortarState]]:
    """
    Assembles internal forces and secant tangent stiffness for bulk elements.
    If mortar_damage_params is None or mortar_state is None:
        Returns K_bulk_elastic @ u_full, K_bulk_elastic, None
    If mortar_damage_params is provided:
        Evaluates Rankine equivalent tensile strain on mortar elements (phase == 1),
        updates trial history kappa, evaluates trial damage d_trial,
        scales mortar element stiffness by (1 - d_trial),
        assembles K_bulk, computes f_bulk = K_bulk @ u_full, and returns trial_mortar_state.
        Aggregate elements (phase == 0) remain 100% elastic (damage = 0).
    """
    if precomputed is None:
        precomputed = precompute_bulk_element_data(
            coh_mesh, mat_mortar, mat_aggregate, thickness, mortar_damage_params
        )

    if mortar_damage_params is None or mortar_state is None:
        K_bulk = precomputed['K_bulk_elastic']
        f_bulk = K_bulk @ u_full
        return f_bulk, K_bulk, None

    all_dofs = precomputed['all_dofs']
    all_B = precomputed['all_B']
    all_ke0 = precomputed['all_ke0']
    mortar_indices = precomputed['mortar_indices']
    row_indices = precomputed['row_indices']
    col_indices = precomputed['col_indices']
    total_dofs = precomputed['total_dofs']
    num_triangles = precomputed['num_triangles']
    eps_0 = precomputed['eps_0']
    eps_u = precomputed['eps_u']
    d_max = precomputed['d_max']

    trial_kappa = mortar_state.kappa.copy()
    trial_damage = np.zeros(num_triangles, dtype=np.float64)

    if len(mortar_indices) > 0:
        u_mortar = u_full[all_dofs[mortar_indices]]
        strains = np.einsum('nij,nj->ni', all_B[mortar_indices], u_mortar)

        center = 0.5 * (strains[:, 0] + strains[:, 1])
        radius = np.sqrt((0.5 * (strains[:, 0] - strains[:, 1])) ** 2 + (0.5 * strains[:, 2]) ** 2)
        eps_1 = center + radius
        tilde_eps = np.maximum(0.0, eps_1)

        k_old = mortar_state.kappa[mortar_indices]
        k_trial_m = np.maximum(k_old, tilde_eps)
        trial_kappa[mortar_indices] = k_trial_m

        active = k_trial_m > eps_0
        d_m = np.zeros(len(mortar_indices), dtype=np.float64)
        if np.any(active):
            eps_u_act = eps_u[mortar_indices][active]
            k_act = k_trial_m[active]
            d_ideal = (eps_u_act * (k_act - eps_0)) / (k_act * (eps_u_act - eps_0))
            d_m[active] = np.minimum(d_ideal, d_max)

        trial_damage[mortar_indices] = d_m

    trial_is_damaged = (trial_damage > 1e-4)
    trial_mortar_state = BulkMortarState(
        kappa=trial_kappa,
        damage=trial_damage,
        is_damaged=trial_is_damaged
    )

    factors = 1.0 - trial_damage
    scaled_ke = factors[:, None, None] * all_ke0

    K_bulk = sp.coo_matrix(
        (scaled_ke.ravel(), (row_indices, col_indices)),
        shape=(total_dofs, total_dofs),
        dtype=np.float64
    ).tocsr()

    f_bulk = K_bulk @ u_full
    return f_bulk, K_bulk, trial_mortar_state


# ==============================================================================
# 6. Nonlinear Incremental-Iterative Newton-Raphson Solver
# ==============================================================================

@dataclass
class NonlinearStepResult:
    """
    Results of a single converged load increment.
    """
    step_index: int
    load_factor: float
    displacement_top: float
    reaction_force_top: float
    reaction_force_bottom: float
    nominal_compressive_stress: float
    nominal_vertical_strain: float
    num_iterations: int
    equilibrium_residual: float
    max_damage: float
    mean_damage: float
    damaged_element_count: int
    convergence_method: str = "Force tolerance"
    max_mortar_damage: float = 0.0
    mean_mortar_damage: float = 0.0
    damaged_mortar_count: int = 0
    displacement_correction_norm: float = 0.0


@dataclass
class NonlinearFEMResult:
    """
    Overall result of the incremental nonlinear cohesive FEM analysis.
    """
    converged: bool
    status_message: str
    steps: List[NonlinearStepResult]
    final_displacements: np.ndarray
    final_state: CohesiveState
    total_time_seconds: float
    displacement_curve: np.ndarray
    reaction_curve: np.ndarray
    stress_curve: np.ndarray
    strain_curve: np.ndarray
    geometry_fingerprint: str = ""
    mesh_fingerprint: str = ""
    last_converged_load_factor: float = 0.0
    actual_displacement_reached: float = 0.0
    total_cutbacks: int = 0
    total_iterations: int = 0
    termination_reason: str = ""
    force_tol: float = 1e-3
    disp_tol: float = 1e-6
    relaxed_force_tol: float = 1e-2
    final_mortar_state: Optional[BulkMortarState] = None
    max_mortar_damage: float = 0.0
    damaged_mortar_count: int = 0
    mortar_damage_curve: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))


def solve_cohesive_fem(
    coh_mesh: CohesiveMesh,
    mat_mortar: MaterialProperties,
    mat_aggregate: MaterialProperties,
    itz_props: ITZProperties,
    top_displacement: float = -0.05,
    num_steps: int = 20,
    thickness: float = 1.0,
    platen_friction: bool = False,
    max_iter_per_step: int = 40,
    force_tol: float = 1e-3,
    disp_tol: float = 1e-6,
    max_cutbacks: int = 8,
    mortar_damage_params: Optional[MortarDamageParameters] = None
) -> NonlinearFEMResult:
    """
    Executes an incremental displacement-controlled Newton-Raphson solver for the cohesive mesoscale concrete model,
    with optional isotropic tensile damage in bulk mortar elements.
    """
    # Input validation
    if top_displacement >= 0.0:
        raise ValueError(
            f"Prescribed vertical displacement must be strictly negative for uniaxial compressive loading, "
            f"got {top_displacement} mm."
        )
    if abs(top_displacement) >= coh_mesh.domain_height:
        raise ValueError(
            f"Prescribed vertical displacement magnitude ({abs(top_displacement):.4f} mm) cannot equal or exceed "
            f"specimen height ({coh_mesh.domain_height:.1f} mm)."
        )
    if num_steps < 1:
        raise ValueError(f"Number of load increments num_steps must be at least 1, got {num_steps}.")
    if thickness <= 0.0:
        raise ValueError(f"Specimen thickness must be strictly positive, got {thickness} mm.")
    if max_iter_per_step < 1:
        raise ValueError(f"Maximum iterations per step max_iter_per_step must be at least 1, got {max_iter_per_step}.")
    if max_cutbacks < 0:
        raise ValueError(f"Maximum cutbacks max_cutbacks must be non-negative, got {max_cutbacks}.")
    if force_tol <= 0.0:
        raise ValueError(f"Force tolerance force_tol must be strictly positive, got {force_tol}.")
    if disp_tol <= 0.0:
        raise ValueError(f"Displacement tolerance disp_tol must be strictly positive, got {disp_tol}.")

    t_start = time.perf_counter()

    total_dofs = 2 * len(coh_mesh.nodes)
    num_elems = len(coh_mesh.cohesive_elements)
    num_triangles = len(coh_mesh.triangles)

    # 1. Precompute bulk element data
    precomputed_bulk = precompute_bulk_element_data(
        coh_mesh, mat_mortar, mat_aggregate, thickness, mortar_damage_params
    )

    # 2. Set up boundary DOFs
    bottom_nodes = coh_mesh.boundary_nodes['bottom']
    top_nodes = coh_mesh.boundary_nodes['top']

    prescribed_dofs_map: Dict[int, float] = {}

    # Bottom support: u_y = 0
    for n in bottom_nodes:
        prescribed_dofs_map[int(2 * n + 1)] = 0.0

    if platen_friction:
        for n in bottom_nodes:
            prescribed_dofs_map[int(2 * n)] = 0.0
        for n in top_nodes:
            prescribed_dofs_map[int(2 * n)] = 0.0
    else:
        # Single node pinned in X to eliminate rigid-body translation
        mid_x = coh_mesh.domain_width / 2.0
        bot_coords_x = coh_mesh.nodes[bottom_nodes, 0]
        center_idx = int(np.argmin(np.abs(bot_coords_x - mid_x)))
        center_node = int(bottom_nodes[center_idx])
        prescribed_dofs_map[int(2 * center_node)] = 0.0

    # Top boundary: unit compressive displacement profile
    top_dof_indices = [int(2 * n + 1) for n in top_nodes]
    for dof in top_dof_indices:
        prescribed_dofs_map[dof] = float(top_displacement)

    prescribed_indices = np.array(sorted(prescribed_dofs_map.keys()), dtype=np.int32)
    prescribed_unit_vals = np.array([prescribed_dofs_map[d] for d in prescribed_indices], dtype=np.float64)

    is_prescribed = np.zeros(total_dofs, dtype=bool)
    is_prescribed[prescribed_indices] = True
    free_indices = np.where(~is_prescribed)[0]

    # Initialize cohesive state
    u_full = np.zeros(total_dofs, dtype=np.float64)
    state = CohesiveState.initialize(num_elems, itz_props.delta_n0)

    # Initialize bulk mortar state if enabled
    mortar_state: Optional[BulkMortarState] = None
    if mortar_damage_params is not None:
        mortar_state = BulkMortarState.initialize(
            num_triangles, coh_mesh.phase, mortar_damage_params.eps_0
        )

    step_results: List[NonlinearStepResult] = []

    current_lambda = 0.0
    d_lambda_initial = 1.0 / num_steps
    d_lambda = d_lambda_initial
    step_attempt = 0
    consecutive_cutbacks = 0
    total_cutbacks = 0
    total_iterations = 0
    steps_since_cutback = 0
    prev_max_damage = 0.0
    prev_dmg_count = 0
    prev_max_mortar_damage = 0.0
    prev_mortar_count = 0

    W = coh_mesh.domain_width
    H = coh_mesh.domain_height
    load_area = W * thickness

    while current_lambda < 1.0 - 1e-9:
        step_attempt += 1
        target_lambda = min(1.0, current_lambda + d_lambda)

        # Store backup for cutback restoration
        u_backup = u_full.copy()
        state_backup = state.copy()
        mortar_state_backup = mortar_state.copy() if mortar_state is not None else None

        converged = False
        iteration = 0
        eq_residual_val = 0.0
        trial_state = state.copy()
        trial_mortar_state = mortar_state.copy() if mortar_state is not None else None
        conv_method = ""
        disp_norm_val = 0.0

        # Newton-Raphson equilibrium iterations
        while iteration < max_iter_per_step:
            iteration += 1
            total_iterations += 1

            # Update prescribed displacement DOFs
            u_full[prescribed_indices] = target_lambda * prescribed_unit_vals

            # Assemble bulk internal forces and tangent stiffness
            f_bulk, K_bulk, trial_mortar_state = assemble_bulk_forces_and_stiffness(
                coh_mesh, mat_mortar, mat_aggregate, u_full,
                mortar_state, mortar_damage_params, thickness, precomputed_bulk
            )

            # Assemble cohesive internal forces and tangent stiffness
            f_coh, K_coh, trial_state = assemble_cohesive_forces_and_stiffness(
                coh_mesh, itz_props, u_full, state, thickness
            )

            # Total internal forces: f_int = f_bulk + f_coh
            f_int = f_bulk + f_coh

            # Residual on free DOFs: r_f = -f_int[free] (f_ext = 0 on free DOFs)
            r_free = -f_int[free_indices]
            residual_norm = float(np.linalg.norm(r_free))
            eq_residual_val = residual_norm

            # Convergence test: residual norm check
            if iteration > 1 and residual_norm < force_tol:
                converged = True
                conv_method = "Force tolerance"
                break

            # Total tangent stiffness
            K_total = K_bulk + K_coh
            K_ff = K_total[free_indices, :][:, free_indices]

            try:
                du_free = spla.spsolve(K_ff.tocsc(), r_free)
            except Exception:
                converged = False
                break

            if np.any(np.isnan(du_free)) or np.any(np.isinf(du_free)):
                converged = False
                break

            u_full[free_indices] += du_free
            disp_norm = float(np.linalg.norm(du_free))
            disp_norm_val = disp_norm

            if disp_norm < disp_tol and residual_norm < 10.0 * force_tol:
                converged = True
                conv_method = "Displacement correction + relaxed force tolerance"
                break

        if converged:
            # Re-evaluate internal forces, cohesive state, mortar state, and residual at final converged displacements
            f_bulk, K_bulk, trial_mortar_state = assemble_bulk_forces_and_stiffness(
                coh_mesh, mat_mortar, mat_aggregate, u_full,
                mortar_state, mortar_damage_params, thickness, precomputed_bulk
            )
            f_coh, K_coh, trial_state = assemble_cohesive_forces_and_stiffness(
                coh_mesh, itz_props, u_full, state, thickness
            )
            f_int = f_bulk + f_coh
            r_free = -f_int[free_indices]
            eq_residual_val = float(np.linalg.norm(r_free))

            # Commit trial states upon confirmed equilibrium convergence
            current_lambda = target_lambda
            state = trial_state.copy()
            if mortar_state is not None and trial_mortar_state is not None:
                mortar_state = trial_mortar_state.copy()

            consecutive_cutbacks = 0
            steps_since_cutback += 1

            # Compute reactions: R = f_int
            reactions = f_int.reshape((-1, 2))
            r_top = float(np.sum(reactions[top_nodes, 1]))
            r_bottom = float(np.sum(reactions[bottom_nodes, 1]))

            nominal_stress = float(abs(r_top) / load_area)
            disp_applied = float(current_lambda * abs(top_displacement))
            nominal_strain = float(disp_applied / H) if H > 0 else 0.0

            d_max = float(np.max(state.damage))
            d_mean = float(np.mean(state.damage))
            d_count = int(np.sum(state.element_damage > 1e-4))

            d_mortar_max = float(np.max(mortar_state.damage)) if mortar_state is not None else 0.0
            d_mortar_mean = (
                float(np.mean(mortar_state.damage[coh_mesh.phase == 1]))
                if mortar_state is not None and np.any(coh_mesh.phase == 1)
                else 0.0
            )
            d_mortar_count = int(np.sum(mortar_state.is_damaged)) if mortar_state is not None else 0

            step_res = NonlinearStepResult(
                step_index=len(step_results) + 1,
                load_factor=current_lambda,
                displacement_top=float(current_lambda * top_displacement),
                reaction_force_top=r_top,
                reaction_force_bottom=r_bottom,
                nominal_compressive_stress=nominal_stress,
                nominal_vertical_strain=nominal_strain,
                num_iterations=iteration,
                equilibrium_residual=eq_residual_val,
                max_damage=d_max,
                mean_damage=d_mean,
                damaged_element_count=d_count,
                convergence_method=conv_method,
                max_mortar_damage=d_mortar_max,
                mean_mortar_damage=d_mortar_mean,
                damaged_mortar_count=d_mortar_count,
                displacement_correction_norm=disp_norm_val
            )
            step_results.append(step_res)

            # Conservative adaptive step-size recovery
            if d_lambda < d_lambda_initial - 1e-12:
                if steps_since_cutback >= 2 and iteration <= 4:
                    delta_dam = d_max - prev_max_damage
                    delta_count = d_count - prev_dmg_count
                    delta_m_dam = d_mortar_max - prev_max_mortar_damage
                    delta_m_count = d_mortar_count - prev_mortar_count
                    if delta_dam < 0.02 and delta_count <= 1 and delta_m_dam < 0.02 and delta_m_count <= 1:
                        d_lambda = min(d_lambda * 1.25, d_lambda_initial)

            prev_max_damage = d_max
            prev_dmg_count = d_count
            prev_max_mortar_damage = d_mortar_max
            prev_mortar_count = d_mortar_count

        else:
            # Cutback required
            consecutive_cutbacks += 1
            total_cutbacks += 1
            steps_since_cutback = 0
            if consecutive_cutbacks > max_cutbacks:
                break

            # Halve step size and restore previous converged state
            d_lambda = d_lambda / 2.0
            u_full = u_backup.copy()
            state = state_backup.copy()
            if mortar_state is not None:
                mortar_state = mortar_state_backup.copy()

    total_time = time.perf_counter() - t_start
    overall_converged = (current_lambda >= 1.0 - 1e-5)

    if overall_converged:
        final_status = "Successfully converged all increments."
        termination_reason = "Completed all prescribed increments to 100% load."
    else:
        final_status = f"Terminated early at load factor {current_lambda:.3f}: cutback limit reached."
        termination_reason = (
            f"Cutback limit reached: exceeded {max_cutbacks} consecutive non-convergent cutbacks "
            f"at load factor λ = {current_lambda:.4f}."
        )

    actual_disp = float(current_lambda * top_displacement)

    disp_curve = np.array([abs(s.displacement_top) for s in step_results], dtype=np.float64)
    reaction_curve = np.array([abs(s.reaction_force_top) for s in step_results], dtype=np.float64)
    stress_curve = np.array([s.nominal_compressive_stress for s in step_results], dtype=np.float64)
    strain_curve = np.array([s.nominal_vertical_strain for s in step_results], dtype=np.float64)

    final_mortar_dmg_max = float(np.max(mortar_state.damage)) if mortar_state is not None else 0.0
    final_mortar_dmg_count = int(np.sum(mortar_state.is_damaged)) if mortar_state is not None else 0
    mortar_dmg_curve = (
        np.array([s.max_mortar_damage for s in step_results], dtype=np.float64)
        if mortar_state is not None
        else np.zeros(0, dtype=np.float64)
    )

    relaxed_force_tol = 10.0 * force_tol

    return NonlinearFEMResult(
        converged=overall_converged,
        status_message=final_status,
        steps=step_results,
        final_displacements=u_full.reshape((-1, 2)),
        final_state=state,
        total_time_seconds=total_time,
        displacement_curve=disp_curve,
        reaction_curve=reaction_curve,
        stress_curve=stress_curve,
        strain_curve=strain_curve,
        geometry_fingerprint=getattr(coh_mesh, 'geometry_fingerprint', ''),
        mesh_fingerprint=getattr(coh_mesh, 'mesh_fingerprint', ''),
        last_converged_load_factor=float(current_lambda),
        actual_displacement_reached=actual_disp,
        total_cutbacks=total_cutbacks,
        total_iterations=total_iterations,
        termination_reason=termination_reason,
        force_tol=force_tol,
        disp_tol=disp_tol,
        relaxed_force_tol=relaxed_force_tol,
        final_mortar_state=mortar_state,
        max_mortar_damage=final_mortar_dmg_max,
        damaged_mortar_count=final_mortar_dmg_count,
        mortar_damage_curve=mortar_dmg_curve
    )
