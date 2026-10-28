"""
core/fem.py
-----------
Linear-elastic 2D plane-stress Finite Element Method (FEM) solver using
three-node constant-strain triangular (CST) elements.

Formulation & Methodology:
- Element: 3-node linear displacement Constant Strain Triangle (CST).
- Kinematics: Small deformation infinitesimal plane-stress (sigma_zz = tau_xz = tau_yz = 0).
- Constitutive: Isotropic linear elasticity with distinct phases for aggregate and mortar.
- Units: Geometry in mm, Moduli (E) and Stresses (sigma) in MPa (N/mm^2), Forces in N, Stiffness in N/mm.
- Assembly: Sparse COO matrix assembled into CSR format for memory-efficient scaling.
- Boundary Conditions: Exact matrix partitioning for prescribed displacement DOFs.
- Reactions & Equilibrium: Full nodal force recovery via R = K @ u with residual equilibrium checks.

SCIENTIFIC DISCLAIMER:
This linear-elastic CST formulation captures initial pre-peak elastic stiffness and stress concentrations.
It does NOT represent nonlinear softening, concrete crushing, micro-crack coalescence, or cohesive ITZ fracture.
"""

import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, Any, List

from core.meshing import ConcreteMesh


# ==============================================================================
# 1. Material Model & Boundary Conditions
# ==============================================================================

@dataclass(frozen=True)
class MaterialProperties:
    """
    Isotropic linear-elastic material properties.
    
    Attributes:
    -----------
    E: float
        Young's modulus in MPa (N/mm^2). Must be positive.
    nu: float
        Poisson's ratio (dimensionless). Must satisfy 0.0 <= nu < 0.5 for stability.
    name: str
        Informative label (e.g. 'Aggregate', 'Mortar').
    """
    E: float
    nu: float
    name: str = "LinearElasticMaterial"

    def __post_init__(self):
        if self.E <= 0.0:
            raise ValueError(f"Young's modulus E must be strictly positive, got {self.E} MPa.")
        if not (-0.99 < self.nu < 0.5):
            raise ValueError(f"Poisson's ratio nu must satisfy -0.99 < nu < 0.5, got {self.nu}.")

    def plane_stress_D(self) -> np.ndarray:
        """
        Computes the 3x3 plane-stress constitutive matrix D relating:
        [sigma_xx, sigma_yy, tau_xy]^T = D @ [epsilon_xx, epsilon_yy, gamma_xy]^T
        Units: MPa.
        """
        factor = self.E / (1.0 - self.nu ** 2)
        D = factor * np.array([
            [1.0,     self.nu, 0.0],
            [self.nu, 1.0,     0.0],
            [0.0,     0.0,     0.5 * (1.0 - self.nu)]
        ], dtype=np.float64)
        return D


@dataclass
class BoundaryConditions:
    """
    Essential (Dirichlet) displacement boundary conditions.
    
    Attributes:
    -----------
    prescribed_dofs: Dict[int, float]
        Mapping from global DOF index (2*node + dir) to prescribed displacement (mm).
        dir = 0 corresponds to u_x, dir = 1 corresponds to u_y.
    description: str
        Summary of the boundary condition setup.
    """
    prescribed_dofs: Dict[int, float] = field(default_factory=dict)
    description: str = "Custom Boundary Conditions"

    @classmethod
    def for_uniaxial_compression(
        cls,
        mesh: ConcreteMesh,
        top_displacement: float = -0.05,
        platen_friction: bool = False
    ) -> 'BoundaryConditions':
        """
        Sets up boundary conditions for displacement-controlled uniaxial compression.
        
        Boundary configuration:
        - Bottom boundary (y = 0):
          * u_y = 0 for all bottom nodes (rigid flat bottom platen).
          * If platen_friction is False (default): u_x = 0 is applied to ONLY the single node
            closest to x = W / 2 on the bottom edge. This eliminates rigid-body translation in X
            without restraining lateral Poisson expansion across the specimen.
          * If platen_friction is True: u_x = 0 is applied to ALL bottom nodes (rough/sticking platen).
        - Top boundary (y = H):
          * u_y = top_displacement for all top nodes (compressive displacement, typically negative).
          * If platen_friction is False: u_x is left free (frictionless top platen).
          * If platen_friction is True: u_x = 0 is applied to ALL top nodes (sticking top platen).
        
        Parameters:
        -----------
        mesh: ConcreteMesh
            The conforming mesh containing 'bottom' and 'top' boundary node arrays.
        top_displacement: float
            Imposed vertical displacement on the top boundary in mm (negative for compression).
        platen_friction: bool
            If True, restrains lateral deformation (u_x = 0) on top and bottom platens.
            If False (default), platens are frictionless to allow pure uniaxial stress state.
        """
        prescribed: Dict[int, float] = {}

        bottom_nodes = mesh.boundary_nodes['bottom']
        top_nodes = mesh.boundary_nodes['top']

        if len(bottom_nodes) == 0:
            raise ValueError("Mesh has no nodes tagged on the 'bottom' boundary.")
        if len(top_nodes) == 0:
            raise ValueError("Mesh has no nodes tagged on the 'top' boundary.")

        # 1. Bottom boundary: support in Y
        for n in bottom_nodes:
            prescribed[int(2 * n + 1)] = 0.0

        if platen_friction:
            for n in bottom_nodes:
                prescribed[int(2 * n)] = 0.0
            for n in top_nodes:
                prescribed[int(2 * n)] = 0.0
        else:
            # Frictionless platen: Pin exactly ONE node in X to eliminate rigid-body translation
            mid_x = mesh.domain_width / 2.0
            bot_coords_x = mesh.nodes[bottom_nodes, 0]
            center_idx = int(np.argmin(np.abs(bot_coords_x - mid_x)))
            center_node = int(bottom_nodes[center_idx])
            prescribed[int(2 * center_node)] = 0.0

        # 2. Top boundary: imposed vertical displacement in Y
        for n in top_nodes:
            prescribed[int(2 * n + 1)] = float(top_displacement)

        desc = (
            f"Uniaxial Compression (Delta_y = {top_displacement:.4f} mm, "
            f"Platens = {'Rough/Sticking' if platen_friction else 'Frictionless'})"
        )
        return cls(prescribed_dofs=prescribed, description=desc)


# ==============================================================================
# 2. CST Element Formulations
# ==============================================================================

def cst_B_matrix(coords: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Computes the 3x6 strain-displacement matrix B and area A for a 3-node triangle.
    
    Parameters:
    -----------
    coords: np.ndarray
        (3, 2) array of nodal coordinates [(x1, y1), (x2, y2), (x3, y3)] in mm.
        Must be oriented counter-clockwise (CCW).
        
    Returns:
    --------
    B: np.ndarray
        (3, 6) strain-displacement matrix relating nodal displacements to
        infinitesimal engineering strains: [eps_xx, eps_yy, gamma_xy]^T = B @ u_e.
    A: float
        Positive element area in mm^2.
    """
    x1, y1 = coords[0]
    x2, y2 = coords[1]
    x3, y3 = coords[2]

    # Signed area: A = 0.5 * ((x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1))
    twice_area = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    if twice_area <= 1e-15:
        # Check if clockwise; if so, invert orientation
        if twice_area < -1e-15:
            # Clockwise detected: swap nodes 1 and 2
            x2, y2, x3, y3 = x3, y3, x2, y2
            twice_area = -twice_area
        else:
            raise ValueError(f"Degenerate or collinear triangular element with area ≈ {twice_area / 2.0:.3e} mm^2.")

    A = 0.5 * twice_area

    # Shape function derivatives: N_i = (alpha_i + beta_i * x + gamma_i * y) / (2A)
    # beta_1 = y2 - y3, gamma_1 = x3 - x2
    # beta_2 = y3 - y1, gamma_2 = x1 - x3
    # beta_3 = y1 - y2, gamma_3 = x2 - x1
    b1 = y2 - y3
    b2 = y3 - y1
    b3 = y1 - y2

    c1 = x3 - x2
    c2 = x1 - x3
    c3 = x2 - x1

    inv_2A = 1.0 / twice_area
    B = inv_2A * np.array([
        [b1, 0.0, b2, 0.0, b3, 0.0],
        [0.0, c1, 0.0, c2, 0.0, c3],
        [c1, b1, c2, b2, c3, b3]
    ], dtype=np.float64)

    return B, A


def cst_element_stiffness(
    coords: np.ndarray,
    D: np.ndarray,
    thickness: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Computes the 6x6 element stiffness matrix for a 3-node CST element:
    k_e = thickness * A * (B^T @ D @ B).
    
    Parameters:
    -----------
    coords: np.ndarray
        (3, 2) nodal coordinates in mm.
    D: np.ndarray
        (3, 3) plane-stress constitutive matrix in MPa.
    thickness: float
        Specimen out-of-plane thickness in mm (default = 1.0 mm).
        
    Returns:
    --------
    k_e: np.ndarray
        (6, 6) symmetric element stiffness matrix in N/mm.
    B: np.ndarray
        (3, 6) strain-displacement matrix.
    A: float
        Element area in mm^2.
    """
    B, A = cst_B_matrix(coords)
    # Exact integration for constant strain and constant D:
    k_e = thickness * A * (B.T @ D @ B)
    # Enforce exact symmetry
    k_e = 0.5 * (k_e + k_e.T)
    return k_e, B, A


# ==============================================================================
# 3. Solver Output Data Structure
# ==============================================================================

@dataclass
class FEMResult:
    """
    Comprehensive output of a 2D linear-elastic FEM solution.
    
    Fields:
    -------
    displacements: np.ndarray
        (N, 2) nodal displacements [u_x, u_y] in mm.
    strains: np.ndarray
        (M, 3) element engineering strains [epsilon_xx, epsilon_yy, gamma_xy].
    stresses: np.ndarray
        (M, 3) element Cauchy stresses [sigma_xx, sigma_yy, tau_xy] in MPa.
    von_mises: np.ndarray
        (M,) element von Mises equivalent stresses in MPa.
    principal_stresses: np.ndarray
        (M, 2) element principal stresses [sigma_1, sigma_2] with sigma_1 >= sigma_2 in MPa.
    reactions: np.ndarray
        (N, 2) nodal reaction forces [R_x, R_y] in N.
    total_reaction_top: float
        Net vertical reaction on top boundary in N (R_y_top).
    total_reaction_bottom: float
        Net vertical reaction on bottom boundary in N (R_y_bottom).
    nominal_compressive_stress: float
        Nominal macro-stress |R_y| / (W * t) in MPa.
    nominal_vertical_strain: float
        Nominal macro-strain |Delta_y| / H (dimensionless).
    effective_modulus: float
        Effective secant Young's modulus E_eff = nominal_stress / nominal_strain in MPa.
    equilibrium_residual_y: float
        Absolute vertical equilibrium error |R_y_top + R_y_bottom| in N.
    equilibrium_residual_x: float
        Absolute horizontal equilibrium error |sum R_x| in N.
    solve_time_seconds: float
        Time taken for matrix assembly and sparse solution in seconds.
    total_dofs: int
        Total degrees of freedom (2 * N).
    thickness: float
        Model thickness in mm.
    """
    displacements: np.ndarray
    strains: np.ndarray
    stresses: np.ndarray
    von_mises: np.ndarray
    principal_stresses: np.ndarray
    reactions: np.ndarray
    total_reaction_top: float
    total_reaction_bottom: float
    nominal_compressive_stress: float
    nominal_vertical_strain: float
    effective_modulus: float
    equilibrium_residual_y: float
    equilibrium_residual_x: float
    solve_time_seconds: float
    total_dofs: int
    thickness: float
    geometry_fingerprint: str = ""


# ==============================================================================
# 4. Global Assembly & Sparse Direct Solver
# ==============================================================================

def solve_fem(
    mesh: ConcreteMesh,
    mat_mortar: MaterialProperties,
    mat_aggregate: MaterialProperties,
    bc: BoundaryConditions,
    thickness: float = 1.0
) -> FEMResult:
    """
    Assembles and solves the linear-elastic 2D plane-stress equilibrium equations
    using three-node constant-strain triangular (CST) elements.
    
    Parameters:
    -----------
    mesh: ConcreteMesh
        The conforming triangular mesh from Phase 3.
    mat_mortar: MaterialProperties
        Linear-elastic properties for the mortar phase (mesh.phase == 1).
    mat_aggregate: MaterialProperties
        Linear-elastic properties for the aggregate phase (mesh.phase == 0).
    bc: BoundaryConditions
        Essential displacement boundary conditions.
    thickness: float
        Specimen out-of-plane thickness in mm (default = 1.0 mm).
        
    Returns:
    --------
    FEMResult:
        Nodal displacements, element stresses, reactions, and homogenized macroscopic metrics.
    """
    t0 = time.perf_counter()

    num_nodes = len(mesh.nodes)
    num_triangles = len(mesh.triangles)
    total_dofs = 2 * num_nodes

    if num_nodes < 3 or num_triangles < 1:
        raise ValueError("Invalid mesh: fewer than 3 nodes or 1 triangle.")
    if thickness <= 0.0:
        raise ValueError(f"Specimen thickness must be strictly positive, got {thickness} mm.")

    D_mortar = mat_mortar.plane_stress_D()
    D_aggregate = mat_aggregate.plane_stress_D()

    # Preallocate COO assembly arrays
    entries_per_element = 36 # 6 x 6
    total_entries = num_triangles * entries_per_element

    row_indices = np.zeros(total_entries, dtype=np.int32)
    col_indices = np.zeros(total_entries, dtype=np.int32)
    matrix_values = np.zeros(total_entries, dtype=np.float64)

    # Store B matrices for strain/stress recovery
    all_B = np.zeros((num_triangles, 3, 6), dtype=np.float64)

    for e in range(num_triangles):
        tri = mesh.triangles[e]
        phase = mesh.phase[e]
        D_elem = D_aggregate if phase == 0 else D_mortar

        coords = mesh.nodes[tri]
        ke, B, _ = cst_element_stiffness(coords, D_elem, thickness)
        all_B[e] = B

        # Global DOF indices: node n has DOFs (2n, 2n + 1)
        dofs = np.array([
            2 * tri[0], 2 * tri[0] + 1,
            2 * tri[1], 2 * tri[1] + 1,
            2 * tri[2], 2 * tri[2] + 1
        ], dtype=np.int32)

        start_idx = e * entries_per_element
        end_idx = start_idx + entries_per_element

        row_indices[start_idx:end_idx] = np.repeat(dofs, 6)
        col_indices[start_idx:end_idx] = np.tile(dofs, 6)
        matrix_values[start_idx:end_idx] = ke.flatten()

    # Build global stiffness matrix in CSR format
    K_global = sp.coo_matrix(
        (matrix_values, (row_indices, col_indices)),
        shape=(total_dofs, total_dofs),
        dtype=np.float64
    ).tocsr()

    # Partition into free and prescribed DOFs
    prescribed_dofs_map = bc.prescribed_dofs
    if not prescribed_dofs_map:
        raise ValueError("No boundary conditions specified. Stiffness matrix is singular without support.")

    prescribed_indices = np.array(sorted(prescribed_dofs_map.keys()), dtype=np.int32)
    prescribed_values = np.array([prescribed_dofs_map[d] for d in prescribed_indices], dtype=np.float64)

    if np.any(prescribed_indices < 0) or np.any(prescribed_indices >= total_dofs):
        invalid_dof = prescribed_indices[(prescribed_indices < 0) | (prescribed_indices >= total_dofs)][0]
        raise IndexError(f"Boundary condition prescribed DOF {invalid_dof} is out of bounds [0, {total_dofs - 1}].")

    is_prescribed = np.zeros(total_dofs, dtype=bool)
    is_prescribed[prescribed_indices] = True
    free_indices = np.where(~is_prescribed)[0]

    if len(free_indices) == 0:
        raise ValueError("All DOFs are prescribed; no free degrees of freedom to solve.")

    # Submatrices
    K_ff = K_global[free_indices, :][:, free_indices]
    K_fp = K_global[free_indices, :][:, prescribed_indices]

    # Equilibrium on free DOFs: K_ff @ u_f + K_fp @ u_p = f_f = 0
    rhs_f = -K_fp @ prescribed_values

    # Solve sparse linear system
    try:
        u_f = spla.spsolve(K_ff.tocsc(), rhs_f)
    except Exception as exc:
        raise RuntimeError(
            f"Direct sparse solver failed to factorize stiffness matrix: {str(exc)}. "
            f"Check for unconstrained rigid-body motion or disconnected elements."
        )

    # Reassemble full displacement vector
    u_full = np.zeros(total_dofs, dtype=np.float64)
    u_full[free_indices] = u_f
    u_full[prescribed_indices] = prescribed_values

    if np.any(np.isnan(u_full)) or np.any(np.isinf(u_full)):
        raise RuntimeError("Solution vector contains NaN or Inf values. System is unstable.")

    # Calculate reaction forces at all DOFs: R = K @ u - f_ext
    reactions_full = K_global @ u_full

    # Reshape nodal arrays to (N, 2)
    displacements = u_full.reshape((num_nodes, 2))
    reactions = reactions_full.reshape((num_nodes, 2))

    # Calculate element strains and stresses
    strains = np.zeros((num_triangles, 3), dtype=np.float64)
    stresses = np.zeros((num_triangles, 3), dtype=np.float64)
    von_mises = np.zeros(num_triangles, dtype=np.float64)
    principal_stresses = np.zeros((num_triangles, 2), dtype=np.float64)

    for e in range(num_triangles):
        tri = mesh.triangles[e]
        phase = mesh.phase[e]
        D_elem = D_aggregate if phase == 0 else D_mortar

        dofs = np.array([
            2 * tri[0], 2 * tri[0] + 1,
            2 * tri[1], 2 * tri[1] + 1,
            2 * tri[2], 2 * tri[2] + 1
        ], dtype=np.int32)
        u_elem = u_full[dofs]

        # Strain recovery: eps = B @ u_elem
        eps = all_B[e] @ u_elem
        # Stress recovery: sigma = D @ eps
        sig = D_elem @ eps

        strains[e] = eps
        stresses[e] = sig

        sxx = sig[0]
        syy = sig[1]
        sxy = sig[2]

        # Von Mises stress (plane stress formulation):
        # sigma_vm = sqrt(sxx^2 - sxx*syy + syy^2 + 3*sxy^2)
        vm_sq = sxx**2 - sxx * syy + syy**2 + 3.0 * (sxy**2)
        von_mises[e] = np.sqrt(max(0.0, vm_sq))

        # Principal stresses: sigma_1,2 = (sxx + syy)/2 +/- sqrt(((sxx - syy)/2)^2 + sxy^2)
        s_avg = 0.5 * (sxx + syy)
        r_rad = np.sqrt((0.5 * (sxx - syy))**2 + sxy**2)
        principal_stresses[e, 0] = s_avg + r_rad # sigma_1 (major)
        principal_stresses[e, 1] = s_avg - r_rad # sigma_2 (minor)

    # Macroscopic metrics & Equilibrium checks
    top_nodes = mesh.boundary_nodes['top']
    bottom_nodes = mesh.boundary_nodes['bottom']

    total_r_top = float(np.sum(reactions[top_nodes, 1]))
    total_r_bottom = float(np.sum(reactions[bottom_nodes, 1]))
    total_r_x = float(np.sum(reactions[:, 0]))

    eq_res_y = float(abs(total_r_top + total_r_bottom))
    eq_res_x = float(abs(total_r_x))

    # Cross-sectional width and nominal height
    W = mesh.domain_width
    H = mesh.domain_height
    load_area = W * thickness

    nominal_compressive_stress = float(abs(total_r_top) / load_area)

    # Nominal strain: average vertical displacement difference across boundary
    top_disp_mean = float(np.mean(displacements[top_nodes, 1]))
    bot_disp_mean = float(np.mean(displacements[bottom_nodes, 1]))
    delta_y = abs(top_disp_mean - bot_disp_mean)
    nominal_vertical_strain = float(delta_y / H) if H > 0 else 0.0

    if nominal_vertical_strain > 1e-12:
        effective_modulus = float(nominal_compressive_stress / nominal_vertical_strain)
    else:
        effective_modulus = 0.0

    solve_time = time.perf_counter() - t0

    return FEMResult(
        displacements=displacements,
        strains=strains,
        stresses=stresses,
        von_mises=von_mises,
        principal_stresses=principal_stresses,
        reactions=reactions,
        total_reaction_top=total_r_top,
        total_reaction_bottom=total_r_bottom,
        nominal_compressive_stress=nominal_compressive_stress,
        nominal_vertical_strain=nominal_vertical_strain,
        effective_modulus=effective_modulus,
        equilibrium_residual_y=eq_res_y,
        equilibrium_residual_x=eq_res_x,
        solve_time_seconds=solve_time,
        total_dofs=total_dofs,
        thickness=thickness,
        geometry_fingerprint=getattr(mesh, 'geometry_fingerprint', '')
    )
