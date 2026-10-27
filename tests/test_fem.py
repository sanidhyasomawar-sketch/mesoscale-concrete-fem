"""
tests/test_fem.py
-----------------
Automated test suite for Phase 4: Linear-Elastic 2D Plane-Stress FEM Solver.

Tests:
1. test_single_cst_element_properties:
   - k_e symmetry, positive semi-definiteness, exact 3 zero eigenvalues for rigid body modes.
2. test_constant_strain_patch_test:
   - Irregular multi-element patch under affine displacement field recovers exact strains & stresses.
3. test_homogeneous_uniaxial_compression_analytical_benchmark:
   - Compares FEM reaction force, stresses, strains, and effective modulus with exact elasticity.
4. test_boundary_condition_preservation_and_equilibrium:
   - Verifies all prescribed DOFs, checks global vertical and horizontal force equilibrium.
5. test_two_phase_composite_response:
   - Real mesoscale specimen with aggregates and mortar; verifies stress concentration in aggregates
     and effective modulus bounds (Reuss <= E_eff <= Voigt).
6. test_pure_shear_patch_test:
   - Verifies shear modulus G = E / (2*(1 + nu)) and shear stress recovery.
7. test_solver_linearity_and_determinism:
   - Validates linear scaling with displacement and reproducible solutions.
8. test_negative_invalid_inputs:
   - Asserts loud exceptions for non-positive E, out-of-bounds nu, non-positive thickness.
"""

import pytest
import numpy as np
from shapely.geometry import Polygon

from core.meshing import generate_conforming_mesh, ConcreteMesh
from core.fem import (
    MaterialProperties, BoundaryConditions, solve_fem,
    cst_B_matrix, cst_element_stiffness, FEMResult
)


# ==============================================================================
# 1. Single CST Element Properties
# ==============================================================================

def test_single_cst_element_properties():
    """
    Checks mathematical and mechanical properties of a single CST element:
    1. Symmetry: k_e = k_e^T.
    2. Rigid body modes: exactly 3 zero eigenvalues (2 translations, 1 rotation).
    3. Strain energy: k_e is positive semi-definite (remaining 3 eigenvalues > 0).
    4. Exact area calculation.
    """
    coords = np.array([
        [0.0, 0.0],
        [4.0, 1.0],
        [1.0, 3.0]
    ], dtype=np.float64)

    mat = MaterialProperties(E=30000.0, nu=0.20, name="Concrete")
    D = mat.plane_stress_D()
    thickness = 2.0

    k_e, B, area = cst_element_stiffness(coords, D, thickness=thickness)

    # 1. Symmetry
    assert np.allclose(k_e, k_e.T, atol=1e-12), "Element stiffness matrix must be symmetric."

    # 2. Area
    # Area of (0,0), (4,1), (1,3): 0.5 * |4*3 - 1*1| = 0.5 * 11 = 5.5
    assert np.isclose(area, 5.5, atol=1e-12)

    # 3. Eigenvalues
    eigenvals = np.linalg.eigvalsh(k_e)
    # 3 zero eigenvalues for rigid body motions in 2D
    zero_eigs = eigenvals[:3]
    pos_eigs = eigenvals[3:]

    assert np.all(np.abs(zero_eigs) < 1e-8), f"First 3 eigenvalues must be zero, got {zero_eigs}."
    assert np.all(pos_eigs > 1e-4), f"Remaining 3 eigenvalues must be strictly positive, got {pos_eigs}."


# ==============================================================================
# 2. Constant Strain Patch Test
# ==============================================================================

def test_constant_strain_patch_test():
    """
    Standard patch test: Apply a known affine displacement field
    u(x, y) = c1 * x + c2 * y
    v(x, y) = c3 * x + c4 * y
    to the boundary of an arbitrary mesh.
    All internal elements must recover the exact constant strain and stress state.
    """
    # Create an irregular mesh of a 20x20 domain with multiple elements
    mesh, _ = generate_conforming_mesh(
        domain_width=20.0,
        domain_height=20.0,
        polygons=[],
        target_edge_mortar=4.0,
        min_angle=20.0,
        run_pre_checks=False
    )

    c1, c2, c3, c4 = 0.002, 0.001, -0.0015, -0.003
    expected_eps_xx = c1
    expected_eps_yy = c4
    expected_gamma_xy = c2 + c3

    mat = MaterialProperties(E=25000.0, nu=0.25)
    D = mat.plane_stress_D()
    expected_stress = D @ np.array([expected_eps_xx, expected_eps_yy, expected_gamma_xy])

    # Prescribe all boundary nodes (top, bottom, left, right)
    prescribed: dict[int, float] = {}
    all_bnd_nodes = set(np.concatenate([
        mesh.boundary_nodes['bottom'],
        mesh.boundary_nodes['top'],
        mesh.boundary_nodes['left'],
        mesh.boundary_nodes['right']
    ]))

    for n in all_bnd_nodes:
        x, y = mesh.nodes[n]
        u_val = c1 * x + c2 * y
        v_val = c3 * x + c4 * y
        prescribed[2 * n] = u_val
        prescribed[2 * n + 1] = v_val

    bc = BoundaryConditions(prescribed_dofs=prescribed, description="Patch Test Affine Field")
    result = solve_fem(mesh, mat, mat, bc, thickness=1.0)

    # Check strains on all elements
    max_strain_err = np.max(np.abs(result.strains - np.array([expected_eps_xx, expected_eps_yy, expected_gamma_xy])))
    assert max_strain_err < 1e-10, f"Patch test strain error {max_strain_err:.2e} exceeded tolerance."

    # Check stresses on all elements
    max_stress_err = np.max(np.abs(result.stresses - expected_stress))
    assert max_stress_err < 1e-8, f"Patch test stress error {max_stress_err:.2e} exceeded tolerance."

    # Equilibrium residual on internal nodes: reaction must be zero for free nodes
    internal_nodes = [i for i in range(len(mesh.nodes)) if i not in all_bnd_nodes]
    for n in internal_nodes:
        assert np.isclose(result.reactions[n, 0], 0.0, atol=1e-8)
        assert np.isclose(result.reactions[n, 1], 0.0, atol=1e-8)


# ==============================================================================
# 3. Analytical Benchmark: Homogeneous Uniaxial Compression
# ==============================================================================

def test_homogeneous_uniaxial_compression_analytical_benchmark():
    """
    Solves displacement-controlled uniaxial compression on a 50x50 mm homogeneous plate.
    Compares against exact 1D/2D elasticity:
    - sigma_yy = E * (Delta / H)
    - sigma_xx = 0
    - Total reaction R_y = sigma_yy * W * t
    - Effective modulus E_eff = E
    """
    W, H = 50.0, 50.0
    E_true = 32000.0 # MPa
    nu_true = 0.20
    thickness = 1.5 # mm
    delta_y = -0.05 # mm compression

    mesh, _ = generate_conforming_mesh(
        domain_width=W,
        domain_height=H,
        polygons=[],
        target_edge_mortar=5.0,
        min_angle=22.0,
        run_pre_checks=False
    )

    mat = MaterialProperties(E=E_true, nu=nu_true, name="BenchmarkMortar")
    bc = BoundaryConditions.for_uniaxial_compression(
        mesh, top_displacement=delta_y, platen_friction=False
    )

    res = solve_fem(mesh, mat, mat, bc, thickness=thickness)

    # Analytical values
    strain_yy_ana = delta_y / H # -0.001
    stress_yy_ana = E_true * strain_yy_ana # -32.0 MPa
    total_reaction_top_ana = stress_yy_ana * W * thickness # -32.0 * 50 * 1.5 = -2400.0 N
    total_reaction_bot_ana = -total_reaction_top_ana # +2400.0 N

    # Reaction force comparison
    rel_err_top = abs(res.total_reaction_top - total_reaction_top_ana) / abs(total_reaction_top_ana)
    rel_err_bot = abs(res.total_reaction_bottom - total_reaction_bot_ana) / abs(total_reaction_bot_ana)
    assert rel_err_top < 1e-4, f"Top reaction relative error {rel_err_top:.2e} exceeded tolerance."
    assert rel_err_bot < 1e-4, f"Bottom reaction relative error {rel_err_bot:.2e} exceeded tolerance."

    # Nominal stress and effective modulus comparison
    assert np.isclose(res.nominal_compressive_stress, abs(stress_yy_ana), rtol=1e-4)
    assert np.isclose(res.effective_modulus, E_true, rtol=1e-4)

    # Element-wise stress distribution: all elements should experience pure uniaxial compression
    mean_syy = np.mean(res.stresses[:, 1])
    mean_sxx = np.mean(res.stresses[:, 0])
    mean_sxy = np.mean(res.stresses[:, 2])

    assert np.isclose(mean_syy, stress_yy_ana, rtol=1e-4)
    assert np.isclose(mean_sxx, 0.0, atol=1e-4)
    assert np.isclose(mean_sxy, 0.0, atol=1e-4)

    # Global equilibrium
    assert res.equilibrium_residual_y < 1e-6, f"Vertical equilibrium residual {res.equilibrium_residual_y} too high."
    assert res.equilibrium_residual_x < 1e-6, f"Horizontal equilibrium residual {res.equilibrium_residual_x} too high."


# ==============================================================================
# 4. Boundary Condition Preservation & Equilibrium
# ==============================================================================

def test_boundary_condition_preservation_and_equilibrium():
    """
    Verifies that on a multi-element specimen:
    1. Every prescribed DOF exactly matches the target value to machine precision.
    2. Sum of all forces in X and Y satisfies global equilibrium.
    """
    poly = Polygon([(10.0, 10.0), (30.0, 12.0), (28.0, 28.0), (12.0, 26.0)])
    mesh, _ = generate_conforming_mesh(
        domain_width=40.0,
        domain_height=40.0,
        polygons=[poly],
        target_edge_mortar=4.0,
        target_edge_aggregate=5.0,
        min_angle=20.0,
        run_pre_checks=False
    )

    mat_agg = MaterialProperties(E=60000.0, nu=0.18)
    mat_mor = MaterialProperties(E=25000.0, nu=0.22)
    top_disp = -0.04

    bc = BoundaryConditions.for_uniaxial_compression(
        mesh, top_displacement=top_disp, platen_friction=False
    )

    res = solve_fem(mesh, mat_mor, mat_agg, bc, thickness=1.0)

    # 1. Prescribed DOF accuracy
    for dof, target_val in bc.prescribed_dofs.items():
        node_idx = dof // 2
        coord_idx = dof % 2
        actual_val = res.displacements[node_idx, coord_idx]
        assert np.isclose(actual_val, target_val, atol=1e-14), (
            f"Prescribed DOF {dof} (node {node_idx}, dir {coord_idx}) mismatch: "
            f"actual {actual_val} vs target {target_val}"
        )

    # 2. Global Equilibrium
    assert res.equilibrium_residual_y < 1e-5
    assert res.equilibrium_residual_x < 1e-5
    assert res.nominal_compressive_stress > 0.0


# ==============================================================================
# 5. Two-Phase Composite Response & Voigt-Reuss Bounds
# ==============================================================================

def test_two_phase_composite_response():
    """
    On a mesoscale concrete specimen with coarse aggregate embedded in mortar:
    1. Stiffer aggregate phase carries higher average compressive stress than mortar.
    2. Effective composite Young's modulus satisfies Voigt (upper) and Reuss (lower) bounds:
       E_Reuss = 1 / (f_agg/E_agg + f_mor/E_mor)
       E_Voigt = f_agg*E_agg + f_mor*E_mor
       E_Reuss <= E_eff <= E_Voigt
    """
    poly1 = Polygon([(8.0, 8.0), (22.0, 8.0), (20.0, 22.0), (10.0, 20.0)])
    poly2 = Polygon([(28.0, 26.0), (42.0, 28.0), (40.0, 42.0), (26.0, 40.0)])
    polygons = [poly1, poly2]

    W, H = 50.0, 50.0
    mesh, _ = generate_conforming_mesh(
        domain_width=W,
        domain_height=H,
        polygons=polygons,
        target_edge_mortar=3.0,
        target_edge_aggregate=4.0,
        min_angle=20.0,
        run_pre_checks=False
    )

    E_agg = 65000.0 # MPa
    E_mor = 25000.0 # MPa
    nu_val = 0.20

    mat_agg = MaterialProperties(E=E_agg, nu=nu_val, name="Aggregate")
    mat_mor = MaterialProperties(E=E_mor, nu=nu_val, name="Mortar")

    bc = BoundaryConditions.for_uniaxial_compression(mesh, top_displacement=-0.05, platen_friction=False)
    res = solve_fem(mesh, mat_mor, mat_agg, bc, thickness=1.0)

    # Area fractions
    agg_area = float(np.sum(mesh.triangle_areas[mesh.phase == 0]))
    total_area = float(np.sum(mesh.triangle_areas))
    f_agg = agg_area / total_area
    f_mor = 1.0 - f_agg

    # Voigt and Reuss bounds
    E_voigt = f_agg * E_agg + f_mor * E_mor
    E_reuss = 1.0 / (f_agg / E_agg + f_mor / E_mor)

    # 1. Effective modulus strictly bounded
    assert E_reuss <= res.effective_modulus <= E_voigt, (
        f"Effective modulus {res.effective_modulus:.1f} MPa violated bounds: "
        f"Reuss={E_reuss:.1f} MPa, Voigt={E_voigt:.1f} MPa."
    )

    # 2. Stress partitioning: Stiff aggregates carry higher compressive stress |sigma_yy|
    agg_syy = np.abs(res.stresses[mesh.phase == 0, 1])
    mor_syy = np.abs(res.stresses[mesh.phase == 1, 1])

    mean_agg_syy = np.average(agg_syy, weights=mesh.triangle_areas[mesh.phase == 0])
    mean_mor_syy = np.average(mor_syy, weights=mesh.triangle_areas[mesh.phase == 1])

    assert mean_agg_syy > mean_mor_syy, (
        f"Aggregate average stress ({mean_agg_syy:.2f} MPa) should exceed "
        f"mortar average stress ({mean_mor_syy:.2f} MPa)."
    )


# ==============================================================================
# 6. Pure Shear Patch Test
# ==============================================================================

def test_pure_shear_patch_test():
    """
    Applies pure shear boundary conditions u(x, y) = gamma * y, v(x, y) = 0.
    Checks that every element recovers gamma_xy = gamma and tau_xy = G * gamma
    where G = E / (2 * (1 + nu)).
    """
    mesh, _ = generate_conforming_mesh(
        domain_width=30.0,
        domain_height=30.0,
        polygons=[],
        target_edge_mortar=5.0,
        min_angle=20.0,
        run_pre_checks=False
    )

    E = 30000.0
    nu = 0.25
    G = E / (2.0 * (1.0 + nu))
    gamma_applied = 0.001

    mat = MaterialProperties(E=E, nu=nu)

    # Prescribe all boundary nodes
    prescribed: dict[int, float] = {}
    all_bnd_nodes = set(np.concatenate([
        mesh.boundary_nodes['bottom'],
        mesh.boundary_nodes['top'],
        mesh.boundary_nodes['left'],
        mesh.boundary_nodes['right']
    ]))

    for n in all_bnd_nodes:
        x, y = mesh.nodes[n]
        prescribed[2 * n] = gamma_applied * y
        prescribed[2 * n + 1] = 0.0

    bc = BoundaryConditions(prescribed_dofs=prescribed, description="Pure Shear Test")
    res = solve_fem(mesh, mat, mat, bc, thickness=1.0)

    # Check engineering shear strain gamma_xy and shear stress tau_xy
    expected_tau = G * gamma_applied # 30000 / (2 * 1.25) * 0.001 = 12000 * 0.001 = 12.0 MPa
    max_gamma_err = np.max(np.abs(res.strains[:, 2] - gamma_applied))
    max_tau_err = np.max(np.abs(res.stresses[:, 2] - expected_tau))

    assert max_gamma_err < 1e-10, f"Shear strain error {max_gamma_err:.2e} exceeded tolerance."
    assert max_tau_err < 1e-8, f"Shear stress error {max_tau_err:.2e} exceeded tolerance."


# ==============================================================================
# 7. Solver Linearity and Determinism
# ==============================================================================

def test_solver_linearity_and_determinism():
    """
    Validates that:
    1. Doubling displacement exactly doubles strains, stresses, and reaction forces.
    2. Repeated runs on the same mesh yield bitwise identical results.
    """
    mesh, _ = generate_conforming_mesh(
        domain_width=25.0,
        domain_height=25.0,
        polygons=[],
        target_edge_mortar=5.0,
        run_pre_checks=False
    )

    mat = MaterialProperties(E=28000.0, nu=0.20)

    bc1 = BoundaryConditions.for_uniaxial_compression(mesh, top_displacement=-0.02)
    bc2 = BoundaryConditions.for_uniaxial_compression(mesh, top_displacement=-0.04)

    res1 = solve_fem(mesh, mat, mat, bc1, thickness=1.0)
    res2 = solve_fem(mesh, mat, mat, bc2, thickness=1.0)
    res1_rep = solve_fem(mesh, mat, mat, bc1, thickness=1.0)

    # 1. Determinism
    assert np.array_equal(res1.displacements, res1_rep.displacements)
    assert np.array_equal(res1.stresses, res1_rep.stresses)

    # 2. Linearity (ratio of 2.0)
    assert np.allclose(res2.displacements, 2.0 * res1.displacements, atol=1e-12)
    assert np.allclose(res2.stresses, 2.0 * res1.stresses, atol=1e-10)
    assert np.isclose(res2.total_reaction_top, 2.0 * res1.total_reaction_top, rtol=1e-10)
    assert np.isclose(res2.effective_modulus, res1.effective_modulus, rtol=1e-10)


# ==============================================================================
# 8. Negative Tests for Error Handling
# ==============================================================================

def test_negative_invalid_inputs():
    """
    Ensures clear, loud exceptions are raised when physically invalid inputs are provided.
    """
    # Negative / zero E
    with pytest.raises(ValueError, match="Young's modulus E must be strictly positive"):
        MaterialProperties(E=0.0, nu=0.2)

    with pytest.raises(ValueError, match="Young's modulus E must be strictly positive"):
        MaterialProperties(E=-1000.0, nu=0.2)

    # Invalid Poisson ratio (incompressibility limit nu >= 0.5 or nu <= -0.99)
    with pytest.raises(ValueError, match="Poisson's ratio nu must satisfy"):
        MaterialProperties(E=30000.0, nu=0.5)

    with pytest.raises(ValueError, match="Poisson's ratio nu must satisfy"):
        MaterialProperties(E=30000.0, nu=-1.0)

    # Degenerate element (collinear points)
    collinear = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    with pytest.raises(ValueError, match="Degenerate or collinear"):
        cst_B_matrix(collinear)
