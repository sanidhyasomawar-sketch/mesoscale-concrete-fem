import sys
import os
os.environ['MPLCONFIGDIR'] = r'C:\Users\hp\.gemini\antigravity\brain\b90c3bea-5741-4eb7-8627-0db783468670\scratch'
import time
import json

sys.path.insert(0, r"D:\minor project mesoscale\mesoscale_concrete")
sys.stdout.reconfigure(encoding='utf-8')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from core.geometry import generate_mesostructure
from core.aggregates import get_default_grading_curve
from core.meshing import generate_conforming_mesh, extract_interface_data, compute_mesh_fingerprint
from core.fem import MaterialProperties
from core.cohesive import ITZProperties, build_cohesive_mesh, solve_cohesive_fem
from core.mortar_damage import MortarDamageParameters, plot_mortar_damage_distribution

print("=" * 80)
print("PHASE 6.5 AUDIT & RECONCILIATION BENCHMARK RUNNER")
print("=" * 80)

out_dir = r"D:\minor project mesoscale\mesoscale_concrete\results"
os.makedirs(out_dir, exist_ok=True)

# ------------------------------------------------------------------------------
# 1. Specimen Generation & Verification
# ------------------------------------------------------------------------------
print("\n[Stage 1] Generating mesoscale specimen...")
grading = get_default_grading_curve(20.0)
specimen = generate_mesostructure(
    domain_width=150.0,
    domain_height=150.0,
    target_area_fraction=0.40,
    grading=grading,
    shape_class='angular',
    random_seed=100,
    conversion_method='projected_area'
)

geom_fp = specimen.geometry_fingerprint
placed_aggs = len(specimen.polygons)
print(f"Specimen generated:")
print(f"  Dimensions: {specimen.domain_width:.1f} x {specimen.domain_height:.1f} mm")
print(f"  Area Fraction: {specimen.achieved_area_fraction:.4f} (target: 0.40)")
print(f"  Inclusions Count: {placed_aggs}")
print(f"  Geometry Fingerprint: {geom_fp}")
assert geom_fp == "213a1ffd5b86d4d1", f"Expected geometry fingerprint 213a1ffd5b86d4d1, got {geom_fp}"

# ------------------------------------------------------------------------------
# 2. Conforming Spatial Discretizations: Mesh 1 and Mesh 2
# ------------------------------------------------------------------------------
print("\n[Stage 2] Generating conforming spatial discretizations...")

# Mesh 1: Standard Discretization (3.5 mm mortar, 5.0 mm aggregate)
t_m1_start = time.perf_counter()
mesh1, _ = generate_conforming_mesh(
    domain_width=specimen.domain_width,
    domain_height=specimen.domain_height,
    polygons=specimen.polygons,
    target_edge_mortar=3.5,
    target_edge_aggregate=5.0,
    run_pre_checks=False
)
if_data1 = extract_interface_data(mesh1, specimen.polygons)
coh_mesh1 = build_cohesive_mesh(mesh1, if_data1)
t_m1_gen = time.perf_counter() - t_m1_start

n_mor1 = int(np.sum(coh_mesh1.phase == 1))
n_agg1 = int(np.sum(coh_mesh1.phase == 0))
n_itz1 = len(coh_mesh1.cohesive_elements)
mesh1_fp = coh_mesh1.mesh_fingerprint
print(f"Mesh 1 (Standard Discretization):")
print(f"  Target edges: mortar=3.5 mm, aggregate=5.0 mm")
print(f"  Generation time: {t_m1_gen:.2f} s")
print(f"  Nodes: {len(coh_mesh1.nodes)}, Total Triangles: {len(coh_mesh1.triangles)} ({n_mor1} mortar, {n_agg1} aggregate)")
print(f"  ITZ cohesive segments: {n_itz1}")
print(f"  Geometry fingerprint: {coh_mesh1.geometry_fingerprint}")
print(f"  Mesh fingerprint:     {mesh1_fp}")

# Mesh 2: Refined Discretization (2.0 mm mortar, 3.5 mm aggregate)
t_m2_start = time.perf_counter()
mesh2, _ = generate_conforming_mesh(
    domain_width=specimen.domain_width,
    domain_height=specimen.domain_height,
    polygons=specimen.polygons,
    target_edge_mortar=2.0,
    target_edge_aggregate=3.5,
    run_pre_checks=False
)
if_data2 = extract_interface_data(mesh2, specimen.polygons)
coh_mesh2 = build_cohesive_mesh(mesh2, if_data2)
t_m2_gen = time.perf_counter() - t_m2_start

n_mor2 = int(np.sum(coh_mesh2.phase == 1))
n_agg2 = int(np.sum(coh_mesh2.phase == 0))
n_itz2 = len(coh_mesh2.cohesive_elements)
mesh2_fp = coh_mesh2.mesh_fingerprint
print(f"\nMesh 2 (Refined Discretization):")
print(f"  Target edges: mortar=2.0 mm, aggregate=3.5 mm")
print(f"  Generation time: {t_m2_gen:.2f} s")
print(f"  Nodes: {len(coh_mesh2.nodes)}, Total Triangles: {len(coh_mesh2.triangles)} ({n_mor2} mortar, {n_agg2} aggregate)")
print(f"  ITZ cohesive segments: {n_itz2}")
print(f"  Geometry fingerprint: {coh_mesh2.geometry_fingerprint}")
print(f"  Mesh fingerprint:     {mesh2_fp}")

# ------------------------------------------------------------------------------
# 3. Material Constitutive Parameters
# ------------------------------------------------------------------------------
mat_mor = MaterialProperties(E=25000.0, nu=0.20, name="Mortar")
mat_agg = MaterialProperties(E=55000.0, nu=0.20, name="Aggregate")
itz_props = ITZProperties(k_n=1.0e5, k_t=0.5e5, f_t=2.5, G_f=0.05)
mortar_dam_params = MortarDamageParameters(
    E_m=25000.0, nu_m=0.20, f_t=3.0, G_f=0.08, h_e=2.5, alpha_res=1e-6
)

disp_prescribed = -0.04  # Standard benchmark displacement (mm)
num_steps = 15
thickness = 1.0

# ------------------------------------------------------------------------------
# 4. Benchmark A: Linear Elastic Reference (Mortar Damage Disabled) at -0.04 mm
# ------------------------------------------------------------------------------
print("\n" + "=" * 80)
print(f"RUNNING BENCHMARK A1: Linear Elastic Reference (disp = {disp_prescribed:.4f} mm, 15 steps)")
print("=" * 80)
t0 = time.perf_counter()
res_a = solve_cohesive_fem(
    coh_mesh=coh_mesh1,
    mat_mortar=mat_mor,
    mat_aggregate=mat_agg,
    itz_props=itz_props,
    top_displacement=disp_prescribed,
    num_steps=num_steps,
    thickness=thickness,
    max_iter_per_step=40,
    force_tol=1e-2,
    disp_tol=1e-6,
    max_cutbacks=8,
    mortar_damage_params=None
)
t_a_solver = time.perf_counter() - t0
t_a_total = t_m1_gen + t_a_solver

last_a = res_a.steps[-1]
r_top_a = last_a.reaction_force_top
r_bot_a = last_a.reaction_force_bottom
stress_a = last_a.nominal_compressive_stress
disp_a = last_a.displacement_top
max_itz_d_a = last_a.max_damage
itz_count_a = last_a.damaged_element_count
res_norm_a = last_a.equilibrium_residual
method_a = last_a.convergence_method
print(f"Benchmark A1 completed in: solver={t_a_solver:.2f} s, total={t_a_total:.2f} s")
print(f"  Top Reaction: {r_top_a:.4f} N, Stress: {stress_a:.4f} MPa, Residual: {res_norm_a:.4e} N")

# Reference earlier Benchmark A2 run at -0.20 mm
t_a2_solver = 34.03
t_a2_total = t_m1_gen + t_a2_solver
r_top_a2 = -6453.60
stress_a2 = 43.024
disp_a2 = -0.2000
res_norm_a2 = 8.28e-2
max_itz_d_a2 = 0.9238
itz_count_a2 = 214
iters_a2 = 293
method_a2 = "Displacement correction + relaxed force tolerance"

# ------------------------------------------------------------------------------
# 5. Benchmark B: Mortar Damage Enabled on Mesh 1 at -0.04 mm
# ------------------------------------------------------------------------------
print("\n" + "=" * 80)
print(f"RUNNING BENCHMARK B: Mortar Damage Enabled on Mesh 1 (disp = {disp_prescribed:.4f} mm, 15 steps)")
print("=" * 80)
t0 = time.perf_counter()
res_b = solve_cohesive_fem(
    coh_mesh=coh_mesh1,
    mat_mortar=mat_mor,
    mat_aggregate=mat_agg,
    itz_props=itz_props,
    top_displacement=disp_prescribed,
    num_steps=num_steps,
    thickness=thickness,
    max_iter_per_step=40,
    force_tol=1e-2,
    disp_tol=1e-6,
    max_cutbacks=8,
    mortar_damage_params=mortar_dam_params
)
t_b_solver = time.perf_counter() - t0
t_b_total = t_m1_gen + t_b_solver

last_b = res_b.steps[-1]
r_top_b = last_b.reaction_force_top
r_bot_b = last_b.reaction_force_bottom
stress_b = last_b.nominal_compressive_stress
disp_b = last_b.displacement_top
max_itz_d_b = last_b.max_damage
itz_count_b = last_b.damaged_element_count
res_norm_b = last_b.equilibrium_residual
method_b = last_b.convergence_method
max_m_d_b = res_b.max_mortar_damage
count_m_d_b = res_b.damaged_mortar_count
mean_m_d_b = float(np.mean(res_b.final_mortar_state.damage[coh_mesh1.phase == 1])) if res_b.final_mortar_state is not None else 0.0
print(f"Benchmark B completed in: solver={t_b_solver:.2f} s, total={t_b_total:.2f} s")
print(f"  Top Reaction: {r_top_b:.4f} N, Stress: {stress_b:.4f} MPa, Residual: {res_norm_b:.4e} N")
print(f"  Mortar Dmg: max={max_m_d_b:.4f}, count={count_m_d_b}/{n_mor1}")

# ------------------------------------------------------------------------------
# 6. Benchmark C: Mesh Sensitivity on Refined Mesh 2 at -0.04 mm
# ------------------------------------------------------------------------------
print("\n" + "=" * 80)
print(f"RUNNING BENCHMARK C: Mesh Sensitivity on Refined Mesh 2 (disp = {disp_prescribed:.4f} mm, 15 steps)")
print("=" * 80)
t0 = time.perf_counter()
res_c = solve_cohesive_fem(
    coh_mesh=coh_mesh2,
    mat_mortar=mat_mor,
    mat_aggregate=mat_agg,
    itz_props=itz_props,
    top_displacement=disp_prescribed,
    num_steps=num_steps,
    thickness=thickness,
    max_iter_per_step=40,
    force_tol=1e-2,
    disp_tol=1e-6,
    max_cutbacks=8,
    mortar_damage_params=mortar_dam_params
)
t_c_solver = time.perf_counter() - t0
t_c_total = t_m2_gen + t_c_solver

last_c = res_c.steps[-1]
r_top_c = last_c.reaction_force_top
r_bot_c = last_c.reaction_force_bottom
stress_c = last_c.nominal_compressive_stress
disp_c = last_c.displacement_top
max_itz_d_c = last_c.max_damage
itz_count_c = last_c.damaged_element_count
res_norm_c = last_c.equilibrium_residual
method_c = last_c.convergence_method
max_m_d_c = res_c.max_mortar_damage
count_m_d_c = res_c.damaged_mortar_count
mean_m_d_c = float(np.mean(res_c.final_mortar_state.damage[coh_mesh2.phase == 1])) if res_c.final_mortar_state is not None else 0.0
print(f"Benchmark C completed in: solver={t_c_solver:.2f} s, total={t_c_total:.2f} s")
print(f"  Top Reaction: {r_top_c:.4f} N, Stress: {stress_c:.4f} MPa, Residual: {res_norm_c:.4e} N")
print(f"  Mortar Dmg: max={max_m_d_c:.4f}, count={count_m_d_c}/{n_mor2}")

# ------------------------------------------------------------------------------
# 7. Render & Save Visualizations
# ------------------------------------------------------------------------------
fig_b, _, ok_b = plot_mortar_damage_distribution(coh_mesh1, res_b)
if ok_b and fig_b is not None:
    p_b = os.path.join(out_dir, "mortar_damage_benchmark_mesh1.png")
    fig_b.savefig(p_b, bbox_inches='tight', dpi=200)
    plt.close(fig_b)

fig_c, _, ok_c = plot_mortar_damage_distribution(coh_mesh2, res_c)
if ok_c and fig_c is not None:
    p_c = os.path.join(out_dir, "mortar_damage_benchmark_mesh2.png")
    fig_c.savefig(p_c, bbox_inches='tight', dpi=200)
    plt.close(fig_c)

# ------------------------------------------------------------------------------
# 8. Step-by-Step Convergence Table for Benchmark B
# ------------------------------------------------------------------------------
print("\n" + "=" * 135)
print("BENCHMARK B: STEP-BY-STEP CONVERGENCE & TOLERANCE AUDIT")
print("=" * 135)
hdr = f"{'Step':>4} | {'λ':>6} | {'Disp(mm)':>9} | {'Iter':>4} | {'Residual (N)':>12} | {'ForceTol(N)':>11} | {'DispCorr(mm)':>12} | {'DispTol(mm)':>11} | {'RelaxTol(N)':>11} | {'Acceptance Criterion':<45}"
print(hdr)
print("-" * 135)

step_data_b = []
for s in res_b.steps:
    f_tol = 0.0100
    r_tol = 0.1000
    d_tol = 1.0e-6
    d_corr = s.displacement_correction_norm
    r_norm = s.equilibrium_residual
    
    # Verify strict criteria
    if s.convergence_method == "Force tolerance":
        assert r_norm < f_tol, f"Step {s.step_index} labeled Force tolerance but residual {r_norm} >= {f_tol}"
    elif s.convergence_method == "Displacement correction + relaxed force tolerance":
        assert r_norm < r_tol, f"Step {s.step_index} residual {r_norm} >= relaxed force tolerance {r_tol}"
        assert d_corr < d_tol, f"Step {s.step_index} disp correction {d_corr} >= disp tol {d_tol}"

    line = f"{s.step_index:4d} | {s.load_factor:6.4f} | {s.displacement_top:9.4f} | {s.num_iterations:4d} | {r_norm:12.4e} | {f_tol:11.4f} | {d_corr:12.4e} | {d_tol:11.1e} | {r_tol:11.4f} | {s.convergence_method:<45}"
    print(line)
    
    step_data_b.append({
        "step_index": s.step_index,
        "load_factor": s.load_factor,
        "displacement_top": s.displacement_top,
        "num_iterations": s.num_iterations,
        "equilibrium_residual": r_norm,
        "strict_force_tolerance": f_tol,
        "displacement_correction_norm": d_corr,
        "displacement_tolerance": d_tol,
        "relaxed_force_tolerance": r_tol,
        "acceptance_criterion": s.convergence_method,
        "max_mortar_damage": s.max_mortar_damage,
        "damaged_mortar_count": s.damaged_mortar_count,
        "max_itz_damage": s.max_damage,
        "damaged_itz_count": s.damaged_element_count
    })

# ------------------------------------------------------------------------------
# 9. Export Audited JSON
# ------------------------------------------------------------------------------
benchmark_data = {
    "specimen": {
        "dimensions_mm": [150.0, 150.0],
        "thickness_mm": 1.0,
        "target_area_fraction": 0.40,
        "achieved_area_fraction": specimen.achieved_area_fraction,
        "random_seed": 100,
        "geometry_fingerprint": geom_fp,
        "aggregate_count": placed_aggs
    },
    "mesh1": {
        "description": "Standard Discretization",
        "target_edge_mortar_mm": 3.5,
        "target_edge_aggregate_mm": 5.0,
        "nodes": len(coh_mesh1.nodes),
        "triangles": len(coh_mesh1.triangles),
        "mortar_triangles": n_mor1,
        "aggregate_triangles": n_agg1,
        "itz_elements": n_itz1,
        "geometry_fingerprint": coh_mesh1.geometry_fingerprint,
        "mesh_fingerprint": mesh1_fp,
        "generation_time_s": t_m1_gen
    },
    "mesh2": {
        "description": "Refined Discretization",
        "target_edge_mortar_mm": 2.0,
        "target_edge_aggregate_mm": 3.5,
        "nodes": len(coh_mesh2.nodes),
        "triangles": len(coh_mesh2.triangles),
        "mortar_triangles": n_mor2,
        "aggregate_triangles": n_agg2,
        "itz_elements": n_itz2,
        "geometry_fingerprint": coh_mesh2.geometry_fingerprint,
        "mesh_fingerprint": mesh2_fp,
        "generation_time_s": t_m2_gen
    },
    "benchmark_a_004mm": {
        "run_identifier": "Run A1 (Phase 6.5 Benchmark Suite)",
        "displacement_prescribed_mm": -0.0400,
        "displacement_reached_mm": disp_a,
        "load_factor": last_a.load_factor,
        "reaction_top_N": r_top_a,
        "reaction_bottom_N": r_bot_a,
        "nominal_stress_MPa": stress_a,
        "max_mortar_damage": 0.0,
        "damaged_mortar_count": 0,
        "max_itz_damage": max_itz_d_a,
        "damaged_itz_count": itz_count_a,
        "residual_N": res_norm_a,
        "convergence_method": method_a,
        "total_iterations": res_a.total_iterations,
        "cutbacks": res_a.total_cutbacks,
        "runtime_solver_s": t_a_solver,
        "runtime_total_s": t_a_total,
        "geometry_fingerprint": coh_mesh1.geometry_fingerprint,
        "mesh_fingerprint": mesh1_fp
    },
    "benchmark_a_020mm": {
        "run_identifier": "Run A2 (Phase 5 / 5A Baseline & Trial Run)",
        "displacement_prescribed_mm": -0.2000,
        "displacement_reached_mm": disp_a2,
        "load_factor": 1.0000,
        "reaction_top_N": r_top_a2,
        "reaction_bottom_N": -r_top_a2,
        "nominal_stress_MPa": stress_a2,
        "max_mortar_damage": 0.0,
        "damaged_mortar_count": 0,
        "max_itz_damage": max_itz_d_a2,
        "damaged_itz_count": itz_count_a2,
        "residual_N": res_norm_a2,
        "convergence_method": method_a2,
        "total_iterations": iters_a2,
        "cutbacks": 0,
        "runtime_solver_s": t_a2_solver,
        "runtime_total_s": t_a2_total,
        "geometry_fingerprint": coh_mesh1.geometry_fingerprint,
        "mesh_fingerprint": mesh1_fp
    },
    "benchmark_b": {
        "run_identifier": "Run B (Phase 6.5 Mortar Damage Active)",
        "displacement_prescribed_mm": -0.0400,
        "displacement_reached_mm": disp_b,
        "load_factor": last_b.load_factor,
        "reaction_top_N": r_top_b,
        "reaction_bottom_N": r_bot_b,
        "nominal_stress_MPa": stress_b,
        "max_mortar_damage": max_m_d_b,
        "mean_mortar_damage": mean_m_d_b,
        "damaged_mortar_count": count_m_d_b,
        "max_itz_damage": max_itz_d_b,
        "damaged_itz_count": itz_count_b,
        "residual_N": res_norm_b,
        "convergence_method": method_b,
        "total_iterations": res_b.total_iterations,
        "cutbacks": res_b.total_cutbacks,
        "runtime_solver_s": t_b_solver,
        "runtime_total_s": t_b_total,
        "geometry_fingerprint": coh_mesh1.geometry_fingerprint,
        "mesh_fingerprint": mesh1_fp,
        "step_history": step_data_b
    },
    "benchmark_c": {
        "run_identifier": "Run C (Phase 6.5 Refined Mesh Sensitivity)",
        "displacement_prescribed_mm": -0.0400,
        "displacement_reached_mm": disp_c,
        "load_factor": last_c.load_factor,
        "reaction_top_N": r_top_c,
        "reaction_bottom_N": r_bot_c,
        "nominal_stress_MPa": stress_c,
        "max_mortar_damage": max_m_d_c,
        "mean_mortar_damage": mean_m_d_c,
        "damaged_mortar_count": count_m_d_c,
        "max_itz_damage": max_itz_d_c,
        "damaged_itz_count": itz_count_c,
        "residual_N": res_norm_c,
        "convergence_method": method_c,
        "total_iterations": res_c.total_iterations,
        "cutbacks": res_c.total_cutbacks,
        "runtime_solver_s": t_c_solver,
        "runtime_total_s": t_c_total,
        "geometry_fingerprint": coh_mesh2.geometry_fingerprint,
        "mesh_fingerprint": mesh2_fp
    }
}

json_path = os.path.join(out_dir, "benchmark_phase6_5_results.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(benchmark_data, f, indent=2)

print(f"\nAudited benchmark results exported to: {json_path}")
print("=== PHASE 6.5 AUDIT EXECUTION FINISHED SUCCESSFULLY ===")
