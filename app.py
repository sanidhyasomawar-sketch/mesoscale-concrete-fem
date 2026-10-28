"""
app.py
------
Streamlit application for Mesoscale Concrete FEM research.
Phase 1: IS 10262:2019 Mix Design
Phase 2: Aggregate Geometry Generation (IS 383:2016, Stereology & RSP)
Phase 3: Conforming Triangular Mesh Generation & Interface Topology (Shewchuk's Triangle)
"""

import io
import time
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection, LineCollection, PolyCollection
from shapely.ops import nearest_points

from core.mix_design import MixDesignInputs, calculate_mix_design
from core.aggregates import (
    IS_383_TABLE_7_ENVELOPES, GradingCurve, get_default_grading_curve
)
from core.geometry import generate_mesostructure, PackingResult
from core.validation import validate_geometry
from core.meshing import (
    generate_conforming_mesh, extract_interface_data, save_mesh_npz,
    ConcreteMesh, InterfaceData
)
from core.mesh_checks import validate_mesh
from core.fem import (
    MaterialProperties, BoundaryConditions, solve_fem, FEMResult
)
from core.cohesive import (
    ITZProperties, CohesiveMesh, CohesiveState, build_cohesive_mesh,
    solve_cohesive_fem, NonlinearFEMResult
)

from core.synchronization import check_mesh_synchronization, invalidate_downstream_state, check_result_provenance

st.set_page_config(page_title="Mesoscale Concrete Model", layout="wide")

# ==============================================================================
# PAGE 1: MIX DESIGN MODULE (IS 10262:2019)
# ==============================================================================

def render_mix_design_page():
    st.title("Phase 1: IS 10262:2019 Mix Design")
    
    st.markdown("""
    This module implements a research-oriented concrete mix-design calculator based on **IS 10262:2019** and **IS 456:2000**.
    It computes the required phase proportions and exposes both:
    1. **Coarse aggregate fraction of total aggregate:** $V_{CA} / (V_{CA} + V_{FA})$
    2. **Coarse aggregate volume fraction of total concrete:** $V_{CA} / V_{concrete}$
    
    The second quantity is transferred to Phase 2 as the target aggregate area fraction.
    """)
    
    with st.sidebar:
        st.header("Mix Design Inputs")
        grade = st.selectbox("Concrete Grade", ["M20", "M25", "M30", "M35", "M40", "M45", "M50"], index=2)
        f_ck = float(grade[1:])
        
        max_size = st.selectbox("Nominal Max Aggregate Size (mm)", [10, 20, 40], index=1)
        fa_zone = st.selectbox("Fine Aggregate Zone", ["I", "II", "III", "IV"], index=1)
        
        slump = st.number_input("Target Slump (mm)", min_value=0, max_value=200, value=50, step=25)
        
        st.subheader("Durability & Exposure (IS 456 Table 5)")
        max_wc_exposure = st.number_input("Max w/c Ratio", min_value=0.30, max_value=0.65, value=0.45, step=0.01)
        min_cement_exposure = st.number_input("Min Cement Content (kg/m³)", min_value=250.0, max_value=450.0, value=320.0, step=10.0)
        
        st.subheader("Assumptions & Adjustments")
        user_wc_strength = st.number_input("w/c Ratio for Target Strength (0 = use empirical)", min_value=0.0, max_value=0.70, value=0.0, step=0.01)
        sp_reduction_pct = st.number_input("SP Water Reduction (%)", min_value=0.0, max_value=40.0, value=20.0, step=1.0)
        sp_dosage_pct = st.number_input("SP Dosage (% by wt. cement)", min_value=0.0, max_value=3.0, value=1.0, step=0.1)
        
        st.subheader("Specific Gravities")
        sg_cement = st.number_input("Cement", value=3.15, step=0.01)
        sg_fa = st.number_input("Fine Aggregate", value=2.74, step=0.01)
        sg_ca = st.number_input("Coarse Aggregate", value=2.74, step=0.01)
        sg_sp = st.number_input("Superplasticizer", value=1.145, step=0.01)
        
        run_btn = st.button("Calculate Mix Design", type="primary")

    if run_btn:
        inputs = MixDesignInputs(
            grade=grade,
            f_ck=f_ck,
            max_size=max_size,
            fa_zone=fa_zone,
            slump=slump,
            sp_reduction_pct=sp_reduction_pct,
            max_wc_exposure=max_wc_exposure,
            min_cement_exposure=min_cement_exposure,
            sg_cement=sg_cement,
            sg_fa=sg_fa,
            sg_ca=sg_ca,
            sg_sp=sg_sp,
            sp_dosage_pct=sp_dosage_pct,
            user_wc_strength=user_wc_strength
        )
        
        results = calculate_mix_design(inputs)
        st.session_state['vol_ca_fraction_of_concrete'] = results.vol_ca_fraction_of_concrete
        st.session_state['max_size'] = max_size
        st.session_state['grade'] = grade
        
        st.header("Calculation Results")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Target Mean Strength", f"{results.target_strength:.2f} MPa")
        col2.metric("Adopted w/c Ratio", f"{results.adopted_wc:.3f}")
        col3.metric("Water Content", f"{results.water_content_kg:.1f} kg/m³")
        
        col4, col5, col6 = st.columns(3)
        col4.metric("Cement Content", f"{results.cement_content_kg:.1f} kg/m³")
        col5.metric("Coarse Aggregate", f"{results.ca_kg:.1f} kg/m³")
        col6.metric("Fine Aggregate", f"{results.fa_kg:.1f} kg/m³")
        
        if results.warnings:
            st.subheader("Data Provenance & Logic Path")
            for w in results.warnings:
                st.info(w)
                
        st.header("Mesoscale Parameters (Transferred to Phase 2)")
        st.markdown(f"""
        * **Coarse Aggregate Fraction (within total aggregate):** `{results.vol_ca_fraction_of_aggregate:.4f}`
          * Formula: $V_{{CA}} / (V_{{CA}} + V_{{FA}})$
        * **Coarse Aggregate Volume Fraction (of total concrete):** `{results.vol_ca_fraction_of_concrete:.4f}`
          * Formula: $V_{{CA}} / V_{{concrete}}$
        
        > **2D MODELLING IDEALIZATION NOTE**:
        > The 3D coarse-aggregate volume fraction of total concrete (`{results.vol_ca_fraction_of_concrete:.4f}`) 
        > is adopted as the target aggregate **area fraction** for the 2D mesoscale model.
        > This is a modeling idealization; the two quantities are not mathematically equivalent.
        """)
        
        st.subheader("Mix Proportions Summary (per m³)")
        df = pd.DataFrame({
            "Material": ["Cement", "Water", "Fine Aggregate", "Coarse Aggregate", "Superplasticizer"],
            "Quantity (kg)": [results.cement_content_kg, results.water_content_kg, results.fa_kg, results.ca_kg, results.sp_kg],
            "Specific Gravity": [sg_cement, 1.00, sg_fa, sg_ca, sg_sp]
        })
        st.table(df)

# ==============================================================================
# PAGE 2: GEOMETRY GENERATION MODULE (PHASE 2)
# ==============================================================================

def render_geometry_page():
    st.title("Phase 2: Aggregate Geometry Generation")
    
    st.markdown("""
    Generates a 2D boundary-conforming mesostructure of coarse aggregates within a mortar matrix.
    
    **Scientific Protocol & Methodology**:
    - **Grading Envelope**: Sieve designations and passing ranges verified from **IS 383:2016 Table 7**.
    - **Mass-to-2D Conversion**: IS 383 sieve grading is mass-based (3D). The model converts this to 2D particle area fractions using stereological projected area scaling ($\propto M_k / d_k$) or direct mass analogy.
    - **Packing Algorithm**: Random Sequential Packing (RSP) with spatial boundary confinement and minimum clearance checks.
    """)
    
    default_fraction = st.session_state.get('vol_ca_fraction_of_concrete', 0.40)
    default_size = st.session_state.get('max_size', 20)
    
    with st.sidebar:
        st.header("Specimen & Material Setup")
        
        target_fraction = st.number_input(
            "Target Aggregate Area Fraction", 
            min_value=0.10, max_value=0.65, value=float(default_fraction), step=0.01,
            help="Adopted from Phase 1 calculated 3D coarse aggregate volume fraction as a 2D modeling idealization."
        )
        
        col_w, col_h = st.columns(2)
        domain_width = col_w.number_input(
            "Width (mm)", value=150.0, step=10.0,
            help="Suggested default: 150 × 150 mm for 20 mm aggregate to satisfy Representative Elementary Volume (REV)."
        )
        domain_height = col_h.number_input(
            "Height (mm)", value=150.0, step=10.0,
            help="Suggested default: 150 × 150 mm for 20 mm aggregate to satisfy Representative Elementary Volume (REV)."
        )
        
        st.subheader("IS 383 Grading Setup")
        nom_size = st.selectbox(
            "Nominal Max Size", [10, 20, 40], 
            index=[10, 20, 40].index(default_size) if default_size in [10, 20, 40] else 1
        )
        
        # Representative Elementary Volume (REV) Advisory
        min_dim = min(domain_width, domain_height)
        rev_threshold = 3.0 * nom_size
        if min_dim < rev_threshold:
            st.warning(
                f"⚠️ **Representative Elementary Volume (REV) Advisory**: Specimen dimension ({min_dim:g} mm) "
                f"is smaller than the recommended REV threshold (≥ 3 × NMAS = {rev_threshold:g} mm) for {nom_size} mm aggregate. "
                "Small specimens are convenient for quick smoke testing and debugging, but discrete particle sizing "
                "means coarse grading bins may be statistically omitted or cause larger deviations from the target area fraction. "
                "For reliable representation of the continuous grading curve, 150 × 150 mm is suggested for 20 mm aggregate "
                "(or choose 10 mm NMAS for 50 × 50 mm specimens)."
            )
        else:
            st.caption(f"✓ Specimen dimension ({min_dim:g} mm) satisfies the recommended REV scale (≥ 3 × NMAS = {rev_threshold:g} mm).")
        
        envelope_choice = st.selectbox(
            "Grading Curve Source", 
            [
                "IS 383:2016 Table 7 Midpoint (Default)",
                "IS 383:2016 Table 7 Lower Limit",
                "IS 383:2016 Table 7 Upper Limit",
                "Custom User Sieve Curve"
            ]
        )
        
        base_env = IS_383_TABLE_7_ENVELOPES[nom_size]
        
        if envelope_choice == "Custom User Sieve Curve":
            st.caption("Enter cumulative % passing for each IS sieve:")
            custom_passing = []
            for s, l, u in zip(base_env.sieve_sizes, base_env.lower_limits, base_env.upper_limits):
                val = st.number_input(f"Sieve {s:g} mm (IS range {l:g}–{u:g}%)", min_value=0.0, max_value=100.0, value=(l+u)/2.0, step=1.0)
                custom_passing.append(val)
            active_curve = GradingCurve(
                nominal_max_size=nom_size,
                sieve_sizes=list(base_env.sieve_sizes),
                percent_passing=custom_passing,
                source_label=f"User Custom ({nom_size} mm)",
                provenance="USER SPECIFIED"
            )
        elif "Lower" in envelope_choice:
            active_curve = get_default_grading_curve(nom_size, "lower")
        elif "Upper" in envelope_choice:
            active_curve = get_default_grading_curve(nom_size, "upper")
        else:
            active_curve = get_default_grading_curve(nom_size, "midpoint")
            
        st.subheader("Conversion & Morphology")
        conv_method_label = st.selectbox(
            "Mass-to-2D Area Conversion",
            [
                "Stereological Projected Area Scaling (Area ~ Mass / d_rep)",
                "Direct Mass-to-Area Analogy (Area ~ Mass)"
            ]
        )
        conv_method = "projected_area" if "Stereological" in conv_method_label else "direct_analogue"
        
        shape_class = st.selectbox("Morphology Class", ["angular", "rounded", "elongated"])
        min_spacing = st.number_input("Min Aggregate Clearance (mm)", value=1.0, min_value=0.2, max_value=5.0, step=0.1)
        random_seed = st.number_input("Random Seed", value=42, step=1)
        
        run_btn = st.button("Generate Mesostructure", type="primary")

    if run_btn:
        try:
            active_curve.validate()
        except ValueError as e:
            st.error(f"Grading validation error: {str(e)}")
            return
            
        with st.spinner("Generating aggregates and executing Random Sequential Packing..."):
            result = generate_mesostructure(
                domain_width=domain_width,
                domain_height=domain_height,
                target_area_fraction=target_fraction,
                grading=active_curve,
                shape_class=shape_class,
                min_spacing=min_spacing,
                random_seed=random_seed,
                conversion_method=conv_method,
                max_attempts_per_particle=2000
            )
            
            validation = validate_geometry(
                result.polygons, domain_width, domain_height, min_spacing=min_spacing
            )
            # Invalidate all downstream meshes and simulation results from prior geometry
            invalidated = invalidate_downstream_state(st.session_state, source="geometry")
            st.session_state['specimen_result'] = result
            if invalidated:
                st.info("ℹ️ Previous mesh and simulation results have been invalidated to maintain synchronization with this new geometry.")
            
        # 1. Summary Metrics
        st.header("Packing & Area Fraction Results")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Target Area Fraction", f"{result.target_area_fraction:.4f}")
        col2.metric("Achieved Area Fraction", f"{result.achieved_area_fraction:.4f}")
        col3.metric("Fraction Difference", f"{result.area_fraction_error:.4f}")
        col4.metric("Placed / Sampled Count", f"{result.total_particles_placed} / {result.total_particles_sampled}")
        
        if result.packing_status == "Target reached":
            st.success(f"**Packing Status**: {result.packing_status} — {result.stop_reason}")
        else:
            st.warning(f"**Packing Status**: {result.packing_status} — {result.stop_reason}")
            st.caption("""
            *Scientific Note on RSP Saturation*: In 2D Random Sequential Packing (RSP) of hard particles 
            with non-zero clearance, the algorithm encounters an observed packing saturation limit 
            (typically 45%–55% for 2D polygons). This is an algorithmic limitation of sequential 
            non-collective placement without particle reorganization/vibration, not a physical property of concrete.
            """)
            
        # 2. Geometry Validation Checks
        st.subheader("Geometry & Topology Validation")
        val_data = [
            ("All polygons simple and valid", validation["all_polygons_valid"]),
            ("No self-intersections", validation["no_self_intersections"]),
            ("All particles contained in domain", validation["contained_in_domain"]),
            ("No particle overlaps (distance > 0)", validation["no_overlaps"]),
            (f"Minimum spacing satisfied (>= {min_spacing} mm)", validation["spacing_satisfied"]),
            ("Mortar matrix remains connected", validation["mortar_connected"]),
        ]
        val_df = pd.DataFrame(val_data, columns=["Validation Criterion", "Status"])
        st.dataframe(val_df, use_container_width=True)
        
        # 3. Visualizations
        st.subheader("Mesostructure and Sieve Grading Plots")
        fig_col1, fig_col2 = st.columns([1, 1])
        
        with fig_col1:
            fig_geom, ax_geom = plt.subplots(figsize=(6, 6))
            ax_geom.set_xlim(0, domain_width)
            ax_geom.set_ylim(0, domain_height)
            ax_geom.set_aspect('equal')
            
            mortar_rect = plt.Rectangle((0, 0), domain_width, domain_height, facecolor='#E8E8E8', edgecolor='black', linewidth=1.5, zorder=0)
            ax_geom.add_patch(mortar_rect)
            
            colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(result.bin_reports)))
            for agg in result.placed_aggregates:
                x, y = agg.polygon.exterior.xy
                c = colors[agg.bin_index % len(colors)]
                mpl_poly = MplPolygon(list(zip(x, y)), closed=True, facecolor=c, edgecolor='#222222', linewidth=0.8, zorder=1)
                ax_geom.add_patch(mpl_poly)
                
            ax_geom.set_title(f"2D Concrete Mesostructure\nShape: {shape_class.capitalize()} | Achieved Area: {result.achieved_area_fraction:.3f} | Seed: {result.random_seed}", fontsize=10)
            ax_geom.set_xlabel("Width (mm)")
            ax_geom.set_ylabel("Height (mm)")
            st.pyplot(fig_geom)
            
        with fig_col2:
            fig_curve, ax_curve = plt.subplots(figsize=(6, 6))
            ax_curve.plot(result.sieve_sizes, result.target_percent_passing, 's--', color='#D9534F', label="IS 383 Target Curve (% Mass)")
            ax_curve.plot(result.sieve_sizes, result.achieved_percent_passing, 'o-', color='#0275D8', linewidth=2, label="Placed 2D Specimen (% Area)")
            
            if nom_size in IS_383_TABLE_7_ENVELOPES:
                env = IS_383_TABLE_7_ENVELOPES[nom_size]
                ax_curve.fill_between(env.sieve_sizes, env.lower_limits, env.upper_limits, color='gray', alpha=0.15, label="IS 383 Table 7 Envelope")
                
            ax_curve.set_xscale('log')
            ax_curve.set_xlabel("Sieve Size (mm) [Log Scale]")
            ax_curve.set_ylabel("Cumulative Passing (%)")
            ax_curve.set_title("Grading: IS 383 Target vs Placed 2D Specimen", fontsize=10)
            ax_curve.set_ylim(-2, 105)
            ax_curve.grid(True, which='both', linestyle=':', alpha=0.6)
            ax_curve.legend(loc='lower right', fontsize=8)
            st.pyplot(fig_curve)
            
        # 4. Grading Comparison Table
        st.subheader("Sieve Analysis & Bin Tracking")
        grading_table_data = []
        for s, t_pct, a_pct in zip(result.sieve_sizes, result.target_percent_passing, result.achieved_percent_passing):
            grading_table_data.append({
                "Sieve Size (mm)": f"{s:g}",
                "IS 383 Target % Passing (Mass)": f"{t_pct:.1f}%",
                "Placed 2D Specimen % Passing (Area)": f"{a_pct:.1f}%",
                "Difference": f"{a_pct - t_pct:+.1f}%"
            })
        st.table(pd.DataFrame(grading_table_data))
        
        # 5. Bin Breakdown Table
        st.subheader("Detailed Particle Bin Breakdown")
        bin_table_data = []
        for br in result.bin_reports:
            bin_table_data.append({
                "Sieve Interval": f"{br.d_min:g} – {br.d_max:g} mm",
                "IS 383 Mass Frac": f"{br.code_mass_fraction:.3f}",
                "Target 2D Area Frac": f"{br.target_area_fraction:.3f}",
                "Sampled (Count / Area mm²)": f"{br.sampled_count} / {br.sampled_area:.1f}",
                "Placed (Count / Area mm²)": f"{br.placed_count} / {br.placed_area:.1f}",
                "Achieved 2D Area Frac": f"{br.achieved_bin_area_fraction:.3f}"
            })
        st.table(pd.DataFrame(bin_table_data))
        
        # 6. Morphology Descriptors Summary
        if result.placed_aggregates:
            st.subheader("Particle Morphology Descriptors")
            aspect_ratios = [a.descriptors.aspect_ratio for a in result.placed_aggregates]
            circularities = [a.descriptors.circularity for a in result.placed_aggregates]
            ang_indices = [a.descriptors.angularity_indicator for a in result.placed_aggregates]
            
            morph_df = pd.DataFrame({
                "Descriptor": [
                    "Mean Aspect Ratio (Major / Minor Axis)",
                    "Mean Circularity (4π · Area / Perimeter²)",
                    "Mean Angularity Indicator (1.0 - Circularity)"
                ],
                "Value": [
                    f"{np.mean(aspect_ratios):.3f} ± {np.std(aspect_ratios):.3f}",
                    f"{np.mean(circularities):.3f} ± {np.std(circularities):.3f}",
                    f"{np.mean(ang_indices):.3f} ± {np.std(ang_indices):.3f}"
                ]
            })
            st.table(morph_df)

    # If specimen is present in session state, allow meshing directly
    if 'specimen_result' in st.session_state:
        render_mesh_section(st.session_state['specimen_result'])

# ==============================================================================
# PAGE 3: CONFORMING MESH GENERATION (PHASE 3)
# ==============================================================================

def render_mesh_section(specimen: PackingResult):
    st.markdown("---")
    st.header("Phase 3: Conforming Triangular Mesh Generation")
    
    current_mesh = st.session_state.get('current_mesh', None)
    sync = check_mesh_synchronization(specimen, current_mesh)

    if current_mesh is not None:
        if not sync.is_synchronized:
            st.warning(
                f"⚠️ **Geometry updated**: {sync.message}\n\n"
                f"- Active Phase 2 Geometry: `{sync.geometry_fingerprint}`\n"
                f"- Previous Mesh Geometry: `{sync.mesh_fingerprint or 'None'}`"
            )
        else:
            st.success(f"✓ **Mesh ready**: {sync.message} (Fingerprint: `{sync.geometry_fingerprint}`)")
    
    st.markdown("""
    Generates a 2D boundary-conforming Constrained Delaunay Triangulation (CDT) using **Shewchuk's Triangle**.
    Enforces all aggregate polygon edges as fixed boundaries, assigns regional material tags ($0 = \text{Aggregate}$, $1 = \text{Mortar}$),
    and extracts interface topology for subsequent zero-thickness cohesive ITZ insertion.
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    target_edge_mortar = col1.number_input("Mortar Target Edge (mm)", value=2.0, min_value=0.5, max_value=10.0, step=0.25)
    target_edge_aggregate = col2.number_input("Aggregate Target Edge (mm)", value=3.5, min_value=0.5, max_value=15.0, step=0.5)
    min_angle = col3.number_input("Min Angle (deg)", value=20.0, min_value=10.0, max_value=32.0, step=1.0)
    max_triangles = col4.number_input("Max Element Cap", value=60000, step=5000)
    
    mesh_btn = st.button("Generate Conforming Mesh", type="primary")
    
    if mesh_btn:
        try:
            t0 = time.perf_counter()
            with st.spinner("Generating Constrained Delaunay Triangulation via Triangle..."):
                # Invalidate any downstream Phase 4 and Phase 5 simulation results
                invalidate_downstream_state(st.session_state, source="mesh")
                mesh, pre_report = generate_conforming_mesh(
                    domain_width=specimen.domain_width,
                    domain_height=specimen.domain_height,
                    polygons=specimen.polygons,
                    target_edge_mortar=target_edge_mortar,
                    target_edge_aggregate=target_edge_aggregate,
                    min_angle=min_angle,
                    max_triangles=int(max_triangles),
                    run_pre_checks=True,
                    geometry_fingerprint=sync.geometry_fingerprint
                )
                if_data = extract_interface_data(mesh, specimen.polygons)
                val_report = validate_mesh(
                    mesh.nodes, mesh.triangles, mesh.aggregate_id, specimen.polygons,
                    specimen.domain_width, specimen.domain_height,
                    angle_fail_threshold=8.0, angle_warn_threshold=20.0
                )
            mesh_time = time.perf_counter() - t0
            
            st.session_state['current_mesh'] = mesh
            st.session_state['current_if_data'] = if_data
            st.session_state['current_val_report'] = val_report
            st.session_state['current_mesh_time'] = mesh_time
            st.success(f"✓ **Mesh ready**: Conforming mesh successfully generated and synchronized with Phase 2 geometry (Fingerprint: `{mesh.geometry_fingerprint}`).")
            sync = check_mesh_synchronization(specimen, mesh)
            
        except Exception as e:
            st.error(f"Meshing failed: {str(e)}")
            return

    if 'current_mesh' in st.session_state and sync.is_synchronized:
        mesh: ConcreteMesh = st.session_state['current_mesh']
        if_data: InterfaceData = st.session_state['current_if_data']
        val_report = st.session_state['current_val_report']
        mesh_time = st.session_state['current_mesh_time']
        
        st.subheader("Mesh Statistics & Quality Summary")
        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        m_col1.metric("Nodes", len(mesh.nodes))
        m_col2.metric("Triangles", len(mesh.triangles))
        n_agg_tri = int(np.sum(mesh.phase == 0))
        n_mor_tri = int(np.sum(mesh.phase == 1))
        m_col3.metric("Agg / Mortar Triangles", f"{n_agg_tri} / {n_mor_tri}")
        m_col4.metric("ITZ Edges", len(if_data.edges))
        m_col5.metric("Meshing Time", f"{mesh_time:.3f} s")
        
        q = val_report.quality
        if q is not None:
            q_col1, q_col2, q_col3, q_col4 = st.columns(4)
            q_col1.metric("Min Interior Angle", f"{q.min_angle:.2f}°")
            q_col2.metric("Median Angle", f"{q.median_angle:.2f}°")
            q_col3.metric("Worst Aspect Ratio", f"{q.worst_aspect_ratio:.2f}")
            q_col4.metric("Elements < 20°", q.sub_20_deg_count)
            
        if val_report.warnings:
            st.warning("Mesh Quality Warnings:\n- " + "\n- ".join(val_report.warnings))
            
        # Display Mesh Plot
        st.subheader("Conforming Triangular Mesh Visualization")
        zoom_col1, zoom_col2 = st.columns([1, 1])
        overlay_itz = zoom_col1.checkbox("Overlay Cohesive ITZ Boundaries (Crimson)", value=True)
        zoom_narrow = zoom_col2.checkbox("Zoom on Narrowest Inter-Aggregate Gap", value=False)
        
        fig_m, ax_m = plt.subplots(figsize=(8, 8), dpi=200)
        ax_m.set_aspect('equal')
        
        tri_patches = []
        colors = []
        for i, tri in enumerate(mesh.triangles):
            pts = mesh.nodes[tri]
            tri_patches.append(MplPolygon(pts, closed=True))
            if mesh.phase[i] == 0:
                colors.append('#4A90E2') # Aggregate Blue
            else:
                colors.append('#EAEAEA') # Mortar Light Gray
                
        p_coll = PatchCollection(tri_patches, facecolors=colors, edgecolors='#555555', linewidths=0.25)
        ax_m.add_collection(p_coll)
        
        if overlay_itz and len(if_data.edges) > 0:
            if_lines = []
            for e in if_data.edges:
                p1, p2 = mesh.nodes[e[0]], mesh.nodes[e[1]]
                if_lines.append([p1, p2])
            line_coll = LineCollection(if_lines, colors='#D0021B', linewidths=1.2, label='ITZ Boundary')
            ax_m.add_collection(line_coll)
            
        if zoom_narrow and len(specimen.polygons) > 1:
            # Find narrowest gap
            min_d = float('inf')
            pair = (0, 1)
            for i in range(len(specimen.polygons)):
                for j in range(i + 1, len(specimen.polygons)):
                    d = specimen.polygons[i].distance(specimen.polygons[j])
                    if d < min_d:
                        min_d = d
                        pair = (i, j)
            np1, np2 = nearest_points(specimen.polygons[pair[0]], specimen.polygons[pair[1]])
            mid_x, mid_y = (np1.x + np2.x) / 2.0, (np1.y + np2.y) / 2.0
            zoom_span = 8.0
            ax_m.set_xlim(mid_x - zoom_span, mid_x + zoom_span)
            ax_m.set_ylim(mid_y - zoom_span, mid_y + zoom_span)
            ax_m.set_title(f"Zoomed Conforming Mesh around Narrowest Gap ({min_d:.2f} mm clearance)", fontsize=10)
        else:
            ax_m.set_xlim(0, mesh.domain_width)
            ax_m.set_ylim(0, mesh.domain_height)
            ax_m.set_title(f"Conforming Mesh (Nodes: {len(mesh.nodes)}, Triangles: {len(mesh.triangles)})", fontsize=10)
            
        ax_m.set_xlabel("X (mm)")
        ax_m.set_ylabel("Y (mm)")
        st.pyplot(fig_m)
        
        # Download Mesh as .npz
        buf = io.BytesIO()
        save_mesh_npz(mesh, buf, interface=if_data)
        buf.seek(0)
        st.download_button(
            label="Download Mesh (.npz)",
            data=buf,
            file_name=f"concrete_mesh_{len(mesh.triangles)}_elements.npz",
            mime="application/octet-stream"
        )


def render_mesh_page():
    st.title("Phase 3: Conforming Triangular Mesh Generation")
    if 'specimen_result' in st.session_state:
        render_mesh_section(st.session_state['specimen_result'])
    else:
        st.info("No concrete mesostructure found in current session. Please first generate an aggregate geometry in 'Phase 2: Geometry Generation'.")

# ==============================================================================
# PAGE 4: LINEAR FEM SOLVER (PHASE 4)
# ==============================================================================

def render_fem_page():
    st.title("Phase 4: Linear-Elastic 2D Plane-Stress FEM Solver")
    
    st.markdown("""
    Direct Finite Element Method (FEM) solver using **three-node constant-strain triangular (CST) elements**.
    Solves displacement-controlled uniaxial compression on the boundary-conforming mesoscale concrete specimen.
    
    **Features & Principles:**
    - Separate isotropic linear-elastic properties for **coarse aggregates** and **mortar matrix**.
    - Displacement-controlled compression with customizable platen friction (frictionless vs sticking).
    - Sparse matrix assembly and direct linear solver via `scipy.sparse`.
    - Element strain recovery ($\epsilon_{xx}, \epsilon_{yy}, \gamma_{xy}$), Cauchy stresses ($\sigma_{xx}, \sigma_{yy}, \tau_{xy}$),
      principal stresses ($\sigma_1, \sigma_2$), and von Mises equivalent stress ($\sigma_{vm}$).
    - Reaction forces and global equilibrium verification ($\sum R_x \approx 0, \sum R_y^{top} + \sum R_y^{bottom} \approx 0$).
    """)

    specimen: Optional[PackingResult] = st.session_state.get('specimen_result', None)
    current_mesh: Optional[ConcreteMesh] = st.session_state.get('current_mesh', None)
    sync = check_mesh_synchronization(specimen, current_mesh)

    if specimen is None:
        st.info("No specimen geometry found in current session. Please first generate an aggregate geometry in **Phase 2: Geometry Generation**.")
        return

    if current_mesh is None:
        st.info("No conforming mesh found in current session. Please generate a mesh in **Phase 3: Conforming Mesh** before running the FEM solver.")
        return

    if not sync.is_synchronized:
        st.warning(
            f"⚠️ **Simulation blocked**: Phase 4 cannot run because the current mesh does not match the latest geometry. Regenerate the conforming mesh in Phase 3.\n\n"
            f"- Active Phase 2 Geometry: `{sync.geometry_fingerprint}`\n"
            f"- Existing Mesh Geometry: `{sync.mesh_fingerprint or 'None'}`"
        )
        return

        return

    mesh: ConcreteMesh = current_mesh
    st.caption(f"✓ **Mesh synchronized**: Conforming mesh matches active Phase 2 geometry (Fingerprint: `{sync.geometry_fingerprint}`).")
    
    st.subheader("1. Specimen Mesh Summary")
    s_col1, s_col2, s_col3, s_col4 = st.columns(4)
    s_col1.metric("Domain Dimensions", f"{mesh.domain_width:.1f} × {mesh.domain_height:.1f} mm")
    s_col2.metric("Nodes", len(mesh.nodes))
    s_col3.metric("Triangles", len(mesh.triangles))
    s_col4.metric("Degrees of Freedom", 2 * len(mesh.nodes))

    st.subheader("2. Material Properties & Boundary Conditions")
    
    with st.expander("Configure Materials & Loading", expanded=True):
        m_col1, m_col2 = st.columns(2)
        
        with m_col1:
            st.markdown("##### Coarse Aggregate Phase")
            e_agg = st.number_input("Aggregate Young's Modulus E_agg (MPa)", min_value=1000.0, max_value=150000.0, value=55000.0, step=1000.0)
            nu_agg = st.number_input("Aggregate Poisson's Ratio ν_agg", min_value=0.05, max_value=0.45, value=0.20, step=0.01)
            
        with m_col2:
            st.markdown("##### Mortar Matrix Phase")
            e_mor = st.number_input("Mortar Young's Modulus E_mor (MPa)", min_value=1000.0, max_value=80000.0, value=25000.0, step=1000.0)
            nu_mor = st.number_input("Mortar Poisson's Ratio ν_mor", min_value=0.05, max_value=0.45, value=0.20, step=0.01)

        b_col1, b_col2, b_col3 = st.columns(3)
        with b_col1:
            disp_y = st.number_input("Top Vertical Displacement Δy (mm)", min_value=-2.0, max_value=-0.001, value=-0.05, step=0.005, format="%.4f")
        with b_col2:
            thickness = st.number_input("Specimen Thickness t (mm)", min_value=0.1, max_value=100.0, value=1.0, step=0.5)
        with b_col3:
            platen_mode = st.selectbox(
                "Platen Boundary Condition",
                ["Frictionless (Standard Uniaxial Benchmark)", "Rough / Sticking Platens (Restrained lateral X)"]
            )
            platen_friction = (platen_mode != "Frictionless (Standard Uniaxial Benchmark)")

    run_fem_btn = st.button("Run Linear FEM Analysis", type="primary")

    if run_fem_btn:
        try:
            with st.spinner("Assembling global stiffness matrix and solving linear system..."):
                mat_aggregate = MaterialProperties(E=float(e_agg), nu=float(nu_agg), name="CoarseAggregate")
                mat_mortar = MaterialProperties(E=float(e_mor), nu=float(nu_mor), name="MortarMatrix")
                
                bc = BoundaryConditions.for_uniaxial_compression(
                    mesh=mesh,
                    top_displacement=float(disp_y),
                    platen_friction=platen_friction
                )
                
                fem_res = solve_fem(
                    mesh=mesh,
                    mat_mortar=mat_mortar,
                    mat_aggregate=mat_aggregate,
                    bc=bc,
                    thickness=float(thickness)
                )
                st.session_state['fem_result'] = fem_res
                st.session_state['fem_params'] = {
                    'e_agg': e_agg, 'nu_agg': nu_agg,
                    'e_mor': e_mor, 'nu_mor': nu_mor,
                    'disp_y': disp_y, 'thickness': thickness,
                    'platen_mode': platen_mode
                }
        except Exception as e:
            st.error(f"FEM Analysis failed: {str(e)}")
            return

    if 'fem_result' in st.session_state:
        res: FEMResult = st.session_state['fem_result']
        if not check_result_provenance(specimen, mesh, res):
            st.warning("⚠️ **Outdated results**: The linear FEM results in memory do not match the current mesh and geometry. Please re-run the simulation above.")
            return
        
        st.subheader("3. Macro-Mechanical Response & Homogenized Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Vertical Reaction |R_y|", f"{abs(res.total_reaction_top) / 1000.0:.3f} kN", f"{abs(res.total_reaction_top):.1f} N")
        col2.metric("Nominal Compressive Stress", f"{res.nominal_compressive_stress:.2f} MPa")
        col3.metric("Effective Young's Modulus", f"{res.effective_modulus:.1f} MPa")
        col4.metric("Nominal Vertical Strain", f"{res.nominal_vertical_strain:.5f}")

        eq_col1, eq_col2, eq_col3 = st.columns(3)
        eq_col1.metric("Vertical Equilibrium Error |ΣRy|", f"{res.equilibrium_residual_y:.2e} N")
        eq_col2.metric("Horizontal Equilibrium Error |ΣRx|", f"{res.equilibrium_residual_x:.2e} N")
        eq_col3.metric("Solver Wall Time", f"{res.solve_time_seconds * 1000.0:.1f} ms")

        st.warning("""
        **SCIENTIFIC & MODELLING LIMITATION NOTE**:
        This solver uses isotropic linear-elastic constant-strain triangular (CST) elements.
        It simulates pre-peak elastic behavior and elastic stress concentrations.
        It does **NOT** capture concrete micro-cracking, strain softening, shear dilatancy,
        aggregate interlock, or nonlinear interfacial decohesion (cohesive ITZ).
        The stresses displayed below represent elastic fields and must not be interpreted as damage or fracture patterns.
        """)

        st.subheader("4. Deformed Configuration & Stress Fields")
        
        viz_tab1, viz_tab2 = st.tabs(["Stress Field Visualization", "Deformed Mesh Configuration"])
        
        with viz_tab1:
            stress_opts = {
                "Vertical Normal Stress (σ_yy) [MPa]": (res.stresses[:, 1], 'coolwarm'),
                "Lateral Normal Stress (σ_xx) [MPa]": (res.stresses[:, 0], 'coolwarm'),
                "In-Plane Shear Stress (τ_xy) [MPa]": (res.stresses[:, 2], 'coolwarm'),
                "von Mises Equivalent Stress (σ_vm) [MPa]": (res.von_mises, 'viridis'),
                "Major Principal Stress (σ_1 - Tensile) [MPa]": (res.principal_stresses[:, 0], 'plasma'),
                "Minor Principal Stress (σ_2 - Compressive) [MPa]": (res.principal_stresses[:, 1], 'coolwarm_r')
            }
            
            selected_field_name = st.selectbox("Select Stress Component to Display", list(stress_opts.keys()))
            field_values, cmap_name = stress_opts[selected_field_name]
            
            # PolyCollection for fast, crisp triangular stress map
            tri_verts = [mesh.nodes[tri] for tri in mesh.triangles]
            fig_stress, ax_stress = plt.subplots(figsize=(8, 7), dpi=200)
            ax_stress.set_aspect('equal')
            
            poly_coll = PolyCollection(tri_verts, array=field_values, cmap=cmap_name, edgecolors='#333333', linewidths=0.2)
            ax_stress.add_collection(poly_coll)
            ax_stress.set_xlim(0, mesh.domain_width)
            ax_stress.set_ylim(0, mesh.domain_height)
            ax_stress.set_xlabel("X (mm)")
            ax_stress.set_ylabel("Y (mm)")
            ax_stress.set_title(f"Elastic Field: {selected_field_name}", fontsize=11)
            
            cbar = plt.colorbar(poly_coll, ax=ax_stress, fraction=0.046, pad=0.04)
            cbar.set_label("Stress (MPa)")
            
            st.pyplot(fig_stress)
            
            # Phase-wise statistics
            agg_vals = field_values[mesh.phase == 0]
            mor_vals = field_values[mesh.phase == 1]
            
            stat_df = pd.DataFrame({
                "Phase": ["Coarse Aggregate", "Mortar Matrix", "Entire Specimen"],
                "Elements": [len(agg_vals), len(mor_vals), len(field_values)],
                "Min (MPa)": [f"{np.min(agg_vals):.2f}", f"{np.min(mor_vals):.2f}", f"{np.min(field_values):.2f}"],
                "Mean (MPa)": [f"{np.mean(agg_vals):.2f}", f"{np.mean(mor_vals):.2f}", f"{np.mean(field_values):.2f}"],
                "Max (MPa)": [f"{np.max(agg_vals):.2f}", f"{np.max(mor_vals):.2f}", f"{np.max(field_values):.2f}"]
            })
            st.table(stat_df)
            
        with viz_tab2:
            scale_factor = st.slider("Displacement Magnification Factor", min_value=1.0, max_value=500.0, value=50.0, step=5.0)
            
            deformed_nodes = mesh.nodes + scale_factor * res.displacements
            
            fig_def, ax_def = plt.subplots(figsize=(8, 7), dpi=200)
            ax_def.set_aspect('equal')
            
            # Original outline
            ax_def.plot(
                [0, mesh.domain_width, mesh.domain_width, 0, 0],
                [0, 0, mesh.domain_height, mesh.domain_height, 0],
                'k--', linewidth=1.0, label='Undeformed Boundary'
            )
            
            # Deformed triangles
            def_verts = [deformed_nodes[tri] for tri in mesh.triangles]
            colors = ['#4A90E2' if p == 0 else '#EAEAEA' for p in mesh.phase]
            def_coll = PolyCollection(def_verts, facecolors=colors, edgecolors='#555555', linewidths=0.25)
            ax_def.add_collection(def_coll)
            
            ax_def.autoscale_view()
            ax_def.set_xlabel("X (mm)")
            ax_def.set_ylabel("Y (mm)")
            ax_def.set_title(f"Deformed Mesostructure (Displacements magnified {scale_factor:.0f}×)", fontsize=11)
            ax_def.legend(loc='upper right')
            
            st.pyplot(fig_def)

# ==============================================================================
# PAGE 5: COHESIVE ITZ NONLINEAR FEM (PHASE 5)
# ==============================================================================

def render_cohesive_page():
    st.title("Phase 5: Cohesive ITZ Modeling & Nonlinear FEM")
    
    st.markdown("""
    Nonlinear Finite Element Method (FEM) solver featuring **4-node zero-thickness cohesive interface elements**
    inserted along the aggregate-mortar Interfacial Transition Zone (ITZ).
    
    **Physical & Numerical Principles:**
    - **Interface Decoupling:** Node duplication along coarse aggregate perimeters creates displacement jumps $[[\\mathbf{u}]]$ across the interface.
    - **Bilinear Constitutive Law:** High initial dummy penalty stiffness ($k_n, k_t$), tensile strength criterion ($f_t^{ITZ}$),
      and linear Mode I softening governed by fracture energy ($G_f^{ITZ}$).
    - **Contact Formulation:** Penetration in compression ($\delta_n < 0$) is resisted by penalty contact stiffness without inducing opening damage.
    - **Nonlinear Newton-Raphson Solver:** Incremental displacement-controlled loading with equilibrium iterations,
      automatic load step cutbacks, and isolated trial state tracking.
    """)

    specimen: Optional[PackingResult] = st.session_state.get('specimen_result', None)
    current_mesh: Optional[ConcreteMesh] = st.session_state.get('current_mesh', None)
    if_data: Optional[InterfaceData] = st.session_state.get('current_if_data', None)
    sync = check_mesh_synchronization(specimen, current_mesh)

    if specimen is None:
        st.info("No specimen geometry found in current session. Please first generate an aggregate geometry in **Phase 2: Geometry Generation**.")
        return

    if current_mesh is None or if_data is None:
        st.info("No conforming mesh found in current session. Please first generate a mesh in **Phase 3: Mesh Generation**.")
        return

    if not sync.is_synchronized:
        st.error(
            f"⚠️ **Simulation blocked**: Phase 5 cannot run because the current mesh does not match the latest geometry. Regenerate the conforming mesh in Phase 3.\n\n"
            f"- Active Phase 2 Geometry: `{sync.geometry_fingerprint}`\n"
            f"- Current Mesh Geometry: `{sync.mesh_fingerprint or 'None'}`"
        )
        return

    mesh: ConcreteMesh = current_mesh
    st.caption(f"✓ **Mesh synchronized**: Conforming mesh matches active Phase 2 geometry (Fingerprint: `{sync.geometry_fingerprint}`).")
    if_data: InterfaceData = st.session_state['current_if_data']

    if len(if_data.edges) == 0:
        st.warning("Current mesh contains zero aggregate-mortar interface edges (homogeneous mortar specimen). Cohesive ITZ modeling requires aggregate inclusions.")
        return

    st.subheader("1. Cohesive Interface Topology Summary")
    t_col1, t_col2, t_col3, t_col4 = st.columns(4)
    t_col1.metric("Specimen Size", f"{mesh.domain_width:.1f} × {mesh.domain_height:.1f} mm")
    t_col2.metric("Original Bulk Nodes", len(mesh.nodes))
    t_col3.metric("ITZ Interface Segments", len(if_data.edges))
    t_col4.metric("Aggregate Inclusions", len(set(if_data.aggregate_id)))

    st.subheader("2. Model & Cohesive Parameters")
    
    with st.expander("Configure Materials, ITZ Properties & Solver Controls", expanded=True):
        p_col1, p_col2 = st.columns(2)
        
        with p_col1:
            st.markdown("##### Cohesive ITZ Properties")
            kn_input = st.number_input("Normal Penalty Stiffness k_n (N/mm³)", value=1.0e5, min_value=1.0e3, max_value=1.0e7, step=1.0e4, format="%.2e")
            kt_input = st.number_input("Tangential Stiffness k_t (N/mm³)", value=0.5e5, min_value=1.0e3, max_value=1.0e7, step=1.0e4, format="%.2e")
            ft_input = st.number_input("ITZ Tensile Strength f_t (MPa)", value=2.5, min_value=0.2, max_value=20.0, step=0.1)
            gf_input = st.number_input("ITZ Mode I Fracture Energy G_f (N/mm)", value=0.05, min_value=0.001, max_value=1.0, step=0.005, format="%.4f")
            
            delta_n0 = (ft_input / kn_input) * 1000.0  # micrometers
            delta_nc = (2.0 * gf_input / ft_input) * 1000.0  # micrometers
            st.caption(f"**Derived Separations:** Initiation $\\delta_{{n0}} = {delta_n0:.3f}\\ \\mu\\text{{m}}$, Critical $\\delta_{{nc}} = {delta_nc:.2f}\\ \\mu\\text{{m}}$")
            
        with p_col2:
            st.markdown("##### Bulk Material Elastic Moduli")
            e_agg_c = st.number_input("Aggregate Young's Modulus E_agg (MPa)", value=55000.0, step=1000.0)
            nu_agg_c = st.number_input("Aggregate Poisson's Ratio ν_agg", value=0.20, step=0.01)
            e_mor_c = st.number_input("Mortar Young's Modulus E_mor (MPa)", value=25000.0, step=1000.0)
            nu_mor_c = st.number_input("Mortar Poisson's Ratio ν_mor", value=0.20, step=0.01)

        st.markdown("##### Loading & Numerical Solver Controls")
        l_col1, l_col2, l_col3, l_col4 = st.columns(4)
        disp_top_c = l_col1.number_input("Imposed Top Displacement Δy (mm)", value=-0.04, min_value=-1.0, max_value=-0.001, step=0.005, format="%.4f")
        n_steps_c = l_col2.number_input("Load Increments", value=15, min_value=5, max_value=50, step=5)
        thickness_c = l_col3.number_input("Specimen Thickness t (mm)", value=1.0, min_value=0.1, max_value=50.0, step=0.5)
        platen_c = l_col4.selectbox("Platen Friction", ["Frictionless Platens", "Rough / Sticking Platens"])
        platen_fric_bool = (platen_c != "Frictionless Platens")

    run_nl_btn = st.button("Run Nonlinear Cohesive FEM Analysis", type="primary")

    if run_nl_btn:
        try:
            with st.spinner("Building cohesive mesh with duplicated interface nodes..."):
                coh_mesh = build_cohesive_mesh(mesh, if_data)
                mat_agg = MaterialProperties(E=float(e_agg_c), nu=float(nu_agg_c), name="Aggregate")
                mat_mor = MaterialProperties(E=float(e_mor_c), nu=float(nu_mor_c), name="Mortar")
                itz_props = ITZProperties(k_n=float(kn_input), k_t=float(kt_input), f_t=float(ft_input), G_f=float(gf_input))

            with st.spinner(f"Solving {int(n_steps_c)} nonlinear Newton-Raphson load increments..."):
                nl_res = solve_cohesive_fem(
                    coh_mesh=coh_mesh,
                    mat_mortar=mat_mor,
                    mat_aggregate=mat_agg,
                    itz_props=itz_props,
                    top_displacement=float(disp_top_c),
                    num_steps=int(n_steps_c),
                    thickness=float(thickness_c),
                    platen_friction=platen_fric_bool,
                    max_iter_per_step=25,
                    force_tol=1e-2
                )
                
            st.session_state['coh_mesh'] = coh_mesh
            st.session_state['nl_res'] = nl_res
            st.session_state['itz_props'] = itz_props

        except Exception as e:
            st.error(f"Nonlinear Cohesive FEM failed: {str(e)}")
            return

    if 'nl_res' in st.session_state:
        nl_res: NonlinearFEMResult = st.session_state['nl_res']
        if not check_result_provenance(specimen, mesh, nl_res):
            st.warning("⚠️ **Outdated results**: The nonlinear FEM results in memory do not match the current mesh and geometry. Please re-run the simulation above.")
            return
        coh_mesh: CohesiveMesh = st.session_state['coh_mesh']
        itz_props: ITZProperties = st.session_state['itz_props']

        st.subheader("3. Macro-Mechanical Nonlinear Response")

        if nl_res.converged:
            st.success(f"Solver Status: {nl_res.status_message} (Completed in {nl_res.total_time_seconds:.2f} s)")
        else:
            st.warning(f"Solver Status: {nl_res.status_message}")

        peak_stress = float(np.max(nl_res.stress_curve)) if len(nl_res.stress_curve) > 0 else 0.0
        peak_reaction = float(np.max(nl_res.reaction_curve)) if len(nl_res.reaction_curve) > 0 else 0.0
        final_damage_max = float(np.max(nl_res.final_state.damage))
        damaged_count = int(np.sum(nl_res.final_state.element_damage > 1e-3))

        c_col1, c_col2, c_col3, c_col4 = st.columns(4)
        c_col1.metric("Peak Compressive Stress", f"{peak_stress:.2f} MPa")
        c_col2.metric("Peak Reaction Force", f"{peak_reaction / 1000.0:.3f} kN", f"{peak_reaction:.1f} N")
        c_col3.metric("Max ITZ Damage d_max", f"{final_damage_max:.3f}", f"{final_damage_max * 100.0:.1f}%")
        c_col4.metric("Damaged ITZ Elements", f"{damaged_count} / {len(coh_mesh.cohesive_elements)}")

        st.warning("""
        **SCIENTIFIC & MODELLING LIMITATION NOTE**:
        This simulation features nonlinear decohesion localized at the Interfacial Transition Zone (ITZ).
        Bulk aggregate and mortar phases remain isotropic linear elastic (bulk cracking and crushing are not active).
        Compressive failure of laboratory concrete is governed by both ITZ debonding and progressive matrix crushing.
        The uncalibrated response curves below demonstrate constitutive interface mechanics and should not be cited
        as validated physical concrete failure data without experimental calibration.
        """)

        st.subheader("4. Visualizations & Nonlinear Mechanics")
        tab_curve, tab_damage, tab_mesh, tab_steps = st.tabs([
            "Macroscopic Response Curves", "ITZ Damage Distribution", "Deformed Interface Mesh", "Convergence History"
        ])

        with tab_curve:
            fig_curves, (ax_c1, ax_c2) = plt.subplots(1, 2, figsize=(12, 5), dpi=200)
            
            # Stress-Strain
            ax_c1.plot(nl_res.strain_curve * 1000.0, nl_res.stress_curve, 'b-o', markersize=4, linewidth=1.5, label='Cohesive ITZ Model')
            ax_c1.set_xlabel("Nominal Compressive Strain (‰ or mm/m)")
            ax_c1.set_ylabel("Nominal Compressive Stress (MPa)")
            ax_c1.set_title("Compressive Stress–Strain Response", fontsize=11)
            ax_c1.grid(True, linestyle='--', alpha=0.6)
            ax_c1.legend(loc='lower right')

            # Force-Displacement
            ax_c2.plot(nl_res.displacement_curve, nl_res.reaction_curve / 1000.0, 'r-s', markersize=4, linewidth=1.5, label='Vertical Reaction')
            ax_c2.set_xlabel("Top Vertical Displacement Δy (mm)")
            ax_c2.set_ylabel("Total Compressive Load (kN)")
            ax_c2.set_title("Load–Displacement Curve", fontsize=11)
            ax_c2.grid(True, linestyle='--', alpha=0.6)
            ax_c2.legend(loc='lower right')

            st.pyplot(fig_curves)

        with tab_damage:
            fig_dam, ax_dam = plt.subplots(figsize=(8, 7), dpi=200)
            ax_dam.set_aspect('equal')

            # Plot background aggregates and mortar
            tri_patches = []
            tri_colors = []
            for i, tri in enumerate(coh_mesh.triangles):
                pts = coh_mesh.nodes[tri]
                tri_patches.append(MplPolygon(pts, closed=True))
                tri_colors.append('#EAEAEA' if coh_mesh.phase[i] == 1 else '#D0E1F9')
            p_coll = PatchCollection(tri_patches, facecolors=tri_colors, edgecolors='#CCCCCC', linewidths=0.2)
            ax_dam.add_collection(p_coll)

            # Overlay cohesive interface segments colored by damage d
            elem_damage_vals = nl_res.final_state.element_damage
            edge_lines = []
            for elem in coh_mesh.cohesive_elements:
                p1 = coh_mesh.nodes[elem.nodes_mor[0]]
                p2 = coh_mesh.nodes[elem.nodes_mor[1]]
                edge_lines.append([p1, p2])

            damage_coll = LineCollection(
                edge_lines,
                array=elem_damage_vals,
                cmap='plasma',
                linewidths=2.5,
                clim=(0.0, 1.0)
            )
            ax_dam.add_collection(damage_coll)
            ax_dam.set_xlim(0, coh_mesh.domain_width)
            ax_dam.set_ylim(0, coh_mesh.domain_height)
            ax_dam.set_xlabel("X (mm)")
            ax_dam.set_ylabel("Y (mm)")
            ax_dam.set_title(f"Computed ITZ Damage Variable d ∈ [0, 1] ({damaged_count} damaged elements)", fontsize=11)

            cbar_dam = plt.colorbar(damage_coll, ax=ax_dam, fraction=0.046, pad=0.04)
            cbar_dam.set_label("Interface Damage d (0 = Intact, 1 = Fully Debonded)")

            st.pyplot(fig_dam)

        with tab_mesh:
            def_mag = st.slider("Interface Crack Opening Magnification", min_value=1.0, max_value=200.0, value=25.0, step=5.0)
            deformed_coords = coh_mesh.nodes + def_mag * nl_res.final_displacements

            fig_def, ax_def = plt.subplots(figsize=(8, 7), dpi=200)
            ax_def.set_aspect('equal')

            def_patches = []
            def_colors = []
            for i, tri in enumerate(coh_mesh.triangles):
                pts = deformed_coords[tri]
                def_patches.append(MplPolygon(pts, closed=True))
                def_colors.append('#F0F0F0' if coh_mesh.phase[i] == 1 else '#4A90E2')
            def_coll = PatchCollection(def_patches, facecolors=def_colors, edgecolors='#444444', linewidths=0.25)
            ax_def.add_collection(def_coll)

            ax_def.plot(
                [0, coh_mesh.domain_width, coh_mesh.domain_width, 0, 0],
                [0, 0, coh_mesh.domain_height, coh_mesh.domain_height, 0],
                'k--', linewidth=1.0, label='Undeformed Boundary'
            )
            ax_def.autoscale_view()
            ax_def.set_xlabel("X (mm)")
            ax_def.set_ylabel("Y (mm)")
            ax_def.set_title(f"Deformed Mesostructure with Debonded ITZ (Displacements magnified {def_mag:.0f}×)", fontsize=11)
            ax_def.legend(loc='upper right')

            st.pyplot(fig_def)

        with tab_steps:
            history_rows = []
            for s in nl_res.steps:
                history_rows.append({
                    "Step": s.step_index,
                    "Load Factor λ": f"{s.load_factor:.3f}",
                    "Top Disp (mm)": f"{s.displacement_top:.4f}",
                    "Reaction (N)": f"{abs(s.reaction_force_top):.1f}",
                    "Stress (MPa)": f"{s.nominal_compressive_stress:.2f}",
                    "Strain (‰)": f"{s.nominal_vertical_strain * 1000.0:.3f}",
                    "Iterations": s.num_iterations,
                    "Residual Norm (N)": f"{s.equilibrium_residual:.2e}",
                    "Max Damage": f"{s.max_damage:.3f}",
                    "Damaged Elements": s.damaged_element_count
                })
            st.dataframe(pd.DataFrame(history_rows), use_container_width=True)

# ==============================================================================
# MAIN ROUTING
# ==============================================================================

if __name__ == "__main__":
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        [
            "Phase 1: Mix Design",
            "Phase 2: Geometry Generation",
            "Phase 3: Conforming Mesh",
            "Phase 4: Linear FEM",
            "Phase 5: Cohesive ITZ"
        ]
    )
    
    if page == "Phase 1: Mix Design":
        render_mix_design_page()
    elif page == "Phase 2: Geometry Generation":
        render_geometry_page()
    elif page == "Phase 3: Conforming Mesh":
        render_mesh_page()
    elif page == "Phase 4: Linear FEM":
        render_fem_page()
    else:
        render_cohesive_page()
