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
from matplotlib.collections import PatchCollection, LineCollection
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
        domain_width = col_w.number_input("Width (mm)", value=150.0, step=10.0)
        domain_height = col_h.number_input("Height (mm)", value=150.0, step=10.0)
        
        st.subheader("IS 383 Grading Setup")
        nom_size = st.selectbox(
            "Nominal Max Size", [10, 20, 40], 
            index=[10, 20, 40].index(default_size) if default_size in [10, 20, 40] else 1
        )
        
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
            st.session_state['specimen_result'] = result
            
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
                mesh, pre_report = generate_conforming_mesh(
                    domain_width=specimen.domain_width,
                    domain_height=specimen.domain_height,
                    polygons=specimen.polygons,
                    target_edge_mortar=target_edge_mortar,
                    target_edge_aggregate=target_edge_aggregate,
                    min_angle=min_angle,
                    max_triangles=int(max_triangles),
                    run_pre_checks=True
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
            
        except Exception as e:
            st.error(f"Meshing failed: {str(e)}")
            return

    if 'current_mesh' in st.session_state:
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
# MAIN ROUTING
# ==============================================================================

if __name__ == "__main__":
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Phase 1: Mix Design", "Phase 2: Geometry Generation", "Phase 3: Conforming Mesh"])
    
    if page == "Phase 1: Mix Design":
        render_mix_design_page()
    elif page == "Phase 2: Geometry Generation":
        render_geometry_page()
    else:
        render_mesh_page()
