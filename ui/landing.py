"""
landing.py
----------
Landing page for the Mesoscale Concrete FEM application.

Design notes
------------
- All custom HTML uses classes defined in ui/styles.py (design token CSS).
- In-page anchor links are replaced by st.button scroll helpers because
  Streamlit iframes block native browser anchor scrolling reliably only in
  full-page mode and even then inconsistently across browsers.
- No invented content: developers, institution, disclaimers, and benchmark
  values are taken verbatim from the project specification.
"""

from typing import Callable
import streamlit as st


def render_landing_page(navigate_to_app: Callable[[], None]) -> None:
    """
    Render the academic landing page.

    Parameters
    ----------
    navigate_to_app : Callable
        Zero-argument callback that navigates to Phase 1 and reruns.
    """

    # ------------------------------------------------------------------
    # HERO
    # ------------------------------------------------------------------
    st.markdown("""
    <div class="hero-box">
        <div class="hero-badge">B.Tech Civil Engineering &mdash; Minor Project</div>
        <h1 class="hero-title">Mesoscale Finite Element Modeling of Concrete</h1>
        <div class="hero-subtitle">
            Effect of Aggregate Distribution and ITZ Properties on Uniaxial Compressive Behavior
        </div>
        <div class="hero-desc">
            An interactive computational pipeline modeling concrete at the mesoscale — explicit
            aggregate, mortar matrix, and Interfacial Transition Zone (ITZ) phases — under
            displacement-controlled uniaxial compression using nonlinear cohesive FEM.
        </div>
    </div>
    """, unsafe_allow_html=True)

    cta_col, _, _ = st.columns([1, 1, 1])
    with cta_col:
        if st.button(
            "Launch Application →",
            type="primary",
            use_container_width=True,
            key="launch_app_hero_btn",
        ):
            navigate_to_app()

    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
    st.divider()

    # ------------------------------------------------------------------
    # SECTION B — Project Introduction
    # ------------------------------------------------------------------
    st.subheader("Project Overview & Motivation")
    st.markdown("""
Concrete is a **multi-phase, heterogeneous composite** material at the mesoscale (0.1–50 mm).
Its mechanical response is governed by the geometrical distribution, volume fraction, and morphology
of coarse aggregates, the cohesive properties of the mortar matrix, and the mechanical compliance
of the **Interfacial Transition Zone (ITZ)**.

The ITZ is the narrow, highly porous region of cement paste surrounding coarse aggregate particles.
Due to higher local water–cement ratios during hydration, it exhibits lower tensile strength and
reduced stiffness than the bulk mortar. Under compressive loading, stress concentrations at aggregate
boundaries cause micro-cracks to preferentially nucleate along the ITZ before bridging into the matrix.

> **Academic purpose**: A transparent, interactive pipeline demonstrating every step of mesoscale
> concrete modeling — from IS 10262:2019 mix design and Random Sequential Packing to boundary-conforming
> mesh generation and nonlinear cohesive FEM. Designed as an academic numerical demonstration, not a
> certified structural design tool.
    """)

    st.divider()

    # ------------------------------------------------------------------
    # SECTION C — Computational Workflow
    # ------------------------------------------------------------------
    st.subheader("Computational Workflow Pipeline")
    st.caption("End-to-end simulation pipeline implemented in this application:")

    workflow_steps = [
        (1, "Concrete Mix Design",
         "IS 10262:2019 calculator: target mean strength, w/c ratio, bulk phase proportions."),
        (2, "Aggregate Grading & Size Distribution",
         "IS 383:2016 sieve grading with stereological 3D-to-2D area projection."),
        (3, "Mesostructure Generation & Morphology",
         "Angular, rounded, or elongated polygon aggregates via Random Sequential Packing (RSP)."),
        (4, "Geometry Validation",
         "Non-overlapping boundaries, domain containment, mortar matrix connectivity."),
        (5, "Conforming Mesh Generation",
         "Boundary-conforming CDT (Shewchuk's Triangle) with minimum interior angle control."),
        (6, "Material Phase Assignment",
         "Phase indices: 0 = Aggregate (elastic), 1 = Mortar; ITZ boundary topology extraction."),
        (7, "Nonlinear FEM Analysis",
         "Displacement-controlled compression — CST elements, incremental Newton-Raphson solver."),
        (8, "Damage Evaluation",
         "Concurrent bulk mortar tensile crack-band softening and zero-thickness ITZ cohesive damage."),
        (9, "Results & Visualization",
         "Cauchy/principal stresses, damage contours, stress-strain curves, convergence logs."),
    ]

    grid_html = '<div class="workflow-grid">'
    for num, title, desc in workflow_steps:
        grid_html += f"""
        <div class="step-card">
            <div class="step-header">
                <span class="step-number">{num}</span>
                <span class="step-title">{title}</span>
            </div>
            <div class="step-body">{desc}</div>
        </div>"""
    grid_html += "</div>"
    st.markdown(grid_html, unsafe_allow_html=True)

    st.divider()

    # ------------------------------------------------------------------
    # SECTION D — Scientific Model Overview
    # ------------------------------------------------------------------
    st.subheader("Scientific Model Overview")
    st.caption("Constitutive laws, kinematic assumptions, and numerical formulations:")

    overview_items = [
        ("🪨 Aggregate Phase",
         "Linear-elastic isotropic continuum. "
         "E&#8336; = 55,000 MPa, &#957;&#8336; = 0.20. "
         "Aggregates remain undamaged throughout loading."),
        ("🧱 Mortar Phase",
         "Isotropic tensile damage (Rankine equivalent strain &#949;&#8321; &gt; &#949;&#8320; = f&#8348;/E&#8344;), "
         "regularized by the B&#780;azant&ndash;Oh crack-band fracture energy formulation (G&#7781;&#8344;). "
         "Residual stiffness &#945;&#7523;&#7497;&#7515; = 10&#8315;&#8310;."),
        ("⚡ ITZ Interfaces",
         "4-node zero-thickness cohesive elements. Bilinear traction-separation law "
         "governed by Mode-I normal opening: penalty stiffness k&#8345;, tensile strength "
         "f&#8348;&#7488;&#7511;&#7503;, fracture energy G&#7781;&#7488;&#7511;&#7503;. "
         "Tangential shear stiffness degrades proportionally with opening damage."),
        ("📐 Kinematics & Elements",
         "3-node Constant Strain Triangle (CST) elements, plane-stress assumption, "
         "unit out-of-plane thickness t = 1.0 mm (default)."),
        ("🔄 Nonlinear Solver",
         "Incremental-iterative Newton-Raphson with displacement control, automatic "
         "load-step cutbacks, and isolated trial state tracking for irreversible damage."),
        ("🎯 Convergence Criteria",
         "Dual-criterion acceptance: strict force residual ‖r‖₂ < 10⁻² N, "
         "OR nanoscale displacement correction ‖Δu‖₂ < 10⁻⁶ mm "
         "with relaxed force bound 10 × force_tol. "
         "Increments meeting the latter are explicitly documented."),
    ]

    ov_html = '<div class="overview-grid">'
    for title, text in overview_items:
        ov_html += f"""
        <div class="overview-card">
            <div class="overview-card-title">{title}</div>
            <div class="overview-card-text">{text}</div>
        </div>"""
    ov_html += "</div>"
    st.markdown(ov_html, unsafe_allow_html=True)

    st.divider()

    # ------------------------------------------------------------------
    # SECTION E — Benchmark Reference
    # ------------------------------------------------------------------
    st.subheader("Benchmark Reference")
    st.markdown("""
    <div class="benchmark-box">
        <div class="benchmark-title">
            &#128202; Phase 6.5 Benchmark &mdash; Preliminary Results (Not Validated)
        </div>
        <div class="benchmark-body">
            Controlled mesh sensitivity benchmark on a standard 150&thinsp;&times;&thinsp;150&thinsp;mm specimen:
            <div class="benchmark-metrics-grid">
                <div class="benchmark-metric-item">
                    <div class="benchmark-metric-label">Specimen</div>
                    <div class="benchmark-metric-value">150 &times; 150 mm (Seed 100)</div>
                </div>
                <div class="benchmark-metric-item">
                    <div class="benchmark-metric-label">Aggregates</div>
                    <div class="benchmark-metric-value">~125 particles &mdash; 39.47% area</div>
                </div>
                <div class="benchmark-metric-item">
                    <div class="benchmark-metric-label">Target Displacement</div>
                    <div class="benchmark-metric-value">&Delta;y = &minus;0.0400 mm</div>
                </div>
                <div class="benchmark-metric-item">
                    <div class="benchmark-metric-label">Global Stress Difference</div>
                    <div class="benchmark-metric-value">&Delta;&sigma; &asymp; 0.56%</div>
                </div>
            </div>
            <strong>Discretization 1 (Standard):</strong>
            4,403 nodes &mdash; 5,979 bulk elements &mdash; 1,297 ITZ segments
            &rarr; <strong>~8.613 MPa</strong><br>
            <strong>Discretization 2 (Fine):</strong>
            10,218 nodes &mdash; 15,931 bulk elements &mdash; 2,086 ITZ segments
            &rarr; <strong>~8.565 MPa</strong><br><br>
            <em>Note</em>: Demonstrates numerical solver stability across discretizations.
            Not proof of mesh independence; some increments required relaxed convergence.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ------------------------------------------------------------------
    # SECTION F — Disclaimer
    # ------------------------------------------------------------------
    st.markdown("""
    <div class="disclaimer-box">
        <div class="disclaimer-title">&#9888;&#65039; Scientific Scope &amp; Disclaimer</div>
        <div class="disclaimer-body">
            <strong>Notice:</strong> This is an academic numerical modeling project.
            Results have <strong>not been experimentally validated</strong> and must
            <strong>NOT</strong> be used for certified structural design, code compliance,
            or mix design approval.<br><br>
            <strong>Documented limitations:</strong>
            <ul>
                <li><strong>2D plane-stress idealization</strong> &mdash; 3D confinement and out-of-plane cracking not captured.</li>
                <li><strong>Tensile softening only</strong> &mdash; no compressive crushing, shear-friction dilatancy.</li>
                <li><strong>Linear elastic aggregates</strong> &mdash; no intragranular cleavage or fracture.</li>
                <li><strong>Mode-I governed ITZ</strong> &mdash; normal-opening damage only; no independent Mode-II criterion.</li>
                <li><strong>Unproven mesh independence</strong> &mdash; crack-band regularization applied but full mesh-independence of compressive strength not demonstrated.</li>
                <li><strong>No experimental calibration</strong> &mdash; material parameters taken from literature ranges.</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ------------------------------------------------------------------
    # SECTION G — Developers & Institution
    # ------------------------------------------------------------------
    st.subheader("Developers & Academic Information")
    st.caption("Undergraduate minor project — Department of Civil Engineering, 7th Semester:")

    devs = [
        ("Sanidhya Somawar",       "SS", "avatar-blue"),
        ("Onkar Prasad Patel",     "OP", "avatar-indigo"),
        ("Priyaranjan Pandey",     "PP", "avatar-teal"),
        ("Abhimanyu Singh Rajput", "AR", "avatar-slate"),
    ]

    dev_html = '<div class="dev-grid">'
    for name, initials, badge_cls in devs:
        dev_html += f"""
        <div class="developer-card">
            <div class="avatar-badge {badge_cls}">{initials}</div>
            <div>
                <div class="dev-name">{name}</div>
                <div class="dev-dept">B.Tech &middot; Civil Engineering</div>
            </div>
        </div>"""
    dev_html += "</div>"
    st.markdown(dev_html, unsafe_allow_html=True)

    st.markdown("""
    <div class="institution-block">
        <p>
            <strong>Institution:</strong> Government Engineering College, Bilaspur<br>
            <strong>Department:</strong> Department of Civil Engineering<br>
            <strong>Semester:</strong> 7th Semester &mdash; B.Tech Minor Project<br>
            <strong>Location:</strong> Bilaspur, Chhattisgarh, India
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Bottom CTA
    st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)
    bot_col1, bot_col2, bot_col3 = st.columns([1, 2, 1])
    with bot_col2:
        if st.button(
            "Proceed to Phase 1: Mix Design →",
            type="primary",
            use_container_width=True,
            key="launch_app_bot_btn",
        ):
            navigate_to_app()
