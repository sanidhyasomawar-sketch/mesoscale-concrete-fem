from pathlib import Path
from typing import Callable

import streamlit as st


_ASSETS = Path(__file__).resolve().parent.parent / "assets"
_FIG_MESH = _ASSETS / "hero_mesh.png"
_FIG_ZOOM = _ASSETS / "mesh_zoom_gap.png"
_FIG_DMG_STD = _ASSETS / "damage_standard_mesh.png"
_FIG_DMG_FINE = _ASSETS / "damage_fine_mesh.png"


# Landing-page-only CSS.
_LANDING_CSS = """
<style>
/* ============================================================
   LANDING PAGE BACKGROUND — SUBTLE TRIANGULAR FEM MESH
   ============================================================ */

.stApp {
    background-color: var(--bg-page);
    background-image:
        radial-gradient(
            circle at 12% 8%,
            rgba(37, 99, 235, 0.045),
            transparent 25rem
        ),
        radial-gradient(
            circle at 88% 42%,
            rgba(37, 99, 235, 0.025),
            transparent 30rem
        ),
        url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='88' height='76' viewBox='0 0 88 76'%3E%3Cg fill='none' stroke='%232563EB' stroke-width='0.7' stroke-opacity='0.16'%3E%3Cpath d='M0 76 L44 0 L88 76 M0 0 L88 0 M0 76 L88 76 M0 0 L44 76 L88 0 M0 38 L44 0 L88 38 L44 76 Z'/%3E%3C/g%3E%3C/svg%3E");
    background-size: auto, auto, 88px 76px;
    background-repeat: no-repeat, no-repeat, repeat;
    background-attachment: fixed;
}

/* ============================================================
   GENERAL LANDING-PAGE RULES
   ============================================================ */

[data-testid="stMarkdownContainer"] .hero-box h1.hero-title {
    color: var(--hero-title) !important;
}

[class*="st-key-launch_app_"] button p {
    color: #FFFFFF !important;
}

[data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] {
    display: none !important;
}

html {
    scroll-behavior: smooth;
}

/* ============================================================
   NAVBAR
   ============================================================ */

.nav-brand {
    font-size: 1rem;
    font-weight: 700;
    color: var(--text);
    padding-top: 0.45rem;
}

.nav-links {
    display: flex;
    gap: 1.6rem;
    justify-content: center;
    padding-top: 0.5rem;
}

.nav-links a {
    font-size: 0.88rem;
    font-weight: 600;
    text-decoration: none;
    color: var(--text-secondary) !important;
}

.nav-links a:hover {
    color: var(--accent) !important;
}

.nav-rule {
    border-bottom: 1px solid var(--border-subtle);
    margin: 0.6rem 0 1.2rem 0;
}

/* ============================================================
   SECTION HEADERS
   ============================================================ */

.sec-head,
.anchor {
    scroll-margin-top: 1rem;
}

.sec-head {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 1rem;
    margin: 2.2rem 0 0.8rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border-subtle);
}

.sec-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--text);
}

.sec-note {
    font-size: 0.82rem;
    color: var(--text-muted);
    text-align: right;
}

/* ============================================================
   FIGURE HEADERS
   ============================================================ */

.fig-head {
    margin: 0.2rem 0 0.5rem 0;
}

.fig-tag {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--accent);
}

.fig-title {
    margin-top: 0.1rem;
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--text);
}

.fig-sub {
    margin-top: 0.15rem;
    font-size: 0.84rem;
    line-height: 1.45;
    color: var(--text-secondary);
}

/* ============================================================
   FIVE-PHASE PIPELINE
   ============================================================ */

.phase-list {
    border-top: 1px solid var(--border-subtle);
}

.phase-row {
    display: grid;
    grid-template-columns: 3rem minmax(0, 1fr) auto;
    align-items: center;
    gap: 1rem;
    padding: 0.85rem 0.25rem;
    border-bottom: 1px solid var(--border-subtle);
}

.phase-num {
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--accent);
}

.phase-title {
    font-size: 0.98rem;
    font-weight: 700;
    color: var(--text);
}

.phase-desc {
    margin-top: 0.1rem;
    font-size: 0.84rem;
    line-height: 1.45;
    color: var(--text-secondary);
}

.phase-tag {
    font-size: 0.74rem;
    font-weight: 600;
    white-space: nowrap;
    padding: 0.2rem 0.6rem;
    border-radius: var(--r-md);
    border: 1px solid var(--border-subtle);
    color: var(--text-secondary);
    background: var(--bg-surface);
}

/* ============================================================
   MESH-SENSITIVITY BENCHMARK
   ============================================================ */

.bench-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
}

.bench-table th,
.bench-table td {
    padding: 0.55rem 0.6rem;
    text-align: right;
    border-bottom: 1px solid var(--border-subtle);
    color: var(--text);
}

.bench-table th:first-child,
.bench-table td:first-child {
    text-align: left;
    color: var(--text-secondary);
}

.bench-table th {
    font-size: 0.76rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: var(--text-muted);
}

.bench-table td.hl {
    font-weight: 700;
    color: var(--accent);
}

.bench-note {
    margin-top: 0.7rem;
    font-size: 0.8rem;
    line-height: 1.5;
    color: var(--text-muted);
}

/* ============================================================
   BUILT-IN CHECKS AND SCOPE NOTE
   ============================================================ */

.check-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 1.5rem;
}

.check-item {
    border-top: 3px solid var(--accent);
    padding-top: 0.7rem;
}

.check-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--text);
}

.check-text {
    margin-top: 0.25rem;
    font-size: 0.84rem;
    line-height: 1.5;
    color: var(--text-secondary);
}

.scope-note {
    margin-top: 1.4rem;
    padding: 0.7rem 0.9rem;
    background: var(--warn-bg);
    border: 1px solid var(--warn-border);
    border-left: 4px solid var(--warn);
    border-radius: var(--r-md);
    font-size: 0.84rem;
    line-height: 1.5;
    color: var(--warn-text);
}

/* ============================================================
   JOURNAL-STYLE HERO
   ============================================================ */

.jt {
    margin: 0.5rem 0 1.6rem 0;
}

.jt-meta {
    font-family: Consolas, "SFMono-Regular", Menlo, monospace;
    font-size: 0.74rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--accent);
}

.jt-title {
    margin-top: 0.9rem;
    max-width: 900px;
    font-family: Georgia, "Times New Roman", serif;
    font-weight: 600;
    font-size: clamp(2.2rem, 4.2vw, 3.4rem);
    line-height: 1.08;
    letter-spacing: -0.02em;
    color: var(--text);
}

.jt-title em {
    color: var(--accent);
    font-style: italic;
}

.jt-sub {
    margin-top: 0.9rem;
    max-width: 900px;
    font-family: Georgia, "Times New Roman", serif;
    font-style: italic;
    font-size: 1.2rem;
    line-height: 1.45;
    color: var(--text-secondary);
}

.jt-rule {
    max-width: 900px;
    height: 3px;
    margin: 1.6rem 0 1.2rem 0;
    border-top: 1px solid var(--text);
    border-bottom: 1px solid var(--text);
}

.jt-abs {
    display: grid;
    grid-template-columns: 7rem minmax(0, 1fr);
    column-gap: 1.2rem;
    row-gap: 0.9rem;
    max-width: 900px;
}

.jt-abs-label {
    font-family: Consolas, "SFMono-Regular", Menlo, monospace;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--text-muted);
    padding-top: 0.25rem;
}

.jt-abs-text {
    font-size: 0.96rem;
    line-height: 1.75;
    color: var(--text-secondary);
}

.jt-kw {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
}

.jt-kw span {
    font-family: Consolas, "SFMono-Regular", Menlo, monospace;
    font-size: 0.72rem;
    padding: 0.2rem 0.65rem;
    border-radius: 999px;
    border: 1px solid var(--border);
    background: var(--bg-surface);
    color: var(--text-secondary);
}

.st-key-launch_app_hero_btn {
    max-width: 300px;
}

/* ============================================================
   RESPONSIVE LAYOUT
   ============================================================ */

@media (max-width: 900px) {
    .check-grid {
        grid-template-columns: 1fr;
    }

    .nav-links {
        display: none;
    }

    .phase-row {
        grid-template-columns: 2.2rem minmax(0, 1fr);
    }

    .phase-tag {
        display: none;
    }

    .sec-head {
        flex-direction: column;
        gap: 0.2rem;
    }

    .sec-note {
        text-align: left;
    }

    .jt-abs {
        grid-template-columns: 1fr;
        row-gap: 0.3rem;
    }
}

/* Respect users who prefer reduced motion. */
@media (prefers-reduced-motion: reduce) {
    html {
        scroll-behavior: auto;
    }
}
</style>
"""


def _figure(
    path: Path,
    tag: str,
    title: str,
    sub: str,
    caption: str,
) -> None:
    """Render one figure with a small header and caption."""
    if not path.exists():
        return

    st.markdown(
        '<div class="fig-head">'
        f'<div class="fig-tag">{tag}</div>'
        f'<div class="fig-title">{title}</div>'
        f'<div class="fig-sub">{sub}</div>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.image(str(path), caption=caption)


def render_landing_page(navigate_to_app: Callable[[], None]) -> None:
    """
    Render the academic landing page.

    Parameters
    ----------
    navigate_to_app : Callable
        Zero-argument callback that navigates to Phase 1 and reruns.
    """

    st.markdown(_LANDING_CSS, unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # TOP NAVBAR
    # ------------------------------------------------------------------

    nav_brand, nav_links, nav_btn = st.columns([2.2, 4.6, 1.6])

    with nav_brand:
        st.markdown(
            '<div class="nav-brand">Mesoscale Concrete FEM</div>',
            unsafe_allow_html=True,
        )

    with nav_links:
        st.markdown(
            '<div class="nav-links">'
            '<a href="#pipeline">Pipeline</a>'
            '<a href="#benchmark">Benchmark</a>'
            '<a href="#checks">Checks</a>'
            '<a href="#details">Details</a>'
            '<a href="#team">Team</a>'
            "</div>",
            unsafe_allow_html=True,
        )

    with nav_btn:
        if st.button(
            "Launch App →",
            type="primary",
            use_container_width=True,
            key="launch_app_nav_btn",
        ):
            navigate_to_app()

    st.markdown('<div class="nav-rule"></div>', unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # HERO
    # ------------------------------------------------------------------

    st.markdown(
        '<div class="jt">'
        '<div class="jt-meta">B.Tech Civil Engineering &middot; Minor Project &middot; GEC Bilaspur</div>'
        '<div class="jt-title">Mesoscale Finite Element Modeling of <em>Concrete</em></div>'
        '<div class="jt-sub">Effect of Aggregate Distribution and ITZ Properties '
        "on Uniaxial Compressive Behavior</div>"
        '<div class="jt-rule"></div>'
        '<div class="jt-abs">'
        '<div class="jt-abs-label">Abstract</div>'
        '<div class="jt-abs-text">An interactive computational pipeline modeling concrete at the '
        'mesoscale &mdash; explicit aggregate, mortar matrix, and Interfacial Transition Zone (ITZ) '
        "phases &mdash; under displacement-controlled uniaxial compression using nonlinear "
        "cohesive FEM.</div>"
        '<div class="jt-abs-label">Keywords</div>'
        '<div class="jt-kw"><span>Mesoscale</span><span>ITZ</span><span>Cohesive FEM</span>'
        '<span>Random Sequential Packing</span><span>Conforming mesh</span></div>'
        "</div></div>",
        unsafe_allow_html=True,
    )

    cta_col, _, _ = st.columns([1, 1, 1])

    with cta_col:
        if st.button(
            "Launch Application →",
            type="primary",
            use_container_width=True,
            key="launch_app_hero_btn",
        ):
            navigate_to_app()

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # TWO LARGE FIGURES
    # ------------------------------------------------------------------

    fig_left, fig_right = st.columns(2, gap="large")

    with fig_left:
        _figure(
            _FIG_MESH,
            "Phase 3 · Conforming mesh",
            "Every aggregate boundary is meshed",
            "Triangle elements follow each aggregate edge, so the ITZ sits on real element edges.",
            "Example output: 100 × 100 mm specimen, 3,457 nodes, 6,676 triangles. "
            "Blue = aggregate, grey = mortar, red line = cohesive ITZ interface.",
        )

    with fig_right:
        _figure(
            _FIG_ZOOM,
            "Mesh quality",
            "Elements in the narrowest mortar gap",
            "Zoom on the closest pair of aggregates in the same specimen: 1.22 mm clearance.",
            "Same 100 × 100 mm example. Dashed line = minimum gap between two aggregates.",
        )

    # ------------------------------------------------------------------
    # THE PIPELINE
    # ------------------------------------------------------------------

    st.markdown(
        '<div class="sec-head" id="pipeline">'
        '<div class="sec-title">The pipeline</div>'
        '<div class="sec-note">Five phases, run in order inside the app</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    phases = [
        (
            "01",
            "Mix Design",
            "IS 10262:2019 mix proportions and IS 383:2016 aggregate grading.",
            "IS 10262",
        ),
        (
            "02",
            "Geometry Generation",
            "Random Sequential Packing of angular, rounded or elongated aggregates.",
            "RSP",
        ),
        (
            "03",
            "Conforming Mesh",
            "Triangle mesh that follows every aggregate boundary, with ITZ interfaces.",
            "Triangle CDT",
        ),
        (
            "04",
            "Linear FEM",
            "Linear-elastic analysis: strains, stresses and an equilibrium check.",
            "CST",
        ),
        (
            "05",
            "Cohesive ITZ",
            "Nonlinear compression with mortar cracking and ITZ cohesive damage.",
            "Nonlinear",
        ),
    ]

    rows_html = '<div class="phase-list">'

    for num, title, desc, tag in phases:
        rows_html += (
            '<div class="phase-row">'
            f'<div class="phase-num">{num}</div>'
            f'<div><div class="phase-title">{title}</div>'
            f'<div class="phase-desc">{desc}</div></div>'
            f'<div class="phase-tag">{tag}</div>'
            "</div>"
        )

    rows_html += "</div>"
    st.markdown(rows_html, unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # MESH-SENSITIVITY BENCHMARK
    # ------------------------------------------------------------------

    st.markdown(
        '<div class="sec-head" id="benchmark">'
        '<div class="sec-title">Does the answer depend on the mesh?</div>'
        '<div class="sec-note">Mesh-sensitivity benchmark · preliminary, not validated</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    tbl_col, dmg_col = st.columns([1, 1.6], gap="large")

    with tbl_col:
        st.markdown(
            '<table class="bench-table">'
            "<tr><th></th><th>Standard</th><th>Fine</th></tr>"
            "<tr><td>Nodes</td><td>4,403</td><td>10,218</td></tr>"
            "<tr><td>Bulk elements</td><td>5,979</td><td>15,931</td></tr>"
            "<tr><td>ITZ segments</td><td>1,297</td><td>2,086</td></tr>"
            '<tr><td>Stress (MPa)</td><td class="hl">8.613</td><td class="hl">8.565</td></tr>'
            "<tr><td>Max mortar damage</td><td>0.052</td><td>0.122</td></tr>"
            "</table>"
            '<div class="bench-note"><strong>Stress difference &asymp; 0.56%.</strong> '
            "150 &times; 150 mm specimen, seed 100, 125 aggregate particles "
            "(39.47% area), target displacement &Delta;y = &minus;0.0400 mm. "
            "Demonstrates numerical solver stability across discretizations. "
            "Not proof of mesh independence; some increments required relaxed convergence.</div>",
            unsafe_allow_html=True,
        )

    with dmg_col:
        d1, d2 = st.columns(2)

        with d1:
            if _FIG_DMG_STD.exists():
                st.image(
                    str(_FIG_DMG_STD),
                    caption="Standard mesh: 2 of 4,228 mortar elements damaged",
                )

        with d2:
            if _FIG_DMG_FINE.exists():
                st.image(
                    str(_FIG_DMG_FINE),
                    caption="Fine mesh: 3 of 12,335 mortar elements damaged",
                )

    # ------------------------------------------------------------------
    # BUILT-IN CHECKS
    # ------------------------------------------------------------------

    st.markdown(
        '<div class="sec-head" id="checks">'
        '<div class="sec-title">Built-in checks</div>'
        '<div class="sec-note">What the code verifies before and between phases</div>'
        "</div>"
        '<div class="check-grid">'
        '<div class="check-item"><div class="check-title">Geometry is validated</div>'
        '<div class="check-text">Aggregates are checked for overlaps and for staying inside the '
        "specimen before a mesh is built.</div></div>"
        '<div class="check-item"><div class="check-title">Results stay tied to their mesh</div>'
        '<div class="check-text">Geometry and mesh fingerprints are carried through the phases, '
        "and a result that does not match its mesh is rejected.</div></div>"
        '<div class="check-item"><div class="check-title">Targets are reported honestly</div>'
        '<div class="check-text">The aggregate area fraction actually achieved is reported next '
        "to the target (for the benchmark: 39.47% against 40%).</div></div>"
        "</div>"
        '<div class="scope-note"><strong>Academic numerical demonstration.</strong> '
        "Results are not experimentally validated and must not be used for "
        "structural design. Full limitations are listed below.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    st.divider()

    # ------------------------------------------------------------------
    # DETAILS
    # ------------------------------------------------------------------

    st.markdown('<div class="anchor" id="details"></div>', unsafe_allow_html=True)
    st.subheader("Details")

    with st.expander("Project overview & motivation", expanded=True):
        st.markdown(
            """
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
            """
        )

    with st.expander("Full computational pipeline (9 steps)"):
        st.caption("End-to-end simulation pipeline implemented in this application:")

        workflow_steps = [
            (
                1,
                "Concrete Mix Design",
                "IS 10262:2019 calculator: target mean strength, w/c ratio, bulk phase proportions.",
            ),
            (
                2,
                "Aggregate Grading & Size Distribution",
                "IS 383:2016 sieve grading with stereological 3D-to-2D area projection.",
            ),
            (
                3,
                "Mesostructure Generation & Morphology",
                "Angular, rounded, or elongated polygon aggregates via Random Sequential Packing (RSP).",
            ),
            (
                4,
                "Geometry Validation",
                "Non-overlapping boundaries, domain containment, mortar matrix connectivity.",
            ),
            (
                5,
                "Conforming Mesh Generation",
                "Boundary-conforming CDT (Shewchuk's Triangle) with minimum interior angle control.",
            ),
            (
                6,
                "Material Phase Assignment",
                "Phase indices: 0 = Aggregate (elastic), 1 = Mortar; ITZ boundary topology extraction.",
            ),
            (
                7,
                "Nonlinear FEM Analysis",
                "Displacement-controlled compression — CST elements, incremental Newton-Raphson solver.",
            ),
            (
                8,
                "Damage Evaluation",
                "Concurrent bulk mortar tensile crack-band softening and zero-thickness ITZ cohesive damage.",
            ),
            (
                9,
                "Results & Visualization",
                "Cauchy/principal stresses, damage contours, stress-strain curves, convergence logs.",
            ),
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
            </div>
            """

        grid_html += "</div>"
        st.markdown(grid_html, unsafe_allow_html=True)

    with st.expander("Scientific model overview"):
        st.caption("Constitutive laws, kinematic assumptions, and numerical formulations:")

        overview_items = [
            (
                "Aggregate Phase",
                "Linear-elastic isotropic continuum. "
                "E&#8336; = 55,000 MPa, &#957;&#8336; = 0.20. "
                "Aggregates remain undamaged throughout loading.",
            ),
            (
                "Mortar Phase",
                "Isotropic tensile damage (Rankine equivalent strain &#949;&#8321; &gt; &#949;&#8320; = f&#8348;/E&#8344;), "
                "regularized by the B&#780;azant&ndash;Oh crack-band fracture energy formulation (G&#7781;&#8344;). "
                "Residual stiffness &#945;&#7523;&#7497;&#7515; = 10&#8315;&#8310;.",
            ),
            (
                "ITZ Interfaces",
                "4-node zero-thickness cohesive elements. Bilinear traction-separation law "
                "governed by Mode-I normal opening: penalty stiffness k&#8345;, tensile strength "
                "f&#8348;&#7488;&#7511;&#7503;, fracture energy G&#7781;&#7488;&#7511;&#7503;. "
                "Tangential shear stiffness degrades proportionally with opening damage.",
            ),
            (
                "Kinematics & Elements",
                "3-node Constant Strain Triangle (CST) elements, plane-stress assumption, "
                "unit out-of-plane thickness t = 1.0 mm (default).",
            ),
            (
                "Nonlinear Solver",
                "Incremental-iterative Newton-Raphson with displacement control, automatic "
                "load-step cutbacks, and isolated trial state tracking for irreversible damage.",
            ),
            (
                "Convergence Criteria",
                "Dual-criterion acceptance: strict force residual ‖r‖₂ < 10⁻² N, "
                "OR nanoscale displacement correction ‖Δu‖₂ < 10⁻⁶ mm "
                "with relaxed force bound 10 × force_tol. "
                "Increments meeting the latter are explicitly documented.",
            ),
        ]

        ov_html = '<div class="overview-grid">'

        for title, text in overview_items:
            ov_html += f"""
            <div class="overview-card">
                <div class="overview-card-title">{title}</div>
                <div class="overview-card-text">{text}</div>
            </div>
            """

        ov_html += "</div>"
        st.markdown(ov_html, unsafe_allow_html=True)

    with st.expander("Scientific scope, limitations & disclaimer"):
        st.markdown(
            """
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
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # ------------------------------------------------------------------
    # DEVELOPERS & INSTITUTION
    # ------------------------------------------------------------------

    st.markdown('<div class="anchor" id="team"></div>', unsafe_allow_html=True)
    st.subheader("Developers & Academic Information")
    st.caption(
        "Undergraduate minor project — Department of Civil Engineering, 7th Semester:"
    )

    devs = [
        ("Sanidhya Somawar", "SS", "avatar-blue"),
        ("Onkar Prasad Patel", "OP", "avatar-indigo"),
        ("Priyaranjan Pandey", "PP", "avatar-teal"),
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
        </div>
        """

    dev_html += "</div>"
    st.markdown(dev_html, unsafe_allow_html=True)

    st.markdown(
        """
    <div class="institution-block">
        <p>
            <strong>Institution:</strong> Government Engineering College, Bilaspur<br>
            <strong>Department:</strong> Department of Civil Engineering<br>
            <strong>Semester:</strong> 7th Semester &mdash; B.Tech Minor Project<br>
            <strong>Location:</strong> Bilaspur, Chhattisgarh, India
        </p>
    </div>
        """,
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------
    # BOTTOM CTA
    # ------------------------------------------------------------------

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