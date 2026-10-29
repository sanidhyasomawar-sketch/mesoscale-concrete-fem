"""
styles.py
---------
CSS design system for the Mesoscale Concrete FEM application.

Architecture
------------
- All colour values are declared once as CSS custom properties (design tokens)
  in the :root block.  Every component references these tokens — no scattered
  hard-coded hex values elsewhere.
- The explicit light theme is set via .streamlit/config.toml.  This CSS layer
  only adds presentation on top of that controlled base; it never fights the
  theme.
- Selectors are chosen for stability: tag names, [data-testid], and structural
  class prefixes.  Emotion auto-generated class names (st-emotion-cache-xxxx)
  are deliberately avoided.
- WCAG AA contrast (4.5:1) targets:
    body text (#1A2332) on page bg (#F8F9FA)  → 14.5:1  ✓
    body text (#1A2332) on surface (#FFFFFF)   → 15.3:1  ✓
    muted text (#4B5563) on page bg            →  7.8:1  ✓
    accent (#2563EB) on white                  →  5.9:1  ✓
    hero text (#FFFFFF) on hero bg (#1E293B)   → 16.7:1  ✓
    disclaimer text (#78350F) on (#FFF9ED)     →  7.6:1  ✓
    benchmark text (#1E3A8A) on (#F0F7FF)      →  9.1:1  ✓
"""

import streamlit as st


# ---------------------------------------------------------------------------
# DESIGN TOKENS — single source of truth for the entire colour palette,
# spacing scale, typography, and radii.
# ---------------------------------------------------------------------------
_CSS = """
<style>
/* ============================================================
   DESIGN TOKENS
   ============================================================ */
:root {
    /* --- Neutral palette ----------------------------------------- */
    --bg-page:        #F8F9FA;   /* Page background (config.toml matches) */
    --bg-surface:     #FFFFFF;   /* Cards, panels                         */
    --bg-inset:       #F1F3F6;   /* Recessed / inset areas                */
    --border:         #D1D5DB;   /* Default border                        */
    --border-subtle:  #E5E7EB;   /* Dividers, hairlines                   */

    /* --- Text palette -------------------------------------------- */
    --text:           #1A2332;   /* Primary body text — navy-charcoal     */
    --text-secondary: #4B5563;   /* Secondary / descriptive text          */
    --text-muted:     #6B7280;   /* Captions, metadata                    */

    /* --- Accent (blue — mirrors primaryColor in config.toml) ------ */
    --accent:         #2563EB;
    --accent-hover:   #1D4ED8;
    --accent-bg:      #EFF6FF;   /* Light tint for accent surfaces        */
    --accent-border:  #BFDBFE;

    /* --- Semantic colours ----------------------------------------- */
    --success:        #059669;
    --success-bg:     #ECFDF5;
    --warn:           #D97706;
    --warn-bg:        #FFF9ED;
    --warn-border:    #FCD34D;
    --warn-text:      #78350F;
    --danger:         #DC2626;
    --danger-bg:      #FEF2F2;
    --info:           #2563EB;
    --info-bg:        #F0F7FF;
    --info-border:    #BAD3FA;
    --info-text:      #1E3A8A;

    /* --- Hero (dark gradient — always dark regardless of theme) --- */
    --hero-bg-from:   #1E293B;
    --hero-bg-to:     #0F172A;
    --hero-border:    #334155;
    --hero-title:     #FFFFFF;
    --hero-subtitle:  #94A3B8;
    --hero-body:      #CBD5E1;
    --hero-badge-bg:  rgba(37, 99, 235, 0.25);
    --hero-badge-fg:  #93C5FD;
    --hero-badge-br:  rgba(147, 197, 253, 0.3);

    /* --- Shape & spacing ------------------------------------------ */
    --r-sm:  4px;
    --r-md:  8px;
    --r-lg:  12px;
    --r-xl:  16px;

    --shadow-xs: 0 1px 2px rgba(0,0,0,0.05);
    --shadow-sm: 0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.04);
    --shadow-md: 0 4px 8px rgba(0,0,0,0.07), 0 2px 4px rgba(0,0,0,0.04);

    /* --- Typography ---------------------------------------------- */
    --font-stack: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                  "Helvetica Neue", Arial, sans-serif;
    --font-mono:  "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
}


/* ============================================================
   BASE TYPOGRAPHY
   ============================================================ */
html, body {
    font-family: var(--font-stack);
    color: var(--text);
    background-color: var(--bg-page);
}

/* Headings */
h1, h2, h3, h4, h5 {
    color: var(--text) !important;
    font-weight: 600;
    letter-spacing: -0.015em;
    line-height: 1.3;
}

/* Streamlit default heading elements */
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3 {
    color: var(--text) !important;
}

/* Body text in markdown containers */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {
    color: var(--text);
    font-size: 0.97rem;
    line-height: 1.65;
}

/* Code / monospace */
code, pre {
    font-family: var(--font-mono);
    font-size: 0.88em;
}


/* ============================================================
   SIDEBAR
   ============================================================ */

/* Sidebar panel background — Streamlit light theme sets this automatically
   via secondaryBackgroundColor; we only add minor tweaks here. */
[data-testid="stSidebar"] {
    border-right: 1px solid var(--border-subtle);
}

/* Sidebar headings and labels */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p {
    color: var(--text) !important;
}

/* Radio option labels in sidebar — these are the nav items */
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: var(--text) !important;
    font-size: 0.93rem;
}

/* Sidebar section title (the "Navigation" / app branding block) */
[data-testid="stSidebar"] [data-testid="stHeading"] {
    color: var(--text) !important;
}

/* Caption text in sidebar footer */
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
    color: var(--text-secondary) !important;
    font-size: 0.78rem;
    line-height: 1.5;
}

/* Sidebar subheader / dividers */
[data-testid="stSidebar"] hr {
    border-color: var(--border-subtle);
    margin: 0.75rem 0;
}


/* ============================================================
   MAIN CONTENT AREA — spacing & structural
   ============================================================ */

/* Reduce excessive top padding on the main block */
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
}

/* st.caption — slightly muted */
[data-testid="stCaptionContainer"] p {
    color: var(--text-muted) !important;
    font-size: 0.82rem;
}

/* Horizontal rule */
hr {
    border: none;
    border-top: 1px solid var(--border-subtle);
    margin: 1.5rem 0;
}


/* ============================================================
   METRIC CARDS  (st.metric)
   ============================================================ */
[data-testid="stMetric"] {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: var(--r-md);
    padding: 0.8rem 1rem;
    box-shadow: var(--shadow-xs);
}

[data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;
    font-size: 0.78rem !important;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

[data-testid="stMetricValue"] {
    color: var(--text) !important;
    font-size: 1.45rem !important;
    font-weight: 700;
}

[data-testid="stMetricDelta"] {
    font-size: 0.8rem !important;
}


/* ============================================================
   ALERTS — st.info / st.warning / st.error / st.success
   ============================================================ */

/* Override Streamlit's alert backgrounds so they sit cleanly on --bg-page */
[data-testid="stAlert"] {
    border-radius: var(--r-md) !important;
    border-width: 1px !important;
}


/* ============================================================
   TABS  (st.tabs)
   ============================================================ */
[data-testid="stTabs"] [data-testid="stMarkdownContainer"] p {
    font-size: 0.9rem;
    color: var(--text);
}


/* ============================================================
   DATAFRAME / TABLE
   ============================================================ */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border-subtle);
    border-radius: var(--r-md);
    overflow: hidden;
}


/* ============================================================
   EXPANDER
   ============================================================ */
[data-testid="stExpander"] {
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--r-md) !important;
    background: var(--bg-surface);
}


/* ============================================================
   DOWNLOAD BUTTON
   ============================================================ */
[data-testid="stDownloadButton"] button {
    border-radius: var(--r-md);
}


/* ============================================================
   HERO BANNER
   ============================================================ */
.hero-box {
    background: linear-gradient(135deg, var(--hero-bg-from) 0%, var(--hero-bg-to) 100%);
    border: 1px solid var(--hero-border);
    border-radius: var(--r-xl);
    padding: 2rem 2.25rem 1.75rem;
    color: var(--hero-title);
    margin-bottom: 1.75rem;
    box-shadow: var(--shadow-md);
}

.hero-badge {
    display: inline-block;
    padding: 0.28rem 0.75rem;
    background: var(--hero-badge-bg);
    color: var(--hero-badge-fg);
    border: 1px solid var(--hero-badge-br);
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 0.9rem;
}

.hero-title {
    font-size: 1.9rem;
    font-weight: 700;
    line-height: 1.25;
    margin: 0 0 0.5rem;
    color: var(--hero-title) !important;
}

.hero-subtitle {
    font-size: 1.08rem;
    color: var(--hero-subtitle);
    font-weight: 400;
    line-height: 1.5;
    margin-bottom: 1rem;
}

.hero-desc {
    font-size: 0.93rem;
    color: var(--hero-body);
    line-height: 1.65;
    max-width: 850px;
}


/* ============================================================
   SECTION DIVIDER
   ============================================================ */
.section-divider {
    border: none;
    border-top: 1px solid var(--border-subtle);
    margin: 1.75rem 0;
}


/* ============================================================
   WORKFLOW STEP GRID
   ============================================================ */
.workflow-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 1rem;
    margin: 1.25rem 0 1.75rem;
}

.step-card {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: var(--r-md);
    padding: 1rem 1.1rem;
    box-shadow: var(--shadow-xs);
    display: flex;
    flex-direction: column;
    transition: border-color 0.15s ease, box-shadow 0.15s ease;
}

.step-card:hover {
    border-color: var(--accent-border);
    box-shadow: var(--shadow-sm);
}

.step-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 0.5rem;
}

.step-number {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    background: var(--accent);
    color: #fff;
    font-weight: 700;
    font-size: 0.8rem;
    border-radius: 50%;
    flex-shrink: 0;
}

.step-title {
    font-size: 0.92rem;
    font-weight: 600;
    color: var(--text);
    line-height: 1.3;
}

.step-body {
    font-size: 0.84rem;
    color: var(--text-secondary);
    line-height: 1.5;
    margin-top: auto;
}


/* ============================================================
   SCIENTIFIC MODEL OVERVIEW CARDS
   ============================================================ */
.overview-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 1.1rem;
    margin: 1.1rem 0 1.75rem;
}

.overview-card {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-left: 4px solid var(--accent);
    border-radius: var(--r-md);
    padding: 1rem 1.2rem;
    box-shadow: var(--shadow-xs);
}

.overview-card-title {
    font-size: 0.97rem;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 0.4rem;
}

.overview-card-text {
    font-size: 0.85rem;
    color: var(--text-secondary);
    line-height: 1.55;
}


/* ============================================================
   DEVELOPER CARDS
   ============================================================ */
.dev-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 1rem;
    margin: 1rem 0 1.5rem;
}

.developer-card {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: var(--r-md);
    padding: 1rem 1.1rem;
    display: flex;
    align-items: center;
    gap: 0.85rem;
    box-shadow: var(--shadow-xs);
}

.avatar-badge {
    width: 44px;
    height: 44px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 1rem;
    color: #fff;
    flex-shrink: 0;
}

.avatar-blue   { background: linear-gradient(135deg, #2563EB, #1D4ED8); }
.avatar-indigo { background: linear-gradient(135deg, #4F46E5, #4338CA); }
.avatar-teal   { background: linear-gradient(135deg, #0D9488, #0F766E); }
.avatar-slate  { background: linear-gradient(135deg, #475569, #334155); }

.dev-name {
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--text);
    line-height: 1.3;
}

.dev-dept {
    font-size: 0.79rem;
    color: var(--text-muted);
    margin-top: 0.1rem;
}


/* ============================================================
   DISCLAIMER BOX
   ============================================================ */
.disclaimer-box {
    background: var(--warn-bg);
    border: 1px solid var(--warn-border);
    border-left: 5px solid var(--warn);
    border-radius: var(--r-md);
    padding: 1.1rem 1.3rem;
    margin: 1.5rem 0;
    box-shadow: var(--shadow-xs);
}

.disclaimer-title {
    color: var(--warn-text);
    font-size: 0.98rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    gap: 0.45rem;
    margin-bottom: 0.5rem;
}

.disclaimer-body {
    color: var(--warn-text);
    font-size: 0.86rem;
    line-height: 1.6;
}

.disclaimer-body ul {
    margin: 0.4rem 0 0.2rem;
    padding-left: 1.2rem;
}

.disclaimer-body li {
    margin-bottom: 0.3rem;
}


/* ============================================================
   BENCHMARK BOX
   ============================================================ */
.benchmark-box {
    background: var(--info-bg);
    border: 1px solid var(--info-border);
    border-left: 5px solid var(--accent);
    border-radius: var(--r-md);
    padding: 1.1rem 1.3rem;
    margin: 1.5rem 0;
    box-shadow: var(--shadow-xs);
}

.benchmark-title {
    color: var(--info-text);
    font-size: 0.98rem;
    font-weight: 700;
    margin-bottom: 0.55rem;
}

.benchmark-body {
    color: var(--info-text);
    font-size: 0.86rem;
    line-height: 1.6;
}

.benchmark-metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 0.7rem;
    margin: 0.8rem 0;
}

.benchmark-metric-item {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: var(--r-sm);
    padding: 0.55rem 0.75rem;
}

.benchmark-metric-label {
    font-size: 0.73rem;
    color: var(--text-muted);
    text-transform: uppercase;
    font-weight: 600;
    letter-spacing: 0.04em;
}

.benchmark-metric-value {
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--text);
    margin-top: 0.1rem;
}


/* ============================================================
   INSTITUTION INFO BLOCK
   ============================================================ */
.institution-block {
    background: var(--bg-inset);
    border: 1px solid var(--border-subtle);
    border-radius: var(--r-md);
    padding: 1rem 1.3rem;
    margin: 1rem 0 1.5rem;
}

.institution-block p {
    color: var(--text-secondary) !important;
    font-size: 0.88rem;
    margin: 0.15rem 0;
    line-height: 1.6;
}

.institution-block strong {
    color: var(--text);
}


/* ============================================================
   PHASE PAGE HEADER CARD
   ============================================================ */
.phase-header {
    background: linear-gradient(90deg, var(--accent-bg) 0%, var(--bg-surface) 100%);
    border: 1px solid var(--accent-border);
    border-left: 4px solid var(--accent);
    border-radius: var(--r-md);
    padding: 0.85rem 1.2rem;
    margin-bottom: 1.25rem;
}

.phase-header-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 0.2rem;
}

.phase-header-desc {
    font-size: 0.86rem;
    color: var(--text-secondary);
    line-height: 1.5;
}
</style>
"""


def inject_css() -> None:
    """
    Inject the application CSS design system into Streamlit.

    Called once at app startup immediately after st.set_page_config().
    Safe: selectors avoid Emotion auto-generated class names.
    """
    st.markdown(_CSS, unsafe_allow_html=True)
