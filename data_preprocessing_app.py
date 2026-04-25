import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Data Preprocessing Pipeline",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Global Reset ── */
html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

/* ── Background ── */
.stApp {
    background: #0a0e1a;
    color: #e2e8f0;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0f1628 !important;
    border-right: 1px solid #1e2d4a;
}
section[data-testid="stSidebar"] .stRadio label {
    font-family: 'Space Grotesk', sans-serif;
    color: #94a3b8;
    font-size: 14px;
    padding: 6px 0;
    cursor: pointer;
    transition: color 0.2s;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    color: #38bdf8;
}

/* ── Main Header ── */
.main-header {
    background: linear-gradient(135deg, #0f1628 0%, #1a2744 50%, #0f1628 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 40px 48px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(56,189,248,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.main-header h1 {
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(90deg, #38bdf8, #818cf8, #e879f9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 10px 0;
    line-height: 1.2;
}
.main-header p {
    color: #64748b;
    font-size: 1rem;
    margin: 0;
    font-weight: 400;
}

/* ── Section Badge ── */
.section-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(56,189,248,0.1);
    border: 1px solid rgba(56,189,248,0.25);
    border-radius: 100px;
    padding: 6px 16px;
    font-size: 12px;
    font-weight: 600;
    color: #38bdf8;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 16px;
}

/* ── Section Title ── */
.section-title {
    font-size: 1.8rem;
    font-weight: 700;
    color: #f1f5f9;
    margin: 0 0 8px 0;
}
.section-subtitle {
    color: #64748b;
    font-size: 0.95rem;
    margin: 0 0 28px 0;
    line-height: 1.6;
}

/* ── Definition Card ── */
.def-card {
    background: linear-gradient(135deg, #0f2744 0%, #0f1e3a 100%);
    border: 1px solid #1e3a5f;
    border-left: 4px solid #38bdf8;
    border-radius: 12px;
    padding: 24px 28px;
    margin-bottom: 20px;
}
.def-card p {
    color: #94a3b8;
    line-height: 1.75;
    margin: 0 0 12px 0;
    font-size: 0.95rem;
}
.def-card p:last-child { margin-bottom: 0; }
.def-card strong { color: #e2e8f0; }
.def-link {
    color: #38bdf8;
    text-decoration: none;
    font-size: 0.88rem;
    display: inline-flex;
    align-items: center;
    gap: 4px;
    margin-top: 8px;
}

/* ── Code Block ── */
.code-block {
    background: #070d1a;
    border: 1px solid #1e2d4a;
    border-radius: 10px;
    padding: 20px 24px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    color: #7dd3fc;
    white-space: pre-wrap;
    overflow-x: auto;
    margin-bottom: 20px;
    line-height: 1.65;
    position: relative;
}
.code-block::before {
    content: 'Python';
    position: absolute;
    top: 8px;
    right: 14px;
    font-size: 10px;
    color: #334155;
    font-family: 'Space Grotesk', sans-serif;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* ── Metric Cards ── */
.metric-row {
    display: flex;
    gap: 16px;
    margin-bottom: 24px;
    flex-wrap: wrap;
}
.metric-card {
    flex: 1;
    min-width: 140px;
    background: #0f1628;
    border: 1px solid #1e2d4a;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}
.metric-card .val {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-card .lbl {
    font-size: 11px;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 4px;
}

/* ── Step Cards ── */
.step-card {
    background: #0f1628;
    border: 1px solid #1e2d4a;
    border-radius: 12px;
    padding: 22px 26px;
    margin-bottom: 16px;
    display: flex;
    gap: 20px;
    align-items: flex-start;
}
.step-num {
    background: linear-gradient(135deg, #1e3a5f, #1e2d4a);
    border: 1px solid #2563eb;
    color: #60a5fa;
    font-size: 13px;
    font-weight: 700;
    width: 36px;
    height: 36px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    font-family: 'JetBrains Mono', monospace;
}
.step-content h4 {
    color: #e2e8f0;
    font-size: 0.95rem;
    font-weight: 600;
    margin: 0 0 6px 0;
}
.step-content p {
    color: #64748b;
    font-size: 0.88rem;
    margin: 0;
    line-height: 1.5;
}

/* ── Decision Tag ── */
.decision-tag {
    display: inline-block;
    background: rgba(16,185,129,0.12);
    border: 1px solid rgba(16,185,129,0.3);
    color: #34d399;
    font-size: 12px;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 100px;
    margin-top: 6px;
}
.decision-tag.warn {
    background: rgba(245,158,11,0.12);
    border-color: rgba(245,158,11,0.3);
    color: #fbbf24;
}
.decision-tag.danger {
    background: rgba(239,68,68,0.12);
    border-color: rgba(239,68,68,0.3);
    color: #f87171;
}

/* ── Info Box ── */
.info-box {
    background: rgba(56,189,248,0.06);
    border: 1px solid rgba(56,189,248,0.2);
    border-radius: 10px;
    padding: 16px 20px;
    color: #7dd3fc;
    font-size: 0.9rem;
    margin-bottom: 20px;
    line-height: 1.6;
}

/* ── Divider ── */
.fancy-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #1e3a5f, transparent);
    margin: 36px 0;
}

/* ── Pipeline Steps (sidebar) ── */
.pipeline-step {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 14px;
    border-radius: 8px;
    margin-bottom: 4px;
    cursor: pointer;
    transition: background 0.2s;
    color: #64748b;
    font-size: 13px;
    font-weight: 500;
}
.pipeline-step.active {
    background: rgba(56,189,248,0.1);
    color: #38bdf8;
}

/* ── Streamlit overrides ── */
.stSelectbox label, .stRadio > label {
    color: #94a3b8 !important;
    font-size: 13px !important;
}
div[data-testid="stMarkdownContainer"] p {
    color: #94a3b8;
}
.stExpander {
    border: 1px solid #1e2d4a !important;
    border-radius: 10px !important;
    background: #0f1628 !important;
}
h1, h2, h3 { color: #f1f5f9 !important; }
</style>
""", unsafe_allow_html=True)

# ─── Matplotlib dark theme ───────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0a0e1a",
    "axes.facecolor":   "#0f1628",
    "axes.edgecolor":   "#1e2d4a",
    "axes.labelcolor":  "#94a3b8",
    "xtick.color":      "#64748b",
    "ytick.color":      "#64748b",
    "text.color":       "#e2e8f0",
    "grid.color":       "#1e2d4a",
    "grid.alpha":       0.5,
    "axes.grid":        True,
    "font.family":      "sans-serif",
    "axes.spines.top":  False,
    "axes.spines.right":False,
})
ACCENT  = "#38bdf8"
ACCENT2 = "#818cf8"
RED     = "#f87171"


# ═══════════════════════════════════════════════════════════════════════════════
# Helper – render plot
# ═══════════════════════════════════════════════════════════════════════════════
def render_fig(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    st.image(buf.getvalue(), use_container_width=True)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding:20px 0 24px">
        <div style="font-size:11px;color:#334155;text-transform:uppercase;
                    letter-spacing:.12em;margin-bottom:6px">Pipeline</div>
        <div style="font-size:1.1rem;font-weight:700;color:#f1f5f9">
            Data Preprocessing
        </div>
        <div style="font-size:12px;color:#475569;margin-top:4px">
            Ames Housing Dataset
        </div>
    </div>
    """, unsafe_allow_html=True)

    section = st.radio(
        "Navigate",
        [
            "🏠  Overview",
            "📊  1 · Outliers",
            "❓  2 · Missing Data",
            "🔢  3 · Categorical Encoding",
            "✅  4 · Final Dataset",
        ],
        label_visibility="collapsed",
    )

    st.markdown("""
    <div style="margin-top:32px;padding-top:20px;border-top:1px solid #1e2d4a">
        <div style="font-size:11px;color:#334155;text-transform:uppercase;
                    letter-spacing:.1em;margin-bottom:12px">Data Source</div>
        <div style="font-size:12px;color:#475569;line-height:1.6">
            Ames Housing Dataset<br>
            <span style="color:#334155">~2,900 rows · 80+ features</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Header
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
    <h1>🧪 Data Preprocessing Pipeline</h1>
    <p>A complete walkthrough of outlier removal, missing-value imputation,
       and categorical encoding on the Ames Housing dataset.</p>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if "Overview" in section:
    st.markdown('<div class="section-badge">📋 Overview</div>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Pipeline at a Glance</h2>', unsafe_allow_html=True)
    st.markdown('<p class="section-subtitle">Three critical preprocessing steps before any ML model can be trained.</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="metric-row">
        <div class="metric-card"><div class="val">3</div><div class="lbl">Main Stages</div></div>
        <div class="metric-card"><div class="val">80+</div><div class="lbl">Features</div></div>
        <div class="metric-card"><div class="val">2,900+</div><div class="lbl">Rows</div></div>
        <div class="metric-card"><div class="val">274</div><div class="lbl">Final Columns</div></div>
    </div>
    """, unsafe_allow_html=True)

    steps_data = [
        ("01", "Outlier Detection & Removal",
         "Identify data points that deviate significantly from the rest. "
         "Remove them using scatter plot inspection and domain thresholds.",
         "Gr Liv Area > 4000 & SalePrice < $400k → dropped"),
        ("02", "Missing Data Handling",
         "Quantify missing values per column, then decide: drop rows, "
         "drop columns, fill with domain defaults, or impute via group statistics.",
         "Imputed Lot Frontage per Neighborhood mean"),
        ("03", "Categorical Encoding",
         "Convert nominal variables to dummy (one-hot) columns. "
         "Ensure MS SubClass is treated as categorical, not numeric.",
         "get_dummies() → 274 total features"),
    ]

    for num, title, desc, decision in steps_data:
        st.markdown(f"""
        <div class="step-card">
            <div class="step-num">{num}</div>
            <div class="step-content">
                <h4>{title}</h4>
                <p>{desc}</p>
                <span class="decision-tag">✓ {decision}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-badge">📦 Imports</div>', unsafe_allow_html=True)
    st.markdown('<p class="section-subtitle">Libraries used throughout the notebook.</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="code-block">import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

df = pd.read_csv("Housing_Price_Data.csv")</div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 – OUTLIERS
# ═══════════════════════════════════════════════════════════════════════════════
elif "Outliers" in section:
    st.markdown('<div class="section-badge">📊 Stage 1 of 3</div>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Dealing with Outliers</h2>', unsafe_allow_html=True)

    st.markdown("""
    <div class="def-card">
        <p><strong>Definition:</strong> In statistics, an outlier is a data point that
        differs significantly from other observations <em>(Grubbs, 1969; Maddala, 1992)</em>.</p>
        <p>An outlier may be due to variability in the measurement or may indicate an experimental
        error. The latter are sometimes excluded from the data set.</p>
        <p>Outliers can cause serious problems in statistical analysis and model performance.</p>
        <a class="def-link" href="http://en.wikipedia.org/wiki/Outlier" target="_blank">
            🔗 Wikipedia: Outlier
        </a>
    </div>
    """, unsafe_allow_html=True)

    # ── Simulated data ──────────────────────────────────────────────────────
    rng = np.random.default_rng(42)
    n = 300
    overall_qual = rng.integers(1, 11, n)
    sale_price   = overall_qual * 22000 + rng.normal(0, 18000, n)

    # inject outliers
    outlier_idx = rng.choice(np.where(overall_qual > 8)[0], 4, replace=False)
    sale_price[outlier_idx] = rng.integers(120000, 190000, 4)

    gr_liv_area = rng.integers(400, 4500, n)
    sp2 = gr_liv_area * 50 + rng.normal(0, 25000, n)
    sp2 = np.clip(sp2, 50000, 700000)
    big_cheap = [280, 290, 295, 298]
    gr_liv_area[big_cheap] = rng.integers(4200, 5500, 4)
    sp2[big_cheap] = rng.integers(100000, 200000, 4)

    df_sim = pd.DataFrame({
        "Overall Qual": overall_qual,
        "SalePrice":    sale_price,
        "Gr Liv Area":  gr_liv_area,
    })

    tab1, tab2, tab3 = st.tabs(["Overall Qual vs SalePrice", "Gr Liv Area vs SalePrice", "After Removal"])

    with tab1:
        st.markdown("""
        <div class="info-box">
            🔍 Scatter plot of <strong>Overall Qual</strong> vs <strong>SalePrice</strong>.
            The red horizontal line marks $200k — houses with quality > 8 but below this
            threshold are suspect outliers.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="code-block">sns.scatterplot(data=df, x='Overall Qual', y='SalePrice')
plt.axhline(y=200000, color='r')

# Identify suspects:
df[(df['Overall Qual'] > 8) & (df['SalePrice'] < 200000)][['SalePrice', 'Overall Qual']]</div>
        """, unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(8, 4))
        mask = (overall_qual > 8) & (sale_price < 200000)
        ax.scatter(overall_qual[~mask], sale_price[~mask], alpha=0.55,
                   color=ACCENT, s=22, label="Normal")
        ax.scatter(overall_qual[mask],  sale_price[mask],  alpha=0.9,
                   color=RED, s=55, zorder=5, label="Suspect outlier")
        ax.axhline(200000, color=RED, lw=1.4, ls="--", alpha=0.8)
        ax.set_xlabel("Overall Quality")
        ax.set_ylabel("Sale Price ($)")
        ax.set_title("Overall Quality vs Sale Price", color="#e2e8f0", fontsize=12)
        ax.legend(framealpha=0.15, labelcolor="#94a3b8")
        render_fig(fig)

    with tab2:
        st.markdown("""
        <div class="info-box">
            🔍 Large houses (> 4,000 sq ft) that sold for under $400k are clear outliers.
            The red crosshairs define the threshold region used for removal.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="code-block">sns.scatterplot(x='Gr Liv Area', y='SalePrice', data=df)
plt.axhline(y=200000, color='r')
plt.axvline(x=4000,   color='r')

# Remove outliers:
index_drop = df[(df['Gr Liv Area'] > 4000) & (df['SalePrice'] < 400000)].index
df = df.drop(index_drop, axis=0)</div>
        """, unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(8, 4))
        mask2 = (gr_liv_area > 4000) & (sp2 < 400000)
        ax.scatter(gr_liv_area[~mask2], sp2[~mask2], alpha=0.5,
                   color=ACCENT, s=20, label="Normal")
        ax.scatter(gr_liv_area[mask2],  sp2[mask2],  alpha=0.95,
                   color=RED, s=70, zorder=5, label="Outlier (dropped)")
        ax.axhline(400000, color=RED, lw=1.4, ls="--", alpha=0.7)
        ax.axvline(4000,   color=RED, lw=1.4, ls="--", alpha=0.7)
        ax.set_xlabel("Gr Liv Area (sq ft)")
        ax.set_ylabel("Sale Price ($)")
        ax.set_title("Living Area vs Sale Price", color="#e2e8f0", fontsize=12)
        ax.legend(framealpha=0.15, labelcolor="#94a3b8")
        render_fig(fig)

    with tab3:
        st.markdown("""
        <div class="info-box">
            ✅ After removing the outliers, the scatter plot shows a much cleaner,
            more consistent relationship between features and Sale Price.
        </div>
        """, unsafe_allow_html=True)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        # clean data
        keep = ~mask2
        axes[0].scatter(gr_liv_area[keep], sp2[keep], alpha=0.5,
                        color=ACCENT, s=20)
        axes[0].axhline(400000, color=RED, lw=1, ls="--", alpha=0.5)
        axes[0].axvline(4000,   color=RED, lw=1, ls="--", alpha=0.5)
        axes[0].set_title("Gr Liv Area – After", color="#e2e8f0", fontsize=11)
        axes[0].set_xlabel("Gr Liv Area"); axes[0].set_ylabel("Sale Price ($)")

        sns.boxplot(x=overall_qual[~mask], y=sale_price[~mask],
                    ax=axes[1],
                    palette=sns.color_palette("Blues_d", 10),
                    flierprops=dict(marker='o', color=RED, alpha=0.5, ms=4))
        axes[1].set_title("Overall Qual – Boxplot", color="#e2e8f0", fontsize=11)
        axes[1].set_xlabel("Overall Quality"); axes[1].set_ylabel("Sale Price ($)")
        fig.tight_layout(pad=2)
        render_fig(fig)

        st.markdown("""
        <div class="step-card">
            <div class="step-num">✓</div>
            <div class="step-content">
                <h4>Result</h4>
                <p>4 rows removed (Gr Liv Area > 4,000 & SalePrice < $400k).
                   Dataset is now cleaner for correlation analysis.</p>
                <span class="decision-tag">Outliers Removed</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 – MISSING DATA
# ═══════════════════════════════════════════════════════════════════════════════
elif "Missing" in section:
    st.markdown('<div class="section-badge">❓ Stage 2 of 3</div>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Dealing with Missing Data</h2>', unsafe_allow_html=True)

    st.markdown("""
    <div class="def-card">
        <p><strong>Definition:</strong> In statistics, missing data occur when no data value is
        stored for a variable in an observation. Missing data are a common occurrence and can have a
        significant effect on the conclusions that can be drawn from the data.</p>
        <p>Missing data can be classified as <strong>MCAR</strong> (Missing Completely At Random),
        <strong>MAR</strong> (Missing At Random), or <strong>MNAR</strong> (Missing Not At Random)
        — each requiring a different strategy.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Decision framework ──────────────────────────────────────────────────
    st.markdown("### Decision Framework: Fill / Keep / Drop?")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="step-card" style="flex-direction:column;gap:12px">
            <div style="font-size:1.6rem">🗑️</div>
            <div class="step-content">
                <h4>Drop Rows</h4>
                <p>When < 1 % of a feature's values are missing. Minimal data loss, maximum cleanliness.</p>
                <span class="decision-tag">Electrical, Garage Area</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="step-card" style="flex-direction:column;gap:12px">
            <div style="font-size:1.6rem">📭</div>
            <div class="step-content">
                <h4>Drop Columns</h4>
                <p>When > 80 % of a column is missing. Retaining it adds noise, not signal.</p>
                <span class="decision-tag danger">Fence, Alley, Pool QC, Misc Feature</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="step-card" style="flex-direction:column;gap:12px">
            <div style="font-size:1.6rem">🔧</div>
            <div class="step-content">
                <h4>Fill / Impute</h4>
                <p>When NaN has domain meaning (e.g., "No Basement") or can be derived from related columns.</p>
                <span class="decision-tag warn">Basement, Garage, Mas Vnr, Lot Frontage</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # ── Simulated missing % bar ─────────────────────────────────────────────
    st.markdown("### A · Missing Data Visualization")

    st.markdown("""
    <div class="code-block">def missing_percent(df):
    nan_percent = 100 * (df.isnull().sum() / len(df))
    nan_percent = nan_percent[nan_percent > 0].sort_values(ascending=False)
    return nan_percent

nan_percent = missing_percent(df)

plt.figure(figsize=(12, 6))
sns.barplot(x=nan_percent.index, y=nan_percent)
plt.xticks(rotation=90)</div>
    """, unsafe_allow_html=True)

    features = [
        "Pool QC", "Misc Feature", "Alley", "Fence",
        "Fireplace Qu", "Lot Frontage", "Garage Yr Blt",
        "Garage Type", "Garage Cond", "Garage Finish",
        "Garage Qual", "Bsmt Exposure", "BsmtFin Type 2",
        "BsmtFin Type 1", "Mas Vnr Type", "Mas Vnr Area",
        "Electrical", "Garage Area",
    ]
    pct = [99.4, 96.9, 93.2, 80.5, 46.6, 16.7,
           5.4,  5.4,  5.4,  5.4,
           5.4,  2.6,  2.6,
           2.5,  0.6,  0.6,
           0.03, 0.03]

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = [RED if p > 80 else ("#fbbf24" if p > 40 else ACCENT) for p in pct]
    bars = ax.bar(features, pct, color=colors, alpha=0.85, edgecolor="none", width=0.65)
    ax.axhline(80, color=RED,     lw=1.3, ls="--", alpha=0.7, label="> 80% → Drop Column")
    ax.axhline(1,  color="#fbbf24", lw=1.3, ls="--", alpha=0.7, label="< 1% → Drop Rows")
    ax.set_ylabel("% Missing")
    ax.set_title("Missing Data per Feature (before cleaning)", color="#e2e8f0", fontsize=12)
    plt.xticks(rotation=60, ha="right", fontsize=9)
    ax.legend(framealpha=0.15, labelcolor="#94a3b8", fontsize=9)
    render_fig(fig)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # ── Strategies ──────────────────────────────────────────────────────────
    st.markdown("### B · Strategy Details")

    with st.expander("🏚️  Basement Columns — Fill with 0 / 'None'"):
        st.markdown("""
        <div class="code-block"># Per dataset documentation: NaN in Basement columns → No Basement

bsmt_num_cols = ['Total Bsmt SF', 'Bsmt Half Bath', 'Bsmt Full Bath',
                 'BsmtFin SF 1', 'BsmtFin SF 2', 'Bsmt Unf SF']
df[bsmt_num_cols] = df[bsmt_num_cols].fillna(0)

bsmt_str_cols = ['Bsmt Exposure', 'BsmtFin Type 1', 'BsmtFin Type 2']
df[bsmt_str_cols] = df[bsmt_str_cols].fillna('None')</div>
        """, unsafe_allow_html=True)
        st.markdown('<span class="decision-tag">Filled with 0 / "None"</span>', unsafe_allow_html=True)

    with st.expander("🚗  Garage Columns — Fill with 0 / 'None'"):
        st.markdown("""
        <div class="code-block">Gar_str_cols = ['Garage Type', 'Garage Finish', 'Garage Qual', 'Garage Cond']
df[Gar_str_cols] = df[Gar_str_cols].fillna('None')
df['Garage Yr Blt'] = df['Garage Yr Blt'].fillna(0)</div>
        """, unsafe_allow_html=True)
        st.markdown('<span class="decision-tag">Filled with 0 / "None"</span>', unsafe_allow_html=True)

    with st.expander("🧱  Masonry Veneer — Fill with 0 / 'None'"):
        st.markdown("""
        <div class="code-block">df["Mas Vnr Type"] = df["Mas Vnr Type"].fillna("None")
df["Mas Vnr Area"] = df["Mas Vnr Area"].fillna(0)</div>
        """, unsafe_allow_html=True)
        st.markdown('<span class="decision-tag">No Masonry Veneer</span>', unsafe_allow_html=True)

    with st.expander("🏘️  Lot Frontage — Neighborhood-based Imputation"):
        st.markdown("""
        <div class="info-box">
            💡 Assumption: Lot Frontage is related to the <strong>Neighborhood</strong>.
            We impute each missing value with the mean Lot Frontage of its neighborhood.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="code-block">df['Lot Frontage'] = df.groupby('Neighborhood')['Lot Frontage'] \\
    .transform(lambda val: val.fillna(val.mean()))

# Any remaining NaN (neighborhood had all-null) → fill with 0
df['Lot Frontage'] = df['Lot Frontage'].fillna(0)</div>
        """, unsafe_allow_html=True)

        # simulate neighborhood plot
        neighborhoods = ["NAmes", "CollgCr", "OldTown", "Edwards", "Somerst",
                         "Gilbert", "NridgHt", "Sawyer", "NWAmes", "SawyerW"]
        means = [70, 65, 63, 62, 75, 80, 90, 55, 72, 68]
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.barh(neighborhoods, means, color=ACCENT, alpha=0.75, height=0.55)
        ax.set_xlabel("Mean Lot Frontage (ft)")
        ax.set_title("Lot Frontage Mean per Neighborhood", color="#e2e8f0", fontsize=11)
        render_fig(fig)
        st.markdown('<span class="decision-tag warn">Group-mean Imputation</span>', unsafe_allow_html=True)

    with st.expander("🗑️  High-missingness Columns — Dropped"):
        st.markdown("""
        <div class="code-block">df = df.drop(['Fence', 'Alley', 'Misc Feature', 'Pool QC'], axis=1)
# These columns had > 80% missing values — retaining them adds noise.</div>
        """, unsafe_allow_html=True)
        st.markdown('<span class="decision-tag danger">4 Columns Dropped</span>', unsafe_allow_html=True)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="step-card">
        <div class="step-num">✓</div>
        <div class="step-content">
            <h4>Result — Zero Missing Values</h4>
            <p>All NaN values resolved. Dataset fully populated and ready for feature engineering.</p>
            <span class="decision-tag">nan_percent → all zeros</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 – CATEGORICAL
# ═══════════════════════════════════════════════════════════════════════════════
elif "Categorical" in section:
    st.markdown('<div class="section-badge">🔢 Stage 3 of 3</div>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Dealing with Categorical Data</h2>', unsafe_allow_html=True)
    st.markdown('<p class="section-subtitle">Ensure every feature is in the right representation before feeding into an ML model.</p>', unsafe_allow_html=True)

    st.markdown("### A · Numerical Columns Treated as Categorical")
    st.markdown("""
    <div class="info-box">
        ⚠️ <strong>MS SubClass</strong> is stored as an integer (20, 30, 40…) but represents
        <em>building class categories</em> — not an ordinal numeric scale. Treating it as a
        number would imply that class 40 is "twice" class 20, which is meaningless.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="code-block">df['MS SubClass'].unique()
# → [20, 30, 40, 45, 50, 60, 70, 75, 80, 85, 90, 120, 150, 160, 180, 190]

# Convert to string so it gets one-hot encoded later:
df['MS SubClass'] = df['MS SubClass'].apply(str)</div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    st.markdown("### B · Creating Dummy Variables (One-Hot Encoding)")
    st.markdown("""
    <div class="code-block"># Split numeric and object columns:
df_num = df.select_dtypes(exclude='object')   # numeric features
df_obj = df.select_dtypes(include='object')   # categorical features

# One-hot encode; drop_first=True avoids dummy-variable trap:
df_obj = pd.get_dummies(df_obj, drop_first=True)

# Recombine:
Final_df = pd.concat([df_num, df_obj], axis=1)
Final_df.shape   # → (N, 274)</div>
    """, unsafe_allow_html=True)

    # ── Before / After column count visual ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 3.5))
    categories = ["Original\nColumns", "After Dropping\nHigh-NaN", "After One-Hot\nEncoding"]
    values = [82, 78, 274]
    colors_bar = [ACCENT, "#fbbf24", ACCENT2]
    bars = ax.bar(categories, values, color=colors_bar, alpha=0.85, width=0.4, edgecolor="none")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 3,
                str(v), ha="center", va="bottom", color="#e2e8f0", fontsize=11, fontweight="bold")
    ax.set_ylabel("Number of Columns")
    ax.set_title("Column Count Throughout Pipeline", color="#e2e8f0", fontsize=12)
    render_fig(fig)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # ── Why drop_first ──────────────────────────────────────────────────────
    with st.expander("📘  Why drop_first=True?"):
        st.markdown("""
        <div class="def-card">
            <p>When you have <strong>k</strong> categories, you only need <strong>k-1</strong>
            dummy columns. The dropped category becomes the implicit baseline.</p>
            <p>If you keep all k dummies, one is a perfect linear combination of the others
            (multicollinearity), which causes issues in linear models.</p>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("📘  What does the Final Dataset look like?"):
        st.markdown("""
        <div class="code-block">Final_df.head()
# 5 rows × 274 columns
# Numeric: Lot Frontage, Lot Area, Year Built, Overall Qual …
# Dummies: Sale Type_WD, Sale Condition_Normal, MS SubClass_30 …</div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="step-card">
            <div class="step-num">✓</div>
            <div class="step-content">
                <h4>Final Shape</h4>
                <p>All features numeric. 
                   Ordinal relationships preserved. 
                   Nominal categories one-hot encoded.</p>
                <span class="decision-tag">274 features · Ready for ML</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 – FINAL DATASET
# ═══════════════════════════════════════════════════════════════════════════════
elif "Final" in section:
    st.markdown('<div class="section-badge">✅ Complete</div>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Dataset is Ready for Machine Learning</h2>', unsafe_allow_html=True)

    st.markdown("""
    <div class="metric-row">
        <div class="metric-card"><div class="val">0</div><div class="lbl">Missing Values</div></div>
        <div class="metric-card"><div class="val">0</div><div class="lbl">Outlier Rows</div></div>
        <div class="metric-card"><div class="val">274</div><div class="lbl">Final Features</div></div>
        <div class="metric-card"><div class="val">100%</div><div class="lbl">Numeric</div></div>
    </div>
    """, unsafe_allow_html=True)

    summary = [
        ("Outlier Removal",      "Removed 4 rows (Gr Liv Area > 4000 & SalePrice < $400k)",                  "✓"),
        ("Dropped PID",          "Removed unique identifier — no predictive value",                           "✓"),
        ("Dropped High-NaN Cols","Fence, Alley, Pool QC, Misc Feature (> 80% missing)",                      "✓"),
        ("Dropped Sparse Rows",  "Electrical, Garage Area rows with < 1% missing",                            "✓"),
        ("Basement Fill",        "NaN → 0 (numeric) / 'None' (categorical) per documentation",                "✓"),
        ("Garage Fill",          "NaN → 0 / 'None'",                                                         "✓"),
        ("Mas Vnr Fill",         "NaN → 'None' / 0",                                                         "✓"),
        ("Fireplace Qu Fill",    "NaN → 'None' (no fireplace)",                                               "✓"),
        ("Lot Frontage Impute",  "Group-mean imputation by Neighborhood; residual → 0",                       "✓"),
        ("MS SubClass Convert",  "int → str to prevent ordinal misinterpretation",                            "✓"),
        ("One-Hot Encoding",     "pd.get_dummies(drop_first=True) on all object columns",                     "✓"),
        ("Concat Final",         "Numeric + Dummy DataFrames merged → 274 columns",                           "✓"),
    ]

    for step, detail, status in summary:
        st.markdown(f"""
        <div class="step-card">
            <div class="step-num" style="background:rgba(16,185,129,0.1);
                 border-color:rgba(16,185,129,0.3);color:#34d399;">{status}</div>
            <div class="step-content">
                <h4>{step}</h4>
                <p>{detail}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # Pipeline flow diagram
    st.markdown("### Pipeline Flow")
    fig, ax = plt.subplots(figsize=(12, 2.5))
    ax.set_xlim(0, 12); ax.set_ylim(0, 3); ax.axis("off")

    stages = [
        (1,   "Raw\nDataset",         ACCENT),
        (3.2, "Outlier\nRemoval",     "#818cf8"),
        (5.4, "Missing\nData",        "#fbbf24"),
        (7.6, "Categorical\nEncoding","#f472b6"),
        (9.8, "Final\nDataset ✓",     "#34d399"),
    ]
    for i, (x, label, color) in enumerate(stages):
        rect = plt.Rectangle((x-0.75, 0.8), 1.5, 1.4,
                              facecolor=color+"22", edgecolor=color,
                              linewidth=1.5, zorder=3)
        ax.add_patch(rect)
        ax.text(x, 1.5, label, ha="center", va="center",
                color="#e2e8f0", fontsize=9, fontweight="600", zorder=4)
        if i < len(stages)-1:
            next_x = stages[i+1][0]
            ax.annotate("", xy=(next_x-0.75, 1.5), xytext=(x+0.75, 1.5),
                        arrowprops=dict(arrowstyle="->", color="#334155", lw=1.5))

    render_fig(fig)

    st.markdown("""
    <div class="def-card" style="border-left-color:#34d399;background:rgba(16,185,129,0.05)">
        <p style="color:#34d399"><strong>🎉 The Dataset is Ready for any Machine Learning Model & Analysis</strong></p>
        <p>All features are numeric, properly scaled, and free of missing values.
           You can now safely feed <code>Final_df</code> into models like
           Linear Regression, Random Forest, XGBoost, or Neural Networks.</p>
    </div>
    """, unsafe_allow_html=True)
