# ============================================================
# SpilloverAI / Zoonotic Spillover Predictor
# Streamlit Dashboard - High-End Editorial Travel Aesthetic
#
# Final research dashboard:
#   - Multi-district environmental monitoring
#   - Environmental Risk Index
#   - Verified KFD ML outbreak signal
#   - Climate / vegetation / forest analysis
#   - Model interpretation and methodology
#
# IMPORTANT:
# Environmental_Risk_Score != ML KFD outbreak probability
# ============================================================

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SpilloverAI | Editorial Intelligence",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 2. FIND PROJECT ROOT AUTOMATICALLY
# ============================================================

def find_project_root():
    current = Path(__file__).resolve().parent
    candidates = [current, *current.parents]
    for folder in candidates:
        if (folder / "data").exists() and (folder / "notebooks").exists():
            return folder
    return Path(__file__).resolve().parents[2]

ROOT_DIR = find_project_root()
DATA_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_MODEL_DIR = ROOT_DIR / "outputs" / "models"
LEGACY_MODEL_DIR = ROOT_DIR / "models"

# ============================================================
# 3. FILE PATHS
# ============================================================

DASHBOARD_DATA_PATH = DATA_DIR / "dashboard_environment_data.csv"
MASTER_FEATURES_PATH = DATA_DIR / "master_features.csv"
MODEL_PATH = OUTPUT_MODEL_DIR / "kfd_outbreak_model_v2.pkl"
MODEL_METADATA_PATH = OUTPUT_MODEL_DIR / "kfd_outbreak_model_v2_metadata.json"

# ============================================================
# 4. DISTRICT COORDINATES
# ============================================================

DISTRICT_COORDINATES = {
    "Shivamogga": (13.9299, 75.5681),
    "Uttara Kannada": (14.7937, 74.6869),
    "Chikkamagaluru": (13.3161, 75.7720),
    "Kodagu": (12.4244, 75.7382),
    "Wayanad": (11.6854, 76.1320),
    "Kannur": (11.8745, 75.3704),
    "North Goa": (15.4909, 73.8278),
    "South Goa": (15.1170, 74.1240),
    "Sindhudurg": (16.3492, 73.5594)
}

# ============================================================
# 5. CUSTOM CSS (EDITORIAL TRAVEL AESTHETIC)
# ============================================================

st.html(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    /* Typography & Global App Background */
    * { font-family: 'Plus Jakarta Sans', sans-serif !important; }
    .stApp {
        background-color: #F7F4EE; /* Warm bone / off-white */
    }

    /* Reduce upper blank space */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    /* Sidebar - Deep Slate Nav */
    [data-testid="stSidebar"] {
        background-color: #0E1113 !important;
        border-right: none;
    }
    [data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] label {
        color: #7B7D7D !important; /* Muted ash gray for subtext */
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Navigation - Minimalist vertical lists with slash prefixes */
    .stRadio div[role="radiogroup"] label {
        margin-bottom: 0.75rem;
        font-size: 0.95rem !important; 
        font-weight: 500 !important;
        text-transform: none;
        letter-spacing: normal;
    }

    /* Input Bars - Ultra minimalist bottom border */
    .stSelectbox div[data-baseweb="select"] {
        background: transparent !important;
        border: none !important;
        border-bottom: 1px solid #333333 !important;
        border-radius: 0 !important;
        box-shadow: none !important;
    }
    .stSelectbox div[data-baseweb="select"] * { color: #FFFFFF !important; }

    /* Pill Buttons */
    .stButton > button, .stDownloadButton > button {
        background-color: #14171A !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 9999px !important; /* Fully rounded pill */
        padding: 0.5rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        background-color: #333 !important;
        box-shadow: 0 4px 14px rgba(0,0,0,0.1);
    }

    /* Section Headings */
    .section-title {
        font-size: 1.75rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        color: #14171A;
        margin-top: 2.5rem;
        margin-bottom: 0.5rem;
        border-bottom: 1px solid #E5E0D8;
        padding-bottom: 0.5rem;
    }
    .section-subtitle {
        color: #7B7D7D;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
        line-height: 1.6;
    }

    /* Metrics & Cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.04em !important;
        color: #14171A !important;
    }
    [data-testid="stMetricLabel"] {
        color: #7B7D7D !important;
        font-weight: 500 !important;
    }
    
    .editorial-card {
        background: #FFFFFF;
        border-radius: 1.5rem; /* rounded-2xl to 3xl */
        padding: 1.75rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.03);
        height: 100%;
        border: 1px solid #FAF7F2;
    }

    /* Signal Indicators */
    .signal-high {
        background: #14171A; /* Solid black background */
        border-radius: 2rem;
        padding: 2.5rem;
        text-align: center;
        box-shadow: 0 12px 35px rgba(0,0,0,0.15);
    }
    .signal-high h2 {
        color: #FFFFFF !important;
        font-size: 3rem !important;
        font-weight: 800;
        letter-spacing: -0.05em;
        margin: 0;
    }
    .signal-low {
        background: #FFFFFF;
        border-radius: 2rem;
        padding: 2.5rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.03);
        border: 1px solid #E5E0D8;
    }
    .signal-low h2 {
        color: #14171A !important;
        font-size: 3rem !important;
        font-weight: 800;
        letter-spacing: -0.05em;
        margin: 0;
    }
    
    /* Methodology tags */
    .method-tag {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #7B7D7D;
        margin-bottom: 0.5rem;
        display: block;
    }
    </style>
    """
)

# ============================================================
# 6. LOAD DATA
# ============================================================

@st.cache_data
def load_environment_data():
    if DASHBOARD_DATA_PATH.exists():
        df = pd.read_csv(DASHBOARD_DATA_PATH)
    elif MASTER_FEATURES_PATH.exists():
        df = pd.read_csv(MASTER_FEATURES_PATH)
    else:
        raise FileNotFoundError(
            "Neither dashboard_environment_data.csv nor master_features.csv could be found."
        )

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df.dropna(subset=["Year"])
    df["Year"] = df["Year"].astype(int)

    if "Temperature_Range" not in df.columns and "Max_Temperature" in df.columns and "Min_Temperature" in df.columns:
        df["Temperature_Range"] = df["Max_Temperature"] - df["Min_Temperature"]

    if "Environmental_Risk_Score" not in df.columns:
        required = ["Temperature", "Rainfall", "NDVI", "Forest_Loss"]
        if all(col in df.columns for col in required):
            df["Environmental_Risk_Score"] = (
                0.30 * df["Temperature"].rank(pct=True) +
                0.25 * df["Rainfall"].rank(pct=True) +
                0.20 * (1 - df["NDVI"].rank(pct=True)) +
                0.25 * df["Forest_Loss"].rank(pct=True)
            ) * 100

    if "Environmental_Risk_Level" not in df.columns and "Environmental_Risk_Score" in df.columns:
        def risk_category(score):
            if pd.isna(score): return "Unavailable"
            if score < 33: return "Low"
            if score < 66: return "Moderate"
            return "High"
        df["Environmental_Risk_Level"] = df["Environmental_Risk_Score"].apply(risk_category)

    df["Latitude"] = df["District"].map(lambda x: DISTRICT_COORDINATES.get(x, (np.nan, np.nan))[0])
    df["Longitude"] = df["District"].map(lambda x: DISTRICT_COORDINATES.get(x, (np.nan, np.nan))[1])
    return df

# ============================================================
# 7. LOAD MODEL
# ============================================================

@st.cache_resource
def load_kfd_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    fallback = LEGACY_MODEL_DIR / "kfd_outbreak_model_v2.pkl"
    if fallback.exists():
        return joblib.load(fallback)
    return None

# ============================================================
# 8. LOAD MODEL METADATA
# ============================================================

@st.cache_data
def load_model_metadata():
    if MODEL_METADATA_PATH.exists():
        with open(MODEL_METADATA_PATH, "r") as file:
            return json.load(file)
    return {}

# ============================================================
# 9. LOAD EVERYTHING
# ============================================================

try:
    environment_df = load_environment_data()
except Exception as error:
    st.error("Unable to load the environmental dataset.")
    st.exception(error)
    st.stop()

kfd_model = load_kfd_model()
model_metadata = load_model_metadata()

# ============================================================
# 10. HELPER FUNCTIONS
# ============================================================

def safe_number(value, decimals=2):
    if pd.isna(value): return "N/A"
    return f"{float(value):.{decimals}f}"

def make_editorial_gauge(score, title, max_val=100, is_prob=False):
    score = float(np.clip(score, 0, max_val))
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "/100" if not is_prob else "", "valueformat": ".3f" if is_prob else "", "font": {"family": "Plus Jakarta Sans", "color": "#14171A"}},
            title={"text": title, "font": {"family": "Plus Jakarta Sans", "color": "#7B7D7D"}},
            gauge={
                "axis": {"range": [0, max_val], "tickcolor": "#14171A"},
                "bar": {"color": "#14171A"}, # Solid Charcoal Black
                "bgcolor": "#F7F4EE",
                "steps": [
                    {"range": [0, max_val*0.5], "color": "#EAE6DF"},
                    {"range": [max_val*0.5, max_val], "color": "#D5D1CA"}
                ]
            }
        )
    )
    fig.update_layout(height=300, margin=dict(l=30, r=30, t=60, b=20), paper_bgcolor="rgba(0,0,0,0)", font={'family': "Plus Jakarta Sans"})
    return fig

def predict_kfd_signal(row):
    if kfd_model is None: return (None, None, None)
    default_features = ["Temperature", "Rainfall", "NDVI", "Temperature_Range"]
    model_features = model_metadata.get("features", default_features)
    missing = [f for f in model_features if f not in row.index]
    if missing:
        return (None, None, "Missing model features: " + ", ".join(missing))
    
    input_data = pd.DataFrame([{feature: row[feature] for feature in model_features}])
    prediction = int(kfd_model.predict(input_data)[0])
    score = None
    if hasattr(kfd_model, "predict_proba"):
        probability_array = kfd_model.predict_proba(input_data)
        score = float(probability_array[0, 1])
    return (prediction, score, None)

# ============================================================
# 11. SIDEBAR (Dark Mode Overlay Style)
# ============================================================

with st.sidebar:
    st.html(
        """
        <div style="margin-bottom: 2rem;">
            <h1 style="font-size: 1.5rem; font-weight: 800; letter-spacing: -0.05em; color: #FFFFFF; margin-bottom: 0;">SpilloverAI</h1>
            <p style="color: #7B7D7D; font-size: 0.8rem; letter-spacing: 0.05em; text-transform: uppercase;">Western Ghats Intelligence</p>
        </div>
        """
    )
    
    # Slash-prefixed navigation to match aesthetic
    page = st.radio(
        "Menu",
        [
            "/ Dashboard",
            "/ Climate",
            "/ Vegetation",
            "/ Forest & Terrain",
            "/ KFD Prediction",
            "/ Methodology"
        ],
        label_visibility="collapsed"
    )

    st.html("<div style='height: 2rem;'></div>")
    st.html("<label>Search / Parameters</label>")

    districts = sorted(environment_df["District"].dropna().unique())
    selected_district = st.selectbox("Destination", districts)

    district_df = environment_df[environment_df["District"] == selected_district].sort_values("Year").copy()
    available_years = sorted(district_df["Year"].unique(), reverse=True)
    selected_year = st.selectbox("Timeframe", available_years)

    st.html("<div style='height: 2rem;'></div>")
    st.caption(f"Dataset: {environment_df['District'].nunique()} districts")
    st.caption(f"Coverage: {int(environment_df['Year'].min())}–{int(environment_df['Year'].max())}")
    
    if kfd_model is None:
        st.caption("⚠️ KFD Model V2 offline")

# ============================================================
# 12. SELECT CURRENT RECORD
# ============================================================

selected_record = environment_df[(environment_df["District"] == selected_district) & (environment_df["Year"] == selected_year)]

if selected_record.empty:
    st.error("No environmental observation exists for this district-year combination.")
    st.stop()

row = selected_record.iloc[0]

temperature = row.get("Temperature", np.nan)
rainfall = row.get("Rainfall", np.nan)
ndvi = row.get("NDVI", np.nan)
forest_loss = row.get("Forest_Loss", np.nan)
elevation = row.get("Elevation", np.nan)
temp_range = row.get("Temperature_Range", np.nan)
environmental_score = row.get("Environmental_Risk_Score", np.nan)
environmental_level = row.get("Environmental_Risk_Level", "Unavailable")

prediction, model_score, prediction_error = predict_kfd_signal(row)
ai_signal = "Unavailable" if prediction is None else ("Elevated" if prediction == 1 else "Lower")

# ============================================================
# 13. EDITORIAL HERO COMPONENT
# ============================================================

def show_hero():
    # Utilizing a high-saturation, warm golden-hour placeholder image for the Western Ghats travel aesthetic
    bg_url = "https://images.unsplash.com/photo-1596423735880-5c62b9f697d4?q=80&w=2000&auto=format&fit=crop"
    
    st.html(
        f"""
        <div style="position: relative; width: 100%; border-radius: 1.5rem; overflow: hidden; margin-bottom: 3rem; background: #0E1113; box-shadow: 0 20px 40px rgba(0,0,0,0.08);">
            <img src="{bg_url}" style="width: 100%; height: 500px; object-fit: cover; opacity: 0.85; filter: saturate(1.2) contrast(1.1);" />
            <div style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; display: flex; align-items: center; padding: 4rem;">
                <div style="background: rgba(255, 255, 255, 0.90); backdrop-filter: blur(16px); padding: 3.5rem; border-radius: 2rem; max-width: 600px; box-shadow: 0 20px 40px rgba(0,0,0,0.1);">
                    <span style="font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.15em; color: #7B7D7D; font-weight: 700;">Environmental Intelligence</span>
                    <h1 style="font-size: 4rem; line-height: 1.05; letter-spacing: -0.05em; color: #14171A; font-weight: 300; margin: 1rem 0 1.5rem 0;">
                        <b>Spillover</b>AI
                    </h1>
                    <p style="font-size: 1.05rem; color: #555555; margin-bottom: 2.5rem; line-height: 1.6;">
                        Multi-source environmental intelligence and experimental machine-learning assessment of Kyasanur Forest Disease outbreak conditions across the Western Ghats.
                    </p>
                    <div style="display: inline-flex; align-items: center; background: #14171A; color: #FFFFFF; border-radius: 9999px; padding: 0.75rem 1.75rem; font-size: 0.9rem; font-weight: 600; letter-spacing: 0.02em;">
                        {selected_district} &nbsp; / &nbsp; {selected_year}
                    </div>
                </div>
            </div>
        </div>
        """
    )

# ============================================================
# 14. ROUTING LOGIC
# ============================================================

if page == "/ Dashboard":
    show_hero()
    
    st.markdown("<p style='color:#7B7D7D; font-size:0.95rem;'>The Environmental Risk Index is a descriptive indicator. The AI Outbreak Signal is generated separately by a verified-label ML model.</p>", unsafe_allow_html=True)

    st.html('<div class="section-title">Environmental Snapshot</div>')

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.metric("Temperature", f"{safe_number(temperature, 1)}°C")
    with c2: st.metric("Rainfall", f"{safe_number(rainfall, 1)}mm")
    with c3: st.metric("NDVI", safe_number(ndvi, 3))
    with c4: st.metric("Forest Loss", safe_number(forest_loss, 3))
    with c5: st.metric("Elevation", f"{safe_number(elevation, 0)}m")

    st.html('<div class="section-title" style="margin-top: 4rem;">Risk & Intelligence</div>')
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.html('<span class="method-tag">Index</span>')
        if not pd.isna(environmental_score):
            st.plotly_chart(make_editorial_gauge(environmental_score, "Environmental Risk Index"), use_container_width=True, config={"displayModeBar": False})
            st.html(
                f"""
                <div class="editorial-card">
                    <p style="font-size:1.25rem; font-weight:700; color:#14171A; margin-bottom:0.5rem; letter-spacing:-0.02em;">Classification: {environmental_level}</p>
                    <p style="color:#7B7D7D; font-size:0.9rem; line-height:1.6; margin:0;">Summarizes environmental conditions; not an epidemiological diagnosis.</p>
                </div>
                """
            )

    with right:
        st.html('<span class="method-tag">AI KFD Outbreak Signal</span>')
        if prediction is None:
            st.warning("The trained KFD model is currently unavailable.")
            if prediction_error: st.caption(prediction_error)
        else:
            is_high = prediction == 1
            css_class = "signal-high" if is_high else "signal-low"
            subtext_color = "#A0A0A0" if is_high else "#7B7D7D"
            st.html(
                f"""
                <div class="{css_class}">
                    <h2>{ai_signal.upper()}</h2>
                    <p style="color:{subtext_color}; font-size:1rem; margin-top:0.5rem; font-weight:500; text-transform:uppercase; letter-spacing:0.1em;">Outbreak Signal</p>
                </div>
                """
            )
            if model_score is not None:
                st.plotly_chart(make_editorial_gauge(model_score, "Model Confidence", max_val=1, is_prob=True), use_container_width=True, config={"displayModeBar": False})
                st.markdown("<p style='color:#7B7D7D; font-size:0.85rem; text-align:center;'>Internal positive-class score. Not a calibrated probability.</p>", unsafe_allow_html=True)

    st.html('<div class="section-title" style="margin-top: 4rem;">Recent Trends</div>')
    trend_left, trend_right = st.columns(2, gap="large")

    # Clean charting aesthetics
    chart_layout = dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans", color="#14171A"),
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(showgrid=False, linecolor="#E5E0D8"),
        yaxis=dict(gridcolor="#E5E0D8", linecolor="#E5E0D8")
    )

    with trend_left:
        temp_fig = px.line(district_df, x="Year", y="Temperature", markers=True, title=f"Temperature — {selected_district}")
        temp_fig.update_traces(line_color="#14171A", marker=dict(size=8, color="#14171A"))
        temp_fig.update_layout(**chart_layout)
        st.plotly_chart(temp_fig, use_container_width=True)

    with trend_right:
        ndvi_fig = px.line(district_df, x="Year", y="NDVI", markers=True, title=f"NDVI — {selected_district}")
        ndvi_fig.update_traces(line_color="#7B7D7D", marker=dict(size=8, color="#7B7D7D"))
        ndvi_fig.update_layout(**chart_layout)
        st.plotly_chart(ndvi_fig, use_container_width=True)

    st.html('<div class="section-title" style="margin-top: 4rem;">Study Coordinates</div>')
    map_df = environment_df[environment_df["Year"] == selected_year][["State", "District", "Latitude", "Longitude", "NDVI", "Temperature"]].drop_duplicates(subset=["State", "District"]).dropna(subset=["Latitude", "Longitude"])
    
    if not map_df.empty:
        map_fig = px.scatter_geo(
            map_df, lat="Latitude", lon="Longitude", hover_name="District",
            hover_data={"State": True, "Temperature": ":.1f", "NDVI": ":.3f", "Latitude": False, "Longitude": False}
        )
        map_fig.update_traces(marker=dict(size=14, color="#14171A", line=dict(width=2, color="#FFFFFF")))
        map_fig.update_geos(
            projection_type="mercator", showland=True, landcolor="#EAE6DF",
            showcountries=True, countrycolor="#D5D1CA", showcoastlines=False,
            center=dict(lat=14.0, lon=75.2), lataxis_range=[10.5, 17.5], lonaxis_range=[72.5, 77.5],
            bgcolor="rgba(0,0,0,0)"
        )
        map_fig.update_layout(height=500, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(map_fig, use_container_width=True)

# ============================================================
# 15. CLIMATE PAGE
# ============================================================
elif page == "/ Climate":
    show_hero()
    st.html('<div class="section-title">Climate Analysis</div>')
    st.html('<div class="section-subtitle">Yearly climate observations for the selected Western Ghats district.</div>')

    temperature_columns = [col for col in ["Temperature", "Min_Temperature", "Max_Temperature"] if col in district_df.columns]
    if temperature_columns:
        temperature_long = district_df[["Year", *temperature_columns]].melt(id_vars="Year", var_name="Temperature_Type", value_name="Temperature_Value")
        fig_temperature = px.line(temperature_long, x="Year", y="Temperature_Value", color="Temperature_Type", markers=True, title=f"Temperature Profile — {selected_district}")
        fig_temperature.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Plus Jakarta Sans", color="#14171A"),
            xaxis=dict(showgrid=False, linecolor="#E5E0D8"), yaxis=dict(gridcolor="#E5E0D8", linecolor="#E5E0D8"),
            colorway=["#14171A", "#7B7D7D", "#A0A0A0"]
        )
        st.plotly_chart(fig_temperature, use_container_width=True)

# ============================================================
# 16. VEGETATION PAGE
# ============================================================
elif page == "/ Vegetation":
    show_hero()
    st.html('<div class="section-title">Vegetation & NDVI</div>')

    ndvi_left, ndvi_right = st.columns([1.5, 1], gap="large")
    with ndvi_left:
        fig_ndvi = px.line(district_df, x="Year", y="NDVI", markers=True, title=f"NDVI Through Time — {selected_district}")
        fig_ndvi.update_traces(line_color="#14171A", marker=dict(size=8))
        fig_ndvi.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Plus Jakarta Sans"), xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#E5E0D8"))
        st.plotly_chart(fig_ndvi, use_container_width=True)

    with ndvi_right:
        st.html('<div class="editorial-card">')
        st.metric("Selected NDVI", safe_number(ndvi, 3))
        if len(district_df) > 1:
            first_ndvi = district_df.sort_values("Year").iloc[0]["NDVI"]
            st.metric("Change from earliest", safe_number(ndvi - first_ndvi, 3))
        st.markdown("<p style='color:#7B7D7D; font-size:0.9rem; margin-top:1rem;'>NDVI is a satellite-derived vegetation indicator. Higher values represent denser vegetation.</p>", unsafe_allow_html=True)
        st.html('</div>')

    st.html('<div class="section-title" style="margin-top: 3rem;">District Comparison</div>')
    comparison = environment_df[environment_df["Year"] == selected_year][["District", "NDVI"]].sort_values("NDVI", ascending=False)
    if not comparison.empty:
        compare_fig = px.bar(comparison, x="District", y="NDVI", title=f"Mean NDVI — {selected_year}")
        compare_fig.update_traces(marker_color="#14171A", marker_line_width=0)
        compare_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Plus Jakarta Sans"), yaxis=dict(gridcolor="#E5E0D8"))
        st.plotly_chart(compare_fig, use_container_width=True)

# ============================================================
# 17. FOREST & TERRAIN PAGE
# ============================================================
elif page == "/ Forest & Terrain":
    show_hero()
    st.html('<div class="section-title">Forest Disturbance & Terrain</div>')
    st.markdown("<p style='color:#7B7D7D;'>Forest_Loss is treated as a district-level indicator, not a yearly causal driver.</p>", unsafe_allow_html=True)

    forest_summary = environment_df.groupby("District", as_index=False).agg(Forest_Loss=("Forest_Loss", "mean"), Elevation=("Elevation", "mean"))
    
    left, right = st.columns(2, gap="large")
    with left:
        forest_fig = px.bar(forest_summary.sort_values("Forest_Loss", ascending=False), x="District", y="Forest_Loss", title="Forest Loss Indicator")
        forest_fig.update_traces(marker_color="#14171A")
        forest_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Plus Jakarta Sans"), yaxis=dict(gridcolor="#E5E0D8"))
        st.plotly_chart(forest_fig, use_container_width=True)

    with right:
        elevation_fig = px.bar(forest_summary.sort_values("Elevation", ascending=False), x="District", y="Elevation", title="Mean Elevation")
        elevation_fig.update_traces(marker_color="#7B7D7D")
        elevation_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Plus Jakarta Sans"), yaxis=dict(gridcolor="#E5E0D8"))
        st.plotly_chart(elevation_fig, use_container_width=True)

# ============================================================
# 18. KFD PREDICTION PAGE
# ============================================================
elif page == "/ KFD Prediction":
    show_hero()
    st.html('<div class="section-title">Experimental KFD Classifier</div>')
    st.html('<div class="section-subtitle">Dynamic environmental features driving the Random Forest V2 model.</div>')

    if kfd_model is None or prediction_error:
        st.error(prediction_error or "Model missing.")
    else:
        p_left, p_right = st.columns([1, 1.5], gap="large")
        with p_left:
            css_class = "signal-high" if prediction == 1 else "signal-low"
            sub_color = "#A0A0A0" if prediction == 1 else "#7B7D7D"
            st.html(
                f"""
                <div class="{css_class}" style="margin-bottom: 2rem;">
                    <h2>{ai_signal.upper()}</h2>
                    <p style="color:{sub_color}; font-size:0.85rem; font-weight:600; text-transform:uppercase; letter-spacing:0.1em; margin-top:0.5rem;">Outbreak Signal</p>
                </div>
                """
            )
            st.metric("District Target", selected_district)
            st.metric("Environmental Year", selected_year)

        with p_right:
            if model_score is not None:
                st.plotly_chart(make_editorial_gauge(model_score, "Model Confidence Score", max_val=1, is_prob=True), use_container_width=True)
            st.markdown("<p style='color:#7B7D7D; font-size:0.85rem;'>Note: Model score is not a calibrated outbreak probability.</p>", unsafe_allow_html=True)

        st.html('<div class="section-title" style="margin-top: 3rem;">Model Importance & Validation</div>')
        
        try:
            rf_estimator = kfd_model.named_steps["model"]
            if hasattr(rf_estimator, "feature_importances_"):
                model_features = model_metadata.get("features", ["Temperature", "Rainfall", "NDVI", "Temperature_Range"])
                importance_df = pd.DataFrame({"Feature": model_features, "Importance": rf_estimator.feature_importances_}).sort_values("Importance", ascending=True)
                importance_fig = px.bar(importance_df, x="Importance", y="Feature", orientation="h", title="Dynamic Random Forest Feature Importance")
                importance_fig.update_traces(marker_color="#14171A")
                importance_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Plus Jakarta Sans"), xaxis=dict(gridcolor="#E5E0D8"))
                st.plotly_chart(importance_fig, use_container_width=True)
        except Exception:
            pass

        assessment = pd.DataFrame([{
            "District": selected_district, "Year": selected_year,
            "Temperature": temperature, "Rainfall": rainfall, "NDVI": ndvi,
            "Temperature_Range": temp_range, "AI_KFD_Signal": ai_signal, "Model_Score": model_score
        }])
        
        st.html("<br>")
        st.download_button(
            label="Download Selected Assessment",
            data=assessment.to_csv(index=False).encode("utf-8"),
            file_name=f"kfd_assessment_{selected_district.replace(' ', '_')}_{selected_year}.csv",
            mime="text/csv"
        )

# ============================================================
# 19. METHODOLOGY PAGE
# ============================================================
elif page == "/ Methodology":
    show_hero()
    st.html('<div class="section-title">Research Methodology</div>')

    st.html(
        """
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1.5rem; margin-bottom: 3rem;">
            <div class="editorial-card">
                <span class="method-tag">01. Study Region</span>
                <p style="color:#14171A; font-weight:500;">Nine KFD-relevant districts across Karnataka, Kerala, Goa and Maharashtra.</p>
            </div>
            <div class="editorial-card">
                <span class="method-tag">02. Climate Variables</span>
                <p style="color:#14171A; font-weight:500;">Annual temperature and rainfall assembled from station/interpolated observations.</p>
            </div>
            <div class="editorial-card">
                <span class="method-tag">03. Satellite Vegetation</span>
                <p style="color:#14171A; font-weight:500;">Sentinel-2 imagery processed in Google Earth Engine for NDVI.</p>
            </div>
            <div class="editorial-card">
                <span class="method-tag">04. Machine Learning</span>
                <p style="color:#14171A; font-weight:500;">Random Forest spatial generalization evaluated using Leave-One-District-Out.</p>
            </div>
        </div>
        """
    )
    
    st.html('<div class="section-title">Interpretation & Limitations</div>')
    st.markdown(
        """
        <ul style="color:#14171A; line-height:1.8; font-size:1.05rem; padding-left:1.5rem; margin-bottom: 3rem;">
            <li>Current results suggest stronger temporal predictive signal within represented regions than geographic transfer to unseen districts.</li>
            <li>Unequal historical climate coverage across districts.</li>
            <li>Classifier score is not a calibrated outbreak probability.</li>
            <li>Environmental associations do not establish definitive causality.</li>
        </ul>
        """, unsafe_allow_html=True
    )

# ============================================================
# 20. FOOTER
# ============================================================
st.html(
    """
    <div style="margin-top: 5rem; padding-top: 2rem; border-top: 1px solid #E5E0D8; text-align: center;">
        <p style="color: #14171A; font-size: 0.95rem; font-weight: 700; letter-spacing: -0.02em; margin: 0;">SpilloverAI • Zoonotic Spillover Predictor</p>
        <p style="color: #7B7D7D; font-size: 0.85rem; margin-top: 0.25rem;">Research prototype for environmental analysis of Kyasanur Forest Disease in the Western Ghats.</p>
    </div>
    """
)