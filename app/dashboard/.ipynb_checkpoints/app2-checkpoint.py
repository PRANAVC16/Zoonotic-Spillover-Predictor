# ============================================================
# SpilloverAI / Zoonotic Spillover Predictor
# Streamlit Dashboard - High-End Editorial Aesthetic
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
    page_title="Zoonotic Spillover | Environmental Intelligence",
    page_icon="🌍",
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

    /* Global Typography & Backgrounds */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    
    .stApp {
        background-color: #F7F4EE; /* Warm bone / off-white */
        color: #14171A; /* Rich charcoal */
    }

    /* Sidebar - Deep Slate Black */
    [data-testid="stSidebar"] {
        background-color: #0E1113 !important;
    }
    [data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] label {
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        color: #7B7D7D !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Minimalist Navigation (Radio Buttons) */
    .stRadio p {
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        line-height: 2 !important;
        color: #FFFFFF !important;
    }
    div[role="radiogroup"] > label {
        margin-bottom: 8px;
    }

    /* Inputs & Search Bars - Ultra-minimalist */
    [data-baseweb="select"] {
        background-color: transparent !important;
        border: none !important;
        border-bottom: 1px solid #7B7D7D !important;
        border-radius: 0px !important;
    }
    [data-baseweb="select"] * {
        color: #14171A !important; 
    }
    [data-testid="stSidebar"] [data-baseweb="select"] {
        border-bottom: 1px solid #333 !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] * {
        color: #FFFFFF !important; 
        background-color: transparent !important;
    }

    /* Buttons - Fully rounded pill */
    .stButton > button {
        background-color: #14171A !important;
        color: #FFFFFF !important;
        border-radius: 9999px !important;
        border: none !important;
        padding: 0.5rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 0.875rem !important;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #7B7D7D !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }

    /* Hero Component (Edge-to-Edge with Glassmorphic Overlay) */
    .hero-container {
        position: relative;
        background: url('https://images.unsplash.com/photo-1542224566-6e85f2e10c08?auto=format&fit=crop&q=80&w=2000') center/cover no-repeat;
        border-radius: 24px;
        min-height: 480px;
        display: flex;
        align-items: center;
        padding: 48px;
        margin-bottom: 40px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.05);
    }
    .hero-glass-card {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        padding: 40px;
        border-radius: 24px;
        max-width: 650px;
        border: 1px solid rgba(255, 255, 255, 0.6);
        box-shadow: 0 8px 32px rgba(0,0,0,0.06);
    }
    .hero-title {
        font-size: 4rem; /* text-6xl */
        font-weight: 700;
        letter-spacing: -0.04em; /* tracking-tight */
        color: #14171A;
        margin-bottom: 12px;
        line-height: 1.05;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: #7B7D7D;
        font-weight: 400;
        line-height: 1.6;
    }

    /* Section Headings */
    .section-title {
        font-size: 2rem; /* text-4xl */
        font-weight: 700;
        letter-spacing: -0.03em;
        margin-top: 32px;
        margin-bottom: 8px;
        color: #14171A;
    }
    .section-subtitle {
        color: #7B7D7D;
        font-size: 1rem;
        margin-bottom: 32px;
    }

    /* Clean Card Overlays */
    .custom-card {
        background: #FFFFFF;
        border-radius: 24px;
        padding: 32px;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.04);
        height: 100%;
        border: 1px solid rgba(0,0,0,0.03);
    }

    /* AI Signal Badges */
    .signal-high {
        padding: 24px;
        background: #14171A; /* Solid Dark */
        border-radius: 24px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
        color: #FFFFFF;
        letter-spacing: -0.02em;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }
    .signal-low {
        padding: 24px;
        background: #FFFFFF;
        border-radius: 24px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
        color: #14171A;
        border: 1px solid rgba(0,0,0,0.05);
        box-shadow: 0 10px 25px rgba(0,0,0,0.04);
    }

    .small-note {
        font-size: 0.85rem;
        color: #7B7D7D;
        line-height: 1.6;
    }

    /* Methodology Itinerary Cards */
    .method-card {
        background: #FFFFFF;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 16px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.03);
        border-top: 4px solid #14171A;
    }

    /* Streamlit Metrics Formatting */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
        font-weight: 600 !important;
        letter-spacing: -0.03em !important;
        color: #14171A !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem !important;
        color: #7B7D7D !important;
        font-weight: 500 !important;
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
        raise FileNotFoundError("Neither dashboard_environment_data.csv nor master_features.csv could be found.")

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df.dropna(subset=["Year"])
    df["Year"] = df["Year"].astype(int)

    if "Temp_Range" not in df.columns and "Max_Temperature" in df.columns and "Min_Temperature" in df.columns:
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

def update_editorial_layout(fig):
    """Applies the editorial theme to Plotly figures."""
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Plus Jakarta Sans', color='#14171A'),
        title_font=dict(size=18, family='Plus Jakarta Sans', color='#14171A'),
        xaxis=dict(showgrid=False, linecolor='#7B7D7D'),
        yaxis=dict(gridcolor='rgba(123, 125, 125, 0.1)', linecolor='#7B7D7D'),
        margin=dict(l=20, r=20, t=55, b=20)
    )
    return fig

def make_environment_gauge(score):
    score = float(np.clip(score, 0, 100))
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "/100", "font": {"family": "Plus Jakarta Sans", "color": "#14171A"}},
            title={"text": "Environmental Risk Index", "font": {"family": "Plus Jakarta Sans", "color": "#7B7D7D", "size": 14}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#14171A"},
                "bar": {"color": "#14171A"},
                "steps": [
                    {"range": [0, 33], "color": "#F7F4EE"},
                    {"range": [33, 66], "color": "#E5E0D8"},
                    {"range": [66, 100], "color": "#D1C9BE"}
                ]
            }
        )
    )
    fig.update_layout(height=300, margin=dict(l=30, r=30, t=60, b=20), paper_bgcolor='rgba(0,0,0,0)')
    return fig

def make_model_gauge(model_score):
    model_score = float(np.clip(model_score, 0, 1))
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=model_score,
            number={"valueformat": ".3f", "font": {"family": "Plus Jakarta Sans", "color": "#14171A"}},
            title={"text": "Model Score", "font": {"family": "Plus Jakarta Sans", "color": "#7B7D7D", "size": 14}},
            gauge={
                "axis": {"range": [0, 1]},
                "bar": {"color": "#14171A"},
                "steps": [
                    {"range": [0, 0.5], "color": "#F7F4EE"},
                    {"range": [0.5, 1], "color": "#D1C9BE"}
                ],
                "threshold": {
                    "line": {"color": "#7B7D7D", "width": 3},
                    "thickness": 0.8,
                    "value": 0.5
                }
            }
        )
    )
    fig.update_layout(height=300, margin=dict(l=30, r=30, t=60, b=20), paper_bgcolor='rgba(0,0,0,0)')
    return fig


def predict_kfd_signal(row):
    if kfd_model is None:
        return None, None, None

    default_features = ["Temperature", "Rainfall", "NDVI", "Temperature_Range"]
    model_features = model_metadata.get("features", default_features)
    missing = [feature for feature in model_features if feature not in row.index]

    if missing:
        return None, None, ("Missing model features: " + ", ".join(missing))

    input_data = pd.DataFrame([{feature: row[feature] for feature in model_features}])
    prediction = int(kfd_model.predict(input_data)[0])
    score = None

    if hasattr(kfd_model, "predict_proba"):
        probability_array = kfd_model.predict_proba(input_data)
        score = float(probability_array[0, 1])

    return prediction, score, None


# ============================================================
# 11. SIDEBAR
# ============================================================

with st.sidebar:
    st.html(
        """
        <div style="font-size: 1.5rem; font-weight: 700; color: #FFF; letter-spacing:-0.05em; margin-bottom:0.5rem;">
        SpilloverAI<br>
        <span style="font-weight:300; font-size:1rem; color:#7B7D7D;">Editorial Intelligence</span>
        </div>
        """
    )

    st.caption("KFD • Western Ghats")
    st.divider()

    # Formatted minimal navigation strings
    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Climate",
            "Vegetation",
            "Forest & Terrain",
            "KFD Prediction",
            " Methodology"
        ],
        label_visibility="collapsed"
    )

    st.divider()
    st.html("<label>Study Area</label>")

    districts = sorted(environment_df["District"].dropna().unique())
    selected_district = st.selectbox("District", districts, label_visibility="collapsed")

    district_df = environment_df[environment_df["District"] == selected_district].sort_values("Year").copy()
    available_years = sorted(district_df["Year"].unique(), reverse=True)
    
    st.html("<label style='margin-top:16px; display:block;'>Year</label>")
    selected_year = st.selectbox("Year", available_years, label_visibility="collapsed")

    st.divider()
    st.caption(f"Dataset: {environment_df['District'].nunique()} districts")
    st.caption(f"Coverage: {int(environment_df['Year'].min())}–{int(environment_df['Year'].max())}")

    if kfd_model is not None:
        st.success("KFD Model V2 loaded")
    else:
        st.warning("KFD Model V2 not found")


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


# ============================================================
# 13. REAL MODEL PREDICTION
# ============================================================

prediction, model_score, prediction_error = predict_kfd_signal(row)

if prediction is None:
    ai_signal = "Unavailable"
elif prediction == 1:
    ai_signal = "Elevated"
else:
    ai_signal = "Lower"


# ============================================================
# 14. COMMON HERO
# ============================================================

def show_hero(title_text):
    st.html(f"""
    <div class="hero-container">
        <div class="hero-glass-card">
            <div class="hero-title">{title_text}</div>
            <div class="hero-subtitle">
                Multi-source environmental intelligence and experimental machine-learning assessment of Kyasanur Forest Disease across the Western Ghats.
                <br><br>
                <b style="color:#14171A;">{selected_district}</b> &nbsp; • &nbsp; <b style="color:#14171A;">{selected_year}</b>
            </div>
        </div>
    </div>
    """)


# ============================================================
# 15. PAGE ROUTING
# ============================================================

if page == "/ Dashboard":
    show_hero("SpilloverAI")

    st.info(
        "The Environmental Risk Index is a descriptive environmental indicator. The AI KFD Outbreak Signal "
        "is generated separately by the verified-label machine-learning model."
    )

    st.html('<div class="section-title">Environmental Snapshot</div>')

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.metric("🌡 Temperature", f"{safe_number(temperature, 1)} °C")
    with c2: st.metric("🌧 Rainfall", f"{safe_number(rainfall, 1)} mm")
    with c3: st.metric("🌿 NDVI", safe_number(ndvi, 3))
    with c4: st.metric("🌲 Forest Loss", safe_number(forest_loss, 3))
    with c5: st.metric("⛰ Elevation", f"{safe_number(elevation, 0)} m")

    st.divider()

    left, right = st.columns([1, 1])
    with left:
        if not pd.isna(environmental_score):
            environment_gauge = make_environment_gauge(environmental_score)
            st.plotly_chart(environment_gauge, use_container_width=True, config={"displayModeBar": False})
            st.html(f"""
                <div class="custom-card">
                    <b style="font-size:1.1rem; color:#14171A;">Environmental Classification:</b> <span style="font-size:1.1rem; color:#7B7D7D;">{environmental_level}</span>
                    <br><br>
                    <span class="small-note">This index summarizes environmental conditions. It is not a disease diagnosis or epidemiological outbreak probability.</span>
                </div>
            """)

    with right:
        st.html('<div style="font-size: 1.25rem; font-weight:600; color:#14171A; margin-bottom:16px; text-align:center;">AI KFD Outbreak Signal</div>')
        if prediction is None:
            st.warning("The trained KFD model is currently unavailable.")
            if prediction_error: st.caption(prediction_error)
        else:
            css_class = "signal-high" if prediction == 1 else "signal-low"
            st.html(f'<div class="{css_class}">{ai_signal.upper()} SIGNAL</div>')
            
            if model_score is not None:
                score_gauge = make_model_gauge(model_score)
                st.plotly_chart(score_gauge, use_container_width=True, config={"displayModeBar": False})
            st.caption("The model score is the classifier's internal positive-class score. It has not been demonstrated to be a calibrated epidemiological outbreak probability.")

    st.divider()

    st.html('<div class="section-title">Recent Environmental Trends</div>')
    trend_left, trend_right = st.columns(2)

    with trend_left:
        temp_fig = px.line(district_df, x="Year", y="Temperature", markers=True, title=f"Temperature Trend — {selected_district}", color_discrete_sequence=['#14171A'])
        st.plotly_chart(update_editorial_layout(temp_fig), use_container_width=True)

    with trend_right:
        ndvi_fig = px.line(district_df, x="Year", y="NDVI", markers=True, title=f"NDVI Trend — {selected_district}", color_discrete_sequence=['#7B7D7D'])
        st.plotly_chart(update_editorial_layout(ndvi_fig), use_container_width=True)

    st.html('<div class="section-title">Western Ghats Study Districts</div>')
    map_df = environment_df[environment_df["Year"] == selected_year][["State", "District", "Latitude", "Longitude", "NDVI", "Temperature"]].drop_duplicates(subset=["State", "District"]).dropna(subset=["Latitude", "Longitude"])

    if not map_df.empty:
        map_fig = px.scatter_geo(map_df, lat="Latitude", lon="Longitude", hover_name="District", 
                                 hover_data={"State": True, "Temperature": ":.1f", "NDVI": ":.3f", "Latitude": False, "Longitude": False},
                                 size_max=15, color_discrete_sequence=['#14171A'])
        map_fig.update_traces(marker=dict(size=12))
        map_fig.update_geos(projection_type="mercator", showland=True, landcolor="#E5E0D8", showcountries=True, countrycolor="#D1C9BE", showcoastlines=True, coastlinecolor="#7B7D7D", center=dict(lat=14.0, lon=75.2), lataxis_range=[10.5, 17.5], lonaxis_range=[72.5, 77.5])
        map_fig.update_layout(height=500, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(map_fig, use_container_width=True)

# ============================================================
# 16. CLIMATE PAGE
# ============================================================
elif page == "/ Climate":
    show_hero("Climate Analysis")
    st.html('<div class="section-subtitle">Yearly climate observations for the selected Western Ghats district.</div>')

    temperature_columns = [col for col in ["Temperature", "Min_Temperature", "Max_Temperature"] if col in district_df.columns]
    if temperature_columns:
        temperature_long = district_df[["Year", *temperature_columns]].melt(id_vars="Year", var_name="Temperature_Type", value_name="Temperature_Value")
        fig_temperature = px.line(temperature_long, x="Year", y="Temperature_Value", color="Temperature_Type", markers=True, title=f"Temperature Profile — {selected_district}", color_discrete_sequence=['#14171A', '#7B7D7D', '#D1C9BE'])
        st.plotly_chart(update_editorial_layout(fig_temperature), use_container_width=True)

# ============================================================
# 17. VEGETATION PAGE
# ============================================================
elif page == "/ Vegetation":
    show_hero("Vegetation & NDVI")
    ndvi_left, ndvi_right = st.columns([1.2, 1])

    with ndvi_left:
        fig_ndvi = px.line(district_df, x="Year", y="NDVI", markers=True, title=f"NDVI Through Time — {selected_district}", color_discrete_sequence=['#14171A'])
        st.plotly_chart(update_editorial_layout(fig_ndvi), use_container_width=True)

    with ndvi_right:
        st.metric("Selected NDVI", safe_number(ndvi, 3))
        if len(district_df) > 1:
            first_ndvi = district_df.sort_values("Year").iloc[0]["NDVI"]
            ndvi_change = ndvi - first_ndvi
            st.metric("Change from first available year", safe_number(ndvi_change, 3))
        
        st.html("""
        <div class="custom-card" style="margin-top:24px;">
            NDVI is used as a satellite-derived vegetation indicator. Higher values generally represent denser or healthier vegetation, while lower values indicate reduced vegetation cover.
        </div>
        """)

    comparison = environment_df[environment_df["Year"] == selected_year][["District", "NDVI"]].sort_values("NDVI", ascending=False)
    if not comparison.empty:
        compare_fig = px.bar(comparison, x="District", y="NDVI", title=f"District NDVI Comparison — {selected_year}", color_discrete_sequence=['#14171A'])
        st.plotly_chart(update_editorial_layout(compare_fig), use_container_width=True)

# ============================================================
# 18. FOREST & TERRAIN PAGE
# ============================================================
elif page == "/ Forest & Terrain":
    show_hero("Forest & Terrain")
    st.warning("Forest_Loss is currently treated as a district-level environmental indicator in the research dataset. It should not be interpreted as a yearly causal driver.")

    forest_summary = environment_df.groupby("District", as_index=False).agg(Forest_Loss=("Forest_Loss", "mean"), Elevation=("Elevation", "mean"))
    left, right = st.columns(2)

    with left:
        forest_fig = px.bar(forest_summary.sort_values("Forest_Loss", ascending=False), x="District", y="Forest_Loss", title="Forest Loss Indicator by District", color_discrete_sequence=['#14171A'])
        st.plotly_chart(update_editorial_layout(forest_fig), use_container_width=True)

    with right:
        elevation_fig = px.bar(forest_summary.sort_values("Elevation", ascending=False), x="District", y="Elevation", title="Mean Elevation by District", color_discrete_sequence=['#7B7D7D'])
        st.plotly_chart(update_editorial_layout(elevation_fig), use_container_width=True)

    st.html('<div class="section-title">Selected District</div>')
    c1, c2 = st.columns(2)
    with c1: st.metric("Forest Loss Indicator", safe_number(forest_loss, 3))
    with c2: st.metric("Mean Elevation", f"{safe_number(elevation, 0)} m")

# ============================================================
# 19. KFD PREDICTION PAGE
# ============================================================
elif page == "/ KFD Prediction":
    show_hero("Experimental Classifier")
    st.html('<div class="section-subtitle">Dynamic environmental features are passed to the spatially evaluated Random Forest V2 model.</div>')

    if kfd_model is None:
        st.error("kfd_outbreak_model_v2.pkl could not be found.")
    elif prediction_error:
        st.error(prediction_error)
    else:
        prediction_left, prediction_right = st.columns([1, 1.2])

        with prediction_left:
            css_class = "signal-high" if prediction == 1 else "signal-low"
            st.html(f'<div class="{css_class}">{ai_signal.upper()} OUTBREAK SIGNAL</div><br>')
            st.metric("District", selected_district)
            st.metric("Environmental Year", selected_year)

        with prediction_right:
            if model_score is not None:
                model_gauge = make_model_gauge(model_score)
                st.plotly_chart(model_gauge, use_container_width=True, config={"displayModeBar": False})

        st.warning("The model score must not be interpreted as a calibrated probability that a KFD outbreak will occur. This is an experimental environmental classification model.")

        st.html('<div class="section-title">Model Inputs</div>')
        model_features = model_metadata.get("features", ["Temperature", "Rainfall", "NDVI", "Temp_Range"])
        input_table = pd.DataFrame({"Feature": model_features, "Value": [row.get(feature, np.nan) for feature in model_features]})
        st.dataframe(input_table, use_container_width=True, hide_index=True)

        try:
            rf_estimator = kfd_model.named_steps["model"]
            if hasattr(rf_estimator, "feature_importances_"):
                importance_df = pd.DataFrame({"Feature": model_features, "Importance": rf_estimator.feature_importances_}).sort_values("Importance", ascending=True)
                importance_fig = px.bar(importance_df, x="Importance", y="Feature", orientation="h", title="Dynamic Random Forest Feature Importance", color_discrete_sequence=['#14171A'])
                st.plotly_chart(update_editorial_layout(importance_fig), use_container_width=True)
                st.caption("Feature importance describes the model's use of variables. It does not demonstrate causality.")
        except Exception:
            st.info("Feature importance is unavailable for the loaded model.")

        st.html('<div class="section-title">Research Validation</div>')
        m1, m2, m3, m4 = st.columns(4)
        with m1: st.metric("Training observations", model_metadata.get("training_observations", "46"))
        with m2: st.metric("Study districts", model_metadata.get("districts", "9"))
        with m3: st.metric("Spatial Balanced Accuracy", f"{float(model_metadata.get('spatial_balanced_accuracy', 0.565)):.3f}")
        with m4: st.metric("Spatial ROC-AUC", f"{float(model_metadata.get('spatial_roc_auc', 0.544)):.3f}")

        st.info("Primary evaluation used Leave-One-District-Out spatial validation. The model showed modest cross-district generalization, while temporal validation within previously represented regions was stronger.")

        assessment = pd.DataFrame([{
            "State": row.get("State", ""), "District": selected_district, "Year": selected_year,
            "Temperature": temperature, "Rainfall": rainfall, "NDVI": ndvi, "Temperature_Range": temp_range,
            "Environmental_Risk_Score": environmental_score, "Environmental_Risk_Level": environmental_level,
            "AI_KFD_Signal": ai_signal, "Model_Score": model_score
        }])

        st.download_button(
            label="Download Selected Assessment",
            data=assessment.to_csv(index=False).encode("utf-8"),
            file_name=f"kfd_assessment_{selected_district.replace(' ', '_')}_{selected_year}.csv",
            mime="text/csv"
        )

# ============================================================
# 20. METHODOLOGY PAGE
# ============================================================
elif page == "/ Methodology":
    show_hero("Methodology")

    st.html("""
        <div class="method-card"><b>1. Study Region</b><br><span style="color:#7B7D7D;">Nine KFD-relevant districts across Karnataka, Kerala, Goa and Maharashtra were included in the Western Ghats study area.</span></div>
        <div class="method-card"><b>2. Climate Variables</b><br><span style="color:#7B7D7D;">Annual temperature and rainfall variables were assembled from station/interpolated climate observations.</span></div>
        <div class="method-card"><b>3. Satellite Vegetation</b><br><span style="color:#7B7D7D;">Sentinel-2 imagery was processed in Google Earth Engine to derive NDVI-based vegetation indicators.</span></div>
        <div class="method-card"><b>4. Forest & Terrain</b><br><span style="color:#7B7D7D;">Hansen Global Forest Change data and SRTM elevation information were incorporated into the environmental dataset.</span></div>
        <div class="method-card"><b>5. Verified Disease Targets</b><br><span style="color:#7B7D7D;">Historical KFD occurrence labels were compiled from source-backed surveillance and government records and merged with district-year environmental observations.</span></div>
        <div class="method-card"><b>6. Machine Learning</b><br><span style="color:#7B7D7D;">Logistic Regression, Decision Tree, Random Forest and XGBoost classifiers were compared. Spatial generalization was evaluated using Leave-One-District-Out validation.</span></div>
        <div class="method-card"><b>7. Feature-Set Ablation</b><br><span style="color:#7B7D7D;">Dynamic environmental features generalized better across unseen districts than the static-only feature set. The primary deployment model therefore uses Temperature, Rainfall, NDVI and Temperature Range.</span></div>
    """)

    st.html('<div class="section-title">Interpretation of Current Results</div>')
    results_table = pd.DataFrame({
        "Experiment": ["Dynamic Random Forest — Spatial", "Combined Decision Tree — Spatial", "Decision Tree — Temporal"],
        "Balanced Accuracy": [0.565, 0.543, 0.833],
        "F1": [0.545, 0.618, 0.909],
        "ROC-AUC": [0.544, 0.354, 0.800]
    })
    st.dataframe(results_table, use_container_width=True, hide_index=True)

    st.html("""
        <div class="custom-card" style="margin-top:24px;">
        The current results suggest stronger temporal predictive signal within represented regions than geographic transfer to completely unseen districts. Consequently, the dashboard should be interpreted as a research prototype rather than an operational disease-surveillance system.
        </div>
    """)

# ============================================================
# 21. FOOTER
# ============================================================
st.divider()
st.html(
    """
    <div style="text-align:center; color:#7B7D7D; font-size:0.8rem; padding:20px 0; font-weight:500; letter-spacing:0.02em;">
    SpilloverAI • Zoonotic Spillover Predictor<br>
    Research prototype for environmental analysis of Kyasanur Forest Disease in the Western Ghats.
    </div>
    """
)