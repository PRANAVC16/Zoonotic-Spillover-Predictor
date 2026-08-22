# ============================================================
# SpilloverAI / Zoonotic Spillover Predictor
# Streamlit Dashboard
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
    page_title="SpilloverAI | KFD Environmental Intelligence",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. FIND PROJECT ROOT AUTOMATICALLY
# ============================================================

def find_project_root():

    current = Path(__file__).resolve().parent

    candidates = [
        current,
        *current.parents
    ]

    for folder in candidates:

        if (
            (folder / "data").exists()
            and
            (folder / "notebooks").exists()
        ):
            return folder

    # Fallback for:
    # project/app/dashboard/app.py

    return Path(__file__).resolve().parents[2]


ROOT_DIR = find_project_root()

DATA_DIR = ROOT_DIR / "data" / "processed"

OUTPUT_MODEL_DIR = (
    ROOT_DIR
    / "outputs"
    / "models"
)

LEGACY_MODEL_DIR = (
    ROOT_DIR
    / "models"
)


# ============================================================
# 3. FILE PATHS
# ============================================================

DASHBOARD_DATA_PATH = (
    DATA_DIR
    / "dashboard_environment_data.csv"
)

MASTER_FEATURES_PATH = (
    DATA_DIR
    / "master_features.csv"
)

MODEL_PATH = (
    OUTPUT_MODEL_DIR
    / "kfd_outbreak_model_v2.pkl"
)

MODEL_METADATA_PATH = (
    OUTPUT_MODEL_DIR
    / "kfd_outbreak_model_v2_metadata.json"
)


# ============================================================
# 4. DISTRICT COORDINATES
# ============================================================

DISTRICT_COORDINATES = {

    "Shivamogga": (
        13.9299,
        75.5681
    ),

    "Uttara Kannada": (
        14.7937,
        74.6869
    ),

    "Chikkamagaluru": (
        13.3161,
        75.7720
    ),

    "Kodagu": (
        12.4244,
        75.7382
    ),

    "Wayanad": (
        11.6854,
        76.1320
    ),

    "Kannur": (
        11.8745,
        75.3704
    ),

    "North Goa": (
        15.4909,
        73.8278
    ),

    "South Goa": (
        15.1170,
        74.1240
    ),

    "Sindhudurg": (
        16.3492,
        73.5594
    )
}


# ============================================================
# 5. CUSTOM CSS
# ============================================================

st.html(
    """
    <style>

    /* Main page */
    .stApp {
        background:
            linear-gradient(
                180deg,
                #f7fbfa 0%,
                #ffffff 42%,
                #f5faf8 100%
            );
    }

    /* Reduce upper blank space */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #071d18,
                #0b3027
            );
    }

    [data-testid="stSidebar"] * {
        color: #f5fffb;
    }

    [data-testid="stSidebar"] label {
        color: #dcf7ed !important;
    }

    /* Hero */
    .hero-container {

        padding:
            34px
            38px;

        border-radius:
            24px;

        background:
            linear-gradient(
                120deg,
                #082d25,
                #0b5543
            );

        color:
            white;

        margin-bottom:
            24px;

        box-shadow:
            0 10px 35px
            rgba(
                0,
                55,
                43,
                0.16
            );
    }

    .hero-title {

        font-size:
            42px;

        font-weight:
            800;

        margin-bottom:
            8px;

        letter-spacing:
            -1px;
    }

    .hero-subtitle {

        font-size:
            17px;

        color:
            #d4f4e7;

        max-width:
            850px;

        line-height:
            1.6;
    }

    /* Section headings */
    .section-title {

        font-size:
            25px;

        font-weight:
            750;

        margin-top:
            15px;

        margin-bottom:
            6px;

        color:
            #0b322a;
    }

    .section-subtitle {

        color:
            #65746f;

        font-size:
            14px;

        margin-bottom:
            18px;
    }

    /* Info cards */
    .custom-card {

        background:
            white;

        border:
            1px solid
            #e3efea;

        border-radius:
            18px;

        padding:
            20px;

        box-shadow:
            0 4px 18px
            rgba(
                20,
                70,
                55,
                0.06
            );

        height:
            100%;
    }

    /* AI signal */
    .signal-high {

        padding:
            22px;

        background:
            #fff3f2;

        border:
            1px solid
            #f2c7c2;

        border-radius:
            16px;

        text-align:
            center;

        font-size:
            26px;

        font-weight:
            800;

        color:
            #a63026;
    }

    .signal-low {

        padding:
            22px;

        background:
            #effaf5;

        border:
            1px solid
            #c8e8d9;

        border-radius:
            16px;

        text-align:
            center;

        font-size:
            26px;

        font-weight:
            800;

        color:
            #176647;
    }

    .small-note {

        font-size:
            13px;

        color:
            #697873;

        line-height:
            1.55;
    }

    .method-card {

        border-left:
            5px solid
            #198c68;

        background:
            white;

        border-radius:
            14px;

        padding:
            20px;

        margin-bottom:
            15px;

        box-shadow:
            0 3px 12px
            rgba(
                0,
                0,
                0,
                0.04
            );
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

        df = pd.read_csv(
            DASHBOARD_DATA_PATH
        )

    elif MASTER_FEATURES_PATH.exists():

        df = pd.read_csv(
            MASTER_FEATURES_PATH
        )

    else:

        raise FileNotFoundError(
            "Neither dashboard_environment_data.csv "
            "nor master_features.csv could be found."
        )


    # --------------------------------------------------------
    # Clean Year
    # --------------------------------------------------------

    df["Year"] = (
        pd.to_numeric(
            df["Year"],
            errors="coerce"
        )
    )

    df = df.dropna(
        subset=["Year"]
    )

    df["Year"] = (
        df["Year"]
        .astype(int)
    )


    # --------------------------------------------------------
    # Temperature range
    # --------------------------------------------------------

    if (
        "Temperature_Range"
        not in df.columns
        and
        "Max_Temperature"
        in df.columns
        and
        "Min_Temperature"
        in df.columns
    ):

        df[
            "Temperature_Range"
        ] = (

            df[
                "Max_Temperature"
            ]

            -

            df[
                "Min_Temperature"
            ]
        )


    # --------------------------------------------------------
    # Environmental Risk Score fallback
    # --------------------------------------------------------

    if (
        "Environmental_Risk_Score"
        not in df.columns
    ):

        required = [
            "Temperature",
            "Rainfall",
            "NDVI",
            "Forest_Loss"
        ]

        if all(
            col in df.columns
            for col in required
        ):

            df[
                "Environmental_Risk_Score"
            ] = (

                0.30
                *
                df[
                    "Temperature"
                ].rank(
                    pct=True
                )

                +

                0.25
                *
                df[
                    "Rainfall"
                ].rank(
                    pct=True
                )

                +

                0.20
                *
                (
                    1
                    -
                    df[
                        "NDVI"
                    ].rank(
                        pct=True
                    )
                )

                +

                0.25
                *
                df[
                    "Forest_Loss"
                ].rank(
                    pct=True
                )

            ) * 100


    # --------------------------------------------------------
    # Environmental Risk Level fallback
    # --------------------------------------------------------

    if (
        "Environmental_Risk_Level"
        not in df.columns
        and
        "Environmental_Risk_Score"
        in df.columns
    ):

        def risk_category(score):

            if pd.isna(score):
                return "Unavailable"

            if score < 33:
                return "Low"

            if score < 66:
                return "Moderate"

            return "High"


        df[
            "Environmental_Risk_Level"
        ] = (

            df[
                "Environmental_Risk_Score"
            ].apply(
                risk_category
            )
        )


    # --------------------------------------------------------
    # Coordinates
    # --------------------------------------------------------

    df[
        "Latitude"
    ] = (
        df["District"]
        .map(
            lambda x:
                DISTRICT_COORDINATES
                .get(
                    x,
                    (np.nan, np.nan)
                )[0]
        )
    )

    df[
        "Longitude"
    ] = (
        df["District"]
        .map(
            lambda x:
                DISTRICT_COORDINATES
                .get(
                    x,
                    (np.nan, np.nan)
                )[1]
        )
    )


    return df


# ============================================================
# 7. LOAD MODEL
# ============================================================

@st.cache_resource
def load_kfd_model():

    # Primary research model
    if MODEL_PATH.exists():

        model = joblib.load(
            MODEL_PATH
        )

        return model


    # Optional fallback
    fallback = (
        LEGACY_MODEL_DIR
        / "kfd_outbreak_model_v2.pkl"
    )

    if fallback.exists():

        model = joblib.load(
            fallback
        )

        return model


    return None


# ============================================================
# 8. LOAD MODEL METADATA
# ============================================================

@st.cache_data
def load_model_metadata():

    if MODEL_METADATA_PATH.exists():

        with open(
            MODEL_METADATA_PATH,
            "r"
        ) as file:

            return json.load(
                file
            )

    return {}


# ============================================================
# 9. LOAD EVERYTHING
# ============================================================

try:

    environment_df = (
        load_environment_data()
    )

except Exception as error:

    st.error(
        "Unable to load the environmental dataset."
    )

    st.exception(
        error
    )

    st.stop()


kfd_model = (
    load_kfd_model()
)

model_metadata = (
    load_model_metadata()
)


# ============================================================
# 10. HELPER FUNCTIONS
# ============================================================

def safe_number(
    value,
    decimals=2
):

    if pd.isna(value):
        return "N/A"

    return (
        f"{float(value):.{decimals}f}"
    )


def make_environment_gauge(
    score
):

    score = float(
        np.clip(
            score,
            0,
            100
        )
    )

    fig = go.Figure(

        go.Indicator(

            mode=(
                "gauge+number"
            ),

            value=score,

            number={
                "suffix": "/100"
            },

            title={
                "text":
                    "Environmental Risk Index"
            },

            gauge={

                "axis": {
                    "range": [
                        0,
                        100
                    ]
                },

                "bar": {
                    "color":
                        "#125b4c"
                },

                "steps": [

                    {
                        "range":
                            [0, 33],

                        "color":
                            "#dff3e8"
                    },

                    {
                        "range":
                            [33, 66],

                        "color":
                            "#fff0c2"
                    },

                    {
                        "range":
                            [66, 100],

                        "color":
                            "#f7d6d2"
                    }
                ]
            }
        )
    )

    fig.update_layout(

        height=300,

        margin=dict(
            l=30,
            r=30,
            t=60,
            b=20
        )
    )

    return fig


def make_model_gauge(
    model_score
):

    model_score = float(
        np.clip(
            model_score,
            0,
            1
        )
    )

    fig = go.Figure(

        go.Indicator(

            mode=(
                "gauge+number"
            ),

            value=model_score,

            number={
                "valueformat":
                    ".3f"
            },

            title={
                "text":
                    "Model Score"
            },

            gauge={

                "axis": {
                    "range": [
                        0,
                        1
                    ]
                },

                "bar": {
                    "color":
                        "#1a6655"
                },

                "steps": [

                    {
                        "range":
                            [0, 0.5],

                        "color":
                            "#dff3e8"
                    },

                    {
                        "range":
                            [0.5, 1],

                        "color":
                            "#f5d6d1"
                    }
                ],

                "threshold": {

                    "line": {
                        "color":
                            "#272727",

                        "width":
                            3
                    },

                    "thickness":
                        0.8,

                    "value":
                        0.5
                }
            }
        )
    )

    fig.update_layout(

        height=300,

        margin=dict(
            l=30,
            r=30,
            t=60,
            b=20
        )
    )

    return fig


def predict_kfd_signal(
    row
):

    if kfd_model is None:

        return (
            None,
            None,
            None
        )


    default_features = [

        "Temperature",
        "Rainfall",
        "NDVI",
        "Temperature_Range"
    ]


    model_features = (
        model_metadata.get(
            "features",
            default_features
        )
    )


    missing = [

        feature

        for feature
        in model_features

        if feature
        not in row.index
    ]


    if missing:

        return (
            None,
            None,
            (
                "Missing model features: "
                +
                ", ".join(
                    missing
                )
            )
        )


    input_data = pd.DataFrame(

        [
            {
                feature:
                    row[feature]

                for feature
                in model_features
            }
        ]
    )


    prediction = int(
        kfd_model.predict(
            input_data
        )[0]
    )


    score = None


    if hasattr(
        kfd_model,
        "predict_proba"
    ):

        probability_array = (
            kfd_model
            .predict_proba(
                input_data
            )
        )

        score = float(
            probability_array[
                0,
                1
            ]
        )


    return (
        prediction,
        score,
        None
    )


# ============================================================
# 11. SIDEBAR
# ============================================================

with st.sidebar:

    st.html(
        """
        SpilloverAI
        Environmental Intelligence
        """
    )

    st.caption(
        "KFD • Western Ghats"
    )

    st.divider()


    page = st.radio(

        "Navigation",

        [
            "Dashboard",
            "Climate",
            "Vegetation",
            "Forest & Terrain",
            "KFD Prediction",
            "Methodology"
        ],

        label_visibility="collapsed"
    )


    st.divider()


    st.html(
        "Study Area"
    )


    districts = sorted(
        environment_df[
            "District"
        ].dropna().unique()
    )


    selected_district = (
        st.selectbox(
            "District",
            districts
        )
    )


    district_df = (
        environment_df[
            environment_df[
                "District"
            ]
            ==
            selected_district
        ]
        .sort_values(
            "Year"
        )
        .copy()
    )


    available_years = sorted(

        district_df[
            "Year"
        ].unique(),

        reverse=True
    )


    selected_year = (
        st.selectbox(
            "Year",
            available_years
        )
    )


    st.divider()


    st.caption(
        f"Dataset: "
        f"{environment_df['District'].nunique()} districts"
    )

    st.caption(
        f"Coverage: "
        f"{int(environment_df['Year'].min())}"
        "–"
        f"{int(environment_df['Year'].max())}"
    )


    if kfd_model is not None:

        st.success(
            "KFD Model V2 loaded"
        )

    else:

        st.warning(
            "KFD Model V2 not found"
        )


# ============================================================
# 12. SELECT CURRENT RECORD
# ============================================================

selected_record = (

    environment_df[

        (
            environment_df[
                "District"
            ]
            ==
            selected_district
        )

        &

        (
            environment_df[
                "Year"
            ]
            ==
            selected_year
        )

    ]
)


if selected_record.empty:

    st.error(
        "No environmental observation exists "
        "for this district-year combination."
    )

    st.stop()


row = (
    selected_record.iloc[0]
)


temperature = row.get(
    "Temperature",
    np.nan
)

rainfall = row.get(
    "Rainfall",
    np.nan
)

ndvi = row.get(
    "NDVI",
    np.nan
)

forest_loss = row.get(
    "Forest_Loss",
    np.nan
)

elevation = row.get(
    "Elevation",
    np.nan
)

temp_range = row.get(
    "Temperature_Range",
    np.nan
)

environmental_score = row.get(
    "Environmental_Risk_Score",
    np.nan
)

environmental_level = row.get(
    "Environmental_Risk_Level",
    "Unavailable"
)


# ============================================================
# 13. REAL MODEL PREDICTION
# ============================================================

prediction, model_score, prediction_error = (
    predict_kfd_signal(
        row
    )
)


if prediction is None:

    ai_signal = "Unavailable"

elif prediction == 1:

    ai_signal = "Elevated"

else:

    ai_signal = "Lower"


# ============================================================
# 14. COMMON HERO
# ============================================================

def show_hero():

    st.html(
        f"""
<div class="hero-container">
    <div class="hero-title">
        SpilloverAI
    </div>

    <div class="hero-subtitle">
        Multi-source environmental intelligence
        and experimental machine-learning
        assessment of Kyasanur Forest Disease
        outbreak conditions across the Western Ghats.

        <br><br>

        <b>{selected_district}</b>
        &nbsp; • &nbsp;
        <b>{selected_year}</b>
    </div>
</div>
        """
    )


# ============================================================
# 15. DASHBOARD PAGE
# ============================================================

if page == "Dashboard":

    show_hero()


    st.info(
        "The Environmental Risk Index is a descriptive "
        "environmental indicator. The AI KFD Outbreak Signal "
        "is generated separately by the verified-label "
        "machine-learning model."
    )


    # --------------------------------------------------------
    # Environmental KPIs
    # --------------------------------------------------------

    st.html(
        '<div class="section-title">'
        'Environmental Snapshot'
        '</div>'
    )


    c1, c2, c3, c4, c5 = (
        st.columns(5)
    )


    with c1:

        st.metric(
            "🌡 Temperature",
            (
                f"{safe_number(temperature, 1)} °C"
            )
        )


    with c2:

        st.metric(
            "🌧 Rainfall",
            (
                f"{safe_number(rainfall, 1)} mm"
            )
        )


    with c3:

        st.metric(
            "🌿 NDVI",
            safe_number(
                ndvi,
                3
            )
        )


    with c4:

        st.metric(
            "🌲 Forest Loss",
            safe_number(
                forest_loss,
                3
            )
        )


    with c5:

        st.metric(
            "⛰ Elevation",
            (
                f"{safe_number(elevation, 0)} m"
            )
        )


    st.divider()


    # --------------------------------------------------------
    # Risk + ML signal
    # --------------------------------------------------------

    left, right = (
        st.columns(
            [1, 1]
        )
    )


    with left:

        if not pd.isna(
            environmental_score
        ):

            environment_gauge = (
                make_environment_gauge(
                    environmental_score
                )
            )

            st.plotly_chart(
                environment_gauge,
                width="stretch",
                config={
                    "displayModeBar":
                        False
                }
            )


            st.html(
                f"""
                <div class="custom-card">

                <b>
                    Environmental classification:
                </b>

                {environmental_level}

                <br><br>

                <span class="small-note">

                This index summarizes environmental
                conditions. It is not a disease
                diagnosis or epidemiological
                outbreak probability.

                </span>

                </div>
                """
            )


    with right:

        st.html(
            "🧠 AI KFD Outbreak Signal"
        )


        if prediction is None:

            st.warning(
                "The trained KFD model is currently unavailable."
            )


            if prediction_error:

                st.caption(
                    prediction_error
                )


        else:

            css_class = (

                "signal-high"

                if prediction == 1

                else "signal-low"
            )


            signal_icon = (

                "⚠️"

                if prediction == 1

                else "✅"
            )


            st.html(
                f"""
                <div class="{css_class}">

                    {signal_icon}
                    {ai_signal.upper()}
                    SIGNAL

                </div>
                """
            )


            if model_score is not None:

                score_gauge = (
                    make_model_gauge(
                        model_score
                    )
                )

                st.plotly_chart(
                    score_gauge,
                    width="stretch",
                    config={
                        "displayModeBar":
                            False
                    }
                )


            st.caption(
                "The model score is the classifier's internal "
                "positive-class score. It has not been "
                "demonstrated to be a calibrated "
                "epidemiological outbreak probability."
            )


    st.divider()


    # --------------------------------------------------------
    # Selected district trends
    # --------------------------------------------------------

    st.html(
        '<div class="section-title">'
        'Recent Environmental Trends'
        '</div>'
    )


    trend_left, trend_right = (
        st.columns(2)
    )


    with trend_left:

        temp_fig = px.line(

            district_df,

            x="Year",

            y="Temperature",

            markers=True,

            title=(
                f"Temperature Trend — "
                f"{selected_district}"
            )
        )


        temp_fig.update_layout(

            xaxis_title="Year",

            yaxis_title=(
                "Temperature (°C)"
            ),

            margin=dict(
                l=20,
                r=20,
                t=55,
                b=20
            )
        )


        st.plotly_chart(
            temp_fig,
            width="stretch"
        )


    with trend_right:

        ndvi_fig = px.line(

            district_df,

            x="Year",

            y="NDVI",

            markers=True,

            title=(
                f"NDVI Trend — "
                f"{selected_district}"
            )
        )


        ndvi_fig.update_layout(

            xaxis_title="Year",

            yaxis_title="NDVI",

            margin=dict(
                l=20,
                r=20,
                t=55,
                b=20
            )
        )


        st.plotly_chart(
            ndvi_fig,
            width="stretch"
        )


    # --------------------------------------------------------
    # Study area
    # --------------------------------------------------------

    st.html(
        '<div class="section-title">'
        'Western Ghats Study Districts'
        '</div>'
    )


    map_df = (

        environment_df[
            environment_df[
                "Year"
            ]
            ==
            selected_year
        ]

        [
            [
                "State",
                "District",
                "Latitude",
                "Longitude",
                "NDVI",
                "Temperature"
            ]
        ]

        .drop_duplicates(
            subset=[
                "State",
                "District"
            ]
        )

        .dropna(
            subset=[
                "Latitude",
                "Longitude"
            ]
        )
    )


    if not map_df.empty:

        map_fig = px.scatter_geo(

            map_df,

            lat="Latitude",

            lon="Longitude",

            hover_name="District",

            hover_data={
                "State":
                    True,

                "Temperature":
                    ":.1f",

                "NDVI":
                    ":.3f",

                "Latitude":
                    False,

                "Longitude":
                    False
            },

            size_max=15
        )


        map_fig.update_traces(
            marker=dict(
                size=12
            )
        )


        map_fig.update_geos(

            projection_type="mercator",

            showland=True,

            landcolor="#edf4ef",

            showcountries=True,

            countrycolor="#b8c8c0",

            showcoastlines=True,

            coastlinecolor="#78958a",

            center=dict(
                lat=14.0,
                lon=75.2
            ),

            lataxis_range=[
                10.5,
                17.5
            ],

            lonaxis_range=[
                72.5,
                77.5
            ]
        )


        map_fig.update_layout(

            height=500,

            margin=dict(
                l=0,
                r=0,
                t=10,
                b=0
            )
        )


        st.plotly_chart(
            map_fig,
            width="stretch"
        )


# ============================================================
# 16. CLIMATE PAGE
# ============================================================

elif page == "Climate":

    show_hero()


    st.html(
        '<div class="section-title">'
        'Climate Analysis'
        '</div>'
    )


    st.html(
        '<div class="section-subtitle">'
        'Yearly climate observations for the selected '
        'Western Ghats district.'
        '</div>'
    )


# --------------------------------------------------------
# Temperature series
# --------------------------------------------------------

temperature_columns = [
    column
    for column in [
        "Temperature",
        "Min_Temperature",
        "Max_Temperature"
    ]
    if column in district_df.columns
]


if temperature_columns:

    temperature_long = (
        district_df[
            [
                "Year",
                *temperature_columns
            ]
        ]
        .melt(
            id_vars="Year",
            var_name="Temperature_Type",
            value_name="Temperature_Value"
        )
    )


    fig_temperature = px.line(
        temperature_long,
        x="Year",
        y="Temperature_Value",
        color="Temperature_Type",
        markers=True,
        title=f"Temperature Profile — {selected_district}"
    )


    fig_temperature.update_layout(
        yaxis_title="Temperature (°C)",
        xaxis_title="Year",
        legend_title="Temperature Type"
    )


    st.plotly_chart(
        fig_temperature,
        width="stretch"
    )


# ============================================================
# 17. VEGETATION PAGE
# ============================================================

elif page == "Vegetation":

    show_hero()


    st.html(
        '<div class="section-title">'
        'Vegetation & NDVI'
        '</div>'
    )


    ndvi_left, ndvi_right = (
        st.columns(
            [1.2, 1]
        )
    )


    with ndvi_left:

        fig_ndvi = px.line(

            district_df,

            x="Year",

            y="NDVI",

            markers=True,

            title=(
                f"NDVI Through Time — "
                f"{selected_district}"
            )
        )


        fig_ndvi.update_layout(

            yaxis_title="Mean NDVI",

            xaxis_title="Year"
        )


        st.plotly_chart(
            fig_ndvi,
            width="stretch"
        )


    with ndvi_right:

        st.metric(
            "Selected NDVI",
            safe_number(
                ndvi,
                3
            )
        )


        if len(
            district_df
        ) > 1:

            first_ndvi = (

                district_df
                .sort_values(
                    "Year"
                )
                .iloc[0][
                    "NDVI"
                ]
            )


            ndvi_change = (
                ndvi
                -
                first_ndvi
            )


            st.metric(

                "Change from first available year",

                safe_number(
                    ndvi_change,
                    3
                )
            )


        st.html(
            """
            NDVI is used as a satellite-derived
            vegetation indicator. Higher values
            generally represent denser or healthier
            vegetation, while lower values indicate
            reduced vegetation cover.
            """
        )


    # --------------------------------------------------------
    # District comparison
    # --------------------------------------------------------

    comparison = (

        environment_df[
            environment_df[
                "Year"
            ]
            ==
            selected_year
        ]

        [
            [
                "District",
                "NDVI"
            ]
        ]

        .sort_values(
            "NDVI",
            ascending=False
        )
    )


    if not comparison.empty:

        compare_fig = px.bar(

            comparison,

            x="District",

            y="NDVI",

            title=(
                f"District NDVI Comparison — "
                f"{selected_year}"
            )
        )


        compare_fig.update_layout(

            xaxis_title="District",

            yaxis_title="Mean NDVI"
        )


        st.plotly_chart(
            compare_fig,
            width="stretch"
        )


# ============================================================
# 18. FOREST & TERRAIN PAGE
# ============================================================

elif page == "Forest & Terrain":

    show_hero()


    st.html(
        '<div class="section-title">'
        'Forest Disturbance & Terrain'
        '</div>'
    )


    st.warning(
        "Forest_Loss is currently treated as a "
        "district-level environmental indicator in "
        "the research dataset. It should not be "
        "interpreted as a yearly causal driver."
    )


    forest_summary = (

        environment_df

        .groupby(
            "District",
            as_index=False
        )

        .agg(

            Forest_Loss=(
                "Forest_Loss",
                "mean"
            ),

            Elevation=(
                "Elevation",
                "mean"
            )
        )
    )


    left, right = (
        st.columns(2)
    )


    with left:

        forest_fig = px.bar(

            forest_summary
            .sort_values(
                "Forest_Loss",
                ascending=False
            ),

            x="District",

            y="Forest_Loss",

            title=(
                "Forest Loss Indicator "
                "by District"
            )
        )


        forest_fig.update_layout(

            xaxis_title="District",

            yaxis_title=(
                "Forest Loss Indicator"
            )
        )


        st.plotly_chart(
            forest_fig,
            width="stretch"
        )


    with right:

        elevation_fig = px.bar(

            forest_summary
            .sort_values(
                "Elevation",
                ascending=False
            ),

            x="District",

            y="Elevation",

            title=(
                "Mean Elevation "
                "by District"
            )
        )


        elevation_fig.update_layout(

            xaxis_title="District",

            yaxis_title=(
                "Elevation (m)"
            )
        )


        st.plotly_chart(
            elevation_fig,
            width="stretch"
        )


    st.html(
        "### Selected District"
    )


    c1, c2 = (
        st.columns(2)
    )


    with c1:

        st.metric(
            "Forest Loss Indicator",
            safe_number(
                forest_loss,
                3
            )
        )


    with c2:

        st.metric(
            "Mean Elevation",
            (
                f"{safe_number(elevation, 0)} m"
            )
        )


# ============================================================
# 19. KFD PREDICTION PAGE
# ============================================================

elif page == "KFD Prediction":

    show_hero()


    st.html(
        '<div class="section-title">'
        'Experimental KFD Outbreak Classifier'
        '</div>'
    )


    st.html(
        '<div class="section-subtitle">'
        'Dynamic environmental features are passed '
        'to the spatially evaluated Random Forest V2 model.'
        '</div>'
    )


    if kfd_model is None:

        st.error(
            "kfd_outbreak_model_v2.pkl "
            "could not be found."
        )

        st.code(
            str(
                MODEL_PATH
            )
        )


    elif prediction_error:

        st.error(
            prediction_error
        )


    else:

        prediction_left, prediction_right = (
            st.columns(
                [1, 1.2]
            )
        )


        with prediction_left:

            css_class = (

                "signal-high"

                if prediction == 1

                else "signal-low"
            )


            signal_icon = (

                "⚠️"

                if prediction == 1

                else "✅"
            )


            st.html(
                f"""
                <div class="{css_class}">

                    {signal_icon}

                    {ai_signal.upper()}
                    OUTBREAK SIGNAL

                </div>
                """
            )


            st.html(
                "<br>"
            )


            st.metric(
                "District",
                selected_district
            )


            st.metric(
                "Environmental Year",
                selected_year
            )


        with prediction_right:

            if model_score is not None:

                model_gauge = (
                    make_model_gauge(
                        model_score
                    )
                )


                st.plotly_chart(
                    model_gauge,
                    width="stretch",
                    config={
                        "displayModeBar":
                            False
                    }
                )


        st.warning(
            "The model score must not be interpreted "
            "as a calibrated probability that a KFD "
            "outbreak will occur. This is an experimental "
            "environmental classification model."
        )


        # ----------------------------------------------------
        # Model inputs
        # ----------------------------------------------------

        st.html(
            "Model Inputs"
        )


        model_features = (
            model_metadata.get(

                "features",

                [
                    "Temperature",
                    "Rainfall",
                    "NDVI",
                    "Temperature_Range"
                ]
            )
        )


        input_table = pd.DataFrame(

            {

                "Feature":
                    model_features,

                "Value":
                    [
                        row.get(
                            feature,
                            np.nan
                        )

                        for feature
                        in model_features
                    ]
            }
        )


        st.dataframe(

            input_table,

            width="stretch",

            hide_index=True
        )


        # ----------------------------------------------------
        # Feature importance
        # ----------------------------------------------------

        st.html(
            "Model Feature Importance"
        )


        try:

            rf_estimator = (
                kfd_model
                .named_steps[
                    "model"
                ]
            )


            if hasattr(
                rf_estimator,
                "feature_importances_"
            ):

                importance_df = (
                    pd.DataFrame(
                        {
                            "Feature":
                                model_features,

                            "Importance":
                                rf_estimator
                                .feature_importances_
                        }
                    )

                    .sort_values(
                        "Importance",
                        ascending=True
                    )
                )


                importance_fig = px.bar(

                    importance_df,

                    x="Importance",

                    y="Feature",

                    orientation="h",

                    title=(
                        "Dynamic Random Forest "
                        "Feature Importance"
                    )
                )


                st.plotly_chart(
                    importance_fig,
                    width="stretch"
                )


                st.caption(
                    "Feature importance describes the "
                    "model's use of variables. It does "
                    "not demonstrate causality."
                )


        except Exception:

            st.info(
                "Feature importance is unavailable "
                "for the loaded model."
            )


        # ----------------------------------------------------
        # Model metadata
        # ----------------------------------------------------

        st.html(
            "Research Validation"
        )


        m1, m2, m3, m4 = (
            st.columns(4)
        )


        with m1:

            st.metric(

                "Training observations",

                model_metadata.get(
                    "training_observations",
                    "46"
                )
            )


        with m2:

            st.metric(

                "Study districts",

                model_metadata.get(
                    "districts",
                    "9"
                )
            )


        with m3:

            spatial_ba = (
                model_metadata.get(
                    "spatial_balanced_accuracy",
                    0.565
                )
            )


            st.metric(

                "Spatial Balanced Accuracy",

                (
                    f"{float(spatial_ba):.3f}"
                )
            )


        with m4:

            spatial_auc = (
                model_metadata.get(
                    "spatial_roc_auc",
                    0.544
                )
            )


            st.metric(

                "Spatial ROC-AUC",

                (
                    f"{float(spatial_auc):.3f}"
                )
            )


        st.info(
            "Primary evaluation used "
            "Leave-One-District-Out spatial validation. "
            "The model showed modest cross-district "
            "generalization, while temporal validation "
            "within previously represented regions was stronger."
        )


        # ----------------------------------------------------
        # Download selected assessment
        # ----------------------------------------------------

        assessment = pd.DataFrame(
            [
                {
                    "State":
                        row.get(
                            "State",
                            ""
                        ),

                    "District":
                        selected_district,

                    "Year":
                        selected_year,

                    "Temperature":
                        temperature,

                    "Rainfall":
                        rainfall,

                    "NDVI":
                        ndvi,

                    "Temperature_Range":
                        temp_range,

                    "Environmental_Risk_Score":
                        environmental_score,

                    "Environmental_Risk_Level":
                        environmental_level,

                    "AI_KFD_Signal":
                        ai_signal,

                    "Model_Score":
                        model_score
                }
            ]
        )


        st.download_button(

            label=(
                "Download Selected Assessment"
            ),

            data=(
                assessment
                .to_csv(
                    index=False
                )
                .encode(
                    "utf-8"
                )
            ),

            file_name=(
                f"kfd_assessment_"
                f"{selected_district.replace(' ', '_')}_"
                f"{selected_year}.csv"
            ),

            mime="text/csv"
        )


# ============================================================
# 20. METHODOLOGY PAGE
# ============================================================

elif page == "Methodology":

    show_hero()


    st.html(
        '<div class="section-title">'
        'Research Methodology'
        '</div>'
    )


    st.html(
        """
        <div class="method-card">

        <b>1. Study Region</b><br>

        Nine KFD-relevant districts across Karnataka,
        Kerala, Goa and Maharashtra were included in
        the Western Ghats study area.

        </div>


        <div class="method-card">

        <b>2. Climate Variables</b><br>

        Annual temperature and rainfall variables were
        assembled from station/interpolated climate
        observations.

        </div>


        <div class="method-card">

        <b>3. Satellite Vegetation</b><br>

        Sentinel-2 imagery was processed in Google
        Earth Engine to derive NDVI-based vegetation
        indicators.

        </div>


        <div class="method-card">

        <b>4. Forest & Terrain</b><br>

        Hansen Global Forest Change data and SRTM
        elevation information were incorporated into
        the environmental dataset.

        </div>


        <div class="method-card">

        <b>5. Verified Disease Targets</b><br>

        Historical KFD occurrence labels were compiled
        from source-backed surveillance and government
        records and merged with district-year
        environmental observations.

        </div>


        <div class="method-card">

        <b>6. Machine Learning</b><br>

        Logistic Regression, Decision Tree,
        Random Forest and XGBoost classifiers were
        compared. Spatial generalization was evaluated
        using Leave-One-District-Out validation.

        </div>


        <div class="method-card">

        <b>7. Feature-Set Ablation</b><br>

        Dynamic environmental features generalized
        better across unseen districts than the
        static-only feature set. The primary deployment
        model therefore uses Temperature, Rainfall,
        NDVI and Temperature Range.

        </div>
        """
    )


    st.html(
        "Final Research Pipeline"
    )


    st.code(
        """
Climate Data
      +
Sentinel-2 NDVI
      +
Forest Change
      +
Elevation
      ↓
Master Environmental Dataset
      ↓
Feature Engineering
      +
Verified KFD Outbreak Labels
      ↓
Model Comparison
      ↓
Leave-One-District-Out Validation
      ↓
Feature-Set Ablation
      ↓
Dynamic Random Forest V2
      ↓
Experimental KFD Outbreak Signal
        """,
        language=None
    )


    st.html(
        "Interpretation of Current Results"
    )


    results_table = pd.DataFrame(
        {

            "Experiment": [

                "Dynamic Random Forest — Spatial",

                "Combined Decision Tree — Spatial",

                "Decision Tree — Temporal"
            ],

            "Balanced Accuracy": [

                0.565,
                0.543,
                0.833
            ],

            "F1": [

                0.545,
                0.618,
                0.909
            ],

            "ROC-AUC": [

                0.544,
                0.354,
                0.800
            ]
        }
    )


    st.dataframe(

        results_table,

        width="stretch",

        hide_index=True
    )


    st.html(
        """
        The current results suggest stronger temporal
        predictive signal within represented regions
        than geographic transfer to completely unseen
        districts. Consequently, the dashboard should
        be interpreted as a research prototype rather
        than an operational disease-surveillance system.
        """
    )


    st.html(
        "Key Limitations"
    )


    st.html(
        """
        - Small verified training dataset.
        - Unequal historical climate coverage across districts.
        - Historical surveillance periods are not perfectly
          harmonized across all sources.
        - Satellite extraction uses representative district
          regions rather than complete administrative polygons.
        - The present forest-loss feature has limited temporal
          variation.
        - The classifier score is not a calibrated outbreak
          probability.
        - Environmental associations do not establish causality.
        """
    )


# ============================================================
# 21. FOOTER
# ============================================================

st.divider()

st.html(
    """
    <div style="
        text-align:center;
        color:#74837e;
        font-size:13px;
        padding:10px 0 15px 0;
    ">

    SpilloverAI • Zoonotic Spillover Predictor
    <br>

    Research prototype for environmental analysis of
    Kyasanur Forest Disease in the Western Ghats.

    </div>
    """
)