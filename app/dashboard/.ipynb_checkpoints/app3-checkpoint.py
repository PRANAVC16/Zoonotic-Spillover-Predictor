# ============================================================
# SpilloverAI
# Editorial Research Dashboard
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
# 1. PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SpilloverAI | KFD",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# 2. PROJECT PATHS
# ============================================================

def find_project_root():

    current = Path(__file__).resolve().parent

    for folder in [current, *current.parents]:

        if (
            (folder / "data").exists()
            and
            (folder / "notebooks").exists()
        ):
            return folder

    return Path(__file__).resolve().parents[2]


ROOT = find_project_root()

DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "outputs" / "models"

DASHBOARD_DATA_PATH = (
    DATA_DIR / "dashboard_environment_data.csv"
)

MASTER_FEATURES_PATH = (
    DATA_DIR / "master_features.csv"
)

MODEL_PATH = (
    MODEL_DIR / "kfd_outbreak_model_v2.pkl"
)

METADATA_PATH = (
    MODEL_DIR / "kfd_outbreak_model_v2_metadata.json"
)


# ============================================================
# 3. HERO IMAGE
# ============================================================

# You can replace this later with your own Western Ghats image.
HERO_IMAGE_URL = (
    "https://images.unsplash.com/"
    "photo-1464822759023-fed622ff2c3b"
    "?auto=format&fit=crop&w=1600&q=85"
)


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
# 5. GLOBAL CSS
# ============================================================

st.html(
    """
<style>

/* ---------------------------------------------------------
   Fonts / page
--------------------------------------------------------- */

html,
body,
[class*="css"] {
    font-family:
        Inter,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;
}

.stApp {
    background: #b6aaa0;
    color: #101817;
}

.block-container {
    max-width: 1320px;
    padding-top: 2rem;
    padding-bottom: 5rem;
}


/* ---------------------------------------------------------
   Hide Streamlit chrome
--------------------------------------------------------- */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header[data-testid="stHeader"] {
    background: transparent;
}


/* ---------------------------------------------------------
   Main shell
--------------------------------------------------------- */

.editorial-shell {

    background: #f3ede7;

    border-radius: 30px;

    padding:
        28px
        34px
        38px
        34px;

    margin-bottom: 24px;

    box-shadow:
        0 20px 60px
        rgba(25, 18, 14, 0.10);
}


/* ---------------------------------------------------------
   Top brand
--------------------------------------------------------- */

.brand-row {

    display: flex;

    align-items: center;

    justify-content: space-between;

    margin-bottom: 14px;
}

.brand {

    font-size: 15px;

    font-weight: 700;

    letter-spacing: -0.02em;
}

.brand-symbol {

    display: inline-flex;

    align-items: center;

    justify-content: center;

    width: 32px;

    height: 32px;

    border-radius: 50%;

    background: #111817;

    color: #f3ede7;

    margin-right: 9px;
}


/* ---------------------------------------------------------
   Navigation
--------------------------------------------------------- */

div[data-testid="stRadio"] > label {
    display: none;
}

div[role="radiogroup"] {

    display: flex;

    flex-direction: row;

    gap: 8px;

    flex-wrap: wrap;
}

div[role="radiogroup"] label {

    background: transparent;

    padding:
        8px
        13px;

    border-radius: 999px;

    font-size: 13px;

    color: #5e5a55;

    transition: 0.2s;
}

div[role="radiogroup"] label:hover {

    background: #e6ddd5;

    color: #111817;
}


/* ---------------------------------------------------------
   Hero
--------------------------------------------------------- */

.hero {

    display: grid;

    grid-template-columns:
        1fr
        1fr;

    min-height: 590px;

    border-radius: 26px;

    overflow: hidden;

    background: #f7f2ed;

    border:
        1px solid
        rgba(20, 25, 24, 0.08);
}

.hero-left {

    position: relative;

    padding:
        54px
        52px;

    display: flex;

    flex-direction: column;

    justify-content: space-between;
}

.hero-eyebrow {

    font-size: 12px;

    text-transform: uppercase;

    letter-spacing: 0.14em;

    color: #6f716b;

    margin-bottom: 25px;
}

.hero-title {

    font-size:
        clamp(
            3.7rem,
            6vw,
            6.5rem
        );

    font-weight: 400;

    letter-spacing: -0.065em;

    line-height: 0.88;

    color: #101817;

    max-width: 690px;
}

.hero-title strong {

    font-weight: 500;

    color: #385d4f;
}

.hero-description {

    max-width: 500px;

    font-size: 14px;

    line-height: 1.7;

    color: #5e625d;

    margin-top: 28px;
}

.hero-meta {

    font-size: 13px;

    line-height: 1.9;

    color: #131918;
}

.hero-meta strong {

    font-weight: 650;
}


/* ---------------------------------------------------------
   Hero image
--------------------------------------------------------- */

.hero-right {

    position: relative;

    min-height: 590px;

    background-size: cover;

    background-position: center;
}

.hero-right::after {

    content: "";

    position: absolute;

    inset: 0;

    background:
        linear-gradient(
            180deg,
            rgba(0,0,0,0.02),
            rgba(0,0,0,0.18)
        );
}


/* ---------------------------------------------------------
   Floating ML card
--------------------------------------------------------- */

.model-card {

    position: absolute;

    z-index: 2;

    right: 34px;

    bottom: 34px;

    width: min(
        340px,
        calc(100% - 68px)
    );

    background:
        rgba(
            248,
            243,
            238,
            0.96
        );

    backdrop-filter: blur(15px);

    border-radius: 22px;

    padding: 22px;

    box-shadow:
        0 14px 45px
        rgba(0,0,0,0.20);
}

.model-kicker {

    font-size: 11px;

    letter-spacing: 0.12em;

    text-transform: uppercase;

    color: #666c66;

    margin-bottom: 8px;
}

.model-signal {

    font-size: 30px;

    letter-spacing: -0.04em;

    font-weight: 600;

    margin-bottom: 13px;
}

.model-divider {

    height: 1px;

    background: #d6cec6;

    margin: 14px 0;
}

.model-meta {

    display: flex;

    justify-content: space-between;

    gap: 10px;

    font-size: 12px;

    line-height: 1.8;

    color: #5f625e;
}


/* ---------------------------------------------------------
   Controls
--------------------------------------------------------- */

.filter-label {

    font-size: 11px;

    text-transform: uppercase;

    letter-spacing: 0.11em;

    color: #76736e;

    margin-bottom: 5px;
}


/* ---------------------------------------------------------
   Section typography
--------------------------------------------------------- */

.section-number {

    font-size: 12px;

    letter-spacing: 0.12em;

    color: #77736e;

    margin-bottom: 18px;
}

.section-heading {

    font-size:
        clamp(
            2.5rem,
            4vw,
            4.3rem
        );

    line-height: 0.98;

    letter-spacing: -0.055em;

    font-weight: 400;

    color: #101817;

    margin-bottom: 14px;
}

.section-copy {

    color: #666b65;

    font-size: 14px;

    line-height: 1.7;

    max-width: 650px;

    margin-bottom: 30px;
}


/* ---------------------------------------------------------
   Metrics
--------------------------------------------------------- */

.metric-grid {

    display: grid;

    grid-template-columns:
        repeat(4, 1fr);

    gap: 1px;

    background: #d8d0c8;

    border:
        1px solid
        #d8d0c8;

    border-radius: 20px;

    overflow: hidden;

    margin: 24px 0;
}

.metric-card {

    background: #f8f4ef;

    padding: 28px 24px;
}

.metric-label {

    color: #77736e;

    font-size: 11px;

    text-transform: uppercase;

    letter-spacing: 0.1em;
}

.metric-value {

    margin-top: 11px;

    font-size: 34px;

    letter-spacing: -0.04em;

    color: #101817;
}

.metric-caption {

    font-size: 11px;

    color: #8a8680;

    margin-top: 5px;
}


/* ---------------------------------------------------------
   Editorial cards
--------------------------------------------------------- */

.editorial-card {

    background: #f8f4ef;

    border:
        1px solid
        #ddd5cd;

    border-radius: 22px;

    padding: 28px;

    min-height: 100%;
}

.card-kicker {

    font-size: 11px;

    letter-spacing: 0.1em;

    text-transform: uppercase;

    color: #7a7772;
}

.card-value {

    font-size: 45px;

    letter-spacing: -0.05em;

    margin: 12px 0;

    color: #111817;
}

.card-description {

    font-size: 13px;

    line-height: 1.65;

    color: #686b66;
}


/* ---------------------------------------------------------
   Research signal
--------------------------------------------------------- */

.signal-elevated {
    color: #9b3c32;
}

.signal-lower {
    color: #28614c;
}


/* ---------------------------------------------------------
   Tables
--------------------------------------------------------- */

[data-testid="stDataFrame"] {

    border-radius: 18px;

    overflow: hidden;
}


/* ---------------------------------------------------------
   Responsive
--------------------------------------------------------- */

@media (max-width: 900px) {

    .hero {
        grid-template-columns: 1fr;
    }

    .hero-right {
        min-height: 450px;
    }

    .metric-grid {
        grid-template-columns:
            repeat(2, 1fr);
    }

}

@media (max-width: 600px) {

    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .hero-left {
        padding: 36px 26px;
    }

    .metric-grid {
        grid-template-columns: 1fr;
    }

}

</style>
"""
)


# ============================================================
# 6. DATA LOADING
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
            "Environmental dataset not found."
        )


    # --------------------------------------------------------
    # Clean years
    # --------------------------------------------------------

    df["Year"] = pd.to_numeric(
        df["Year"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["Year"]
    )

    df["Year"] = (
        df["Year"].astype(int)
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

        df["Temperature_Range"] = (

            df["Max_Temperature"]
            -
            df["Min_Temperature"]
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
            column in df.columns
            for column in required
        ):

            df["Environmental_Risk_Score"] = (

                0.30
                *
                df["Temperature"].rank(
                    pct=True
                )

                +

                0.25
                *
                df["Rainfall"].rank(
                    pct=True
                )

                +

                0.20
                *
                (
                    1
                    -
                    df["NDVI"].rank(
                        pct=True
                    )
                )

                +

                0.25
                *
                df["Forest_Loss"].rank(
                    pct=True
                )

            ) * 100


    # --------------------------------------------------------
    # Risk label
    # --------------------------------------------------------

    if (
        "Environmental_Risk_Level"
        not in df.columns
    ):

        def classify_risk(score):

            if pd.isna(score):
                return "Unavailable"

            if score < 33:
                return "Low"

            if score < 66:
                return "Moderate"

            return "High"


        df["Environmental_Risk_Level"] = (

            df[
                "Environmental_Risk_Score"
            ].apply(
                classify_risk
            )
        )


    # --------------------------------------------------------
    # Coordinates
    # --------------------------------------------------------

    df["Latitude"] = (
        df["District"].map(
            lambda district:
                DISTRICT_COORDINATES.get(
                    district,
                    (np.nan, np.nan)
                )[0]
        )
    )

    df["Longitude"] = (
        df["District"].map(
            lambda district:
                DISTRICT_COORDINATES.get(
                    district,
                    (np.nan, np.nan)
                )[1]
        )
    )


    return df


@st.cache_resource
def load_model():

    if MODEL_PATH.exists():

        return joblib.load(
            MODEL_PATH
        )

    return None


@st.cache_data
def load_metadata():

    if METADATA_PATH.exists():

        with open(
            METADATA_PATH,
            "r"
        ) as file:

            return json.load(file)

    return {}


try:

    environment_df = (
        load_environment_data()
    )

except Exception as error:

    st.error(
        "Unable to load environmental data."
    )

    st.exception(error)

    st.stop()


kfd_model = load_model()

model_metadata = load_metadata()


# ============================================================
# 7. HELPERS
# ============================================================

def fmt(value, decimals=2):

    if pd.isna(value):
        return "N/A"

    return (
        f"{float(value):.{decimals}f}"
    )


def predict_kfd(row):

    if kfd_model is None:

        return None, None


    features = model_metadata.get(
        "features",
        [
            "Temperature",
            "Rainfall",
            "NDVI",
            "Temperature_Range"
        ]
    )


    if any(
        feature not in row.index
        for feature in features
    ):

        return None, None


    input_df = pd.DataFrame(
        [
            {
                feature:
                    row[feature]

                for feature
                in features
            }
        ]
    )


    prediction = int(
        kfd_model.predict(
            input_df
        )[0]
    )


    score = None


    if hasattr(
        kfd_model,
        "predict_proba"
    ):

        score = float(
            kfd_model.predict_proba(
                input_df
            )[0, 1]
        )


    return prediction, score


def style_figure(
    fig,
    height=430
):

    fig.update_layout(

        height=height,

        paper_bgcolor="rgba(0,0,0,0)",

        plot_bgcolor="rgba(0,0,0,0)",

        font=dict(
            color="#232927",
            size=12
        ),

        title=dict(
            font=dict(
                size=18
            ),
            x=0
        ),

        margin=dict(
            l=20,
            r=20,
            t=55,
            b=30
        ),

        legend_title_text=""
    )


    fig.update_xaxes(
        showgrid=False,
        zeroline=False
    )


    fig.update_yaxes(
        gridcolor="rgba(30,30,30,0.08)",
        zeroline=False
    )


    return fig


def section_header(
    number,
    title,
    copy
):

    st.html(
        f"""
<div class="section-number">
    {number}
</div>

<div class="section-heading">
    {title}
</div>

<div class="section-copy">
    {copy}
</div>
        """
    )


# ============================================================
# 8. TOP BRAND
# ============================================================

st.html(
    """
<div class="brand-row">

    <div class="brand">

        <span class="brand-symbol">
            ◐
        </span>

        SpilloverAI

    </div>

    <div style="
        font-size:12px;
        color:#6d6b67;
        letter-spacing:0.06em;
    ">
        KFD / WESTERN GHATS
    </div>

</div>
    """
)


# ============================================================
# 9. NAVIGATION
# ============================================================

page = st.radio(

    "Navigation",

    [
        "Overview",
        "Climate",
        "Vegetation",
        "Forest & Terrain",
        "KFD Model",
        "Research"
    ],

    horizontal=True
)


st.markdown(
    "<div style='height:10px'></div>",
    unsafe_allow_html=True
)


# ============================================================
# 10. FILTERS
# ============================================================

filter_left, filter_mid, filter_right = (
    st.columns(
        [1.5, 1, 4]
    )
)


with filter_left:

    districts = sorted(
        environment_df[
            "District"
        ]
        .dropna()
        .unique()
    )

    selected_district = (
        st.selectbox(
            "District",
            districts
        )
    )


district_df = (

    environment_df[
        environment_df["District"]
        ==
        selected_district
    ]

    .sort_values("Year")

    .copy()
)


with filter_mid:

    years = sorted(
        district_df[
            "Year"
        ].unique(),
        reverse=True
    )

    selected_year = (
        st.selectbox(
            "Year",
            years
        )
    )


selected_record = (

    district_df[
        district_df["Year"]
        ==
        selected_year
    ]
)


if selected_record.empty:

    st.error(
        "No environmental data found."
    )

    st.stop()


row = selected_record.iloc[0]


# ============================================================
# 11. CURRENT VALUES
# ============================================================

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


prediction, model_score = (
    predict_kfd(row)
)


if prediction == 1:

    ai_signal = "Elevated"

elif prediction == 0:

    ai_signal = "Lower"

else:

    ai_signal = "Unavailable"


signal_class = (

    "signal-elevated"

    if prediction == 1

    else "signal-lower"
)


score_text = (

    fmt(
        model_score,
        3
    )

    if model_score
    is not None

    else "N/A"
)


# ============================================================
# 12. HERO
# ============================================================

def render_hero():

    st.html(
        f"""
<div class="hero">

    <div class="hero-left">

        <div>

            <div class="hero-eyebrow">
                Environmental intelligence /
                zoonotic disease
            </div>

            <div class="hero-title">

                Environmental
                intelligence for
                <strong>spillover risk.</strong>

            </div>

            <div class="hero-description">

                Satellite, climate and machine-learning
                analysis of Kyasanur Forest Disease
                across Western Ghats study districts.

            </div>

        </div>


        <div class="hero-meta">

            <strong>{selected_district}</strong><br>

            Environmental year /
            {selected_year}<br>

            KFD research framework /
            2015–2024

        </div>

    </div>


    <div
        class="hero-right"
        style="
            background-image:
            url('{HERO_IMAGE_URL}');
        "
    >

        <div class="model-card">

            <div class="model-kicker">
                AI outbreak signal
            </div>

            <div
                class="
                    model-signal
                    {signal_class}
                "
            >
                {ai_signal}
            </div>

            <div class="model-divider"></div>

            <div class="model-meta">

                <span>
                    Model score
                </span>

                <strong>
                    {score_text}
                </strong>

            </div>

            <div class="model-meta">

                <span>
                    Model
                </span>

                <strong>
                    Dynamic RF V2
                </strong>

            </div>

            <div class="model-meta">

                <span>
                    Region
                </span>

                <strong>
                    {selected_district}
                </strong>

            </div>

        </div>

    </div>

</div>
        """
    )


# ============================================================
# 13. OVERVIEW
# ============================================================

if page == "Overview":

    render_hero()


    st.markdown(
        "<div style='height:45px'></div>",
        unsafe_allow_html=True
    )


    section_header(

        "01 / OVERVIEW",

        "Reading the environment.",

        (
            "A compact view of the environmental "
            "conditions used by the SpilloverAI "
            "research framework."
        )
    )


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    st.html(
        f"""
<div class="metric-grid">

    <div class="metric-card">

        <div class="metric-label">
            Temperature
        </div>

        <div class="metric-value">
            {fmt(temperature, 1)}°
        </div>

        <div class="metric-caption">
            Mean annual °C
        </div>

    </div>


    <div class="metric-card">

        <div class="metric-label">
            Rainfall
        </div>

        <div class="metric-value">
            {fmt(rainfall, 0)}
        </div>

        <div class="metric-caption">
            Annual mm
        </div>

    </div>


    <div class="metric-card">

        <div class="metric-label">
            Vegetation
        </div>

        <div class="metric-value">
            {fmt(ndvi, 3)}
        </div>

        <div class="metric-caption">
            NDVI
        </div>

    </div>


    <div class="metric-card">

        <div class="metric-label">
            Environmental index
        </div>

        <div class="metric-value">
            {fmt(environmental_score, 0)}
        </div>

        <div class="metric-caption">
            {environmental_level} / 100
        </div>

    </div>

</div>
        """
    )


    # --------------------------------------------------------
    # Narrative + trends
    # --------------------------------------------------------

    left, right = st.columns(
        [1, 1.7]
    )


    with left:

        st.html(
            f"""
<div class="editorial-card">

    <div class="card-kicker">
        Current research assessment
    </div>

    <div
        class="
            card-value
            {signal_class}
        "
    >
        {ai_signal}
    </div>

    <div class="card-description">

        The AI output is produced by the
        spatially evaluated Dynamic Random
        Forest V2 classifier.

        <br><br>

        Its score should be interpreted as
        an experimental model signal rather
        than a calibrated epidemiological
        probability.

    </div>

</div>
            """
        )


    with right:

        fig = px.line(

            district_df,

            x="Year",

            y=[
                "Temperature",
                "NDVI"
            ],

            markers=True,

            title=(
                f"Environmental trajectory · "
                f"{selected_district}"
            )
        )


        fig = style_figure(
            fig,
            390
        )


        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "displayModeBar":
                    False
            }
        )


    st.markdown(
        "<div style='height:35px'></div>",
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # Map
    # --------------------------------------------------------

    section_header(

        "02 / STUDY AREA",

        "Nine districts. One ecological system.",

        (
            "The research dataset spans KFD-relevant "
            "regions across Karnataka, Kerala, Goa "
            "and Maharashtra."
        )
    )


    map_df = (

        environment_df[
            environment_df["Year"]
            ==
            selected_year
        ]

        [
            [
                "District",
                "State",
                "Latitude",
                "Longitude",
                "Temperature",
                "NDVI"
            ]
        ]

        .drop_duplicates(
            "District"
        )

        .dropna(
            subset=[
                "Latitude",
                "Longitude"
            ]
        )
    )


    fig_map = px.scatter_geo(

        map_df,

        lat="Latitude",

        lon="Longitude",

        hover_name="District",

        hover_data={
            "State": True,
            "Temperature": ":.1f",
            "NDVI": ":.3f",
            "Latitude": False,
            "Longitude": False
        }
    )


    fig_map.update_traces(
        marker=dict(
            size=14,
            color="#385d4f"
        )
    )


    fig_map.update_geos(

        projection_type="mercator",

        showland=True,

        landcolor="#eee6de",

        showocean=True,

        oceancolor="#d9e3df",

        showcountries=True,

        countrycolor="#8e9690",

        center=dict(
            lat=14,
            lon=75
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


    fig_map.update_layout(

        height=520,

        paper_bgcolor="rgba(0,0,0,0)",

        margin=dict(
            l=0,
            r=0,
            t=0,
            b=0
        )
    )


    st.plotly_chart(
        fig_map,
        use_container_width=True,
        config={
            "displayModeBar":
                False
        }
    )


# ============================================================
# 14. CLIMATE
# ============================================================

elif page == "Climate":

    render_hero()


    st.markdown(
        "<div style='height:45px'></div>",
        unsafe_allow_html=True
    )


    section_header(

        "01 / CLIMATE",

        "Climate through time.",

        (
            "Temperature and rainfall conditions "
            "for the selected study district."
        )
    )


    temperature_columns = [

        column

        for column in [

            "Temperature",
            "Min_Temperature",
            "Max_Temperature"

        ]

        if column
        in district_df.columns
    ]


    if temperature_columns:

        temp_long = (

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


        fig = px.line(

            temp_long,

            x="Year",

            y="Temperature_Value",

            color="Temperature_Type",

            markers=True,

            title="Temperature profile"
        )


        fig = style_figure(
            fig,
            450
        )


        fig.update_yaxes(
            title="Temperature (°C)"
        )


        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "displayModeBar":
                    False
            }
        )


    st.markdown(
        "<div style='height:20px'></div>",
        unsafe_allow_html=True
    )


    rainfall_fig = px.bar(

        district_df,

        x="Year",

        y="Rainfall",

        title="Annual rainfall"
    )


    rainfall_fig = style_figure(
        rainfall_fig,
        430
    )


    rainfall_fig.update_yaxes(
        title="Rainfall (mm)"
    )


    st.plotly_chart(
        rainfall_fig,
        use_container_width=True,
        config={
            "displayModeBar":
                False
        }
    )


    st.markdown("#### Climate records")


    climate_columns = [

        column

        for column in [

            "Year",
            "Temperature",
            "Min_Temperature",
            "Max_Temperature",
            "Temperature_Range",
            "Rainfall"

        ]

        if column
        in district_df.columns
    ]


    st.dataframe(

        district_df[
            climate_columns
        ].sort_values(
            "Year",
            ascending=False
        ),

        use_container_width=True,

        hide_index=True
    )


# ============================================================
# 15. VEGETATION
# ============================================================

elif page == "Vegetation":

    render_hero()


    st.markdown(
        "<div style='height:45px'></div>",
        unsafe_allow_html=True
    )


    section_header(

        "02 / VEGETATION",

        "Watching vegetation change.",

        (
            "Sentinel-derived NDVI provides a "
            "consistent indicator of vegetation "
            "condition across the study region."
        )
    )


    left, right = st.columns(
        [1.7, 1]
    )


    with left:

        fig = px.line(

            district_df,

            x="Year",

            y="NDVI",

            markers=True,

            title=(
                f"NDVI trajectory · "
                f"{selected_district}"
            )
        )


        fig = style_figure(
            fig,
            420
        )


        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "displayModeBar":
                    False
            }
        )


    with right:

        first_ndvi = (

            district_df
            .sort_values("Year")
            .iloc[0]["NDVI"]
        )


        ndvi_change = (
            ndvi
            -
            first_ndvi
        )


        st.html(
            f"""
<div class="editorial-card">

    <div class="card-kicker">
        Current NDVI
    </div>

    <div class="card-value">
        {fmt(ndvi, 3)}
    </div>

    <div class="card-description">

        Change from first available year:

        <br><br>

        <strong>
            {fmt(ndvi_change, 3)}
        </strong>

        <br><br>

        NDVI represents vegetation density and
        condition rather than disease occurrence.

    </div>

</div>
            """
        )


    comparison = (

        environment_df[
            environment_df["Year"]
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


    fig = px.bar(

        comparison,

        x="District",

        y="NDVI",

        title=(
            f"Vegetation across districts · "
            f"{selected_year}"
        )
    )


    fig = style_figure(
        fig,
        430
    )


    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar":
                False
        }
    )


# ============================================================
# 16. FOREST & TERRAIN
# ============================================================

elif page == "Forest & Terrain":

    render_hero()


    st.markdown(
        "<div style='height:45px'></div>",
        unsafe_allow_html=True
    )


    section_header(

        "03 / LANDSCAPE",

        "Forest disturbance & terrain.",

        (
            "Landscape variables provide ecological "
            "context, but our ablation study showed "
            "that static features generalized poorly "
            "across unseen districts."
        )
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


    left, right = st.columns(2)


    with left:

        fig = px.bar(

            forest_summary
            .sort_values(
                "Forest_Loss",
                ascending=False
            ),

            x="District",

            y="Forest_Loss",

            title="Forest loss indicator"
        )


        fig = style_figure(
            fig,
            430
        )


        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "displayModeBar":
                    False
            }
        )


    with right:

        fig = px.bar(

            forest_summary
            .sort_values(
                "Elevation",
                ascending=False
            ),

            x="District",

            y="Elevation",

            title="Mean elevation"
        )


        fig = style_figure(
            fig,
            430
        )


        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "displayModeBar":
                    False
            }
        )


    st.info(
        "Forest Loss and Elevation are retained for "
        "environmental interpretation, but the final "
        "spatial KFD model uses only dynamic features."
    )


# ============================================================
# 17. KFD MODEL
# ============================================================

elif page == "KFD Model":

    render_hero()


    st.markdown(
        "<div style='height:45px'></div>",
        unsafe_allow_html=True
    )


    section_header(

        "04 / MODEL",

        "An experimental outbreak signal.",

        (
            "The primary classifier uses dynamic "
            "environmental variables selected after "
            "feature-set ablation and spatial validation."
        )
    )


    if kfd_model is None:

        st.error(
            "KFD model V2 could not be loaded."
        )

        st.code(
            str(MODEL_PATH)
        )


    else:

        left, right = st.columns(
            [1, 1.45]
        )


        with left:

            st.html(
                f"""
<div class="editorial-card">

    <div class="card-kicker">
        AI KFD signal
    </div>

    <div
        class="
            card-value
            {signal_class}
        "
    >
        {ai_signal}
    </div>

    <div class="card-description">

        Model score

        <br>

        <strong style="
            font-size:25px;
            color:#111817;
        ">
            {score_text}
        </strong>

        <br><br>

        The score is not a calibrated
        epidemiological outbreak probability.

    </div>

</div>
                """
            )


        with right:

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


            feature_table = pd.DataFrame(
                {

                    "Environmental feature":
                        model_features,

                    "Observed value":
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


            st.markdown(
                "#### Model inputs"
            )


            st.dataframe(

                feature_table,

                use_container_width=True,

                hide_index=True
            )


    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    if kfd_model is not None:

        estimator = (
            kfd_model
            .named_steps["model"]
        )


        if hasattr(
            estimator,
            "feature_importances_"
        ):

            features = (
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


            importance_df = (

                pd.DataFrame(
                    {

                        "Feature":
                            features,

                        "Importance":
                            estimator
                            .feature_importances_
                    }
                )

                .sort_values(
                    "Importance",
                    ascending=True
                )
            )


            fig = px.bar(

                importance_df,

                x="Importance",

                y="Feature",

                orientation="h",

                title=(
                    "Environmental feature importance"
                )
            )


            fig = style_figure(
                fig,
                400
            )


            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    "displayModeBar":
                        False
                }
            )


    # --------------------------------------------------------
    # Validation summary
    # --------------------------------------------------------

    section_header(

        "05 / VALIDATION",

        "Promising temporally. Limited spatially.",

        (
            "Spatial validation tested unseen districts, "
            "while temporal validation examined later "
            "observations in represented regions."
        )
    )


    validation_df = pd.DataFrame(
        {

            "Experiment": [

                "Dynamic Random Forest · Spatial",

                "Combined Decision Tree · Spatial",

                "Decision Tree · Temporal"
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

        validation_df,

        use_container_width=True,

        hide_index=True
    )


    st.warning(
        "Current results support the dashboard as a "
        "research prototype, not an operational "
        "public-health forecasting system."
    )


    # --------------------------------------------------------
    # Download
    # --------------------------------------------------------

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

        "Download selected assessment",

        data=assessment
        .to_csv(
            index=False
        )
        .encode(
            "utf-8"
        ),

        file_name=(
            f"kfd_assessment_"
            f"{selected_district.replace(' ', '_')}_"
            f"{selected_year}.csv"
        ),

        mime="text/csv"
    )


# ============================================================
# 18. RESEARCH
# ============================================================

elif page == "Research":

    render_hero()


    st.markdown(
        "<div style='height:45px'></div>",
        unsafe_allow_html=True
    )


    section_header(

        "06 / RESEARCH",

        "How SpilloverAI was built.",

        (
            "The research framework integrates "
            "multi-source environmental observations "
            "with source-backed KFD outbreak labels."
        )
    )


    methodology = [

        (
            "01",
            "Climate",
            (
                "Annual temperature and rainfall "
                "observations were assembled for "
                "Western Ghats study districts."
            )
        ),

        (
            "02",
            "Satellite vegetation",
            (
                "Sentinel-2 imagery was processed "
                "through Google Earth Engine to "
                "derive NDVI."
            )
        ),

        (
            "03",
            "Landscape",
            (
                "Hansen Global Forest Change and "
                "SRTM elevation were incorporated "
                "as contextual environmental layers."
            )
        ),

        (
            "04",
            "Disease targets",
            (
                "KFD occurrence labels were compiled "
                "from source-backed surveillance and "
                "government records."
            )
        ),

        (
            "05",
            "Machine learning",
            (
                "Logistic Regression, Decision Tree, "
                "Random Forest and XGBoost were "
                "evaluated."
            )
        ),

        (
            "06",
            "Spatial validation",
            (
                "Leave-One-District-Out validation "
                "tested generalization to completely "
                "unseen districts."
            )
        )
    ]


    for number, title, text in methodology:

        st.html(
            f"""
<div
    class="editorial-card"
    style="
        margin-bottom:12px;
        min-height:auto;
    "
>

    <div class="card-kicker">
        {number}
    </div>

    <div style="
        font-size:25px;
        letter-spacing:-0.03em;
        margin:7px 0 9px 0;
    ">
        {title}
    </div>

    <div class="card-description">
        {text}
    </div>

</div>
            """
        )


    st.markdown(
        "<div style='height:25px'></div>",
        unsafe_allow_html=True
    )


    section_header(

        "07 / LIMITATIONS",

        "What the model cannot claim.",

        (
            "Responsible interpretation is part "
            "of the research design."
        )
    )


    st.markdown(
        """
- The verified ML dataset remains small.
- Climate availability is uneven across districts and years.
- Historical surveillance periods are not perfectly harmonized.
- Satellite extraction uses representative district regions rather than complete administrative boundaries.
- The current forest-loss representation has limited temporal variation.
- Model scores are not calibrated epidemiological probabilities.
- Feature importance does not establish causality.
- Spatial generalization remains modest.
        """
    )


# ============================================================
# 19. FOOTER
# ============================================================

st.markdown(
    "<div style='height:45px'></div>",
    unsafe_allow_html=True
)


st.html(
    """
<div style="
    border-top:1px solid #8f857d;
    padding-top:20px;
    display:flex;
    justify-content:space-between;
    font-size:11px;
    color:#675f59;
">

    <span>
        SpilloverAI / KFD Research Framework
    </span>

    <span>
        Western Ghats · 2015–2024
    </span>

</div>
    """
)