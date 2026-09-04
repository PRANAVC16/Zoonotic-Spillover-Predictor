# SpilloverAI

## Environmental Intelligence for Kyasanur Forest Disease in the Western Ghats

SpilloverAI is a research-oriented geospatial machine-learning project that investigates whether routinely obtainable environmental variables can provide useful predictive signal for **reported Kyasanur Forest Disease (KFD) outbreak occurrence** across the Western Ghats.

The project combines climate observations, satellite-derived vegetation indicators, forest-change information, terrain data, verified historical KFD records, machine-learning models, spatial validation, and an interactive Streamlit dashboard in a single reproducible workflow.

> **Research status:** Technical pipeline complete.
> **Intended use:** Research and environmental intelligence prototype — not an operational public-health forecasting or diagnostic system.

---

## Table of Contents

* [Project Objective](#project-objective)
* [Research Question](#research-question)
* [Study Area](#study-area)
* [What We Built](#what-we-built)
* [Data Sources and Environmental Variables](#data-sources-and-environmental-variables)
* [Research Workflow](#research-workflow)
* [Machine-Learning Experiments](#machine-learning-experiments)
* [Final Model Results](#final-model-results)
* [Key Research Findings](#key-research-findings)
* [Interactive Dashboard](#interactive-dashboard)
* [Repository Structure](#repository-structure)
* [Getting Started](#getting-started)
* [Running the Dashboard](#running-the-dashboard)
* [Research Outputs](#research-outputs)
* [Version History](#version-history)
* [Limitations](#limitations)
* [Future Work](#future-work)
* [Responsible Interpretation](#responsible-interpretation)

---

## Project Objective

The objective of SpilloverAI is to evaluate whether **climate, vegetation, landscape, and topographic indicators** contain useful information for identifying environmental conditions associated with reported KFD outbreak occurrence.

Rather than treating disease prediction as a purely clinical classification problem, SpilloverAI explores the broader environmental context in which zoonotic disease risk emerges.

The long-term vision is a reusable environmental-intelligence framework that can later be extended to other zoonotic diseases when suitable epidemiological data are available.

---

## Research Question

> **Can satellite-derived environmental indicators and climate variables predict the spatial and temporal occurrence of Kyasanur Forest Disease outbreaks in the Western Ghats?**

The project therefore evaluates two distinct forms of generalization:

1. **Spatial generalization** — can a model classify KFD occurrence in a district that was not seen during training?
2. **Temporal generalization** — can earlier observations help classify later outbreak occurrence within already represented regions?

---

## Study Area

The study covers nine KFD-relevant districts across the Western Ghats:

| State       | District       |
| ----------- | -------------- |
| Karnataka   | Shivamogga     |
| Karnataka   | Uttara Kannada |
| Karnataka   | Chikkamagaluru |
| Karnataka   | Kodagu         |
| Kerala      | Wayanad        |
| Kerala      | Kannur         |
| Goa         | North Goa      |
| Goa         | South Goa      |
| Maharashtra | Sindhudurg     |

The environmental study period spans **2015–2024 where source coverage is available**.

Because historical climate coverage is incomplete for some districts, the final environmental dataset contains fewer than the theoretical 90 district-year observations.

---

## What We Built

SpilloverAI currently includes:

* Automated multi-district climate extraction and aggregation
* Sentinel-2 NDVI extraction using Google Earth Engine
* Hansen Global Forest Change processing
* SRTM elevation extraction
* A merged environmental master dataset
* Environmental feature engineering
* Source-backed KFD outbreak target construction
* A verified machine-learning training dataset
* Logistic Regression, Decision Tree, Random Forest, and XGBoost experiments
* Leave-One-District-Out spatial validation
* Secondary temporal validation
* Dynamic-vs-static feature-set ablation
* Publication-ready figures and result tables
* A trained research model
* An editorial-style Streamlit research dashboard

---

## Data Sources and Environmental Variables

### Climate

Climate observations were assembled using Meteostat station data and spatial interpolation.

Core climate variables include:

* Mean temperature
* Minimum temperature
* Maximum temperature
* Annual rainfall
* Temperature range

Wind speed and atmospheric pressure were retained in the master dataset where available, but were excluded from the primary final model.

### Vegetation

Vegetation condition was derived from **Sentinel-2 imagery** using Google Earth Engine.

The primary satellite feature is:

* **NDVI — Normalized Difference Vegetation Index**

NDVI was calculated using the Sentinel-2 near-infrared and red bands and aggregated over representative district regions.

### Forest Change

Forest disturbance information was derived from the **Hansen Global Forest Change** dataset.

The current representation is primarily used as a district-level environmental context variable rather than a strong year-specific predictor.

### Elevation

Mean elevation was derived from:

* **SRTM / USGS SRTMGL1_003**

### KFD Targets

Historical KFD occurrence records were compiled from source-backed government, surveillance, and research documents.

The primary machine-learning target is:

```text
Outbreak_Label = 1  → reported KFD occurrence
Outbreak_Label = 0  → verified non-outbreak observation
```

Case counts are retained where available, but the primary V1 ML experiment is **binary outbreak-occurrence classification**, not case-count regression.

---

## Research Workflow

```text
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
Model-Ready Training Dataset
      ↓
Logistic Regression
Decision Tree
Random Forest
XGBoost
      ↓
Spatial Validation
Leave-One-District-Out
      ↓
Temporal Validation
      ↓
Feature-Set Ablation
      ↓
Final Research Model
      ↓
Streamlit Dashboard
      +
Publication Figures & Tables
```

---

## Machine-Learning Experiments

Four classifiers were evaluated:

* Logistic Regression
* Decision Tree
* Random Forest
* XGBoost

A Dummy Classifier was also included as a baseline.

### Primary Validation Strategy

The main experiment uses **Leave-One-District-Out cross-validation**.

Each fold:

1. holds out one district,
2. trains the model on the remaining districts,
3. evaluates predictions on the unseen district.

This avoids relying solely on a conventional random train/test split, which could allow observations from the same district to appear in both training and testing data.

### Feature-Set Ablation

Three environmental feature groups were compared:

**Dynamic features**

* Temperature
* Rainfall
* NDVI
* Temperature Range

**Static features**

* Forest Loss
* Elevation

**Combined features**

* Dynamic + Static variables

The ablation experiment showed that the dynamic feature set produced better cross-district generalization than static-only features.

---

## Final Model Results

The final deployment-oriented model is a **Dynamic Random Forest** using:

* Temperature
* Rainfall
* NDVI
* Temperature Range

The verified training dataset contains:

* **46 observations**
* **9 districts**
* **23 outbreak observations**
* **23 non-outbreak observations**

### Leave-One-District-Out Performance

| Metric            |     Score |
| ----------------- | --------: |
| Accuracy          | **0.565** |
| Balanced Accuracy | **0.565** |
| Precision         | **0.571** |
| Recall            | **0.522** |
| F1-score          | **0.545** |
| ROC-AUC           | **0.544** |
| PR-AUC            | **0.501** |

These values indicate **modest cross-district predictive performance**, not operational forecasting capability.

### Secondary Temporal Experiment

A Decision Tree produced stronger results on a small later-period temporal holdout:

| Metric            | Score |
| ----------------- | ----: |
| Accuracy          | 0.875 |
| Balanced Accuracy | 0.833 |
| Precision         | 0.833 |
| Recall            | 1.000 |
| F1-score          | 0.909 |
| ROC-AUC           | 0.800 |
| PR-AUC            | 0.807 |

Because this evaluation contains a small number of later-year observations and previously represented geographic regions, it is treated as a **secondary result rather than the headline model performance**.

---

## Key Research Findings

The experiments support four main conclusions:

1. **Cross-district prediction is difficult.**
   Environmental conditions alone produced only modest generalization to completely unseen districts.

2. **Temporal prediction appears more promising.**
   Later outbreak occurrence within already represented regions showed substantially stronger performance in the secondary temporal experiment.

3. **Dynamic environmental variables generalized better than static variables.**
   Temperature, rainfall, NDVI, and temperature range performed better under spatial validation than static-only elevation and forest-loss features.

4. **Environmental ML should be interpreted as decision-support research, not epidemiological certainty.**
   The model identifies statistical environmental patterns but does not establish causality or provide calibrated outbreak probabilities.

---

## Interactive Dashboard

The Streamlit dashboard provides an interactive research interface for exploring:

* District and year selection
* Temperature and rainfall trends
* NDVI trajectories
* Forest-loss and elevation context
* Study-area geography
* Environmental Risk Index
* Experimental KFD ML outbreak signal
* Model inputs
* Random Forest feature importance
* Validation results
* Research methodology
* Scientific limitations

The interface deliberately keeps two concepts separate:

```text
Environmental Risk Index
        ≠
AI KFD Outbreak Signal
```

The first is a descriptive environmental indicator.

The second is produced by the trained machine-learning model.

---

## Repository Structure

```text
Zoonotic-Spillover-Predictor/
│
├── app/
│   ├── assets/
│   └── dashboard/
│       └── app.py
│
├── data/
│   ├── raw/
│   │   └── outbreaks/
│   ├── interim/
│   └── processed/
│
├── models/
│
├── notebooks/
│   ├── 01_climate_data_exploration.ipynb
│   ├── 02_satellite_ndvi_analysis.ipynb
│   ├── ...
│   ├── 14_feature_engineering.ipynb
│   ├── 15_machine_learning_pipeline.ipynb
│   ├── 16_dashboard_integration.ipynb
│   ├── 17_research_paper_assets.ipynb
│   └── 18_verified_kfd_targets.ipynb
│
├── outputs/
│   └── models/
│       ├── kfd_outbreak_model_v1.pkl
│       ├── kfd_outbreak_model_v2.pkl
│       └── kfd_outbreak_model_v2_metadata.json
│
├── paper/
│   ├── figures/
│   ├── tables/
│   ├── references/
│   ├── research_log.md
│   ├── data_dictionary.md
│   └── environmental_variables.md
│
├── src/
│
├── PROJECT_OVERVIEW.md
├── README.md
└── LICENSE
```

> The exact repository structure may evolve as the research paper and deployment workflow are finalized.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/PRANAVC16/Zoonotic-Spillover-Predictor.git
cd Zoonotic-Spillover-Predictor
```

### 2. Create a virtual environment

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

If the repository contains `requirements.txt`:

```bash
pip install -r requirements.txt
```

Core packages used across the project include:

```text
pandas
numpy
scikit-learn
xgboost
joblib
matplotlib
plotly
streamlit
earthengine-api
geemap
meteostat
```

Google Earth Engine analysis additionally requires Earth Engine authentication and an authorized Google Cloud / Earth Engine project.

---

## Running the Dashboard

From the repository root:

```bash
streamlit run app/dashboard/app.py
```

The application will open locally, typically at:

```text
http://localhost:8501
```

The dashboard automatically searches for the project root and loads the processed environmental data and the final saved KFD model from the repository structure.

---

## Research Outputs

Publication-ready assets are stored under:

```text
paper/figures/
paper/tables/
```

Generated outputs include:

* Climate trend figures
* NDVI trends
* District environmental profiles
* Correlation matrix
* Spatial model comparison
* Spatial-vs-temporal validation
* Feature-set ablation
* Confusion matrix
* ROC curve
* Precision–Recall curve
* Final Random Forest feature importance
* Dataset coverage tables
* Model comparison tables
* Temporal-validation results
* Final model metrics

These assets form the quantitative basis of the accompanying research manuscript.

---
## Version History
V3 — Current canonical research version

* Corrected Sentinel-2 district-year NDVI extraction
* Added Cloud Score+ masking
* Rebuilt environmental and training datasets
* Repeated model comparison, temporal validation, and feature ablation
* Selected Dynamic Decision Tree as the final spatial model
* Regenerated dashboard inputs and publication assets
* Updated manuscript results and interpretation

V2 — Superseded research model

* Dynamic Random Forest used as the primary spatial model
* Superseded after the NDVI provenance audit
* Retained only for research history and reproducibility

V1 — Initial research pipeline

* Multi-source environmental data integration
* Prototype environmental risk index
* Early modelling and dashboard workflow

---

## Limitations

The current study has several important limitations:

* The verified ML dataset is small.
* Historical climate coverage is uneven across districts.
* Some surveillance periods are not perfectly harmonized across source documents.
* Sentinel-2 coverage is less complete in the earliest part of the study period.
* Representative district regions are used rather than complete official administrative polygons.
* Climate interpolation can introduce spatial uncertainty in complex Western Ghats terrain.
* The current forest-loss feature has limited temporal variation.
* Historical KFD surveillance may contain under-reporting or differences in reporting practices.
* Model scores are not calibrated epidemiological probabilities.
* Feature importance does not imply causal relationships.
* Spatial generalization remains modest.

These limitations are explicitly retained as part of the research interpretation.

---

## Future Work

Potential extensions include:

* Official district-boundary polygon analysis
* Improved gridded climate datasets such as ERA5-Land or equivalent products
* Truly annual forest-loss features using `lossyear`
* Larger and more consistently harmonized KFD target datasets
* Seasonal rather than annual environmental predictors
* Lagged climate and vegetation features
* Model calibration
* Explainability using SHAP
* Spatiotemporal ML models
* Live Google Earth Engine inference
* Interactive risk maps and time sliders
* Automated report generation
* Expansion to additional zoonotic diseases such as Nipah where suitable ground-truth data are available

---

## Responsible Interpretation

SpilloverAI is **not**:

* a clinical diagnostic system,
* an individual infection-risk calculator,
* a confirmed outbreak-warning system,
* a calibrated epidemiological forecasting tool,
* or evidence that an environmental feature causes KFD.

The platform should be interpreted as a **research prototype for environmental intelligence and machine-learning experimentation around zoonotic disease occurrence**.

---

## Author

**Pranav Choudhary**
---

## License

See the repository's [`LICENSE`](LICENSE) file for licensing information.

Data products used in the project remain subject to the terms and licensing conditions of their original providers.

---

## Citation

A formal citation will be added after completion of the accompanying research manuscript.

For now, if you use or reference this repository, please cite the project as:

```text
Choudhary, P. SpilloverAI: Environmental Intelligence for
Kyasanur Forest Disease in the Western Ghats.
Research prototype, 2026.
```

---

## Acknowledgement

This project integrates open environmental, remote-sensing, and epidemiological information to explore how data science can contribute to more transparent and reproducible zoonotic-disease research.

If you are interested in the methodology, dataset construction, machine-learning experiments, or future collaboration, see [`PROJECT_OVERVIEW.md`](PROJECT_OVERVIEW.md) for a concise introduction to the project.
