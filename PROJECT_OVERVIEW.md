SpilloverAI

Environmental Intelligence for Kyasanur Forest Disease in the Western Ghats

Objective. To investigate whether routinely obtainable climate, satellite and landscape indicators can provide useful predictive signal for reported KFD outbreak occurrence, while building a reproducible and interpretable geospatial machine-learning framework that can later be extended to other zoonotic diseases.

What we built

SpilloverAI is a Western Ghats research prototype covering nine KFD-relevant districts across Karnataka, Kerala, Goa and Maharashtra, with environmental observations spanning 2015-2024 where data were available. The pipeline combines climate variables, Sentinel-2 NDVI, Hansen forest-change information, SRTM elevation and source-backed KFD outbreak records. These sources were transformed into a master environmental dataset, engineered features, verified disease targets, a model-ready training dataset, publication figures/tables and an interactive Streamlit dashboard.

Research and machine-learning workflow

Environmental features were merged with verified KFD occurrence labels. Logistic Regression, Decision Tree, Random Forest and XGBoost were compared. Primary evaluation used Leave-One-District-Out validation so each test fold represented a geographically unseen district, while a secondary temporal experiment evaluated later observations in regions already represented during training. Feature-set ablation showed that dynamic variables - temperature, rainfall, NDVI and temperature range - generalized better across unseen districts than the static-only landscape features.

Current findings

The final spatial model is a Dynamic Random Forest trained on 46 verified observations, evenly split between 23 outbreak and 23 non-outbreak records. Under Leave-One-District-Out validation it achieved 56.5% balanced accuracy, precision 0.571, recall 0.522, F1 0.545, ROC-AUC 0.544 and PR-AUC 0.501. These results indicate modest cross-district generalization rather than strong operational forecasting performance. A secondary temporal Decision Tree experiment produced stronger results (balanced accuracy 0.833, F1 0.909 and ROC-AUC 0.800), but on a very small later-period test set. The main scientific insight is therefore that environmental variables show more promising temporal signal within represented regions than geographic transfer to entirely unseen districts.

What the project is - and is not

SpilloverAI is a research prototype for environmental intelligence around KFD. It predicts reported outbreak occurrence from environmental observations; it does not diagnose disease, predict individual infection, establish causality or provide a calibrated epidemiological outbreak probability. Its contribution lies in the reproducible integration of remote sensing, climate data, verified disease evidence, spatial validation and interpretable machine learning.

Project outcome and next step

The V1 technical pipeline, validated model, dashboard and research assets are complete. The immediate next objective is to document the work in a research paper, publish the repository cleanly on GitHub and present the dashboard as a transparent research demonstrator. Future versions can improve district boundaries, temporal forest-loss features, climate coverage, target volume, model calibration and extension to additional zoonotic diseases.