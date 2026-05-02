# Freddie Mac Mortgage Risk Modeling

This repository builds loan-month panel datasets from Freddie Mac mortgage data, merges macroeconomic and environmental features, and compares linear, spline, tree, neural-network, and exploratory representation-learning approaches for mortgage default and prepayment prediction.

The current modeling workflow focuses on two binary monthly outcomes:

- `y_prepay`: prepayment event, defined from zero balance code `01`
- `y_default`: default event, defined from delinquency status reaching 3+ months

## Repository Structure

```text
data/
  step_1_mortgage_data_creation_and_analysis_origination.ipynb
  step_2_performance_data_creation_and_analysis.ipynb
  process_air_data.ipynb
  step_3_Merge_other_data.ipynb

model_dev/
  01_Feature_Selection_and_baseline.ipynb
  02_Model_options_Cubic_Spline.ipynb
  03_Model_options_Trees.ipynb
  04_Model_options_neural_network.ipynb
  05_Unsupervised/
    lda_panel_analysis.py
    lda_panel_prepay_analysis.py
    umap_panel_analysis.py
    umap_panel_prepay_analysis.py
  artifacts/

src/
  main.py
```

## Data Pipeline

The raw and processed data are expected under:

```text
/Users/jinyecai/Desktop/ML_Mortgage
```

If running on another machine, update the `ROOT` / `PROC` paths inside the notebooks.

### 1. Origination Data

Notebook: `data/step_1_mortgage_data_creation_and_analysis_origination.ipynb`

Main output:

```text
processed_mortgage/orig_only_2016_2025.parquet
```

This notebook reads Freddie Mac origination files, applies file-layout based column names, normalizes column names, performs basic cleaning, and creates origination EDA outputs.

### 2. Performance Labels and Loan-Month Panels

Notebook: `data/step_2_performance_data_creation_and_analysis.ipynb`

Main outputs:

```text
processed_mortgage/perf_performance_labeled_for_default_2016_2025.parquet
processed_mortgage/perf_performance_labeled_for_prepay_2016_2025.parquet
processed_mortgage/panel_default_modeling_2016_2025.parquet
processed_mortgage/panel_prepay_modeling_2016_2025.parquet
```

This notebook labels the first observed event per loan and keeps loan-month rows up to the first event or censoring month.

### Optional: Air Data and AQI EDA

Notebook: `data/process_air_data.ipynb`

This companion notebook reads annual county-level AQI CSV files from `data/air data/`, merges them for exploratory analysis, creates derived shares such as `Good Days % of Total`, and summarizes state-year air-quality patterns using metrics such as median AQI.

Figure outputs are written under:

```text
figures/air data/
```

The macro merge notebook below also reads annual AQI county files and collapses them to state-year median AQI for the modeling panels.

### 3. Macro, HPI, and AQI Merge

Notebook: `data/step_3_Merge_other_data.ipynb`

Main outputs:

```text
processed_mortgage/macro_merged/panel_default_with_macro.parquet
processed_mortgage/macro_merged/panel_prepay_with_macro.parquet
```

Merged external features include FRED macro series such as `unrate`, `cpi`, `fedfunds`, and `gs10`; Freddie PMMS `pmms30`; FHFA state HPI features; and state-year median AQI. The notebook also derives spread features such as `mortgage_treasury_spread`, `spread_at_origination`, and `incentive_rate`.

Use a `FRED_API_KEY` environment variable when possible. Avoid committing API keys.

## Modeling Pipeline

### Notebook 01: Feature Selection and Baseline

Notebook: `model_dev/01_Feature_Selection_and_baseline.ipynb`

This notebook:

- reads the macro-merged panels
- performs feature engineering
- writes cleaned modeling parquet files
- fits baseline logistic-style models
- performs L1 feature screening
- saves L1-selected source variables and metrics

Important leakage control: `current_actual_upb` is excluded because it is a post-origination balance variable and can leak zero-balance/event status into prepayment and default models.

Key outputs:

```text
processed_mortgage/cleaned/default_fe.parquet
processed_mortgage/cleaned/prepay_fe.parquet
model_dev/artifacts/baseline_logistic_metrics_compare.csv
model_dev/artifacts/l1_regularized_metrics_compare.csv
model_dev/artifacts/l1_selected_source_features.csv
```

### Notebook 02: Cubic Spline Model

Notebook: `model_dev/02_Model_options_Cubic_Spline.ipynb`

This notebook fits logistic-style models on the L1-selected source variables, adding cubic spline expansions for continuous variables.

Key outputs:

```text
model_dev/artifacts/l1_vs_spline_model_comparison.csv
```

### Notebook 03: Tree Models

Notebook: `model_dev/03_Model_options_Trees.ipynb`

This notebook fits tree-based models on the original baseline source-variable set, not the L1-selected list:

- Random Forest
- Histogram Gradient Boosting

It also saves ROC-AUC comparison tables, tree variable-importance charts, and L1-vs-tree feature-screening comparison tables.

Key outputs:

```text
model_dev/artifacts/tree_model_metrics_compare.csv
model_dev/artifacts/tree_permutation_source_feature_importance.csv
model_dev/artifacts/tree_impurity_source_feature_importance.csv
model_dev/artifacts/prepay_roc_auc_model_comparison.csv
model_dev/artifacts/default_roc_auc_model_comparison.csv
model_dev/artifacts/prepay_top15_variable_importance.png
model_dev/artifacts/default_top15_variable_importance.png
model_dev/artifacts/prepay_tree_l1_feature_screening_comparison.csv
model_dev/artifacts/default_tree_l1_feature_screening_comparison.csv
```

### Notebook 04: Neural Network Model

Notebook: `model_dev/04_Model_options_neural_network.ipynb`

This notebook reads the engineered feature panels for default and prepay, applies the same loan-level train/test split idea used elsewhere, and fits an `MLPClassifier` with hidden layers `(40, 20, 10)`.

The workflow:

- loads engineered feature panels from `data/default_fe.parquet` and `data/prepay_fe.parquet` in the current notebook version; adjust these paths if your cleaned panels remain under `processed_mortgage/cleaned/`
- drops leakage-prone identifiers and servicing columns such as `current_actual_upb`
- imputes numeric and categorical features
- one-hot encodes categorical columns
- rescales the design matrix with `MaxAbsScaler`
- evaluates held-out ROC-AUC, PR-AUC, log loss, and Brier score
- saves SHAP explainers and loss-curve plots

Key outputs:

```text
figures/default_loss_curve.png
figures/prepay_loss_curve.png
data/default_explainer.pkl
data/prepay_explainer.pkl
```

Create `figures/` first if it does not already exist in your local setup.

## Exploratory Representation Learning

The `model_dev/05_Unsupervised/` scripts are optional diagnostics rather than part of the core supervised benchmark table.

They currently expect local copies or symlinks of the macro-merged panel files under:

```text
model_dev/05_Unsupervised/data2/panel_default_with_macro.parquet
model_dev/05_Unsupervised/data2/panel_prepay_with_macro.parquet
```

Their plots are written under:

```text
model_dev/05_Unsupervised/processed_mortgage/
```

### UMAP + KMeans Scripts

- `umap_panel_analysis.py`: standardizes nine continuous panel features, assigns KMeans clusters, profiles cluster composition, and exports 2-D UMAP plots colored by cluster, monthly default status, and origination channel.
- `umap_panel_prepay_analysis.py`: applies the same workflow for `y_prepay`, including silhouette-based cluster selection on a sample and 2-D UMAP diagnostics for prepayment behavior.

### Discriminant Analysis Scripts

- `lda_panel_analysis.py`: explores one-dimensional supervised separation for default using flexible discriminant analysis and kernel-LDA style projections.
- `lda_panel_prepay_analysis.py`: fits one-dimensional LDA and kernel-LDA style projections for prepay and prints ranked coefficient loadings.

Example commands:

```bash
python model_dev/05_Unsupervised/umap_panel_analysis.py
python model_dev/05_Unsupervised/umap_panel_prepay_analysis.py
python model_dev/05_Unsupervised/lda_panel_analysis.py
python model_dev/05_Unsupervised/lda_panel_prepay_analysis.py
```

## Evaluation Setup

Models are evaluated using a loan-level train/test split. All monthly rows for a loan stay either in train or test, preventing the same loan from appearing in both sets.

The split is stratified by loan-level ever-event status so rare events, especially default, are represented similarly in train and test.

Metrics saved across the supervised modeling notebooks include:

- ROC-AUC
- PR-AUC / average precision
- log loss
- Brier score

The compact comparison tables below focus on ROC-AUC.

## Current ROC-AUC Summary

### Prepayment

| Model | Input features | ROC-AUC |
|---|---:|---:|
| Baseline logistic-style model | 46 | 0.961537 |
| L1 selected model | 33 | 0.961675 |
| Cubic spline model | 113 | 0.963840 |
| Neural network (MLP) | 46 | 0.966200 |
| Random Forest | 46 | 0.969081 |
| Histogram Gradient Boosting | 46 | 0.972572 |

### Default

| Model | Input features | ROC-AUC |
|---|---:|---:|
| Baseline logistic-style model | 46 | 0.783611 |
| L1 selected model | 17 | 0.798036 |
| Cubic spline model | 70 | 0.832383 |
| Neural network (MLP) | 46 | 0.802000 |
| Random Forest | 46 | 0.850188 |
| Histogram Gradient Boosting | 46 | 0.885922 |

`Input features` refers to the model design-matrix feature count after one-hot encoding or spline expansion, not simply raw source-variable count.

## Model Interpretation

Tree importance is computed with held-out permutation importance using average precision scoring. This is useful for rare-event outcomes because it measures how much PR-AUC drops when an original source variable is shuffled.

Saved tree-model plots:

- `model_dev/artifacts/prepay_top15_variable_importance.png`
- `model_dev/artifacts/default_top15_variable_importance.png`

The L1-vs-tree comparison tables compare:

- L1 top 15 selected source variables
- Histogram Gradient Boosting top 15 permutation-importance variables

Each table has three columns:

- `both_selected`
- `L1_only`
- `tree_only`

Notebook 04 also displays a top-15 SHAP bar summary during execution and saves reusable SHAP explainer objects under `data/`.

## Suggested Run Order

Run the core data and supervised modeling notebooks in this order:

1. `data/step_1_mortgage_data_creation_and_analysis_origination.ipynb`
2. `data/step_2_performance_data_creation_and_analysis.ipynb`
3. Optional AQI EDA: `data/process_air_data.ipynb`
4. `data/step_3_Merge_other_data.ipynb`
5. `model_dev/01_Feature_Selection_and_baseline.ipynb`
6. `model_dev/02_Model_options_Cubic_Spline.ipynb`
7. `model_dev/03_Model_options_Trees.ipynb`
8. `model_dev/04_Model_options_neural_network.ipynb`

The optional `05_Unsupervised` scripts can be run after Step 3 once the macro-merged panel files are available under `model_dev/05_Unsupervised/data2/`.

If you change leakage exclusions, feature engineering, or target definitions, rerun the downstream notebooks and scripts that depend on those outputs so the model artifacts stay consistent.

## Environment

The notebooks and scripts use Python with common data-science packages:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
pyarrow
requests
openpyxl
jupyter
shap
umap-learn
```

Install example:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy pyarrow requests openpyxl jupyter shap umap-learn
```

## Notes

- Raw Freddie Mac data and large processed parquet files are not stored in this repository.
- Several notebook and script paths are local-machine specific or relative to the current working directory. Adjust `ROOT`, `PROC`, and local parquet paths when reproducing the workflow on another machine.
