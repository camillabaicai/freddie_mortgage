# Freddie Mac Mortgage Risk Modeling

This repository builds loan-month panel datasets from Freddie Mac mortgage data, merges macroeconomic and environmental features, and compares several default and prepayment prediction models.

The current modeling workflow focuses on two binary monthly outcomes:

- `y_prepay`: prepayment event, defined from zero balance code `01`
- `y_default`: default event, defined from delinquency status reaching 3+ months

## Repository Structure

```text
data/
  step_1_mortgage_data_creation_and_analysis_origination.ipynb
  step_2_performance_data_creation_and_analysis.ipynb
  step_3_Merge_other_data.ipynb

model_dev/
  01_Feature_Selection_and_baseline.ipynb
  02_Model_options_Cubic_Spline.ipynb
  03_Model_options_Trees.ipynb
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

## Evaluation Setup

Models are evaluated using a loan-level train/test split. All monthly rows for a loan stay either in train or test, preventing the same loan from appearing in both sets.

The split is stratified by loan-level ever-event status so rare events, especially default, are represented similarly in train and test.

Metrics saved across notebooks include:

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
| Random Forest | 46 | 0.969081 |
| Histogram Gradient Boosting | 46 | 0.972572 |

### Default

| Model | Input features | ROC-AUC |
|---|---:|---:|
| Baseline logistic-style model | 46 | 0.783611 |
| L1 selected model | 17 | 0.798036 |
| Cubic spline model | 70 | 0.832383 |
| Random Forest | 46 | 0.850188 |
| Histogram Gradient Boosting | 46 | 0.885922 |

`Input features` refers to the model design-matrix feature count after one-hot encoding or spline expansion, not simply raw source-variable count.

## Feature Importance

Tree importance is computed with held-out permutation importance using average precision scoring. This is useful for rare-event outcomes because it measures how much PR-AUC drops when an original source variable is shuffled.

Saved plots:

- `model_dev/artifacts/prepay_top15_variable_importance.png`
- `model_dev/artifacts/default_top15_variable_importance.png`

The L1-vs-tree comparison tables compare:

- L1 top 15 selected source variables
- Histogram Gradient Boosting top 15 permutation-importance variables

Each table has three columns:

- `both_selected`
- `L1_only`
- `tree_only`

## Suggested Run Order

Run notebooks in this order:

1. `data/step_1_mortgage_data_creation_and_analysis_origination.ipynb`
2. `data/step_2_performance_data_creation_and_analysis.ipynb`
3. `data/step_3_Merge_other_data.ipynb`
4. `model_dev/01_Feature_Selection_and_baseline.ipynb`
5. `model_dev/02_Model_options_Cubic_Spline.ipynb`
6. `model_dev/03_Model_options_Trees.ipynb`

If you change leakage exclusions, feature engineering, or target definitions, rerun notebooks 01 through 03 so the model artifacts stay consistent.

## Environment

The notebooks use Python with common data-science packages:

```text
pandas
numpy
matplotlib
scikit-learn
scipy
pyarrow
requests
openpyxl
jupyter
```

Install example:

```bash
pip install pandas numpy matplotlib scikit-learn scipy pyarrow requests openpyxl jupyter
```

## Notes

- Raw Freddie Mac data and large processed parquet files are not stored in this repository.
- Several notebooks currently use absolute local paths. Update these paths before running elsewhere.
- The project is exploratory and notebook-driven; the authoritative outputs are the CSV and PNG files under `model_dev/artifacts/`.
