import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import umap

from sklearn.preprocessing import SplineTransformer
from sklearn.pipeline import make_pipeline
from sklearn.kernel_approximation import Nystroem
    

def main():
    ROOT = Path(__file__).parent
    parquet_file = ROOT / "data2" / "panel_default_with_macro.parquet"
        
    print(f"Loading panel dataset from {parquet_file}...")
    df = pd.read_parquet(parquet_file)
    
    print("Preparing data for LDA...")
    
    features = [
        "credit_score",
        "original_upb",
        "original_loan_to_value_ltv",
        "dti",
        "original_interest_rate",
        "unrate",
        "pmms30",
        "cpi_yoy",
        "median_aqi"
    ]
    
    target_col = "y_default"
    
    # Clean up outliers
    if "dti" in df.columns:
        df.loc[df['dti'] > 100, 'dti'] = np.nan
        
    # Drop rows with missing values in our features and target
    analysis_df = df[features + [target_col]].dropna().copy()
    
    # Filter out missing data codes
    analysis_df = analysis_df[analysis_df['credit_score'] < 9000]
    
    print(f"Data shape after cleaning: {analysis_df.shape}")
    
    X = analysis_df[features]
    y = analysis_df[target_col]
    
    print("Standardizing the dataset...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    OUTDIR = ROOT / "processed_mortgage"
    OUTDIR.mkdir(parents=True, exist_ok=True)
    
    '''
    
    print("Fitting Linear Discriminant Analysis (LDA)...")
    # LDA for binary classification produces exactly 1 component
    lda = LinearDiscriminantAnalysis(n_components=1)
    
    # Fit and transform all data
    lda_projection = lda.fit_transform(X_scaled, y)
    
    analysis_df['lda_1d'] = lda_projection[:, 0]
    
    # Print LDA Coefficients to see feature importance
    print("\n--- LDA Coefficients ---")
    coef_df = pd.DataFrame({
        'Feature': features,
        'Coefficient': lda.coef_[0]
    })
    coef_df['Abs_Coefficient'] = coef_df['Coefficient'].abs()
    coef_df = coef_df.sort_values(by='Abs_Coefficient', ascending=False).drop(columns=['Abs_Coefficient'])
    print(coef_df.to_string(index=False))
    print("------------------------\n")
    
    print("Plotting results...")
    OUTDIR = ROOT / "processed_mortgage"
    OUTDIR.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: KDE Density Plot
    plt.figure(figsize=(10, 6))
    
    # Using KDE plot to visualize the overlapping distributions
    sns.kdeplot(
        data=analysis_df, 
        x="lda_1d", 
        hue=target_col, 
        common_norm=False, 
        fill=True, 
        palette={0: 'blue', 1: 'red'},
        alpha=0.5
    )
    
    plt.title('1D LDA Projection Density (Panel Data - Defaults vs Non-Defaults)')
    plt.xlabel('Linear Discriminant 1')
    plt.ylabel('Density')
    
    plt.tight_layout()
    output_plot = OUTDIR / "lda_panel_default.png"
    plt.savefig(output_plot)
    print(f"Plot saved to {output_plot}")
    
    # Plot 2: Histogram Density Plot
    # Because of class imbalance (millions of 0s, thousands of 1s), use stat='density'
    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=analysis_df, 
        x="lda_1d", 
        hue=target_col, 
        stat='density', 
        common_norm=False, 
        palette={0: 'blue', 1: 'red'},
        bins=50,
        alpha=0.4
    )
    plt.title('1D LDA Projection Histogram (Panel Data - Defaults vs Non-Defaults)')
    plt.xlabel('Linear Discriminant 1')
    plt.ylabel('Density')
    
    plt.tight_layout()
    output_hist = OUTDIR / "lda_panel_default_hist.png"
    plt.savefig(output_hist, dpi=300)
    print(f"Histogram saved to {output_hist}")

    print("\nRunning Supervised 1D UMAP on a sample...")
    # Sample for UMAP because 1.7M rows is too much
    n_samples = min(10000, len(analysis_df))
    sample_df = analysis_df.sample(n=n_samples, random_state=42).copy()
    
    X_umap_sample = scaler.transform(sample_df[features])
    y_umap_sample = sample_df[target_col].values
    
    # Supervised UMAP takes y as an argument
    reducer = umap.UMAP(n_components=1, random_state=42)
    embedding = reducer.fit_transform(X_umap_sample, y=y_umap_sample)
    
    sample_df['umap_1d'] = embedding[:, 0]
    
    print("Plotting Supervised UMAP results...")
    
    # Plot 3: Supervised 1D UMAP KDE
    plt.figure(figsize=(10, 6))
    sns.kdeplot(
        data=sample_df, 
        x="umap_1d", 
        hue=target_col, 
        common_norm=False, 
        fill=True, 
        palette={0: 'blue', 1: 'red'},
        alpha=0.5
    )
    plt.title('Supervised 1D UMAP Density (Panel Data - Defaults vs Non-Defaults)')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('Density')
    
    plt.tight_layout()
    output_umap = OUTDIR / "supervised_umap_1d_default.png"
    plt.savefig(output_umap, dpi=300)
    print(f"Plot saved to {output_umap}")
    '''

    print("\nFitting Flexible Discriminant Analysis (FDA) via Splines...")
    # Using splines to simulate FDA
    fda = make_pipeline(
        SplineTransformer(n_knots=4, degree=3),
        LinearDiscriminantAnalysis(n_components=1)
    )
    fda_projection = fda.fit_transform(X_scaled, y)
    analysis_df['fda_1d'] = fda_projection[:, 0]
    
    print("Plotting FDA results...")
    plt.figure(figsize=(10, 6))
    sns.kdeplot(
        data=analysis_df, x="fda_1d", hue=target_col, 
        common_norm=False, fill=True, palette={0: 'blue', 1: 'red'}, alpha=0.5
    )
    plt.title('Flexible Discriminant Analysis (FDA) Density')
    plt.xlabel('FDA Component 1')
    plt.ylabel('Density')
    plt.tight_layout()
    output_fda = OUTDIR / "fda_panel_default.png"
    plt.savefig(output_fda, dpi=300)
    print(f"Plot saved to {output_fda}")

    print("\nFitting Kernel LDA (via Nystroem RBF approximation)...")
    klda = make_pipeline(
        Nystroem(kernel='rbf', gamma=0.1, n_components=100, random_state=42),
        LinearDiscriminantAnalysis(n_components=1)
    )
    klda_projection = klda.fit_transform(X_scaled, y)
    analysis_df['klda_1d'] = klda_projection[:, 0]
    
    print("Plotting Kernel LDA results...")
    plt.figure(figsize=(10, 6))
    sns.kdeplot(
        data=analysis_df, x="klda_1d", hue=target_col, 
        common_norm=False, fill=True, palette={0: 'blue', 1: 'red'}, alpha=0.5
    )
    plt.title('Kernel LDA Density')
    plt.xlabel('KLDA Component 1')
    plt.ylabel('Density')
    plt.tight_layout()
    output_klda = OUTDIR / "klda_panel_default.png"
    plt.savefig(output_klda)
    print(f"Plot saved to {output_klda}")

if __name__ == "__main__":
    main()
