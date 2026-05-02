import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.kernel_approximation import Nystroem

def main():
    ROOT = Path(__file__).parent
    parquet_file = ROOT / "data2" / "panel_prepay_with_macro.parquet"
    
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
    
    target_col = "y_prepay"
    
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
    
    # --- Regular LDA ---
    print("\nFitting Linear Discriminant Analysis (LDA)...")
    lda = LinearDiscriminantAnalysis(n_components=1)
    lda_projection = lda.fit_transform(X_scaled, y)
    analysis_df['lda_1d'] = lda_projection[:, 0]
    
    # Print LDA Coefficients
    print("\n--- LDA Coefficients ---")
    coef_df = pd.DataFrame({
        'Feature': features,
        'Coefficient': lda.coef_[0]
    })
    coef_df['Abs_Coefficient'] = coef_df['Coefficient'].abs()
    coef_df = coef_df.sort_values(by='Abs_Coefficient', ascending=False).drop(columns=['Abs_Coefficient'])
    print(coef_df.to_string(index=False))
    print("------------------------\n")
    
    print("Plotting Regular LDA results...")
    plt.figure(figsize=(10, 6))
    sns.kdeplot(
        data=analysis_df, 
        x="lda_1d", 
        hue=target_col, 
        common_norm=False, 
        fill=True, 
        palette={0: 'blue', 1: 'red'},
        alpha=0.5
    )
    plt.title('1D LDA Projection Density (Panel Data - Prepay vs Non-Prepay)')
    plt.xlabel('Linear Discriminant 1')
    plt.ylabel('Density')
    plt.tight_layout()
    output_plot = OUTDIR / "lda_panel_prepay.png"
    plt.savefig(output_plot)
    print(f"Plot saved to {output_plot}")
    
    # --- Kernel LDA ---
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
        data=analysis_df, 
        x="klda_1d", 
        hue=target_col, 
        common_norm=False, 
        fill=True, 
        palette={0: 'blue', 1: 'red'}, 
        alpha=0.5
    )
    plt.title('Kernel LDA Density (Panel Data - Prepay vs Non-Prepay)')
    plt.xlabel('KLDA Component 1')
    plt.ylabel('Density')
    plt.tight_layout()
    output_klda = OUTDIR / "klda_panel_prepay.png"
    plt.savefig(output_klda, dpi=300)
    print(f"Plot saved to {output_klda}")

if __name__ == "__main__":
    main()
