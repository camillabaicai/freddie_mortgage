import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

import umap
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def main():
    ROOT = Path(__file__).parent
    parquet_file = ROOT / "data2" / "panel_prepay_with_macro.parquet"
    
    print(f"Loading panel dataset from {parquet_file}...")
    df = pd.read_parquet(parquet_file)
    
    print("Preparing data for unsupervised learning...")
    
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
    
    # Categorical/Metadata columns to keep for profiling
    profiling_cols = ["postal_code", "channel", "first_time_homebuyer_flag"]
    target_col = "y_prepay"
    keep_cols = list(set([target_col] + [c for c in profiling_cols if c in df.columns]))
    
    # Clean up outliers
    if "dti" in df.columns:
        df.loc[df['dti'] > 100, 'dti'] = np.nan
        
    # Drop rows with missing values in our features
    analysis_df = df[features + keep_cols].dropna(subset=features).copy()
    
    # Filter out missing data codes
    analysis_df = analysis_df[analysis_df['credit_score'] < 9000]
    
    print(f"Data shape after cleaning: {analysis_df.shape}")
    
    print("Standardizing the full dataset...")
    scaler = StandardScaler()
    scaled_full_features = scaler.fit_transform(analysis_df[features])
    
    # Small sample for Silhouette evaluation
    n_silhouette_samples = min(10000, len(analysis_df))
    np.random.seed(42)
    silhouette_indices = np.random.choice(len(analysis_df), size=n_silhouette_samples, replace=False)
    scaled_silhouette_features = scaled_full_features[silhouette_indices]
    
    print("Finding the best number of clusters using Silhouette Score (on sample)...")
    best_k = 2
    best_score = -1
    best_kmeans = None
    
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(scaled_full_features)
        
        sample_clusters = kmeans.predict(scaled_silhouette_features)
        score = silhouette_score(scaled_silhouette_features, sample_clusters)
        
        print(f"  k={k}, Silhouette Score (10k sample): {score:.4f}")
        if score > best_score:
            best_score = score
            best_k = k
            best_kmeans = kmeans
            
    print(f"Optimal number of clusters: {best_k} (Score: {best_score:.4f})")
    
    analysis_df['cluster'] = best_kmeans.predict(scaled_full_features)
    
    profile_df = analysis_df.groupby('cluster')[features].mean()
    profile_df['count'] = analysis_df.groupby('cluster').size()
    print(profile_df)
    print("---------------------------------------\n")
    
    for cluster_id in sorted(analysis_df['cluster'].unique()):
        print(f"Cluster {cluster_id}:")
        cluster_data = analysis_df[analysis_df['cluster'] == cluster_id]
        
        if 'postal_code' in cluster_data.columns:
            top_zips = cluster_data['postal_code'].value_counts().head(3)
            print(f"  Top 3 Postal Codes: {', '.join(f'{k} ({v})' for k, v in top_zips.items())}")
            
        if 'channel' in cluster_data.columns:
            top_chan = cluster_data['channel'].value_counts().head(1)
            if not top_chan.empty:
                print(f"  Top Channel: {top_chan.index[0]} ({top_chan.iloc[0]})")
                
        if 'first_time_homebuyer_flag' in cluster_data.columns:
            fthb_counts = cluster_data['first_time_homebuyer_flag'].value_counts(normalize=True)
            if 'Y' in fthb_counts:
                fthb_y_pct = fthb_counts.get('Y', 0) * 100
            elif 1 in fthb_counts:
                fthb_y_pct = fthb_counts.get(1, 0) * 100
            else:
                fthb_y_pct = 0
            print(f"  First-Time Homebuyer: {fthb_y_pct:.1f}%")
            
        if target_col in cluster_data.columns:
            target_rate = cluster_data[target_col].mean() * 100
            print(f"  Prepay Rate (Monthly): {target_rate:.4f}%")
            
    
    # UMAP dimensionality reduction on sample
    n_samples = min(10000, len(analysis_df))
    sample_df = analysis_df.sample(n=n_samples, random_state=42).copy()
    print(f"Sampled {n_samples} rows for UMAP visualization.")
    
    print("Running UMAP dimensionality reduction...")
    scaled_umap_features = scaler.transform(sample_df[features])
    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding = reducer.fit_transform(scaled_umap_features)
    
    sample_df['umap_1'] = embedding[:, 0]
    sample_df['umap_2'] = embedding[:, 1]
    
    global_cluster_counts = analysis_df['cluster'].value_counts()
    smallest_cluster = global_cluster_counts.idxmin()
    if global_cluster_counts[smallest_cluster] < (len(analysis_df) * 0.05):
        print(f"Dropping cluster {smallest_cluster} from plot as it is a small outlier group.")
        sample_df = sample_df[sample_df['cluster'] != smallest_cluster]
        
    print("Plotting results...")
    OUTDIR = ROOT / "processed_mortgage"
    OUTDIR.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: By Cluster
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x='umap_1', y='umap_2', hue='cluster', palette='viridis', 
        data=sample_df, s=15, alpha=0.8, edgecolor='none'
    )
    plt.title('UMAP Projection (Panel Data - KMeans Clusters)')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    output_plot1 = OUTDIR / "umap_panel_prepay_clusters.png"
    plt.savefig(output_plot)
    print(f"Plot saved to {output_plot1}")

    # Plot 2: By Prepay Target
    # Sort the dataframe so that prepays (1) are plotted last, appearing on top
    sample_df = sample_df.sort_values(by=target_col)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x='umap_1', y='umap_2', hue=target_col, palette={0: 'blue', 1: 'red'},
        data=sample_df, s=15, alpha=0.8, edgecolor='none', hue_order=[0, 1]
    )
    plt.title('UMAP Projection (Panel Data - Monthly Prepay Status)')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.legend(title='Prepay Status', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    output_plot2 = OUTDIR / "umap_panel_prepay_status.png"
    plt.savefig(output_plot2)
    print(f"Plot saved to {output_plot2}")
    
if __name__ == "__main__":
    main()
