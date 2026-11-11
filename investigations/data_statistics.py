
# Libraries & Modules
import os
import sys
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":

    result_folder = "D:/exp_RESULTS\CDT_exp/061926/loss/baseline_autoencoding_loss/volumenormalisation_lambda_s00_005_01_02_03_04_dec1L_T1POST_T2_FLAIR_slices_all/final_comparison_results.csv"
    

    # 1. Loading and data preparation
    df = pd.read_csv(result_folder)
    df = df.sort_values(by=['File', 'Experiment', 'Slice'])
    metrics = [ 'MNRMSE', 'MSSIM']
    df = df[(df['Slice'] <= 10)].copy()
    # Generate global summary table
    summary_df = df.groupby('Experiment')[metrics].agg(['mean', 'std']).round(4)
    print("--- Global Summary Table (Mean and Std) ---")
    print(summary_df)

    # 2. Configure visualization style
    sns.set_theme(style="whitegrid", context="talk") 
    subjects = df['File'].unique()

    # Using a distinct color palette
    palette_choice = "deep" 

    # 3. Visualization loop
    for subject in subjects:
        df_subject = df[df['File'] == subject]
        
        # Create figure with contrast and clear layout
        fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 16), sharex=True)
        fig.suptitle(f"Performance Analysis - Subject: {subject}", fontsize=18, fontweight='bold', y=1.02)

        for i, metric in enumerate(metrics):
            # Use style and hue for maximum visual distinction
            sns.lineplot(
                data=df_subject,
                x='Slice',
                y=metric,
                hue='Experiment',
                style='Experiment',  # Differentiates line styles
                markers=True,        # Differentiates marker shapes
                dashes=True,         # Differentiates line patterns
                ax=axes[i],
                markersize=7,
                linewidth=2.5,
                palette=palette_choice 
            )
            
            # Apply log scale only to error metrics
            if metric in ['NRMSE', 'MNRMSE']:
                axes[i].set_yscale("log")
                
            axes[i].set_title(f'Metric: {metric}', fontsize=14, fontweight='bold')
            axes[i].set_ylabel(metric)
            axes[i].grid(True, which="both", ls="-", alpha=0.6)
            
            # Optimized legend positioning
            if i == 0:
                axes[i].legend(title='Methods', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
            else:
                axes[i].get_legend().remove()

        plt.xlabel('Slice Index', fontsize=14)
        plt.tight_layout()
        plt.show()