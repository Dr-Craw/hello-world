#!/usr/bin/env python3
"""
TDM Results Comparison Tool

Quickly compare results from different model versions.
Shows side-by-side MAE and generates comparison plots.

Run:
    python tdm_compare_results.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).resolve().parent


def load_results(pattern):
    """Load all CSV files matching pattern."""
    files = list(BASE.glob(pattern))
    results = {}
    for f in files:
        # Extract version from filename
        name = f.stem
        if "_v2" in name:
            version = "v2 (enhanced)"
        elif "_v1" in name:
            version = "v1"
        else:
            version = "v1 (baseline)"
        
        # Extract model type
        if "lstm" in name.lower():
            model = "LSTM"
        elif "ann" in name.lower() or "tdm_blind" in name.lower():
            model = "ANN"
        else:
            model = "Unknown"
        
        key = f"{model}_{version}"
        results[key] = pd.read_csv(f)
    
    return results


def compare_mae_tables():
    """Compare MAE by well across different models."""
    print("="*70)
    print("MAE BY WELL COMPARISON")
    print("="*70)
    
    # Load all MAE files
    mae_files = list(BASE.glob("*_mae_by_well*.csv"))
    
    if not mae_files:
        print("No MAE files found. Run training scripts first.")
        return None
    
    # Read and merge
    dfs = []
    for f in mae_files:
        df = pd.read_csv(f)
        
        # Standardize column names
        if "well_id" not in df.columns and df.shape[1] >= 1:
            df = df.rename(columns={df.columns[0]: "well_id"})
        
        mae_col = [c for c in df.columns if "mae" in c.lower() or c != "well_id"][0]
        
        # Extract label from filename
        if "lstm" in f.name:
            model = "LSTM"
        else:
            model = "ANN"
        
        if "_v2" in f.name:
            version = "v2"
        else:
            version = "v1"
        
        label = f"{model}_{version}"
        df = df[["well_id", mae_col]].rename(columns={mae_col: label})
        dfs.append(df)
    
    # Merge all
    result = dfs[0]
    for df in dfs[1:]:
        result = result.merge(df, on="well_id", how="outer")
    
    # Sort by average MAE
    result["avg_mae"] = result.select_dtypes(include=[np.number]).mean(axis=1)
    result = result.sort_values("avg_mae", ascending=False)
    
    print("\nPer-Well MAE (bpd):")
    print(result.to_string(index=False))
    
    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    summary = result.select_dtypes(include=[np.number]).describe().T
    summary["improvement_vs_v1"] = np.nan
    
    for col in summary.index:
        if "v2" in col and "avg" not in col:
            v1_col = col.replace("v2", "v1")
            if v1_col in summary.index:
                v2_mean = summary.loc[col, "mean"]
                v1_mean = summary.loc[v1_col, "mean"]
                pct_change = ((v2_mean - v1_mean) / v1_mean) * 100
                summary.loc[col, "improvement_vs_v1"] = pct_change
    
    print(summary[["mean", "std", "min", "max", "improvement_vs_v1"]].to_string())
    
    return result


def plot_mae_comparison(mae_df):
    """Create visual comparison of MAE across models."""
    if mae_df is None:
        return
    
    plot_dir = BASE / "plots"
    plot_dir.mkdir(exist_ok=True)
    
    # Prepare data (drop avg_mae column)
    plot_data = mae_df.drop(columns=["avg_mae"]).set_index("well_id")
    
    # 1. Bar chart by well
    fig, ax = plt.subplots(figsize=(14, 6))
    plot_data.plot(kind="bar", ax=ax, width=0.8)
    ax.set_title("MAE by Well - Model Comparison", fontsize=14, fontweight="bold")
    ax.set_xlabel("Well ID")
    ax.set_ylabel("MAE (bpd)")
    ax.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(plot_dir / "mae_comparison_by_well.png", dpi=150)
    print(f"\n✓ Saved: {plot_dir / 'mae_comparison_by_well.png'}")
    plt.close(fig)
    
    # 2. Box plot comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_data.boxplot(ax=ax)
    ax.set_title("MAE Distribution - Model Comparison", fontsize=14, fontweight="bold")
    ax.set_xlabel("Model")
    ax.set_ylabel("MAE (bpd)")
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(plot_dir / "mae_comparison_boxplot.png", dpi=150)
    print(f"✓ Saved: {plot_dir / 'mae_comparison_boxplot.png'}")
    plt.close(fig)
    
    # 3. Improvement heatmap (v2 vs v1)
    v1_cols = [c for c in plot_data.columns if "v1" in c]
    v2_cols = [c for c in plot_data.columns if "v2" in c]
    
    if v1_cols and v2_cols:
        fig, ax = plt.subplots(figsize=(12, len(plot_data) * 0.3 + 2))
        
        improvements = pd.DataFrame(index=plot_data.index)
        for v2_col in v2_cols:
            v1_col = v2_col.replace("v2", "v1")
            if v1_col in plot_data.columns:
                pct_change = ((plot_data[v2_col] - plot_data[v1_col]) / plot_data[v1_col]) * 100
                improvements[v2_col] = pct_change
        
        im = ax.imshow(improvements.values, cmap='RdYlGn_r', aspect='auto', vmin=-50, vmax=50)
        
        ax.set_xticks(range(len(improvements.columns)))
        ax.set_xticklabels(improvements.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(improvements.index)))
        ax.set_yticklabels(improvements.index)
        ax.set_title("Improvement: v2 vs v1 (% change in MAE)", fontsize=14, fontweight="bold")
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("% Change (negative = better)", rotation=270, labelpad=20)
        
        # Add text annotations
        for i in range(len(improvements.index)):
            for j in range(len(improvements.columns)):
                val = improvements.values[i, j]
                if not np.isnan(val):
                    color = 'white' if abs(val) > 25 else 'black'
                    ax.text(j, i, f'{val:.1f}%', ha='center', va='center', color=color, fontsize=8)
        
        fig.tight_layout()
        fig.savefig(plot_dir / "mae_improvement_heatmap.png", dpi=150)
        print(f"✓ Saved: {plot_dir / 'mae_improvement_heatmap.png'}")
        plt.close(fig)


def compare_predictions():
    """Compare prediction accuracy across models."""
    print("\n" + "="*70)
    print("PREDICTION COMPARISON")
    print("="*70)
    
    pred_files = list(BASE.glob("*_predictions_blind*.csv"))
    
    if not pred_files:
        print("No prediction files found.")
        return None
    
    results = []
    for f in pred_files:
        df = pd.read_csv(f)
        
        if "oil_rate_bpd" not in df.columns or "pred_oil_rate_bpd" not in df.columns:
            continue
        
        # Extract metadata
        if "lstm" in f.name:
            model = "LSTM"
        else:
            model = "ANN"
        
        if "_v2" in f.name:
            version = "v2"
        else:
            version = "v1"
        
        # Calculate metrics
        mae = (df["oil_rate_bpd"] - df["pred_oil_rate_bpd"]).abs().mean()
        rmse = np.sqrt(((df["oil_rate_bpd"] - df["pred_oil_rate_bpd"]) ** 2).mean())
        mape = ((df["oil_rate_bpd"] - df["pred_oil_rate_bpd"]).abs() / df["oil_rate_bpd"].replace(0, np.nan)).mean() * 100
        
        # R-squared
        ss_res = ((df["oil_rate_bpd"] - df["pred_oil_rate_bpd"]) ** 2).sum()
        ss_tot = ((df["oil_rate_bpd"] - df["oil_rate_bpd"].mean()) ** 2).sum()
        r2 = 1 - (ss_res / ss_tot)
        
        results.append({
            "Model": f"{model}_{version}",
            "MAE (bpd)": mae,
            "RMSE (bpd)": rmse,
            "MAPE (%)": mape,
            "R² Score": r2,
            "Samples": len(df)
        })
    
    results_df = pd.DataFrame(results).sort_values("MAE (bpd)")
    print("\n" + results_df.to_string(index=False))
    
    # Save to CSV
    results_df.to_csv(BASE / "model_comparison_summary.csv", index=False)
    print(f"\n✓ Saved: {BASE / 'model_comparison_summary.csv'}")
    
    return results_df


def main():
    """Run all comparisons."""
    print("="*70)
    print("TDM RESULTS COMPARISON TOOL")
    print("="*70)
    
    # Compare MAE by well
    mae_df = compare_mae_tables()
    
    # Plot comparisons
    if mae_df is not None:
        plot_mae_comparison(mae_df)
    
    # Compare overall predictions
    pred_summary = compare_predictions()
    
    # Final summary
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    if pred_summary is not None and len(pred_summary) > 1:
        best = pred_summary.iloc[0]
        print(f"\n✅ Best Model: {best['Model']}")
        print(f"   MAE: {best['MAE (bpd)']:.2f} bpd")
        print(f"   R² Score: {best['R² Score']:.3f}")
        
        if len(pred_summary) > 1:
            baseline = pred_summary[pred_summary["Model"].str.contains("v1")].iloc[0] if any(pred_summary["Model"].str.contains("v1")) else pred_summary.iloc[1]
            improvement = ((baseline["MAE (bpd)"] - best["MAE (bpd)"]) / baseline["MAE (bpd)"]) * 100
            
            if improvement > 5:
                print(f"\n💡 Using enhanced features improved accuracy by {improvement:.1f}%")
                print("   Recommendation: Deploy the v2 (enhanced) model")
            elif improvement > 0:
                print(f"\n💡 Modest improvement of {improvement:.1f}%")
                print("   Recommendation: Enhanced features help slightly; consider further tuning")
            else:
                print(f"\n⚠️  Enhanced features did not improve accuracy")
                print("   Recommendation: Stick with baseline or try different feature engineering")
    
    print("\n" + "="*70)
    print("\nAll comparison plots saved to: plots/")
    print("Run 'python tdm_plotter.py' to see per-well predictions")


if __name__ == "__main__":
    main()
