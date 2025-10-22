#!/usr/bin/env python3
"""
TDM Feature Engineering Module

Centralizes all feature engineering for both ANN and LSTM models.
Includes physics-based derived features for production forecasting.

Usage:
    from tdm_features import add_derived_features
    prod = add_derived_features(prod)
"""

import numpy as np
import pandas as pd


def add_derived_features(prod_df, include_lags=True, lag_list=(1, 2, 3)):
    """
    Add derived features to production dataframe.
    
    Parameters:
    -----------
    prod_df : pd.DataFrame
        Production timeseries with columns: well_id, date, oil_rate_bpd, etc.
    include_lags : bool
        Whether to add lagged features (needed for ANN, not for LSTM)
    lag_list : tuple
        Which lags to create (e.g., (1,2,3) for t-1, t-2, t-3)
    
    Returns:
    --------
    pd.DataFrame with additional derived features
    """
    df = prod_df.copy()
    df = df.sort_values(["well_id", "date"]).reset_index(drop=True)
    
    # ==========================================
    # 1. DECLINE ANALYSIS
    # ==========================================
    print("  → Adding decline features...")
    
    # Month-over-month decline rate
    df["oil_decline_rate"] = df.groupby("well_id")["oil_rate_bpd"].pct_change()
    df["oil_decline_rate"] = df["oil_decline_rate"].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # 3-month moving average decline (smoother signal)
    df["oil_decline_ma3"] = (
        df.groupby("well_id")["oil_decline_rate"]
          .transform(lambda x: x.rolling(3, min_periods=1).mean())
    )
    
    # ==========================================
    # 2. DEPLETION INDICATORS
    # ==========================================
    print("  → Adding depletion features...")
    
    # Cumulative oil production (proxy for reservoir depletion)
    # Approximate: rate * 30 days/month
    df["cum_oil_bbl"] = df.groupby("well_id")["oil_rate_bpd"].cumsum() * 30
    
    # Cumulative gas production
    df["cum_gas_mscf"] = df.groupby("well_id")["gas_rate_mscfd"].cumsum() * 30
    
    # Well age (months since first production)
    df["months_online"] = df.groupby("well_id").cumcount()
    
    # ==========================================
    # 3. PRESSURE & DRAWDOWN
    # ==========================================
    print("  → Adding pressure features...")
    
    # Normalized pressure (current / initial maximum)
    # Declining normalized pressure often indicates depletion
    df["whp_normalized"] = (
        df["whp_psia"] / 
        df.groupby("well_id")["whp_psia"].transform("max")
    ).fillna(1.0)
    
    # Pressure decline rate
    df["whp_decline_rate"] = df.groupby("well_id")["whp_psia"].pct_change()
    df["whp_decline_rate"] = df["whp_decline_rate"].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Temperature change (can indicate flow regime changes)
    df["wht_change"] = df.groupby("well_id")["wht_degF"].diff().fillna(0)
    
    # ==========================================
    # 4. RATE RATIOS (Key physics relationships)
    # ==========================================
    print("  → Adding ratio features...")
    
    # Gas-Oil Ratio (GOR) - increases as reservoir depletes
    # Convert to standard units: scf/bbl
    df["gor_scf_bbl"] = (
        (df["gas_rate_mscfd"] * 1000) / 
        df["oil_rate_bpd"].replace(0, np.nan)
    ).fillna(0).replace([np.inf, -np.inf], 0)
    
    # Water Cut (already in data, but add rate of change)
    df["watercut_change"] = df.groupby("well_id")["watercut_frac"].diff().fillna(0)
    
    # Liquid rate (oil + water)
    df["liquid_rate_bpd"] = df["oil_rate_bpd"] + df["water_rate_bpd"]
    
    # Oil fraction of total liquid
    df["oil_fraction"] = (
        df["oil_rate_bpd"] / 
        df["liquid_rate_bpd"].replace(0, np.nan)
    ).fillna(1.0).clip(0, 1)
    
    # ==========================================
    # 5. OPERATIONAL INDICATORS
    # ==========================================
    print("  → Adding operational features...")
    
    # Choke change (step changes can indicate interventions)
    df["choke_change"] = df.groupby("well_id")["choke_64"].diff().fillna(0)
    
    # Production efficiency: oil rate per unit pressure
    # Higher = better well performance
    df["prod_efficiency"] = (
        df["oil_rate_bpd"] / 
        df["whp_psia"].replace(0, np.nan)
    ).fillna(0).replace([np.inf, -np.inf], 0)
    
    # ==========================================
    # 6. ROLLING STATISTICS (Trend indicators)
    # ==========================================
    print("  → Adding rolling statistics...")
    
    # 3-month moving averages
    for col in ["oil_rate_bpd", "gas_rate_mscfd", "watercut_frac"]:
        df[f"{col}_ma3"] = (
            df.groupby("well_id")[col]
              .transform(lambda x: x.rolling(3, min_periods=1).mean())
        )
    
    # 3-month standard deviation (volatility measure)
    df["oil_rate_std3"] = (
        df.groupby("well_id")["oil_rate_bpd"]
          .transform(lambda x: x.rolling(3, min_periods=1).std())
    ).fillna(0)
    
    # Ratio of current rate to 3-month average
    # < 1 = declining, > 1 = increasing
    df["oil_rate_vs_ma3"] = (
        df["oil_rate_bpd"] / 
        df["oil_rate_bpd_ma3"].replace(0, np.nan)
    ).fillna(1.0)
    
    # ==========================================
    # 7. LAGGED FEATURES (for ANN model)
    # ==========================================
    if include_lags:
        print(f"  → Adding lags: {lag_list}...")
        
        # Core variables to lag
        lag_cols = [
            "oil_rate_bpd", "gas_rate_mscfd", "water_rate_bpd",
            "watercut_frac", "whp_psia", "wht_degF", "choke_64",
            "inj_idw",  # if you're using injection features
            # Derived features worth lagging:
            "oil_decline_rate", "gor_scf_bbl", "watercut_change",
            "whp_normalized", "oil_rate_vs_ma3"
        ]
        
        # Only lag columns that exist
        lag_cols = [c for c in lag_cols if c in df.columns]
        
        for col in lag_cols:
            for lag in lag_list:
                df[f"{col}_tminus{lag}"] = df.groupby("well_id")[col].shift(lag)
    
    # ==========================================
    # 8. HANDLE INFINITE/MISSING VALUES
    # ==========================================
    print("  → Cleaning infinite and missing values...")
    
    # Replace any remaining inf with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaNs in derived features with 0 (conservative choice)
    # Alternatively, you could forward-fill or drop
    derived_cols = [c for c in df.columns if c not in prod_df.columns]
    df[derived_cols] = df[derived_cols].fillna(0)
    
    print(f"  ✓ Added {len(derived_cols)} derived features")
    
    return df


def get_feature_groups():
    """
    Returns dictionary of feature groups for easy selection.
    
    Usage:
        groups = get_feature_groups()
        base_features = groups["base"]
        all_features = groups["base"] + groups["derived"]
    """
    return {
        "base": [
            "oil_rate_bpd", "gas_rate_mscfd", "water_rate_bpd",
            "watercut_frac", "whp_psia", "wht_degF", "choke_64"
        ],
        "decline": [
            "oil_decline_rate", "oil_decline_ma3"
        ],
        "depletion": [
            "cum_oil_bbl", "cum_gas_mscf", "months_online"
        ],
        "pressure": [
            "whp_normalized", "whp_decline_rate", "wht_change"
        ],
        "ratios": [
            "gor_scf_bbl", "watercut_change", "liquid_rate_bpd", 
            "oil_fraction"
        ],
        "operational": [
            "choke_change", "prod_efficiency"
        ],
        "rolling": [
            "oil_rate_bpd_ma3", "gas_rate_mscfd_ma3", "watercut_frac_ma3",
            "oil_rate_std3", "oil_rate_vs_ma3"
        ],
        "injection": [
            "inj_idw"  # if using
        ]
    }


def select_features_for_model(df, model_type="ann", include_injection=False):
    """
    Select appropriate features for a given model type.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with all features
    model_type : str
        "ann" (uses lagged features) or "lstm" (uses sequences)
    include_injection : bool
        Whether to include injection features
    
    Returns:
    --------
    list of column names to use as features
    """
    groups = get_feature_groups()
    
    if model_type == "ann":
        # ANN uses lagged versions of features
        # Find all columns ending with _tminus1, _tminus2, etc.
        lag_features = [c for c in df.columns if "_tminus" in c]
        return lag_features
    
    elif model_type == "lstm":
        # LSTM uses current timestep features (no lags needed)
        features = (
            groups["base"] + 
            groups["decline"] + 
            groups["depletion"] +
            groups["pressure"] +
            groups["ratios"] +
            groups["operational"] +
            groups["rolling"]
        )
        
        if include_injection:
            features += groups["injection"]
        
        # Only return features that exist in dataframe
        return [c for c in features if c in df.columns]
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# ==========================================
# EXAMPLE USAGE
# ==========================================
if __name__ == "__main__":
    """
    Demonstration of feature engineering pipeline
    """
    from pathlib import Path
    
    # Load data
    base = Path(__file__).resolve().parent
    prod = pd.read_csv(base / "production_timeseries.csv", parse_dates=["date"])
    
    print("Original columns:", prod.columns.tolist())
    print(f"Original shape: {prod.shape}")
    
    # Add placeholder injection feature (since we don't have real coords yet)
    prod["inj_idw"] = 0.0
    
    # Apply feature engineering
    print("\nAdding derived features...")
    prod_enhanced = add_derived_features(prod, include_lags=True, lag_list=(1, 2, 3))
    
    print(f"\nEnhanced shape: {prod_enhanced.shape}")
    print(f"New columns added: {prod_enhanced.shape[1] - prod.shape[1]}")
    
    # Show feature groups
    print("\n" + "="*60)
    print("FEATURE GROUPS:")
    print("="*60)
    groups = get_feature_groups()
    for name, cols in groups.items():
        available = [c for c in cols if c in prod_enhanced.columns]
        print(f"{name:15s}: {len(available):2d} features")
    
    # Show what each model would use
    print("\n" + "="*60)
    print("MODEL FEATURE SELECTION:")
    print("="*60)
    
    ann_features = select_features_for_model(prod_enhanced, model_type="ann")
    print(f"ANN would use {len(ann_features)} lagged features")
    print(f"Example: {ann_features[:5]}")
    
    lstm_features = select_features_for_model(prod_enhanced, model_type="lstm")
    print(f"\nLSTM would use {len(lstm_features)} current features")
    print(f"Features: {lstm_features}")
    
    # Save enhanced dataset
    prod_enhanced.to_csv(base / "production_timeseries_enhanced.csv", index=False)
    print(f"\n✓ Saved enhanced dataset to production_timeseries_enhanced.csv")
