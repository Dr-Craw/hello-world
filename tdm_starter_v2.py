#!/usr/bin/env python3
"""
TDM Starter v2 (Enhanced ANN with Derived Features)

Changes from v1:
- Uses tdm_features module for feature engineering
- Includes physics-based derived features
- Configurable feature selection
- Tracks which features are most important

Run:
    python tdm_starter_v2.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Import our feature engineering module
from tdm_features import add_derived_features, select_features_for_model

BASE = Path(__file__).resolve().parent


def load_data():
    prod = pd.read_csv(BASE / "production_timeseries.csv", parse_dates=["date"])
    static = pd.read_csv(BASE / "wells_static.csv")
    return prod, static


def build_dataset(use_derived_features=True):
    """
    Build enhanced dataset with derived features.
    
    Parameters:
    -----------
    use_derived_features : bool
        If True, adds physics-based derived features
    """
    prod, static = load_data()
    prod = prod[prod["status"] == "producer"].copy()
    
    # Add placeholder injection feature (until we have real coordinates)
    prod["inj_idw"] = 0.0
    
    if use_derived_features:
        print("Adding derived features...")
        prod = add_derived_features(prod, include_lags=True, lag_list=(1, 2, 3))
    else:
        print("Using only base features with lags...")
        # Simple lag addition (original approach)
        base_cols = ["oil_rate_bpd", "gas_rate_mscfd", "water_rate_bpd",
                     "watercut_frac", "whp_psia", "wht_degF", "choke_64", "inj_idw"]
        for col in base_cols:
            for lag in (1, 2, 3):
                prod[f"{col}_tminus{lag}"] = prod.groupby("well_id")[col].shift(lag)
    
    # Drop rows with NaN in lagged features
    prod = prod.dropna().reset_index(drop=True)
    
    return prod, static


def train_eval(prod, blind_months=9, mlp_config=None):
    """
    Train ANN and evaluate on blind test set.
    
    Parameters:
    -----------
    prod : pd.DataFrame
        Production data with features
    blind_months : int
        Number of months to hold out for blind testing
    mlp_config : dict
        Custom MLPRegressor parameters (optional)
    """
    prod = prod.sort_values(["date", "well_id"])
    cutoff = prod["date"].max() - pd.offsets.MonthBegin(blind_months)
    
    train = prod[prod["date"] < cutoff].copy()
    test = prod[prod["date"] >= cutoff].copy()
    
    # Select features (all lagged columns)
    features = [c for c in prod.columns if "_tminus" in c]
    target = "oil_rate_bpd"
    
    print(f"\nUsing {len(features)} features")
    print(f"Train samples: {len(train):,}")
    print(f"Test samples:  {len(test):,}")
    print(f"Blind cutoff:  {cutoff.date()}")
    
    Xtr, ytr = train[features].values, train[target].values
    Xte, yte = test[features].values, test[target].values
    
    # Scale features
    x_scaler = StandardScaler()
    Xtr_s = x_scaler.fit_transform(Xtr)
    Xte_s = x_scaler.transform(Xte)
    
    # Scale target
    y_scaler = StandardScaler()
    ytr_s = y_scaler.fit_transform(ytr.reshape(-1, 1)).ravel()
    yte_s = y_scaler.transform(yte.reshape(-1, 1)).ravel()
    
    # Default MLP configuration
    default_config = {
        "hidden_layer_sizes": (128, 64),
        "activation": "relu",
        "solver": "adam",
        "alpha": 1e-3,
        "learning_rate": "adaptive",
        "learning_rate_init": 1e-3,
        "batch_size": 256,
        "max_iter": 3000,
        "tol": 1e-5,
        "early_stopping": True,
        "validation_fraction": 0.15,
        "n_iter_no_change": 30,
        "random_state": 0,
        "verbose": True
    }
    
    if mlp_config:
        default_config.update(mlp_config)
    
    print("\nTraining ANN with config:")
    for k, v in default_config.items():
        if k not in ["verbose"]:
            print(f"  {k:25s}: {v}")
    
    mlp = MLPRegressor(**default_config)
    mlp.fit(Xtr_s, ytr_s)
    
    # Predictions
    pred_tr_s = mlp.predict(Xtr_s)
    pred_te_s = mlp.predict(Xte_s)
    pred_tr = y_scaler.inverse_transform(pred_tr_s.reshape(-1, 1)).ravel()
    pred_te = y_scaler.inverse_transform(pred_te_s.reshape(-1, 1)).ravel()
    
    # Metrics
    mae_tr = mean_absolute_error(ytr, pred_tr)
    mae_te = mean_absolute_error(yte, pred_te)
    
    print(f"\n{'='*60}")
    print(f"RESULTS:")
    print(f"{'='*60}")
    print(f"Train MAE: {mae_tr:,.2f} bpd")
    print(f"Blind MAE: {mae_te:,.2f} bpd")
    print(f"Improvement over train: {((mae_te - mae_tr) / mae_tr * 100):+.1f}%")
    
    # Output predictions
    test_out = test[["well_id", "date", target]].copy()
    test_out["pred_oil_rate_bpd"] = pred_te
    
    # Per-well MAE
    per_well = (
        test_out.assign(abs_err=lambda d: (d[target] - d["pred_oil_rate_bpd"]).abs())
                .groupby("well_id")["abs_err"].mean()
                .sort_values(ascending=False)
                .rename("blind_mae_bpd")
    )
    
    print(f"\nPer-well MAE (top 5 worst):")
    print(per_well.head().to_string())
    
    return test_out, per_well, mlp, features


def analyze_feature_importance(mlp, feature_names, top_n=20):
    """
    Approximate feature importance using input layer weights.
    Higher absolute weights suggest more important features.
    
    Note: This is a rough approximation. For true importance,
    consider permutation importance or SHAP values.
    """
    # Get weights from first layer
    weights = mlp.coefs_[0]  # shape: (n_features, n_hidden_units)
    
    # Compute average absolute weight per feature
    importance = np.abs(weights).mean(axis=1)
    
    # Create dataframe
    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    }).sort_values("importance", ascending=False)
    
    print(f"\n{'='*60}")
    print(f"TOP {top_n} MOST IMPORTANT FEATURES (by avg abs weight):")
    print(f"{'='*60}")
    print(imp_df.head(top_n).to_string(index=False))
    
    return imp_df


def main(use_derived_features=True, blind_months=9, mlp_config=None):
    """
    Main training pipeline.
    
    Parameters:
    -----------
    use_derived_features : bool
        Whether to include physics-based derived features
    blind_months : int
        Blind test period
    mlp_config : dict
        Custom MLP hyperparameters
    """
    print("="*60)
    print("TDM ANN v2 - Enhanced with Derived Features")
    print("="*60)
    
    # Build dataset
    prod, static = build_dataset(use_derived_features=use_derived_features)
    
    # Train and evaluate
    test_out, per_well, mlp, features = train_eval(
        prod, 
        blind_months=blind_months,
        mlp_config=mlp_config
    )
    
    # Analyze feature importance
    importance_df = analyze_feature_importance(mlp, features, top_n=20)
    
    # Save outputs
    suffix = "_v2" if use_derived_features else "_v1"
    test_out.to_csv(BASE / f"tdm_oil_predictions_blind{suffix}.csv", index=False)
    per_well.to_csv(BASE / f"tdm_blind_mae_by_well{suffix}.csv")
    importance_df.to_csv(BASE / f"tdm_feature_importance{suffix}.csv", index=False)
    
    print(f"\n✓ Wrote outputs with suffix '{suffix}'")
    
    return test_out, per_well, mlp, importance_df


if __name__ == "__main__":
    # Example 1: Run with derived features (recommended)
    main(use_derived_features=True, blind_months=9)
    
    # Example 2: Run with custom hyperparameters
    # custom_config = {
    #     "hidden_layer_sizes": (256, 128, 64),
    #     "alpha": 1e-4,
    #     "learning_rate_init": 5e-4,
    # }
    # main(use_derived_features=True, mlp_config=custom_config)
    
    # Example 3: Compare with baseline (no derived features)
    # main(use_derived_features=False, blind_months=9)
