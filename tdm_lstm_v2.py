#!/usr/bin/env python3
"""
TDM LSTM v2 (Enhanced with Derived Features)

Changes from v1:
- Uses tdm_features module for feature engineering
- Includes physics-based derived features
- No lagged features needed (LSTM handles sequences)
- Configurable sequence length

Run:
    python tdm_lstm_v2.py
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

import tensorflow as tf
from tensorflow.keras import layers, callbacks, models

# Import our feature engineering module
from tdm_features import add_derived_features, select_features_for_model

BASE = Path(__file__).resolve().parent


def load_data():
    prod = pd.read_csv(BASE / "production_timeseries.csv", parse_dates=["date"])
    static = pd.read_csv(BASE / "wells_static.csv")
    return prod, static


def build_dynamic_table(use_derived_features=True):
    """
    Build production table with optional derived features.
    
    Parameters:
    -----------
    use_derived_features : bool
        Whether to include physics-based derived features
    """
    prod, static = load_data()
    prod = prod[prod["status"] == "producer"].copy()
    
    # Add placeholder injection feature
    prod["inj_idw"] = 0.0
    
    if use_derived_features:
        print("Adding derived features...")
        # Don't include lags for LSTM (it handles sequences natively)
        prod = add_derived_features(prod, include_lags=False)
        # Select features appropriate for LSTM
        feature_cols = select_features_for_model(prod, model_type="lstm", include_injection=True)
    else:
        print("Using only base features...")
        feature_cols = [
            "oil_rate_bpd", "gas_rate_mscfd", "water_rate_bpd",
            "watercut_frac", "whp_psia", "wht_degF", "choke_64", "inj_idw"
        ]
    
    # Sort for sequence construction
    prod = prod.sort_values(["well_id", "date"]).reset_index(drop=True)
    
    return prod, feature_cols


def make_sequences(df, feature_cols, seq_len=12, blind_months=9):
    """
    Build sequences for LSTM training.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Production data
    feature_cols : list
        Column names to use as features
    seq_len : int
        Sequence length (number of historical months)
    blind_months : int
        Months to hold out for testing
    
    Returns:
    --------
    (X_tr, y_tr, meta_tr), (X_te, y_te, meta_te), feature_cols
    """
    # Time-based cutoff for blind tail
    cutoff = df["date"].max() - pd.offsets.MonthBegin(blind_months)
    
    print(f"\nSequence configuration:")
    print(f"  Sequence length: {seq_len} months")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Blind cutoff: {cutoff.date()}")
    
    X_tr, y_tr, wid_tr, dt_tr = [], [], [], []
    X_te, y_te, wid_te, dt_te = [], [], [], []
    
    wells_skipped = 0
    
    for wid, g in df.groupby("well_id"):
        g = g.sort_values("date").reset_index(drop=True)
        
        # Drop rows with missing data in required columns
        required_cols = feature_cols + ["oil_rate_bpd"]
        if g[required_cols].isna().any().any():
            g = g.dropna(subset=required_cols).reset_index(drop=True)
        
        if len(g) <= seq_len:
            wells_skipped += 1
            continue
        
        # Build sequences
        for end_idx in range(seq_len, len(g)):
            # Sequence covers [end_idx - seq_len : end_idx-1]
            Xwin = g.loc[end_idx - seq_len:end_idx - 1, feature_cols].values.astype(np.float32)
            y_next = float(g.loc[end_idx, "oil_rate_bpd"])
            end_date = g.loc[end_idx, "date"]
            
            if end_date < cutoff:
                X_tr.append(Xwin)
                y_tr.append(y_next)
                wid_tr.append(wid)
                dt_tr.append(end_date)
            else:
                X_te.append(Xwin)
                y_te.append(y_next)
                wid_te.append(wid)
                dt_te.append(end_date)
    
    # Convert to arrays
    X_tr = np.array(X_tr)
    y_tr = np.array(y_tr)
    X_te = np.array(X_te)
    y_te = np.array(y_te)
    
    tr_meta = pd.DataFrame({"well_id": wid_tr, "date": dt_tr})
    te_meta = pd.DataFrame({"well_id": wid_te, "date": dt_te})
    
    print(f"\nSequence statistics:")
    print(f"  Wells skipped (insufficient data): {wells_skipped}")
    print(f"  Training sequences: {len(X_tr):,}")
    print(f"  Test sequences: {len(X_te):,}")
    
    return (X_tr, y_tr, tr_meta), (X_te, y_te, te_meta), feature_cols


def build_lstm(input_timesteps, input_features, lstm_config=None):
    """
    Build LSTM model with configurable architecture.
    
    Parameters:
    -----------
    input_timesteps : int
        Sequence length
    input_features : int
        Number of features per timestep
    lstm_config : dict
        Custom architecture parameters (optional)
    
    Returns:
    --------
    Compiled Keras model
    """
    # Default configuration
    default_config = {
        "lstm_units": 64,
        "lstm_layers": 1,
        "dense_units": 32,
        "dropout": 0.0,
        "learning_rate": 1e-3,
        "loss": "mae"
    }
    
    if lstm_config:
        default_config.update(lstm_config)
    
    cfg = default_config
    
    # Build model
    model = models.Sequential()
    model.add(layers.Input(shape=(input_timesteps, input_features)))
    
    # LSTM layers
    for i in range(cfg["lstm_layers"]):
        return_sequences = (i < cfg["lstm_layers"] - 1)  # Only last LSTM doesn't return sequences
        model.add(layers.LSTM(
            cfg["lstm_units"],
            return_sequences=return_sequences,
            dropout=cfg["dropout"] if cfg["dropout"] > 0 else 0.0
        ))
    
    # Dense layers
    model.add(layers.Dense(cfg["dense_units"], activation="relu"))
    model.add(layers.Dense(1))  # oil_rate_bpd prediction
    
    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg["learning_rate"]),
        loss=cfg["loss"],
        metrics=["mae"]
    )
    
    print(f"\nModel architecture:")
    print(f"  LSTM layers: {cfg['lstm_layers']} x {cfg['lstm_units']} units")
    print(f"  Dense layer: {cfg['dense_units']} units")
    print(f"  Dropout: {cfg['dropout']}")
    print(f"  Learning rate: {cfg['learning_rate']}")
    print(f"  Loss: {cfg['loss']}")
    
    return model


def main(seq_len=12, blind_months=9, batch_size=32, epochs=200, patience=20,
         random_state=0, use_derived_features=True, lstm_config=None):
    """
    Main LSTM training pipeline.
    
    Parameters:
    -----------
    seq_len : int
        Number of historical months to use
    blind_months : int
        Blind test period
    batch_size : int
        Training batch size
    epochs : int
        Maximum training epochs
    patience : int
        Early stopping patience
    random_state : int
        Random seed
    use_derived_features : bool
        Whether to include physics-based features
    lstm_config : dict
        Custom LSTM architecture parameters
    """
    print("="*60)
    print("TDM LSTM v2 - Enhanced with Derived Features")
    print("="*60)
    
    np.random.seed(random_state)
    tf.random.set_seed(random_state)
    
    # Build dataset
    df, feature_cols = build_dynamic_table(use_derived_features=use_derived_features)
    
    # Make sequences
    (Xtr, ytr, meta_tr), (Xte, yte, meta_te), feature_cols = make_sequences(
        df, feature_cols, seq_len=seq_len, blind_months=blind_months
    )
    
    if len(Xtr) == 0 or len(Xte) == 0:
        raise RuntimeError("Not enough sequences to train/test. Check your data and parameters.")
    
    # Scale features (flatten time into rows, fit scaler on train only)
    nT, nF = Xtr.shape[1], Xtr.shape[2]
    Xtr_flat = Xtr.reshape(-1, nF)
    Xte_flat = Xte.reshape(-1, nF)
    
    x_scaler = StandardScaler()
    Xtr_flat_s = x_scaler.fit_transform(Xtr_flat)
    Xte_flat_s = x_scaler.transform(Xte_flat)
    
    Xtr_s = Xtr_flat_s.reshape(-1, nT, nF)
    Xte_s = Xte_flat_s.reshape(-1, nT, nF)
    
    # Scale target
    y_scaler = StandardScaler()
    ytr_s = y_scaler.fit_transform(ytr.reshape(-1, 1)).ravel()
    yte_s = y_scaler.transform(yte.reshape(-1, 1)).ravel()
    
    # Build model
    model = build_lstm(
        input_timesteps=nT,
        input_features=nF,
        lstm_config=lstm_config
    )
    
    # Callbacks
    es = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )
    rlrop = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=max(5, patience // 2),
        min_lr=1e-5,
        verbose=1
    )
    
    print(f"\nTraining configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Max epochs: {epochs}")
    print(f"  Early stopping patience: {patience}")
    print(f"  Validation split: 15%")
    
    # Train
    history = model.fit(
        Xtr_s, ytr_s,
        validation_split=0.15,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        shuffle=False,  # Preserve time ordering
        callbacks=[es, rlrop]
    )
    
    # Predictions (inverse target scaling)
    pred_tr_s = model.predict(Xtr_s, verbose=0).ravel()
    pred_te_s = model.predict(Xte_s, verbose=0).ravel()
    
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
    
    # Blind outputs
    out = meta_te.copy()
    out["oil_rate_bpd"] = yte
    out["pred_oil_rate_bpd"] = pred_te
    out = out.sort_values(["well_id", "date"]).reset_index(drop=True)
    
    # Per-well MAE
    per_well = (
        out.assign(abs_err=lambda d: (d["oil_rate_bpd"] - d["pred_oil_rate_bpd"]).abs())
           .groupby("well_id")["abs_err"].mean()
           .sort_values(ascending=False)
           .rename("blind_mae_bpd")
           .to_frame()
    )
    
    print(f"\nPer-well MAE (top 5 worst):")
    print(per_well.head().to_string())
    
    # Save outputs
    suffix = "_v2" if use_derived_features else "_v1"
    out.to_csv(BASE / f"tdm_lstm_oil_predictions_blind{suffix}.csv", index=False)
    per_well.to_csv(BASE / f"tdm_lstm_blind_mae_by_well{suffix}.csv")
    
    print(f"\n✓ Wrote outputs with suffix '{suffix}'")
    
    # Save training history
    hist_df = pd.DataFrame(history.history)
    hist_df["epoch"] = range(len(hist_df))
    hist_df.to_csv(BASE / f"tdm_lstm_training_history{suffix}.csv", index=False)
    
    return out, per_well, model, history


if __name__ == "__main__":
    # Example 1: Run with derived features and default settings
    main(
        seq_len=12,
        blind_months=9,
        use_derived_features=True
    )
    
    # Example 2: Try longer sequences
    # main(
    #     seq_len=24,
    #     blind_months=9,
    #     use_derived_features=True
    # )
    
    # Example 3: Try deeper LSTM with dropout
    # custom_lstm = {
    #     "lstm_units": 128,
    #     "lstm_layers": 2,
    #     "dense_units": 64,
    #     "dropout": 0.2,
    #     "learning_rate": 5e-4
    # }
    # main(
    #     seq_len=12,
    #     blind_months=9,
    #     use_derived_features=True,
    #     lstm_config=custom_lstm
    # )
    
    # Example 4: Compare with baseline (no derived features)
    # main(
    #     seq_len=12,
    #     blind_months=9,
    #     use_derived_features=False
    # )
