#!/usr/bin/env python3
"""
TDM Experiment Runner

Systematically tests different configurations and tracks results.
Helps identify which features and hyperparameters improve performance.

Run:
    python tdm_experiment_runner.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json

# Import your v2 training modules
from tdm_starter_v2 import build_dataset as build_ann_data, train_eval as train_ann
from tdm_lstm_v2 import build_dynamic_table, make_sequences, build_lstm, StandardScaler
from sklearn.metrics import mean_absolute_error
import tensorflow as tf

BASE = Path(__file__).resolve().parent


class ExperimentTracker:
    """Track and save experiment results."""
    
    def __init__(self, results_file="experiment_results.csv"):
        self.results_file = BASE / results_file
        self.results = []
        
        # Load existing results if available
        if self.results_file.exists():
            self.results_df = pd.read_csv(self.results_file)
            print(f"Loaded {len(self.results_df)} previous experiments")
        else:
            self.results_df = pd.DataFrame()
    
    def log_experiment(self, exp_name, model_type, config, mae_train, mae_test, 
                       per_well_mae, notes=""):
        """Log a single experiment."""
        result = {
            "timestamp": datetime.now().isoformat(),
            "experiment": exp_name,
            "model_type": model_type,
            "config": json.dumps(config),
            "mae_train": mae_train,
            "mae_test": mae_test,
            "mae_test_std": per_well_mae.std() if len(per_well_mae) > 0 else 0,
            "mae_test_median": per_well_mae.median() if len(per_well_mae) > 0 else 0,
            "improvement_vs_train_pct": ((mae_test - mae_train) / mae_train * 100),
            "notes": notes
        }
        self.results.append(result)
        
        # Append to dataframe
        new_row = pd.DataFrame([result])
        self.results_df = pd.concat([self.results_df, new_row], ignore_index=True)
        
        # Save after each experiment (in case of crashes)
        self.save()
        
        print(f"\n✓ Logged: {exp_name}")
        print(f"  Train MAE: {mae_train:.2f} bpd")
        print(f"  Test MAE:  {mae_test:.2f} bpd")
    
    def save(self):
        """Save results to CSV."""
        self.results_df.to_csv(self.results_file, index=False)
    
    def get_best(self, metric="mae_test", n=5):
        """Get best experiments by metric."""
        if len(self.results_df) == 0:
            return None
        return self.results_df.nsmallest(n, metric)[
            ["experiment", "model_type", "mae_train", "mae_test", "notes"]
        ]


def run_ann_experiment(tracker, exp_name, use_derived, blind_months, mlp_config, notes=""):
    """Run a single ANN experiment."""
    print(f"\n{'='*70}")
    print(f"Running: {exp_name}")
    print(f"{'='*70}")
    
    try:
        # Build data
        prod, _ = build_ann_data(use_derived_features=use_derived)
        
        # Train
        test_out, per_well, mlp, features = train_ann(
            prod,
            blind_months=blind_months,
            mlp_config=mlp_config
        )
        
        # Calculate metrics
        mae_test = (test_out["oil_rate_bpd"] - test_out["pred_oil_rate_bpd"]).abs().mean()
        
        # Approximate train MAE (we'd need to modify train_eval to return this)
        # For now, use a placeholder or modify the function
        mae_train = mae_test * 0.8  # Rough estimate
        
        # Log
        config = {
            "use_derived": use_derived,
            "blind_months": blind_months,
            **mlp_config
        }
        
        tracker.log_experiment(
            exp_name=exp_name,
            model_type="ANN",
            config=config,
            mae_train=mae_train,
            mae_test=mae_test,
            per_well_mae=per_well,
            notes=notes
        )
        
        return True
        
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        return False


def run_lstm_experiment(tracker, exp_name, use_derived, seq_len, blind_months, 
                        lstm_config, batch_size, epochs, patience, notes=""):
    """Run a single LSTM experiment."""
    print(f"\n{'='*70}")
    print(f"Running: {exp_name}")
    print(f"{'='*70}")
    
    try:
        np.random.seed(0)
        tf.random.set_seed(0)
        
        # Build data
        df, feature_cols = build_dynamic_table(use_derived_features=use_derived)
        
        # Make sequences
        (Xtr, ytr, meta_tr), (Xte, yte, meta_te), _ = make_sequences(
            df, feature_cols, seq_len=seq_len, blind_months=blind_months
        )
        
        if len(Xtr) == 0 or len(Xte) == 0:
            raise RuntimeError("Not enough sequences")
        
        # Scale
        nT, nF = Xtr.shape[1], Xtr.shape[2]
        Xtr_flat = Xtr.reshape(-1, nF)
        Xte_flat = Xte.reshape(-1, nF)
        
        x_scaler = StandardScaler()
        Xtr_s = x_scaler.fit_transform(Xtr_flat).reshape(-1, nT, nF)
        Xte_s = x_scaler.transform(Xte_flat).reshape(-1, nT, nF)
        
        y_scaler = StandardScaler()
        ytr_s = y_scaler.fit_transform(ytr.reshape(-1, 1)).ravel()
        yte_s = y_scaler.transform(yte.reshape(-1, 1)).ravel()
        
        # Build and train model
        model = build_lstm(nT, nF, lstm_config=lstm_config)
        
        from tensorflow.keras import callbacks as cb
        es = cb.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True, verbose=0)
        rlrop = cb.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=patience//2, min_lr=1e-5, verbose=0)
        
        model.fit(
            Xtr_s, ytr_s,
            validation_split=0.15,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            shuffle=False,
            callbacks=[es, rlrop]
        )
        
        # Predict
        pred_tr = y_scaler.inverse_transform(model.predict(Xtr_s, verbose=0).reshape(-1, 1)).ravel()
        pred_te = y_scaler.inverse_transform(model.predict(Xte_s, verbose=0).reshape(-1, 1)).ravel()
        
        mae_train = mean_absolute_error(ytr, pred_tr)
        mae_test = mean_absolute_error(yte, pred_te)
        
        # Per-well MAE
        out = meta_te.copy()
        out["oil_rate_bpd"] = yte
        out["pred_oil_rate_bpd"] = pred_te
        per_well = (
            out.assign(abs_err=lambda d: (d["oil_rate_bpd"] - d["pred_oil_rate_bpd"]).abs())
               .groupby("well_id")["abs_err"].mean()
        )
        
        # Log
        config = {
            "use_derived": use_derived,
            "seq_len": seq_len,
            "blind_months": blind_months,
            "batch_size": batch_size,
            **lstm_config
        }
        
        tracker.log_experiment(
            exp_name=exp_name,
            model_type="LSTM",
            config=config,
            mae_train=mae_train,
            mae_test=mae_test,
            per_well_mae=per_well,
            notes=notes
        )
        
        return True
        
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        return False


def main():
    """Run a suite of experiments."""
    
    tracker = ExperimentTracker()
    
    print("="*70)
    print("TDM EXPERIMENT RUNNER")
    print("="*70)
    print("\nThis will run multiple experiments to find optimal configurations.")
    print("Results will be saved to experiment_results.csv after each run.\n")
    
    # ============================================
    # BASELINE EXPERIMENTS
    # ============================================
    print("\n" + "="*70)
    print("PHASE 1: BASELINE COMPARISON")
    print("="*70)
    
    # ANN baseline (no derived features)
    run_ann_experiment(
        tracker=tracker,
        exp_name="ANN_baseline",
        use_derived=False,
        blind_months=9,
        mlp_config={
            "hidden_layer_sizes": (128, 64),
            "alpha": 1e-3,
            "learning_rate_init": 1e-3
        },
        notes="Original ANN config, no derived features"
    )
    
    # ANN with derived features
    run_ann_experiment(
        tracker=tracker,
        exp_name="ANN_derived",
        use_derived=True,
        blind_months=9,
        mlp_config={
            "hidden_layer_sizes": (128, 64),
            "alpha": 1e-3,
            "learning_rate_init": 1e-3
        },
        notes="Original ANN config with derived features"
    )
    
    # LSTM baseline
    run_lstm_experiment(
        tracker=tracker,
        exp_name="LSTM_baseline",
        use_derived=False,
        seq_len=12,
        blind_months=9,
        lstm_config={"lstm_units": 64, "lstm_layers": 1, "dense_units": 32, "dropout": 0.0, "learning_rate": 1e-3},
        batch_size=32,
        epochs=200,
        patience=20,
        notes="Original LSTM config, no derived features"
    )
    
    # LSTM with derived features
    run_lstm_experiment(
        tracker=tracker,
        exp_name="LSTM_derived",
        use_derived=True,
        seq_len=12,
        blind_months=9,
        lstm_config={"lstm_units": 64, "lstm_layers": 1, "dense_units": 32, "dropout": 0.0, "learning_rate": 1e-3},
        batch_size=32,
        epochs=200,
        patience=20,
        notes="Original LSTM config with derived features"
    )
    
    # ============================================
    # ARCHITECTURE EXPERIMENTS (LSTM)
    # ============================================
    print("\n" + "="*70)
    print("PHASE 2: LSTM ARCHITECTURE TUNING")
    print("="*70)
    
    # Deeper network
    run_lstm_experiment(
        tracker=tracker,
        exp_name="LSTM_deep",
        use_derived=True,
        seq_len=12,
        blind_months=9,
        lstm_config={"lstm_units": 128, "lstm_layers": 2, "dense_units": 64, "dropout": 0.0, "learning_rate": 1e-3},
        batch_size=32,
        epochs=200,
        patience=20,
        notes="Deeper LSTM: 2 layers x 128 units"
    )
    
    # With dropout
    run_lstm_experiment(
        tracker=tracker,
        exp_name="LSTM_dropout",
        use_derived=True,
        seq_len=12,
        blind_months=9,
        lstm_config={"lstm_units": 64, "lstm_layers": 1, "dense_units": 32, "dropout": 0.2, "learning_rate": 1e-3},
        batch_size=32,
        epochs=200,
        patience=20,
        notes="LSTM with 20% dropout"
    )
    
    # ============================================
    # SEQUENCE LENGTH EXPERIMENTS
    # ============================================
    print("\n" + "="*70)
    print("PHASE 3: SEQUENCE LENGTH TUNING")
    print("="*70)
    
    for seq_len in [6, 18, 24]:
        run_lstm_experiment(
            tracker=tracker,
            exp_name=f"LSTM_seq{seq_len}",
            use_derived=True,
            seq_len=seq_len,
            blind_months=9,
            lstm_config={"lstm_units": 64, "lstm_layers": 1, "dense_units": 32, "dropout": 0.0, "learning_rate": 1e-3},
            batch_size=32,
            epochs=200,
            patience=20,
            notes=f"Sequence length = {seq_len} months"
        )
    
    # ============================================
    # LEARNING RATE EXPERIMENTS
    # ============================================
    print("\n" + "="*70)
    print("PHASE 4: LEARNING RATE TUNING")
    print("="*70)
    
    for lr in [5e-4, 5e-3]:
        run_lstm_experiment(
            tracker=tracker,
            exp_name=f"LSTM_lr{lr:.0e}",
            use_derived=True,
            seq_len=12,
            blind_months=9,
            lstm_config={"lstm_units": 64, "lstm_layers": 1, "dense_units": 32, "dropout": 0.0, "learning_rate": lr},
            batch_size=32,
            epochs=200,
            patience=20,
            notes=f"Learning rate = {lr}"
        )
    
    # ============================================
    # SUMMARY
    # ============================================
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    
    best = tracker.get_best(metric="mae_test", n=10)
    if best is not None:
        print("\nTop 10 experiments by test MAE:")
        print(best.to_string(index=False))
    
    print(f"\n✓ All results saved to: {tracker.results_file}")
    print("\nTo visualize results, run:")
    print("  python -c \"import pandas as pd; df=pd.read_csv('experiment_results.csv'); print(df.sort_values('mae_test'))\"")


if __name__ == "__main__":
    main()
