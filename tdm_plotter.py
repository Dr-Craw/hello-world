#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
PLOTS_DIR = BASE / "plots"
PER_WELL_DIR = PLOTS_DIR / "per_well"

def load_csv(path):
    p = Path(path)
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df

def first_existing(*candidates):
    for c in candidates:
        df = load_csv(c)
        if df is not None:
            return df, Path(c)
    return None, None

def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    PER_WELL_DIR.mkdir(parents=True, exist_ok=True)

    # === History ===
    prod, _ = first_existing(BASE / "production_timeseries.csv")
    if prod is None:
        raise FileNotFoundError("production_timeseries.csv not found. Cannot plot history.")

    # producers only (if status missing, keep all)
    prod = prod[prod.get("status", "producer") == "producer"].copy()
    wells = sorted(prod["well_id"].unique())

    # === Predictions (prefer v2, then v1, then unsuffixed) ===
    ann, ann_path = first_existing(
        BASE / "tdm_oil_predictions_blind_v2.csv",
        BASE / "tdm_oil_predictions_blind_v1.csv",
        BASE / "tdm_oil_predictions_blind.csv"
    )
    lstm, lstm_path = first_existing(
        BASE / "tdm_lstm_oil_predictions_blind_v2.csv",
        BASE / "tdm_lstm_oil_predictions_blind_v1.csv",
        BASE / "tdm_lstm_oil_predictions_blind.csv"
    )

    mae_ann, mae_ann_path = first_existing(
        BASE / "tdm_blind_mae_by_well_v2.csv",
        BASE / "tdm_blind_mae_by_well_v1.csv",
        BASE / "tdm_blind_mae_by_well.csv"
    )
    mae_lstm, mae_lstm_path = first_existing(
        BASE / "tdm_lstm_blind_mae_by_well_v2.csv",
        BASE / "tdm_lstm_blind_mae_by_well_v1.csv",
        BASE / "tdm_lstm_blind_mae_by_well.csv"
    )

    preds = {}
    if ann is not None and {"well_id","date","pred_oil_rate_bpd"}.issubset(ann.columns):
        preds["ANN"] = ann.rename(columns={"pred_oil_rate_bpd": "pred"})
    if lstm is not None and {"well_id","date","pred_oil_rate_bpd"}.issubset(lstm.columns):
        preds["LSTM"] = lstm.rename(columns={"pred_oil_rate_bpd": "pred"})

    if not preds:
        print("No prediction files found (looked for v2/v1/unsuffixed). Nothing to plot.")
        return

    # === Per-well plots ===
    for wid in wells:
        g_hist = prod.loc[prod["well_id"] == wid, ["date", "oil_rate_bpd"]].sort_values("date")

        # Collect predictions for this well
        frames = []
        for name, df in preds.items():
            gp = df.loc[df["well_id"] == wid, ["date", "pred"]].copy().sort_values("date")
            if not gp.empty:
                gp["model"] = name
                frames.append(gp)

        if not frames:
            # No predictions for this well; skip
            continue

        g_pred = pd.concat(frames, ignore_index=True)
        blind_start = g_pred["date"].min() if not g_pred.empty else None

        fig, ax = plt.subplots(figsize=(9, 4.5))
        g_hist.plot(x="date", y="oil_rate_bpd", ax=ax, linewidth=1.5, label="History")

        for name in sorted(preds.keys()):
            gp = g_pred[g_pred["model"] == name]
            if not gp.empty:
                gp.plot(x="date", y="pred", ax=ax, linewidth=1.5, label=f"{name} (blind)")

        if blind_start is not None:
            ax.axvline(blind_start, linestyle="--", linewidth=1.0, alpha=0.8)

        ax.set_title(f"{wid} — Oil Rate (bpd)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Oil Rate (bpd)")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(PER_WELL_DIR / f"{wid}.png", dpi=150)
        plt.close(fig)

    # === MAE summary (if available) ===
    def _normalize_mae(df, tag):
        if df is None:
            return None
        if {"well_id", "blind_mae_bpd"}.issubset(df.columns):
            out = df[["well_id", "blind_mae_bpd"]].copy()
        elif "well_id" in df.columns and df.shape[1] >= 2:
            mae_col = [c for c in df.columns if c != "well_id"][0]
            out = df[["well_id", mae_col]].copy().rename(columns={mae_col: "blind_mae_bpd"})
        else:
            return None
        out = out.rename(columns={"blind_mae_bpd": f"{tag}_blind_mae_bpd"})
        return out

    a = _normalize_mae(mae_ann, "ANN")
    l = _normalize_mae(mae_lstm, "LSTM")

    if a is not None or l is not None:
        merged = a
        if merged is None:
            merged = l
        elif l is not None:
            merged = pd.merge(a, l, on="well_id", how="outer")

        merged.to_csv(PLOTS_DIR / "summary_ann_vs_lstm_mae.csv", index=False)

        if a is not None and l is not None:
            m2 = merged.sort_values("well_id").set_index("well_id")
            fig, ax = plt.subplots(figsize=(12, 6))
            m2.plot(kind="bar", ax=ax)
            ax.set_title("Blind MAE by Well — ANN vs LSTM (bpd)")
            ax.set_xlabel("Well")
            ax.set_ylabel("MAE (bpd)")
            ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            fig.savefig(PLOTS_DIR / "summary_ann_vs_lstm_mae.png", dpi=150)
            plt.close(fig)

    # Helpful breadcrumbs
    def _p(p): return p.name if p else "—"
    print("\nInputs used:")
    print(f"  ANN preds : {_p(ann_path)}")
    print(f"  LSTM preds: {_p(lstm_path)}")
    print(f"  ANN MAE   : {_p(mae_ann_path)}")
    print(f"  LSTM MAE  : {_p(mae_lstm_path)}")
    print(f"\nPer-well figures → {PER_WELL_DIR}")
    print(f"Summary outputs  → {PLOTS_DIR}")

if __name__ == "__main__":
    main()
