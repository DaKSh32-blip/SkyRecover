"""The Delay Forecaster: predict which flights will leave late, 60 min before departure.

How we test it honestly:
  * TIME SPLIT. Train on earlier dates, test on later dates the model has never
    seen. (Shuffling randomly would let the model peek at the same storm day
    in both training and testing, which inflates the score.)
  * BASELINES. A model is only impressive compared to something simple.
    We compare against "guess the usual delay for that airport and hour".
  * TWO MODELS.
      planning : schedule only (could run the night before)
      day-of   : schedule + live info (inbound plane, airport's last 3 hours)
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, precision_score, recall_score, roc_auc_score

from skyrecover import config
from skyrecover.features import CATEGORICAL, LIVE_FEATURES, SCHEDULE_FEATURES

PARAMS = dict(
    n_estimators=2000,       # max trees; early stopping picks the real number
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=50,    # each leaf needs 50+ flights, so it can't memorize single flights
    subsample=0.8, subsample_freq=1,
    colsample_bytree=0.8,
    verbose=-1,
)


@dataclass
class Split:
    train: pd.DataFrame
    valid: pd.DataFrame   # used only to decide when to stop training
    test: pd.DataFrame    # never touched until the final score


def time_split(df: pd.DataFrame, test_from: str, valid_days: int = 14) -> Split:
    test_start = pd.Timestamp(test_from)
    valid_start = test_start - pd.Timedelta(days=valid_days)
    return Split(
        train=df[df["sched_dep"] < valid_start],
        valid=df[(df["sched_dep"] >= valid_start) & (df["sched_dep"] < test_start)],
        test=df[df["sched_dep"] >= test_start],
    )


def baseline_predict(train: pd.DataFrame, test: pd.DataFrame):
    """Usual delay and usual late-rate for that (airport, hour), learned from training data.

    We use the MEDIAN delay, not the average: when you're scored on average
    error in minutes, the median is the best single guess. A baseline should be
    as strong as a simple method can be, otherwise beating it means nothing.
    """
    g = train.groupby(["origin", "dep_hour"], observed=True)
    stats = g.agg(median_delay=("target_delay", "median"), late_rate=("target_late", "mean")).reset_index()
    out = test[["origin", "dep_hour"]].merge(stats, on=["origin", "dep_hour"], how="left")
    delay = out["median_delay"].fillna(train["target_delay"].median()).to_numpy()
    late = out["late_rate"].fillna(train["target_late"].mean()).to_numpy()
    return delay, late


def _scores(y_delay, y_late, pred_delay, pred_prob) -> dict:
    flagged = pred_prob >= 0.5
    return {
        "mae_minutes": round(float(mean_absolute_error(y_delay, pred_delay)), 2),
        "auc": round(float(roc_auc_score(y_late, pred_prob)), 4),
        "precision": round(float(precision_score(y_late, flagged, zero_division=0)), 3),
        "recall": round(float(recall_score(y_late, flagged, zero_division=0)), 3),
    }


def train_and_evaluate(features: pd.DataFrame, test_from: str = "2024-03-01") -> dict:
    s = time_split(features, test_from)
    y_test_delay, y_test_late = s.test["target_delay"], s.test["target_late"]

    results = {
        "data": {
            "train_flights": len(s.train), "valid_flights": len(s.valid), "test_flights": len(s.test),
            "train_range": f"{s.train['sched_dep'].min():%Y-%m-%d} to {s.train['sched_dep'].max():%Y-%m-%d}",
            "test_range": f"{s.test['sched_dep'].min():%Y-%m-%d} to {s.test['sched_dep'].max():%Y-%m-%d}",
            "late_rate_test": round(float(y_test_late.mean()), 3),
        },
        "models": {},
    }

    # Baseline 1: always guess "on time"
    zeros = np.zeros(len(s.test))
    results["models"]["always_on_time"] = {
        "mae_minutes": round(float(mean_absolute_error(y_test_delay, zeros)), 2),
        "auc": 0.5, "precision": 0.0, "recall": 0.0,
    }
    # Baseline 2: usual delay for that airport + hour
    b_delay, b_late = baseline_predict(s.train, s.test)
    results["models"]["airport_hour_average"] = _scores(y_test_delay, y_test_late, b_delay, b_late)

    # Baseline 3: the ripple formula from piece 3 on its own (no machine learning).
    # Predicted delay = what the inbound plane pushes onto us. This tests whether
    # ML adds anything beyond simple slack math.
    pushed = s.test["inbound_pushed_known"].fillna(0).to_numpy()
    rip = _scores(y_test_delay, y_test_late, pushed, pushed)  # bigger push = more likely late
    rip["precision"] = round(float(precision_score(y_test_late, pushed >= config.DELAY_THRESHOLD_MINUTES, zero_division=0)), 3)
    rip["recall"] = round(float(recall_score(y_test_late, pushed >= config.DELAY_THRESHOLD_MINUTES, zero_division=0)), 3)
    results["models"]["ripple_formula"] = rip

    trained = {}
    for name, cols in [("planning", SCHEDULE_FEATURES), ("day_of", SCHEDULE_FEATURES + LIVE_FEATURES)]:
        cats = [c for c in cols if c in CATEGORICAL]
        stop = [lgb.early_stopping(100, verbose=False)]

        warnings.filterwarnings("ignore", message=".*eval_set.*")  # harmless LightGBM API notice
        reg = lgb.LGBMRegressor(objective="l1", **PARAMS)   # l1 = directly minimizes average error
        reg.fit(s.train[cols], s.train["target_delay"], eval_set=[(s.valid[cols], s.valid["target_delay"])],
                categorical_feature=cats, callbacks=stop)

        clf = lgb.LGBMClassifier(objective="binary", **PARAMS)
        clf.fit(s.train[cols], s.train["target_late"], eval_set=[(s.valid[cols], s.valid["target_late"])],
                categorical_feature=cats, callbacks=stop)

        pred_delay = np.clip(reg.predict(s.test[cols]), 0, None)
        pred_prob = clf.predict_proba(s.test[cols])[:, 1]
        results["models"][name] = _scores(y_test_delay, y_test_late, pred_delay, pred_prob)

        imp = pd.Series(clf.booster_.feature_importance("gain"), index=cols)
        results["models"][name]["top_features"] = (imp / imp.sum()).sort_values(ascending=False).head(6).round(3).to_dict()
        trained[name] = (reg, clf, cols)

    return results, trained


def save(results: dict, trained: dict) -> None:
    config.MODELS_DIR.mkdir(exist_ok=True)
    config.REPORTS_DIR.mkdir(exist_ok=True)
    for name, (reg, clf, _) in trained.items():
        reg.booster_.save_model(str(config.MODELS_DIR / f"{name}_delay.txt"))
        clf.booster_.save_model(str(config.MODELS_DIR / f"{name}_late.txt"))
    (config.REPORTS_DIR / "forecast_metrics.json").write_text(json.dumps(results, indent=2))
    (config.REPORTS_DIR / "forecast_report.md").write_text(to_markdown(results))


def to_markdown(r: dict) -> str:
    d = r["data"]
    names = {"always_on_time": "Always guess on time", "airport_hour_average": "Airport + hour typical delay",
             "ripple_formula": "Ripple formula only (no ML)",
             "planning": "SkyRecover, planning (schedule only)", "day_of": "SkyRecover, day-of (schedule + live)"}
    lines = [
        "# Delay Forecaster Results", "",
        f"Trained on {d['train_flights']:,} Delta flights ({d['train_range']}), "
        f"tested on {d['test_flights']:,} flights it had never seen ({d['test_range']}). "
        f"{d['late_rate_test']:.0%} of test flights left 15+ minutes late.", "",
        "| Model | Avg error (min) | AUC | Precision | Recall |", "|---|---|---|---|---|",
    ]
    for k, m in r["models"].items():
        lines.append(f"| {names[k]} | {m['mae_minutes']} | {m['auc']} | {m['precision']} | {m['recall']} |")
    lines += ["", "**Avg error**: average minutes the predicted delay is off by (lower is better).",
              "**AUC**: how well it ranks late flights above on-time ones (0.5 = coin flip, 1.0 = perfect).",
              "**Precision**: of flights it flags as late, the share that really were.",
              "**Recall**: of flights that really were late, the share it caught."]
    for k in ["planning", "day_of"]:
        lines += ["", f"Top signals, {names[k]}:", ""]
        lines += [f"- {f}: {v:.1%}" for f, v in r["models"][k]["top_features"].items()]
    return "\n".join(lines) + "\n"
