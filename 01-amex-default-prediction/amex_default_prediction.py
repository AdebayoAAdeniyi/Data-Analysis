"""
American Express Default Prediction (Kaggle competition, 2022).

Task: from up to 13 monthly credit card statements per customer, predict the
probability that the customer defaults (does not pay back the balance).
The features are anonymised and grouped by prefix:

    D_* delinquency   S_* spend   P_* payment   B_* balance   R_* risk

and the competition lists 11 of them as categorical:
B_30, B_38, D_114, D_116, D_117, D_120, D_126, D_63, D_64, D_66, D_68.

Pipeline
--------
1. Load the statement level data (CSV or Parquet), optionally only the first
   N customers so it fits on a laptop. Whole customers are kept: a row limit
   could cut a customer's history in half.
2. Drop columns that are mostly empty (threshold learned on the training data).
3. Aggregate the statements of each customer into one row: mean, std, min,
   max, last value and last minus mean (recent trend) for numeric columns;
   last value, number of distinct values and count for categorical columns;
   and the number of statements. Missing values are NOT replaced by 0 before
   aggregation (0 is a real value for many features and would bias the means).
4. Remove one of every pair of features with |correlation| > threshold.
   Optionally rank the rest with the null importance method.
5. Train with stratified K fold cross validation:
     * LightGBM (gradient boosted trees, handles missing values natively)
     * Logistic regression baseline (median imputation + scaling)
   and report out of fold (OOF) scores: the competition metric, ROC AUC,
   Gini, default rate captured at 4 %, log loss, Brier score, and accuracy,
   precision and recall at the F1 optimal threshold.
6. Plot feature importance, ROC curves, the calibration curve and the
   threshold trade off.
7. Write a submission with one *probability* per customer, averaged over the
   fold models. The competition metric only uses the ranking of the
   predictions, so 0/1 labels throw information away.

Usage::

    python amex_default_prediction.py --train data/train_data.parquet \\
        --labels data/train_labels.csv --test data/test_data.parquet \\
        --max-customers 50000
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score, log_loss,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ID, DATE, TARGET = "customer_ID", "S_2", "target"
CAT_FEATURES = ["B_30", "B_38", "D_114", "D_116", "D_117", "D_120", "D_126",
                "D_63", "D_64", "D_66", "D_68"]
SEED = 42


# --------------------------------------------------------------------------- #
# 1. Competition metric
# --------------------------------------------------------------------------- #
def amex_metric(y_true, y_pred) -> tuple[float, float, float]:
    """Official metric M = 0.5 * (G + D).

    G: normalised weighted Gini coefficient.
    D: share of all defaults captured in the 4 % of (weighted) customers with
       the highest predicted risk.
    Non defaults get weight 20, because the organisers down sampled them by
    a factor of 20. Returns (M, G, D).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    order = np.argsort(-y_pred, kind="mergesort")
    y = y_true[order]
    w = np.where(y == 0, 20.0, 1.0)

    # D: default rate captured at 4 %
    cutoff = int(0.04 * w.sum())
    in_top = np.cumsum(w) <= cutoff
    d = y[in_top].sum() / y.sum()

    # G: normalised weighted Gini
    def weighted_gini(y_sorted, w_sorted):
        random = np.cumsum(w_sorted / w_sorted.sum())
        lorentz = np.cumsum(y_sorted * w_sorted) / (y_sorted * w_sorted).sum()
        return ((lorentz - random) * w_sorted).sum()

    perfect = np.sort(y_true)[::-1]
    g = weighted_gini(y, w) / weighted_gini(perfect, np.where(perfect == 0, 20.0, 1.0))
    return 0.5 * (g + d), g, d


# --------------------------------------------------------------------------- #
# 2. Loading
# --------------------------------------------------------------------------- #
def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_statements(path: Path, max_customers: int | None) -> pd.DataFrame:
    t0 = time.time()
    df = read_table(path)
    if max_customers:
        keep = df[ID].drop_duplicates().iloc[:max_customers]
        df = df[df[ID].isin(keep)]
    df[DATE] = pd.to_datetime(df[DATE])
    df = df.sort_values([ID, DATE]).reset_index(drop=True)
    # float32 halves the memory and keeps enough precision; float16 (used in
    # many public notebooks) keeps only about 3 significant digits.
    num = df.select_dtypes("float64").columns
    df[num] = df[num].astype("float32")
    print(f"Loaded {path.name}: {len(df):,} statements, {df[ID].nunique():,} customers, "
          f"{df.shape[1]} columns in {time.time() - t0:.1f}s")
    return df


# --------------------------------------------------------------------------- #
# 3. Feature engineering
# --------------------------------------------------------------------------- #
def columns_to_keep(train: pd.DataFrame, max_missing: float) -> list[str]:
    miss = train.drop(columns=[ID, DATE]).isna().mean()
    dropped = miss[miss > max_missing].index.tolist()
    print(f"Dropping {len(dropped)} columns with more than {max_missing:.0%} missing values")
    return [c for c in train.columns if c not in dropped and c not in (ID, DATE)]


def aggregate_customers(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """One row per customer."""
    t0 = time.time()
    cats = [c for c in cols if c in CAT_FEATURES]
    nums = [c for c in cols if c not in CAT_FEATURES]
    g = df.groupby(ID, sort=False)

    num_agg = g[nums].agg(["mean", "std", "min", "max", "last"])
    num_agg.columns = [f"{c}_{s}" for c, s in num_agg.columns]
    for c in nums:
        num_agg[f"{c}_last_minus_mean"] = num_agg[f"{c}_last"] - num_agg[f"{c}_mean"]

    parts = [num_agg, g.size().rename("n_statements")]
    if cats:
        cat_df = df[[ID] + cats].copy()
        for c in cats:   # encode as strings so '-1', 'CO', 1.0 are all treated alike
            cat_df[c] = cat_df[c].astype("string").fillna("missing")
        cg = cat_df.groupby(ID, sort=False)
        last = cg[cats].last().add_suffix("_last")
        nunique = cg[cats].nunique().add_suffix("_nunique")
        parts += [last, nunique]

    out = pd.concat(parts, axis=1)
    print(f"Aggregated to {out.shape[0]:,} customers x {out.shape[1]:,} features "
          f"in {time.time() - t0:.1f}s")
    return out


def encode_categoricals(train: pd.DataFrame, test: pd.DataFrame | None):
    """One hot encode the '*_last' categorical columns with the categories seen
    in training, so train and test get identical columns."""
    cat_cols = [c for c in train.columns if c.endswith("_last") and
                c[:-5] in CAT_FEATURES]
    train = pd.get_dummies(train, columns=cat_cols, dtype="uint8")
    if test is not None:
        test = pd.get_dummies(test, columns=cat_cols, dtype="uint8")
        test = test.reindex(columns=train.columns, fill_value=0)
    return train, test


def correlation_filter(X: pd.DataFrame, threshold: float) -> list[str]:
    """Return the columns to drop: for every pair with |r| > threshold the
    later column is dropped."""
    corr = X.astype("float32").corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    to_drop = [c for c in upper.columns if (upper[c] > threshold).any()]
    print(f"Correlation filter |r| > {threshold}: dropping {len(to_drop)} of "
          f"{X.shape[1]} features")
    return to_drop


def null_importance(X: pd.DataFrame, y: pd.Series, n_runs: int = 20) -> pd.DataFrame:
    """Null importance (Olivier Grellier's method): compare each feature's real
    LightGBM gain with its gain when the target is randomly shuffled. Features
    whose real importance is not above the noise level are candidates to drop."""
    import lightgbm as lgb
    params = dict(objective="binary", boosting_type="rf", subsample=0.632,
                  subsample_freq=1, colsample_bytree=0.5, num_leaves=63,
                  max_depth=8, verbose=-1, seed=SEED)

    def gains(target):
        booster = lgb.train(params, lgb.Dataset(X, target), num_boost_round=200)
        return booster.feature_importance(importance_type="gain")

    actual = gains(y)
    rng = np.random.default_rng(SEED)
    null = np.vstack([gains(rng.permutation(y.values)) for _ in range(n_runs)])
    score = np.log(1e-10 + actual / (1 + np.percentile(null, 75, axis=0)))
    return (pd.DataFrame({"feature": X.columns, "actual_gain": actual,
                          "null_gain_p75": np.percentile(null, 75, axis=0),
                          "score": score})
            .sort_values("score", ascending=False))


# --------------------------------------------------------------------------- #
# 4. Models and cross validation
# --------------------------------------------------------------------------- #
def lgbm_model():
    from lightgbm import LGBMClassifier
    return LGBMClassifier(n_estimators=3000, learning_rate=0.03, num_leaves=63,
                          min_child_samples=50, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.5, reg_lambda=2.0, random_state=SEED,
                          n_jobs=-1, verbose=-1)


def logreg_model():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=0.05, max_iter=2000))


def cross_validate_model(name: str, make_model, X: pd.DataFrame, y: pd.Series,
                         X_test: pd.DataFrame | None, n_folds: int):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test)) if X_test is not None else None
    importances = []
    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        t0 = time.time()
        model = make_model()
        if name == "LightGBM":
            from lightgbm import early_stopping, log_evaluation
            cbs = [early_stopping(100, verbose=False), log_evaluation(0)]
            try:   # LightGBM >= 4.6 names the validation data eval_X / eval_y
                model.fit(X.iloc[tr], y.iloc[tr], eval_X=(X.iloc[va],), eval_y=(y.iloc[va],),
                          eval_metric="auc", callbacks=cbs)
            except TypeError:   # older LightGBM versions
                model.fit(X.iloc[tr], y.iloc[tr], eval_set=[(X.iloc[va], y.iloc[va])],
                          eval_metric="auc", callbacks=cbs)
            importances.append(pd.Series(model.booster_.feature_importance("gain"),
                                         index=X.columns))
        else:
            model.fit(X.iloc[tr], y.iloc[tr])
        oof[va] = model.predict_proba(X.iloc[va])[:, 1]       # column 1 = P(default)
        if X_test is not None:
            test_pred += model.predict_proba(X_test)[:, 1] / n_folds
        m, _, _ = amex_metric(y.iloc[va], oof[va])
        print(f"  {name} fold {fold}: amex metric {m:.4f}  AUC "
              f"{roc_auc_score(y.iloc[va], oof[va]):.4f}  ({time.time() - t0:.1f}s)")
    imp = pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False) \
        if importances else None
    return oof, test_pred, imp


def summarise(name: str, y: pd.Series, p: np.ndarray) -> dict:
    m, g, d = amex_metric(y, p)
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    best = int(np.argmax(f1[:-1]))
    t = thr[best]
    lab = (p >= t).astype(int)
    return {"model": name, "amex_metric": m, "gini": g, "default_capture_4pct": d,
            "roc_auc": roc_auc_score(y, p), "log_loss": log_loss(y, np.clip(p, 1e-7, 1 - 1e-7)),
            "brier": brier_score_loss(y, p), "f1_threshold": t,
            "accuracy_at_t": accuracy_score(y, lab), "precision_at_t": precision_score(y, lab),
            "recall_at_t": recall_score(y, lab), "f1_at_t": f1_score(y, lab)}


# --------------------------------------------------------------------------- #
# 5. Figures
# --------------------------------------------------------------------------- #
def make_figures(stmts: pd.DataFrame, y: pd.Series, oofs: dict, imp: pd.Series | None,
                 out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    y.value_counts().sort_index().plot.bar(ax=axes[0], color=["steelblue", "indianred"])
    axes[0].set_title(f"Target (default rate {y.mean():.1%})")
    axes[0].set_xticklabels(["No default", "Default"], rotation=0)
    stmts.groupby(ID).size().value_counts().sort_index().plot.bar(ax=axes[1], color="grey")
    axes[1].set_title("Statements per customer")
    miss = stmts.drop(columns=[ID, DATE]).isna().mean().sort_values(ascending=False).head(25)
    miss.plot.bar(ax=axes[2], color="darkorange")
    axes[2].set_title("Most incomplete columns (share missing)")
    fig.tight_layout()
    fig.savefig(out / "01_data_overview.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    if imp is not None:
        fig, ax = plt.subplots(figsize=(8, 9))
        imp.head(30)[::-1].plot.barh(ax=ax, color="teal")
        ax.set_title("LightGBM: top 30 features by mean gain over folds")
        fig.savefig(out / "02_feature_importance.png", dpi=110, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for name, p in oofs.items():
        fpr, tpr, _ = roc_curve(y, p)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y, p):.3f})")
        frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
        axes[1].plot(mean_pred, frac_pos, marker="o", label=name)
        prec, rec, thr = precision_recall_curve(y, p)
        f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
        axes[2].plot(thr, f1[:-1], label=f"{name} F1")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set(title="ROC curve (out of fold)", xlabel="False positive rate",
                ylabel="True positive rate")
    axes[1].plot([0, 1], [0, 1], "k--", lw=1)
    axes[1].set(title="Calibration (10 quantile bins)", xlabel="Mean predicted probability",
                ylabel="Observed default rate")
    axes[2].set(title="F1 score versus decision threshold", xlabel="Threshold", ylabel="F1")
    for ax in axes:
        ax.legend()
    fig.savefig(out / "03_roc_calibration_threshold.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", type=Path, required=True, help="train_data (.csv or .parquet)")
    parser.add_argument("--labels", type=Path, required=True, help="train_labels (.csv)")
    parser.add_argument("--test", type=Path, default=None, help="test_data (.csv or .parquet)")
    parser.add_argument("--max-customers", type=int, default=None,
                        help="use only the first N customers (for limited memory)")
    parser.add_argument("--max-missing", type=float, default=0.70)
    parser.add_argument("--corr-threshold", type=float, default=0.95)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--null-importance", type=int, default=0, metavar="RUNS",
                        help="if > 0, rank features with this many null importance runs")
    parser.add_argument("--figures", type=Path, default=Path("figures"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)

    stmts = load_statements(args.train, args.max_customers)
    labels = read_table(args.labels).set_index(ID)[TARGET]
    keep = columns_to_keep(stmts, args.max_missing)

    X = aggregate_customers(stmts, keep)
    y = labels.reindex(X.index)
    if y.isna().any():
        raise ValueError(f"{int(y.isna().sum())} customers have no label")
    y = y.astype(int)
    print(f"Customers: {len(y):,}   default rate: {y.mean():.2%}")

    X_test = None
    if args.test is not None:
        test_stmts = load_statements(args.test, None)
        X_test = aggregate_customers(test_stmts, keep)
        del test_stmts

    X, X_test = encode_categoricals(X, X_test)
    drop = correlation_filter(X, args.corr_threshold)
    X = X.drop(columns=drop)
    if X_test is not None:
        X_test = X_test[X.columns]

    if args.null_importance > 0:
        ni = null_importance(X, y, args.null_importance)
        ni.to_csv(args.results / "null_importance.csv", index=False)
        print("Top 15 features by null importance score:")
        print(ni.head(15).round(3).to_string(index=False))

    print(f"\nFinal feature matrix: {X.shape[0]:,} x {X.shape[1]:,}")
    oofs, preds, imp = {}, {}, None
    for name, maker in [("LightGBM", lgbm_model), ("Logistic regression", logreg_model)]:
        print(f"\n{name}, {args.folds} fold stratified CV")
        oof, tp, fold_imp = cross_validate_model(name, maker, X, y, X_test, args.folds)
        oofs[name], preds[name] = oof, tp
        if fold_imp is not None:
            imp = fold_imp

    summary = pd.DataFrame([summarise(n, y, p) for n, p in oofs.items()])
    print("\nOut of fold results:")
    print(summary.round(4).to_string(index=False))
    summary.round(5).to_csv(args.results / "oof_summary.csv", index=False)
    if imp is not None:
        imp.rename("mean_gain").to_csv(args.results / "lightgbm_feature_importance.csv")
    make_figures(stmts, y, oofs, imp, args.figures)

    if X_test is not None:
        best = summary.sort_values("amex_metric", ascending=False)["model"].iloc[0]
        sub = pd.DataFrame({ID: X_test.index, "prediction": preds[best]})
        sub.to_csv(args.results / "submission.csv", index=False)
        print(f"\nSubmission from {best}: {len(sub):,} customers -> "
              f"{args.results / 'submission.csv'} (probabilities, not 0/1 labels)")


if __name__ == "__main__":
    main()
