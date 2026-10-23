"""
Medical insurance cost analysis and prediction.

What this script does
---------------------
1. Loads the insurance data set (age, sex, bmi, children, smoker, region, charges).
2. Runs an exploratory data analysis (EDA) and saves every figure to ``figures/``.
3. Tests the most important hypothesis statistically (do smokers cost more?).
4. Compares several regression models on the same train/test split and with
   5 fold cross validation, and saves the metrics to ``results/``.

Run it from this folder with::

    python insurance_cost_analysis.py --data data/insurance.csv

The data file is the public "Medical Cost Personal Datasets" file
(columns: age, sex, bmi, children, smoker, region, charges). Some copies name
the target column ``expenses`` instead of ``charges``; both are accepted.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write figures to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

RANDOM_STATE = 0
TARGET = "charges"
sns.set_theme(style="whitegrid", font_scale=1.1)


# --------------------------------------------------------------------------- #
# 1. Data loading and cleaning
# --------------------------------------------------------------------------- #
def load_data(path: Path) -> pd.DataFrame:
    """Read the csv file and standardise the column names."""
    df = pd.read_csv(path)
    df = df.rename(columns={"expenses": TARGET})
    expected = {"age", "sex", "bmi", "children", "smoker", "region", TARGET}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Data file is missing columns: {sorted(missing)}")
    return df


def add_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """Binary encodings plus one domain driven feature.

    ``is_smoker`` and ``is_male`` are explicit 0/1 columns. This avoids the
    ambiguity of LabelEncoder, which codes categories alphabetically
    (female = 0, male = 1; no = 0, yes = 1).

    ``obese_smoker`` marks smokers with BMI >= 30 (the WHO obesity cut off).
    The EDA below shows that charges for smokers jump at this threshold, so a
    linear model needs this interaction to describe the data well.
    """
    out = df.copy()
    out["is_smoker"] = (out["smoker"] == "yes").astype(int)
    out["is_male"] = (out["sex"] == "male").astype(int)
    out["obese"] = (out["bmi"] >= 30).astype(int)
    out["obese_smoker"] = out["obese"] * out["is_smoker"]
    return out


# --------------------------------------------------------------------------- #
# 2. Exploratory data analysis
# --------------------------------------------------------------------------- #
def save(fig: plt.Figure, folder: Path, name: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / name, dpi=120, bbox_inches="tight")
    plt.close(fig)


def describe_data(df: pd.DataFrame) -> None:
    print("=" * 70)
    print(f"Rows: {len(df)}   Columns: {df.shape[1]}")
    print("\nMissing values per column:")
    print(df.isna().sum().to_string())
    print("\nSummary statistics:")
    print(df.describe().round(2).T.to_string())

    num = add_model_features(df)[["age", "bmi", "children", "is_smoker",
                                  "is_male", TARGET]]
    print("\nPearson correlation of each feature with charges:")
    print(num.corr()[TARGET].drop(TARGET).sort_values().round(3).to_string())


def smoker_hypothesis_test(df: pd.DataFrame) -> dict:
    """Do smokers have higher charges? Charges are right skewed, so a
    non parametric Mann Whitney U test is more appropriate than a t test."""
    smokers = df.loc[df.smoker == "yes", TARGET]
    non_smokers = df.loc[df.smoker == "no", TARGET]
    u_stat, p_value = stats.mannwhitneyu(smokers, non_smokers, alternative="greater")
    result = {
        "n_smokers": len(smokers),
        "n_non_smokers": len(non_smokers),
        "median_smokers": smokers.median(),
        "median_non_smokers": non_smokers.median(),
        "median_ratio": smokers.median() / non_smokers.median(),
        "mann_whitney_U": u_stat,
        "p_value": p_value,
    }
    print("\nHypothesis test: smokers have higher charges than non smokers")
    for k, v in result.items():
        print(f"  {k:<20} {v:,.4g}" if isinstance(v, float) else f"  {k:<20} {v}")

    # Median charges by smoking status and obesity (the key interaction)
    table = (df.assign(obese=np.where(df.bmi >= 30, "BMI >= 30", "BMI < 30"))
               .pivot_table(index="smoker", columns="obese", values=TARGET,
                            aggfunc="median").round(0))
    print("\nMedian charges by smoking status and obesity:")
    print(table.to_string())
    return result


def plot_eda(df: pd.DataFrame, out: Path) -> None:
    palette = {"no": "#4C72B0", "yes": "#DD8452"}

    # Distribution of the target: strongly right skewed
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    sns.histplot(df[TARGET], bins=40, kde=True, ax=axes[0], color="teal")
    axes[0].set_title("Distribution of charges")
    sns.histplot(data=df, x=TARGET, hue="smoker", bins=40, kde=True,
                 palette=palette, ax=axes[1])
    axes[1].set_title("Charges for smokers and non smokers")
    save(fig, out, "01_charges_distribution.png")

    # Sex x smoker violin plot
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.violinplot(data=df, x="sex", y=TARGET, hue="smoker", split=True,
                   palette=palette, ax=ax, inner="quartile")
    ax.set_title("Charges by sex and smoking status")
    save(fig, out, "02_sex_smoker_violin.png")

    # Counts of smokers by sex
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.countplot(data=df, x="smoker", hue="sex", ax=ax)
    ax.set_title("Number of smokers and non smokers by sex")
    save(fig, out, "03_smoker_counts.png")

    # Age vs charges: three visible bands
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(data=df, x="age", y=TARGET, hue="smoker", palette=palette,
                    alpha=0.7, ax=ax)
    ax.set_title("Charges increase with age; smoking shifts the whole curve")
    save(fig, out, "04_age_vs_charges.png")

    # BMI vs charges with regression lines for each group
    g = sns.lmplot(data=df, x="bmi", y=TARGET, hue="smoker", palette=palette,
                   height=6, aspect=1.4, scatter_kws={"alpha": 0.6})
    g.ax.axvline(30, ls="--", color="grey")
    g.ax.text(30.3, df[TARGET].max() * 0.95, "BMI = 30", color="grey")
    g.ax.set_title("BMI only matters for smokers")
    g.figure.savefig(out / "05_bmi_vs_charges.png", dpi=120, bbox_inches="tight")
    plt.close(g.figure)

    # Charges for BMI above and below 30
    fig, ax = plt.subplots(figsize=(9, 4.5))
    tmp = df.assign(bmi_group=np.where(df.bmi >= 30, "BMI >= 30", "BMI < 30"))
    sns.boxplot(data=tmp, x=TARGET, y="bmi_group", hue="smoker",
                palette=palette, ax=ax)
    ax.set_title("Charges by obesity and smoking status")
    save(fig, out, "06_bmi_group_boxplot.png")

    # Age distribution and children counts
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    sns.histplot(df["age"], bins=47, ax=axes[0], color="goldenrod")
    axes[0].set_title("Distribution of age")
    sns.countplot(data=df, x="children", ax=axes[1], color="slateblue")
    axes[1].set_title("Number of children")
    save(fig, out, "07_age_children.png")

    # Correlation heatmap of the numeric encodings
    num = add_model_features(df)[["age", "bmi", "children", "is_smoker",
                                  "is_male", TARGET]]
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="vlag", center=0,
                square=True, ax=ax)
    ax.set_title("Correlation matrix")
    save(fig, out, "08_correlation_heatmap.png")


# --------------------------------------------------------------------------- #
# 3. Modelling
# --------------------------------------------------------------------------- #
BASE_NUMERIC = ["age", "bmi", "children", "is_smoker", "is_male"]


def make_models() -> dict[str, tuple[Pipeline, list[str]]]:
    """Return {name: (pipeline, feature list)}.

    Every preprocessing step (scaling, PCA, one hot encoding) lives inside the
    pipeline, so it is fitted on the training data only. That is how the
    original notebook's PCA mistake (fitting a second PCA on the test set) is
    avoided.
    """
    def region_ohe() -> ColumnTransformer:
        # A new transformer per model, so pipelines never share fitted state
        return ColumnTransformer(
            [("region", OneHotEncoder(drop="first"), ["region"])],
            remainder="passthrough",
        )

    feats = BASE_NUMERIC + ["region"]
    feats_inter = BASE_NUMERIC + ["obese", "obese_smoker", "region"]

    return {
        "Linear regression": (
            Pipeline([("prep", region_ohe()), ("model", LinearRegression())]), feats),
        "PCA + linear regression": (
            Pipeline([("prep", region_ohe()), ("scale", StandardScaler()),
                      ("pca", PCA()), ("model", LinearRegression())]), feats),
        "Polynomial (degree 2) regression": (
            Pipeline([("prep", region_ohe()),
                      ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                      ("model", LinearRegression())]), feats),
        "Linear + obesity x smoker interaction": (
            Pipeline([("prep", region_ohe()), ("model", LinearRegression())]), feats_inter),
        "Random forest": (
            Pipeline([("prep", region_ohe()),
                      ("model", RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                                                      random_state=1, n_jobs=-1))]), feats),
        "Gradient boosting": (
            Pipeline([("prep", region_ohe()),
                      ("model", GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                                          learning_rate=0.05,
                                                          random_state=1))]), feats),
    }


def evaluate_models(df: pd.DataFrame, fig_dir: Path | None) -> pd.DataFrame:
    data = add_model_features(df)
    y = data[TARGET]
    train_idx, test_idx = train_test_split(data.index, random_state=RANDOM_STATE)
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    rows, fitted = [], {}
    for name, (pipe, feats) in make_models().items():
        X = data[feats]
        pipe.fit(X.loc[train_idx], y.loc[train_idx])
        pred_tr = pipe.predict(X.loc[train_idx])
        pred_te = pipe.predict(X.loc[test_idx])
        cv_r2 = cross_val_score(pipe, X, y, cv=cv, scoring="r2")
        rows.append({
            "model": name,
            "train_R2": r2_score(y.loc[train_idx], pred_tr),
            "test_R2": r2_score(y.loc[test_idx], pred_te),
            "test_RMSE": np.sqrt(mean_squared_error(y.loc[test_idx], pred_te)),
            "test_MAE": mean_absolute_error(y.loc[test_idx], pred_te),
            "cv5_R2_mean": cv_r2.mean(),
            "cv5_R2_std": cv_r2.std(),
        })
        fitted[name] = (pipe, feats, pred_tr, pred_te)

    results = pd.DataFrame(rows).set_index("model")
    print("\nModel comparison (same 75/25 split, random_state=0, plus 5 fold CV):")
    print(results.round(3).to_string())

    if fig_dir is not None:
        plot_residuals(fitted, y.loc[train_idx], y.loc[test_idx], results, fig_dir)

    # Permutation importance of the best model by cross validated R2
    best = results["cv5_R2_mean"].idxmax()
    pipe, feats, _, _ = fitted[best]
    imp = permutation_importance(pipe, data.loc[test_idx, feats], y.loc[test_idx],
                                 n_repeats=20, random_state=RANDOM_STATE)
    imp_df = (pd.DataFrame({"feature": feats, "importance_mean": imp.importances_mean,
                            "importance_std": imp.importances_std})
                .sort_values("importance_mean", ascending=False))
    print(f"\nPermutation importance on the test set ({best}):")
    print(imp_df.round(4).to_string(index=False))
    return results, imp_df, best


def plot_residuals(fitted, y_train, y_test, results, out: Path) -> None:
    n = len(fitted)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(6 * ((n + 1) // 2), 11),
                             sharey=True)
    for ax, (name, (_, _, pred_tr, pred_te)) in zip(axes.ravel(), fitted.items()):
        ax.scatter(pred_tr, pred_tr - y_train, s=18, alpha=0.4, c="black",
                   label=f"train R² = {results.loc[name, 'train_R2']:.3f}")
        ax.scatter(pred_te, pred_te - y_test, s=18, alpha=0.7, c="c",
                   label=f"test R² = {results.loc[name, 'test_R2']:.3f}")
        ax.axhline(0, color="red", lw=2)
        ax.set_title(name)
        ax.set_xlabel("Predicted charges")
        ax.legend(loc="upper left")
    fig.subplots_adjust(hspace=0.35)
    axes.ravel()[0].set_ylabel("Residual (predicted − actual)")
    axes.ravel()[(n + 1) // 2].set_ylabel("Residual (predicted − actual)")
    save(fig, out, "09_residuals_by_model.png")


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("data/insurance.csv"))
    parser.add_argument("--figures", type=Path, default=Path("figures"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    df = load_data(args.data)
    describe_data(df)
    smoker_hypothesis_test(df)
    if not args.no_plots:
        plot_eda(df, args.figures)

    results, importance, best = evaluate_models(df, None if args.no_plots else args.figures)
    args.results.mkdir(parents=True, exist_ok=True)
    results.round(4).to_csv(args.results / "model_comparison.csv")
    importance.round(4).to_csv(args.results / "permutation_importance.csv", index=False)
    print(f"\nBest model by 5 fold CV R²: {best}")
    print(f"Figures saved to {args.figures}/, metrics saved to {args.results}/")


if __name__ == "__main__":
    main()
