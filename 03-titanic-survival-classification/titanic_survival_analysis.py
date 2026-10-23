"""
Titanic survival analysis and classification.

What this script does
---------------------
1. Audits data quality (missing Age, Cabin and Embarked values).
2. Engineers features (Title from the name, family size, travelling alone,
   minor flag, known cabin) with all imputation statistics learned from the
   training data only.
3. Explores survival by sex, class, port, age, fare and family, and saves the
   figures to ``figures/``.
4. Compares ten classifiers with stratified 10 fold cross validation.
5. Builds an interpretable logistic regression: recursive feature elimination
   with cross validation (RFECV), a grid search for the regularisation strength
   C, a hold out evaluation (accuracy, log loss, ROC AUC, confusion matrix), and
   odds ratios for every selected feature.
6. Draws a depth 3 decision tree that can be read like a flow chart.
7. Optionally writes a Kaggle submission when ``--test`` is given.

Run it from this folder with::

    python titanic_survival_analysis.py --train data/train.csv [--test data/test.csv]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, log_loss, roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold, cross_validate,
                                     train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree

RANDOM_STATE = 2
sns.set_theme(style="whitegrid", font_scale=1.1)

RARE_TITLES = ["Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major", "Rev",
               "Sir", "Jonkheer", "Dona"]


# --------------------------------------------------------------------------- #
# 1. Data quality
# --------------------------------------------------------------------------- #
def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    rep = pd.DataFrame({"missing": df.isna().sum(),
                        "percent": (df.isna().mean() * 100).round(2)})
    return rep[rep.missing > 0].sort_values("missing", ascending=False)


def get_title(name: str) -> str:
    """'Braund, Mr. Owen Harris' -> 'Mr'."""
    match = re.search(r" ([A-Za-z]+)\.", name)
    return match.group(1) if match else ""


# --------------------------------------------------------------------------- #
# 2. Feature engineering (fit on train, apply to any data)
# --------------------------------------------------------------------------- #
class TitanicFeatures:
    """Small fit/transform helper so that the test set is processed with
    statistics learned on the training set (no information leaks from test)."""

    def fit(self, df: pd.DataFrame) -> "TitanicFeatures":
        d = self._base(df)
        # Age is right skewed, so the median is a better fill value than the
        # mean. The median *per title* is better still: "Master" is a boy,
        # "Miss" is often a young woman, "Mrs" is older.
        self.age_by_title_ = d.groupby("Title")["Age"].median()
        self.age_median_ = d["Age"].median()
        self.fare_median_ = d["Fare"].median()
        self.embarked_mode_ = d["Embarked"].mode()[0]
        return self

    @staticmethod
    def _base(df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        d["Title"] = d["Name"].apply(get_title)
        d["Title"] = (d["Title"].replace(RARE_TITLES, "Rare")
                                .replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}))
        return d

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        d = self._base(df)
        d["AgeWasMissing"] = d["Age"].isna().astype(int)
        d["Age"] = d["Age"].fillna(d["Title"].map(self.age_by_title_)).fillna(self.age_median_)
        d["Fare"] = d["Fare"].fillna(self.fare_median_)
        d["Embarked"] = d["Embarked"].fillna(self.embarked_mode_)
        d["HasCabin"] = d["Cabin"].notna().astype(int)   # 77% missing, but missingness is informative
        d["FamilySize"] = d["SibSp"] + d["Parch"] + 1
        d["TravelAlone"] = (d["FamilySize"] == 1).astype(int)
        d["IsMinor"] = (d["Age"] <= 16).astype(int)
        d["Sex_male"] = (d["Sex"] == "male").astype(int)
        d["LogFare"] = np.log1p(d["Fare"])
        d = pd.get_dummies(d, columns=["Pclass", "Embarked", "Title"], dtype=int)
        # Make sure train and test always have the same dummy columns
        for col in MODEL_FEATURES:
            if col not in d:
                d[col] = 0
        return d

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


# One dummy per category is left out as the reference level (Pclass_3,
# Embarked_S, Title_Mr). Keeping every dummy makes the columns perfectly
# collinear, and then the logistic coefficients cannot be interpreted.
MODEL_FEATURES = ["Age", "LogFare", "FamilySize", "TravelAlone", "IsMinor", "HasCabin",
                  "Sex_male", "Pclass_1", "Pclass_2", "Embarked_C", "Embarked_Q",
                  "Title_Master", "Title_Miss", "Title_Mrs", "Title_Rare"]


# --------------------------------------------------------------------------- #
# 3. Exploratory analysis
# --------------------------------------------------------------------------- #
def save(fig, folder: Path, name: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / name, dpi=120, bbox_inches="tight")
    plt.close(fig)


def survival_tables(raw: pd.DataFrame, feat: pd.DataFrame) -> None:
    print("\nSurvival rate by group (training data):")
    d = feat.assign(Pclass=raw["Pclass"], Sex=raw["Sex"], Embarked=raw["Embarked"],
                    Title=TitanicFeatures._base(raw)["Title"],
                    FareQuartile=pd.qcut(raw["Fare"], 4),
                    AgeBand=pd.cut(feat["Age"], [0, 16, 32, 48, 64, 81]))
    for col in ["Sex", "Pclass", "Embarked", "Title", "TravelAlone", "FamilySize",
                "FareQuartile", "AgeBand", "HasCabin"]:
        t = d.groupby(col, observed=True)["Survived"].agg(["mean", "count"]).round(3)
        t.columns = ["survival_rate", "passengers"]
        print(f"\n{t.to_string()}")


def plot_eda(raw: pd.DataFrame, feat: pd.DataFrame, out: Path) -> None:
    # Raw versus imputed age
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.histplot(raw["Age"].dropna(), bins=30, stat="density", color="teal",
                 alpha=0.5, label="Raw age (714 known values)", ax=ax)
    sns.histplot(feat["Age"], bins=30, stat="density", color="brown", alpha=0.4,
                 label="After median-by-title imputation", ax=ax)
    ax.legend()
    ax.set_title("Age before and after imputation")
    save(fig, out, "01_age_imputation.png")

    # Age and fare densities for survivors and non survivors
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, col, xlim in [(axes[0], "Age", (-5, 85)), (axes[1], "Fare", (-10, 200))]:
        sns.kdeplot(feat.loc[feat.Survived == 1, col], fill=True, color="darkturquoise",
                    label="Survived", ax=ax)
        sns.kdeplot(feat.loc[feat.Survived == 0, col], fill=True, color="lightcoral",
                    label="Died", ax=ax)
        ax.set_xlim(*xlim)
        ax.legend()
        ax.set_title(f"{col}: survivors vs non survivors")
    save(fig, out, "02_age_fare_density.png")

    # Survival rate by categorical variables
    d = raw.assign(Title=TitanicFeatures._base(raw)["Title"],
                   TravelAlone=feat["TravelAlone"], FamilySize=feat["FamilySize"])
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, col in zip(axes.ravel(), ["Sex", "Pclass", "Embarked", "Title",
                                      "TravelAlone", "FamilySize"]):
        sns.barplot(data=d, x=col, y="Survived", ax=ax, color="mediumturquoise",
                    errorbar=("ci", 95))
        ax.set_ylim(0, 1)
        ax.set_title(f"Survival rate by {col}")
    fig.tight_layout()
    save(fig, out, "03_survival_by_group.png")

    # Survival by age (children were prioritised)
    fig, ax = plt.subplots(figsize=(14, 4.5))
    by_age = feat.assign(AgeInt=feat["Age"].round()).groupby("AgeInt")["Survived"].mean()
    ax.bar(by_age.index, by_age.values, color="lightseagreen")
    ax.axvline(16.5, color="grey", ls="--")
    ax.text(17, 0.95, "age 16", color="grey")
    ax.set_xlabel("Age (rounded)")
    ax.set_ylabel("Survival rate")
    ax.set_title("Survival rate by age: a large share of children survived")
    save(fig, out, "04_survival_by_age.png")

    # Correlation heatmap
    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(feat[MODEL_FEATURES + ["Survived"]].corr(), annot=True, fmt=".2f",
                cmap="RdYlGn", center=0, ax=ax, annot_kws={"size": 7})
    ax.set_title("Correlation between engineered features and survival")
    save(fig, out, "05_correlation_heatmap.png")


# --------------------------------------------------------------------------- #
# 4. Model comparison
# --------------------------------------------------------------------------- #
def compare_classifiers(X: pd.DataFrame, y: pd.Series, out: Path | None) -> pd.DataFrame:
    """Every model is wrapped with a scaler where scale matters, and evaluated
    with the same stratified 10 fold split."""
    scaled = lambda m: make_pipeline(StandardScaler(), m)  # noqa: E731
    models = {
        "KNeighbors (k=3)": scaled(KNeighborsClassifier(3)),
        "SVC (RBF)": scaled(SVC(random_state=RANDOM_STATE)),  # ROC AUC uses decision_function
        "Decision tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                                random_state=RANDOM_STATE),
        "AdaBoost": AdaBoostClassifier(random_state=RANDOM_STATE),
        "Gradient boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "Gaussian naive Bayes": GaussianNB(),
        "Linear discriminant": LinearDiscriminantAnalysis(),
        "Quadratic discriminant": QuadraticDiscriminantAnalysis(reg_param=0.1),
        "Logistic regression": scaled(LogisticRegression(max_iter=1000)),
    }
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    rows = []
    for name, model in models.items():
        res = cross_validate(model, X, y, cv=cv, scoring=["accuracy", "roc_auc"])
        rows.append({"classifier": name,
                     "accuracy_mean": res["test_accuracy"].mean(),
                     "accuracy_std": res["test_accuracy"].std(),
                     "roc_auc_mean": res["test_roc_auc"].mean()})
    table = pd.DataFrame(rows).sort_values("accuracy_mean", ascending=False)
    print("\nClassifier comparison, stratified 10 fold CV on the training data:")
    print(table.round(3).to_string(index=False))

    if out is not None:
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(table.classifier, table.accuracy_mean, xerr=table.accuracy_std,
                color="steelblue", capsize=3)
        ax.invert_yaxis()
        ax.set_xlim(0.6, 0.9)
        ax.set_xlabel("Mean accuracy (10 fold CV, error bar = 1 std)")
        ax.set_title("Classifier comparison for predicting survival")
        save(fig, out, "06_classifier_comparison.png")
    return table


# --------------------------------------------------------------------------- #
# 5. Logistic regression: selection, tuning, evaluation, interpretation
# --------------------------------------------------------------------------- #
def logistic_regression_study(X: pd.DataFrame, y: pd.Series, out: Path | None):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y,
                                              random_state=RANDOM_STATE)
    scaler = StandardScaler().fit(X_tr)
    Xs_tr = pd.DataFrame(scaler.transform(X_tr), columns=X.columns, index=X_tr.index)
    Xs_te = pd.DataFrame(scaler.transform(X_te), columns=X.columns, index=X_te.index)
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

    # (a) RFECV on the training part only
    rfecv = RFECV(LogisticRegression(max_iter=1000), step=1, cv=cv,
                  scoring="accuracy", min_features_to_select=1).fit(Xs_tr, y_tr)
    selected = list(X.columns[rfecv.support_])
    print(f"\nRFECV optimal number of features: {rfecv.n_features_}")
    print(f"Selected features: {selected}")

    if out is not None:
        scores = rfecv.cv_results_["mean_test_score"]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(range(1, len(scores) + 1), scores, marker="o")
        ax.axvline(rfecv.n_features_, color="grey", ls="--")
        ax.set_xlabel("Number of features selected")
        ax.set_ylabel("Mean CV accuracy")
        ax.set_title("Recursive feature elimination with cross validation")
        save(fig, out, "07_rfecv.png")

    # (b) Tune C with several scorers, refit on accuracy
    grid = GridSearchCV(LogisticRegression(max_iter=1000),
                        {"C": np.logspace(-3, 2, 21)}, cv=cv, refit="accuracy",
                        scoring={"accuracy": "accuracy", "roc_auc": "roc_auc",
                                 "neg_log_loss": "neg_log_loss"},
                        return_train_score=True).fit(Xs_tr[selected], y_tr)
    best_c = grid.best_params_["C"]
    cvr = grid.cv_results_
    i = grid.best_index_
    print(f"\nBest C = {best_c:.4g}")
    print(f"CV accuracy {cvr['mean_test_accuracy'][i]:.3f} ± {cvr['std_test_accuracy'][i]:.3f}, "
          f"CV ROC AUC {cvr['mean_test_roc_auc'][i]:.3f} ± {cvr['std_test_roc_auc'][i]:.3f}, "
          f"CV log loss {-cvr['mean_test_neg_log_loss'][i]:.3f}")

    if out is not None:
        fig, ax = plt.subplots(figsize=(9, 6))
        for key, color in [("accuracy", "g"), ("roc_auc", "k")]:
            ax.semilogx(cvr["param_C"].data.astype(float), cvr[f"mean_test_{key}"],
                        color=color, label=f"{key} (CV)")
            ax.semilogx(cvr["param_C"].data.astype(float), cvr[f"mean_train_{key}"],
                        color=color, ls="--", alpha=0.6, label=f"{key} (train)")
        ax.axvline(best_c, color="grey", ls=":")
        ax.set_xlabel("C (inverse of regularisation strength)")
        ax.set_ylabel("Score")
        ax.legend()
        ax.set_title("Grid search for the regularisation strength")
        save(fig, out, "08_grid_search_C.png")

    # (c) Hold out evaluation
    model = grid.best_estimator_
    proba = model.predict_proba(Xs_te[selected])[:, 1]
    pred = (proba >= 0.5).astype(int)
    print("\nHold out test set (20%, stratified):")
    print(f"  accuracy {accuracy_score(y_te, pred):.3f}   log loss {log_loss(y_te, proba):.3f}"
          f"   ROC AUC {roc_auc_score(y_te, proba):.3f}")
    print(classification_report(y_te, pred, target_names=["Died", "Survived"], digits=3))

    fpr, tpr, thr = roc_curve(y_te, proba)
    idx = int(np.argmax(tpr >= 0.95))
    print(f"A threshold of {thr[idx]:.3f} gives sensitivity {tpr[idx]:.3f} and "
          f"specificity {1 - fpr[idx]:.3f} (false positive rate {fpr[idx]:.1%}).")

    if out is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        axes[0].plot(fpr, tpr, color="coral",
                     label=f"ROC curve (AUC = {roc_auc_score(y_te, proba):.3f})")
        axes[0].plot([0, 1], [0, 1], "k--")
        axes[0].plot([0, fpr[idx], fpr[idx]], [tpr[idx], tpr[idx], 0], "b:")
        axes[0].set_xlabel("False positive rate (1 − specificity)")
        axes[0].set_ylabel("True positive rate (sensitivity)")
        axes[0].legend(loc="lower right")
        axes[0].set_title("ROC curve, hold out set")
        ConfusionMatrixDisplay(confusion_matrix(y_te, pred),
                               display_labels=["Died", "Survived"]).plot(ax=axes[1],
                                                                          colorbar=False)
        axes[1].set_title("Confusion matrix at threshold 0.5")
        save(fig, out, "09_roc_confusion.png")

    # (d) Interpretation: odds ratio per one standard deviation change
    # Odds ratio per one unit of the original feature: divide the standardised
    # coefficient by the feature's standard deviation. For a 0/1 feature one
    # unit is "absent -> present" (for Sex_male: female -> male).
    sd = pd.Series(scaler.scale_, index=X.columns)[selected].to_numpy()
    coefs = pd.DataFrame({"feature": selected, "coefficient": model.coef_[0]})
    coefs["odds_ratio_per_1sd"] = np.exp(coefs["coefficient"])
    coefs["odds_ratio_per_unit"] = np.exp(coefs["coefficient"] / sd)
    coefs = coefs.sort_values("coefficient")
    print("\nLogistic regression coefficients (standardised inputs):")
    print(coefs.round(3).to_string(index=False))
    return {"selected": selected, "C": best_c, "scaler": scaler, "model": model,
            "coefficients": coefs}


def draw_decision_tree(X: pd.DataFrame, y: pd.Series, out: Path) -> None:
    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=20,
                                  random_state=RANDOM_STATE).fit(X, y)
    fig, ax = plt.subplots(figsize=(22, 9))
    plot_tree(tree, feature_names=list(X.columns), class_names=["Died", "Survived"],
              filled=True, rounded=True, impurity=True, proportion=False, ax=ax,
              fontsize=10)
    ax.set_title("Depth 3 decision tree (gini impurity, min 20 passengers per leaf)")
    save(fig, out, "10_decision_tree.png")


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--test", type=Path, default=None,
                        help="optional Kaggle test.csv; writes results/submission.csv")
    parser.add_argument("--figures", type=Path, default=Path("figures"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    fig_dir = None if args.no_plots else args.figures
    args.results.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.train)
    print(f"Training rows: {len(raw)}   survival rate: {raw.Survived.mean():.3f}")
    print("\nMissing values:")
    print(missing_report(raw).to_string())
    print(f"\nAge mean {raw.Age.mean():.2f}, median {raw.Age.median():.2f} "
          f"(right skew, so the median is the safer fill value)")

    fe = TitanicFeatures()
    feat = fe.fit_transform(raw)
    print("\nMedian age used for imputation, by title:")
    print(fe.age_by_title_.round(1).to_string())
    survival_tables(raw, feat)
    if fig_dir is not None:
        plot_eda(raw, feat, fig_dir)

    X, y = feat[MODEL_FEATURES], feat["Survived"]
    table = compare_classifiers(X, y, fig_dir)
    table.round(4).to_csv(args.results / "classifier_comparison.csv", index=False)

    lr = logistic_regression_study(X, y, fig_dir)
    lr["coefficients"].round(4).to_csv(args.results / "logistic_coefficients.csv", index=False)
    if fig_dir is not None:
        draw_decision_tree(X, y, fig_dir)

    if args.test is not None:
        test_raw = pd.read_csv(args.test)
        Xt = fe.transform(test_raw)[MODEL_FEATURES]
        # Refit the tuned logistic model on all training rows before predicting
        scaler = StandardScaler().fit(X)
        final = LogisticRegression(C=lr["C"], max_iter=1000).fit(
            pd.DataFrame(scaler.transform(X), columns=X.columns)[lr["selected"]], y)
        Xt_s = pd.DataFrame(scaler.transform(Xt), columns=X.columns)[lr["selected"]]
        sub = pd.DataFrame({"PassengerId": test_raw["PassengerId"],
                            "Survived": final.predict(Xt_s)})
        sub.to_csv(args.results / "submission.csv", index=False)
        print(f"\nWrote {len(sub)} predictions to {args.results / 'submission.csv'}")


if __name__ == "__main__":
    main()
