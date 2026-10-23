"""
Exploratory text analysis and classification of news articles by source type.

Data: ``news_articles.csv`` from the Kaggle data set "Source based Fake News
Classification" (2,096 articles published in Oct to Nov 2016). Columns used:
author (index), published, title, text, language, site_url, main_img_url,
type, label, title_without_stopwords, text_without_stopwords, hasImage.

What this script does
---------------------
1. Cleans the data and measures article length.
2. Describes the classes: ``type`` has 8 values (bias, bs, conspiracy, fake,
   hate, junksci, satire, state) and ``label`` has 2 (Real, Fake). The script
   checks how ``label`` relates to ``type`` and how strongly each website is
   tied to one type, because that decides how the models must be evaluated.
3. Finds the most frequent unigrams, bigrams and trigrams and draws word clouds.
4. Lists websites that publish articles labelled Real, Fake, or both.
5. Trains TF-IDF text classifiers under three evaluation designs:
     A. text + website domain, random split   (the design of the original notebook)
     B. text only, random split
     C. text only, split by website            (are the models able to classify
                                                articles from websites never seen
                                                in training?)
   Design C is the honest test of generalisation. If the scores drop a lot
   from A to C, the model was mostly recognising websites, not writing style.

Run it from this folder with::

    python fake_news_analysis.py --data data/news_articles.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 1
sns.set_theme(style="whitegrid")


# --------------------------------------------------------------------------- #
# 1. Load and clean
# --------------------------------------------------------------------------- #
def load_data(path: Path) -> pd.DataFrame:
    # The first column is the author. Authors repeat, so keep it as a normal
    # column with a clean 0..n-1 index (duplicate index labels break crosstab).
    df = pd.read_csv(path, encoding="latin-1")
    print(f"Raw rows: {len(df)}")
    print("Missing values per column:")
    print(df.isna().sum().to_string())
    df = df.dropna().copy()
    df["word_count"] = df["text_without_stopwords"].str.split().str.len()
    df["domain"] = df["site_url"].str.lower().str.strip()
    print(f"\nRows after dropping missing values: {len(df)}")
    return df


# --------------------------------------------------------------------------- #
# 2. Class structure and the website effect
# --------------------------------------------------------------------------- #
def describe_classes(df: pd.DataFrame) -> dict:
    print("\nArticles per type:")
    counts = df["type"].value_counts()
    print(pd.DataFrame({"articles": counts, "percent": (counts / len(df) * 100).round(2)})
          .to_string())

    print("\nMean word count and share of articles with an image, by type:")
    print(df.groupby("type")[["word_count", "hasImage"]].mean().round(2).to_string())

    ct = pd.crosstab(df["type"], df["label"])
    print("\nType versus label (article counts):")
    print(ct.to_string())
    deterministic = bool((ct.gt(0).sum(axis=1) == 1).all())
    print(f"\nEvery type maps to exactly one label: {deterministic}")

    # How 'pure' is each website? share of its articles in its majority type
    purity = df.groupby("domain")["type"].agg(lambda s: s.value_counts(normalize=True).iloc[0])
    weighted = (purity * df["domain"].value_counts()).sum() / len(df)
    single = (purity == 1).mean()
    print(f"\nWebsites: {df['domain'].nunique()}")
    print(f"Websites whose articles all share one type: {single:.1%}")
    print(f"Article weighted website purity: {weighted:.1%}")
    print("  (if this is close to 100%, knowing the website almost tells you the type,")
    print("   so a model that sees the domain can score highly without reading the text)")
    return {"type_to_label_deterministic": deterministic, "site_purity": weighted}


def websites_by_label(df: pd.DataFrame) -> None:
    real = set(df.loc[df.label == "Real", "domain"])
    fake = set(df.loc[df.label == "Fake", "domain"])
    print(f"\nWebsites with articles labelled Real ({len(real)}): {sorted(real)}")
    print(f"\nWebsites with articles labelled Fake ({len(fake)}): {sorted(fake)}")
    print(f"\nWebsites with both labels ({len(real & fake)}): {sorted(real & fake)}")


# --------------------------------------------------------------------------- #
# 3. N-grams, word clouds and figures
# --------------------------------------------------------------------------- #
def top_ngrams(corpus: pd.Series, n: int = 20, ngram: int = 1) -> pd.DataFrame:
    """One function for unigrams, bigrams and trigrams (the original notebook
    had three copies of the same code)."""
    vec = CountVectorizer(ngram_range=(ngram, ngram))
    bag = vec.fit_transform(corpus)
    freq = np.asarray(bag.sum(axis=0)).ravel()
    order = freq.argsort()[::-1][:n]
    terms = vec.get_feature_names_out()
    return pd.DataFrame({"term": terms[order], "count": freq[order]})


def save(fig, folder: Path, name: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / name, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_eda(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    df["word_count"].plot(kind="hist", bins=150, ax=axes[0], title="Words per article")
    np.log10(df["word_count"]).plot(kind="hist", bins=50, ax=axes[1],
                                    title="log10(words per article)")
    save(fig, out, "01_word_count_hist.png")

    order = df["type"].value_counts().index
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    sns.countplot(data=df, x="type", order=order, ax=axes[0])
    axes[0].set_title("Articles per type")
    sns.violinplot(data=df, x="type", y="word_count", order=order, ax=axes[1], cut=0)
    axes[1].set_title("Words per article by type")
    save(fig, out, "02_type_counts_and_length.png")

    ct = pd.crosstab(df["type"], df["label"], normalize="index")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ct.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title("Share of each label within each type")
    save(fig, out, "03_label_vs_type.png")

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for ax, k, name in zip(axes, [1, 2, 3], ["unigrams", "bigrams", "trigrams"]):
        t = top_ngrams(df["text_without_stopwords"], 20, k)
        ax.barh(t["term"][::-1], t["count"][::-1], color="steelblue")
        ax.set_title(f"Top 20 {name}")
    fig.tight_layout()
    save(fig, out, "04_top_ngrams.png")

    try:
        from wordcloud import STOPWORDS, WordCloud
    except ImportError:
        print("wordcloud is not installed, skipping word clouds")
        return
    types = df["type"].unique().tolist()
    fig, axes = plt.subplots(2, 4, figsize=(24, 9))
    for ax, t in zip(axes.ravel(), types):
        text = " ".join(df.loc[df.type == t, "text_without_stopwords"])
        wc = WordCloud(stopwords=set(STOPWORDS), max_words=300, width=600, height=400,
                       background_color="black", random_state=42).generate(text)
        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(t.upper())
        ax.axis("off")
    for ax in axes.ravel()[len(types):]:
        ax.axis("off")
    save(fig, out, "05_wordclouds_by_type.png")


# --------------------------------------------------------------------------- #
# 4. Classification under three evaluation designs
# --------------------------------------------------------------------------- #
def make_models() -> dict:
    tfidf = lambda: TfidfVectorizer(stop_words="english", sublinear_tf=True,  # noqa: E731
                                    min_df=2, max_df=0.9)
    return {
        "Linear SVM": make_pipeline(tfidf(), LinearSVC(C=1.0, class_weight="balanced")),
        "Logistic regression": make_pipeline(tfidf(), LogisticRegression(
            C=10, max_iter=2000, class_weight="balanced")),
        "Random forest": make_pipeline(tfidf(), RandomForestClassifier(
            n_estimators=300, random_state=0, n_jobs=-1, class_weight="balanced")),
        "AdaBoost (trees depth 10)": make_pipeline(tfidf(), AdaBoostClassifier(
            DecisionTreeClassifier(max_depth=10), n_estimators=5, random_state=1)),
    }


def run_design(df: pd.DataFrame, target: str, design: str) -> list[dict]:
    if design == "A: text + domain, random split":
        X = df["domain"] + " " + df["text_without_stopwords"]
    else:
        X = df["text_without_stopwords"]
    y = df[target]

    if design.endswith("split by website"):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=RANDOM_STATE)
        tr, te = next(gss.split(X, y, groups=df["domain"]))
        X_tr, X_te, y_tr, y_te = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
    else:
        strat = y if y.value_counts().min() >= 2 else None
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=strat,
                                                  random_state=RANDOM_STATE)
    rows = []
    for name, model in make_models().items():
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        rows.append({"target": target, "design": design, "model": name,
                     "n_test": len(y_te), "accuracy": accuracy_score(y_te, pred),
                     "macro_F1": f1_score(y_te, pred, average="macro"),
                     "majority_baseline": (y_te == y_tr.mode()[0]).mean()})
    return rows


def evaluate(df: pd.DataFrame) -> pd.DataFrame:
    designs = ["A: text + domain, random split",
               "B: text only, random split",
               "C: text only, split by website"]
    rows = []
    for target in ["type", "label"]:
        for d in designs:
            rows += run_design(df, target, d)
    res = pd.DataFrame(rows)
    print("\nClassification results (30% test set):")
    print(res.round(3).to_string(index=False))
    print("\nmacro_F1 weighs every class equally, so it exposes poor performance on")
    print("small classes (for example 'fake') that accuracy hides.")
    return res


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("data/news_articles.csv"))
    parser.add_argument("--figures", type=Path, default=Path("figures"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    df = load_data(args.data)
    print("\nWords per article:")
    print(df["word_count"].describe().round(1).to_string())
    describe_classes(df)
    websites_by_label(df)
    for k, name in [(1, "unigrams"), (2, "bigrams"), (3, "trigrams")]:
        print(f"\nTop 10 {name}:")
        print(top_ngrams(df["text_without_stopwords"], 10, k).to_string(index=False))
    if not args.no_plots:
        plot_eda(df, args.figures)

    res = evaluate(df)
    args.results.mkdir(parents=True, exist_ok=True)
    res.round(4).to_csv(args.results / "classification_results.csv", index=False)
    print(f"\nResults saved to {args.results / 'classification_results.csv'}")


if __name__ == "__main__":
    main()
