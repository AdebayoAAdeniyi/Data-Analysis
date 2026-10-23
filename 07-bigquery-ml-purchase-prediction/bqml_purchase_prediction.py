"""
Predict whether a website session ends in a purchase with BigQuery ML.

BigQuery ML trains machine learning models with SQL, inside the data
warehouse, so the data never has to be exported. This script builds a
logistic regression that predicts whether a Google Merchandise Store session
contains a transaction, using the public data set
``bigquery-public-data.google_analytics_sample`` (Aug 2016 to Aug 2017).

Pipeline
--------
1. Create a BigQuery dataset to hold the models (location US, same as the data).
2. Train two logistic regression models on sessions from 2016-08-01 to
   2017-06-30:
     * ``purchase_model``           : plain logistic regression
     * ``purchase_model_balanced``  : the same, with ``auto_class_weights=TRUE``
       because only a small percentage of sessions contain a purchase
3. Inspect training (``ML.TRAINING_INFO``) and the learned weights (``ML.WEIGHTS``).
4. Evaluate both models on a later, unseen period, 2017-07-01 to 2017-08-01
   (``ML.EVALUATE``, ``ML.ROC_CURVE``). Training on the past and testing on the
   future is how such a model would be used in practice.
5. Predict purchases per country and per visitor (``ML.PREDICT``).

Requirements::

    pip install google-cloud-bigquery db-dtypes pandas matplotlib
    gcloud auth application-default login

Usage::

    python bqml_purchase_prediction.py --project YOUR_GCP_PROJECT [--dataset bqml_tutorial]

Training reads well under 1 GB, but BigQuery ML training is billed separately
from queries; check the current BigQuery ML pricing page before running.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SOURCE = "`bigquery-public-data.google_analytics_sample.ga_sessions_*`"
TRAIN_PERIOD = "_TABLE_SUFFIX BETWEEN '20160801' AND '20170630'"
EVAL_PERIOD = "_TABLE_SUFFIX BETWEEN '20170701' AND '20170801'"

# The label must be binary (0 = no purchase, 1 = purchase). Using the number
# of transactions directly would turn the problem into a multiclass one.
FEATURES_SQL = """
  IF(totals.transactions IS NULL, 0, 1)   AS label,
  IFNULL(device.operatingSystem, '')      AS os,
  device.isMobile                         AS is_mobile,
  IFNULL(geoNetwork.country, '')          AS country,
  IFNULL(totals.pageviews, 0)             AS pageviews"""


def sql_create(dataset: str, model: str, balanced: bool) -> str:
    opts = "model_type = 'logistic_reg', input_label_cols = ['label']"
    if balanced:
        opts += ", auto_class_weights = TRUE"
    # CREATE OR REPLACE: the model is retrained every time the script runs.
    # (CREATE MODEL IF NOT EXISTS would silently keep an old model.)
    return f"""
CREATE OR REPLACE MODEL `{dataset}.{model}`
OPTIONS ({opts}) AS
SELECT {FEATURES_SQL}
FROM {SOURCE}
WHERE {TRAIN_PERIOD}"""


def eval_data() -> str:
    return f"(SELECT {FEATURES_SQL} FROM {SOURCE} WHERE {EVAL_PERIOD})"


def sql_training_info(dataset, model):
    return f"SELECT * FROM ML.TRAINING_INFO(MODEL `{dataset}.{model}`) ORDER BY iteration"


def sql_weights(dataset, model):
    # Numeric features have one weight; categorical features get one weight
    # per category, returned in the nested category_weights column.
    return f"""
SELECT processed_input, weight,
       ARRAY_LENGTH(category_weights) AS n_categories
FROM ML.WEIGHTS(MODEL `{dataset}.{model}`)
ORDER BY ABS(IFNULL(weight, 0)) DESC"""


def sql_evaluate(dataset, model):
    return f"SELECT * FROM ML.EVALUATE(MODEL `{dataset}.{model}`, {eval_data()})"


def sql_roc(dataset, model):
    # Passing the evaluation data explicitly avoids the error
    # "The ROC curve metrics of the model was not calculated during training".
    return f"SELECT * FROM ML.ROC_CURVE(MODEL `{dataset}.{model}`, {eval_data()})"


def sql_predict_by(dataset, model, key):
    extra = ", fullVisitorId" if key == "fullVisitorId" else ""
    return f"""
SELECT {key},
       SUM(label)           AS actual_purchase_sessions,
       SUM(predicted_label) AS predicted_purchase_sessions
FROM ML.PREDICT(MODEL `{dataset}.{model}`,
     (SELECT {FEATURES_SQL}{extra} FROM {SOURCE} WHERE {EVAL_PERIOD}))
GROUP BY {key}
ORDER BY actual_purchase_sessions DESC
LIMIT 10"""


# --------------------------------------------------------------------------- #
def run(client, sql: str, max_gb: float) -> pd.DataFrame:
    from google.cloud import bigquery
    cfg = bigquery.QueryJobConfig(maximum_bytes_billed=int(max_gb * 1024 ** 3))
    return client.query(sql, job_config=cfg).to_dataframe()


def plot_roc(rocs: dict[str, pd.DataFrame], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, roc in rocs.items():
        r = roc.sort_values("false_positive_rate")
        fpr = np.r_[0, r["false_positive_rate"].to_numpy(), 1]
        tpr = np.r_[0, r["recall"].to_numpy(), 1]
        auc = np.trapezoid(tpr, fpr) if hasattr(np, "trapezoid") else np.trapz(tpr, fpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC ≈ {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title("ROC curves on July 2017 sessions (unseen period)")
    ax.legend(loc="lower right")
    fig.savefig(out / "roc_curves.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", required=True)
    parser.add_argument("--dataset", default="bqml_tutorial")
    parser.add_argument("--max-gb", type=float, default=2.0)
    parser.add_argument("--print-sql", action="store_true",
                        help="only print the SQL statements, do not run anything")
    parser.add_argument("--figures", type=Path, default=Path("figures"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()

    models = {"purchase_model": False, "purchase_model_balanced": True}
    if args.print_sql:
        for m, bal in models.items():
            for sql in [sql_create(args.dataset, m, bal), sql_evaluate(args.dataset, m),
                        sql_roc(args.dataset, m)]:
                print(sql, ";\n")
        print(sql_predict_by(args.dataset, "purchase_model", "country"), ";")
        return

    from google.cloud import bigquery
    client = bigquery.Client(project=args.project)
    ds = bigquery.Dataset(f"{args.project}.{args.dataset}")
    ds.location = "US"  # must match the location of the public data set
    client.create_dataset(ds, exists_ok=True)
    args.results.mkdir(parents=True, exist_ok=True)

    evals, rocs = [], {}
    for model, balanced in models.items():
        print(f"\n=== Training {model} (auto_class_weights={balanced}) ===")
        client.query(sql_create(args.dataset, model, balanced)).result()

        info = run(client, sql_training_info(args.dataset, model), args.max_gb)
        print(info.to_string(index=False))
        w = run(client, sql_weights(args.dataset, model), args.max_gb)
        print("\nLearned weights (numeric features; categorical features have "
              "one weight per category):")
        print(w.to_string(index=False))

        ev = run(client, sql_evaluate(args.dataset, model), args.max_gb)
        ev.insert(0, "model", model)
        evals.append(ev)
        rocs[model] = run(client, sql_roc(args.dataset, model), args.max_gb)

    evals = pd.concat(evals, ignore_index=True)
    print("\n=== Evaluation on 2017-07-01 to 2017-08-01 ===")
    print(evals.round(4).to_string(index=False))
    evals.to_csv(args.results / "evaluation.csv", index=False)
    plot_roc(rocs, args.figures)

    for key in ["country", "fullVisitorId"]:
        df = run(client, sql_predict_by(args.dataset, "purchase_model_balanced", key),
                 args.max_gb)
        print(f"\n=== Actual versus predicted purchase sessions by {key} ===")
        print(df.to_string(index=False))
        df.to_csv(args.results / f"predicted_by_{key}.csv", index=False)


if __name__ == "__main__":
    main()
