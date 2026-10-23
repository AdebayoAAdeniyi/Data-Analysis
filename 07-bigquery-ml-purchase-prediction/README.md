# Purchase Prediction with BigQuery ML

**Question:** Can we predict, from a few session attributes, whether a visit to the Google Merchandise Store will end in a purchase?

**Why it matters:** Purchase propensity scores drive remarketing lists, ad bidding and website personalisation. BigQuery ML trains and serves models **with SQL, inside the data warehouse**, so there is no export, no separate infrastructure, and analysts who know SQL can build and evaluate models themselves.

**Skills shown:** BigQuery ML (`CREATE MODEL`, `ML.TRAINING_INFO`, `ML.WEIGHTS`, `ML.EVALUATE`, `ML.ROC_CURVE`, `ML.PREDICT`), binary classification on rare events, class weighting, time based validation, and diagnosing a broken model from its metrics.

---

## Data and features

The source is `bigquery-public-data.google_analytics_sample.ga_sessions_*` (one row per session, 2016-08-01 to 2017-08-01).

| Feature | Source field |
|---|---|
| `label` | **1 if the session has at least one transaction, else 0** |
| `os` | `device.operatingSystem` |
| `is_mobile` | `device.isMobile` |
| `country` | `geoNetwork.country` |
| `pageviews` | `totals.pageviews` |

**Validation by time:** the model trains on 2016-08-01 to 2017-06-30 and is evaluated on 2017-07-01 to 2017-08-01, a period it has never seen. A deployed model is used the same way, trained on the past and applied to the future.

## How to run

```bash
pip install google-cloud-bigquery db-dtypes pandas matplotlib
gcloud auth application-default login
python bqml_purchase_prediction.py --project YOUR_GCP_PROJECT --print-sql   # review the SQL first
python bqml_purchase_prediction.py --project YOUR_GCP_PROJECT
```

The script trains two models: a plain logistic regression, and one with `auto_class_weights = TRUE` to handle the rare positive class. It prints the training curve, the learned weights and the evaluation metrics, saves ROC curves to `figures/`, and compares actual with predicted purchase sessions by country and by visitor. BigQuery ML training is billed; check current pricing before running.

> **Verification note.** No Google Cloud credentials were available where this code was rebuilt, so the new models have **not been run yet**. The SQL follows the documented BigQuery ML syntax and keeps the structure of the original notebook, which ran successfully. Please run it once and add the new metrics to this README.

---

## What the original run showed, and why it was wrong

The original notebook (`Machine-Learning-on-google-cloud-dataset-using-BigQuery.ipynb`) reported this on July 2017:

| precision | recall | accuracy | f1 | log loss | ROC AUC |
|---|---|---|---|---|---|
| 0.101 | 0.079 | 0.985 | 0.083 | 4.353 | **0.333** |

An ROC AUC of 0.33 is **worse than random guessing (0.5)**, and a log loss of 4.35 is very high, while accuracy looks excellent at 98.5 %. That combination is the signature of a broken setup on rare events. Purchases are rare (fewer than 3 % of sessions, judging by the transaction and user counts in project 06), so a model that predicts "no purchase" for every session already reaches about the same accuracy. Three causes were found in the code:

| Issue | Evidence | Fix |
|---|---|---|
| **The label was the transaction count** (`IF(totals.transactions IS NULL, 0, totals.transactions)`), not 0/1 | The column has values 0 to 25, so BigQuery ML built a *multiclass* logistic regression; that explains the ROC AUC and the log loss | `IF(totals.transactions IS NULL, 0, 1)`, which matches the Google tutorial this was based on |
| **The training window included the test window** (`_TABLE_SUFFIX BETWEEN '20160801' AND '20180630'` covers July 2017) | The model was evaluated on data it had trained on | Training ends 2017-06-30; evaluation is July 2017 only |
| **`CREATE MODEL IF NOT EXISTS` for the second, "cleaned" model** | Its `ML.TRAINING_INFO` table is identical to the first model's, to the last digit, because the existing model was kept and never retrained | `CREATE OR REPLACE MODEL` |

Two smaller problems were fixed as well:

* `ML.ROC_CURVE` failed with *"The ROC curve metrics of the model was not calculated during training"*. The ROC plot in the notebook came from a `roc` variable left over from an earlier session, so it did not belong to this model. The evaluation data are now passed to `ML.ROC_CURVE` explicitly.
* A comment about "mean absolute error ... 7 minutes" had been copied from a different (regression) tutorial and did not apply here. It was removed.

Even with the broken label, the original per country prediction showed the most important pattern: the **United States had 1,073 purchase sessions in July 2017**, against 19 for Canada and 2 or fewer for every other country listed. The model predicted only 220 for the US, which shows how much a rare positive class is underestimated without class weighting. That is why the rebuilt script adds the `auto_class_weights = TRUE` model.
