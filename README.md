# Data Analysis and Machine Learning Portfolio

A **Kaggle credit risk competition** entry and six practice projects, all end to end in **Python and SQL**. Together they cover exploratory analysis, statistical testing, regression, classification, text mining, relational database design and cloud analytics on Google BigQuery.

Every project is a runnable script with a command line interface, saved figures and results, and a README. Each README explains **the question, why it matters, the method, the findings, and what was corrected** when the original notebooks were reviewed. The aim is twofold: to show how I work, and to give anyone doing similar work a clear, correct template to learn from.

---

## Featured project: Kaggle competition

| Project | Question | Main techniques | Result |
|---|---|---|---|
| [01 · American Express Default Prediction](01-amex-default-prediction) | Will a credit card customer default, judged from up to 13 monthly statements? | Customer level time series aggregation, feature selection (correlation filter, permutation and null importance), LightGBM, gradient boosting, neural network, K fold CV, the official competition metric, calibration | Solo entry in a featured Kaggle competition, finishing 4,137th of 4,874 teams. A later review found the main reasons (a small data subset, 0/1 submissions, a bug that misaligned predictions with customers), and the rebuilt pipeline fixes all three |

This is real, externally scored work: a featured Kaggle competition run by American Express on anonymised credit card data, where every team was ranked on a hidden test set.

## Practice and skill building projects

These projects use well known public data sets to practise and demonstrate specific techniques.

| # | Project | Question | Main techniques | Headline result |
|---|---|---|---|---|
| 02 | [Insurance cost prediction](02-insurance-cost-prediction) | What drives medical insurance charges? | EDA, Mann Whitney U test, linear / polynomial / tree models, pipelines, CV | A domain feature (*smoker AND BMI ≥ 30*) lets a linear model reach test R² **0.899**, matching random forest and gradient boosting while staying interpretable |
| 03 | [Titanic survival](03-titanic-survival-classification) | Who survived, and can it be predicted? | Leakage free feature engineering, 10 classifier benchmark, RFECV, grid search, ROC, odds ratios | Tuned logistic regression: CV accuracy **0.834**, ROC AUC **0.881**; a man had about one sixth of a comparable woman's odds of survival |
| 04 | [Fake news source types](04-fake-news-source-classification) | Can the text reveal the type of an unreliable source? | n-grams, word clouds, TF-IDF, linear and ensemble classifiers, grouped split | Original accuracy 0.933; the evaluation was redesigned to test whether the model reads text or recognises websites |
| 05 | [MySQL relational database](05-mysql-relational-database) | How do you design and query a consistent database from Python? | Schema design, foreign keys, joins, GROUP BY / HAVING, transactions, parameterised SQL | Fully re-runnable build and query script with an ER diagram and a sample run |
| 06 | [BigQuery web analytics + soccer SQL](06-bigquery-web-analytics-and-sql) | Where do e-commerce visitors come from, and who buys? | BigQuery SQL on nested data, cost control, window functions, CTEs, SQLite | The US made up 55 % of the top 10 countries' users but almost all of their transactions (11,440 of 11,711) |
| 07 | [BigQuery ML purchase prediction](07-bigquery-ml-purchase-prediction) | Will a website session end in a purchase? | BigQuery ML logistic regression, time based validation, class weights | Diagnosed why the original model scored ROC AUC 0.33 (a multiclass label and train/test overlap) and fixed it |

## Skills demonstrated

| Area | Tools and methods |
|---|---|
| Data wrangling | pandas, NumPy, missing value strategies, Parquet, memory aware dtypes, customer level aggregation |
| Statistics | hypothesis testing (Mann Whitney U), correlation analysis, distribution analysis, odds ratios, calibration |
| Machine learning | scikit-learn pipelines, linear / polynomial / logistic regression, SVM, k-NN, naive Bayes, LDA / QDA, decision trees, random forests, AdaBoost, gradient boosting, LightGBM, PCA |
| Model evaluation | stratified and grouped K fold CV, hold out sets, time based validation, ROC AUC, macro F1, log loss, Brier score, threshold selection, a custom competition metric |
| Feature work | domain driven features, regex extraction, TF-IDF, RFECV, permutation importance, null importance, correlation filtering |
| SQL | MySQL, SQLite, Google BigQuery: DDL, constraints, joins, subqueries, CTEs, window functions, transactions, BigQuery ML |
| Visualisation | Matplotlib, seaborn, Plotly choropleth maps, word clouds, decision tree plots |
| Engineering practice | reproducible scripts with CLIs, fixed random seeds, no data leakage, credentials kept out of code, cost guards for cloud queries |

## A note on rigour

While turning the original notebooks into scripts, I re-checked every result against its code. Several real errors turned up, and each one is documented in its project README:

* competition submissions where `predict_proba` output was flattened, **misaligning probabilities with customer IDs** (01)
* a classifier comparison that was predicting the **age band instead of survival** (03)
* PCA **fitted on the test set** (02)
* a BigQuery ML "binary" model that was really **multiclass**, with a training window that **overlapped the test window**, and a retrained model that was **never actually retrained** (07)
* a pie chart that **split the Americas 50/50**, overstating South American traffic more than tenfold (06)

Finding and fixing mistakes like these, and explaining why they matter, is as much a part of data science as building the models. I came to data science from computational chemistry research, where a result counts only once it has been checked against its method.

## About me

I hold bachelor's, master's and PhD degrees in **chemistry**, specialising in **computational chemistry**, and I was a university lecturer for about a decade from 2016. I completed a degree in **Computing Science at the University of Alberta** (June 2026). To understand the financial sector in depth, I passed the **LLQP** and the **Canadian Securities Course (CSC)**.

I combine these backgrounds to build **FINWISE**, the main service of my company **Strategic Financial Building** ([www.strategicfinancialbuilding.ca](https://www.strategicfinancialbuilding.ca)). FINWISE aims to make financial planning simple and free for families, individuals and corporations, with a customised subscription version for financial professionals.

## Repository layout

```
Data-Analysis/
├── README.md                  ← you are here
├── requirements.txt
├── 01-amex-default-prediction/          ← Kaggle competition
│   ├── README.md
│   ├── amex_default_prediction.py
│   └── figures/
├── 02-insurance-cost-prediction/        ← practice projects from here on
│   ├── README.md
│   ├── insurance_cost_analysis.py
│   ├── figures/                         ← generated by the script
│   └── results/                         ← metrics tables and run log
├── 03-titanic-survival-classification/
├── 04-fake-news-source-classification/
├── 05-mysql-relational-database/
├── 06-bigquery-web-analytics-and-sql/
└── 07-bigquery-ml-purchase-prediction/
```

## Quick start

```bash
git clone <this repository>
cd Data-Analysis
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cd 02-insurance-cost-prediction   # runs in seconds; the AmEx data need a Kaggle download
mkdir -p data && curl -L -o data/insurance.csv \
  https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv
python insurance_cost_analysis.py
```

Each project README lists its own data source and command. Data files are not stored in the repository; they come from Kaggle, public GitHub mirrors, or Google's public BigQuery data sets.

## Verification status

| Project | How the current code was verified |
|---|---|
| 02, 03 | Run end to end on the real data; every number in the README comes from these runs |
| 05 | Run end to end against a live MariaDB 10.11 server (MySQL compatible); output in `sample_output.txt` |
| 01, 04, 06 (soccer) | Run end to end on synthetic data with the real schema (the Kaggle files were not downloadable in the build environment); reported numbers are from the original notebook runs on the real data |
| 06 (BigQuery), 07 | SQL checked with a BigQuery SQL parser, plotting code tested; needs a Google Cloud project for a live run |
