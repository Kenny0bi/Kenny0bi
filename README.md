# Kenny Obidele

I build data systems end to end and then prove they work. Data science, data engineering, and analytics, with a portfolio that leans healthcare: pipelines that run on real FDA data, warehouses built to clinical standards, and models graded against cohorts where I planted the ground truth on purpose.

One rule holds across everything here: every number in every README comes from actually running the code. When a shipped model turned out to be worse than random, the README says so and walks through the diagnosis, because finding that bug was the real work.

## Six projects worth your time

**[adverse-event-pipeline](https://github.com/Kenny0bi/adverse-event-pipeline)**
Pharmacovigilance signal detection on 1.2M real FDA FAERS cases: PRR, ROR, BCPNN, and MGPS with fitted priors, FDR control, and stratified confounding screens. It re-runs itself quarterly through GitHub Actions and refuses to pass unless six known drug-event signals resurface.

**[product-analytics-engine](https://github.com/Kenny0bi/product-analytics-engine)**
50K synthetic users and 2M+ events in DuckDB, engineered to behave like a real product (power-law usage, churn, seasonality). Funnels, cohort retention, Bayesian and sequential A/B testing, Holt-Winters forecasting, Dagster, FastAPI, and a dashboard where the funnel is a river and retention is a field of comet trails.

**[real-time-vitals-pipeline](https://github.com/Kenny0bi/real-time-vitals-pipeline)**
Streaming patient vitals through Kafka and Spark into TimescaleDB, with MEWS clinical scoring, anomaly detection, and a live telemetry-wall dashboard.

**[ehr-to-omop-warehouse](https://github.com/Kenny0bi/ehr-to-omop-warehouse)**
Synthea EHR data transformed into OMOP CDM v5.4 twice, once in pandas and once in dbt, with quality gates between every stage and a cohort engine on top.

**[cancer-survival-prediction](https://github.com/Kenny0bi/cancer-survival-prediction)**
Multi-modal survival modeling: Cox PH, XGBoost-AFT, and DeepSurv on structured data, BioBERT on clinical notes, SHAP for attribution. Validated on a synthetic cohort with planted effects, so there is an oracle to beat, and the multi-modal models beat it.

**[Financial-Risk-mdp-RL](https://github.com/Kenny0bi/Financial-Risk-mdp-RL)**
A 5-state credit-risk Markov chain for a neobank and a Q-learning intervention agent, including a written diagnosis of why the textbook-default discount factor provably cannot work in this environment while gamma=0.3 converges cleanly.

## How I work

I verify instead of assuming. Synthetic data gets engineered until its funnels, retention curves, and seasonality look like a real product's, because downstream analytics are only as honest as the data underneath. Statistics get pinned to hand-calculated values in tests. Charts get designed from the shape of the data rather than pulled from a library's defaults, then actually rendered and looked at before they ship.

## Stack

Python, SQL, DuckDB, PostgreSQL/TimescaleDB, Kafka, Spark, dbt, Dagster, pandas, NumPy, scikit-learn, XGBoost, PyTorch, statsmodels, FastAPI, Streamlit, Plotly, Docker, GitHub Actions.

Also at [github.com/khennyG](https://github.com/khennyG).
