# Kenny Obidele

I build the whole thing: the pipeline that moves the data, the model that learns from it, and the proof that both actually work.

## Scored in public

Work that is measured by somebody other than me, against everyone else, on the same tasks.

**[harmattan](https://github.com/Kenny0bi/harmattan)**
A probabilistic influenza forecaster for the CDC's FluSight hub, and an independent scorer built from the papers to go with it. It reproduces the CDC Flu Modeling Unit's own published 2025-26 season evaluation from the raw submissions: absolute WIS, MAE and interval coverage match their table for 58 of 58 models to within 0.005. My model then enters that field under a sealed holdout and finishes 36th of 59, beating the CDC's reference baseline by 9.1%. The writeup keeps the methodology error that produced my first attempt, where four hyperparameters tuned against a single season beat the CDC's own ensemble on that season and landed exactly on the baseline out of sample, along with the measurement that explains it: a damped trend beats a flat forecast by 14.3% in one season and 3.2% in the next, so the edge belonged to the season, not the model.

**[iyipada](https://github.com/Kenny0bi/iyipada)**
Drift detection where the threshold is calculated instead of quoted. The rule every monitoring tool ships with, PSI above 0.10 means watch and above 0.25 means act, has no statistical grounding: I measured what it actually delivers on data where nothing changed, and it fires 85% of the time at 100 samples per batch and is mathematically incapable of firing at 1,000. The library derives the null distribution instead, verifies it against simulation across thirty cells, and hands you a threshold that delivers the false-alarm rate you asked for. It also benchmarks six statistics against six kinds of drift after calibrating them all to the same rate, which is the only way that comparison means anything, and finds that the industry default is beaten on the two hardest cases by a statistic almost nobody uses.

**[idogba](https://github.com/Kenny0bi/idogba)**
A reproduction that went further than the paper it reproduced. Every tutorial says to rebalance imbalanced classes with SMOTE or resampling; across 26 benchmark datasets, 4 model families and 4,160 cross-validated fits, that improved AUC on 49% of runs, which is a coin flip, and made average precision worse. What it reliably did was destroy the predicted probabilities, which no rank-based metric can see, because training at a resampled base rate adds a constant to the log-odds. In direct simulation at a 1% event rate the corrections inflate predicted risk to 36.6 times the true risk while AUC moves in the third decimal, and on a clinical decision curve SMOTE gives up 44% of the model's net benefit. The published warning was only ever tested on linear models; I show it is severe there, mild for tree ensembles, and that gradient boosting is badly miscalibrated before anyone touches the imbalance. Then the part nobody does: refitting one intercept on held-out data repairs all of it and provably cannot cost a point of AUC.

**[ami](https://github.com/Kenny0bi/ami)** · [model on Hugging Face](https://huggingface.co/kenny0bi/ami-yoruba-diacritics)
A Yoruba diacritic restoration model, and the most personal thing here: I grew up speaking Yoruba, and typed Yoruba drops the marks that make it readable, collapsing oko (farm), ọkọ (husband) and ọkọ̀ (vehicle) into one string. This 1.29M-parameter character BiLSTM puts them back at 9,600 chars/sec on a 2014 CPU: 92.4% character accuracy on the MENYO-20k test split against 77.1% for a dictionary lookup, with the whole margin exactly where it should be, on the ambiguous words only context can decide. The design makes corruption structurally impossible (six mark classes, base letters untouchable, illegal combinations masked from the softmax), the information analysis that motivates it is measured (the marks carry 1.64 bits per character), and the evaluation records the bug that nearly cheated the baseline out of 16 points.

**[asotele](https://github.com/Kenny0bi/asotele)** · [live scoreboard](https://kenny0bi.github.io/asotele/)
A public forecast ledger. Every Friday a GitHub Action files probabilistic forecasts (23 quantiles) of verifiable public quantities: worldwide M4.5+ earthquake counts, the warmest day of the week in New York and Seattle, committed to git before the target week begins, then scored against reality by the weighted interval score once outcomes mature. The commit history is the notary; misses stay on the ledger with the same weight as hits. The forecasters were calibrated by a 130-week backtest before launch (held-out coverage 0.50/0.80/0.97 against nominal 0.50/0.80/0.95 on the earthquake target), and the propriety of the scoring rule, which is what makes the record ungameable, is asserted in the tests by simulation. The first round is already on the books.

**[oogun](https://github.com/Kenny0bi/oogun)** · [live piece](https://kenny0bi.github.io/oogun/)
A data journalism piece on the FDA drug shortage list: 67 drugs in active shortage, the median already waiting 4.5 years, four on the list since 2012, drawn in full as an interactive page. Its core is the mathematics most coverage misses: a snapshot of ongoing shortages over-samples long ones in proportion to their length (the inspection paradox), so I show the 3.9x exaggeration in a simulation where the truth is known, then run renewal theory backwards to estimate what durations the observed ages actually imply, caveats on the label. Every number on the page is pinned by a test, so a data refresh that changes the story fails loudly instead of leaving stale prose.

## Core ML engineering

The from-scratch series, complete at ten: I build the machinery frameworks hide, prove it correct against the production tool, and publish the real numbers, including the failures.

**[ember](https://github.com/Kenny0bi/ember)**
A tensor autograd engine on raw NumPy: full broadcasting, batched matmul, causal attention, AdamW, cosine schedules. Every op gradient-checked against PyTorch at 1e-9; trained side by side with PyTorch from identical weights, the loss curves agree to 2.7e-07. Then it trains a 624K-parameter GPT on Shakespeare end to end, PyTorch nowhere in the loop.

**[forge-lm](https://github.com/Kenny0bi/forge-lm)**
LLM pretraining as a controlled experiment: a from-scratch byte-level BPE tokenizer (byte-exact roundtrip, incremental trainer), a RoPE/RMSNorm/SwiGLU decoder with numerical invariant tests, and eleven full training runs across five model sizes and two fixed compute budgets on a 2014 laptop CPU. The 33M model loses to the 1.3M one at every budget, the compute frontier fits a clean power law, and the tier my own background processes contaminated is in the repo next to its clean rerun, because throughput accounting caught it.

**[tinyserve-engine](https://github.com/Kenny0bi/tinyserve-engine)**
An LLM inference engine from scratch in NumPy: hand-written safetensors parser, GPT-2's BPE reimplemented byte-exactly, KV cache with rollback, per-channel INT8 (340MB to 85MB at +0.1% perplexity), and provably-exact speculative decoding. Verified token-for-token against HuggingFace, then profiled: one non-contiguous matmul held 81% of every decode step, and fixing it bought 6x. Includes the cost model for why speculative decoding loses on this CPU and when it flips.

**[quantlab](https://github.com/Kenny0bi/quantlab)**
Post-training quantization from the papers: RTN at every scale granularity, GPTQ with the full Hessian pipeline, AWQ with per-layer alpha search scored by the calibration Hessian. 34 measured wikitext-2 configs on GPT-2: 4-bit GPTQ lands within 9% of fp32 at an 8x memory cut, 3-bit is a 22x win over naive rounding, and the writeup includes the scale-granularity confound that made my first benchmark half method, half accounting.

**[gradmesh](https://github.com/Kenny0bi/gradmesh)**
Distributed data-parallel training with the collectives built from raw TCP sockets: a ring all-reduce that hits the bandwidth-optimal bound to the byte, a trainer proven bitwise-consistent across workers and within 3e-08 of single-process training, and the measured three-regime economics of when data parallelism pays. The bug ledger (mutual-sendall deadlock, TCP self-connection, float32 non-associativity) is the curriculum.

**[hive-index](https://github.com/Kenny0bi/hive-index)**
Vector search from the papers: HNSW (full layer machinery, neighbor-selection heuristic) and IVF-PQ (residual codebooks, ADC lookup tables) in NumPy, benchmarked against FAISS on 100K real SIFT vectors with exact in-repo ground truth. Same recall curves at every operating point, the 8-byte quantization ceiling reproduced and cured with the paper's rerank, and honesty about the 130x QPS gap: recall is the algorithm, QPS is the engineering.

**[relay-serve](https://github.com/Kenny0bi/relay-serve)**
A model serving system built around the dynamic batcher: the fire-full-or-fire-on-time policy with priority classes and load shedding, proven by four invariant tests and Poisson load tests to the breaking point. Measured on four cores: unbatched serving collapses at 1,200 rps (median 3ms to 757ms, thousands shed) while the batcher holds under 9ms to 2,400 rps and delivers 2.5x the throughput, with batch size self-adapting from 1.4 to 11.4.

**[fuse](https://github.com/Kenny0bi/fuse)**
A small ML graph compiler: ONNX loaded into an IR, then identity elimination, constant folding, conv+batchnorm fusion (exact algebra, all 20 in resnet18), activation fusion, dead-code sweeps, and liveness-based memory planning (13.8MB peak activations down to 4.0MB). 141 nodes to 32 with outputs identical to 1.2e-06, executed on a from-scratch NumPy runtime verified against onnxruntime before any pass ran.

**[pulse-recsys](https://github.com/Kenny0bi/pulse-recsys)**
A recommender on 25 million real MovieLens ratings: a DuckDB temporal split with no leakage by construction, a hand-differentiated two-tower with in-batch sampled softmax, and fairly tuned baselines. The first model lost to a popularity counter (recall 0.018 vs 0.058); a float64 gradcheck cleared the code, the missing logQ correction was the culprit, and one line of math took it to 0.085, beating everything. Item embeddings served through hive-index's HNSW at 0.99 recall.

**[loom](https://github.com/Kenny0bi/loom)**
The MLOps capstone: file-backed experiment tracking with data fingerprints, a registry that refuses to promote unevaluated versions and keeps rollback receipts, per-feature PSI drift monitoring that emits decisions, and canary rollouts gated by a z-test with ties going to the incumbent. Proven on a simulated production year: it detects real concept drift, retrains itself, earns promotion on live traffic at z=5.4, and destroys a poisoned model at z=-12.2 with no human in the loop.

## Data science, data engineering, analytics

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

## Stack

**Languages** Python, SQL, JavaScript, HTML, CSS, Bash, LaTeX

**Data engineering** point-in-time data reconstruction and revision handling in surveillance feeds, Kafka, Spark, dbt, Dagster, DuckDB, PostgreSQL, TimescaleDB, Redis, Parquet, batch and streaming ETL, data quality gates, dimensional and OMOP CDM modeling

**Machine learning** low-resource NLP (Yoruba diacritic restoration, character-level sequence labelling, constrained decoding via legality masks), PyTorch model training and publishing models and datasets to the Hugging Face Hub, tokenizer analysis (BPE byte fallback, cross-lingual token-premium measurement on FLORES-200 parallel text, tiktoken and Hugging Face tokenizers), class-imbalance methods and their costs (SMOTE, random over- and undersampling), leakage-safe resampling protocols, drift detection and model monitoring (PSI, Jensen-Shannon, Wasserstein, Kolmogorov-Smirnov, total variation), probabilistic and quantile forecasting (weighted interval score, pinball loss, empirical predictive distributions from nearest-neighbour residuals, damped-trend models with partial pooling), PyTorch, scikit-learn, XGBoost, Hugging Face Transformers (BioBERT), survival analysis (Cox PH, AFT, DeepSurv), reinforcement learning (DQN, Q-learning, Gymnasium), K-means and PCA, SHAP explainability, recommender systems (implicit-feedback ALS, BPR and SVD matrix factorization, TF-IDF content models, hybrid ranking with learned blend weights, MMR diversity re-ranking), contextual bandits (LinUCB, Thompson Sampling, annealed epsilon-greedy), ranking evaluation (NDCG, MAP, MRR, catalog coverage, intra-list diversity, novelty), t-SNE embedding cartography

**ML engineering** reverse-mode automatic differentiation (from-scratch tensor engine in NumPy), transformer internals (causal attention, RoPE, RMSNorm, SwiGLU, pre-norm blocks, GELU, weight tying) implemented from primitives, BPE tokenization from scratch, LLM pretraining pipelines (gradient accumulation, warmup and cosine schedules, deterministic checkpoint resume), scaling-law experiments and power-law fitting, optimizer internals (AdamW bias correction, decoupled decay, global-norm clipping), gradient checking and finite-difference verification, numerical-parity testing against PyTorch, inference optimization (KV caching, prefill/decode phase analysis, speculative decoding), quantization algorithms from papers (GPTQ Hessian pipeline, AWQ activation-aware scaling, RTN across scale granularities, layerwise sensitivity attribution), distributed training internals (ring all-reduce over sockets, gradient synchronization, parallel-efficiency and comm/compute analysis), approximate nearest-neighbor search (HNSW, IVF-PQ, recall/QPS benchmarking against FAISS), serving-systems engineering (dynamic batching, priority scheduling, load shedding, Poisson load testing, latency-percentile analysis), ML compilation (ONNX IR, operator fusion, constant folding, liveness-based memory planning, differential testing against onnxruntime), research-to-production model shipping (torch-to-ONNX export with parity proofs, dynamic INT8 quantization with full test-set re-evaluation, latency and cold-start benchmarking, in-browser inference with onnxruntime-web), production retrieval modeling (two-tower training, in-batch sampled softmax with logQ correction, temporal evaluation, recall/NDCG harnesses), MLOps platform engineering (experiment tracking, model registry and lineage, PSI/KS drift monitoring, canary analysis, automated retraining and rollback), profiling and BLAS-aware NumPy performance work, safetensors and BPE internals, throughput benchmarking and contention forensics, Manim mathematical animation

**Statistics** information theory in practice (entropy analyses driving model design), renewal theory and length-biased sampling (the inspection paradox, age-distribution inversion by MLE), reproduction and extension of published results, model calibration (calibration slope and intercept, Brier decomposition, decision curve analysis and net benefit), prior-shift correction for resampled base rates, null-distribution derivation and threshold calibration, false-alarm and power analysis by simulation, multiple-testing control (Benjamini-Hochberg, Sidak), sequential change detection (CUSUM, Page-Hinkley) calibrated to average run length, permutation tests, proper scoring rules and forecast evaluation (WIS and its sharpness/overprediction/underprediction decomposition, PIT calibration, pairwise relative skill), leave-one-season-out cross-validation, vintage-aware (point-in-time) backtesting, frequentist and Bayesian A/B testing, sequential testing (SPRT), power analysis, disproportionality methods (PRR, ROR, BCPNN, MGPS), empirical-Bayes shrinkage, FDR control, Mantel-Haenszel confounding screens, Holt-Winters forecasting, anomaly detection (Isolation Forest, control charts), statsmodels, SciPy

**Serving and apps** interactive data-journalism pages (hand-built SVG and vanilla JS, GitHub Pages), FastAPI, Streamlit, Plotly, Firebase (Auth, Realtime Database), REST API design

**Engineering practice** self-operating pipelines on scheduled GitHub Actions (a forecast ledger that files, scores and publishes itself weekly), publishing installable libraries with a documented public API, Docker, docker-compose, GitHub Actions CI and scheduled unattended jobs, pytest, mypy, ruff, Git, reproducible pipelines that validate their own output
