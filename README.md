# Kenny Obidele

I build the whole thing: the pipeline that moves the data, the model that learns from it, and the proof that both actually work.

## Core ML engineering

The from-scratch series: I build the machinery frameworks hide, prove it correct against the production tool, and publish the real numbers.

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

**Data engineering** Kafka, Spark, dbt, Dagster, DuckDB, PostgreSQL, TimescaleDB, Parquet, batch and streaming ETL, data quality gates, dimensional and OMOP CDM modeling

**Machine learning** PyTorch, scikit-learn, XGBoost, Hugging Face Transformers (BioBERT), survival analysis (Cox PH, AFT, DeepSurv), reinforcement learning (DQN, Q-learning, Gymnasium), K-means and PCA, SHAP explainability

**ML engineering** reverse-mode automatic differentiation (from-scratch tensor engine in NumPy), transformer internals (causal attention, RoPE, RMSNorm, SwiGLU, pre-norm blocks, GELU, weight tying) implemented from primitives, BPE tokenization from scratch, LLM pretraining pipelines (gradient accumulation, warmup and cosine schedules, deterministic checkpoint resume), scaling-law experiments and power-law fitting, optimizer internals (AdamW bias correction, decoupled decay, global-norm clipping), gradient checking and finite-difference verification, numerical-parity testing against PyTorch, inference optimization (KV caching, prefill/decode phase analysis, speculative decoding), quantization algorithms from papers (GPTQ Hessian pipeline, AWQ activation-aware scaling, RTN across scale granularities, layerwise sensitivity attribution), distributed training internals (ring all-reduce over sockets, gradient synchronization, parallel-efficiency and comm/compute analysis), approximate nearest-neighbor search (HNSW, IVF-PQ, recall/QPS benchmarking against FAISS), profiling and BLAS-aware NumPy performance work, safetensors and BPE internals, throughput benchmarking and contention forensics, Manim mathematical animation

**Statistics** frequentist and Bayesian A/B testing, sequential testing (SPRT), power analysis, disproportionality methods (PRR, ROR, BCPNN, MGPS), empirical-Bayes shrinkage, FDR control, Mantel-Haenszel confounding screens, Holt-Winters forecasting, anomaly detection (Isolation Forest, control charts), statsmodels, SciPy

**Serving and apps** FastAPI, Streamlit, Plotly, Firebase (Auth, Realtime Database), REST API design

**Engineering practice** Docker, docker-compose, GitHub Actions CI, pytest, mypy, ruff, Git, reproducible pipelines that validate their own output
