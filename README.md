# quant-rl-portfolio-transformer

PPO reinforcement-learning agent for multi-asset portfolio management (AAPL, MSFT, GOOGL, AMZN, TSLA). Phase 3 of a 3-phase quantitative trading pipeline.

## Quickstart

```bash
git clone --recurse-submodules \
    https://github.com/HamzaKhan2002/quant-rl-portfolio-transformer
pip install -r requirements.txt
python src/train.py
```

If you cloned without `--recurse-submodules`, initialise dependencies with:

```bash
git submodule update --init --recursive
```

## Submodule structure

| Phase | Role | Path |
|-------|------|------|
| Phase 0 (shared utils) | Common utilities | `deps/quant-shared-utils` |
| Phase 1 (regime model) | Market regime detection | `deps/quant-multi-transformer-regimes` |
| Phase 2 (sentiment)    | NLP sentiment pipeline | `deps/quant-nlp-claude-grok-sentiment` |

Data paths are configured in `config.yaml` under `data.sentiment_path` and `data.regime_path` and resolve to CSV files inside the submodule checkouts.

## Project layout

```
src/
  data_loader.py       — load + merge Phase 1/2 features
  feature_engineer.py  — price features, normalisation
  portfolio_env.py     — Gymnasium PortfolioEnv (PPO environment)
  ppo_agent.py         — PPO factory and training callbacks
  train.py             — PPO training loop
  tune.py              — Optuna hyperparameter tuning
  evaluate.py          — evaluation against baselines
tests/
  test_data_loader.py
  test_feature_engineer.py
  test_portfolio_env.py
  test_ppo_agent.py
  test_evaluate.py
  test_tune.py
deps/                  — git submodules (populated by --recurse-submodules)
output/                — model checkpoints and evaluation artefacts (gitignored)
```

## Hyperparameter Tuning

Sequential (single core):

```bash
python -m src.tune --n-trials 50
```

Parallel (recommended — open N terminals simultaneously):

```bash
python -m src.tune --n-trials 13   # terminal 1
python -m src.tune --n-trials 13   # terminal 2
python -m src.tune --n-trials 13   # terminal 3
python -m src.tune --n-trials 13   # terminal 4
# 52 trials total, each terminal uses 1 CPU core
```

All processes share a single SQLite study (`output/optuna_study.db`) via Optuna's
storage backend. Do NOT use `n_jobs > 1` in `study.optimize()` — Optuna's default
thread-based parallelism is blocked by Python's GIL for CPU-bound SB3 training.
The multi-terminal approach gives true multiprocessing.
