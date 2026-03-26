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
  train.py             — PPO training loop
  evaluate.py          — evaluation against baselines
tests/
  test_data_loader.py
  test_feature_engineer.py
  test_portfolio_env.py
  test_evaluate.py
deps/                  — git submodules (populated by --recurse-submodules)
output/                — model checkpoints and evaluation artefacts (gitignored)
```
