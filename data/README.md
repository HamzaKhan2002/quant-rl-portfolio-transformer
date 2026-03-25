# data/ — not tracked in git

Data files must be sourced from upstream sibling repos via environment variables.
See .env.example at the repo root for the expected variable names.

  SENTIMENT_FEATURES_PATH → Phase 2 output: sentiment_features.csv
  REGIME_PROBS_PATH        → Phase 1 output: regime_probs.csv

Default sibling-repo layout:

```
../quant-nlp-claude-grok-sentiment/output/sentiment_features.csv
../quant-multi-transformer-regimes/output/regime_probs.csv
```
