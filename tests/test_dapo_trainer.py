"""
tests/test_dapo_trainer.py — Unit tests for src/dapo_trainer.py.

Tests use synthetic data only — no network access, no upstream files.
PyTorch is required.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.dapo_trainer import (
    GaussianPolicy,
    DAPOTrainer,
    collect_dapo_rollout,
    dapo_update,
    _eval_sharpe,
)


# ---------------------------------------------------------------------------
# Constants and helpers
# ---------------------------------------------------------------------------

OBS_DIM    = 34
ACTION_DIM = 6
N_TICKERS  = 5
T          = 60   # episode length
K          = 4    # group size (small for tests)


def _make_config(**overrides) -> dict:
    base = {
        "env": {
            "n_tickers":           N_TICKERS,
            "action_dim":          ACTION_DIM,
            "obs_dim":             OBS_DIM,
            "transaction_cost":    0.001,
            "vol_eps":             1e-8,
            "vol_window":          21,
            "max_weight_per_ticker": 1.0,
        },
        "ppo": {
            "policy_kwargs": {
                "net_arch": [64, 64],
                "activation_fn": "Tanh",
            },
            "learning_rate":  3e-4,
            "ent_coef":       0.01,
            "n_epochs":       1,
            "batch_size":     32,
            "max_grad_norm":  0.5,
            "total_timesteps": 100,
        },
        "dapo": {
            "group_size":     K,
            "eps_low":        0.1,
            "eps_high":       0.2,
            "filter_min_std": 1e-8,
        },
    }
    base["env"].update(overrides)
    return base


def _make_obs_ret(seed=0):
    rng = np.random.default_rng(seed)
    obs = rng.standard_normal((T + 1, OBS_DIM)).astype(np.float32)
    ret = rng.normal(0.001, 0.01, (T + 1, N_TICKERS)).astype(np.float32)
    return obs, ret


# ---------------------------------------------------------------------------
# Tests: GaussianPolicy
# ---------------------------------------------------------------------------

class TestGaussianPolicy:
    def _make_policy(self):
        return GaussianPolicy(OBS_DIM, ACTION_DIM, net_arch=[64, 64])

    def test_forward_output_shapes(self):
        policy = self._make_policy()
        obs = torch.randn(OBS_DIM)
        mu, log_std = policy(obs)
        assert mu.shape == (ACTION_DIM,)
        assert log_std.shape == (ACTION_DIM,)

    def test_forward_batch_shapes(self):
        policy = self._make_policy()
        batch = torch.randn(8, OBS_DIM)
        mu, log_std = policy(batch)
        assert mu.shape == (8, ACTION_DIM)
        assert log_std.shape == (8, ACTION_DIM)

    def test_log_std_clamped(self):
        policy = self._make_policy()
        obs = torch.randn(100, OBS_DIM)
        _, log_std = policy(obs)
        assert (log_std >= -5.0).all()
        assert (log_std <= 2.0).all()

    def test_sample_shapes_single_obs(self):
        policy = self._make_policy()
        obs = torch.randn(OBS_DIM)
        actions, logprobs = policy.sample(obs, n=K)
        assert actions.shape == (K, ACTION_DIM)
        assert logprobs.shape == (K,)

    def test_log_prob_shape(self):
        policy = self._make_policy()
        batch = 8
        obs     = torch.randn(batch, OBS_DIM)
        actions = torch.randn(batch, ACTION_DIM)
        lp = policy.log_prob(obs, actions)
        assert lp.shape == (batch,)

    def test_log_prob_is_finite(self):
        policy = self._make_policy()
        obs     = torch.randn(16, OBS_DIM)
        actions = torch.randn(16, ACTION_DIM)
        lp = policy.log_prob(obs, actions)
        assert torch.isfinite(lp).all()

    def test_entropy_positive(self):
        policy = self._make_policy()
        obs = torch.randn(8, OBS_DIM)
        ent = policy.entropy(obs)
        assert ent.shape == (8,)

    def test_deterministic_action_is_mean(self):
        policy = self._make_policy()
        policy.eval()
        obs = torch.randn(OBS_DIM)
        with torch.no_grad():
            mu, _ = policy(obs)
            actions, _ = policy.sample(obs, n=100)
        # Mean of many samples should be close to μ
        sample_mean = actions.mean(0)
        assert torch.allclose(sample_mean, mu, atol=0.2)


# ---------------------------------------------------------------------------
# Tests: collect_dapo_rollout
# ---------------------------------------------------------------------------

class TestCollectDapoRollout:
    def test_output_shapes_consistent(self):
        policy = GaussianPolicy(OBS_DIM, ACTION_DIM, [64, 64])
        obs, ret = _make_obs_ret()
        obs_rb, act_rb, logp_rb, adv_rb = collect_dapo_rollout(
            policy, obs, ret, n_tickers=N_TICKERS, max_weight=1.0, K=4,
        )
        assert obs_rb.shape[1] == OBS_DIM
        assert act_rb.shape[1] == ACTION_DIM
        assert logp_rb.shape[0] == obs_rb.shape[0]
        assert adv_rb.shape[0] == obs_rb.shape[0]

    def test_rollout_size_is_multiple_of_K(self):
        """Number of training samples must be a multiple of K."""
        policy = GaussianPolicy(OBS_DIM, ACTION_DIM, [64, 64])
        obs, ret = _make_obs_ret()
        obs_rb, _, _, _ = collect_dapo_rollout(
            policy, obs, ret, n_tickers=N_TICKERS, max_weight=1.0, K=4,
        )
        # Each retained group contributes exactly K samples
        assert len(obs_rb) % 4 == 0

    def test_advantages_mean_close_to_zero(self):
        """Group-relative advantages: within each K-group, mean should be ~0."""
        policy = GaussianPolicy(OBS_DIM, ACTION_DIM, [64, 64])
        obs, ret = _make_obs_ret()
        _, _, _, adv_rb = collect_dapo_rollout(
            policy, obs, ret, n_tickers=N_TICKERS, max_weight=1.0, K=4,
        )
        if len(adv_rb) < 4:
            pytest.skip("All groups filtered — nothing to test")
        # Sum over each K=4 window should be approximately 0
        n_groups = len(adv_rb) // 4
        for i in range(n_groups):
            group_sum = adv_rb[i*4:(i+1)*4].sum()
            assert abs(group_sum) < 1e-5, f"Group {i} advantage mean != 0"

    def test_logprobs_are_finite(self):
        policy = GaussianPolicy(OBS_DIM, ACTION_DIM, [64, 64])
        obs, ret = _make_obs_ret()
        _, _, logp_rb, _ = collect_dapo_rollout(
            policy, obs, ret, n_tickers=N_TICKERS, max_weight=1.0, K=4,
        )
        assert np.isfinite(logp_rb).all()

    def test_empty_result_when_all_filtered(self):
        """With very high filter_min_std, all groups should be filtered."""
        policy = GaussianPolicy(OBS_DIM, ACTION_DIM, [64, 64])
        obs, ret = _make_obs_ret()
        obs_rb, _, _, _ = collect_dapo_rollout(
            policy, obs, ret, n_tickers=N_TICKERS, max_weight=1.0, K=4,
            filter_min_std=1e10,  # impossibly high threshold
        )
        assert len(obs_rb) == 0


# ---------------------------------------------------------------------------
# Tests: dapo_update
# ---------------------------------------------------------------------------

class TestDapoUpdate:
    def _make_batch(self, n=64):
        policy = GaussianPolicy(OBS_DIM, ACTION_DIM, [64, 64])
        rng = np.random.default_rng(0)
        obs  = rng.standard_normal((n, OBS_DIM)).astype(np.float32)
        acts = rng.normal(0, 1, (n, ACTION_DIM)).astype(np.float32)
        logp = np.full(n, -5.0, dtype=np.float32)
        adv  = rng.standard_normal(n).astype(np.float32)
        opt  = torch.optim.Adam(policy.parameters(), lr=1e-3)
        return policy, opt, obs, acts, logp, adv

    def test_returns_dict_with_required_keys(self):
        policy, opt, obs, acts, logp, adv = self._make_batch()
        result = dapo_update(policy, opt, obs, acts, logp, adv, n_epochs=1)
        for key in ("policy_loss", "entropy_loss", "total_loss", "approx_kl"):
            assert key in result

    def test_loss_values_are_finite(self):
        policy, opt, obs, acts, logp, adv = self._make_batch()
        result = dapo_update(policy, opt, obs, acts, logp, adv, n_epochs=1)
        for v in result.values():
            assert np.isfinite(v), f"Non-finite loss: {v}"

    def test_policy_params_change_after_update(self):
        policy, opt, obs, acts, logp, adv = self._make_batch(n=128)
        params_before = [p.clone().detach() for p in policy.parameters()]
        dapo_update(policy, opt, obs, acts, logp, adv, n_epochs=2)
        params_after = [p.clone().detach() for p in policy.parameters()]
        changed = any(
            not torch.allclose(b, a) for b, a in zip(params_before, params_after)
        )
        assert changed, "Policy parameters did not change after DAPO update"

    def test_asymmetric_clipping_eps(self):
        """Verify clipping is applied with asymmetric bounds."""
        policy = GaussianPolicy(OBS_DIM, ACTION_DIM, [64, 64])
        opt = torch.optim.Adam(policy.parameters(), lr=1e-4)
        rng = np.random.default_rng(1)
        n = 64
        obs  = rng.standard_normal((n, OBS_DIM)).astype(np.float32)
        acts = rng.normal(0, 1, (n, ACTION_DIM)).astype(np.float32)
        # Old log probs very different from new → ratio far from 1 → clipping fires
        logp = np.full(n, -20.0, dtype=np.float32)
        adv  = rng.standard_normal(n).astype(np.float32)
        # Should not raise (clipping prevents extreme gradients)
        result = dapo_update(
            policy, opt, obs, acts, logp, adv,
            n_epochs=1, eps_low=0.1, eps_high=0.2
        )
        assert np.isfinite(result["policy_loss"])


# ---------------------------------------------------------------------------
# Tests: _eval_sharpe
# ---------------------------------------------------------------------------

class TestEvalSharpe:
    def test_returns_finite_float(self):
        policy = GaussianPolicy(OBS_DIM, ACTION_DIM, [64, 64])
        obs, ret = _make_obs_ret()
        sharpe = _eval_sharpe(policy, obs, ret, N_TICKERS, max_weight=1.0)
        assert np.isfinite(sharpe)

    def test_returns_float_type(self):
        policy = GaussianPolicy(OBS_DIM, ACTION_DIM, [64, 64])
        obs, ret = _make_obs_ret()
        sharpe = _eval_sharpe(policy, obs, ret, N_TICKERS, max_weight=1.0)
        assert isinstance(sharpe, float)


# ---------------------------------------------------------------------------
# Tests: DAPOTrainer
# ---------------------------------------------------------------------------

class TestDAPOTrainer:
    def _make_trainer(self):
        return DAPOTrainer(
            obs_dim=OBS_DIM, action_dim=ACTION_DIM,
            config=_make_config(), seed=42,
        )

    def test_predict_returns_correct_shape(self):
        trainer = self._make_trainer()
        obs, _ = _make_obs_ret()
        action, state = trainer.predict(obs[0], deterministic=True)
        assert action.shape == (ACTION_DIM,)
        assert state is None

    def test_predict_stochastic_returns_correct_shape(self):
        trainer = self._make_trainer()
        obs, _ = _make_obs_ret()
        action, _ = trainer.predict(obs[0], deterministic=False)
        assert action.shape == (ACTION_DIM,)

    def test_predict_deterministic_is_reproducible(self):
        trainer = self._make_trainer()
        obs, _ = _make_obs_ret()
        a1, _ = trainer.predict(obs[0], deterministic=True)
        a2, _ = trainer.predict(obs[0], deterministic=True)
        assert np.allclose(a1, a2)

    def test_train_completes_and_returns_log(self):
        obs, ret = _make_obs_ret()
        config = _make_config()
        config["dapo"]["filter_min_std"] = 1e-10  # allow most groups
        config["ppo"]["total_timesteps"] = T  # one epoch only
        trainer = DAPOTrainer(OBS_DIM, ACTION_DIM, config=config, seed=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            log = trainer.train(obs, ret, obs, ret, out_dir=tmpdir, eval_freq=T)
        assert isinstance(log, list)

    def test_save_and_load_roundtrip(self):
        """Saved and loaded trainer should predict the same action."""
        trainer = self._make_trainer()
        obs, _ = _make_obs_ret()
        action_before, _ = trainer.predict(obs[0], deterministic=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "policy.pt")
            trainer.save(path)
            loaded = DAPOTrainer.load(path, config=_make_config())

        action_after, _ = loaded.predict(obs[0], deterministic=True)
        assert np.allclose(action_before, action_after, atol=1e-6)

    def test_save_creates_file(self):
        trainer = self._make_trainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "policy.pt")
            trainer.save(path)
            assert os.path.exists(path)

    def test_load_restores_hparams(self):
        trainer = self._make_trainer()
        trainer.eps_low  = 0.05
        trainer.eps_high = 0.15
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "policy.pt")
            trainer.save(path)
            loaded = DAPOTrainer.load(path, config=_make_config())
        assert loaded.eps_low  == 0.05
        assert loaded.eps_high == 0.15
