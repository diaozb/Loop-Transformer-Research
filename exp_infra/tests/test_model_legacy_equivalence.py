from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXP_INFRA = os.path.join(REPO_ROOT, "exp_infra")
SRC = os.path.join(REPO_ROOT, "src")
sys.path.insert(0, EXP_INFRA)
sys.path.insert(0, SRC)

from models import build_general_model as build_legacy_model  # noqa: E402
from ltf.config import ModelConfig  # noqa: E402
from ltf.models import build_looped_model  # noqa: E402


def _legacy_conf(**overrides):
    payload = dict(
        family="gpt2",
        n_dims=6,
        n_positions=32,
        n_embd=16,
        n_layer=1,
        n_head=4,
        linear_embedding=True,
        use_wpe=False,
        wpe_mode=None,
        use_rope=False,
        rope_theta=10000.0,
    )
    payload.update(overrides)
    return SimpleNamespace(**payload)


class ModelLegacyEquivalenceTest(unittest.TestCase):
    def assert_mode_matches(self, **overrides):
        conf = _legacy_conf(**overrides)
        model_conf = ModelConfig(**vars(conf))
        torch.manual_seed(17)
        legacy = build_legacy_model(conf).eval()
        migrated = build_looped_model(model_conf).eval()
        migrated.load_state_dict(legacy.state_dict())

        torch.manual_seed(23)
        xs = torch.zeros(3, 9, 6)
        token_ids = torch.randint(low=0, high=6, size=(3, 9))
        xs.scatter_(2, token_ids.unsqueeze(-1), 1.0)

        with torch.no_grad():
            old_logits = legacy.looped_forward(xs, horizon=5)
            new_logits = migrated.looped_forward(xs, horizon=5)

        self.assertEqual(len(old_logits), len(new_logits))
        for old, new in zip(old_logits, new_logits):
            self.assertLessEqual((old - new).abs().max().item(), 1e-6)

    def test_nope_matches_legacy(self):
        self.assert_mode_matches(use_wpe=False, wpe_mode=None, use_rope=False)

    def test_rope_matches_legacy(self):
        self.assert_mode_matches(use_wpe=False, wpe_mode=None, use_rope=True)

    def test_wpe_all_matches_legacy(self):
        self.assert_mode_matches(use_wpe=True, wpe_mode="all", use_rope=False)

    def test_wpe_once_matches_legacy(self):
        self.assert_mode_matches(use_wpe=True, wpe_mode="once", use_rope=False)


if __name__ == "__main__":
    unittest.main()
