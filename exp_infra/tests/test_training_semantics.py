from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXP_INFRA = os.path.join(REPO_ROOT, "exp_infra")
SRC = os.path.join(REPO_ROOT, "src")
sys.path.insert(0, EXP_INFRA)
sys.path.insert(0, SRC)

from models import build_general_model as build_legacy_model  # noqa: E402
from train_ponder import PonderLoopedModel as LegacyPonderLoopedModel  # noqa: E402
from ltf.config import EvalConfig, LoggingConfig, ModelConfig, RunConfig, TaskConfig, TrainerConfig  # noqa: E402
from ltf.data import generate_copy, generate_parity, prepare_inputs  # noqa: E402
from ltf.models import PonderLoopedModel, build_looped_model  # noqa: E402
from ltf.training import fixed_loop_loss, run_training  # noqa: E402


def _conf(**overrides):
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


class TrainingSemanticsTest(unittest.TestCase):
    def test_fixed_loop_loss_matches_legacy_selection(self):
        conf = _conf()
        model_conf = ModelConfig(**vars(conf))
        torch.manual_seed(17)
        legacy = build_legacy_model(conf).eval()
        migrated = build_looped_model(model_conf).eval()
        migrated.load_state_dict(legacy.state_dict())

        batch = generate_copy(
            batch_size=8,
            max_len=8,
            min_length=1,
            max_length_exclusive=7,
            prob_one=0.5,
        )
        xs = prepare_inputs(batch.inputs, linear_embedding=True, n_dims=6, device="cpu")
        horizon = 9

        with torch.no_grad():
            old_states = legacy.looped_forward(xs, horizon=horizon)
            selected = []
            for sample_idx in range(batch.lengths.shape[0]):
                selected.append(old_states[int(batch.lengths[sample_idx].item()) - 1][sample_idx])
            old_logits = torch.stack(selected, dim=0)
            old_loss = F.cross_entropy(old_logits[batch.mask == 1], batch.targets[batch.mask == 1])
            new_loss = fixed_loop_loss(migrated, xs, batch, horizon=horizon, task_name="copy")

        self.assertLessEqual(abs(old_loss.item() - new_loss.item()), 1e-6)

    def test_ponder_forward_matches_legacy_wrapper(self):
        conf = _conf(use_rope=True)
        model_conf = ModelConfig(**vars(conf))
        torch.manual_seed(17)
        legacy_base = build_legacy_model(conf).eval()
        legacy = LegacyPonderLoopedModel(legacy_base).eval()
        migrated_base = build_looped_model(model_conf).eval()
        migrated = PonderLoopedModel(migrated_base).eval()
        migrated.load_state_dict(legacy.state_dict())

        batch = generate_parity(
            batch_size=8,
            max_len=8,
            min_length=1,
            max_length_exclusive=7,
        )
        xs = prepare_inputs(batch.inputs, linear_embedding=True, n_dims=6, device="cpu")
        halt_mask = batch.mask.bool()

        with torch.no_grad():
            old_logits, old_p = legacy.forward_ponder(xs, max_steps=5, halt_mask=halt_mask)
            new = migrated.forward_ponder(xs, max_steps=5, halt_mask=halt_mask)

        self.assertLessEqual((old_logits - new.logits_steps).abs().max().item(), 1e-6)
        self.assertLessEqual((old_p - new.p_steps).abs().max().item(), 1e-6)
        self.assertTrue(torch.allclose(new.p_steps.sum(dim=0), torch.ones(xs.shape[0]), atol=1e-6))
        self.assertTrue(torch.equal(new.lambda_steps[-1], torch.ones(xs.shape[0])))

    def test_post_train_eval_writes_default_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = RunConfig(
                seed=11,
                device="cpu",
                task=TaskConfig(name="copy", n_dims=6, train_length_start=2, train_length_end=3, test_length=2),
                model=ModelConfig(n_positions=32, n_embd=16, n_layer=1, n_head=4, use_wpe=False, use_rope=False),
                trainer=TrainerConfig(
                    name="fixed_loop",
                    train_steps=1,
                    batch_size=2,
                    eval_batch_size=2,
                    eval_every=1,
                ),
                eval=EvalConfig(run_after_train=True, after_train_checkpoint="best"),
                logging=LoggingConfig(output_root=tmp),
            )

            run_dir = Path(run_training(config, console_log=False))
            eval_dir = run_dir / "eval" / "default_best"
            metrics_json = eval_dir / "eval_metrics.json"
            metrics_csv = eval_dir / "eval_metrics.csv"

            self.assertTrue(metrics_json.exists())
            self.assertTrue(metrics_csv.exists())
            row = json.loads(metrics_json.read_text(encoding="utf-8"))
            self.assertEqual(row["checkpoint_tag"], "best")
            self.assertEqual(row["task"], "copy")
            self.assertIn("accuracy", row)


if __name__ == "__main__":
    unittest.main()
