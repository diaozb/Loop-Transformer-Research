from __future__ import annotations

import os
import sys
import unittest

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXP_INFRA = os.path.join(REPO_ROOT, "exp_infra")
SRC = os.path.join(REPO_ROOT, "src")
sys.path.insert(0, EXP_INFRA)
sys.path.insert(0, SRC)

from generate_training_data import (  # noqa: E402
    generate_prompt_matrix_copy,
    generate_prompt_matrix_mod_add,
    generate_prompt_matrix_parity,
)
from ltf.data import generate_copy, generate_mod_add, generate_parity  # noqa: E402


class DataLegacyEquivalenceTest(unittest.TestCase):
    def assert_batch_equal(self, new_batch, legacy_tuple):
        legacy_inputs, legacy_lengths, legacy_targets, legacy_mask = legacy_tuple
        self.assertTrue(torch.equal(new_batch.inputs, legacy_inputs))
        self.assertTrue(torch.equal(new_batch.lengths, legacy_lengths))
        self.assertTrue(torch.equal(new_batch.targets, legacy_targets))
        self.assertTrue(torch.equal(new_batch.mask, legacy_mask))

    def test_parity_matches_legacy(self):
        kwargs = dict(b=16, max_len=22, min_num_digits=1, max_num_digits=21)
        np.random.seed(123)
        legacy = generate_prompt_matrix_parity(**kwargs)
        np.random.seed(123)
        new = generate_parity(
            batch_size=16,
            max_len=22,
            min_length=1,
            max_length_exclusive=21,
        )
        self.assert_batch_equal(new, legacy)

    def test_copy_matches_legacy(self):
        kwargs = dict(b=16, max_len=21, min_num_digits=1, max_num_digits=20, prob_one=0.8)
        np.random.seed(123)
        legacy = generate_prompt_matrix_copy(**kwargs)
        np.random.seed(123)
        new = generate_copy(
            batch_size=16,
            max_len=21,
            min_length=1,
            max_length_exclusive=20,
            prob_one=0.8,
        )
        self.assert_batch_equal(new, legacy)

    def test_mod_add_matches_legacy(self):
        kwargs = dict(b=16, max_len=21, min_num_digits=1, max_num_digits=20, modulus=11)
        np.random.seed(123)
        legacy = generate_prompt_matrix_mod_add(**kwargs)
        np.random.seed(123)
        new = generate_mod_add(
            batch_size=16,
            max_len=21,
            min_length=1,
            max_length_exclusive=20,
            modulus=11,
        )
        self.assert_batch_equal(new, legacy)


if __name__ == "__main__":
    unittest.main()
