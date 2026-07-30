#!/usr/bin/env python3
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = PACKAGE_ROOT / "scripts/train.py"
CONFIG_PATH = PACKAGE_ROOT / "configs/train_stage_B_continuation_3gpu_no_replacement.json"
MODEL_CONFIG_PATH = PACKAGE_ROOT / "configs/model_large.json"


def load_train_module():
    spec = importlib.util.spec_from_file_location("cropgenome_train", TRAIN_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class StageBContinuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = load_train_module()

    def test_frozen_three_gpu_protocol(self):
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        self.assertEqual(config["micro_batch_size"], 4)
        self.assertEqual(config["grad_accum_steps"], 3)
        self.assertEqual(config["expected_world_size"], 3)
        self.assertEqual(4 * 3 * 3, 36)
        self.assertFalse(config["ddp_static_graph"])
        self.assertTrue(config["ddp_gradient_as_bucket_view"])
        self.assertEqual(config["sampler_mode"], "global_no_replacement")
        self.assertEqual(config["save_every"], 500)
        self.assertEqual(config["eval_every"], 1000)
        self.assertFalse(config["early_stopping_enabled"])
        self.assertEqual(config["resume_contract"]["checkpoint_step"], 14000)
        self.assertEqual(config["launch_gate"]["a100_gpu_indices"], [0, 1, 2])

    def test_model_parameter_count_matches_public_summary(self):
        config = json.loads(MODEL_CONFIG_PATH.read_text(encoding="utf-8"))
        model = self.train.CropGenomeFM(config)
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 369505287)

    def test_no_replacement_plan_is_unique_and_rank_disjoint(self):
        references = {
            4096: np.arange(0, 12, dtype=np.uint64),
            8192: np.arange(100, 106, dtype=np.uint64),
        }
        first = self.train.NoReplacementBatchPlan(references, global_batch_size=6, seed=19)
        second = self.train.NoReplacementBatchPlan(references, global_batch_size=6, seed=19)
        observed = []
        for position in range(first.total_batches):
            global_batch = first.global_batch(position)
            self.assertEqual(global_batch.tolist(), second.global_batch(position).tolist())
            local_batches = [first.local_batch(position, rank, world_size=3) for rank in range(3)]
            self.assertEqual([len(batch) for batch in local_batches], [2, 2, 2])
            self.assertEqual(np.concatenate(local_batches).tolist(), global_batch.tolist())
            observed.extend(global_batch.tolist())
        self.assertEqual(len(observed), len(set(observed)))
        self.assertEqual(set(observed), set(range(12)) | set(range(100, 106)))

    def test_no_replacement_stream_resumes_exact_position(self):
        class FakeDataset:
            def reference_catalog_by_length(self):
                return {
                    4: np.arange(0, 12, dtype=np.uint64),
                    8: np.arange(100, 112, dtype=np.uint64),
                }

            def materialize_reference(self, reference, rng):
                reference = int(reference)
                length = 4 if reference < 100 else 8
                return torch.full((length,), reference, dtype=torch.long), reference % 7

        dataset = FakeDataset()
        stream = self.train.NoReplacementBatchStream(
            dataset, micro_batch_size=2, rank=0, world_size=3, seed=23
        )
        next(stream)
        state = stream.state_dict()
        expected = next(stream)
        resumed = self.train.NoReplacementBatchStream(
            dataset, micro_batch_size=2, rank=0, world_size=3, seed=23, state=state
        )
        actual = next(resumed)
        self.assertEqual(
            [int(sequence[0]) for sequence, _ in actual],
            [int(sequence[0]) for sequence, _ in expected],
        )
        self.assertEqual(resumed.state_dict()["batch_position"], 2)

    def test_ddp_accumulation_syncs_only_final_microbatch(self):
        model = mock.Mock()
        model.no_sync.side_effect = lambda: self.train.nullcontext()
        for microbatch_index in range(3):
            with self.train.gradient_accumulation_sync_context(
                model,
                ddp_enabled=True,
                microbatch_index=microbatch_index,
                accumulation_steps=3,
            ):
                pass
        self.assertEqual(model.no_sync.call_count, 2)

    def test_step_rank_rng_is_reproducible_and_rank_specific(self):
        self.train.seed_training_step(base_seed=20260609, step=14001, rank=0)
        first = torch.rand(8)
        self.train.seed_training_step(base_seed=20260609, step=14001, rank=0)
        replayed = torch.rand(8)
        self.train.seed_training_step(base_seed=20260609, step=14001, rank=1)
        other_rank = torch.rand(8)
        self.assertTrue(torch.equal(first, replayed))
        self.assertFalse(torch.equal(first, other_rank))


if __name__ == "__main__":
    unittest.main()
