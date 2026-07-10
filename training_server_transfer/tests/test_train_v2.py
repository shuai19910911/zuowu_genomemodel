import csv
import gzip
import importlib.util
import json
import os
import signal
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = PROJECT_ROOT / "training_server_transfer" / "scripts" / "train.py"
CHECK_PACKAGE_PATH = PROJECT_ROOT / "training_server_transfer" / "scripts" / "check_package.py"
RUN_STAGE_PATH = PROJECT_ROOT / "training_server_transfer" / "scripts" / "run_stage.sh"


def load_train_module():
    spec = importlib.util.spec_from_file_location("cropgenome_train", TRAIN_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_check_package_module():
    spec = importlib.util.spec_from_file_location("cropgenome_check_package", CHECK_PACKAGE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CropGenomeFMV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = load_train_module()
        cls.check_package = load_check_package_module()

    def tiny_cfg(self):
        return {
            "model_name": "tiny-v2-test",
            "backend_priority": ["hyena_lite"],
            "d_model": 16,
            "n_layers": 2,
            "expand": 1,
            "mlp_ratio": 1.25,
            "dropout": 0.0,
            "conv_kernel": 5,
            "attention_every": 0,
            "rc_equivariant": True,
            "gradient_checkpointing": False,
            "mlm_probability": 0.25,
            "mlm_loss_weight": 0.7,
            "rc_consistency_weight": 0.03,
            "region_classification_weight": 0.11,
            "region_label_smoothing": 0.0,
            "rc_selection_weight": 0.07,
            "region_labels": ["background", "coding", "gene_body", "promoter", "splice", "tes", "utr"],
        }

    def make_batch(self):
        return [
            (torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.long), 1),
            (torch.tensor([3, 2, 1, 0, 3, 2, 1, 0], dtype=torch.long), 4),
        ]

    def test_package_preflight_reports_training_runtime(self):
        runtime = self.check_package.check_training_runtime()

        self.assertEqual(Path(runtime["python"]).resolve(), Path(os.sys.executable).resolve())
        self.assertIn("numpy", runtime)
        self.assertIn("torch", runtime)
        self.assertIn("cuda_available", runtime)

    def test_safe_relative_path_rejects_current_directory(self):
        with self.assertRaisesRegex(ValueError, "unsafe"):
            self.train.safe_relative_path(".", "output_dir")

    def test_collate_returns_region_labels_and_masks(self):
        batch = [
            (torch.tensor([0, 1, 2, 3, 0, 1], dtype=torch.long), 1),
            (torch.tensor([3, 2, 1, 0], dtype=torch.long), 4),
        ]
        input_ids, labels, attention_mask, region_labels = self.train.collate_and_mask(
            batch,
            mask_prob=0.25,
            force_mask_per_sequence=True,
        )
        self.assertEqual(tuple(input_ids.shape), (2, 6))
        self.assertEqual(tuple(labels.shape), (2, 6))
        self.assertEqual(tuple(attention_mask.shape), (2, 6))
        self.assertEqual(region_labels.tolist(), [1, 4])
        self.assertTrue((labels != -100).any(dim=1).all().item())

    def test_n_and_pad_are_not_masked_for_mlm(self):
        batch = [
            (torch.tensor([0, 4, 1], dtype=torch.long), 1),
            (torch.tensor([2], dtype=torch.long), 4),
        ]
        _, labels, _, _ = self.train.collate_and_mask(
            batch,
            mask_prob=1.0,
            force_mask_per_sequence=True,
        )
        self.assertEqual(labels[0].tolist(), [0, -100, 1])
        self.assertEqual(labels[1].tolist(), [2, -100, -100])

    def test_region_bucket_window_index_mapping(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "windows.tsv.gz"
            with gzip.open(path, "wt", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(
                    fh,
                    delimiter="\t",
                    fieldnames=["split", "offset", "length", "region_bucket"],
                )
                writer.writeheader()
                writer.writerow({"split": "train", "offset": 0, "length": 8, "region_bucket": "coding"})
                writer.writerow({"split": "val", "offset": 8, "length": 8, "region_bucket": "promoter"})
                writer.writerow({"split": "train", "offset": 16, "length": 8, "region_bucket": "unknown_bucket"})
            label_map = self.train.build_region_label_map(["background", "coding", "promoter"])
            offsets, lengths, region_labels = self.train.load_window_index(path, "train", label_map)
        self.assertEqual(offsets.tolist(), [0, 16])
        self.assertEqual(lengths.tolist(), [8, 8])
        self.assertEqual(region_labels.tolist(), [1, self.train.REGION_IGNORE_INDEX])

    def test_stage_window_dataset_yields_region_label_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            stage_dir = root / "inputs" / "Stage_B"
            stage_dir.mkdir(parents=True)
            (stage_dir / "shard_000001.input_ids.bin").write_bytes(bytes([0, 1, 2, 3, 0, 1, 2, 3]))
            with gzip.open(stage_dir / "shard_000001.windows.tsv.gz", "wt", encoding="utf-8", newline="") as fh:
                fields = ["stage", "split", "assembly_id", "species", "genus", "contig_id", "start0", "end0", "context", "region_bucket", "source_region_type", "shard", "offset", "length"]
                writer = csv.DictWriter(fh, delimiter="\t", fieldnames=fields)
                writer.writeheader()
                writer.writerow({"stage": "Stage_B", "split": "train", "assembly_id": "ASM", "species": "S", "genus": "G", "contig_id": "c", "start0": 0, "end0": 8, "context": 8, "region_bucket": "splice", "source_region_type": "splice", "shard": "shard_000001", "offset": 0, "length": 8})
            with (stage_dir / "manifest.tsv").open("w", encoding="utf-8", newline="") as fh:
                fields = ["shard", "input_ids", "windows", "tokens", "windows_count", "input_sha256", "windows_sha256"]
                writer = csv.DictWriter(fh, delimiter="\t", fieldnames=fields)
                writer.writeheader()
                writer.writerow({"shard": "shard_000001", "input_ids": "shard_000001.input_ids.bin", "windows": "shard_000001.windows.tsv.gz", "tokens": 8, "windows_count": 1, "input_sha256": "NA", "windows_sha256": "NA"})
            dataset = self.train.StageWindowDataset(
                root,
                "inputs/Stage_B",
                "train",
                seed=123,
                rc_prob=0.0,
                region_label_map=self.train.build_region_label_map(self.tiny_cfg()["region_labels"]),
                deterministic=True,
            )
            seq, region_label = next(iter(dataset))
        self.assertEqual(seq.tolist(), [0, 1, 2, 3, 0, 1, 2, 3])
        self.assertEqual(region_label, self.tiny_cfg()["region_labels"].index("splice"))

    def test_multitask_loss_is_finite_and_reports_region_metrics(self):
        cfg = self.tiny_cfg()
        model = self.train.CropGenomeFM(cfg)
        input_ids, labels, attention_mask, region_labels = self.train.collate_and_mask(
            self.make_batch(),
            mask_prob=0.25,
            force_mask_per_sequence=True,
        )
        loss, metrics = self.train.pretraining_loss(
            model,
            input_ids,
            labels,
            attention_mask,
            rc_weight=cfg["rc_consistency_weight"],
            region_labels=region_labels,
            region_weight=cfg["region_classification_weight"],
            mlm_weight=cfg["mlm_loss_weight"],
            region_label_smoothing=cfg["region_label_smoothing"],
            rc_selection_weight=cfg["rc_selection_weight"],
        )
        self.assertTrue(torch.isfinite(loss).item())
        for key in ["mlm_loss", "rc_loss", "region_loss", "region_acc", "region_valid_count", "selection_loss"]:
            self.assertIn(key, metrics)
            self.assertTrue(torch.isfinite(metrics[key]).item(), key)
        expected_selection = metrics["mlm_loss"] + cfg["rc_selection_weight"] * metrics["rc_loss"]
        self.assertTrue(torch.allclose(metrics["selection_loss"], expected_selection, atol=1e-6))
        expected_total = cfg["mlm_loss_weight"] * metrics["mlm_loss"] + cfg["rc_consistency_weight"] * metrics["rc_loss"] + cfg["region_classification_weight"] * metrics["region_loss"]
        self.assertTrue(torch.allclose(loss.detach(), expected_total, atol=1e-6))

    def test_pretraining_loss_exposes_component_losses_and_counts(self):
        cfg = self.tiny_cfg()
        model = self.train.CropGenomeFM(cfg)
        input_ids, labels, attention_mask, region_labels = self.train.collate_and_mask(
            self.make_batch(),
            mask_prob=0.25,
            force_mask_per_sequence=True,
        )
        loss, metrics, components, counts = self.train.pretraining_loss(
            model,
            input_ids,
            labels,
            attention_mask,
            rc_weight=cfg["rc_consistency_weight"],
            region_labels=region_labels,
            region_weight=cfg["region_classification_weight"],
            mlm_weight=cfg["mlm_loss_weight"],
            region_label_smoothing=cfg["region_label_smoothing"],
            rc_selection_weight=cfg["rc_selection_weight"],
            return_components=True,
        )

        self.assertEqual(set(components), {"mlm", "rc", "region"})
        self.assertTrue(all(value.requires_grad for value in components.values()))
        self.assertEqual(counts["mlm"], int(labels.ne(-100).sum()))
        self.assertEqual(counts["rc"], int(attention_mask.sum()))
        self.assertEqual(counts["region"], int(region_labels.ne(self.train.REGION_IGNORE_INDEX).sum()))
        self.assertTrue(torch.isfinite(loss).item())
        self.assertIn("selection_loss", metrics)

    def test_no_region_config_has_no_region_head_and_no_region_loss(self):
        cfg = self.tiny_cfg()
        cfg["region_classification_weight"] = 0.0
        model = self.train.CropGenomeFM(cfg)
        self.assertIsNone(model.region_head)
        input_ids, labels, attention_mask, region_labels = self.train.collate_and_mask(
            self.make_batch(),
            mask_prob=0.25,
            force_mask_per_sequence=True,
        )
        loss, metrics = self.train.pretraining_loss(
            model,
            input_ids,
            labels,
            attention_mask,
            rc_weight=cfg["rc_consistency_weight"],
            region_labels=region_labels,
            region_weight=0.0,
            mlm_weight=cfg["mlm_loss_weight"],
            rc_selection_weight=cfg["rc_selection_weight"],
        )
        expected_total = cfg["mlm_loss_weight"] * metrics["mlm_loss"] + cfg["rc_consistency_weight"] * metrics["rc_loss"]
        self.assertTrue(torch.allclose(loss.detach(), expected_total, atol=1e-6))
        self.assertEqual(float(metrics["region_loss"]), 0.0)

    def test_all_n_batch_has_finite_zero_mlm_loss(self):
        cfg = self.tiny_cfg()
        model = self.train.CropGenomeFM(cfg)
        batch = [(torch.tensor([4, 4, 4], dtype=torch.long), 1)]
        input_ids, labels, attention_mask, region_labels = self.train.collate_and_mask(
            batch,
            mask_prob=1.0,
            force_mask_per_sequence=True,
        )
        loss, metrics = self.train.pretraining_loss(
            model,
            input_ids,
            labels,
            attention_mask,
            rc_weight=0.0,
            region_labels=region_labels,
            region_weight=0.0,
            mlm_weight=1.0,
        )
        self.assertTrue(torch.isfinite(loss).item())
        self.assertEqual(float(metrics["mlm_loss"]), 0.0)

    def test_update_best_tracking_ignores_region_loss(self):
        best_selection, best_step, stale_evals, selection, improved = self.train.update_best_tracking(
            {"loss": 0.1, "mlm_loss": 0.50, "rc_loss": 10.0, "region_loss": 0.01, "selection_loss": 1.20},
            step=1000,
            best_selection=1.00,
            best_step=500,
            stale_evals=0,
            min_delta=0.002,
        )
        self.assertFalse(improved)
        self.assertEqual(best_selection, 1.00)
        self.assertEqual(best_step, 500)
        self.assertEqual(stale_evals, 1)
        best_selection, best_step, stale_evals, selection, improved = self.train.update_best_tracking(
            {"loss": 10.0, "mlm_loss": 9.00, "rc_loss": 0.0, "region_loss": 9.0, "selection_loss": 0.90},
            step=2000,
            best_selection=1.00,
            best_step=500,
            stale_evals=1,
            min_delta=0.002,
        )
        self.assertTrue(improved)
        self.assertEqual(best_selection, 0.90)
        self.assertEqual(best_step, 2000)
        self.assertEqual(stale_evals, 0)

    def test_safe_paths_and_checkpoint_loading(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with self.assertRaises(ValueError):
                self.train.resolve_under(root, "../escape", "output_dir")
            with self.assertRaises(ValueError):
                self.train.resolve_under(root, "/tmp/escape", "resume")
            ckpt_path = root / "checkpoint.pt"
            torch.save({"model": {}, "step": 0}, ckpt_path)
            loaded = self.train.safe_torch_load_checkpoint(ckpt_path)
        self.assertEqual(loaded["step"], 0)

    def test_partial_state_load_skips_new_or_mismatched_keys(self):
        cfg = self.tiny_cfg()
        model = self.train.CropGenomeFM(cfg)
        state = model.state_dict()
        legacy_state = {
            key: value.clone()
            for key, value in state.items()
            if not key.startswith("region_head")
        }
        first_key = next(iter(legacy_state))
        legacy_state[first_key] = legacy_state[first_key][:-1]
        report = self.train.load_model_state_safely(
            model,
            legacy_state,
            allow_partial=True,
        )
        self.assertIn(first_key, report["skipped_shape_mismatch"])
        self.assertTrue(any(key.startswith("region_head") for key in report["missing_keys"]))

    def test_partial_state_load_strips_module_prefix(self):
        cfg = self.tiny_cfg()
        model = self.train.CropGenomeFM(cfg)
        state = {f"module.{key}": value.clone() for key, value in model.state_dict().items()}
        report = self.train.load_model_state_safely(model, state, allow_partial=True)
        self.assertFalse(report["unexpected_keys"])
        self.assertFalse(report["skipped_shape_mismatch"])
        self.assertGreater(len(report["loaded_keys"]), 0)

    def test_resume_restores_best_tracking_from_checkpoint(self):
        ckpt = {"best_selection_loss": 1.2375, "best_step": 1000, "stale_evals": 2}
        with tempfile.TemporaryDirectory() as tmpdir:
            best_selection, best_step, stale_evals, source = self.train.restore_best_tracking_from_resume(
                ckpt,
                Path(tmpdir) / "missing_early_state.json",
                start_step=1000,
            )
        self.assertEqual(source, "checkpoint")
        self.assertEqual(best_step, 1000)
        self.assertEqual(stale_evals, 2)
        self.assertAlmostEqual(best_selection, 1.2375)

    def test_resume_restores_best_tracking_from_early_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "early_stopping_state.json"
            state_path.write_text(
                '{"step": 1000, "best_step": 1000, "best_selection_loss": 1.25, "stale_evals": 1}',
                encoding="utf-8",
            )
            best_selection, best_step, stale_evals, source = self.train.restore_best_tracking_from_resume(
                {"step": 1000},
                state_path,
                start_step=1000,
            )
        self.assertEqual(source, "early_stopping_state")
        self.assertEqual(best_step, 1000)
        self.assertEqual(stale_evals, 1)
        self.assertAlmostEqual(best_selection, 1.25)

    def test_stage_reset_starts_fresh_best_tracking(self):
        ckpt = {"best_selection_loss": 1.1, "best_step": 14000, "stale_evals": 2}
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "early_stopping_state.json"
            state_path.write_text(
                '{"step": 14000, "best_step": 14000, "best_selection_loss": 1.1, "stale_evals": 2}',
                encoding="utf-8",
            )
            best_selection, best_step, stale_evals, source = self.train.restore_best_tracking_from_resume(
                ckpt,
                state_path,
                start_step=0,
                reset_tracking=True,
            )

        self.assertEqual(source, "fresh")
        self.assertEqual(best_step, 0)
        self.assertEqual(stale_evals, 0)
        self.assertEqual(best_selection, float("inf"))

    def test_dry_run_batch_selection_requires_exact_sequence_length(self):
        def batch(length):
            shape = (1, length)
            return (
                torch.zeros(shape, dtype=torch.long),
                torch.zeros(shape, dtype=torch.long),
                torch.ones(shape, dtype=torch.bool),
                torch.zeros(1, dtype=torch.long),
            )

        selected, scanned = self.train.next_dry_run_batch(
            iter([batch(8192), batch(65536)]),
            required_sequence_length=65536,
            max_batches=2,
        )

        self.assertEqual(tuple(selected[0].shape), (1, 65536))
        self.assertEqual(scanned, 2)

    def test_dilated_attention_connects_distant_chunks(self):
        block = self.train.LocalSelfAttentionBlock(
            dim=4,
            num_heads=1,
            chunk_size=4,
            dropout=0.0,
            dilation=4,
        )
        source = torch.randn(1, 16, 4, requires_grad=True)
        output = block(source, torch.ones(1, 16, dtype=torch.bool))
        output[0, 0].sum().backward()

        self.assertGreater(float(source.grad[0, 12].abs().sum()), 0.0)

    def test_attention_dilation_schedule_keeps_checkpoint_keys_compatible(self):
        base_cfg = self.tiny_cfg()
        base_cfg.update({"n_layers": 3, "attention_every": 1, "attention_heads": 1, "attention_chunk_size": 4})
        base_model = self.train.CropGenomeFM(base_cfg)
        long_cfg = dict(base_cfg)
        long_cfg["attention_dilations"] = [1, 2, 4]
        long_model = self.train.CropGenomeFM(long_cfg)
        dilations = [
            layer.dilation
            for layer in long_model.layers
            if isinstance(layer, self.train.LocalSelfAttentionBlock)
        ]

        self.assertEqual(dilations, [1, 2, 4])
        self.assertEqual(set(base_model.state_dict()), set(long_model.state_dict()))
        long_model.load_state_dict(base_model.state_dict(), strict=True)

    def test_pretraining_objective_is_normalized_by_component_counts(self):
        cfg = {
            "mlm_loss_weight": 1.0,
            "rc_consistency_weight": 0.5,
            "region_classification_weight": 0.25,
        }
        global_counts = {"mlm": 68, "rc": 68, "region": 2}
        first = self.train.weighted_pretraining_objective(
            {"mlm": torch.tensor(1.0), "rc": torch.tensor(2.0), "region": torch.tensor(4.0)},
            {"mlm": 4, "rc": 4, "region": 1},
            global_counts,
            cfg,
        )
        second = self.train.weighted_pretraining_objective(
            {"mlm": torch.tensor(3.0), "rc": torch.tensor(6.0), "region": torch.tensor(8.0)},
            {"mlm": 64, "rc": 64, "region": 1},
            global_counts,
            cfg,
        )
        expected = (1.0 * 4 + 3.0 * 64) / 68
        expected += 0.5 * (2.0 * 4 + 6.0 * 64) / 68
        expected += 0.25 * (4.0 + 8.0) / 2

        self.assertAlmostEqual(float(first + second), expected, places=6)

    def test_metric_summary_uses_token_and_region_denominators(self):
        cfg = {
            "mlm_loss_weight": 1.0,
            "rc_consistency_weight": 0.5,
            "region_classification_weight": 0.25,
            "rc_selection_weight": 0.1,
        }
        summary = self.train.summarize_pretraining_metrics(
            {"mlm": torch.tensor(196.0), "rc": torch.tensor(392.0), "region": torch.tensor(12.0), "region_correct": torch.tensor(1.0)},
            {"mlm": torch.tensor(68), "rc": torch.tensor(68), "region": torch.tensor(2)},
            cfg,
        )

        self.assertAlmostEqual(summary["mlm_loss"], 196.0 / 68.0)
        self.assertAlmostEqual(summary["rc_loss"], 392.0 / 68.0)
        self.assertAlmostEqual(summary["region_loss"], 6.0)
        self.assertAlmostEqual(summary["region_acc"], 0.5)
        self.assertAlmostEqual(summary["selection_loss"], 196.0 / 68.0 + 0.1 * 392.0 / 68.0)

    def test_deterministic_window_order_is_reproducible_and_without_replacement(self):
        first = self.train.deterministic_window_order([2, 0, 3], seed=17)
        second = self.train.deterministic_window_order([2, 0, 3], seed=17)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 5)
        self.assertEqual(set(first), {(0, 0), (0, 1), (2, 0), (2, 1), (2, 2)})

    def test_component_counts_are_summed_across_accumulation_batches(self):
        batches = [
            (
                torch.zeros((1, 4), dtype=torch.long),
                torch.tensor([[0, -100, 1, -100]], dtype=torch.long),
                torch.ones((1, 4), dtype=torch.bool),
                torch.tensor([2], dtype=torch.long),
            ),
            (
                torch.zeros((1, 8), dtype=torch.long),
                torch.tensor([[0, 1, 2, 3, -100, -100, -100, -100]], dtype=torch.long),
                torch.ones((1, 8), dtype=torch.bool),
                torch.tensor([self.train.REGION_IGNORE_INDEX], dtype=torch.long),
            ),
        ]

        counts = self.train.count_pretraining_components(batches, torch.device("cpu"))

        self.assertEqual(counts.tolist(), [6.0, 12.0, 1.0])

    def test_stop_controller_records_termination_signal(self):
        controller = self.train.StopController()

        controller.request(signal.SIGTERM, None)

        self.assertTrue(controller.requested)
        self.assertEqual(controller.signal_number, signal.SIGTERM)

    def test_resume_policy_distinguishes_warmstart_from_exact_resume(self):
        cfg = {
            "allow_partial_resume": True,
            "resume_optimizer": False,
            "reset_step_on_resume": True,
        }

        warmstart = self.train.resolve_resume_policy(cfg, "warmstart")
        exact = self.train.resolve_resume_policy(cfg, "exact")

        self.assertEqual(warmstart, {"allow_partial": False, "resume_optimizer": False, "reset_step": True})
        self.assertEqual(exact, {"allow_partial": False, "resume_optimizer": True, "reset_step": False})

    def test_checkpoint_save_is_atomic_and_preserves_previous_file_on_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            target = checkpoint_dir / "checkpoint_best.pt"
            target.write_bytes(b"previous-checkpoint")
            model = torch.nn.Linear(2, 2)
            optimizer = torch.optim.AdamW(model.parameters())

            def partial_write_then_fail(_payload, path):
                Path(path).write_bytes(b"partial")
                raise RuntimeError("simulated interrupted save")

            with mock.patch.object(self.train.torch, "save", side_effect=partial_write_then_fail):
                with self.assertRaisesRegex(RuntimeError, "simulated interrupted save"):
                    self.train.save_checkpoint(
                        checkpoint_dir,
                        model,
                        optimizer,
                        step=3,
                        cfg={},
                        model_cfg={},
                        rank=0,
                        filename="checkpoint_best.pt",
                    )

            self.assertEqual(target.read_bytes(), b"previous-checkpoint")
            self.assertEqual(list(checkpoint_dir.glob("*.tmp")), [])

    def test_stage_c1_launcher_uses_locked_checkpoint_and_supports_dry_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            package_root = tmp / "training_server_transfer"
            scripts_dir = package_root / "scripts"
            scripts_dir.mkdir(parents=True)
            launcher = scripts_dir / "run_stage.sh"
            launcher.write_text(RUN_STAGE_PATH.read_text(encoding="utf-8"), encoding="utf-8")
            checkpoint = package_root / "runs" / "Stage_B_cropgenome_fm_v2_stable" / "checkpoints" / "checkpoint_stage_B_8k_final.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.touch()
            command_log = tmp / "commands.log"
            fake_mamba = tmp / "mamba"
            fake_mamba.write_text(
                '#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$COMMAND_LOG"\n',
                encoding="utf-8",
            )
            fake_mamba.chmod(0o755)
            fake_sha256sum = tmp / "sha256sum"
            fake_sha256sum.write_text(
                '#!/usr/bin/env bash\nprintf "c81bce39ec448845e929e755530bc7023a345cca42234ff7fb776f5f39c83fed  %s\\n" "$1"\n',
                encoding="utf-8",
            )
            fake_sha256sum.chmod(0o755)
            env = os.environ.copy()
            env.update({
                "COMMAND_LOG": str(command_log),
                "DRY_RUN": "1",
                "STAGE_C1_RESUME": "runs/wrong_checkpoint.pt",
                "PATH": f"{tmp}:{env['PATH']}",
            })
            subprocess.run(
                ["bash", str(launcher), "Stage_C1", "1"],
                cwd=package_root,
                env=env,
                check=True,
            )
            commands = command_log.read_text(encoding="utf-8").splitlines()

        self.assertEqual(len(commands), 2)
        self.assertIn(
            "--resume runs/Stage_B_cropgenome_fm_v2_stable/checkpoints/checkpoint_stage_B_8k_final.pt",
            commands[1],
        )
        self.assertIn("--model-config configs/model_stage_C1_64k.json", commands[1])
        self.assertIn("--resume-mode warmstart", commands[1])
        self.assertTrue(commands[1].endswith("--dry-run"))

    def test_stage_c1_launcher_prefers_explicit_environment_prefix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            package_root = tmp / "training_server_transfer"
            scripts_dir = package_root / "scripts"
            scripts_dir.mkdir(parents=True)
            launcher = scripts_dir / "run_stage.sh"
            launcher.write_text(RUN_STAGE_PATH.read_text(encoding="utf-8"), encoding="utf-8")
            checkpoint = package_root / "runs" / "Stage_B_cropgenome_fm_v2_stable" / "checkpoints" / "checkpoint_stage_B_8k_final.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.touch()
            command_log = tmp / "commands.log"
            env_prefix = tmp / "env"
            env_bin = env_prefix / "bin"
            env_bin.mkdir(parents=True)
            fake_python = env_bin / "python"
            fake_python.write_text(
                '#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$COMMAND_LOG"\n',
                encoding="utf-8",
            )
            fake_python.chmod(0o755)
            fake_sha256sum = tmp / "sha256sum"
            fake_sha256sum.write_text(
                '#!/usr/bin/env bash\nprintf "c81bce39ec448845e929e755530bc7023a345cca42234ff7fb776f5f39c83fed  %s\\n" "$1"\n',
                encoding="utf-8",
            )
            fake_sha256sum.chmod(0o755)
            env = os.environ.copy()
            env.update({
                "COMMAND_LOG": str(command_log),
                "CONDA_ENV_PREFIX": str(env_prefix),
                "DRY_RUN": "1",
                "PATH": f"{tmp}:{env['PATH']}",
            })
            subprocess.run(
                ["bash", str(launcher), "Stage_C1", "1"],
                cwd=package_root,
                env=env,
                check=True,
            )
            commands = command_log.read_text(encoding="utf-8").splitlines()

        self.assertEqual(len(commands), 2)
        self.assertTrue(commands[0].startswith("scripts/check_package.py "))
        self.assertTrue(commands[1].startswith("scripts/train.py "))

    def test_stage_c1_launcher_rejects_checkpoint_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            package_root = tmp / "training_server_transfer"
            scripts_dir = package_root / "scripts"
            scripts_dir.mkdir(parents=True)
            launcher = scripts_dir / "run_stage.sh"
            launcher.write_text(RUN_STAGE_PATH.read_text(encoding="utf-8"), encoding="utf-8")
            checkpoint = package_root / "runs" / "Stage_B_cropgenome_fm_v2_stable" / "checkpoints" / "checkpoint_stage_B_8k_final.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"wrong checkpoint")
            command_log = tmp / "commands.log"
            fake_mamba = tmp / "mamba"
            fake_mamba.write_text(
                '#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$COMMAND_LOG"\n',
                encoding="utf-8",
            )
            fake_mamba.chmod(0o755)
            fake_sha256sum = tmp / "sha256sum"
            fake_sha256sum.write_text(
                '#!/usr/bin/env bash\nprintf "deadbeef  %s\\n" "$1"\n',
                encoding="utf-8",
            )
            fake_sha256sum.chmod(0o755)
            env = os.environ.copy()
            env.update({
                "COMMAND_LOG": str(command_log),
                "PATH": f"{tmp}:{env['PATH']}",
            })
            result = subprocess.run(
                ["bash", str(launcher), "Stage_C1", "1"],
                cwd=package_root,
                env=env,
                check=False,
            )

        self.assertEqual(result.returncode, 4)
        self.assertFalse(command_log.exists())

    def test_stage_c1_launcher_rejects_duplicate_live_pid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            package_root = Path(tmpdir) / "training_server_transfer"
            scripts_dir = package_root / "scripts"
            scripts_dir.mkdir(parents=True)
            launcher = scripts_dir / "run_stage.sh"
            launcher.write_text(RUN_STAGE_PATH.read_text(encoding="utf-8"), encoding="utf-8")
            run_dir = package_root / "runs" / "Stage_C1"
            run_dir.mkdir(parents=True)
            (run_dir / "launcher.pid").write_text(f"{os.getpid()}\n", encoding="utf-8")

            result = subprocess.run(
                ["bash", str(launcher), "Stage_C1", "1"],
                cwd=package_root,
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(result.returncode, 6)
        self.assertIn("already running", result.stderr)

    def test_stage_c1_config_starts_a_fresh_early_stopped_stage(self):
        config_path = PROJECT_ROOT / "training_server_transfer" / "configs" / "train_stage_C1.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertEqual(config["dry_run_sequence_length"], 65536)
        self.assertGreaterEqual(config["eval_batches"], 256)
        self.assertLessEqual(config["save_every"], 500)
        self.assertFalse(config["allow_partial_resume"])
        self.assertFalse(config["resume_optimizer"])
        self.assertTrue(config["reset_step_on_resume"])
        self.assertTrue(config["early_stopping_enabled"])
        self.assertGreater(config["early_stopping_min_steps"], config["warmup_steps"])
        self.assertGreater(config["early_stopping_patience_evals"], 0)
        self.assertGreater(config["early_stopping_min_delta"], 0.0)

    def test_stage_c1_model_config_uses_full_dilation_schedule(self):
        config_path = PROJECT_ROOT / "training_server_transfer" / "configs" / "model_stage_C1_64k.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertEqual(config["attention_dilations"], [1, 2, 4, 8, 16, 32, 64, 128])
        self.assertEqual(config["attention_chunk_size"], 512)
        self.assertEqual(config["n_layers"] // config["attention_every"], len(config["attention_dilations"]))


if __name__ == "__main__":
    unittest.main()
