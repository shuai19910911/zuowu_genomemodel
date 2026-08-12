#!/usr/bin/env python3
import importlib.util
import hashlib
import os
import types
import unittest
from pathlib import Path

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO / "scripts" / "extract_public_dna_embeddings.py"


class FakeTokenizer:
    def __call__(self, sequences, **kwargs):
        self.kwargs = kwargs
        return {"input_ids": [list(range(len(sequence) // 6 + 2)) for sequence in sequences]}


class ForwardTokenizer:
    all_special_ids = []

    def __call__(self, sequences, **kwargs):
        batch = len(sequences)
        return {
            "input_ids": torch.tensor([[1, 2]] * batch),
            "attention_mask": torch.ones((batch, 2), dtype=torch.long),
            "token_type_ids": torch.zeros((batch, 2), dtype=torch.long),
            "special_tokens_mask": torch.zeros((batch, 2), dtype=torch.long),
        }


class StrictForwardModel:
    config = types.SimpleNamespace(d_model=2, rcps=False)

    def forward(self, input_ids, output_hidden_states=False, return_dict=True):
        hidden = input_ids.float().unsqueeze(-1).repeat(1, 1, 2)
        return types.SimpleNamespace(last_hidden_state=hidden)

    __call__ = forward


class PublicTokenizerAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("public_dna_embedding_audit", MODULE_PATH)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_reports_untruncated_lengths_and_overflow_count(self):
        tokenizer = FakeTokenizer()
        audit = self.module.token_length_audit(tokenizer, ["A" * 60, "C" * 120], max_tokens=16)
        self.assertEqual(audit["samples"], 2)
        self.assertEqual(audit["min_tokens"], 12)
        self.assertEqual(audit["max_tokens"], 22)
        self.assertEqual(audit["truncated_samples"], 1)
        self.assertFalse(tokenizer.kwargs["truncation"])
        self.assertFalse(tokenizer.kwargs["padding"])

    def test_embed_strings_only_forwards_supported_tokenizer_fields(self):
        embeddings = self.module.embed_strings(
            StrictForwardModel(),
            ForwardTokenizer(),
            ["AC"],
            torch.device("cpu"),
            max_tokens=2,
        )

        self.assertEqual(embeddings.shape, (1, 2))
        self.assertTrue((embeddings == 1.5).all())

    def test_extract_task_streams_float16_embeddings_to_disk(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "tokens.bin"
            source.write_bytes(bytes([0, 1, 2, 3]))
            samples = [
                {
                    "sample_id": "train", "split": "train",
                    "input_path": str(source), "offset": 0, "length": 2,
                },
                {
                    "sample_id": "test", "split": "test",
                    "input_path": str(source), "offset": 2, "length": 2,
                },
            ]
            sink, rc_ids, _, _, _ = self.module.extract_task(
                StrictForwardModel(), ForwardTokenizer(), samples,
                batch_size=1, context=2, max_tokens=2,
                device=torch.device("cpu"), include_rc_test=True,
                require_no_truncation=True,
                inference_dtype=torch.float32,
                workspace=Path(directory) / "parts",
            )
            try:
                self.assertIsInstance(sink.embeddings, np.memmap)
                self.assertEqual(sink.embeddings.dtype, np.float16)
                self.assertEqual(sink.forward_written, 2)
                self.assertEqual(sink.rc_written, 1)
                np.testing.assert_array_equal(rc_ids, np.asarray(["test"]))
            finally:
                sink.cleanup()

    def test_explicit_float32_inference_and_nonfinite_rejection(self):
        self.assertEqual(
            self.module.resolve_inference_dtype("float32", torch.device("cuda")),
            torch.float32,
        )
        self.assertEqual(
            self.module.resolve_inference_dtype("bfloat16", torch.device("cpu")),
            torch.float32,
        )
        self.module.require_finite_embeddings(np.ones((2, 3), dtype=np.float32), "finite")
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            self.module.require_finite_embeddings(
                np.asarray([[0.0, np.nan]], dtype=np.float32), "bad",
            )

    def test_auto_model_masked_lm_wrapper_is_unwrapped_to_trained_backbone(self):
        backbone = object()

        class FakeWrapperForMaskedLM:
            bert = backbone

        class FakeAutoModel:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return FakeWrapperForMaskedLM()

        class FakeTokenizerFactory:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return object()

        originals = {
            "AutoModel": self.module.AutoModel,
            "AutoTokenizer": self.module.AutoTokenizer,
            "register_model_plugin": self.module.register_model_plugin,
            "clean_model_runtime_cache": self.module.clean_model_runtime_cache,
        }
        self.module.AutoModel = FakeAutoModel
        self.module.AutoTokenizer = FakeTokenizerFactory
        self.module.register_model_plugin = lambda path: types.SimpleNamespace(
            tokenizer=None, force_trust_remote_code=None, config_overrides=None, mode="test",
        )
        self.module.clean_model_runtime_cache = lambda path: None
        try:
            _, loaded, mode = self.module.load_hf_model(
                "snapshot", None, True, torch.float32, True, "auto",
            )
            self.assertIs(loaded, backbone)
            self.assertEqual(mode, "test+auto_mlm_bert_backbone")
        finally:
            for name, value in originals.items():
                setattr(self.module, name, value)

    def test_validates_exact_existing_task_cache_before_resume(self):
        import tempfile

        samples = [
            {
                "sample_id": "train-1", "label": "1", "split": "train",
                "species": "Species one", "assembly_id": "assembly-1",
            },
            {
                "sample_id": "test-1", "label": "0", "split": "test",
                "species": "Species one", "assembly_id": "assembly-1",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task.npz"
            np.savez(
                path,
                embeddings=np.ones((2, 3), dtype=np.float16),
                sample_ids=np.asarray(["train-1", "test-1"]),
                labels=self.module.labels_array(samples),
                splits=np.asarray(["train", "test"]),
                species=np.asarray(["Species one", "Species one"]),
                assemblies=np.asarray(["assembly-1", "assembly-1"]),
                group_ids=np.asarray(["assembly-1", "assembly-1"]),
                rc_embeddings=np.ones((1, 3), dtype=np.float16),
                rc_sample_ids=np.asarray(["test-1"]),
            )
            assert self.module.validate_existing_task_cache(
                path, samples, include_rc_test=True,
            ) == {"embedding_dim": 3, "rc_test_samples": 1}
            np.savez(
                path,
                embeddings=np.ones((2, 3), dtype=np.float16),
                sample_ids=np.asarray(["wrong", "test-1"]),
                labels=self.module.labels_array(samples),
                splits=np.asarray(["train", "test"]),
                species=np.asarray(["Species one", "Species one"]),
                assemblies=np.asarray(["assembly-1", "assembly-1"]),
                group_ids=np.asarray(["assembly-1", "assembly-1"]),
                rc_embeddings=np.ones((1, 3), dtype=np.float16),
                rc_sample_ids=np.asarray(["test-1"]),
            )
            with self.assertRaisesRegex(RuntimeError, "sample_ids"):
                self.module.validate_existing_task_cache(
                    path, samples, include_rc_test=True,
                )

    def test_weight_hash_cache_reuses_stat_bound_verification(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weight.bin"
            cache = Path(directory) / "verification.json"
            path.write_bytes(b"abc")
            expected = hashlib.sha256(b"abc").hexdigest()
            original = self.module.sha256_path
            calls = []

            def counted(value, *args, **kwargs):
                calls.append(Path(value))
                return original(value, *args, **kwargs)

            self.module.sha256_path = counted
            try:
                self.assertEqual(self.module.verified_sha256_path(path, expected, cache), expected)
                self.assertEqual(self.module.verified_sha256_path(path, expected, cache), expected)
                self.assertEqual(calls, [path])
                before = path.stat().st_mtime_ns
                path.write_bytes(b"abd")
                os.utime(path, ns=(before + 1_000_000, before + 1_000_000))
                with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
                    self.module.verified_sha256_path(path, expected, cache)
                self.assertEqual(calls, [path, path])
            finally:
                self.module.sha256_path = original


if __name__ == "__main__":
    unittest.main()
