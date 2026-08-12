import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
MODULE_PATH = ROOT / "scripts" / "extract_cropgenome_bench_v1_embeddings.py"


class FakeCropModel:
    def __call__(self, input_ids, attention_mask, return_aux=False):
        hidden = input_ids.float().unsqueeze(-1).repeat(1, 1, 3)
        return None, {"hidden": hidden}


class FakeTrainModule:
    class CropGenomeFM:
        @staticmethod
        def sequence_pool(hidden, attention_mask):
            weights = attention_mask.to(hidden.dtype).unsqueeze(-1)
            return (hidden * weights).sum(1) / weights.sum(1).clamp_min(1)

    @staticmethod
    def reverse_complement_tokens(input_ids):
        return torch.flip(input_ids, dims=(1,))


def test_cropgenome_extract_task_streams_float16_embeddings_to_disk(tmp_path):
    spec = importlib.util.spec_from_file_location("crop_embedding_streaming", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "tokens.bin"
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

    sink, rc_ids, _ = module.extract_task(
        FakeCropModel(), FakeTrainModule(), samples,
        batch_size=1, context=2, device=torch.device("cpu"),
        include_rc_test=True, workspace=tmp_path / "parts",
    )
    try:
        assert isinstance(sink.embeddings, np.memmap)
        assert sink.embeddings.dtype == np.float16
        assert sink.forward_written == 2
        assert sink.rc_written == 1
        np.testing.assert_array_equal(rc_ids, np.asarray(["test"]))
    finally:
        sink.cleanup()
