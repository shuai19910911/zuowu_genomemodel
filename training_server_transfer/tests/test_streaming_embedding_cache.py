import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training_server_transfer"))

from downstream_v4.streaming_embeddings import DiskBackedEmbeddingAccumulator


def test_disk_backed_accumulator_writes_float16_batches_and_atomic_npz(tmp_path):
    workspace = tmp_path / "parts"
    output = tmp_path / "task.npz"
    sink = DiskBackedEmbeddingAccumulator(workspace, forward_rows=3, rc_rows=1)
    sink.append(
        np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        np.asarray([[5.0, 6.0]], dtype=np.float32),
    )
    sink.append(np.asarray([[7.0, 8.0]], dtype=np.float32))

    assert isinstance(sink.embeddings, np.memmap)
    assert isinstance(sink.rc_embeddings, np.memmap)
    assert sink.embeddings.dtype == np.float16
    assert sink.forward_written == 3
    assert sink.rc_written == 1

    sink.save_npz(
        output,
        sample_ids=np.asarray(["a", "b", "c"]),
        labels=np.asarray([0, 1, 0], dtype=np.int8),
        splits=np.asarray(["train", "validation", "test"]),
        species=np.asarray(["plant", "plant", "plant"]),
        assemblies=np.asarray(["asm", "asm", "asm"]),
        group_ids=np.asarray(["g", "g", "g"]),
        rc_sample_ids=np.asarray(["c"]),
    )

    assert output.is_file()
    assert not workspace.exists()
    assert not output.with_name(output.name + ".tmp.npz").exists()
    with np.load(output, allow_pickle=False) as payload:
        assert set(payload.files) == {
            "embeddings", "sample_ids", "labels", "splits", "species",
            "assemblies", "group_ids", "rc_embeddings", "rc_sample_ids",
        }
        np.testing.assert_array_equal(
            payload["embeddings"],
            np.asarray([[1, 2], [3, 4], [7, 8]], dtype=np.float16),
        )
        np.testing.assert_array_equal(
            payload["rc_embeddings"], np.asarray([[5, 6]], dtype=np.float16),
        )


def test_disk_backed_accumulator_rejects_incomplete_or_nonfinite_output(tmp_path):
    incomplete = DiskBackedEmbeddingAccumulator(
        tmp_path / "incomplete", forward_rows=2, rc_rows=0,
    )
    incomplete.append(np.asarray([[1.0, 2.0]], dtype=np.float32))
    with pytest.raises(RuntimeError, match="incomplete"):
        incomplete.save_npz(tmp_path / "incomplete.npz")
    incomplete.cleanup()

    nonfinite = DiskBackedEmbeddingAccumulator(
        tmp_path / "nonfinite", forward_rows=1, rc_rows=0,
    )
    with pytest.raises(RuntimeError, match="non-finite"):
        nonfinite.append(np.asarray([[np.nan, 1.0]], dtype=np.float32))
    nonfinite.cleanup()

    overflow = DiskBackedEmbeddingAccumulator(
        tmp_path / "overflow", forward_rows=1, rc_rows=0,
    )
    with pytest.raises(RuntimeError, match="non-finite.*float16"):
        overflow.append(np.asarray([[1.0e10, 1.0]], dtype=np.float32))
    overflow.cleanup()
