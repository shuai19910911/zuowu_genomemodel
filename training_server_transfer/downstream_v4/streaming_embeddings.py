"""Bounded-RAM, disk-backed assembly of canonical embedding cache NPZ files."""

import os
import shutil
from pathlib import Path

import numpy as np


class DiskBackedEmbeddingAccumulator:
    """Write batch embeddings directly to float16 NPY memmaps before atomic NPZ save."""

    RESERVED_FIELDS = {"embeddings", "rc_embeddings"}

    def __init__(self, workspace, forward_rows, rc_rows):
        self.workspace = Path(workspace)
        self.forward_rows = int(forward_rows)
        self.rc_rows = int(rc_rows)
        if self.forward_rows <= 0 or self.rc_rows < 0:
            raise ValueError("embedding row counts must be positive/non-negative")
        if self.workspace.exists():
            shutil.rmtree(str(self.workspace))
        self.workspace.mkdir(parents=True)
        self.embeddings = None
        self.rc_embeddings = None
        self.forward_written = 0
        self.rc_written = 0
        self.embedding_dim = None

    @staticmethod
    def _finite(values, label):
        if not np.isfinite(values).all():
            raise RuntimeError(f"non-finite {label} embeddings")

    def _allocate(self, embedding_dim):
        embedding_dim = int(embedding_dim)
        if embedding_dim <= 0:
            raise ValueError("embedding dimension must be positive")
        self.embedding_dim = embedding_dim
        self.embeddings = np.lib.format.open_memmap(
            self.workspace / "embeddings.npy",
            mode="w+",
            dtype=np.float16,
            shape=(self.forward_rows, embedding_dim),
        )
        if self.rc_rows:
            self.rc_embeddings = np.lib.format.open_memmap(
                self.workspace / "rc_embeddings.npy",
                mode="w+",
                dtype=np.float16,
                shape=(self.rc_rows, embedding_dim),
            )
        else:
            self.rc_embeddings = np.empty((0, embedding_dim), dtype=np.float16)

    def append(self, forward, reverse=None):
        forward = np.asarray(forward)
        if forward.ndim != 2 or not len(forward):
            raise ValueError("forward embeddings must be a non-empty rank-2 batch")
        reverse = (
            np.empty((0, forward.shape[1]), dtype=forward.dtype)
            if reverse is None else np.asarray(reverse)
        )
        if reverse.ndim != 2 or reverse.shape[1] != forward.shape[1]:
            raise ValueError("reverse-complement embedding width mismatch")
        self._finite(forward, "forward")
        self._finite(reverse, "reverse-complement")
        with np.errstate(over="ignore", invalid="ignore"):
            forward_float16 = forward.astype(np.float16, copy=False)
            reverse_float16 = reverse.astype(np.float16, copy=False)
        self._finite(forward_float16, "forward float16")
        self._finite(reverse_float16, "reverse-complement float16")
        if self.embeddings is None:
            self._allocate(forward.shape[1])
        if forward.shape[1] != self.embedding_dim:
            raise ValueError("forward embedding width changed between batches")
        forward_end = self.forward_written + len(forward)
        reverse_end = self.rc_written + len(reverse)
        if forward_end > self.forward_rows or reverse_end > self.rc_rows:
            raise RuntimeError("embedding batch exceeds declared row count")
        self.embeddings[self.forward_written:forward_end] = forward_float16
        if len(reverse):
            self.rc_embeddings[self.rc_written:reverse_end] = reverse_float16
        self.forward_written = forward_end
        self.rc_written = reverse_end

    def _assert_complete(self):
        if self.embeddings is None:
            raise RuntimeError("embedding output is incomplete: no batches were written")
        if self.forward_written != self.forward_rows or self.rc_written != self.rc_rows:
            raise RuntimeError(
                "embedding output is incomplete: "
                f"forward={self.forward_written}/{self.forward_rows}, "
                f"rc={self.rc_written}/{self.rc_rows}"
            )

    @staticmethod
    def _close_memmap(values):
        if isinstance(values, np.memmap):
            values.flush()
            mapping = getattr(values, "_mmap", None)
            if mapping is not None:
                mapping.close()

    def save_npz(self, output_path, **metadata):
        self._assert_complete()
        overlap = self.RESERVED_FIELDS & set(metadata)
        if overlap:
            raise ValueError(f"reserved embedding fields supplied as metadata: {sorted(overlap)}")
        self.embeddings.flush()
        if isinstance(self.rc_embeddings, np.memmap):
            self.rc_embeddings.flush()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_name(output_path.name + ".tmp.npz")
        try:
            np.savez(
                temporary,
                embeddings=self.embeddings,
                rc_embeddings=self.rc_embeddings,
                **metadata,
            )
            os.replace(str(temporary), str(output_path))
        except Exception:
            try:
                temporary.unlink()
            except OSError:
                pass
            raise
        self.cleanup()

    def cleanup(self):
        self._close_memmap(self.embeddings)
        self._close_memmap(self.rc_embeddings)
        self.embeddings = None
        self.rc_embeddings = None
        if self.workspace.exists():
            shutil.rmtree(str(self.workspace))
