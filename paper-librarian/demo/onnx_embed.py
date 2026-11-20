"""all-MiniLM-L6-v2 via onnxruntime: same vectors as the sentence-transformers embedder, without torch.

Torch alone needs more memory than Render's free tier (512 MB) allows, so the hosted demo uses this.
"""

from __future__ import annotations

import os
import threading

import numpy as np

MODEL_REPO = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS = 256  # the model's max_seq_length
BATCH = 8  # small batches keep peak memory low on the free tier


class OnnxEmbedder:
    dim = 384

    def __init__(self) -> None:
        self._session = None
        self._tokenizer = None
        self._lock = threading.Lock()

    def warm_up(self) -> None:
        self._load()

    def _load(self):
        with self._lock:
            if self._session is None:
                import onnxruntime as ort
                from huggingface_hub import hf_hub_download
                from tokenizers import Tokenizer

                tok = Tokenizer.from_file(hf_hub_download(MODEL_REPO, "tokenizer.json"))
                tok.enable_truncation(MAX_TOKENS)
                tok.enable_padding()
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = int(os.environ.get("EMBED_THREADS", "1"))
                # The default arena grows to fit the largest batch and never shrinks; on a 512 MB
                # instance that alone OOM-kills add_paper. Allocate per run instead.
                opts.enable_cpu_mem_arena = False
                opts.enable_mem_pattern = False
                self._session = ort.InferenceSession(
                    hf_hub_download(MODEL_REPO, "onnx/model.onnx"), opts, providers=["CPUExecutionProvider"]
                )
                self._tokenizer = tok
        return self._session, self._tokenizer

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        session, tok = self._load()
        wanted = {i.name for i in session.get_inputs()}
        out: list[list[float]] = []
        for start in range(0, len(texts), BATCH):
            enc = tok.encode_batch(texts[start : start + BATCH])
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            feeds = {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)}
            hidden = session.run(None, {k: v for k, v in feeds.items() if k in wanted})[0]
            m = mask[..., None].astype(np.float32)
            pooled = (hidden * m).sum(1) / np.clip(m.sum(1), 1e-9, None)
            pooled /= np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12, None)
            out.extend(pooled.tolist())
        return out
