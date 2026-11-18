"""Build step: pre-load the demo library so visitors start with linked papers to search."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from librarian.ingest import ingest  # noqa: E402
from librarian.store import Store  # noqa: E402
from onnx_embed import OnnxEmbedder  # noqa: E402

PAPERS = [
    "1706.03762",  # Attention Is All You Need
    "1810.04805",  # BERT
    "1908.10084",  # Sentence-BERT
    "2005.11401",  # Retrieval-Augmented Generation
]

db = Path(os.environ.get("DEMO_DB", Path(__file__).parent / "data" / "library.db"))
db.parent.mkdir(parents=True, exist_ok=True)
emb = OnnxEmbedder()
store = Store(db, embedding_dim=emb.dim)
for arxiv_id in PAPERS:
    result = ingest(arxiv_id, store, emb, progress=lambda m: print("  ", m, flush=True))
    print(f"{arxiv_id}: {result.paper.title} ({result.chunk_count} chunks)", flush=True)
store.close()
