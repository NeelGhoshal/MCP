import pytest

from librarian.search import fts_query, make_snippet, rrf, search
from librarian.store import Chunk, Paper, Store

DIM = 4

# Each "topic" gets its own axis, so the fake embedder makes semantic matches predictable.
TOPICS = {"attention": 0, "convolution": 1, "recurrent": 2, "graph": 3}


class TopicEmbedder:
    dim = DIM

    def embed(self, texts):
        out = []
        for t in texts:
            v = [0.01] * DIM
            for word, axis in TOPICS.items():
                if word in t.lower():
                    v[axis] += 1.0
            out.append(v)
        return out


@pytest.fixture
def store(tmp_path):
    with Store(tmp_path / "library.db", embedding_dim=DIM) as s:
        papers = {
            "arxiv:a": (2017, ["Self-attention lets every token attend to every other token.",
                               "We train on WMT 2014 English-German."]),
            "arxiv:b": (2015, ["Deep convolution networks for image recognition.",
                               "Residual connections ease optimization."]),
            "arxiv:c": (2020, ["Recurrent networks process tokens sequentially.",
                               "Graph neural networks pass messages between nodes."]),
        }
        emb = TopicEmbedder()
        for pid, (year, texts) in papers.items():
            s.add_paper(Paper(id=pid, title=f"Paper {pid[-1].upper()}", source="arxiv", year=year, arxiv_id=pid))
            chunks = [Chunk(seq=i, section="Method", page=i + 1, text=t) for i, t in enumerate(texts)]
            s.add_chunks(pid, chunks, emb.embed(texts))
        yield s


def test_fts_query_sanitizes_operators():
    assert fts_query('what is "attention"? (AND) NEAR*') == '"what" OR "is" OR "attention" OR "and" OR "near"'
    assert fts_query("?? !") is None


def test_rrf_rewards_agreement():
    fused = rrf({"semantic": [1, 2, 3], "keyword": [3, 4]})
    assert fused[0][0] == 3 and fused[0][2] == ["semantic", "keyword"]
    assert {cid for cid, _, _ in fused} == {1, 2, 3, 4}


def test_keyword_mode(store):
    hits = search(store, "WMT English", mode="keyword")
    assert hits[0].paper_id == "arxiv:a" and hits[0].matched_by == ["keyword"]
    assert hits[0].page == 2


def test_semantic_mode_finds_without_word_overlap(store):
    # "convolutional" never appears verbatim, but the fake embedder maps it to the convolution axis.
    hits = search(store, "convolutional models", TopicEmbedder(), mode="semantic", limit=1)
    assert hits[0].paper_id == "arxiv:b"


def test_hybrid_combines_both(store):
    hits = search(store, "attention tokens", TopicEmbedder())
    assert hits[0].paper_id == "arxiv:a"
    assert set(hits[0].matched_by) == {"semantic", "keyword"}
    assert hits[0].title == "Paper A" and hits[0].year == 2017


def test_filters(store):
    hits = search(store, "networks", TopicEmbedder(), year_from=2016)
    assert hits and {h.paper_id for h in hits} <= {"arxiv:a", "arxiv:c"}
    assert all(h.year >= 2016 for h in hits)
    hits = search(store, "networks", TopicEmbedder(), paper_ids=["arxiv:b"])
    assert hits and {h.paper_id for h in hits} == {"arxiv:b"}
    assert search(store, "networks", TopicEmbedder(), year_to=2010) == []


def test_per_paper_cap_and_limit(store):
    hits = search(store, "networks tokens", TopicEmbedder(), per_paper=1, limit=10)
    assert len({h.paper_id for h in hits}) == len(hits)
    assert len(search(store, "networks tokens", TopicEmbedder(), limit=2)) == 2


def test_bad_inputs(store):
    assert search(store, "   ", TopicEmbedder()) == []
    with pytest.raises(ValueError):
        search(store, "x", None, mode="hybrid")
    with pytest.raises(ValueError):
        search(store, "x", TopicEmbedder(), mode="fuzzy")


def test_snippet_centres_on_term():
    text = " ".join(["filler"] * 100) + " positional encoding uses sines " + " ".join(["tail"] * 100)
    snip = make_snippet(text, ["positional"], width=80)
    assert "positional" in snip and snip.startswith("…") and snip.endswith("…")
    assert len(snip) <= 82
    assert make_snippet("short text", ["x"]) == "short text"


def test_snippet_prefers_window_with_most_terms():
    text = "of the " + " ".join(["filler"] * 80) + " self attention replaces recurrence " + " ".join(["tail"] * 80)
    snip = make_snippet(text, ["why", "of", "self", "attention", "recurrence"], width=80)
    assert "self attention replaces recurrence" in snip
