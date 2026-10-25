import pytest

from librarian.citations import Matcher, compact, link_paper, reference_arxiv_id, split_references
from librarian.store import Chunk, Paper, Store

DIM = 4
I = "    "  # INDENT marker from page_text()


def paper(pid, title, arxiv_id=None):
    return Paper(id=pid, title=title, source="arxiv" if arxiv_id else "pdf", arxiv_id=arxiv_id, sha256=pid)


@pytest.fixture
def store(tmp_path):
    with Store(tmp_path / "library.db", embedding_dim=DIM) as s:
        yield s


# --- splitting ---


def test_split_bracketed():
    raw = "\n".join([
        f"{I}[1] Y. Bengio, R. Ducharme. A neural probabilistic language model. Journal of Ma-",
        "",
        f"{I}chine Learning Research, 2003.",
        f"{I}[2] Y. Bengio, Y. LeCun. Scaling learning algorithms towards AI. 2007.",
        f"{I}[3] T. Brants. Large language models in machine translation. 2007.",
    ])
    refs = split_references(raw)
    assert len(refs) == 3
    assert refs[0] == "Y. Bengio, R. Ducharme. A neural probabilistic language model. Journal of Machine Learning Research, 2003."
    assert refs[2].startswith("T. Brants.")


def test_split_dotted_requires_sequence():
    raw = "1. A. Smith. Gadgets. 2019.\n2. B. Lee. Gizmos. 2020.\n3. C. Wu. Widgets and\nthings. 2021."
    assert split_references(raw) == ["A. Smith. Gadgets. 2019.", "B. Lee. Gizmos. 2020.", "C. Wu. Widgets and things. 2021."]
    # Numbers that don't count up (e.g. numbered lines inside a hanging-indent list) aren't markers.
    shuffled = "7. A. Smith. Gadgets. 2019.\n3. B. Lee. Gizmos. 2020.\n12. C. Wu. Widgets. 2021."
    assert len(split_references(shuffled)) == 3 and split_references(shuffled)[0].startswith("7.")


def test_split_hanging_indent_author_year():
    raw = "\n".join([
        "Alan Akbik, Duncan Blythe, and Roland Vollgraf.",
        f"{I}2018. Contextual string embeddings for sequence",
        f"{I}labeling. In Proceedings of COLING.",
        "",
        "Rami Al-Rfou, Dokook Choe, and Llion Jones. 2018.",
        f"{I}Character-level language modeling with deeper self-attention.",
        "continued at the top of a new column without indent",
        "Rie Kubota Ando and Tong Zhang. 2005. A framework for learning predictive structures.",
    ])
    refs = split_references(raw)
    assert len(refs) == 3
    assert refs[0] == (
        "Alan Akbik, Duncan Blythe, and Roland Vollgraf. 2018. Contextual string embeddings for sequence "
        "labeling. In Proceedings of COLING."
    )
    assert refs[1].endswith("without indent")


def test_split_drops_fragments():
    assert split_references("[1] ok\n[2] A. Real. Reference title here. 2020.\n[3] x") == [
        "A. Real. Reference title here. 2020."
    ]


# --- matching ---


@pytest.mark.parametrize(
    "ref, expected",
    [
        ("Ba et al. Layer normalization. arXiv preprint arXiv:1607.06450, 2016.", "1607.06450"),
        ("Bahdanau et al. CoRR, abs/1409.0473, 2014.", "1409.0473"),
        ("See https://arxiv.org/abs/2310.06825v2", "2310.06825"),
        ("Smith. A paper. NeurIPS 2019, pages 1234.5678.", None),
    ],
)
def test_reference_arxiv_id(ref, expected):
    assert reference_arxiv_id(ref) == expected


def test_compact_ignores_case_punctuation_and_ligatures():
    assert compact("Pre-training of Deep, Efﬁcient  Nets") == compact("pretraining of deep efficient nets")


LIBRARY = [
    paper("arxiv:1706.03762", "Attention Is All You Need", "1706.03762"),
    paper("arxiv:1810.04805", "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", "1810.04805"),
    paper("arxiv:1301.3781", "Efficient Estimation of Word Representations in Vector Space", "1301.3781"),
    paper("pdf:short", "BERT"),
]


@pytest.mark.parametrize(
    "ref, expected",
    [
        # exact title
        ("Ashish Vaswani et al. 2017. Attention is all you need. In NIPS.", "arxiv:1706.03762"),
        # arXiv id wins even with a garbled title
        ("J. Devlin. Bert: pre-tra... arXiv preprint arXiv:1810.04805, 2018.", "arxiv:1810.04805"),
        # hyphenation, case and punctuation differences
        ("Devlin et al. BERT: Pre-training of deep bidirec- tional transformers for language understanding. NAACL",
         "arxiv:1810.04805"),
        # small typo: fuzzy match
        ("T. Mikolov. Efficient estimaton of word representations in vector space. ICLR 2013.", "arxiv:1301.3781"),
        # similar words, different paper
        ("T. Mikolov et al. Distributed representations of words and phrases and their compositionality.", None),
        # very short titles never match on title alone
        ("Smith. Bert and Ernie: a study. 2020.", None),
    ],
)
def test_matcher(ref, expected):
    assert Matcher(LIBRARY).match(ref) == expected


def test_matcher_prefers_longest_matching_title():
    papers = [paper("pdf:a", "Deep Residual Learning"), paper("pdf:b", "Deep Residual Learning for Image Recognition")]
    assert Matcher(papers).match("K. He. Deep residual learning for image recognition. CVPR 2016.") == "pdf:b"


# --- linking ---


def test_link_paper_forward_and_backward(store):
    transformer = store.add_paper(paper("arxiv:1706.03762", "Attention Is All You Need", "1706.03762"))
    resnet = store.add_paper(paper("arxiv:1512.03385", "Deep Residual Learning for Image Recognition", "1512.03385"))
    link_paper(store, transformer, "[1] K. He et al. Deep residual learning for image recognition. CVPR.\n"
                                   "[2] D. Bahdanau. Neural machine translation by jointly learning to align. 2014.\n"
                                   "[3] Devlin et al. BERT: Pre-training of deep bidirectional transformers. 2018.")

    assert [c.matched_paper_id for c in store.get_citations(transformer.id)] == [resnet.id, None, None]
    assert [c.paper_id for c in store.get_cited_by(resnet.id)] == [transformer.id]

    # BERT arrives later: the Transformer's dangling reference to it gets linked.
    bert = store.add_paper(paper("arxiv:1810.04805", "BERT: Pre-training of Deep Bidirectional Transformers", "1810.04805"))
    result = link_paper(store, bert, "[1] A. Vaswani. Attention is all you need. 2017.\n[2] X. Other. Unrelated work here. 2019.\n[3] Y. Else. Another unrelated one. 2018.")
    assert (result.references, result.linked, result.cited_by) == (3, 1, 1)
    assert store.get_citations(transformer.id)[2].matched_paper_id == bert.id
    assert {c.paper_id for c in store.get_cited_by(transformer.id)} == {bert.id}


def test_link_paper_replaces_previous_citations(store):
    p = store.add_paper(paper("pdf:x", "Some Paper Title Here"))
    link_paper(store, p, "[1] A. One. First reference title. 2019.\n[2] B. Two. Second reference. 2020.\n[3] C. Three. Third one. 2021.")
    link_paper(store, p, "[1] A. One. Only reference now. 2019.\n[2] B. Two. Second reference. 2020.\n[3] C. Three. Third one. 2021.")
    assert [c.seq for c in store.get_citations("pdf:x")] == [0, 1, 2]
    assert store.get_citations("pdf:x")[0].raw_reference == "A. One. Only reference now. 2019."


def test_self_citation_not_linked(store):
    p = store.add_paper(paper("pdf:self", "Widgets: A Comprehensive Study"))
    result = link_paper(store, p, "[1] Me. Widgets: a comprehensive study. 2019.\n[2] B. Two. Second reference. 2020.\n[3] C. Three. Third one. 2021.")
    assert result.linked == 0 and result.cited_by == 0


def test_link_rolls_back_with_caller_transaction(store):
    p = store.add_paper(paper("pdf:x", "Some Paper Title Here"))
    with pytest.raises(RuntimeError):
        with store.transaction():
            link_paper(store, p, "[1] A. One. A reference title. 2019.\n[2] B. Two. Another title. 2020.\n[3] C. Three. Third. 2021.", commit=False)
            raise RuntimeError("boom")
    assert store.get_citations("pdf:x") == []
