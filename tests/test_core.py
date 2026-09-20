"""Tests de la logique pure (aucune dépendance lourde). Lancer : `uv run pytest tests -q`
ou simplement `python tests/test_core.py`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.parsing.chunking import (  # noqa: E402
    RawChunk,
    TextChunker,
    join_lines,
    normalize_rows,
    overlap_ratio,
    rows_to_markdown_chunks,
    split_long_text,
    table_quality,
    tail_at_boundary,
)
from src.retrieval.lexical import BM25Index, tokenize  # noqa: E402
from src.retrieval.query_planner import plan_queries, rules_decompose  # noqa: E402
from src.utils import extract_percent_figures, find_unsupported_figures, normalize_text  # noqa: E402

BOX = (0.0, 0.0, 100.0, 20.0)


# ------------------------------------------------------------------ utils
def test_normalize_text():
    assert normalize_text("  Dérivés   ÉLIGIBLES ") == "derives eligibles"


def test_extract_percent_figures_ranges_and_decimals():
    assert extract_percent_figures("Actions : 0 - 35 % ; taux 65-100 % ; frais 1,50%") == ["0", "35", "65", "100", "1.5"]


def test_find_unsupported_figures():
    sources = "Le compartiment actions représente entre 0 et 35 % de l'actif net, frais de 1,5 %."
    answer = "| Actions | 0 % | 35 % |\n| Dérivés | 0 % | 20 % |\n| Frais | 1.50 % |"
    # 0 est trivial, 35 et 1.5 existent dans la source, 20 est inventé.
    assert find_unsupported_figures(answer, sources) == ["20 %"]


# ------------------------------------------------------------------ lexical
def test_tokenize_strips_accents_plurals_and_stopwords():
    assert tokenize("Les limites d'exposition des dérivés") == ["limite", "exposition", "derive"]


def test_bm25_prefers_exact_terms():
    docs = [
        "Politique de distribution des dividendes",
        "Le compartiment peut investir jusqu'à 35 % en actions",
        "Frais de gestion annuels et commissions de surperformance",
    ]
    index = BM25Index(docs)
    assert index.search("limite actions 35")[0][0] == 1
    assert index.search("frais de gestion")[0][0] == 2
    assert index.search("mot inexistant zzz") == []


# ------------------------------------------------------------------ chunking
def test_join_lines_dehyphenates_and_keeps_bullets():
    assert join_lines(["Le fonds peut inves-", "tir en actions", "• limite : 35 %"]) == "Le fonds peut investir en actions\n• limite : 35 %"


def test_overlap_ratio():
    assert overlap_ratio((0, 0, 10, 10), (5, 0, 15, 10)) == 0.5
    assert overlap_ratio((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_split_long_text_bounds_and_content():
    sentence = "Le compartiment investit dans des obligations. "
    text = sentence * 60  # ~2800 caractères, un seul « paragraphe »
    parts = split_long_text(text, 500)
    assert all(len(p) <= 500 for p in parts)
    assert len(parts) >= 5
    assert "".join(parts).replace(" ", "").replace("\n", "") == text.replace(" ", "")


def test_split_long_text_handles_unbroken_string():
    parts = split_long_text("x" * 2500, 1000)
    assert [len(p) for p in parts] == [1000, 1000, 500]


def test_tail_at_boundary_starts_on_word():
    tail = tail_at_boundary("Première phrase complète. Seconde phrase qui se termine ici", 30)
    assert tail and not tail.startswith(("ère", "hrase")) and tail.endswith("ici")


def _fill(chunker, n, section=None, page=1):
    for i in range(n):
        # Chaque paragraphe se termine par un mot unique (finN) : le recouvrement devient observable.
        chunker.add(f"Paragraphe numéro {i} " + "mot " * 40 + f"fin{i}", page, (0, i * 10, 100, i * 10 + 9), section)


def test_chunker_respects_size_and_overlaps():
    c = TextChunker(chunk_size=400, overlap=100, min_chunk_chars=100)
    _fill(c, 12)
    c.flush()
    assert len(c.chunks) > 3
    assert all(len(ch.text) <= 400 + 100 + 4 for ch in c.chunks)  # taille + recouvrement
    # Le recouvrement : la fin (mot unique inclus) d'un chunk réapparaît au début du suivant.
    for prev, nxt in zip(c.chunks, c.chunks[1:]):
        last_marker = prev.text.split()[-1]  # ex. « fin3 »
        assert last_marker.startswith("fin")
        assert last_marker in nxt.text.split()[:30], (last_marker, nxt.text[:120])


def test_chunker_crosses_pages_and_reports_range():
    c = TextChunker(chunk_size=1000, overlap=0, min_chunk_chars=100)
    c.add("A " * 100, 1, (0, 700, 100, 780), None)  # fin de page 1
    c.add("B " * 100, 2, (0, 20, 100, 100), None)  # début de page 2
    c.flush()
    assert len(c.chunks) == 1
    assert (c.chunks[0].page_start, c.chunks[0].page_end) == (1, 2)


def test_chunker_heading_starts_new_chunk_only_when_buffer_is_big_enough():
    c = TextChunker(chunk_size=1000, overlap=0, min_chunk_chars=250)
    c.add("Intro " * 60, 1, BOX, "Intro")  # ~360 car. : assez grand
    c.add("2. Limites", 1, BOX, "Limites", is_heading=True)
    c.add("Les actions sont limitées à 35 %. " * 3, 1, BOX, "Limites")
    c.flush()
    assert [ch.section for ch in c.chunks] == ["Intro", "Limites"]

    tiny = TextChunker(chunk_size=1000, overlap=0, min_chunk_chars=250)
    tiny.add("Court.", 1, BOX, "A")
    tiny.add("Titre B", 1, BOX, "B", is_heading=True)  # trop petit : pas de coupure
    tiny.flush()
    assert len(tiny.chunks) == 1


def test_chunker_table_caption_and_order():
    c = TextChunker(chunk_size=1000, overlap=0, min_chunk_chars=50)
    c.add("Texte avant. " * 10, 3, (0, 100, 100, 150), "Sec")
    c.add("Tableau 2 : Limites par classe d'actifs", 3, (0, 160, 100, 172), "Sec")
    caption = c.pop_caption(page=3, table_top=180)
    assert caption == "Tableau 2 : Limites par classe d'actifs"
    c.flush()
    c.emit_external(RawChunk("table", f"{caption}\n| a | b |", 3, 3, (0, 180, 100, 300), "Sec"))
    c.add("Texte après. " * 10, 3, (0, 320, 100, 380), "Sec")
    c.flush()
    assert [ch.kind for ch in c.chunks] == ["text", "table", "text"]
    assert "Tableau 2" not in c.chunks[0].text  # la légende a quitté le texte pour suivre le tableau


def test_chunker_caption_not_taken_when_far_from_table():
    c = TextChunker()
    c.add("Un paragraphe bref", 1, (0, 10, 100, 30), None)
    assert c.pop_caption(page=1, table_top=400) is None


# ------------------------------------------------------------------ tableaux
def test_normalize_rows_and_markdown():
    raw = [["Classe", None, "Min", "Max"], [None, None, None, None], ["Actions", "", "0 %", "35 %"], ["Taux", "", "65 %", "100 %"]]
    rows = normalize_rows(raw)
    assert rows[0] == ["Classe", "Min", "Max"]  # colonne et ligne vides supprimées
    assert table_quality(rows)
    (md,) = rows_to_markdown_chunks(rows, 5000)
    assert md.splitlines()[0] == "| Classe | Min | Max |"
    assert "| Actions | 0 % | 35 % |" in md


def test_markdown_split_repeats_header():
    rows = [["Nom", "Valeur"]] + [[f"ligne {i}", str(i)] for i in range(40)]
    parts = rows_to_markdown_chunks(rows, 300)
    assert len(parts) > 1
    assert all(p.startswith("| Nom | Valeur |") for p in parts)
    assert sum(p.count("ligne") for p in parts) == 40


def test_table_quality_flags_sparse_tables():
    sparse = [["A", "", ""], ["", "", "x"], ["", "", ""]]
    assert normalize_rows(sparse) == [["A", ""], ["", "x"]]  # ligne et colonne vides retirées
    assert not table_quality(normalize_rows(sparse))  # 50 % de cellules vides : Markdown douteux
    assert not table_quality([["seule ligne", "x"]])
    assert not table_quality([["a", "", "", ""], ["", "", "", "b"]])


# ------------------------------------------------------------------ planificateur
def test_rules_decompose_enumeration():
    q = "Quelles contraintes par classe d'actifs (actions, obligations, devises et dérivés) dans la stratégie ?"
    queries = rules_decompose(q)
    assert queries[0] == q
    assert len(queries) == 5
    assert "Quelles contraintes par classe d'actifs dérivés dans la stratégie ?" in queries


def test_rules_decompose_ignores_non_enumerations():
    assert rules_decompose("Que dit la page 3 (voir annexe complète du document) ?") == [
        "Que dit la page 3 (voir annexe complète du document) ?"
    ]
    assert plan_queries("Simple question", "rules") == ["Simple question"]
    assert plan_queries("Q (a, b)", "off") == ["Q (a, b)"]


if __name__ == "__main__":  # exécution sans pytest
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    sys.exit(1 if failures else 0)
