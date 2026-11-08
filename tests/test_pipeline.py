"""Test offline: không cần Tesseract, Ollama, Chroma hay mô hình embedding."""
import json
from pathlib import Path

from langchain_core.documents import Document

from english_agent import chunking, ocr
from english_agent.config import Settings
from english_agent.retrieval import assemble, fix_text
from english_agent.tutor import NOT_FOUND, Tutor

# ----------------------------------------------------------------- sách giả: 2 Unit
BOOK = """# Page 1

![Page 1](images/page_0001.jpg)

## Contents

- 1 Greetings hello, hi
- 2 Health: injuries bruise, bandage

---

# Page 2

![Page 2](images/page_0002.jpg)

## Greetings

### A Saying hello

Hello and hi are common greetings. We say hello when we meet someone. Good morning is more formal.

---

# Page 3

![Page 3](images/page_0003.jpg)

## Exercises

1.1 Complete the greeting with a suitable word from the opposite page and say it aloud twice.

1.2 Match each greeting on the left with a reply on the right. Then practise them with a friend.

---

# Page 4

![Page 4](images/page_0004.jpg)

## Health: injuries

### A Common injuries

She cut her finger. He twisted his ankle. She has a bruise on her leg after she fell off her bike.

---

# Page 5

![Page 5](images/page_0005.jpg)

## Exercises

2.1 Complete each sentence with a word about injuries from the opposite page and check it.

2.2 Match the injury on the left with the treatment on the right and then explain your choices.
"""


def make_book(tmp_path: Path) -> Path:
    (tmp_path / "images").mkdir()
    for n in (4, 2):
        (tmp_path / "images" / f"page_{n:04d}_fig1.jpg").write_bytes(b"x")
    md = tmp_path / "book.md"
    md.write_text(BOOK, encoding="utf-8")
    return md


# ----------------------------------------------------------------- chunking
def test_units_are_paired_from_exercise_numbers(tmp_path):
    pages = chunking.load_pages(make_book(tmp_path))
    assert chunking.find_units(pages) == {
        1: {"content": 2, "exercise": 3},
        2: {"content": 4, "exercise": 5},
    }


def test_run_writes_chunks_and_parents_and_they_load_back(tmp_path):
    md = make_book(tmp_path)
    n_chunks, n_units = chunking.run(md, tmp_path / "chunks.jsonl", tmp_path / "parents.json")
    assert n_units == 2 and n_chunks > 0
    docs = chunking.load_chunks(tmp_path / "chunks.jsonl")
    kinds = {d.metadata["chunk_type"] for d in docs}
    assert {"reference", "exercise", "toc"} <= kinds
    ref = next(d for d in docs if d.metadata["chunk_type"] == "reference" and d.metadata["unit_no"] == 2)
    assert ref.metadata["parent_id"] == "unit-002"
    assert ref.metadata["scan_page"] == 4
    assert ref.page_content.startswith("Unit 2: Health: injuries (reference)")
    assert ref.metadata["figure_images"] == "images/page_0004_fig1.jpg"
    parents = chunking.load_parents(tmp_path / "parents.json")
    assert parents["unit-002"]["pages"] == [4, 5] and "bruise" in parents["unit-002"]["reference"]


def test_exercises_are_split_per_exercise(tmp_path):
    md = make_book(tmp_path)
    docs, _ = chunking.build_documents(md)
    nums = sorted(d.metadata["exercise_no"] for d in docs if d.metadata["chunk_type"] == "exercise")
    assert nums == ["1.1", "1.2", "2.1", "2.2"]


def test_clean_removes_footer_and_dot_leaders():
    out = chunking.clean("English Vocabulary in Use (pre-intermediate & intermediate) 123\nI have a ......... on my leg")
    assert "English Vocabulary" not in out and "____" in out


# ----------------------------------------------------------------- OCR: nhận dạng tiêu đề (hàm thuần)
def line(text, h=20, top=500, par=(1, 1)):
    return {"text": text, "h": h, "top": top, "par": par}


def test_heading_levels():
    med = 20
    assert ocr.level_of(line("60 Health: injuries", h=34, top=50), med, 1500) == 2
    assert ocr.level_of(line("A Common injuries", h=24), med, 1500) == 3
    assert ocr.level_of(line("NOTE"), med, 1500) == 4
    assert ocr.level_of(line("She cut her finger and went home."), med, 1500) == 0


def test_toc_page_becomes_bullets_not_headings():
    lines = [line(f"{n} Topic name here", h=30, top=100 * n) for n in range(37, 42)]
    lines.append(line("37 Prepositions: place 38 Adverbs: frequency", top=10))
    md = ocr.to_markdown(lines, 1500)
    assert "#" not in md.replace("##", "") or md.count("##") == 0
    assert "- 38 Adverbs: frequency" in md


# ----------------------------------------------------------------- retrieval + tutor (không DB, không LLM)
def fake_hits(cfg):
    d1 = Document(page_content="x", metadata={"parent_id": "unit-002", "figure_images": "images/page_0004_fig1.jpg"})
    d2 = Document(page_content="y", metadata={"parent_id": "unit-001", "figure_images": ""})
    far = Document(page_content="z", metadata={"parent_id": "unit-009", "figure_images": ""})
    return [(d1, 0.31), (d2, 0.50), (far, 0.90)]


PARENTS = {
    "unit-001": {"unit_no": 1, "title": "Greetings", "pages": [2, 3], "reference": "hello", "exercises": "1.1 ..."},
    "unit-002": {"unit_no": 2, "title": "Health: injuries", "pages": [4, 5], "reference": "bruise", "exercises": "2.1 ..."},
}


def test_assemble_drops_far_hits_and_collects_images(tmp_path):
    cfg = Settings(root=tmp_path)
    r = assemble(fake_hits(cfg), PARENTS, practice=False, cfg=cfg)
    assert r.unit_ids == ["unit-002", "unit-001"]  # unit-009 xa quá ngưỡng, không có trong parents
    assert "Unit 2: Health: injuries - scan page 4" in r.context
    assert r.images[0].name == "page_0004.jpg" and r.images[-1].name == "page_0004_fig1.jpg"


def test_assemble_practice_uses_exercise_side(tmp_path):
    cfg = Settings(root=tmp_path)
    r = assemble(fake_hits(cfg), PARENTS, practice=True, cfg=cfg)
    assert "2.1 ..." in r.context and r.images[0].name == "page_0005.jpg"


class FakeRetriever:
    def __init__(self, cfg, found=True):
        self.cfg, self.found = cfg, found

    def search(self, q):
        hits = fake_hits(self.cfg) if self.found else [(fake_hits(self.cfg)[2][0], 0.9)]
        return assemble(hits, PARENTS, False, self.cfg)


class FakeLLM:
    def invoke(self, messages):
        self.last = messages
        return type("R", (), {"content": "ok"})()


def test_tutor_answers_and_refuses_off_topic(tmp_path):
    cfg = Settings(root=tmp_path)
    llm = FakeLLM()
    ans = Tutor(cfg, FakeRetriever(cfg), llm).ask("bruise?")
    assert ans.text == "ok" and ans.units == ["unit-002", "unit-001"]
    assert "Unit 2: Health: injuries" in llm.last[1][1]
    assert Tutor(cfg, FakeRetriever(cfg, found=False), llm).ask("thời tiết?").text == NOT_FOUND


def test_fix_text_recovers_vietnamese_from_bad_terminal_locale():
    broken = "tôi muốn".encode("utf-8").decode("ascii", "surrogateescape")
    assert fix_text(broken) == "tôi muốn"


def test_settings_paths_are_derived_from_root(tmp_path):
    cfg = Settings(root=tmp_path)
    assert cfg.book_md == tmp_path / "build" / "book.md"
    assert cfg.pages_dir == tmp_path / "pages" and cfg.chroma_dir == tmp_path / "chroma_db"
    assert cfg.chunks_path.parent == cfg.parents_path.parent == cfg.build_dir
