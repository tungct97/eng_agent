"""Giai đoạn 2 - Chunking: book.md -> chunks.jsonl + parents.json (LangChain).

Sách có cấu trúc cố định nên chia theo cấu trúc, không cắt theo độ dài:
  - Mỗi Unit = trang nội dung + trang "Exercises" ngay sau (số Unit lấy từ số bài tập "60.1, 60.2").
  - Chunk con (~900 ký tự) dùng để tìm kiếm, luôn có dòng ngữ cảnh "Unit 60: Health: injuries (reference)".
  - Cả Unit lưu thành "parent" để đưa cho LLM khi cần ngữ cảnh đầy đủ.
  - Metadata giữ số trang + đường dẫn ảnh để gửi lại hình minh hoạ.

Vào : data/build/book.md (+ data/build/images/)
Ra   : data/build/chunks.jsonl, data/build/parents.json
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHILD_SIZE, CHILD_OVERLAP, MIN_CHARS = 900, 120, 40

PAGE_HEAD = re.compile(r"^\s*# Page (\d+)\s*\n")
PAGE_IMG = re.compile(r"\]\((images/page_\d+\.jpg)\)")
IMG_LINE = re.compile(r"^!\[.*?\]\(.*?\)\s*$", re.M)
EXERCISE_HEAD = re.compile(r"^#+\s*Exercises?\b", re.M)
EX_NUM = re.compile(r"^(?:#+\s*)?(\d{1,3})\.\d{1,2}\b", re.M)
EX_SPLIT = re.compile(r"^(?:#+\s*)?(?=\d{1,3}\.\d{1,2}\s)", re.M)
ANS_MARK = re.compile(r"^(?:-\s*)?(?:Unit\s+(\d{1,3})\b|(\d{1,3})\.\d{1,2}\b)")
TOC_ENTRY = re.compile(r"^(?:-\s*)?(\d{1,3})\.?\s+([A-Z].{8,})$", re.M)
FOOTER = re.compile(r"English Vocabulary in Use\s*\(pre-intermediate", re.I)
DOTS = re.compile(r"\.{3,}")


# ------------------------------------------------------------------ đọc & làm sạch
def load_pages(md_path: Path) -> dict[int, dict]:
    pages = {}
    for part in re.split(r"\n---\n", md_path.read_text(encoding="utf-8")):
        m = PAGE_HEAD.match(part)
        if not m:
            continue
        n = int(m.group(1))
        img = PAGE_IMG.search(part)
        pages[n] = {"text": IMG_LINE.sub("", part[m.end():]).strip(), "image": img.group(1) if img else None}
    return pages


def clean(text: str) -> str:
    """Bỏ chân trang, dòng rác OCR; đổi chỗ trống '.....' thành '____'."""
    out = []
    for line in text.split("\n"):
        s = line.strip()
        if not s:
            out.append("")
            continue
        if len(s) < 90 and FOOTER.search(s):
            continue
        core = DOTS.sub("", s)
        if len(core) > 8 and sum(c.isalpha() or c == " " for c in core) / len(core) < 0.55:
            continue  # dòng toàn ký hiệu rác
        out.append(DOTS.sub(" ____ ", s))
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def unit_title(content: str) -> str:
    lines = [l.strip() for l in content.split("\n") if l.strip()]
    if not lines:
        return "Untitled"
    first = lines[0].lstrip("# ").strip()
    if first.endswith(",") and len(lines) > 1:  # tiêu đề dài bị ngắt thành 2 dòng
        first += " " + lines[1].lstrip("# ").strip()
    toks = first.split()
    while len(toks) > 1 and (not re.search(r"[A-Za-z]{2,}", toks[0])
                             or (toks[0].islower() and any(t[:1].isupper() for t in toks[1:3]))):
        toks.pop(0)  # rác OCR đầu tiêu đề: ",£*)" hoặc "alia"
    return " ".join(toks) or "Untitled"


# ------------------------------------------------------------------ cấu trúc sách
def find_units(pages: dict[int, dict]) -> dict[int, dict]:
    """unit_no -> {content_page, exercise_page}. Trang Exercises đứng ngay sau trang nội dung."""
    units = {}
    for n, p in pages.items():
        if not EXERCISE_HEAD.search(p["text"]):
            continue
        votes = Counter(int(m.group(1)) for m in EX_NUM.finditer(p["text"]))
        if votes and (n - 1) in pages:
            units[votes.most_common(1)[0][0]] = {"content": n - 1, "exercise": n}
    return units


def answer_key_pages(pages: dict[int, dict], after: int) -> list[int]:
    """Các trang Answer key: nằm sau Unit cuối, có nhiều dòng '60.1' hoặc 'Unit 60'."""
    keep, misses = [], 0
    for n in sorted(k for k in pages if k > after):
        hits = sum(bool(ANS_MARK.match(l.strip())) for l in pages[n]["text"].split("\n"))
        if hits >= 3:
            keep.append(n)
            misses = 0
        else:
            misses += 1
            if keep and misses >= 3:
                break
    return keep


# ------------------------------------------------------------------ tạo chunk
class Builder:
    def __init__(self, md_path: Path):
        self.root = md_path.parent
        self.pages = load_pages(md_path)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHILD_SIZE,
            chunk_overlap=CHILD_OVERLAP,
            separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ", " ", ""],
            keep_separator="start",
        )
        self.docs: list[Document] = []

    def _meta(self, page: int, **extra) -> dict:
        figs = sorted((self.root / "images").glob(f"page_{page:04d}_fig*.jpg"))
        return {
            "source": "book.md",
            "scan_page": page,
            "page_image": self.pages[page]["image"] or "",
            "figure_images": "|".join(f"images/{f.name}" for f in figs),  # Chroma/FAISS chỉ nhận giá trị đơn
            **extra,
        }

    def _add(self, text: str, page: int, unit: int | None, title: str, kind: str,
             min_chars: int = MIN_CHARS, **extra) -> None:
        for part in self.splitter.split_text(text):
            if len(part.strip()) < min_chars:
                continue
            ctx = f"Unit {unit}: {title} ({kind})\n" if unit else f"{title} ({kind})\n"
            meta = self._meta(page, unit_no=unit or 0, unit_title=title, chunk_type=kind,
                              parent_id=f"unit-{unit:03d}" if unit else "", **extra)
            self.docs.append(Document(page_content=ctx + part.strip(), metadata=meta))

    def build(self) -> tuple[list[Document], dict]:
        units = find_units(self.pages)
        parents = {}

        for u in sorted(units):
            c_page, e_page = units[u]["content"], units[u]["exercise"]
            content = clean(self.pages[c_page]["text"])
            exercises = clean(self.pages[e_page]["text"])
            title = unit_title(content)
            parents[f"unit-{u:03d}"] = {"unit_no": u, "title": title, "pages": [c_page, e_page],
                                        "reference": content, "exercises": exercises}
            self._add(content, c_page, u, title, "reference")
            for piece in EX_SPLIT.split(exercises):  # mỗi bài tập 60.1, 60.2... là một chunk
                m = re.match(r"(\d{1,3}\.\d{1,2})\b", piece.strip())
                if m:
                    self._add(piece, e_page, u, title, "exercise", exercise_no=m.group(1))

        # mục lục: mỗi dòng một chunk, giúp trả lời "bài nào nói về giới từ?"
        first_content = min((v["content"] for v in units.values()), default=0)
        seen = set()
        for n in sorted(k for k in self.pages if k < first_content):
            for m in TOC_ENTRY.finditer(clean(self.pages[n]["text"])):
                u = int(m.group(1))
                if 1 <= u <= 100 and u not in seen:
                    seen.add(u)
                    self._add(m.group(2), n, u, f"Contents entry for unit {u}", "toc", min_chars=8)  # mục lục vốn ngắn

        # đáp án: gom theo Unit
        last_ex = max((v["exercise"] for v in units.values()), default=0)
        buckets: dict[int, list] = defaultdict(list)
        cur = None
        for n in answer_key_pages(self.pages, last_ex):
            for line in clean(self.pages[n]["text"]).split("\n"):
                m = ANS_MARK.match(line.strip())
                if m:
                    u = int(m.group(1) or m.group(2))
                    if cur is None or cur <= u <= cur + 3:  # chặn khớp nhầm kiểu "2.5 hours"
                        cur = u
                if cur is not None:
                    buckets[cur].append((n, line))
        for u, rows in sorted(buckets.items()):
            title = parents.get(f"unit-{u:03d}", {}).get("title", f"Unit {u}")
            self._add("\n".join(l for _, l in rows), rows[0][0], u, title, "answer")
        return self.docs, parents


def build_documents(md_path: str | Path) -> tuple[list[Document], dict]:
    return Builder(Path(md_path)).build()




def run(book_md: Path, chunks_path: Path, parents_path: Path) -> tuple[int, int]:
    """Chạy chunking và ghi 2 file kết quả. Trả về (số chunk, số unit)."""
    docs, parents = build_documents(book_md)
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    with open(chunks_path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps({"text": d.page_content, "metadata": d.metadata}, ensure_ascii=False) + "\n")
    parents_path.write_text(json.dumps(parents, ensure_ascii=False, indent=1), encoding="utf-8")
    return len(docs), len(parents)


def load_chunks(chunks_path: Path) -> list[Document]:
    with open(chunks_path, encoding="utf-8") as f:
        return [Document(page_content=r["text"], metadata=r["metadata"]) for r in map(json.loads, f)]


def load_parents(parents_path: Path) -> dict:
    return json.loads(Path(parents_path).read_text(encoding="utf-8"))
