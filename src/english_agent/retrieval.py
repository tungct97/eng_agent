"""Giai đoạn 4a - Retrieval: câu hỏi -> chunk con -> Unit đầy đủ (parent) + ảnh minh hoạ.

Tìm bằng chunk nhỏ cho chính xác, nhưng đưa cho LLM cả Unit để có đủ ngữ cảnh.
Ảnh được xác định từ metadata (không nhờ LLM gọi công cụ): ảnh trang của Unit, cộng ảnh cắt nếu có.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document

from .chunking import load_parents
from .config import Settings

PRACTICE = re.compile(r"bài tập|luyện tập|exercise|practice|quiz|test me", re.I)
CONTEXT_CHARS = 4500  # mỗi Unit đưa cho LLM tối đa chừng này ký tự


@dataclass
class Retrieval:
    practice: bool = False
    unit_ids: list[str] = field(default_factory=list)
    context: str = ""
    unit_images: dict[str, list[Path]] = field(default_factory=dict)  # Unit -> ảnh trang (+ ảnh cắt) của Unit đó
    hits: list[tuple[Document, float]] = field(default_factory=list)

    @property
    def found(self) -> bool:
        return bool(self.unit_ids)

    @property
    def images(self) -> list[Path]:
        return [p for uid in self.unit_ids for p in self.unit_images.get(uid, [])]


def fix_text(s: str) -> str:
    """Terminal có locale ASCII biến chữ Việt thành ký tự surrogate lỗi -> khôi phục lại UTF-8."""
    return s.encode("utf-8", "surrogateescape").decode("utf-8", "replace").strip()


def unit_key(unit_no: int) -> str:
    return f"unit-{unit_no:03d}"


def assemble(
    hits: list[tuple[Document, float]], parents: dict, practice: bool, cfg: Settings, pinned: tuple[str, ...] = ()
) -> Retrieval:
    """Phần thuần logic (không đụng DB/mô hình, dễ test): chọn Unit, dựng ngữ cảnh, gom ảnh.

    `pinned`: Unit đang học, luôn được đưa vào ngữ cảnh trước (cả nội dung lẫn bài tập), kể cả khi tìm kiếm không ra.
    """
    unit_ids: list[str] = [u for u in pinned if u in parents][: cfg.max_units]
    pinned_ids = set(unit_ids)
    figures: dict[str, list[str]] = {}
    for doc, dist in hits:
        if dist > cfg.max_distance:  # quá xa câu hỏi -> bỏ, tránh trả lời bằng nội dung không liên quan
            continue
        pid = doc.metadata.get("parent_id")
        if pid and pid in parents and pid not in unit_ids and len(unit_ids) < cfg.max_units:
            unit_ids.append(pid)
        if pid in unit_ids and doc.metadata.get("figure_images"):
            figures.setdefault(pid, []).extend(doc.metadata["figure_images"].split("|"))

    side = 1 if practice else 0  # pages = [trang nội dung, trang bài tập]
    parts, unit_images = [], {}
    for pid in unit_ids:
        p = parents[pid]
        if pid in pinned_ids:  # bài đang học: đưa cả nội dung lẫn bài tập
            body = p["reference"][:3000] + "\n\n[Exercises of this unit]\n" + p["exercises"][:2000]
            page = p["pages"][0]
        else:
            body = p["exercises"] if practice else p["reference"]
            page = p["pages"][side]
            body = body[:CONTEXT_CHARS]
        parts.append(f"[Unit {p['unit_no']}: {p['title']} - scan page {page}]\n{body}")
        unit_images[pid] = [cfg.build_dir / "images" / f"page_{page:04d}.jpg"] + [
            cfg.build_dir / f for f in dict.fromkeys(figures.get(pid, []))
        ]
    return Retrieval(practice, unit_ids, "\n\n".join(parts), unit_images, hits)


class Retriever:
    def __init__(self, cfg: Settings):
        from .indexing import get_db

        if not cfg.chroma_dir.exists():
            raise FileNotFoundError(f"Chưa có {cfg.chroma_dir}. Chạy trước: eng-agent index")
        self.cfg = cfg
        self.db = get_db(cfg)
        self.parents = load_parents(cfg.parents_path)

    def search(self, question: str, pinned: tuple[str, ...] = ()) -> Retrieval:
        practice = bool(PRACTICE.search(question))
        kinds = ["exercise"] if practice else ["reference", "toc"]
        hits = self.db.similarity_search_with_score(
            question, k=self.cfg.top_k, filter={"chunk_type": {"$in": kinds}}
        )
        return assemble(hits, self.parents, practice, self.cfg, pinned)
