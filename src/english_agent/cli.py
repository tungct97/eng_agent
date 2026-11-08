"""Một CLI duy nhất cho cả pipeline:

    eng-agent ocr      # ảnh trang  -> data/build/book.md + images/
    eng-agent chunk    # book.md    -> data/build/chunks.jsonl + parents.json
    eng-agent index    # chunks     -> data/chroma_db
    eng-agent build    # chạy cả 3 bước trên
    eng-agent search "bruise"   # xem tìm kiếm trả về gì (không cần LLM)
    eng-agent ask               # hỏi đáp với LLM local (Ollama)

(cũng chạy được bằng: python -m english_agent <lệnh>)
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from .config import Settings


def cmd_ocr(cfg: Settings, a) -> None:
    from . import ocr

    book = ocr.run(cfg.pages_dir, cfg.build_dir, lang=a.lang, psm=a.psm, limit=a.limit, figures=a.figures, pdf=a.pdf)
    print(f"OCR xong -> {book}")


def cmd_chunk(cfg: Settings, a) -> None:
    from . import chunking

    if not cfg.book_md.exists():
        raise SystemExit(f"Chưa có {cfg.book_md}. Chạy trước: eng-agent ocr")
    n_chunks, n_units = chunking.run(cfg.book_md, cfg.chunks_path, cfg.parents_path)
    print(f"{n_chunks} chunks, {n_units} units -> {cfg.chunks_path}")


def cmd_index(cfg: Settings, a) -> None:
    from . import indexing

    print(f"Đã lưu {indexing.build_index(cfg)} chunks vào {cfg.chroma_dir}")


def cmd_build(cfg: Settings, a) -> None:
    cmd_ocr(cfg, a)
    cmd_chunk(cfg, a)
    cmd_index(cfg, a)


def cmd_search(cfg: Settings, a) -> None:
    from .retrieval import Retriever, fix_text

    r = Retriever(cfg).search(fix_text(a.query))
    for doc, dist in r.hits:
        m = doc.metadata
        print(f"{dist:.3f}  unit {m['unit_no']:>3} {m['unit_title'][:38]:<38} [{m['chunk_type']}] scan page {m['scan_page']}")
    print("-> Unit dùng làm ngữ cảnh:", r.unit_ids or f"(không có, đều xa hơn {cfg.max_distance})")
    for p in r.images:
        print("   ảnh:", p)


def cmd_ask(cfg: Settings, a) -> None:
    from .tutor import Tutor

    tutor = Tutor(cfg)

    def one(q: str) -> None:
        ans = tutor.ask(q)
        print(ans.text)
        for p in ans.images:
            print("[ảnh]", p)

    if a.question:
        return one(a.question)
    while True:
        try:
            q = input("\nbạn> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q:
            one(q)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="eng-agent", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data-dir", type=Path, help="thư mục dữ liệu (mặc định ./data)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_ocr_opts(p):
        p.add_argument("--limit", type=int, help="chỉ OCR N trang đầu (để thử)")
        p.add_argument("--psm", type=int, default=4, help="bố cục Tesseract: 3=tự chia cột, 4=một cột, 6=một khối")
        p.add_argument("--lang", default="eng")
        p.add_argument("--figures", action="store_true", help="tự cắt các vùng tranh minh hoạ")
        p.add_argument("--pdf", action="store_true", help="xuất thêm book.pdf (cần: uv sync --extra pdf)")

    p = sub.add_parser("ocr", help="ảnh trang -> book.md")
    add_ocr_opts(p)
    p.set_defaults(fn=cmd_ocr)
    sub.add_parser("chunk", help="book.md -> chunks.jsonl + parents.json").set_defaults(fn=cmd_chunk)
    sub.add_parser("index", help="chunks.jsonl -> Chroma").set_defaults(fn=cmd_index)
    p = sub.add_parser("build", help="ocr + chunk + index")
    add_ocr_opts(p)
    p.set_defaults(fn=cmd_build)
    p = sub.add_parser("search", help="xem kết quả tìm kiếm cho một câu hỏi")
    p.add_argument("query")
    p.set_defaults(fn=cmd_search)
    p = sub.add_parser("ask", help="hỏi đáp với LLM local")
    p.add_argument("-q", "--question", help="hỏi một câu rồi thoát (bỏ qua để vào chế độ hội thoại)")
    p.set_defaults(fn=cmd_ask)

    a = ap.parse_args(argv)
    cfg = Settings()
    if a.data_dir:
        cfg = replace(cfg, data_dir=a.data_dir)
    a.fn(cfg, a)


if __name__ == "__main__":
    main()
