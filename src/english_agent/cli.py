"""Một CLI duy nhất cho cả pipeline:

    eng-agent ocr      # ảnh trang  -> build/book.md + build/images/
    eng-agent chunk    # book.md    -> build/chunks.jsonl + build/parents.json
    eng-agent index    # chunks     -> chroma_db/
    eng-agent build    # chạy cả 3 bước trên
    eng-agent build --skip-ocr   # chỉ chunk + index (dùng build/book.md có sẵn, không cần ảnh scan)
    eng-agent search "bruise"   # xem tìm kiếm trả về gì (không cần LLM)
    eng-agent ask               # hội thoại với LLM local (Ollama), lưu vào conversations.db
    eng-agent ask -c            # tiếp tục phiên gần nhất
    eng-agent sessions          # các phiên đã lưu;  eng-agent history 3  # xem lại phiên #3

(cũng chạy được bằng: python -m english_agent <lệnh>)
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from .config import Settings

try:  # bật chỉnh sửa dòng kiểu readline cho input(): xoá/di chuyển đúng với chữ có dấu UTF-8 (tránh ký tự lỗi �)
    import readline  # noqa: F401
except ImportError:  # Windows thuần không có readline
    pass


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
    if not a.skip_ocr:  # --skip-ocr: dùng build/book.md có sẵn, chỉ chunk + index
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


HELP = """Lệnh trong lúc chat:
  /learn [N]   học theo Unit (bỏ N = Unit tiếp theo chưa học xong)
  /next        xong Unit này, sang Unit tiếp theo
  /progress    xem tiến độ học
  /style [mô tả | off]   đặt cách học, vd: /style mỗi lần đưa 1 từ tiếng Việt, tôi dịch sang tiếng Anh
  /new         mở phiên mới
  /help        hiện trợ giúp      /quit  thoát
Cũng có thể nói tự nhiên: "dạy tôi unit 5", "bài tiếp"."""


def _print_answer(ans) -> None:
    from .tutor import PHASE_LABEL

    print(ans.text)
    for p in ans.images:
        print("[ảnh]", p)
    if ans.unit_no and ans.part:
        label = PHASE_LABEL.get(ans.phase or "", "tóm tắt")
        print(f"[Unit {ans.unit_no} · phần {ans.part[0]}/{ans.part[1]} · {label}]")


def cmd_ask(cfg: Settings, a) -> None:
    from .tutor import Tutor

    tutor = Tutor(cfg)
    if a.session:
        sid = a.session
        try:
            tutor.store.session(sid)
        except KeyError as e:
            raise SystemExit(f"{e.args[0]}. Xem các phiên có sẵn: eng-agent sessions")
    elif a.cont and tutor.store.latest_session():
        sid = tutor.store.latest_session()
    else:
        sid = tutor.new_session()

    if a.question:
        return _print_answer(tutor.chat(sid, a.question))

    past = tutor.store.history(sid)
    print(f"Phiên #{sid} ({len(past)} tin nhắn đã lưu trong {cfg.db_path}). Gõ /help để xem lệnh.")
    for m in past[-4:]:  # nhắc lại vài tin gần nhất khi tiếp tục phiên cũ
        print(f"{'bạn' if m.role == 'user' else 'tutor'}> {m.content[:200]}")
    while True:
        try:
            q = input("\nbạn> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        cmd, _, arg = q.partition(" ")
        if cmd in ("/quit", "/exit"):
            break
        if cmd == "/help":
            print(HELP)
        elif cmd == "/new":
            sid = tutor.new_session()
            print(f"Phiên mới #{sid}")
        elif cmd == "/style":
            arg = arg.strip()
            if arg.lower() in ("off", "tắt", "xoá", "xoa"):
                tutor.store.set_style(sid, None)
                print("Đã tắt cách học riêng.")
            elif arg:
                tutor.store.set_style(sid, arg[:300])
                print(f"Đã đặt cách học: {arg[:300]}")
            else:
                print("Cách học hiện tại:", tutor.store.session(sid).get("style") or "(mặc định)")
        elif cmd == "/progress":
            print("\n".join(tutor.progress_lines()))
        elif cmd in ("/learn", "/next"):
            q = f"dạy tôi unit {arg.strip()}" if cmd == "/learn" and arg.strip().isdigit() else (
                "dạy tôi học theo unit" if cmd == "/learn" else "bài tiếp")
            _print_answer(tutor.chat(sid, q))
        else:
            try:
                _print_answer(tutor.chat(sid, q))
            except Exception as e:  # ví dụ Ollama chưa chạy; tin nhắn chưa được lưu nên gõ lại được
                print(f"Lỗi: {e}")


def cmd_sessions(cfg: Settings, a) -> None:
    from .memory import ConversationStore

    rows = ConversationStore(cfg.db_path).list_sessions()
    if not rows:
        print("Chưa có phiên nào.")
    for r in rows:
        unit = f" | đang học Unit {r['current_unit']}" if r["current_unit"] else ""
        print(f"#{r['id']:<3} {r['updated_at']}  {r['messages']:>3} tin{unit}  {r['title'] or '(trống)'}")


def cmd_history(cfg: Settings, a) -> None:
    from .memory import ConversationStore

    for m in ConversationStore(cfg.db_path).history(a.session):
        print(f"[{m.created_at}] {'bạn' if m.role == 'user' else 'tutor'}> {m.content}\n")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="eng-agent", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--root", type=Path, help="thư mục gốc chứa pages/, build/, chroma_db/ (mặc định: thư mục hiện tại)")
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
    p = sub.add_parser("build", help="ocr + chunk + index (thêm --skip-ocr để chỉ chunk + index)")
    add_ocr_opts(p)
    p.add_argument("--skip-ocr", action="store_true", help="bỏ qua OCR, dùng build/book.md có sẵn")
    p.set_defaults(fn=cmd_build)
    p = sub.add_parser("search", help="xem kết quả tìm kiếm cho một câu hỏi")
    p.add_argument("query")
    p.set_defaults(fn=cmd_search)
    p = sub.add_parser("ask", help="hội thoại với LLM local (lưu vào conversations.db)")
    p.add_argument("-q", "--question", help="hỏi một câu rồi thoát (bỏ qua để vào chế độ hội thoại)")
    p.add_argument("--start-unit", type=int, help="Unit bắt đầu học (mặc định 6, hoặc ENG_AGENT_START_UNIT); Unit đã xong thì bỏ qua")
    p.add_argument("-c", "--continue", dest="cont", action="store_true", help="tiếp tục phiên gần nhất")
    p.add_argument("-s", "--session", type=int, help="tiếp tục phiên theo số (xem bằng `eng-agent sessions`)")
    p.set_defaults(fn=cmd_ask)
    sub.add_parser("sessions", help="liệt kê các phiên đã lưu").set_defaults(fn=cmd_sessions)
    p = sub.add_parser("history", help="in toàn bộ một phiên đã lưu")
    p.add_argument("session", type=int)
    p.set_defaults(fn=cmd_history)

    a = ap.parse_args(argv)
    cfg = Settings()
    if a.root:
        cfg = replace(cfg, root=a.root)
    if getattr(a, "start_unit", None):
        cfg = replace(cfg, start_unit=a.start_unit)
    a.fn(cfg, a)


if __name__ == "__main__":
    main()
