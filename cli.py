"""Usage:
  python cli.py ingest --images ./pages --db book.db [--limit N] [--workers 4]
  python cli.py reindex --db book.db            # re-chunk from stored extraction, no vision calls
  python cli.py chat --images ./pages --db book.db --out ./sent
  (Drive instead of a local folder: --drive-folder FOLDER_ID --sa-json service_account.json)
"""
from __future__ import annotations

import argparse
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from eng_agent.store import HashEmbedder, SentenceTransformerEmbedder, Store

BASE_DIR = Path(__file__).resolve().parent
# def make_source(a):
#     from eng_agent.sources import DriveSource, LocalDirSource, drive_service
#
#     if a.drive_folder:
#         return DriveSource(drive_service(a.sa_json), a.drive_folder)
#     return LocalDirSource(a.images)
def make_source(a):
    from eng_agent.sources import LocalDirSource

    return LocalDirSource(a.images)

def make_embedder(a):
    return HashEmbedder() if a.embedder == "hash" else SentenceTransformerEmbedder()


def page_number(name: str) -> int:
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else 0


def cmd_ingest(a):
    import anthropic

    from eng_agent.extract import extract_page

    client = anthropic.Anthropic()
    source, store = make_source(a), Store(a.db, make_embedder(a))
    todo = [p for p in source.list_pages() if not store.has_page(p)]
    if a.limit:
        todo = todo[: a.limit]
    print(f"{len(todo)} pages to extract")

    def work(pid):
        try:
            return pid, extract_page(client, source.get_bytes(pid)), None
        except Exception as e:
            return pid, None, e

    with ThreadPoolExecutor(a.workers) as ex:
        for pid, page, err in ex.map(work, todo):
            if err:
                print(f"FAILED {pid}: {err}", file=sys.stderr)
                continue
            store.save_page(pid, page_number(pid), page)  # resumable: saved page by page
            print(f"ok {pid}")
    print(f"{store.build_index()} chunks indexed")


def cmd_reindex(a):
    print(f"{Store(a.db, make_embedder(a)).build_index()} chunks indexed")


def cmd_chat(a):
    import anthropic

    from eng_agent.agent import Tutor

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tutor = Tutor(anthropic.Anthropic(), Store(a.db, make_embedder(a)), make_source(a))
    n = 0
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        r = tutor.ask(q)
        print(r.text)
        for caption, png in r.images:  # the "screenshot" is produced here, on demand
            n += 1
            p = out / f"visual_{n}.png"
            p.write_bytes(png)
            print(f"[image] {caption} -> {p}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["ingest", "reindex", "chat"])
    ap.add_argument("--images", default=str(BASE_DIR / "pages"))   # đổi: mặc định là thư mục pages
    ap.add_argument("--db", default=str(BASE_DIR / "book.db"))     # đổi: db nằm cạnh cli.py
    ap.add_argument("--out", default=str(BASE_DIR / "sent"))
    # đổi: ảnh cắt gửi ra thư mục sent
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--embedder", choices=["e5", "hash"], default="e5")
    a = ap.parse_args()
    {"ingest": cmd_ingest, "reindex": cmd_reindex, "chat": cmd_chat}[a.command](a)
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("command", choices=["ingest", "reindex", "chat"])
#     ap.add_argument("--images")
#     ap.add_argument("--drive-folder")
#     ap.add_argument("--sa-json")
#     ap.add_argument("--db", default="book.db")
#     ap.add_argument("--out", default="sent")
#     ap.add_argument("--limit", type=int)
#     ap.add_argument("--workers", type=int, default=4)
#     ap.add_argument("--embedder", choices=["e5", "hash"], default="e5")
#     a = ap.parse_args()
#     if a.command != "reindex" and not (a.images or a.drive_folder):
#         ap.error("--images or --drive-folder is required")
#     {"ingest": cmd_ingest, "reindex": cmd_reindex, "chat": cmd_chat}[a.command](a)


if __name__ == "__main__":
    main()
