"""Embed các chunk và lưu vào Chroma (chạy 1 lần; chạy lại sẽ xoá và tạo mới).

Cài:  uv add langchain-huggingface langchain-chroma sentence-transformers
Chạy: uv run python index_book.py            # tạo chroma_db/ và parents.json
      uv run python index_book.py --test     # thêm: thử vài câu hỏi để xem tìm kiếm có ổn không
"""
import json
import sys

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from chunk_book import build_documents

MD_PATH = "build/book.md"
EMBED_MODEL = "BAAI/bge-m3"  # đa ngôn ngữ, không cần tiền tố query/passage; nhẹ hơn: "intfloat/multilingual-e5-base"
DB_DIR, COLLECTION = "chroma_db", "evu"


def make_embeddings() -> HuggingFaceEmbeddings:
    # normalize -> vector độ dài 1, dùng cosine ổn định
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL, encode_kwargs={"normalize_embeddings": True})


def make_db(emb: HuggingFaceEmbeddings) -> Chroma:
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=emb,
        persist_directory=DB_DIR,
        collection_metadata={"hnsw:space": "cosine"},
    )


def main() -> None:
    docs, parents = build_documents(MD_PATH)
    with open("parents.json", "w", encoding="utf-8") as f:
        json.dump(parents, f, ensure_ascii=False)
    print(f"{len(docs)} chunks, {len(parents)} units")

    emb = make_embeddings()
    db = make_db(emb)
    db.delete_collection()  # chạy lại không bị trùng
    db = make_db(emb)

    ids = [f"{d.metadata['chunk_type']}-{d.metadata['unit_no']}-{i}" for i, d in enumerate(docs)]
    for s in range(0, len(docs), 64):
        db.add_documents(docs[s : s + 64], ids=ids[s : s + 64])
        print(f"  đã lưu {min(s + 64, len(docs))}/{len(docs)}", flush=True)

    if "--test" in sys.argv:
        for q in ["vết bầm tím và bong gân nói thế nào", "bruise", "how to say sorry I'm late", "giới từ chỉ nơi chốn"]:
            print(f"\n? {q}")
            for d, dist in db.similarity_search_with_score(q, k=3):
                m = d.metadata
                print(f"  {dist:.3f}  unit {m['unit_no']} {m['unit_title'][:40]} [{m['chunk_type']}]")


if __name__ == "__main__":
    main()
