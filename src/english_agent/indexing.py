"""Giai đoạn 3 - Indexing: chunks.jsonl -> embedding -> Chroma.

Vào : data/build/chunks.jsonl (kết quả của giai đoạn chunking)
Ra   : data/chroma_db/
Dùng get_embeddings()/get_db() chung với retrieval để embedding lúc lưu và lúc hỏi luôn là cùng một mô hình.
"""
from __future__ import annotations

from .chunking import load_chunks
from .config import Settings


def get_embeddings(cfg: Settings):
    from langchain_huggingface import HuggingFaceEmbeddings

    # normalize -> vector độ dài 1, cosine ổn định. bge-m3 không cần tiền tố query:/passage:
    return HuggingFaceEmbeddings(model_name=cfg.embed_model, encode_kwargs={"normalize_embeddings": True})


def get_db(cfg: Settings, embeddings=None):
    from langchain_chroma import Chroma

    return Chroma(
        collection_name=cfg.collection,
        embedding_function=embeddings or get_embeddings(cfg),
        persist_directory=str(cfg.chroma_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )


def build_index(cfg: Settings, batch: int = 64) -> int:
    """Xoá và tạo lại collection từ chunks.jsonl (chạy lại không bị trùng). Trả về số chunk đã lưu."""
    if not cfg.chunks_path.exists():
        raise FileNotFoundError(f"Chưa có {cfg.chunks_path}. Chạy trước: eng-agent chunk")
    docs = load_chunks(cfg.chunks_path)
    emb = get_embeddings(cfg)
    db = get_db(cfg, emb)
    db.delete_collection()
    db = get_db(cfg, emb)

    ids = [f"{d.metadata['chunk_type']}-{d.metadata['unit_no']}-{i}" for i, d in enumerate(docs)]
    for s in range(0, len(docs), batch):
        db.add_documents(docs[s : s + batch], ids=ids[s : s + batch])
        print(f"  đã lưu {min(s + batch, len(docs))}/{len(docs)}", flush=True)
    return len(docs)
