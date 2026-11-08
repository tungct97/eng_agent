"""Hỏi đáp: tìm chunk -> mở rộng thành cả Unit -> LLM local (Ollama). Ảnh lấy từ metadata, không nhờ LLM.

Cài:  uv add langchain-ollama
      (và cài Ollama:  curl -fsSL https://ollama.com/install.sh | sh ;  ollama pull qwen2.5:7b-instruct)
Chạy: uv run python ask.py
"""
import json
import re

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

EMBED_MODEL = "BAAI/bge-m3"  # phải trùng với index_book.py
LLM_MODEL = "qwen2.5:7b-instruct"  # máy yếu: "qwen2.5:3b-instruct"
IMG_ROOT = "build"  # thư mục chứa images/

emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL, encode_kwargs={"normalize_embeddings": True})
db = Chroma(collection_name="evu", embedding_function=emb, persist_directory="chroma_db")
parents = json.load(open("parents.json", encoding="utf-8"))
# num_ctx quan trọng: mặc định của Ollama chỉ ~2k token, ngữ cảnh dài sẽ bị cắt âm thầm
llm = ChatOllama(model=LLM_MODEL, temperature=0.2, num_ctx=8192)

PRACTICE = re.compile(r"bài tập|luyện tập|exercise|practice|quiz|test me", re.I)

SYSTEM = """You are an English tutor for a Vietnamese learner. Use ONLY the textbook excerpts provided.
- If the excerpts do not contain the answer, say so; do not invent textbook content.
- Explain in simple English and give the Vietnamese meaning of key words in brackets.
- Mention the unit and page you used.
- The excerpts come from OCR and may contain garbled text; ignore anything that does not make sense.
- End with one short practice question."""


def retrieve(question: str, k: int = 6, max_units: int = 2):
    practice = bool(PRACTICE.search(question))
    kinds = ["exercise"] if practice else ["reference", "toc"]
    hits = db.similarity_search_with_score(question, k=k, filter={"chunk_type": {"$in": kinds}})
    unit_ids, figures = [], []
    for doc, _ in hits:
        pid = doc.metadata["parent_id"]
        if pid and pid not in unit_ids and len(unit_ids) < max_units:
            unit_ids.append(pid)
        if pid in unit_ids and doc.metadata["figure_images"]:
            figures += doc.metadata["figure_images"].split("|")
    return practice, unit_ids, figures


def build_context(unit_ids: list[str], practice: bool) -> str:
    parts = []
    for pid in unit_ids:
        p = parents[pid]
        page = p["pages"][1] if practice else p["pages"][0]
        body = p["exercises"] if practice else p["reference"]
        parts.append(f"[Unit {p['unit_no']}: {p['title']} - scan page {page}]\n{body[:4500]}")
    return "\n\n".join(parts)


def ask(question: str) -> tuple[str, list[str]]:
    practice, unit_ids, figures = retrieve(question)
    if not unit_ids:
        return "Mình không tìm thấy nội dung phù hợp trong sách.", []
    ctx = build_context(unit_ids, practice)
    answer = llm.invoke([("system", SYSTEM), ("human", f"Textbook excerpts:\n{ctx}\n\nQuestion: {question}")]).content
    images = [f"{IMG_ROOT}/images/page_{parents[u]['pages'][1 if practice else 0]:04d}.jpg" for u in unit_ids]
    images += [f"{IMG_ROOT}/{f}" for f in dict.fromkeys(figures)]
    return answer, images


if __name__ == "__main__":
    while True:
        try:
            q = input("\nbạn> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q:
            text, imgs = ask(q)
            print(text)
            for p in imgs:
                print("[ảnh]", p)
