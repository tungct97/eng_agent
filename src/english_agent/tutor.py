"""Giai đoạn 4b - Tutor: ngữ cảnh từ retrieval + LLM local (Ollama) -> câu trả lời + ảnh."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .config import Settings
from .retrieval import Retriever, fix_text

SYSTEM = """You are an English tutor for a Vietnamese learner. Use ONLY the textbook excerpts provided.
- If the excerpts do not contain the answer, say so; do not invent textbook content.
- Explain in simple English and give the Vietnamese meaning of key words in brackets.
- Mention the unit and page you used.
- The excerpts come from OCR and may contain garbled text; ignore anything that does not make sense.
- End with one short practice question."""

NOT_FOUND = "Mình không tìm thấy nội dung phù hợp trong sách."


@dataclass
class Answer:
    text: str
    images: list[Path] = field(default_factory=list)
    units: list[str] = field(default_factory=list)


class Tutor:
    def __init__(self, cfg: Settings, retriever: Retriever | None = None, llm=None):
        self.cfg = cfg
        self.retriever = retriever or Retriever(cfg)
        if llm is None:
            from langchain_ollama import ChatOllama

            llm = ChatOllama(model=cfg.llm_model, temperature=0.2, num_ctx=cfg.llm_num_ctx)
        self.llm = llm

    def ask(self, question: str) -> Answer:
        question = fix_text(question)
        r = self.retriever.search(question)
        if not r.found:
            return Answer(NOT_FOUND)
        msg = f"Textbook excerpts:\n{r.context}\n\nQuestion: {question}"
        text = self.llm.invoke([("system", SYSTEM), ("human", msg)]).content
        return Answer(text, [p for p in r.images if p.exists()], r.unit_ids)
