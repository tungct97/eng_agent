"""Giai đoạn 4b - Tutor: hội thoại nhiều lượt với LLM local (Ollama).

Điểm khác với hỏi-đáp từng câu riêng lẻ:
  - nhớ lịch sử: các lượt trước của phiên được đưa lại cho LLM, nên "bài tập vừa rồi", "tiếp đi" đều hiểu được;
  - viết lại câu hỏi nối tiếp thành câu tìm kiếm đầy đủ trước khi tìm trong sách;
  - có chế độ học theo Unit: chọn Unit tiếp theo theo tiến độ, dạy từng phần nhỏ, chấm câu trả lời của bạn;
  - mọi tin nhắn và tiến độ được lưu vào SQLite (memory.py), mở lại phiên cũ để học tiếp.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .config import Settings
from .memory import ConversationStore, Message
from .retrieval import Retriever, assemble, fix_text, unit_key

SYSTEM = """You are a friendly English tutor for a Vietnamese learner, in an ongoing conversation.
- For textbook content use ONLY the excerpts provided in the latest message; do not invent textbook content.
  If there is no relevant excerpt, answer briefly from the conversation and say it is not from the book.
- Reply in the language the learner is using (Vietnamese or English). Keep English examples in English and give the
  Vietnamese meaning of key words in brackets.
- Use the conversation so far. Do not repeat what you already explained.
- If your previous message ended with a practice question, treat the learner's reply as their ANSWER: say whether it is
  correct, show a corrected version if needed, and explain the mistake in one or two sentences.
- Keep replies short (under about 200 words). Mention the unit and page you used.
- The excerpts come from OCR and may contain garbled text; ignore anything that does not make sense."""

LESSON_RULES = """The learner is studying ONE unit step by step.
- Teach only a small part at a time: the next section (A, B, C...) of the unit, 3-5 key words with Vietnamese meanings and
  one example from the excerpts.
- Then ask exactly ONE short practice question and wait for the answer. Do not answer it yourself.
- After the learner answers, correct it kindly and move on to the next part.
- When every section is covered, summarise the unit in three lines and tell the learner to say "bài tiếp" for the next unit."""

REWRITE = """Conversation so far:
{history}

Learner's last message: {question}

Write ONE short English search query (max 12 words) to find the relevant lesson in an English vocabulary textbook.
If the last message is an English sentence attempt or an answer, use the topic of the conversation.
Output only the query."""

NO_EXCERPT = "(no relevant excerpt found in the textbook)"
ALL_DONE = "Bạn đã học hết các Unit trong sách rồi, chúc mừng! Bạn có thể nói 'dạy unit N' để ôn lại một Unit bất kỳ."

UNIT_NO = re.compile(r"\bunit\s*(\d{1,3})\b", re.I)
LEARN = re.compile(r"\b(dạy|teach)\b|học (theo )?(từng )?unit|bắt đầu (học|bài)|start (a )?lesson|let'?s learn|learn unit", re.I)
NEXT = re.compile(r"unit (tiếp|kế|mới|sau)|bài (tiếp|kế|mới|sau)|next (unit|lesson)", re.I)


@dataclass
class Answer:
    text: str
    images: list[Path] = field(default_factory=list)
    units: list[str] = field(default_factory=list)
    mode: str = "chat"  # chat | lesson | finished
    unit_no: int | None = None  # Unit đang học (nếu có)


class Tutor:
    def __init__(self, cfg: Settings, retriever: Retriever | None = None, llm=None, store: ConversationStore | None = None):
        self.cfg = cfg
        self.retriever = retriever or Retriever(cfg)
        self.parents = self.retriever.parents
        self.store = store or ConversationStore(cfg.db_path)
        if llm is None:
            from langchain_ollama import ChatOllama

            llm = ChatOllama(model=cfg.llm_model, temperature=0.2, num_ctx=cfg.llm_num_ctx)
        self.llm = llm
        self.unit_numbers = sorted(p["unit_no"] for p in self.parents.values())

    # ------------------------------------------------------------ phiên
    def new_session(self) -> int:
        return self.store.new_session()

    def progress_lines(self) -> list[str]:
        label = {"done": "đã xong", "learning": "đang học"}
        out = [
            f"Unit {u:>3} {self.parents[unit_key(u)]['title']}: {label[s]}"
            for u, s in sorted(self.store.progress().items())
            if unit_key(u) in self.parents
        ]
        done = sum(s == "done" for s in self.store.progress().values())
        return out + [f"-> {done}/{len(self.unit_numbers)} Unit đã xong"]

    # ------------------------------------------------------------ điều hướng
    def _route(self, text: str, sess: dict) -> tuple[str, int | None, bool, tuple[int, ...]]:
        """Trả về (mode, unit đang học, vừa mở Unit mới?, các Unit cần ghim vào ngữ cảnh)."""
        current = sess["current_unit"]
        m = UNIT_NO.search(text)
        wanted = int(m.group(1)) if m and unit_key(int(m.group(1))) in self.parents else None

        if NEXT.search(text):  # chuyển Unit: đánh dấu Unit hiện tại là xong
            if current:
                self.store.set_progress(current, "done")
            nxt = self.store.next_unit(self.unit_numbers)
            return ("lesson", nxt, True, (nxt,)) if nxt else ("finished", None, False, ())
        if LEARN.search(text):  # bắt đầu / tiếp tục học theo Unit
            unit = wanted or current or self.store.next_unit(self.unit_numbers)
            if not unit:
                return "finished", None, False, ()
            return "lesson", unit, unit != current, (unit,)
        pinned = tuple(dict.fromkeys(u for u in (wanted, current) if u))  # đang học dở thì luôn bám Unit đó
        return ("lesson" if current else "chat"), current, False, pinned

    def _standalone_query(self, text: str, history: list[Message]) -> str:
        """Câu hỏi nối tiếp ('tiếp đi', câu trả lời bài tập) -> câu tìm kiếm tự đứng được."""
        if not history or not self.cfg.rewrite_queries:
            return text
        transcript = "\n".join(f"{m.role}: {m.content[:300]}" for m in history[-4:])
        try:
            out = self.llm.invoke([("human", REWRITE.format(history=transcript, question=text))]).content
        except Exception:
            return text
        out = out.strip().strip("\"'")
        return out if 3 <= len(out) <= 200 else text

    # ------------------------------------------------------------ một lượt chat
    def chat(self, session_id: int, text: str) -> Answer:
        text = fix_text(text)
        if not text:
            return Answer("")
        sess = self.store.session(session_id)
        history = self.store.history(session_id, limit=self.cfg.history_messages)
        seen_units = {u for m in self.store.history(session_id) for u in m.units}

        mode, unit, started, pinned_units = self._route(text, sess)
        if mode == "finished":
            self.store.set_current_unit(session_id, None)
            return self._save(session_id, text, Answer(ALL_DONE, mode="finished"))

        if unit:
            self.store.set_current_unit(session_id, unit)
            if self.store.progress().get(unit) != "done":
                self.store.set_progress(unit, "learning")
        pinned = tuple(unit_key(u) for u in pinned_units)

        if started:  # vừa mở Unit mới: chỉ cần chính Unit đó, không cần tìm kiếm
            r = assemble([], self.parents, False, self.cfg, pinned)
        else:
            r = self.retriever.search(self._standalone_query(text, history), pinned)

        system = SYSTEM + ("\n\n" + LESSON_RULES if unit else "")
        human = [f"Textbook excerpts (for your use only):\n{r.context or NO_EXCERPT}"]
        if started and unit:
            human.append(
                f"The learner has just started Unit {unit} ({self.parents[unit_key(unit)]['title']}). "
                "Begin the lesson now: state the goal in one line, then teach the first part."
            )
        human.append(f"Learner: {text}")

        messages = [("system", system)]
        messages += [(m.role, m.content[: self.cfg.history_chars]) for m in history]
        messages.append(("human", "\n\n".join(human)))
        reply = self.llm.invoke(messages).content

        # chỉ gửi ảnh khi có Unit mới chưa từng hiện trong phiên (tránh gửi lại ảnh cũ mỗi lượt)
        images = [p for uid in r.unit_ids if uid not in seen_units for p in r.unit_images[uid] if p.exists()]
        return self._save(session_id, text, Answer(reply, images, r.unit_ids, mode, unit))

    def _save(self, session_id: int, user_text: str, ans: Answer) -> Answer:
        self.store.add_message(session_id, "user", user_text)
        self.store.add_message(session_id, "assistant", ans.text, ans.units, ans.images)
        return ans
