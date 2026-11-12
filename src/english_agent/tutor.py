"""Giai đoạn 4b - Tutor: hội thoại nhiều lượt với LLM local (Ollama).

  - nhớ lịch sử: các lượt trước của phiên được đưa lại cho LLM;
  - viết lại câu hỏi nối tiếp thành câu tìm kiếm đầy đủ trước khi tìm trong sách;
  - học theo Unit: CODE (không phải LLM) quyết định mỗi lượt dạy phần nào. Unit được chia thành các phần nhỏ,
    bước học được lưu trong DB, mỗi lượt chỉ đưa đúng MỘT phần cho LLM. Nhờ vậy LLM không thể lặp lại bài cũ
    hay quên mình đã dạy tới đâu (mô hình 7B làm việc này rất kém nếu phải tự nhớ);
  - mọi tin nhắn, bước học và tiến độ được lưu vào SQLite (memory.py).
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
- If your previous message ended with a practice question, treat the learner's reply as their ANSWER and judge it
  honestly: say it is correct only if it really is; otherwise show the correct answer and explain the mistake briefly.
- Keep replies short (under about 200 words). Mention the unit and page you used.
- The excerpts come from OCR and may contain garbled text; ignore anything that does not make sense."""

LESSON_RULES = """The learner is studying one unit step by step. The program decides which part of the unit and which kind of
activity to use in each turn: follow the instructions for this turn exactly, do only that one activity, and never repeat
a word, sentence or question that already appears in the conversation.
- The learner can NOT see the textbook. Never tell them to look at or do an exercise, a page or a number (such as
  "exercise 3.1"); you write every word, sentence and question out in full yourself, using ONLY vocabulary from the excerpt.
- After the learner answers, ALWAYS state the correct answer yourself. Never ask the learner to check it or to report progress."""

STYLE_NOTE = (
    "The learner also asked for this way of studying; follow it where it does not conflict with the instructions for this turn:\n"
    '"{style}"'
)

LESSON_SIDE = """The learner asked a side question in the middle of a lesson. Answer it briefly from the excerpts, then remind
them to answer your last question, or to say "tiếp tục" to go on."""

# Mỗi Unit chia thành các phần (lesson_parts). Kế hoạch học: 1) giới thiệu từ vựng từng phần,
# 2) từ tiếng Việt -> viết tiếng Anh, 3) câu tiếng Việt -> dịch sang tiếng Anh. Code chọn bước, LLM chỉ thực hiện.
WORDS_PER_PART = 2      # số lượt "từ Việt -> Anh" cho mỗi phần
SENTENCES_PER_PART = 2  # số lượt "câu Việt -> Anh" cho mỗi phần


def lesson_plan(n_parts: int) -> list[tuple[str, int]]:
    plan = [("teach", i) for i in range(n_parts)]
    plan += [("words", i) for i in range(n_parts) for _ in range(WORDS_PER_PART)]
    plan += [("sentences", i) for i in range(n_parts) for _ in range(SENTENCES_PER_PART)]
    return plan


PHASE_TASK = {
    "teach": (
        "PHASE 1 - vocabulary. Introduce ONLY the part above. Pick 4-6 real vocabulary words or phrases that appear in it "
        "(never headings, labels, grammar terms or instructions of the book). Use EXACTLY this layout, one block per word:\n"
        "1. **english word** /pronunciation if the excerpt gives it/ - Vietnamese meaning\n"
        "   Example: one short ENGLISH sentence using the word (take the book's own example when the excerpt has one)\n"
        "   Nghĩa: Vietnamese translation of that sentence\n"
        "Every word MUST have an English example sentence and its Vietnamese translation; never skip them and never write the "
        "example in Vietnamese only. Do not ask any question; end by telling the learner to say \"tiếp tục\" when they have "
        "memorised these words."
    ),
    "words": (
        "PHASE 2 - word practice. Give ONE Vietnamese word or phrase taken from the part above and ask the learner to write "
        "it in English. Write the Vietnamese item out in full, choose one not used earlier in this conversation, then wait."
    ),
    "sentences": (
        "PHASE 3 - sentence practice. Write ONE short Vietnamese sentence that uses vocabulary from the part above and ask the "
        "learner to translate it into English. Write the sentence out in full, make it new, then wait."
    ),
}
PHASE_LABEL = {"teach": "từ vựng", "words": "viết từ", "sentences": "dịch câu"}


def lesson_instruction(kind: str, phase: str | None, summary: bool) -> str:
    """Chỉ dẫn cho LLM ở lượt này. kind: start | continue | reply."""
    judge = (
        "First judge the learner's reply to your previous question honestly (it is an answer, not a new request), "
        "say the correct answer yourself and correct any mistake briefly. "
        if kind == "reply"
        else ""
    )
    if summary:
        return judge + (
            'All parts of this unit have been taught and practised: summarise the unit in three lines and tell the learner to say '
            '"bài tiếp" for the next unit. Do not ask more questions.'
        )
    opening = "The learner has just started this unit. State the goal of the unit in one line. " if kind == "start" else ""
    return opening + judge + PHASE_TASK[phase or "teach"]


REWRITE = """Conversation so far:
{history}

Learner's last message: {question}

Write ONE short English search query (max 12 words) to find the relevant lesson in an English vocabulary textbook.
If the last message is an English sentence attempt or an answer, use the topic of the conversation.
Output only the query."""

NO_EXCERPT = "(no relevant excerpt found in the textbook)"
ALL_DONE = "Bạn đã học hết các Unit trong sách rồi, chúc mừng! Bạn có thể nói 'dạy unit N' để ôn lại một Unit bất kỳ."

UNIT_NO = re.compile(r"\bunit\s*(\d{1,3})\b", re.I)
LEARN = re.compile(
    r"\b(dạy|teach)\b|\bhọc\b[^.?!\n]{0,40}\bunit\b|theo (từng )?unit|(giúp|muốn|cho|hãy|cùng) (tôi |mình |em )?(học|luyện)\b|"
    r"bắt đầu (học|bài)|start (a )?lesson|let'?s learn|learn unit",
    re.I,
)
NEXT = re.compile(r"unit (tiếp|kế|mới|sau)|bài (tiếp|kế|mới|sau)|next (unit|lesson)", re.I)
STYLE_RE = re.compile(
    r"bằng cách|theo cách|bắt tôi|hỏi tôi|đố tôi|cho tôi (làm|dịch)|mỗi lần|từng (từ|câu)\b|một từ|1 từ|"
    r"one (word|question|at a time)|quiz me|test me|flashcard",
    re.I,
)
CONTINUE_RE = re.compile(r"^\s*(tiếp tục|tiếp đi|tiếp nào|tiếp|continue|go on|next part|phần (tiếp|sau|kế))\b", re.I)
QUESTION = re.compile(r"\?|là gì|nghĩa là|nghĩa của|giải thích|what (does|is)|how (do|to)|explain|cho (tôi )?ví dụ", re.I)

MIN_PART, MAX_PART = 250, 1100  # độ dài (ký tự) của mỗi phần dạy


def lesson_parts(reference: str) -> list[str]:
    """Chia nội dung một Unit thành các phần nhỏ để dạy lần lượt (theo tiêu đề A/B/C; gộp phần quá ngắn, cắt phần quá dài)."""
    pieces = [p.strip() for p in re.split(r"\n(?=#{2,4} )", reference) if p.strip()]
    merged: list[str] = []
    for p in pieces:
        if merged and (len(merged[-1]) < MIN_PART or len(p) < MIN_PART):
            merged[-1] += "\n\n" + p
        else:
            merged.append(p)
    parts: list[str] = []
    for m in merged:
        if len(m) <= MAX_PART:
            parts.append(m)
            continue
        cur = ""
        for para in re.split(r"\n{2,}", m):
            if cur and len(cur) + len(para) > MAX_PART:
                parts.append(cur)
                cur = para
            else:
                cur = f"{cur}\n\n{para}".strip()
        if cur:
            parts.append(cur)
    return parts


@dataclass
class Route:
    mode: str  # chat | lesson | finished
    unit: int | None = None
    kind: str = "chat"  # chat | start | continue | reply | side
    pinned: tuple[int, ...] = ()


@dataclass
class Answer:
    text: str
    images: list[Path] = field(default_factory=list)
    units: list[str] = field(default_factory=list)
    mode: str = "chat"  # chat | lesson | finished
    unit_no: int | None = None  # Unit đang học (nếu có)
    part: tuple[int, int] | None = None  # (phần vừa dạy, tổng số phần); (n, n) = đã dạy xong
    phase: str | None = None  # teach | words | sentences


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
        self._parts: dict[str, list[str]] = {}

    # ------------------------------------------------------------ phiên
    def new_session(self) -> int:
        return self.store.new_session()

    def progress_lines(self) -> list[str]:
        label = {"done": "đã xong", "learning": "đang học"}
        prog = self.store.progress()
        out = [
            f"Unit {u:>3} {self.parents[unit_key(u)]['title']}: {label[s]}"
            for u, s in sorted(prog.items())
            if unit_key(u) in self.parents
        ]
        done = sum(s == "done" for s in prog.values())
        return out + [f"-> {done}/{len(self.unit_numbers)} Unit đã xong"]

    def parts_of(self, unit: int) -> list[str]:
        key = unit_key(unit)
        if key not in self._parts:
            self._parts[key] = lesson_parts(self.parents[key]["reference"]) or [self.parents[key]["reference"]]
        return self._parts[key]

    # ------------------------------------------------------------ điều hướng
    def _route(self, text: str, sess: dict) -> Route:
        current = sess["current_unit"]
        m = UNIT_NO.search(text)
        wanted = int(m.group(1)) if m and unit_key(int(m.group(1))) in self.parents else None

        if NEXT.search(text):  # chuyển Unit: đánh dấu Unit hiện tại là xong
            if current:
                self.store.set_progress(current, "done")
            nxt = self.store.next_unit(self.unit_numbers)
            return Route("lesson", nxt, "start", (nxt,)) if nxt else Route("finished")
        if LEARN.search(text):  # bắt đầu / tiếp tục học theo Unit
            unit = wanted or current or self.store.next_unit(self.unit_numbers)
            if not unit:
                return Route("finished")
            return Route("lesson", unit, "continue" if unit == current else "start", (unit,))
        if not current:
            return Route("chat", None, "chat", tuple(u for u in (wanted,) if u))
        if QUESTION.search(text) or (wanted and wanted != current):  # câu hỏi phụ giữa chừng bài học
            return Route("lesson", current, "side", tuple(dict.fromkeys(u for u in (wanted, current) if u)))
        if CONTINUE_RE.search(text):
            return Route("lesson", current, "continue", (current,))
        return Route("lesson", current, "reply", (current,))  # còn lại = câu trả lời cho câu luyện tập vừa hỏi

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

        style = sess.get("style") or ""
        if len(text) <= 160 and STYLE_RE.search(text):  # "hãy dạy tôi bằng cách đưa 1 từ rồi bắt tôi dịch" -> nhớ cho cả phiên
            style = text[:300]
            self.store.set_style(session_id, style)

        route = self._route(text, sess)
        if route.mode == "finished":
            self.store.set_current_unit(session_id, None)
            return self._save(session_id, text, Answer(ALL_DONE, mode="finished"))

        unit = route.unit
        if unit:
            if sess["current_unit"] != unit:
                self.store.set_current_unit(session_id, unit)  # đổi Unit -> bước học về 0
            if self.store.progress().get(unit) != "done":
                self.store.set_progress(unit, "learning")
        step = self.store.session(session_id)["lesson_step"] if unit else 0
        pinned = tuple(unit_key(u) for u in route.pinned)

        part, cur_phase = None, None
        system, instr = SYSTEM, ""
        kind = route.kind
        if unit and kind in ("start", "continue", "reply"):
            total = len(lesson_plan(len(self.parts_of(unit))))
            if kind == "start":
                step = 0
            if step >= total and kind != "start" and self.store.progress().get(unit) == "done":
                kind = "side"  # Unit đã dạy xong và đã tóm tắt: chỉ trả lời tự do, không tóm tắt lại

        if unit and kind in ("start", "continue", "reply"):
            parts, title = self.parts_of(unit), self.parents[unit_key(unit)]["title"]
            n = len(parts)
            plan = lesson_plan(n)
            r = assemble([], self.parents, False, self.cfg, pinned)  # chỉ dùng để lấy Unit + ảnh; KHÔNG tìm thêm Unit khác
            system += "\n\n" + LESSON_RULES
            if style:
                system += "\n\n" + STYLE_NOTE.format(style=style)
            if step < len(plan):
                phase, idx = plan[step]
                page = self.parents[unit_key(unit)]["pages"][0]
                context = f"[Unit {unit}: {title} - part {idx + 1} of {n} - scan page {page}]\n{parts[idx]}"
                instr = lesson_instruction(kind, phase, summary=False)
                self.store.set_lesson_step(session_id, step + 1)
                part = (idx + 1, n)
                cur_phase = phase
            else:  # đã dạy hết các phần -> tóm tắt, đánh dấu xong
                context = f"[Unit {unit}: {title} - all {n} parts have been taught]"
                instr = lesson_instruction(kind, None, summary=True)
                self.store.set_progress(unit, "done")
                part = (n, n)
        else:
            r = self.retriever.search(self._standalone_query(text, history), pinned)
            context = r.context
            if unit and kind == "side":
                system += "\n\n" + LESSON_SIDE

        human = [f"Textbook excerpts (for your use only):\n{context or NO_EXCERPT}"]
        if instr:
            human.append(f"Instructions for this turn: {instr}")
        human.append(f"Learner: {text}")

        messages = [("system", system)]
        messages += [(m.role, m.content[: self.cfg.history_chars]) for m in history]
        messages.append(("human", "\n\n".join(human)))
        reply = self.llm.invoke(messages).content

        # chỉ gửi ảnh khi có Unit mới chưa từng hiện trong phiên (tránh gửi lại ảnh cũ mỗi lượt)
        images = [p for uid in r.unit_ids if uid not in seen_units for p in r.unit_images[uid] if p.exists()]
        mode = "lesson" if unit else "chat"
        return self._save(session_id, text, Answer(reply, images, r.unit_ids, mode, unit, part, cur_phase))

    def _save(self, session_id: int, user_text: str, ans: Answer) -> Answer:
        self.store.add_message(session_id, "user", user_text)
        self.store.add_message(session_id, "assistant", ans.text, ans.units, ans.images)
        return ans
