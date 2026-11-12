"""Test hội thoại: bộ nhớ SQLite, lịch sử đưa lại cho LLM, học theo Unit từng phần. Không cần Ollama/Chroma."""
import sqlite3

from langchain_core.documents import Document

from english_agent.config import Settings
from english_agent.memory import ConversationStore
from english_agent.retrieval import assemble
from english_agent.tutor import (
    ALL_DONE, CONTINUE_RE, LEARN, LESSON_RULES, NEXT, QUESTION, STYLE_NOTE, STYLE_RE, Tutor, lesson_parts, lesson_plan,
)


def sec(title, word, n=300):
    return f"### {title}\n\n" + (f"{word} " * (n // (len(word) + 1))).strip()


REF1 = "## Greetings\n\n" + "\n\n".join([sec("A Saying hello", "hello"), sec("B Saying bye", "goodbye"), sec("C Thanking", "thanks")])
PARENTS = {
    "unit-001": {"unit_no": 1, "title": "Greetings", "pages": [2, 3], "reference": REF1, "exercises": "1.1 ..."},
    "unit-002": {"unit_no": 2, "title": "Health: injuries", "pages": [4, 5], "reference": "bruise", "exercises": "2.1 ..."},
}


class FakeRetriever:
    def __init__(self, cfg):
        self.cfg, self.parents, self.calls = cfg, PARENTS, []

    def search(self, q, pinned=()):
        self.calls.append((q, pinned))
        hit = Document(page_content="x", metadata={"parent_id": "unit-002", "figure_images": ""})
        return assemble([(hit, 0.3)], PARENTS, False, self.cfg, pinned)


class Msg:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    def __init__(self):
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        if "search query" in messages[0][1]:  # lời gọi viết lại câu hỏi
            return Msg("bruise injury treatment")
        return Msg(f"reply {len(self.calls)}")

    def main_calls(self):
        return [c for c in self.calls if c[0][0] == "system"]

    def human(self, i=-1):  # tin nhắn cuối (có đoạn trích + chỉ dẫn) của lượt gọi chính thứ i
        return self.main_calls()[i][-1][1]


def make(tmp_path, db=":memory:"):
    cfg = Settings(root=tmp_path)
    llm, store = FakeLLM(), ConversationStore(db)
    return cfg, llm, store, Tutor(cfg, FakeRetriever(cfg), llm, store)


# ----------------------------------------------------------------- chia Unit thành các phần
def test_lesson_parts_follow_sections():
    parts = lesson_parts(REF1)
    assert len(parts) == 3
    assert "hello hello" in parts[0] and "goodbye" in parts[1] and "thanks" in parts[2]
    assert "goodbye" not in parts[0]


def test_lesson_parts_merge_short_and_split_long():
    assert len(lesson_parts("## T\n\n### A x\n\nshort\n\n### B y\n\nalso short")) == 1
    long = "## T\n\n" + "\n\n".join(f"paragraph {i} " + "word " * 80 for i in range(6))  # ~2.4k ký tự, một tiêu đề
    parts = lesson_parts(long)
    assert len(parts) >= 2 and all(len(p) <= 1300 for p in parts)


# ----------------------------------------------------------------- kho lưu SQLite
def test_store_orders_messages_limits_and_titles(tmp_path):
    store = ConversationStore(tmp_path / "c.db")
    sid = store.new_session()
    for i in range(5):
        store.add_message(sid, "user" if i % 2 == 0 else "assistant", f"m{i}", units=["unit-001"], images=["a.jpg"])
    assert [m.content for m in store.history(sid)] == ["m0", "m1", "m2", "m3", "m4"]
    assert [m.content for m in store.history(sid, limit=2)] == ["m3", "m4"]
    assert store.history(sid, limit=0) == []
    assert store.history(sid)[0].units == ["unit-001"] and store.history(sid)[0].images == ["a.jpg"]
    row = store.list_sessions()[0]
    assert row["messages"] == 5 and row["title"] == "m0"


def test_store_survives_reopen_and_tracks_progress(tmp_path):
    path = tmp_path / "c.db"
    store = ConversationStore(path)
    sid = store.new_session()
    store.add_message(sid, "user", "hello")
    store.set_current_unit(sid, 3)
    store.set_progress(1, "done")
    store.set_progress(2, "learning")
    store.set_progress(2, "done")
    store.close()

    again = ConversationStore(path)
    assert again.latest_session() == sid
    assert again.session(sid)["current_unit"] == 3
    assert again.history(sid)[0].content == "hello"
    assert again.progress() == {1: "done", 2: "done"}
    assert again.next_unit([1, 2, 3]) == 3 and again.next_unit([1, 2]) is None


def test_lesson_step_resets_only_when_unit_changes(tmp_path):
    store = ConversationStore(":memory:")
    sid = store.new_session()
    store.set_current_unit(sid, 5)
    store.set_lesson_step(sid, 2)
    store.set_current_unit(sid, 5)  # cùng Unit: giữ nguyên bước
    assert store.session(sid)["lesson_step"] == 2
    store.set_current_unit(sid, 6)  # đổi Unit: về 0
    assert store.session(sid)["lesson_step"] == 0


def test_old_database_without_lesson_step_is_migrated(tmp_path):
    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.executescript(
        "CREATE TABLE sessions(id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT, current_unit INTEGER,"
        " created_at TEXT NOT NULL, updated_at TEXT NOT NULL);"
        "INSERT INTO sessions(title, current_unit, created_at, updated_at) VALUES ('cũ', 5, 'x', 'x');"
    )
    old.commit()
    old.close()
    store = ConversationStore(path)
    assert store.session(1)["lesson_step"] == 0 and store.session(1)["current_unit"] == 5
    assert store.session(1)["style"] is None


# ----------------------------------------------------------------- điều hướng
def test_intent_patterns():
    for text in ["bạn có thể dạy tôi học tiếng anh được không?", "bạn hãy dạy tôi theo từng unit được khong?",
                 "dạy tôi unit 5", "học theo từng unit", "teach me", "start a lesson"]:
        assert LEARN.search(text), text
    for text in ["bài tiếp", "unit tiếp theo", "next unit", "Next lesson please"]:
        assert NEXT.search(text), text
    for text in ["tiếp tục", "tiếp đi", "continue", "phần tiếp"]:
        assert CONTINUE_RE.search(text), text
    for text in ["bruise nghĩa là gì", "what does swap mean?", "giải thích giúp tôi"]:
        assert QUESTION.search(text), text
    # câu trả lời bài tập KHÔNG được bị coi là lệnh hay câu hỏi
    for text in ["dọn cái bảng", "đổi vị trí với ai đó", "I went to school, you let stay home", "Dọn cái bảng"]:
        assert not (LEARN.search(text) or NEXT.search(text) or CONTINUE_RE.search(text) or QUESTION.search(text)), text


# ----------------------------------------------------------------- bài học từng phần
def test_lesson_plan_order():
    plan = lesson_plan(3)
    assert [p for p, _ in plan[:3]] == ["teach"] * 3 and [i for _, i in plan[:3]] == [0, 1, 2]  # học hết từ vựng trước
    assert plan[3:9] == [("words", 0)] * 2 + [("words", 1)] * 2 + [("words", 2)] * 2
    assert plan[9:] == [("sentences", 0)] * 2 + [("sentences", 1)] * 2 + [("sentences", 2)] * 2
    assert len(lesson_plan(1)) == 5


def test_lesson_walks_through_plan_without_repeating(tmp_path):
    """Mỗi lượt đúng MỘT hoạt động: giới thiệu từng phần, rồi từ Việt -> Anh, rồi câu Việt -> Anh; không tìm thêm Unit lạ."""
    cfg, llm, store, tutor = make(tmp_path)
    sid = tutor.new_session()

    a1 = tutor.chat(sid, "bạn có thể dạy tôi học tiếng anh được không?")
    assert (a1.mode, a1.unit_no, a1.part, a1.phase) == ("lesson", 1, (1, 3), "teach")
    assert "hello hello" in llm.human() and "goodbye" not in llm.human()  # chỉ phần 1
    assert "PHASE 1" in llm.human() and "just started this unit" in llm.human()

    a2 = tutor.chat(sid, "tiếp tục")
    assert a2.part == (2, 3) and a2.phase == "teach" and "goodbye" in llm.human() and "hello hello" not in llm.human()
    a3 = tutor.chat(sid, "tiếp tục")
    assert a3.part == (3, 3) and "thanks" in llm.human()

    a4 = tutor.chat(sid, "ok")  # bắt đầu luyện từ: phần 1
    assert a4.phase == "words" and a4.part == (1, 3) and "PHASE 2" in llm.human() and "hello hello" in llm.human()
    a5 = tutor.chat(sid, "hello")  # trả lời -> chấm + nêu đáp án + câu từ kế tiếp
    assert a5.phase == "words" and "judge the learner's reply" in llm.human()
    assert "correct answer yourself" in llm.human()

    for i in range(9):  # đến bước cuối của kế hoạch (bước 15)
        tutor.chat(sid, "answer")
    last = tutor.chat(sid, "answer")
    assert last.phase == "sentences" and last.part == (3, 3) and "PHASE 3" in llm.human()

    fin = tutor.chat(sid, "answer")  # hết kế hoạch -> tóm tắt, đánh dấu xong
    assert fin.part == (3, 3) and fin.phase is None and "summarise the unit" in llm.human()
    assert store.progress() == {1: "done"}
    assert tutor.retriever.calls == []
    assert all("exercise 1." not in c[-1][1] for c in llm.main_calls())


def test_after_summary_it_does_not_summarise_again(tmp_path):
    cfg, llm, store, tutor = make(tmp_path)
    sid = tutor.new_session()
    for text in ["dạy tôi học"] + ["a"] * len(lesson_plan(3)):  # cả kế hoạch + tóm tắt
        tutor.chat(sid, text)
    ans = tutor.chat(sid, "cảm ơn")  # Unit đã xong: trả lời tự do (có tìm sách), không tóm tắt lại
    assert ans.part is None and "summarise" not in llm.human()
    assert len(tutor.retriever.calls) == 1


def test_side_question_does_not_advance_lesson(tmp_path):
    cfg, llm, store, tutor = make(tmp_path)
    sid = tutor.new_session()
    tutor.chat(sid, "dạy tôi học")
    ans = tutor.chat(sid, "bruise nghĩa là gì?")
    assert ans.part is None and store.session(sid)["lesson_step"] == 1  # vẫn đứng ở bước 2
    assert tutor.retriever.calls == [("bruise injury treatment", ("unit-001",))]  # câu hỏi được viết lại + ghim Unit
    assert "side question" in llm.main_calls()[-1][0][1]
    nxt = tutor.chat(sid, "dọn cái bảng")
    assert nxt.part == (2, 3)


def test_continue_moves_on_without_grading(tmp_path):
    cfg, llm, store, tutor = make(tmp_path)
    sid = tutor.new_session()
    tutor.chat(sid, "dạy tôi học")
    ans = tutor.chat(sid, "tiếp tục")
    assert ans.part == (2, 3) and "judge" not in llm.human()


def test_lesson_step_survives_restart(tmp_path):
    db = tmp_path / "c.db"
    cfg, llm, store, tutor = make(tmp_path, db)
    sid = tutor.new_session()
    tutor.chat(sid, "dạy tôi học")
    store.close()

    cfg2, llm2, store2, tutor2 = make(tmp_path, db)
    ans = tutor2.chat(sid, "dọn cái bảng")
    assert ans.part == (2, 3)  # nhớ đã dạy tới bước 1
    msgs = llm2.main_calls()[0]
    assert [m[0] for m in msgs[:3]] == ["system", "user", "assistant"]  # lịch sử cũ được nạp lại


def test_next_marks_done_then_finishes(tmp_path):
    cfg, llm, store, tutor = make(tmp_path)
    sid = tutor.new_session()
    tutor.chat(sid, "dạy tôi học")
    ans = tutor.chat(sid, "bài tiếp")
    assert ans.unit_no == 2 and ans.part == (1, 1) and store.progress() == {1: "done", 2: "learning"}
    fin = tutor.chat(sid, "bài tiếp")
    assert fin.mode == "finished" and fin.text == ALL_DONE
    assert store.progress() == {1: "done", 2: "done"} and store.session(sid)["current_unit"] is None


def test_learn_specific_unit(tmp_path):
    cfg, llm, store, tutor = make(tmp_path)
    sid = tutor.new_session()
    first = tutor.chat(sid, "dạy tôi unit 2")
    assert first.unit_no == 2 and first.part == (1, 1)
    assert "unit-002" in tutor.chat(sid, "dạy tôi unit 2").units  # nói lại: tiếp tục Unit 2, không mở lại từ đầu


# ----------------------------------------------------------------- chat tự do
def test_free_chat_has_no_lesson_rules_and_no_rewrite_without_history(tmp_path):
    cfg, llm, store, tutor = make(tmp_path)
    sid = tutor.new_session()
    ans = tutor.chat(sid, "bruise nghĩa là gì")
    assert ans.mode == "chat" and ans.unit_no is None
    assert tutor.retriever.calls == [("bruise nghĩa là gì", ())]
    assert "one unit step by step" not in llm.main_calls()[0][0][1]


def test_images_are_sent_only_for_new_units(tmp_path):
    cfg, llm, store, tutor = make(tmp_path)
    (cfg.build_dir / "images").mkdir(parents=True)
    (cfg.build_dir / "images" / "page_0002.jpg").write_bytes(b"x")
    sid = tutor.new_session()
    assert [p.name for p in tutor.chat(sid, "dạy tôi học").images] == ["page_0002.jpg"]
    assert tutor.chat(sid, "dọn cái bảng").images == []  # Unit 1 đã gửi ảnh rồi


def test_history_is_trimmed(tmp_path):
    cfg, llm, store, tutor = make(tmp_path)
    sid = tutor.new_session()
    for i in range(10):
        tutor.chat(sid, f"câu {i}")
    msgs = llm.main_calls()[-1]
    assert len(msgs) == 1 + cfg.history_messages + 1


def test_cli_lists_and_prints_saved_sessions(tmp_path, capsys):
    from english_agent.cli import main

    store = ConversationStore(tmp_path / "conversations.db")
    sid = store.new_session()
    store.add_message(sid, "user", "dạy tôi unit 5")
    store.add_message(sid, "assistant", "Unit 5 nói về ...")
    store.set_current_unit(sid, 5)
    store.close()

    main(["--root", str(tmp_path), "sessions"])
    out = capsys.readouterr().out
    assert f"#{sid}" in out and "2 tin" in out and "đang học Unit 5" in out and "dạy tôi unit 5" in out
    main(["--root", str(tmp_path), "history", str(sid)])
    out = capsys.readouterr().out
    assert "bạn> dạy tôi unit 5" in out and "tutor> Unit 5 nói về" in out


# ----------------------------------------------------------------- cách học do người học yêu cầu
ASK = "hãy dạy tôi tiếng anh, bằng cách đưa ra 1 từ vựng và bắt tôi dịch nó sang tiếng anh"


def test_style_pattern():
    assert STYLE_RE.search(ASK)
    for text in ["dictionary nghĩa là từ điển", "dọn cái bảng", "dạy tôi học"]:
        assert not STYLE_RE.search(text)


def test_style_is_noted_in_system_prompt_and_persisted(tmp_path):
    db = tmp_path / "s.db"
    cfg, llm, store, tutor = make(tmp_path, db)
    sid = tutor.new_session()
    tutor.chat(sid, ASK)
    assert store.session(sid)["style"] == ASK
    assert ASK in llm.main_calls()[-1][0][1] and "can NOT see the textbook" in LESSON_RULES
    store.close()
    cfg2, llm2, store2, tutor2 = make(tmp_path, db)  # cách học được nhớ sau khi khởi động lại
    tutor2.chat(sid, "dọn cái bảng")
    assert ASK in llm2.main_calls()[-1][0][1]


def test_long_plan_message_is_not_stored_as_style(tmp_path):
    cfg, llm, store, tutor = make(tmp_path)
    sid = tutor.new_session()
    tutor.chat(sid, "hãy giúp tôi học theo từng unit, bằng cách " + "giới thiệu từ vựng rồi cho tôi làm bài tập " * 6)
    assert store.session(sid)["style"] is None
