"""Test hội thoại: bộ nhớ SQLite, lịch sử đưa lại cho LLM, học theo Unit từng phần. Không cần Ollama/Chroma."""
import json
import sqlite3

from langchain_core.documents import Document

from english_agent.config import Settings
from english_agent.memory import ConversationStore
from english_agent.retrieval import assemble
from english_agent.tutor import (
    ALL_DONE, CONTINUE_RE, LEARN, LESSON_RULES, NEXT, QUESTION, STYLE_NOTE, STYLE_RE, Tutor, lesson_parts, practice_phase,
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
    cfg = Settings(root=tmp_path, start_unit=1)
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
    assert again.next_unit([1, 2, 3, 4], start=4) == 4 and again.next_unit([1, 2, 3], start=4) is None


def test_lesson_step_resets_only_when_unit_changes(tmp_path):
    store = ConversationStore(":memory:")
    sid = store.new_session()
    store.set_current_unit(sid, 5)
    store.set_lesson(sid, 2, 4, "intro")
    store.set_current_unit(sid, 5)  # cùng Unit: giữ nguyên vị trí
    s = store.session(sid)
    assert (s["lesson_part"], s["lesson_step"], s["part_intro"]) == (2, 4, "intro")
    store.set_current_unit(sid, 6)  # đổi Unit: về 0
    s = store.session(sid)
    assert (s["lesson_part"], s["lesson_step"], s["part_intro"]) == (0, 0, None)


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
    assert store.session(1)["lesson_step"] == 0 and store.session(1)["lesson_part"] == 0
    assert store.session(1)["current_unit"] == 5
    assert store.session(1)["style"] is None


# ----------------------------------------------------------------- điều hướng
ASK_LONG = (
    "hãy giúp tôi học tiếng anh theo từng unit. đầu tiên hãy cho tôi học thuộc các từ vựng bằng cách bạn sẽ giới thiệu "
    "các từ vựng trong unit đó, chia làm 2 hoặc 3 phần cho dễ học. xong rồi cho tôi làm bài tập"
)


def test_long_learning_request_starts_a_lesson(tmp_path):
    """Lỗi thật: câu này không khớp LEARN nên rơi vào chat tự do (tìm sách, trả lời một lượt, bịa bài 1.4)."""
    cfg, llm, store, tutor = make(tmp_path)
    ans = tutor.chat(tutor.new_session(), ASK_LONG)
    assert ans.mode == "lesson" and ans.phase == "teach" and ans.part == (1, 3) and tutor.retriever.calls == []


def test_intent_patterns():
    for text in ["bạn có thể dạy tôi học tiếng anh được không?", "bạn hãy dạy tôi theo từng unit được khong?",
                 "dạy tôi unit 5", "học theo từng unit", ASK_LONG, "teach me", "start a lesson"]:
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
def test_practice_phase_cycles_words_then_sentences():
    assert [practice_phase(k) for k in range(10)] == ["words"] * 3 + ["sentences"] * 2 + ["words"] * 3 + ["sentences"] * 2


def test_part_flow_teach_then_practice_loop_until_continue(tmp_path):
    """Luồng mong muốn: giới thiệu phần 1 -> luyện lặp (từ Việt->Anh, rồi câu Việt->Anh, rồi lại từ đầu) -> 'tiếp tục' -> phần 2."""
    cfg, llm, store, tutor = make(tmp_path)
    sid = tutor.new_session()

    a1 = tutor.chat(sid, "bạn có thể dạy tôi học tiếng anh được không?")
    assert (a1.mode, a1.unit_no, a1.part, a1.phase) == ("lesson", 1, (1, 3), "teach")
    assert "hello hello" in llm.human() and "goodbye" not in llm.human()  # chỉ phần 1
    assert "PHASE 1" in llm.human() and "just started this unit" in llm.human()
    assert "Every word MUST have an English example sentence" in llm.human()
    assert "FIRST practice question" in llm.human()
    assert store.session(sid)["part_intro"] == a1.text  # bài giới thiệu được nhớ để luyện đúng từ đã dạy

    phases = []
    for i in range(7):  # 7 câu trả lời liên tiếp vẫn ở phần 1, KHÔNG tự sang phần 2
        ans = tutor.chat(sid, f"answer {i}")
        assert ans.part == (1, 3) and "judge the learner's reply" in llm.human()
        assert a1.text in llm.human()  # từ luyện lấy từ bài giới thiệu
        phases.append(ans.phase)
    assert phases == ["words", "words", "sentences", "sentences", "words", "words", "words"]
    assert "hello hello" in llm.human()  # vẫn ngữ cảnh phần 1

    nxt = tutor.chat(sid, "tiếp tục")  # chỉ khi người học nói vậy mới sang phần 2
    assert nxt.part == (2, 3) and nxt.phase == "teach" and "goodbye" in llm.human() and "hello hello" not in llm.human()
    assert "judge" not in llm.human() and store.session(sid)["lesson_step"] == 1
    assert store.session(sid)["part_intro"] == nxt.text

    tutor.chat(sid, "ok")  # (đang ở phần 2: 'ok' chỉ là một câu trả lời)
    assert store.session(sid)["lesson_part"] == 1
    tutor.chat(sid, "tiếp tục")  # phần 3
    assert store.session(sid)["lesson_part"] == 2 and store.progress() == {1: "learning"}

    rev = tutor.chat(sid, "tiếp tục")  # hết phần 3 -> VÒNG ÔN cả Unit (chưa tóm tắt, chưa xong)
    assert rev.phase == "review" and rev.part == (3, 3) and store.progress() == {1: "learning"}
    human = llm.human()
    assert "UNIT REVIEW" in human and "judge" not in human
    assert "hello hello" not in human  # ôn bằng bài giới thiệu đã dạy, không đưa lại văn bản sách
    intros = json.loads(store.session(sid)["unit_intros"])
    assert len(intros) == 3 and all(i in human for i in intros)  # từ của CẢ 3 phần

    for i in range(4):  # ôn lặp lại, không tự kết thúc
        again = tutor.chat(sid, f"bản dịch {i}")
        assert again.phase == "review" and "judge the learner's reply" in llm.human()
    assert store.progress() == {1: "learning"}

    last = tutor.chat(sid, "ok")  # người học nói ok -> tóm tắt, đánh dấu xong
    assert last.part == (3, 3) and last.phase is None and "summarise the unit" in llm.human()
    assert store.progress() == {1: "done"}
    assert tutor.retriever.calls == []  # suốt bài học không tìm thêm Unit lạ
    assert all("exercise 1." not in c[-1][1] for c in llm.main_calls())


def test_after_summary_it_does_not_summarise_again(tmp_path):
    cfg, llm, store, tutor = make(tmp_path)
    sid = tutor.new_session()
    tutor.chat(sid, "dạy tôi học")
    for _ in range(3):  # 'tiếp tục' x3 -> qua 3 phần, vào vòng ôn
        tutor.chat(sid, "tiếp tục")
    tutor.chat(sid, "ok")  # kết thúc vòng ôn -> tóm tắt
    ans = tutor.chat(sid, "cảm ơn")  # Unit đã xong: trả lời tự do (có tìm sách), không tóm tắt lại
    assert ans.part is None and "summarise" not in llm.human()
    assert len(tutor.retriever.calls) == 1


def test_side_question_does_not_advance_lesson(tmp_path):
    cfg, llm, store, tutor = make(tmp_path)
    sid = tutor.new_session()
    tutor.chat(sid, "dạy tôi học")
    ans = tutor.chat(sid, "bruise nghĩa là gì?")
    assert ans.part is None and store.session(sid)["lesson_step"] == 1  # vị trí không đổi
    assert tutor.retriever.calls == [("bruise injury treatment", ("unit-001",))]  # câu hỏi được viết lại + ghim Unit
    assert "side question" in llm.main_calls()[-1][0][1]
    nxt = tutor.chat(sid, "dọn cái bảng")  # rồi trả lời tiếp câu luyện tập
    assert nxt.part == (1, 3) and nxt.phase == "words" and store.session(sid)["lesson_step"] == 2


def test_continue_moves_on_without_grading(tmp_path):
    cfg, llm, store, tutor = make(tmp_path)
    sid = tutor.new_session()
    tutor.chat(sid, "dạy tôi học")
    ans = tutor.chat(sid, "tiếp tục")
    assert ans.part == (2, 3) and "judge" not in llm.human()


def test_lesson_position_survives_restart(tmp_path):
    db = tmp_path / "c.db"
    cfg, llm, store, tutor = make(tmp_path, db)
    sid = tutor.new_session()
    tutor.chat(sid, "dạy tôi học")
    tutor.chat(sid, "answer")
    store.close()

    cfg2, llm2, store2, tutor2 = make(tmp_path, db)
    ans = tutor2.chat(sid, "answer 2")
    assert ans.part == (1, 3) and store2.session(sid)["lesson_step"] == 3  # nhớ đang luyện ở phần 1
    assert "reply 1" in llm2.human()  # bài giới thiệu được lấy lại từ DB
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


def test_lesson_starts_from_configured_unit(tmp_path):
    cfg = Settings(root=tmp_path, start_unit=2)
    llm, store = FakeLLM(), ConversationStore(":memory:")
    tutor = Tutor(cfg, FakeRetriever(cfg), llm, store)
    ans = tutor.chat(tutor.new_session(), "dạy tôi học")
    assert ans.unit_no == 2  # bỏ qua Unit 1
    assert Settings().start_unit == 6
