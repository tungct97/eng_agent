"""Test hội thoại: bộ nhớ SQLite, lịch sử đưa lại cho LLM, chế độ học theo Unit. Không cần Ollama/Chroma."""
from langchain_core.documents import Document

from english_agent.config import Settings
from english_agent.memory import ConversationStore
from english_agent.retrieval import assemble
from english_agent.tutor import ALL_DONE, LEARN, NEXT, Tutor

PARENTS = {
    "unit-001": {"unit_no": 1, "title": "Greetings", "pages": [2, 3], "reference": "hello and hi", "exercises": "1.1 ..."},
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


def make(tmp_path, db=":memory:"):
    cfg = Settings(root=tmp_path)
    llm, store = FakeLLM(), ConversationStore(db)
    return cfg, llm, store, Tutor(cfg, FakeRetriever(cfg), llm, store)


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
    assert row["messages"] == 5 and row["title"] == "m0"  # tiêu đề = câu đầu của người học


def test_store_survives_reopen_and_tracks_progress(tmp_path):
    path = tmp_path / "c.db"
    store = ConversationStore(path)
    sid = store.new_session()
    store.add_message(sid, "user", "hello")
    store.set_current_unit(sid, 3)
    store.set_progress(1, "done")
    store.set_progress(2, "learning")
    store.set_progress(2, "done")  # cập nhật, không trùng
    store.close()

    again = ConversationStore(path)
    assert again.latest_session() == sid
    assert again.session(sid)["current_unit"] == 3
    assert again.history(sid)[0].content == "hello"
    assert again.progress() == {1: "done", 2: "done"}
    assert again.next_unit([1, 2, 3]) == 3
    assert again.next_unit([1, 2]) is None


# ----------------------------------------------------------------- điều hướng
def test_intent_patterns():
    for text in ["bạn có thể dạy tôi học tiếng anh được không?", "bạn hãy dạy tôi theo từng unit được khong?",
                 "dạy tôi unit 5", "học theo từng unit", "teach me", "start a lesson"]:
        assert LEARN.search(text), text
    for text in ["bài tiếp", "unit tiếp theo", "next unit", "Next lesson please"]:
        assert NEXT.search(text), text
    for text in ["I went to school, you let stay home to do your homework", "tiếp tục", "tôi bị bầm tím", "bruise là gì"]:
        assert not LEARN.search(text) and not NEXT.search(text), text


# ----------------------------------------------------------------- hội thoại
def test_lesson_starts_at_next_unit_without_searching(tmp_path):
    cfg, llm, store, tutor = make(tmp_path)
    sid = tutor.new_session()
    ans = tutor.chat(sid, "bạn có thể dạy tôi học tiếng anh được không?")
    assert (ans.mode, ans.unit_no) == ("lesson", 1)
    assert store.session(sid)["current_unit"] == 1 and store.progress() == {1: "learning"}
    assert tutor.retriever.calls == []  # Unit mới: không cần tìm kiếm
    system, human = llm.main_calls()[0][0][1], llm.main_calls()[0][-1][1]
    assert "ONE unit step by step" in system
    assert "Unit 1: Greetings" in human and "Begin the lesson now" in human


def test_followup_gets_history_pinned_unit_and_rewritten_query(tmp_path):
    cfg, llm, store, tutor = make(tmp_path)
    sid = tutor.new_session()
    first = tutor.chat(sid, "dạy tôi học")
    second = tutor.chat(sid, "I went to school, you let stay home")
    msgs = llm.main_calls()[1]
    assert msgs[1] == ("user", "dạy tôi học") and msgs[2] == ("assistant", first.text)  # LLM thấy lượt trước
    assert tutor.retriever.calls[-1] == ("bruise injury treatment", ("unit-001",))  # câu tìm kiếm được viết lại + ghim Unit
    assert second.unit_no == 1 and store.session(sid)["current_unit"] == 1  # vẫn đang học Unit 1


def test_next_marks_done_then_finishes(tmp_path):
    cfg, llm, store, tutor = make(tmp_path)
    sid = tutor.new_session()
    tutor.chat(sid, "dạy tôi học")
    ans = tutor.chat(sid, "bài tiếp")
    assert ans.unit_no == 2 and store.progress() == {1: "done", 2: "learning"}
    fin = tutor.chat(sid, "bài tiếp")
    assert fin.mode == "finished" and fin.text == ALL_DONE
    assert store.progress() == {1: "done", 2: "done"} and store.session(sid)["current_unit"] is None


def test_learn_specific_unit(tmp_path):
    cfg, llm, store, tutor = make(tmp_path)
    sid = tutor.new_session()
    assert tutor.chat(sid, "dạy tôi unit 2").unit_no == 2
    assert "unit-002" in tutor.chat(sid, "dạy tôi unit 2").units  # nói lại thì tiếp tục Unit 2, không mở lại từ đầu


def test_free_chat_has_no_lesson_rules_and_no_rewrite_without_history(tmp_path):
    cfg, llm, store, tutor = make(tmp_path)
    sid = tutor.new_session()
    ans = tutor.chat(sid, "bruise nghĩa là gì")
    assert ans.mode == "chat" and ans.unit_no is None
    assert tutor.retriever.calls == [("bruise nghĩa là gì", ())]  # chưa có lịch sử -> giữ nguyên câu hỏi
    assert "ONE unit step by step" not in llm.main_calls()[0][0][1]


def test_conversation_is_saved_and_resumes_after_restart(tmp_path):
    db = tmp_path / "c.db"
    cfg, llm, store, tutor = make(tmp_path, db)
    sid = tutor.new_session()
    tutor.chat(sid, "dạy tôi học")
    tutor.chat(sid, "I went to school")
    store.close()

    cfg2, llm2, store2, tutor2 = make(tmp_path, db)
    tutor2.chat(sid, "tiếp tục")
    msgs = llm2.main_calls()[0]
    assert [m[0] for m in msgs[:5]] == ["system", "user", "assistant", "user", "assistant"]  # lịch sử cũ được nạp lại
    assert len(store2.history(sid)) == 6
    assert store2.session(sid)["current_unit"] == 1  # nhớ cả Unit đang học


def test_images_are_sent_only_for_new_units(tmp_path):
    cfg, llm, store, tutor = make(tmp_path)
    (cfg.build_dir / "images").mkdir(parents=True)
    (cfg.build_dir / "images" / "page_0002.jpg").write_bytes(b"x")
    sid = tutor.new_session()
    assert [p.name for p in tutor.chat(sid, "dạy tôi học").images] == ["page_0002.jpg"]
    assert tutor.chat(sid, "I went to school").images == []  # Unit 1 đã gửi ảnh rồi


def test_history_is_trimmed(tmp_path):
    cfg, llm, store, tutor = make(tmp_path)
    sid = tutor.new_session()
    for i in range(10):
        tutor.chat(sid, f"câu {i}")
    msgs = llm.main_calls()[-1]
    assert len(msgs) == 1 + cfg.history_messages + 1  # system + lịch sử giới hạn + lượt hiện tại


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
