"""Bộ nhớ hội thoại lưu trong SQLite (conversations.db): phiên chat, tin nhắn, tiến độ học từng Unit.

Dùng SQLite thay vì Chroma vì đây là dữ liệu có thứ tự và có quan hệ (tin nhắn thuộc phiên, tiến độ theo Unit),
cần đọc lại đúng thứ tự và cập nhật, không cần tìm theo độ giống nghĩa. Chỉ dùng thư viện chuẩn.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions(
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    title        TEXT,
    current_unit INTEGER,           -- Unit đang học trong phiên này (NULL = chat tự do)
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages(
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role       TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content    TEXT NOT NULL,
    units      TEXT NOT NULL DEFAULT '[]',   -- JSON: các Unit đã dùng để trả lời
    images     TEXT NOT NULL DEFAULT '[]',   -- JSON: đường dẫn ảnh đã gửi kèm
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
CREATE TABLE IF NOT EXISTS progress(
    unit_no    INTEGER PRIMARY KEY,
    status     TEXT NOT NULL CHECK (status IN ('learning', 'done')),
    updated_at TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Message:
    role: str
    content: str
    units: list[str] = field(default_factory=list)
    images: list[str] = field(default_factory=list)
    created_at: str = ""


class ConversationStore:
    def __init__(self, path: Path | str):
        path = Path(path)
        if str(path) != ":memory:":
            path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        with self.db:
            self.db.executescript(SCHEMA)

    def close(self) -> None:
        self.db.close()

    # ------------------------------------------------------------ phiên
    def new_session(self, title: str | None = None) -> int:
        now = _now()
        with self.db:
            cur = self.db.execute(
                "INSERT INTO sessions(title, created_at, updated_at) VALUES (?,?,?)", (title, now, now)
            )
        return int(cur.lastrowid)

    def session(self, sid: int) -> dict:
        row = self.db.execute("SELECT * FROM sessions WHERE id=?", (sid,)).fetchone()
        if row is None:
            raise KeyError(f"Không có phiên #{sid}")
        return dict(row)

    def latest_session(self) -> int | None:
        row = self.db.execute("SELECT id FROM sessions ORDER BY updated_at DESC, id DESC LIMIT 1").fetchone()
        return int(row["id"]) if row else None

    def list_sessions(self) -> list[dict]:
        rows = self.db.execute(
            """SELECT s.id, s.title, s.current_unit, s.created_at, s.updated_at, COUNT(m.id) AS messages
               FROM sessions s LEFT JOIN messages m ON m.session_id = s.id
               GROUP BY s.id ORDER BY s.updated_at DESC, s.id DESC"""
        ).fetchall()
        return [dict(r) for r in rows]

    def set_current_unit(self, sid: int, unit: int | None) -> None:
        with self.db:
            self.db.execute("UPDATE sessions SET current_unit=?, updated_at=? WHERE id=?", (unit, _now(), sid))

    # ------------------------------------------------------------ tin nhắn
    def add_message(self, sid: int, role: str, content: str, units=(), images=()) -> None:
        now = _now()
        with self.db:
            self.db.execute(
                "INSERT INTO messages(session_id, role, content, units, images, created_at) VALUES (?,?,?,?,?,?)",
                (sid, role, content, json.dumps(list(units)), json.dumps([str(i) for i in images]), now),
            )
            # tiêu đề phiên = câu đầu tiên của người học
            self.db.execute(
                "UPDATE sessions SET updated_at=?, title=COALESCE(title, ?) WHERE id=?",
                (now, content[:60] if role == "user" else None, sid),
            )

    def history(self, sid: int, limit: int | None = None) -> list[Message]:
        """Các tin nhắn của phiên theo thứ tự thời gian; `limit` = chỉ lấy N tin gần nhất."""
        rows = self.db.execute(
            "SELECT role, content, units, images, created_at FROM messages WHERE session_id=? ORDER BY id", (sid,)
        ).fetchall()
        if limit is not None:
            rows = rows[-limit:] if limit > 0 else []
        return [
            Message(r["role"], r["content"], json.loads(r["units"]), json.loads(r["images"]), r["created_at"])
            for r in rows
        ]

    # ------------------------------------------------------------ tiến độ học
    def set_progress(self, unit: int, status: str) -> None:
        with self.db:
            self.db.execute(
                "INSERT INTO progress(unit_no, status, updated_at) VALUES (?,?,?) "
                "ON CONFLICT(unit_no) DO UPDATE SET status=excluded.status, updated_at=excluded.updated_at",
                (unit, status, _now()),
            )

    def progress(self) -> dict[int, str]:
        return {r["unit_no"]: r["status"] for r in self.db.execute("SELECT unit_no, status FROM progress")}

    def next_unit(self, all_units: list[int]) -> int | None:
        """Unit đầu tiên (theo thứ tự trong sách) chưa học xong; None nếu đã học hết."""
        done = {u for u, s in self.progress().items() if s == "done"}
        return next((u for u in sorted(all_units) if u not in done), None)
