"""Cấu hình duy nhất cho cả pipeline: đường dẫn, tên mô hình, ngưỡng. Các giai đoạn không tự đặt hằng số riêng.

Thư mục gốc dữ liệu mặc định là "." (nơi bạn chạy lệnh, thường là gốc repo) với bố cục:
    pages/      ảnh scan gốc (không commit)
    build/      book.md + images/ (commit) và chunks.jsonl + parents.json (sinh ra, không commit)
    chroma_db/  vector DB (không commit)
    conversations.db  hội thoại + tiến độ học (không commit)
Đổi bằng biến môi trường:
    ENG_AGENT_ROOT, ENG_AGENT_EMBED_MODEL, ENG_AGENT_LLM_MODEL, ENG_AGENT_START_UNIT
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    root: Path = field(default_factory=lambda: Path(os.getenv("ENG_AGENT_ROOT", ".")))

    # mô hình (embedding dùng ở cả index lẫn truy vấn nên PHẢI lấy từ đây, không viết lại ở chỗ khác)
    embed_model: str = field(default_factory=lambda: os.getenv("ENG_AGENT_EMBED_MODEL", "BAAI/bge-m3"))
    llm_model: str = field(default_factory=lambda: os.getenv("ENG_AGENT_LLM_MODEL", "qwen2.5:7b-instruct"))
    llm_num_ctx: int = 8192  # mặc định của Ollama chỉ ~2k token, ngữ cảnh dài sẽ bị cắt âm thầm

    # truy vấn
    top_k: int = 6  # số chunk con lấy về
    max_units: int = 2  # số Unit đầy đủ đưa cho LLM
    max_distance: float = 0.55  # cosine distance tối đa; xa hơn coi như ngoài sách
    collection: str = "evu"

    # hội thoại / học theo Unit
    start_unit: int = field(default_factory=lambda: int(os.getenv("ENG_AGENT_START_UNIT", "6")))  # Unit đầu tiên khi bắt đầu học
    history_messages: int = 8  # số tin nhắn gần nhất đưa lại cho LLM (4 lượt hỏi-đáp)
    history_chars: int = 1200  # cắt mỗi tin trong lịch sử cho gọn
    rewrite_queries: bool = True  # viết lại câu hỏi nối tiếp thành câu tìm kiếm đầy đủ (tốn thêm 1 lần gọi LLM)

    # ---- đường dẫn dẫn xuất ----
    @property
    def pages_dir(self) -> Path:  # ảnh scan gốc (bạn tự đặt vào)
        return self.root / "pages"

    @property
    def build_dir(self) -> Path:  # kết quả OCR + chunking
        return self.root / "build"

    @property
    def book_md(self) -> Path:
        return self.build_dir / "book.md"

    @property
    def chunks_path(self) -> Path:
        return self.build_dir / "chunks.jsonl"

    @property
    def parents_path(self) -> Path:
        return self.build_dir / "parents.json"

    @property
    def db_path(self) -> Path:  # hội thoại + tiến độ học (SQLite)
        return self.root / "conversations.db"

    @property
    def chroma_dir(self) -> Path:
        return self.root / "chroma_db"
