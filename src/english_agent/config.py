"""Cấu hình duy nhất cho cả pipeline: đường dẫn, tên mô hình, ngưỡng. Các giai đoạn không tự đặt hằng số riêng.

Thư mục dữ liệu mặc định là ./data (tính từ nơi bạn chạy lệnh, thường là gốc repo); đổi bằng biến môi trường:
    ENG_AGENT_DATA, ENG_AGENT_EMBED_MODEL, ENG_AGENT_LLM_MODEL
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("ENG_AGENT_DATA", "data")))

    # mô hình (embedding dùng ở cả index lẫn truy vấn nên PHẢI lấy từ đây, không viết lại ở chỗ khác)
    embed_model: str = field(default_factory=lambda: os.getenv("ENG_AGENT_EMBED_MODEL", "BAAI/bge-m3"))
    llm_model: str = field(default_factory=lambda: os.getenv("ENG_AGENT_LLM_MODEL", "qwen2.5:7b-instruct"))
    llm_num_ctx: int = 8192  # mặc định của Ollama chỉ ~2k token, ngữ cảnh dài sẽ bị cắt âm thầm

    # truy vấn
    top_k: int = 6  # số chunk con lấy về
    max_units: int = 2  # số Unit đầy đủ đưa cho LLM
    max_distance: float = 0.55  # cosine distance tối đa; xa hơn coi như ngoài sách
    collection: str = "evu"

    # ---- đường dẫn dẫn xuất ----
    @property
    def pages_dir(self) -> Path:  # ảnh scan gốc (bạn tự đặt vào)
        return self.data_dir / "pages"

    @property
    def build_dir(self) -> Path:  # kết quả OCR + chunking
        return self.data_dir / "build"

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
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma_db"
