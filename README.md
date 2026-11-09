# english-agent

Trợ lý học tiếng Anh chạy **hoàn toàn local**, dựa trên một cuốn giáo trình scan (mỗi trang là một ảnh):
hỏi bằng tiếng Việt hoặc tiếng Anh, nhận câu trả lời lấy từ sách kèm ảnh trang minh hoạ.

## Cài đặt nhanh (sau khi clone)

Repo đã có sẵn `build/` (kết quả OCR của sách), nên **không cần ảnh scan và không cần OCR** để chạy hỏi đáp.

```bash
git clone https://github.com/tungct97/eng_agent.git && cd eng_agent

# 1. Công cụ hệ thống (Ubuntu / WSL)
sudo apt install zstd tesseract-ocr          # tesseract chỉ cần nếu bạn chạy lại bước OCR
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:7b-instruct              # máy yếu: qwen2.5:3b-instruct (đặt ENG_AGENT_LLM_MODEL)

# 2. Thư viện Python — chọn MỘT trong hai
uv sync                                      # khuyên dùng (dùng pyproject.toml + uv.lock)
# hoặc:  python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

# 3. Tạo chunk + vector DB từ build/book.md, rồi hỏi
uv run eng-agent build --skip-ocr               # chunk + index trong một lệnh (lần đầu tải mô hình embedding ~2 GB)
uv run eng-agent ask
```

(Nếu dùng pip, bỏ `uv run` và gõ thẳng `eng-agent ...`.)

## Luồng xử lý

```
pages/*.png                                ảnh scan gốc (không commit; chỉ cần khi chạy OCR)
      │  eng-agent ocr      (Tesseract, không dùng LLM)
      ▼
build/book.md + build/images/              chữ + tiêu đề theo bậc (# Page, ## Unit, ### A/B/C) + ảnh đã nén   [commit]
      │  eng-agent chunk    (LangChain splitter, chia theo cấu trúc Unit)
      ▼
build/chunks.jsonl + build/parents.json    chunk con (để tìm) + Unit đầy đủ (để đưa cho LLM)   [sinh lại được]
      │  eng-agent index    (bge-m3 embedding)
      ▼
chroma_db/                                 vector DB   [sinh lại được]
      │  eng-agent ask      (tìm chunk -> mở rộng thành Unit -> Ollama)
      ▼
câu trả lời + đường dẫn ảnh trang
```

Mỗi bước đọc đúng file mà bước trước ghi ra, nên chạy riêng từng bước hoặc `eng-agent build` cho ba bước đầu đều được.

## Lệnh

```bash
uv run eng-agent ocr --limit 5     # (cần pages/) thử OCR 5 trang, mở build/book.md xem
uv run eng-agent build             # ocr + chunk + index cả cuốn
uv run eng-agent build --skip-ocr  # chỉ chunk + index (dùng build/book.md có sẵn)
uv run eng-agent chunk             # book.md -> chunks.jsonl + parents.json
uv run eng-agent index             # chunks -> Chroma
uv run eng-agent search "bruise"   # xem tìm kiếm trả về gì (không cần LLM)
uv run eng-agent ask               # hỏi đáp (hội thoại)
uv run eng-agent ask -q "vết bầm tím nói tiếng Anh thế nào?"
```

Đổi cách chunk mà không OCR lại: `eng-agent chunk && eng-agent index`.

Cấu hình qua biến môi trường `ENG_AGENT_ROOT`, `ENG_AGENT_EMBED_MODEL`, `ENG_AGENT_LLM_MODEL`
(hoặc `--root`). Ngưỡng và kích thước nằm trong `src/english_agent/config.py`.

## Cấu trúc mã

```
src/english_agent/
  config.py      đường dẫn, tên mô hình, ngưỡng (MỘT chỗ duy nhất)
  ocr.py         giai đoạn 1: ảnh -> book.md
  chunking.py    giai đoạn 2: book.md -> chunks.jsonl + parents.json
  indexing.py    giai đoạn 3: chunks.jsonl -> Chroma (embedding dùng chung với retrieval)
  retrieval.py   giai đoạn 4a: câu hỏi -> Unit + ảnh
  tutor.py       giai đoạn 4b: LLM local trả lời
  cli.py         lệnh `eng-agent`
tests/           test offline (không cần Tesseract/Ollama/Chroma)
```

## Test

```bash
uv run pytest
```

## Lưu ý

- `pages/` (ảnh scan gốc, ~240 MB) và `chroma_db/` không được commit. Để chạy lại OCR, tự đặt ảnh vào `pages/`
  (tên `page_0001.png`, `page_0002.png`, …).
- `requirements.txt` được giữ cho người dùng pip; nguồn chuẩn là `pyproject.toml`. Khi đổi dependencies, cập nhật bằng
  `uv export --no-hashes --no-dev --no-emit-project -o requirements.txt`.
- `build/` chứa nội dung giáo trình (có bản quyền của nhà xuất bản). Nếu để repo public, hãy cân nhắc.
