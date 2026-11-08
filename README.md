# english-agent

Trợ lý học tiếng Anh chạy **hoàn toàn local**, dựa trên một cuốn giáo trình scan (mỗi trang là một ảnh):
hỏi bằng tiếng Việt hoặc tiếng Anh, nhận câu trả lời lấy từ sách kèm ảnh trang minh hoạ.

## Luồng xử lý

```
data/pages/*.png
      │  eng-agent ocr      (Tesseract, không dùng LLM)
      ▼
data/build/book.md + images/          chữ + tiêu đề theo bậc (# Page, ## Unit, ### A/B/C) + ảnh đã nén
      │  eng-agent chunk    (LangChain splitter, chia theo cấu trúc Unit)
      ▼
data/build/chunks.jsonl + parents.json    chunk con (để tìm) + Unit đầy đủ (để đưa cho LLM)
      │  eng-agent index    (bge-m3 embedding)
      ▼
data/chroma_db
      │  eng-agent ask      (tìm chunk -> mở rộng thành Unit -> Ollama)
      ▼
câu trả lời + đường dẫn ảnh trang
```

Mỗi bước đọc đúng file mà bước trước ghi ra, nên chạy riêng từng bước hoặc `eng-agent build` cho cả ba đều được.

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

## Cài đặt (Ubuntu / WSL)

```bash
sudo apt install tesseract-ocr zstd
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:7b-instruct        # máy yếu: qwen2.5:3b-instruct (đặt ENG_AGENT_LLM_MODEL)
uv sync
```

## Dùng

Đặt ảnh scan vào `data/pages/` (tên `page_0001.png`, `page_0002.png`, …), rồi chạy từ gốc repo:

```bash
uv run eng-agent ocr --limit 5     # thử 5 trang, mở data/build/book.md xem
uv run eng-agent build             # OCR + chunk + index cả cuốn
uv run eng-agent search "bruise"   # kiểm tra tìm kiếm (không cần LLM)
uv run eng-agent ask               # hỏi đáp
uv run eng-agent ask -q "vết bầm tím nói tiếng Anh thế nào?"
```

Đổi chunking mà không OCR lại: `eng-agent chunk && eng-agent index`.

Cấu hình qua biến môi trường: `ENG_AGENT_DATA`, `ENG_AGENT_EMBED_MODEL`, `ENG_AGENT_LLM_MODEL`
(hoặc `--data-dir`). Ngưỡng và kích thước nằm trong `config.py`.

## Test

```bash
uv run pytest
```

## Lưu ý về dữ liệu

`data/` được `.gitignore`: ảnh scan, kết quả OCR và vector DB **không** được đưa lên git
(nặng, và nội dung sách thuộc bản quyền nhà xuất bản). Chỉ commit mã nguồn.
