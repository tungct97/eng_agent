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
uv run eng-agent ask               # hội thoại mới (lưu vào conversations.db)
uv run eng-agent ask -c            # tiếp tục phiên gần nhất
uv run eng-agent ask -q "vết bầm tím nói tiếng Anh thế nào?"
uv run eng-agent sessions          # liệt kê các phiên đã lưu
uv run eng-agent history 3         # xem lại phiên #3
```

Đổi cách chunk mà không OCR lại: `eng-agent chunk && eng-agent index`.

Cấu hình qua biến môi trường `ENG_AGENT_ROOT`, `ENG_AGENT_EMBED_MODEL`, `ENG_AGENT_LLM_MODEL`
(hoặc `--root`). Ngưỡng và kích thước nằm trong `src/english_agent/config.py`.

## Hội thoại và học theo Unit

`eng-agent ask` là một cuộc hội thoại thật, không phải từng câu hỏi rời nhau:

- **Nhớ lịch sử:** các lượt trước của phiên được đưa lại cho LLM, nên "bài tập vừa rồi", câu trả lời của bạn cho
  câu hỏi luyện tập, hay "tiếp tục" đều được hiểu đúng và được chấm.
- **Hiểu câu nối tiếp khi tìm sách:** câu hỏi ngắn được viết lại thành câu tìm kiếm đầy đủ dựa trên cuộc trò chuyện.
- **Học theo Unit:** nói "dạy tôi học" (hoặc `/learn`, `/learn 5`) để bắt đầu từ Unit chưa học xong tiếp theo.
  Mỗi Unit được chia 2–5 phần và học theo kế hoạch cố định do code điều khiển: (1) giới thiệu từ vựng từng phần,
  (2) tutor đưa từ tiếng Việt, bạn viết tiếng Anh, (3) tutor đưa câu tiếng Việt, bạn dịch sang tiếng Anh. Tutor tự
  viết ra mọi từ/câu và tự nêu đáp án, không bắt bạn tìm bài tập trong sách. Gõ "tiếp tục" sau phần từ vựng;
  nói "bài tiếp" (hoặc `/next`) để sang Unit sau.
- **Lưu vào SQLite** (`conversations.db`): `sessions` (phiên + Unit đang học), `messages` (từng tin, kèm Unit và ảnh đã
  dùng), `progress` (Unit nào đang học / đã xong). Mở lại bằng `ask -c` hoặc `ask -s <số phiên>`; tiến độ dùng chung
  cho mọi phiên.
- **Cách học theo ý bạn:** nói ví dụ "dạy tôi bằng cách đưa 1 từ và bắt tôi dịch sang tiếng Anh" (hoặc `/style ...`,
  `/style off`) thì tutor nhớ cách đó cho cả phiên, mỗi lượt chỉ hỏi MỘT mục, tự viết đầy đủ câu hỏi (người học không
  thấy sách nên không bao giờ nói "bài 3.1") và luôn tự nêu đáp án đúng sau khi bạn trả lời.
- Ảnh trang chỉ được gửi khi một Unit mới xuất hiện trong phiên, không gửi lặp mỗi lượt.

Lệnh trong lúc chat: `/learn [N]`, `/next`, `/style [mô tả|off]`, `/progress`, `/new`, `/help`, `/quit`.

## Cấu trúc mã

```
src/english_agent/
  config.py      đường dẫn, tên mô hình, ngưỡng (MỘT chỗ duy nhất)
  ocr.py         giai đoạn 1: ảnh -> book.md
  chunking.py    giai đoạn 2: book.md -> chunks.jsonl + parents.json
  indexing.py    giai đoạn 3: chunks.jsonl -> Chroma (embedding dùng chung với retrieval)
  retrieval.py   giai đoạn 4a: câu hỏi -> Unit + ảnh
  memory.py      lưu hội thoại + tiến độ học (SQLite)
  tutor.py       giai đoạn 4b: hội thoại nhiều lượt với LLM local
  cli.py         lệnh `eng-agent`
tests/           test offline (không cần Tesseract/Ollama/Chroma)
```

## Test

```bash
uv run pytest
```

## Lưu ý

- `pages/` (ảnh scan gốc, ~240 MB), `chroma_db/` và `conversations.db` (hội thoại của bạn) không được commit. Để chạy lại OCR, tự đặt ảnh vào `pages/`
  (tên `page_0001.png`, `page_0002.png`, …).
- `requirements.txt` được giữ cho người dùng pip; nguồn chuẩn là `pyproject.toml`. Khi đổi dependencies, cập nhật bằng
  `uv export --no-hashes --no-dev --no-emit-project -o requirements.txt`.
- `build/` chứa nội dung giáo trình (có bản quyền của nhà xuất bản). Nếu để repo public, hãy cân nhắc.
