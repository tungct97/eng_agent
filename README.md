# English tutor agent (from textbook page images)

Ý chính: **không lưu ảnh cắt**. Mỗi chunk chỉ lưu `page_id` + `bbox` (toạ độ 0..1 so với cả trang).
Khi agent trả lời và cần minh hoạ, `render.py` mở ảnh trang gốc, cắt đúng vùng đó rồi gửi đi.

## Luồng
1. `ingest`: mỗi ảnh trang -> vision LLM -> JSON (unit, printed_page, sections, figures + bbox). Lưu JSON thô vào SQLite.
2. `build_index`: chunk theo section (có header Unit > Heading), thêm caption minh hoạ vào text để tìm được,
   embed bằng multilingual-e5 (hỏi tiếng Việt vẫn tìm ra chunk tiếng Anh).
3. `chat`: Claude có 2 tool: `search_lessons`, `show_visual`. Khi `show_visual` được gọi, ảnh được cắt tại chỗ.

## Chạy
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=...
    python cli.py ingest --images ./pages --db book.db --limit 5     # thử 5 trang trước
    python cli.py ingest --images ./pages --db book.db               # chạy tiếp (bỏ qua trang đã xong)
    python cli.py chat   --images ./pages --db book.db --out ./sent

Đọc thẳng từ Google Drive (không tải ảnh về): `--drive-folder FOLDER_ID --sa-json service_account.json`
(chia sẻ thư mục cho email của service account).

Đổi cách chunk: `python cli.py reindex --db book.db` (không tốn thêm lệnh vision).

Test: `python -m pytest tests` (chạy offline, không cần API key).
