"""Ảnh trang sách -> build/book.md (chữ OCR + ảnh) và build/book.pdf (tuỳ chọn). Không dùng LLM.

Cài (WSL/Ubuntu):
    sudo apt install tesseract-ocr
    uv add pytesseract pillow pypdf numpy opencv-python-headless

Chạy:
    uv run python ocr_pages.py --images pages --out build --limit 5            # ảnh cả trang + chữ
    uv run python ocr_pages.py --images pages --out build --figures --limit 5  # + tự cắt tranh minh hoạ
    thêm --pdf nếu cũng muốn xuất book.pdf (nặng, chậm hơn)

Kết quả:
    build/book.md
    build/images/page_0129.jpg            (ảnh cả trang, đã nén)
    build/images/page_0129_fig1.jpg ...   (chỉ khi dùng --figures)
"""
from __future__ import annotations

import argparse
import io
import re
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageOps

EXT = {".png", ".jpg", ".jpeg", ".webp"}


def page_no(p: Path) -> int:
    m = re.search(r"(\d+)", p.stem)
    return int(m.group(1)) if m else 0


def prep(img: Image.Image) -> Image.Image:
    img = ImageOps.exif_transpose(img).convert("L")
    if img.width < 1800:  # scan nhỏ -> phóng to để OCR chính xác hơn
        k = 1800 / img.width
        img = img.resize((round(img.width * k), round(img.height * k)), Image.LANCZOS)
    return ImageOps.autocontrast(img)


def tidy(text: str) -> str:
    text = re.sub(r"-\n(?=[a-z])", "", text)  # nối từ bị ngắt dòng
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def ocr_lines(gray: Image.Image, lang: str, psm: int) -> list[dict]:
    """OCR thành danh sách dòng, mỗi dòng có chữ, chiều cao chữ, vị trí và id đoạn."""
    d = pytesseract.image_to_data(gray, lang=lang, config=f"--psm {psm}", output_type=pytesseract.Output.DICT)
    groups: dict[tuple, dict] = {}
    for i, word in enumerate(d["text"]):
        if not word.strip() or float(d["conf"][i]) < 0:
            continue
        key = (d["block_num"][i], d["par_num"][i], d["line_num"][i])
        g = groups.setdefault(key, {"words": [], "h": [], "top": d["top"][i], "par": key[:2]})
        g["words"].append(word)
        g["h"].append(d["height"][i])
    return [
        {"text": " ".join(g["words"]), "h": float(np.median(g["h"])), "top": g["top"], "par": g["par"]}
        for g in groups.values()
    ]


LETTER_HEAD = re.compile(r"^[A-H]\s+[A-Z][^.?!:;,]*$")  # "A Common injuries"
UNIT_HEAD = re.compile(r"^\d{1,3}\s+[A-Z]")  # "60 Health: injuries"


def level_of(line: dict, med_h: float, page_h: int) -> int:
    """0 = chữ thường; 2 = tên Unit; 3 = mục A/B/C hoặc dòng chữ lớn; 4 = nhãn nhỏ (NOTE, Exercise...)."""
    t, words = line["text"].strip(), len(line["text"].split())
    big = line["h"] >= 1.35 * med_h
    if words <= 10 and UNIT_HEAD.match(t) and (big or line["top"] < 0.2 * page_h):
        return 2
    if words <= 8 and big and not t.endswith((".", ",")):
        return 2 if line["top"] < 0.2 * page_h else 3
    if words <= 7 and LETTER_HEAD.match(t):
        return 3
    if words <= 3 and (t.isupper() and len(t) > 2 or re.match(r"^(Exercises?|Study tip|Tip|Remember)\b", t, re.I)):
        return 4
    return 0


TOC_ENTRY = re.compile(r"(?:^|\s)(?=\d{1,3}\s+[A-Z])")  # chỗ bắt đầu một mục "37 Prepositions..."


def is_toc(lines: list[dict]) -> bool:
    """Trang mục lục: có từ 4 dòng trở lên bắt đầu bằng 'số + chữ hoa'."""
    return sum(bool(UNIT_HEAD.match(l["text"].strip())) for l in lines) >= 4


def toc_markdown(lines: list[dict], page_h: int) -> str:
    """Mục lục: mỗi mục là một gạch đầu dòng; chỉ dòng không có số đầu mà chữ lớn mới là tiêu đề."""
    med_h = float(np.median([l["h"] for l in lines]))
    out = []
    for l in sorted(lines, key=lambda x: x["top"]):
        t = l["text"].strip().replace("|", " ")
        entries = [e.strip() for e in TOC_ENTRY.split(t) if e.strip()]
        if len(entries) == 1 and not UNIT_HEAD.match(t):
            big = l["h"] >= 1.35 * med_h and len(t.split()) <= 8
            out.append(("## " if big else "") + t)
        else:
            out += [f"- {e}" for e in entries]  # tách các mục bị ghép chung một dòng (mục lục 2 cột)
    return "\n".join(out)


def to_markdown(lines: list[dict], page_h: int) -> str:
    if not lines:
        return ""
    if is_toc(lines):
        return toc_markdown(lines, page_h)
    med_h = float(np.median([l["h"] for l in lines]))
    out, para, last_par = [], [], None

    def flush():
        nonlocal para
        if para:
            out.append(tidy(" ".join(para)))
            para = []

    for l in lines:
        lv = level_of(l, med_h, page_h)
        if lv:
            flush()
            out.append("#" * lv + " " + l["text"].strip())
            last_par = None
            continue
        if last_par is not None and l["par"] != last_par:
            flush()
        para.append(l["text"])
        last_par = l["par"]
    flush()
    return "\n\n".join(out)


def save_jpg(img: Image.Image, path: Path, max_w: int = 1200, quality: int = 80) -> None:
    img = img.convert("RGB")
    if img.width > max_w:
        img = img.resize((max_w, round(img.height * max_w / img.width)), Image.LANCZOS)
    img.save(path, "JPEG", quality=quality, optimize=True)


def find_figures(gray: Image.Image, lang: str, min_area: float = 0.02, min_side: float = 0.06):
    """Tìm vùng tranh = phần không-trắng còn lại sau khi xoá chữ. Trả về list (x0,y0,x1,y1) theo pixel, từ trên xuống.

    Đây là heuristic: hộp màu có chữ đôi khi bị nhận là tranh, tranh có nhãn chữ có thể bị cắt dính nhãn.
    """
    arr = np.array(gray)
    h, w = arr.shape
    data = pytesseract.image_to_data(gray, lang=lang, config="--psm 3", output_type=pytesseract.Output.DICT)
    mask = (arr < 235).astype(np.uint8) * 255
    for i, word in enumerate(data["text"]):
        if word.strip() and float(data["conf"][i]) > 40:  # xoá chữ đã nhận ra
            x, y, bw, bh = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            mask[max(0, y - 2) : y + bh + 2, max(0, x - 2) : x + bw + 2] = 0
    k = max(15, w // 50)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))  # bỏ nét mảnh/nhiễu
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw * bh >= min_area * w * h and bw >= min_side * w and bh >= min_side * h and bw * bh < 0.9 * w * h:
            boxes.append((x, y, x + bw, y + bh))
    return sorted(boxes, key=lambda b: (b[1], b[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="pages")
    ap.add_argument("--out", default="build")
    ap.add_argument("--lang", default="eng")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--psm", type=int, default=4, help="chế độ bố cục Tesseract: 3=tự chia cột, 4=một cột, 6=một khối")
    ap.add_argument("--figures", action="store_true", help="tự cắt các vùng tranh minh hoạ")
    ap.add_argument("--pdf", action="store_true", help="xuất thêm book.pdf")
    a = ap.parse_args()

    files = sorted((p for p in Path(a.images).iterdir() if p.suffix.lower() in EXT), key=page_no)
    if a.limit:
        files = files[: a.limit]
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)

    writer = None
    if a.pdf:
        from pypdf import PdfWriter

        writer = PdfWriter()

    md = []
    for i, p in enumerate(files, 1):
        original = Image.open(p)
        gray = prep(original)
        text = to_markdown(ocr_lines(gray, a.lang, a.psm), gray.height)
        n = page_no(p)

        parts = [f"# Page {n}\n"]
        page_jpg = f"page_{n:04d}.jpg"
        save_jpg(ImageOps.exif_transpose(original), out / "images" / page_jpg)
        parts.append(f"![Page {n}](images/{page_jpg})\n")

        if a.figures:
            color = ImageOps.exif_transpose(original).convert("RGB").resize(gray.size, Image.LANCZOS)
            for j, (x0, y0, x1, y1) in enumerate(find_figures(gray, a.lang), 1):
                name = f"page_{n:04d}_fig{j}.jpg"
                save_jpg(color.crop((x0, y0, x1, y1)), out / "images" / name, max_w=900)
                parts.append(f"![Illustration {j} (page {n})](images/{name})\n")

        parts.append(text + "\n")
        md.append("\n".join(parts))

        if writer is not None:
            writer.append(io.BytesIO(pytesseract.image_to_pdf_or_hocr(gray, lang=a.lang, extension="pdf")))
        print(f"[{i}/{len(files)}] {p.name}")

    (out / "book.md").write_text("\n---\n\n".join(md), encoding="utf-8")
    if writer is not None:
        with open(out / "book.pdf", "wb") as f:
            writer.write(f)
    print(f"xong -> {out}/book.md (ảnh trong {out}/images)")


if __name__ == "__main__":
    main()