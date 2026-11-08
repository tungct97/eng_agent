import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

from eng_agent.agent import Tutor
from eng_agent.extract import normalize_page, parse_json
from eng_agent.render import render_visual
from eng_agent.schema import VisualRef
from eng_agent.sources import LocalDirSource
from eng_agent.store import HashEmbedder, Store, chunk_page

RAW = {
    "unit": "60 Health: injuries",
    "printed_page": "122",
    "sections": [
        {"heading": "A Common injuries", "type": "vocab",
         "text": "She cut her finger. He twisted his ankle. She has a bruise.", "figure_ids": ["f1", "bad"]},
        {"heading": "B Hospital treatment", "type": "note",
         "text": "He was rushed to hospital and needed ten stitches.", "figure_ids": []},
    ],
    "figures": [
        {"id": "f1", "bbox": [100, 200, 900, 500], "kind": "illustration", "caption": "Five people with injuries"},
        {"id": "f2", "bbox": [5, 5, 6, 6], "kind": "illustration", "caption": "too small"},
    ],
}


def make_pages(tmp_path):
    img = Image.new("RGB", (1000, 1400), "white")
    for x in range(100, 900):  # coloured band at the figure position
        for y in range(280, 700):
            img.putpixel((x, y), (200, 30, 30))
    img.save(tmp_path / "page_0129.png")
    return LocalDirSource(tmp_path)


def test_parse_json_tolerates_fences():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_normalize_drops_bad_figures_and_links():
    page = normalize_page(RAW)
    assert [f["id"] for f in page["figures"]] == ["f1"]
    assert page["figures"][0]["bbox"] == (0.1, 0.2, 0.9, 0.5)
    assert page["sections"][0]["figure_ids"] == ["f1"]


def test_chunk_carries_visual_metadata():
    chunks = chunk_page(normalize_page(RAW), "page_0129.png", 129)
    assert len(chunks) == 2
    assert chunks[0].visuals[0].bbox == (0.1, 0.2, 0.9, 0.5)
    assert chunks[1].visuals == []
    assert chunks[0].printed_page == "122"


def test_store_search_and_reindex(tmp_path):
    store = Store(str(tmp_path / "t.db"), HashEmbedder())
    store.save_page("page_0129.png", 129, normalize_page(RAW))
    assert store.build_index() == 2
    top = store.search("bruise ankle finger", k=1)[0].chunk
    assert top.section == "A Common injuries" and top.visuals
    assert store.build_index(max_chars=30) > 2  # re-chunk without extraction


def test_render_crops_only_the_box(tmp_path):
    src = make_pages(tmp_path)
    ref = VisualRef("page_0129.png", (0.1, 0.2, 0.9, 0.5))
    png = render_visual(src, ref)
    im = Image.open(io.BytesIO(png))
    assert im.size[0] < 1000 and im.size[1] < 700  # smaller than the page
    assert im.getpixel((im.width // 2, im.height // 2)) == (200, 30, 30)


def test_path_traversal_blocked(tmp_path):
    src = make_pages(tmp_path)
    try:
        src.get_bytes("../x.png")
    except ValueError:
        return
    raise AssertionError("traversal not blocked")


class FakeClient:
    """Scripted model: search -> show_visual -> final answer."""

    def __init__(self):
        self.calls = 0
        self.messages = SimpleNamespace(create=self.create)

    def create(self, **kw):
        self.calls += 1
        if self.calls == 1:
            blk = SimpleNamespace(type="tool_use", id="t1", name="search_lessons", input={"query": "injuries"})
            return SimpleNamespace(stop_reason="tool_use", content=[blk])
        if self.calls == 2:
            last = kw["messages"][-1]["content"][0]["content"]
            vid = json.loads(last)[0]["visual_ids"][0]
            blk = SimpleNamespace(type="tool_use", id="t2", name="show_visual", input={"visual_id": vid})
            return SimpleNamespace(stop_reason="tool_use", content=[blk])
        return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text="Here you go.")])


def test_agent_sends_screenshot_on_demand(tmp_path):
    src = make_pages(tmp_path)
    store = Store(str(tmp_path / "t.db"), HashEmbedder())
    store.save_page("page_0129.png", 129, normalize_page(RAW))
    store.build_index()
    reply = Tutor(FakeClient(), store, src).ask("injuries?")
    assert reply.text == "Here you go."
    assert len(reply.images) == 1 and reply.images[0][1][:4] == b"\x89PNG"
