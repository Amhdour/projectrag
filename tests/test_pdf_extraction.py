from pathlib import Path

import app.main as main


class FakePage:
    def __init__(self, text: str) -> None:
        self._text = text

    def extract_text(self) -> str:
        return self._text


class FakeReader:
    def __init__(self, _path: str) -> None:
        self.pages = [FakePage("First page"), FakePage("Second page")]


def test_extract_pdf_pages_and_format(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(main, "PdfReader", FakeReader)
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    pages = main.extract_pdf_pages(pdf_path)
    assert pages == [
        {"page": 1, "text": "First page"},
        {"page": 2, "text": "Second page"},
    ]

    formatted = main.format_pdf_text(pages)
    assert formatted == "--- Page 1 ---\n\nFirst page\n\n--- Page 2 ---\n\nSecond page"
