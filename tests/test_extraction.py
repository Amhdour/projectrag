from pathlib import Path

import app.main as main


def test_normalize_text_rules() -> None:
    raw = "Line one  \r\n\r\n\r\nLine two  \r\n\r\n\r\n\r\nLine three"
    normalized = main.normalize_text(raw)
    assert normalized == "Line one\n\nLine two\n\nLine three"


def test_extract_text_from_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(main, "PROCESSED_ROOT", tmp_path)
    source = tmp_path / "source.txt"
    source.write_text("Hello  \r\n\r\n\r\nWorld  ", encoding="utf-8")

    extracted = main.extract_text_from_file(source)
    assert extracted == "Hello\n\nWorld"

    saved_path = main.save_processed_text("doc123", extracted)
    assert saved_path.read_text(encoding="utf-8") == "Hello\n\nWorld"
