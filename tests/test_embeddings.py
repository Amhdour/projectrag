import app.embeddings as embeddings


def test_fake_embeddings_are_deterministic(monkeypatch) -> None:
    monkeypatch.setenv("EMBEDDINGS_DISABLED", "true")
    monkeypatch.setenv("EMBEDDINGS_DIMENSIONS", "6")

    client = embeddings.EmbeddingsClient()
    first = client.embed_texts(["hello", "world"])
    second = client.embed_texts(["hello", "world"])

    assert first == second
    assert len(first) == 2
    assert len(first[0]) == 6


def test_fake_embeddings_vary_by_input(monkeypatch) -> None:
    monkeypatch.setenv("EMBEDDINGS_DISABLED", "true")
    monkeypatch.setenv("EMBEDDINGS_DIMENSIONS", "6")

    client = embeddings.EmbeddingsClient()
    vectors = client.embed_texts(["alpha", "beta"])

    assert vectors[0] != vectors[1]
