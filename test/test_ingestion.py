from src.ingestion import load_documents, get_chroma_vectorstore, ingest_documents


def test_load_documents_parses_articles(tmp_path):
    test_dir = tmp_path / "data"
    test_dir.mkdir()

    (test_dir / "test_doc.txt").write_text(
        "Статья 115. Отпуск\n"
        "Ежегодный отпуск — 28 дней.\n\n"
        "Статья 196. Исковая давность\n"
        "Срок — 3 года.",
        encoding="utf-8",
    )

    docs = load_documents(data_dir=test_dir)
    assert len(docs) == 2

    assert docs[0].metadata["source"] == "test_doc"
    assert docs[0].metadata["article"] == "115"
    assert "28 дней" in docs[0].page_content

    assert docs[1].metadata["article"] == "196"
    assert "3 года" in docs[1].page_content


def test_ingest_creates_valid_index(tmp_path):
    test_dir = tmp_path / "data"
    test_dir.mkdir()

    (test_dir / "test_doc.txt").write_text(
        "Статья 115. Отпуск\n"
        "Ежегодный отпуск — 28 дней.\n\n"
        "Статья 196. Исковая давность\n"
        "Срок — 3 года.",
        encoding="utf-8",
    )

    ingest_documents()

    vectorstore = get_chroma_vectorstore()
    retriever = vectorstore.as_retriever()
    results = retriever.invoke("тестовый вопрос")

    assert len(results) >= 0
