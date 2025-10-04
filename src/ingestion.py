from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path


def load_documents(data_dir: Path | None = None) -> list[Document]:
    """
    Loads files from DATA_DIR directory.

    Only files with .txt extensions are supported.

    Splits files into separate articles.
    Each articles are expected to begin with "Статья X."

    returns:
        list[Documents]: list of documents with metadata
    """
    if data_dir is None:
        from src.config import DATA_DIR

        data_dir = DATA_DIR

    documents = []

    for file_path in data_dir.glob("*.txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        articles = text.split("Статья ")[1:]

        for article in articles:
            if not article.strip():
                continue

            lines = article.strip().split("\n", 1)

            article_num = lines[0].split(".")[0].strip()
            article_content = lines[1] if len(lines) > 1 else lines[0]

            document = Document(
                page_content=f"Статья {article_num}. {article_content}",
                metadata={"source": file_path.stem, "article": article_num},
            )

            documents.append(document)

    return documents


def get_chroma_vectorstore(
    chroma_persist_dir: Path | None = None, embedding_model: str | None = None
) -> Chroma:
    """
    Returns initialized ChromaDB.

    If index exist - returns existing index.
    If not - create index and then return it.

    Returns:
        Chroma: ChromaDB index.
    """
    if chroma_persist_dir is None:
        from src.config import CHROMA_PATH

        chroma_persist_dir = CHROMA_PATH

    if embedding_model is None:
        from src.config import EMBEDDING_MODEL

        embedding_model = EMBEDDING_MODEL

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    return Chroma(persist_directory=str(CHROMA_PATH), embedding_function=embeddings)


def ingest_documents():
    documents = load_documents()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)

    vector_store = get_chroma_vectorstore()
    vector_store.add_documents(chunks)


if __name__ == "__main__":
    ingest_documents()
