import re
from langchain_core.documents import Document
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

from src.config import DATA_DIR
from src.config import CHROMA_PATH, EMBEDDING_MODEL


def load_documents() -> list[Document]:
    """
    Loads files from DATA_DIR directory.
    
    Only files with .txt extensions are supported.
    
    Splits files into separate articles.
    Each articles are expected to begin with "Статья X."
    
    returns:
        list[Documents]: list of documents with metadata
    """
    documents = []
    
    for file_path in DATA_DIR.glob("*.txt"):
        with open(file_path, 'r', encoding='utf-8') as f:
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
                metadata={
                    "source": file_path.stem,
                    "article": article_num
                }
            )
            
            documents.append(document)
            
    return documents

def get_chroma_vectorstore() -> Chroma:
    """
    Returns initialized ChromaDB.
    
    If index exist - returns existing index.
    If not - create index and then return it.
    
    Returns:
        Chroma: ChromaDB index.
    """
    
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=EMBEDDING_MODEL
    )
    
    return Chroma(
        persist_directory=str(CHROMA_PATH),
        embedding_function=embeddings
    ) 