from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


from src.config import CHROMA_PATH, EMBEDDING_MODEL
from src.ingestion import get_chroma_vectorstore

vectorstore = get_chroma_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

print(retriever)