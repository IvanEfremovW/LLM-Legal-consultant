from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import torch
from transformers import BitsAndBytesConfig

from src.config import USE_4BIT, LLM_MODEL
from src.ingestion import get_chroma_vectorstore

vectorstore = get_chroma_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

quantization_config = (
    BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    if USE_4BIT
    else None
)

llm = HuggingFacePipeline.from_model_id(
    model_id=LLM_MODEL,
    task="text-generation",
    device_map="auto",
    model_kwargs=dict(quantization_config=quantization_config),
    pipeline_kwargs=dict(max_new_tokens=512, repetition_penalty=1.2, temperature=0.3),
)

chat_model = ChatHuggingFace(llm=llm)

prompt = ChatPromptTemplate.from_template("""
Ты — профессиональный юридический консультант. 
Ответь на вопрос, используя ТОЛЬКО приведённый ниже контекст.

Правила:
1. Если в контексте нет информации для ответа на вопрос — обозначь это.
2. В конце ответа цитируй источник в формате: [Название документа, Статья X].
3. Не выдумывай статьи или нормы.
4. Не отвечай на вопросы, не относящиеся к юридической тематике.

Вопрос: {question}

Контекст:
{context}

Ответ:
""")


def format_docs(docs):
    """Форматирует документы для промпта."""
    return "\n\n".join(
        [
            f"[{doc.metadata.get('source', 'Документ')}, {doc.metadata.get('article', 'Статья')}]: {doc.page_content}"
            for doc in docs
        ]
    )


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
