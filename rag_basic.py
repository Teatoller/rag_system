# rag_basic.py
"""
Simple RAG System with LangChain - 100% via HF Inference API
No local PyTorch, no sentence-transformers, no GPU needed!

It's a great way to test RAG concepts on any machine, including Intel Macs.
"""

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.llms import LLM
from huggingface_hub import InferenceClient
from typing import Any, List, Optional
from dotenv import load_dotenv
import os


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
load_dotenv()
HF_TOKEN    = os.getenv("HF_TOKEN")
LLM_MODEL   = "HuggingFaceTB/SmolLM3-3B"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # used via API, not locally


# ─────────────────────────────────────────────
# CUSTOM LLM — HF Inference API
# ─────────────────────────────────────────────
class HFInferenceLLM(LLM):
    """LangChain LLM wrapper using HF Inference API for text generation."""

    model: str = LLM_MODEL
    token: str = HF_TOKEN
    max_tokens: int = 256

    @property
    def _llm_type(self) -> str:
        return "hf-inference-api"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        client = InferenceClient(provider="hf-inference", token=self.token)
        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content


# ─────────────────────────────────────────────
# CUSTOM EMBEDDINGS — HF Inference API
# ─────────────────────────────────────────────
class HFInferenceEmbeddings(Embeddings):
    """LangChain Embeddings wrapper using HF Inference API — no local model needed."""

    def __init__(self, token: str, model: str = EMBED_MODEL):
        self.token = token
        self.model = model
        self.client = InferenceClient(provider="hf-inference", token=token)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.client.feature_extraction(text, model=self.model) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self.client.feature_extraction(text, model=self.model)


# ─────────────────────────────────────────────
# 1. CREATE SAMPLE DOCUMENT
# ─────────────────────────────────────────────
with open("sample_doc.txt", "w") as f:
    f.write("""
Artificial Intelligence (AI) is transforming software development.

Key AI Technologies:
- Machine Learning: Systems that learn from data
- Natural Language Processing: Understanding human language
- Computer Vision: Analyzing images and videos
- Robotics: Intelligent physical systems

AI in Development:
- Code generation with tools like GitHub Copilot
- Automated testing and bug detection
- Intelligent code review
- Predictive analytics for project management

RAG (Retrieval-Augmented Generation):
RAG combines retrieval of relevant information with generation.
It helps LLMs provide accurate, context-specific answers by
first finding relevant documents, then generating responses
based on that retrieved context.

Benefits of RAG:
- Reduces hallucination
- Provides source attribution
- Enables domain-specific knowledge
- Keeps information up-to-date
""")

loader = TextLoader("sample_doc.txt")
documents = loader.load()
print(f"✅ Loaded {len(documents)} documents")


# ─────────────────────────────────────────────
# 2. SPLIT INTO CHUNKS
# ─────────────────────────────────────────────
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)

chunks = text_splitter.split_documents(documents)
print(f"✅ Created {len(chunks)} chunks")

for i, chunk in enumerate(chunks[:2]):
    print(f"\n--- Chunk {i+1} ---")
    print(chunk.page_content[:200] + "...")


# ─────────────────────────────────────────────
# 3. EMBEDDINGS via HF Inference API
# ─────────────────────────────────────────────
embeddings = HFInferenceEmbeddings(token=HF_TOKEN)
print("✅ Embeddings ready (via HF Inference API)")


# ─────────────────────────────────────────────
# 4. VECTOR STORE
# ─────────────────────────────────────────────
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
print("✅ Vector store created")


# ─────────────────────────────────────────────
# 5. LLM via HF Inference API
# ─────────────────────────────────────────────
llm = HFInferenceLLM()
print(f"✅ LLM ready ({LLM_MODEL})")


# ─────────────────────────────────────────────
# 6. BUILD RAG CHAIN
# ─────────────────────────────────────────────
template = """Answer the question based only on the following context.
If you cannot answer from the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("✅ RAG chain ready")


# ─────────────────────────────────────────────
# 7. ASK QUESTIONS
# ─────────────────────────────────────────────
questions = [
    "What is RAG?",
    "What are the benefits of RAG?",
    "How is AI used in software development?",
    "What is machine learning?"
]

print("\n" + "="*60)
print("TESTING RAG SYSTEM")
print("="*60)

for question in questions:
    print(f"\n❓ Question: {question}")
    print("-" * 60)

    answer = rag_chain.invoke(question)
    print(f"💬 Answer: {answer}")

    retrieved_docs = retriever.invoke(question)
    print(f"\n📄 Sources used ({len(retrieved_docs)} chunks):")
    for i, doc in enumerate(retrieved_docs, 1):
        preview = doc.page_content[:100].replace('\n', ' ')
        print(f"   {i}. {preview}...")

    print("="*60)
