# rag_multi_docs.py
"""
Multi-Document RAG System with LangChain - 100% via HF Inference API
Extends the simple RAG pattern to support multiple documents.
No local PyTorch, no sentence-transformers, no GPU needed!
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
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ─────────────────────────────────────────────
# CUSTOM LLM — HF Inference API  (same as working script)
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
# CUSTOM EMBEDDINGS — HF Inference API  (same as working script)
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
# HELPER — load multiple documents  (was missing from original snippet)
# ─────────────────────────────────────────────
def load_multiple_documents(file_paths: List[str]):
    """Load and return all documents from a list of file paths."""
    all_docs = []
    for path in file_paths:
        loader = TextLoader(path)
        docs = loader.load()
        all_docs.extend(docs)
        print(f"  📄 Loaded: {path} ({len(docs)} doc)")
    return all_docs


# ─────────────────────────────────────────────
# SHARED UTILITIES  (same objects as working script)
# ─────────────────────────────────────────────
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)

embeddings = HFInferenceEmbeddings(token=HF_TOKEN)
llm = HFInferenceLLM()

print("✅ Embeddings ready (via HF Inference API)")
print(f"✅ LLM ready ({LLM_MODEL})")


# ─────────────────────────────────────────────
# 1. CREATE KNOWLEDGE BASE FILES
# ─────────────────────────────────────────────
with open("ai_concepts.txt", "w") as f:
    f.write("""
Deep Learning is a subset of machine learning using neural networks
with multiple layers. It excels at pattern recognition in images,
speech, and text.

Transfer Learning allows using pre-trained models as a starting point
for new tasks, saving time and computational resources.

Fine-tuning adjusts a pre-trained model for specific tasks or domains.
""")

with open("python_tips.txt", "w") as f:
    f.write("""
Python Best Practices:

1. Use virtual environments (venv) for project isolation
2. Follow PEP 8 style guide for readable code
3. Write docstrings for functions and classes
4. Use type hints for better code documentation
5. Handle exceptions explicitly with try-except blocks
""")

with open("langchain_guide.txt", "w") as f:
    f.write("""
LangChain is a framework for building LLM applications.

Key Components:
- Models: LLM integrations (OpenAI, Anthropic, etc.)
- Prompts: Templates for model inputs
- Chains: Sequences of operations
- Memory: Conversation history management
- Agents: Autonomous decision-making systems

Common Use Cases:
- Chatbots and conversational AI
- Question answering over documents
- Text summarization
- Code generation assistants
""")

print("✅ Knowledge base files written")


# ─────────────────────────────────────────────
# 2. LOAD ALL DOCUMENTS
# ─────────────────────────────────────────────
all_files = ["ai_concepts.txt", "python_tips.txt", "langchain_guide.txt"]
print(f"\nLoading {len(all_files)} documents...")
all_docs = load_multiple_documents(all_files)
print(f"✅ Loaded {len(all_docs)} documents total")


# ─────────────────────────────────────────────
# 3. SPLIT & STORE IN VECTOR DB
# ─────────────────────────────────────────────
chunks = text_splitter.split_documents(all_docs)
print(f"✅ Created {len(chunks)} chunks")

vectorstore_multi = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db_multi"
)
print("✅ Vector store created")



# ─────────────────────────────────────────────
# 4. BUILD RAG CHAIN  (LCEL style — same pattern as working script)
# ─────────────────────────────────────────────
template = """Answer the question based only on the following context.
If you cannot answer from the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)

retriever = vectorstore_multi.as_retriever(search_kwargs={"k": 5})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("✅ RAG chain ready")


# ─────────────────────────────────────────────
# 5. ASK CROSS-DOCUMENT QUESTIONS
# ─────────────────────────────────────────────
cross_doc_questions = [
    "How can I use Python for AI development?",
    "What's the relationship between deep learning and transfer learning?",
    "How does LangChain help build AI applications?"
]

print("\n" + "="*60)
print("TESTING MULTI-DOCUMENT RAG SYSTEM")
print("="*60)

for q in cross_doc_questions:
    answer = rag_chain.invoke(q)
    retrieved_docs = retriever.invoke(q)
    print(f"\n❓ {q}")
    print(f"💬 {answer}")
    print(f"📚 Sources: {len(retrieved_docs)} chunks used")
    print("="*60)
