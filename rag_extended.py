# rag_extended.py
"""
Extended Multi-Document RAG System with LangChain - 100% via HF Inference API

Layers added on top of rag_multi_doc.py:
  1. Memory      — simple buffer keeping full conversation history
  2. Streaming   — generator-based token streaming for use in apps
  3. Multi-query — LLM rewrites question N ways, merges retrieved docs
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
from typing import Any, Generator, Iterator, List, Optional
from dotenv import load_dotenv
import os

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
load_dotenv()
HF_TOKEN    = os.getenv("HF_TOKEN")
LLM_MODEL   = "HuggingFaceTB/SmolLM3-3B"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
NUM_QUERY_VARIANTS = 3   # how many rewrites multi-query generates


# ─────────────────────────────────────────────
# CUSTOM LLM — HF Inference API  (+ streaming support)
# ─────────────────────────────────────────────
class HFInferenceLLM(LLM):
    """
    LangChain LLM wrapper using HF Inference API.
    Adds stream_call() generator for chunk-by-chunk output.
    """

    model: str = LLM_MODEL
    token: str = HF_TOKEN
    max_tokens: int = 512   # bumped up — multi-query needs more room

    @property
    def _llm_type(self) -> str:
        return "hf-inference-api"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        """Standard blocking call — used internally by LangChain chains."""
        client = InferenceClient(provider="hf-inference", token=self.token)
        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content

    def stream_call(self, prompt: str) -> Generator[str, None, None]:
        """
        ── STREAMING ──
        Generator that yields text chunks as they arrive from the API.
        Use this when you want to display output token-by-token in an app.

        Example usage:
            for chunk in llm.stream_call("What is RAG?"):
                my_app.write(chunk)          # Streamlit
                websocket.send(chunk)        # WebSocket server
                sys.stdout.write(chunk)      # Terminal
        """
        client = InferenceClient(provider="hf-inference", token=self.token)
        stream: Iterator = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
            max_tokens=self.max_tokens,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


# ─────────────────────────────────────────────
# CUSTOM EMBEDDINGS — HF Inference API
# ─────────────────────────────────────────────
class HFInferenceEmbeddings(Embeddings):
    """LangChain Embeddings wrapper using HF Inference API — no local model needed."""

    def __init__(self, token: str, model: str = EMBED_MODEL):
        self.token  = token
        self.model  = model
        self.client = InferenceClient(provider="hf-inference", token=token)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.client.feature_extraction(t, model=self.model) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self.client.feature_extraction(text, model=self.model)


# ─────────────────────────────────────────────
# HELPER — load multiple documents
# ─────────────────────────────────────────────
def load_multiple_documents(file_paths: List[str]):
    all_docs = []
    for path in file_paths:
        loader = TextLoader(path)
        docs   = loader.load()
        all_docs.extend(docs)
        print(f"  📄 Loaded: {path} ({len(docs)} doc)")
    return all_docs


# ─────────────────────────────────────────────
# ── LAYER 1: MEMORY ──────────────────────────
# Simple buffer — stores every (question, answer) pair for the session.
# Injected into the prompt so the LLM has full conversation context.
# ─────────────────────────────────────────────
class ConversationBuffer:
    """
    Stores the full conversation history as a plain list of turns.
    Call .add(question, answer) after each exchange.
    Call .format() to get a formatted string ready for the prompt.
    Call .clear() to reset the session.
    """

    def __init__(self):
        self.history: List[dict] = []   # [{"question": ..., "answer": ...}]

    def add(self, question: str, answer: str) -> None:
        self.history.append({"question": question, "answer": answer})

    def format(self) -> str:
        if not self.history:
            return "No previous conversation."
        lines = []
        for i, turn in enumerate(self.history, 1):
            lines.append(f"Turn {i}:")
            lines.append(f"  Human: {turn['question']}")
            lines.append(f"  AI:    {turn['answer']}")
        return "\n".join(lines)

    def clear(self) -> None:
        self.history = []

    def __len__(self) -> int:
        return len(self.history)


# ─────────────────────────────────────────────
# ── LAYER 3: MULTI-QUERY RETRIEVER ───────────
# Rewrites the question N ways, retrieves docs for each variant,
# then deduplicates by page_content before returning the merged set.
# ─────────────────────────────────────────────
REWRITE_TEMPLATE = """You are an AI assistant helping improve document retrieval.
Given the question below, generate {n} different versions of it that mean the same
thing but are phrased differently. These variants will be used to retrieve documents,
so make each one distinct in wording and angle.

Output ONLY the {n} questions, one per line, no numbering, no extra text.

Original question: {question}"""

def generate_query_variants(question: str, llm: HFInferenceLLM, n: int = NUM_QUERY_VARIANTS) -> List[str]:
    """Ask the LLM to rephrase the question N ways."""
    prompt = REWRITE_TEMPLATE.format(n=n, question=question)
    raw    = llm._call(prompt)
    # Split on newlines, strip blanks, keep up to n variants
    variants = [line.strip() for line in raw.strip().splitlines() if line.strip()][:n]
    # Always include the original so we never lose coverage
    return [question] + variants


def multi_query_retrieve(question: str, retriever, llm: HFInferenceLLM):
    """
    ── MULTI-QUERY ──
    1. Generate N rephrased versions of the question
    2. Run each through the retriever
    3. Deduplicate by page_content
    4. Return the merged, unique doc list
    """
    variants = generate_query_variants(question, llm)
    print(f"\n🔀 Multi-query variants ({len(variants)} total):")
    for i, v in enumerate(variants):
        tag = "(original)" if i == 0 else f"(variant {i})"
        print(f"   {tag} {v}")

    seen    = set()
    all_docs = []
    for variant in variants:
        docs = retriever.invoke(variant)
        for doc in docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                all_docs.append(doc)

    print(f"   → {len(all_docs)} unique chunks retrieved across all variants")
    return all_docs


# ─────────────────────────────────────────────
# SHARED UTILITIES
# ─────────────────────────────────────────────
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)

embeddings = HFInferenceEmbeddings(token=HF_TOKEN)
llm        = HFInferenceLLM()
memory     = ConversationBuffer()

print("✅ Embeddings ready (via HF Inference API)")
print(f"✅ LLM ready ({LLM_MODEL})")
print("✅ Memory buffer initialised")


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
# 2. LOAD, SPLIT, EMBED
# ─────────────────────────────────────────────
all_files = ["ai_concepts.txt", "python_tips.txt", "langchain_guide.txt"]
print(f"\nLoading {len(all_files)} documents...")
all_docs = load_multiple_documents(all_files)

chunks = text_splitter.split_documents(all_docs)
print(f"✅ Created {len(chunks)} chunks")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db_extended"
)
print("✅ Vector store created")

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# ─────────────────────────────────────────────
# 3. PROMPT TEMPLATE  (includes memory slot)
# ─────────────────────────────────────────────
RAG_TEMPLATE = """You are a helpful assistant. Use the context and conversation
history below to answer the question. If the answer is not in the context,
say "I don't have enough information to answer that."

Conversation history:
{history}

Context from documents:
{context}

Question: {question}

Answer:"""


# ─────────────────────────────────────────────
# 4. CORE ASK FUNCTION
#    Wires together: multi-query → memory → streaming
# ─────────────────────────────────────────────
def ask(question: str, stream: bool = True) -> str:
    """
    Full pipeline:
      1. Multi-query retrieval  → richer, deduplicated context
      2. Memory injection       → conversation history in prompt
      3. Streaming generation   → yields chunks if stream=True

    Args:
        question: The user's question
        stream:   If True, prints tokens as they arrive and returns full answer.
                  If False, returns the complete answer string silently.

    Returns:
        The complete answer string (regardless of stream mode).
    """
    # Step 1 — multi-query retrieval
    docs    = multi_query_retrieve(question, retriever, llm)
    context = "\n\n".join(doc.page_content for doc in docs)

    # Step 2 — build prompt with memory
    prompt = RAG_TEMPLATE.format(
        history=memory.format(),
        context=context,
        question=question,
    )

    # Step 3 — generate answer
    print(f"\n💬 Answer: ", end="", flush=True)
    full_answer = ""

    if stream:
        # ── generator streaming ──
        for chunk in llm.stream_call(prompt):
            print(chunk, end="", flush=True)   # swap this line for your app's sink
            full_answer += chunk
        print()  # newline after streamed output
    else:
        full_answer = llm._call(prompt)
        print(full_answer)

    # Step 4 — save to memory buffer
    memory.add(question, full_answer)
    print(f"🧠 Memory: {len(memory)} turn(s) stored")

    return full_answer


# ─────────────────────────────────────────────
# 5. RUN A MULTI-TURN CONVERSATION
#    Each question builds on the last via memory.
# ─────────────────────────────────────────────
conversation = [
    "What is deep learning?",
    "How does that relate to transfer learning?",   # refers back to deep learning
    "How can I use Python best practices with LangChain?",
    "Can you summarise everything we've discussed?",  # tests full memory recall
]

print("\n" + "="*60)
print("MULTI-TURN RAG WITH MEMORY + STREAMING + MULTI-QUERY")
print("="*60)

for question in conversation:
    print(f"\n❓ Question: {question}")
    print("-" * 60)
    ask(question, stream=True)
    print("="*60)

# ─────────────────────────────────────────────
# 6. SHOW FULL MEMORY BUFFER AT END
# ─────────────────────────────────────────────
print("\n📋 Full conversation history stored in memory:")
print("-" * 60)
print(memory.format())