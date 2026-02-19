# 📚 RAG System - Production Patterns with HuggingFace API

> Three implementations showing RAG progression from basic retrieval
> to advanced multi-query with conversation memory.

**Challenge Solved:** Intel Mac PyTorch compatibility → HuggingFace Inference API  
**Built:** February 2026  
**Framework:** LangChain + ChromaDB + HuggingFace

---

## 🎯 Three Implementations

### 1️⃣ Basic RAG
**File:** `rag_basic.py`

Core RAG pattern using HuggingFace Inference API (no local PyTorch).

**Features:**
- ✅ Document loading and chunking
- ✅ HF Inference API for embeddings (no local models)
- ✅ ChromaDB vector store
- ✅ Basic retrieval + generation
- ✅ **Solved:** PyTorch compatibility on Intel Mac

**Key Innovation:** Custom `HFInferenceEmbeddings` class bypasses
local PyTorch requirements entirely.

---

### 2️⃣ Multi-Document RAG
**File:** `rag_multi_docs.py`

Scales basic RAG to multiple knowledge sources.

**Features:**
- ✅ Multiple document loading
- ✅ Cross-document queries
- ✅ Source attribution (which docs were used)
- ✅ Same HF API pattern (no PyTorch)

**Use Case:** Knowledge bases spanning multiple files/topics.

---

### 3️⃣ Extended RAG
**File:** `rag_extended.py`

Production-grade RAG with advanced features.

**Features:**
- ✅ **Conversation Memory** - Multi-turn dialogue
- ✅ **Streaming** - Token-by-token output
- ✅ **Multi-Query Retrieval** - LLM rewrites questions for better retrieval
- ✅ Deduplicated results
- ✅ Full conversation history

**Use Case:** Production chatbots, conversational RAG systems.

---

## 🔧 The PyTorch Problem & Solution

### The Challenge

```python
# ❌ Traditional approach (fails on Intel Mac)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
# Requires PyTorch > 2.4
# Not available on Intel Mac with Python 3.11
```

### The Solution

```python
# ✅ Our approach (works everywhere)
from huggingface_hub import InferenceClient

class HFInferenceEmbeddings(Embeddings):
    def embed_query(self, text: str) -> List[float]:
        return self.client.feature_extraction(text, model=self.model)

# Runs on HuggingFace servers, not your Mac
# No PyTorch needed locally
```

**Key Insight:** Move computation to the cloud via API instead
of fighting local dependencies.

---

## 🚀 Quick Start

### Prerequisites

```bash
# 1. Get HuggingFace token
# Sign up: https://huggingface.co/join
# Get token: Settings → Access Tokens

# 2. Install dependencies
pip install -r requirements.txt

# 3. create .env: Add your token
HF_TOKEN = "your_token_here"
```

### Run Basic RAG

```bash
python rag_basic.py
```

**Output:**
```
✅ Loaded 1 documents
✅ Created 4 chunks
✅ Embeddings ready (via HF Inference API)
✅ Vector store created
✅ LLM ready (HuggingFaceTB/SmolLM3-3B)
✅ RAG chain ready

❓ Question: What is RAG?
💬 Answer: RAG combines retrieval of relevant information 
           with generation...
📄 Sources used (3 chunks)
```

### Run Multi-Document RAG

```bash
python rag_multi_docs.py
```

Creates 3 knowledge base files, loads all, answers cross-document questions.

### Run Extended RAG

```bash
python rag_extended.py
```

**Demo conversation:**
```
❓ Question: What is deep learning?
💬 Answer: [streaming...] Deep learning is a subset of 
           machine learning using neural networks...

❓ Question: How does that relate to transfer learning?
           ↑ Agent remembers previous context!
💬 Answer: [streaming...] Transfer learning allows using 
           pre-trained deep learning models...
```

---

## 🏗️ Architecture

### Basic RAG Flow
```
User Question
    ↓
Embed via HF API
    ↓
Vector Search (ChromaDB)
    ↓
Retrieve Top K Chunks
    ↓
Build Prompt (Question + Context)
    ↓
LLM via HF API
    ↓
Answer
```

### Extended RAG Flow
```
User Question
    ↓
Multi-Query Rewriting (LLM generates N variants)
    ↓
Retrieve docs for each variant
    ↓
Deduplicate results
    ↓
Build Prompt (Question + Context + Memory)
    ↓
Streaming LLM Response
    ↓
Save to Conversation Memory
    ↓
Answer (with full context for next turn)
```

---

## 💡 Key Concepts

### From Basic Version:
- Document loading and chunking
- Embedding generation (remote)
- Vector similarity search
- Retrieval + generation pattern
- **HF Inference API workaround**

### From Multi-Doc Version:
- Multiple document sources
- Cross-document retrieval
- Source attribution
- Scaling RAG to larger knowledge bases

### From Extended Version:
- **Conversation Memory** - `ConversationBuffer` class
- **Streaming** - Generator-based token output
- **Multi-Query** - LLM rewrites questions for better coverage
- Deduplication by content
- Full conversation history management

---

## 📦 Project Structure

```
rag-system/
├── rag_basic.py           # Core RAG + HF API fix
├── rag_multi_docs.py      # Multiple documents
├── rag_extended.py        # Memory + streaming + multi-query
├── requirements.txt       # Dependencies
├── README.md             # This file
└── sample_docs/          # Created by scripts
    ├── sample_doc.txt
    ├── ai_concepts.txt
    ├── python_tips.txt
    └── langchain_guide.txt
```

---

## 🔑 Custom Components

### HFInferenceLLM
```python
class HFInferenceLLM(LLM):
    """LangChain LLM using HF Inference API"""
    
    def _call(self, prompt: str) -> str:
        # Standard blocking call
        
    def stream_call(self, prompt: str) -> Generator:
        # Streaming generator for token-by-token output
```

### HFInferenceEmbeddings
```python
class HFInferenceEmbeddings(Embeddings):
    """LangChain Embeddings using HF Inference API"""
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Batch embedding
        
    def embed_query(self, text: str) -> List[float]:
        # Single query embedding
```

### ConversationBuffer
```python
class ConversationBuffer:
    """Simple memory buffer for multi-turn conversations"""
    
    def add(self, question: str, answer: str) -> None:
        # Store turn
        
    def format(self) -> str:
        # Format for prompt injection
```

---

## 🎓 Learning Progression

```
Step 1: rag_basic.py
        ↓
      Learn RAG fundamentals
      Solve PyTorch dependency issue
        ↓
Step 2: rag_multi_docs.py
        ↓
      Scale to multiple documents
      Cross-document retrieval
        ↓
Step 3: rag_extended.py
        ↓
      Add memory, streaming, multi-query
      Production patterns
```

---

## 🌍 Production Deployment

### Replace HF Tokens

### Scale to Larger Documents

```python
# Adjust chunking for your use case
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Larger for books
    chunk_overlap=200,    # More overlap for context
)
```

### Use Better LLMs

```python
# Swap to more powerful models
LLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"  # Better quality
# or
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"  # Good balance
```

---

## 📊 Comparison

| Feature | Basic | Multi-Doc | Extended |
|---------|-------|-----------|----------|
| Documents | Single | Multiple | Multiple |
| Memory | ❌ | ❌ | ✅ |
| Streaming | ❌ | ❌ | ✅ |
| Multi-Query | ❌ | ❌ | ✅ |
| Source Attribution | ✅ | ✅ | ✅ |
| HF API (no PyTorch) | ✅ | ✅ | ✅ |
| Lines of Code | ~150 | ~180 | ~320 |

---

## 🔗 Related Projects

- **Weather Agent** - LangGraph agents with memory
- **HuggingFace Experiments** - Model inference patterns

---

## 🧠 Problem-Solving Highlight

**Challenge:** Intel Mac + Python 3.11 + PyTorch > 2.4 incompatibility

**Failed Approaches:**
- Downgrading Python versions
- Installing older PyTorch
- Using different embedding libraries

**Solution:** 
Move embedding computation to HuggingFace servers via Inference API.
Created custom `HFInferenceEmbeddings` class that works on ANY machine.

**Lesson:** Sometimes the best solution is architectural (API over local),
not configurational (fighting dependencies).

---

## 👤 Author

**Steven**  
Progression: Basic RAG → Multi-query with memory in 1 week

**Key Achievement:** Solved PyTorch dependency hell on Intel Mac
by pivoting to cloud-based inference.

**Technologies:**
- LangChain for RAG orchestration
- ChromaDB for vector storage
- HuggingFace Inference API (embeddings + LLM)
- Python 3.11 on Intel Mac

---

## 📝 Notes

This project demonstrates both RAG understanding AND 
real-world problem-solving. The progression from basic to 
extended shows ability to:

1. Implement core patterns (basic RAG)
2. Scale solutions (multi-document)
3. Add production features (memory, streaming, multi-query)
4. Solve real deployment challenges (PyTorch workaround)

All three implementations are production-ready for their
respective use cases.