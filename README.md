## LangChain Components: Load, Split, Embed, and Store

LangChain provides essential components to build LLM-powered applications. Below are the core components used in **Retrieval-Augmented Generation (RAG)** and other NLP workflows.

## 1. Load (Document Loading)
**Purpose:** Load raw data from various sources such as text files, PDFs, URLs, or databases.

**Common Loaders:**
- `TextLoader` (for `.txt` files)
- `PDFLoader` (for `.pdf` documents)
- `WebBaseLoader` (for URLs)
- `CSVLoader` (for `.csv` files)

**Example:**
```python
from langchain_community.document_loaders import TextLoader

loader = TextLoader("sample.txt")
documents = loader.load()
```

---

## 2. Split (Text Splitting)
**Purpose:** Break large documents into manageable chunks for efficient processing.

**Common Splitters:**
- `CharacterTextSplitter` (splits text based on character count)
- `RecursiveCharacterTextSplitter` (handles structured documents more effectively)
- `TokenTextSplitter` (splits text based on token count)

**Example:**
```python
from langchain_text_splitters import CharacterTextSplitter

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)
```

---

## 3. Embed (Vector Embeddings)
**Purpose:** Convert text into numerical representations (embeddings) that capture meaning.

**Common Embedding Models:**
- `OllamaEmbeddings` (local embedding model)
- `HuggingFaceEmbeddings` (uses Sentence Transformers)
- `OpenAIEmbeddings` (uses OpenAI API)

**Example:**
```python
from langchain_community.embeddings import OllamaEmbeddings

embeddings = OllamaEmbeddings()
vectorized_data = embeddings.embed_documents(["This is a sample text."])
```

---

## 4. Store (Vector Storage)
**Purpose:** Store and retrieve embeddings efficiently using a vector database.

**Common Vector Databases:**
- `FAISS` (lightweight and efficient)
- `ChromaDB` (popular local storage option)
- `Pinecone` (scalable cloud-based vector database)

**Example:**
```python
from langchain_community.vectorstores import FAISS

db = FAISS.from_documents(chunks, embeddings)
```

---

## Summary
| Component | Purpose | Example |
|-----------|---------|---------|
| **Load** | Load documents from various sources | `TextLoader("file.txt")` |
| **Split** | Divide documents into smaller chunks | `CharacterTextSplitter()` |
| **Embed** | Convert text into vector embeddings | `OllamaEmbeddings()` |
| **Store** | Store and retrieve embeddings | `FAISS.from_documents()` |

By combining these components, you can build powerful applications such as **chatbots, search engines, and AI assistants** using LangChain. 🚀

