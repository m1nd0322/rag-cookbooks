"""
Test script to check LangChain compatibility with version 1.1.0+
Tests all imports used in the advanced_rag_techniques notebooks
"""

import sys
print(f"Python version: {sys.version}")

# Test core LangChain imports
print("\n=== Testing LangChain Core Imports ===")
try:
    from langchain_core.prompts import ChatPromptTemplate
    print("[OK] langchain_core.prompts.ChatPromptTemplate")
except ImportError as e:
    print(f"[FAIL] langchain_core.prompts.ChatPromptTemplate: {e}")

try:
    from langchain_core.runnables import RunnablePassthrough
    print("[OK] langchain_core.runnables.RunnablePassthrough")
except ImportError as e:
    print(f"[FAIL] langchain_core.runnables.RunnablePassthrough: {e}")

try:
    from langchain_core.output_parsers import StrOutputParser
    print("[OK] langchain_core.output_parsers.StrOutputParser")
except ImportError as e:
    print(f"[FAIL] langchain_core.output_parsers.StrOutputParser: {e}")

try:
    from langchain_core.documents import Document
    print("[OK] langchain_core.documents.Document")
except ImportError as e:
    print(f"[FAIL] langchain_core.documents.Document: {e}")

try:
    from langchain_core.load import dumps, loads
    print("[OK] langchain_core.load.dumps, loads")
except ImportError as e:
    print(f"[FAIL] langchain_core.load: {e}")

try:
    from langchain_core.stores import InMemoryStore
    print("[OK] langchain_core.stores.InMemoryStore")
except ImportError as e:
    print(f"[FAIL] langchain_core.stores.InMemoryStore: {e}")

# Test community imports
print("\n=== Testing LangChain Community Imports ===")
try:
    from langchain_community.document_loaders import CSVLoader
    print("[OK] langchain_community.document_loaders.CSVLoader")
except ImportError as e:
    print(f"[FAIL] langchain_community.document_loaders.CSVLoader: {e}")

try:
    from langchain_community.vectorstores import Pinecone, Chroma, FAISS, Qdrant
    print("[OK] langchain_community.vectorstores (Pinecone, Chroma, FAISS, Qdrant)")
except ImportError as e:
    print(f"[FAIL] langchain_community.vectorstores: {e}")

try:
    from langchain_community.retrievers import BM25Retriever
    print("[OK] langchain_community.retrievers.BM25Retriever")
except ImportError as e:
    print(f"[FAIL] langchain_community.retrievers.BM25Retriever: {e}")

try:
    from langchain_community.document_compressors import LLMChainExtractor
    print("[OK] langchain_community.document_compressors.LLMChainExtractor")
except ImportError as e:
    print(f"[FAIL] langchain_community.document_compressors.LLMChainExtractor: {e}")

# Test text splitters
print("\n=== Testing Text Splitters ===")
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    print("[OK] langchain_text_splitters.RecursiveCharacterTextSplitter")
except ImportError as e:
    print(f"[FAIL] langchain_text_splitters.RecursiveCharacterTextSplitter: {e}")

# Test retrievers
print("\n=== Testing Retrievers ===")
try:
    from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever, ParentDocumentRetriever
    print("[OK] langchain.retrievers (EnsembleRetriever, ContextualCompressionRetriever, ParentDocumentRetriever)")
except ImportError as e:
    print(f"[FAIL] langchain.retrievers: {e}")

# Test OpenAI
print("\n=== Testing LangChain OpenAI ===")
try:
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    print("[OK] langchain_openai (OpenAIEmbeddings, ChatOpenAI)")
except ImportError as e:
    print(f"[FAIL] langchain_openai: {e}")

# Test Weaviate
print("\n=== Testing Weaviate ===")
try:
    from langchain_weaviate.vectorstores import WeaviateVectorStore
    print("[OK] langchain_weaviate.vectorstores.WeaviateVectorStore")
except ImportError as e:
    print(f"[FAIL] langchain_weaviate: {e}")

# Check LangChain versions
print("\n=== LangChain Versions ===")
try:
    import langchain
    print(f"langchain: {langchain.__version__}")
except:
    print("langchain: Not installed")

try:
    import langchain_core
    print(f"langchain_core: {langchain_core.__version__}")
except:
    print("langchain_core: Not installed")

try:
    import langchain_community
    print(f"langchain_community: {langchain_community.__version__}")
except:
    print("langchain_community: Not installed")

print("\n=== Test Complete ===")
