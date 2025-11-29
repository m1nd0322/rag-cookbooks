# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is an educational cookbook repository from Athina AI containing 16 Jupyter notebooks demonstrating advanced and agentic Retrieval-Augmented Generation (RAG) techniques. The repository progresses from naive RAG to sophisticated agentic approaches, with each notebook including evaluation using Athina AI.

**Structure:**
- `advanced_rag_techniques/` - 8 notebooks covering RAG patterns (naive, hybrid, HyDE, parent document retriever, fusion, contextual, rewrite-retrieve-read, unstructured)
- `agentic_rag_techniques/` - 5 notebooks covering agentic patterns (basic agentic, corrective RAG, self RAG, adaptive RAG, DeepSeek example)
- `agent_techniques/` - 3 notebooks covering agent frameworks (ReAct, Reflexion, ReWOO)
- `data/` - Sample datasets (context.csv with historical facts, tesla_q3.pdf)

## Development Environment

**Designed for Google Colab:**
- All notebooks are optimized for Google Colab execution
- Each notebook is self-contained and can run independently
- API keys are expected via Google Colab's `userdata` (e.g., `userdata.get('OPENAI_API_KEY')`)

**Python Package Setup:**
- A `requirements.txt` file is now available in the root directory
- Install all core dependencies: `pip install -r requirements.txt`
- Note: Some packages (athina, yfinance) have complex dependencies and should be installed separately if needed
- Tested with Python 3.11

## Running Notebooks

**In Google Colab:**
1. Open the notebook directly in Colab using the badge links in README.md
2. Add required API keys to Colab Secrets (userdata)
3. Run cells sequentially from top to bottom

**Locally:**
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Optional packages (install separately if needed):
   - `pip install athina>=1.7.0` - For RAG evaluation
   - `pip install yfinance>=0.2.0` - For finance data (one notebook)
4. Open notebooks: `jupyter notebook`
5. Modify API key loading from `userdata.get()` to `os.getenv()` or direct strings

## Common Notebook Structure

Every notebook follows this consistent pattern:

1. **Installation** - Install required dependencies via pip
2. **API Key Setup** - Load API keys (OpenAI, Pinecone, Athina AI, etc.)
3. **Indexing** - Load documents, split into chunks, create embeddings, initialize vector store
4. **Retriever Setup** - Configure retrieval strategy (simple, hybrid, multi-query, etc.)
5. **RAG Chain/Agent** - Build the generation pipeline or agentic workflow
6. **Inference** - Run example queries to test the implementation
7. **Evaluation** - Evaluate RAG performance using Athina AI metrics

## Technology Stack

**Core Frameworks:**
- **LangChain** - Main orchestration framework for RAG pipelines
- **LangGraph** - State machine framework for agentic workflows (uses StateGraph, START, END nodes)
- **LangSmith** - Tracing and monitoring (via `set_debug(True)`)
- **Athina AI** - RAG evaluation and monitoring

**LLM Providers:**
- OpenAI (GPT-4o, GPT-3.5-turbo) - Primary
- Google Gemini (gemini-2.0-flash-exp) - Alternative
- DeepSeek - Used in some examples

**Vector Databases:**
- Pinecone - Cloud-native (naive_rag.ipynb)
- Chromadb - Embedded (hybrid_rag.ipynb, parent_document_retriever.ipynb, contextual_rag.ipynb, corrective_rag.ipynb)
- FAISS - Facebook's similarity search (basic_agentic_rag.ipynb, self_rag.ipynb, adaptive_rag.ipynb, basic_unstructured_rag.ipynb)
- Qdrant - Open-source vector search (fusion_rag.ipynb)
- Weaviate - Open-source vector DB (hyde_rag.ipynb)

**Common Patterns:**
- Embeddings: OpenAI Embeddings or HuggingFace (BAAI/bge-small-en-v1.5)
- Document Loaders: CSVLoader, PyPDFLoader, UnstructuredMarkdownLoader
- Text Splitting: RecursiveCharacterTextSplitter, CharacterTextSplitter
- Hybrid Search: BM25 + vector search using EnsembleRetriever
- Agentic Tools: create_retriever_tool, TavilySearchResults

## Key Architectural Patterns

**Advanced RAG Patterns:**
- **Hybrid Retrieval**: Combines dense vector search with sparse BM25 for better recall
- **Multi-Query**: Generates multiple query variations for RAG Fusion with reciprocal rank fusion
- **Parent-Child**: Stores small chunks for retrieval but returns larger parent documents for context
- **Contextual Compression**: Uses LLMChainExtractor to compress retrieved docs to relevant portions only
- **Query Rewriting**: Rewrites user query before retrieval to improve results

**Agentic Patterns:**
- **LangGraph State Machines**: Define workflow with TypedDict state, conditional edges, and nodes
- **Self-Reflection**: Agents grade document relevance, check hallucinations, answer quality
- **Routing**: Classify queries to route to vector store, web search, or direct LLM
- **Corrective RAG**: Grade retrieved docs, filter irrelevant ones, fall back to web search if needed
- **Tool-based Agents**: Use create_react_agent or AgentExecutor with retriever and search tools

## Evaluation with Athina AI

All notebooks include RAG evaluation using Athina AI's framework:

```python
from athina.evals import RagasContextRelevancy, RagasAnswerRelevancy, RagasFaithfulness
from athina.loaders import RagasLoader

# Run evaluations
dataset = RagasLoader().load(...)
RagasContextRelevancy().run_batch(data=dataset).to_df()
RagasFaithfulness().run_batch(data=dataset).to_df()
RagasAnswerRelevancy().run_batch(data=dataset).to_df()
```

**Key Metrics:**
- Context Relevancy - Are retrieved documents relevant to the query?
- Faithfulness - Is the answer grounded in the retrieved context?
- Answer Relevancy - Does the answer directly address the query?

## Modifying Notebooks

**When adding new techniques:**
- Follow the 7-step structure (Install → API → Index → Retrieve → Chain → Infer → Eval)
- Include Athina AI evaluation at the end
- Make notebooks self-contained with inline pip installs
- Add example queries that demonstrate the technique's strengths
- Include comments explaining key differences from naive RAG

**When updating existing notebooks:**
- Preserve the self-contained nature (don't extract shared utilities)
- Keep compatibility with Google Colab's userdata API
- Update both the technique implementation AND evaluation sections
- Test in Colab before committing

**Common dependencies pattern:**
```python
!pip install -qU langchain langchain-openai langchain-community athina
# Add technique-specific deps (e.g., chromadb, pinecone, etc.)
```

**Note on Pinecone:** The package was renamed from `pinecone-client` to `pinecone`. Older notebooks may use `pinecone-client` imports, but both work with the new `pinecone` package.

## API Keys Required

Most notebooks require these API keys in Google Colab Secrets:
- `OPENAI_API_KEY` - For LLM and embeddings (required for almost all notebooks)
- `ATHINA_API_KEY` - For RAG evaluation (required for all notebooks)
- Vector DB keys (one of):
  - `PINECONE_API_KEY` - For Pinecone notebooks
  - No key needed for Chromadb/FAISS (embedded)
  - `QDRANT_API_KEY` and `QDRANT_URL` - For Qdrant notebooks
  - `WEAVIATE_API_KEY` and `WEAVIATE_URL` - For Weaviate notebooks
- `TAVILY_API_KEY` - For agentic notebooks using web search
- `LANGSMITH_API_KEY` - Optional, for tracing in LangSmith-enabled notebooks

## Common Issues and Solutions

**Issue: ModuleNotFoundError for a package**
- Each notebook has its own pip install cell - run it first
- For local development, you may need to install packages globally or in a venv

**Issue: API key errors**
- In Colab: Add keys to Secrets (left sidebar lock icon)
- Locally: Replace `userdata.get('KEY')` with `os.getenv('KEY')` and set environment variables

**Issue: Vector store already exists errors**
- Chromadb: Delete the persist directory or change the collection name
- Pinecone: Delete the index or use a different index name
- FAISS: Re-run the indexing cell to overwrite

**Issue: LangGraph state machine errors**
- Ensure all nodes return the full state dict with updates
- Check conditional edge functions return valid next node names
- Verify StateGraph has START and END connections
