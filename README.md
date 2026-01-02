# Research Synthesizer

**Enterprise-grade research synthesis powered by dual-LLM reasoning + synthesis architecture**

A lightweight, extensible research-assistant framework that combines deep analytical reasoning with polished report generation. Ingests local and remote documents, builds semantic vector indexes, and produces source-grounded technical outputs using configurable LLM pipelines.

> **Perfect for:** Research teams, knowledge synthesis, competitive analysis, literature reviews, technical due diligence, and rapid prototyping.

---

## ✨ Why Research Synthesizer?

### 🚀 **Speed & Responsiveness**
- **Instant UI startup** (~1-2s) with lazy model loading
- **Metadata-based document scanning** 500-1000x faster than hash-based approaches
- **Non-blocking operations** - UI stays responsive during indexing, reranking, synthesis
- **Efficient batch processing** for embeddings and reranking

### 🎯 **Accuracy & Grounding**
- **Dual-LLM architecture**: Separate reasoning (deep analysis) and synthesis (report generation) stages
- **Source-grounded output** - Every claim traceable to original documents
- **Multi-stage validation** - Reasoning validation + synthesis validation + quality checks
- **Smart reranking** - Cross-encoder reranking ensures top-quality results

### 🔀 **Multi-Source Intelligence**
- **Local documents** (PDF, TXT, MD) with semantic search
- **ArXiv papers** for academic research
- **Web search** via Tavily API for current information
- **Automatic deduplication** across sources

### 🛠️ **Developer-Friendly**
- **Modular architecture** - Easy to extend with new LLM providers or retrieval sources
- **Comprehensive configuration** - Control every aspect without code changes
- **Type-safe codebase** - Full type hints and LangGraph state management
- **Async operations** - Background tasks don't block the UI
- **Programmatic API** - Use as a Python library for custom workflows

---

## 🎬 Quick Start

### 1. Install

```bash
git clone <repository-url>
cd research-synthesizer-v1
pip install -r requirements.txt
```

### 2. Configure

Edit [config/settings.py](config/settings.py):
```python
"reasoning_model": "deepseek-r1:14b",      # Deep analysis
"synthesis_model": "qwen2.5:7b-instruct",  # Report generation
"embedding_model": "BAAI/bge-base-en-v1.5" # Semantic search
```

### 3. Run

```bash
python main.py
```

Open browser: `http://127.0.0.1:7860`

### 4. Research

1. Add documents to `./research_docs/`
2. Click "Rescan Documents" 
3. Enter your research question
4. Get source-grounded technical report in seconds

---

## 🎯 Core Capabilities

### Dual-LLM Architecture
- **Reasoning Model** (Stage 1): Deep analysis and critical thinking
- **Synthesis Model** (Stage 2): Polished report generation  
- **Single-Stage Mode**: Direct synthesis for speed
- **Lazy Loading**: Models initialize only on first use (instant UI startup)

### Multi-Source Research
- **Local documents**: PDF, TXT, MD with semantic search
- **ArXiv papers**: Direct academic paper search
- **Web search**: Current information via Tavily API
- **Automatic deduplication** across all sources

### Advanced Retrieval & Ranking
- **FAISS vector store** with stat-based document scanning (500-1000x faster than hash-based)
- **Semantic search** with embeddings for relevance
- **Cross-encoder reranking** for quality ranking
- **Batch processing** for efficiency
- **Smart deduplication** across sources and within vectorstore

### UI Features
- **Main Tab**: Research interface with query input and mode selection
- **Reports Tab**: Browse, load, and download previous reports
- **Tools Tab**: Database maintenance (async, non-blocking)
  - Rescan documents
  - Check for duplicates
  - Rebuild indexes
  - Optimize cache
  - Full database reset with backup
- **Config Tab**: Runtime information and status
- **Real-time progress** indicators and task status

---

## 📖 Core Usage Workflow

1. **Add Documents** 
   - Place PDF, TXT, or MD files in `./research_docs/`
   - Click "Rescan Documents" (runs async, non-blocking)

2. **Configure Research** (Optional)
   - Toggle Dual-LLM mode (reasoning + synthesis vs. direct synthesis)
   - Select sources: Local Docs, ArXiv, Web
   - Adjust temperature for creativity vs. consistency

3. **Run Query**
   - Enter research question
   - Click "Research"
   - First run loads models (~30-60s)
   - Subsequent runs use cached models (much faster)

4. **Review & Export**
   - View formatted report with citations
   - Access reasoning analysis (Dual-LLM mode)
   - Download or browse previous reports

---

## 📊 Project Structure

```
research-synthesizer-v1/
├── assets/                       # UI styling
│   └── style.css
├── config/                       # Configuration
│   ├── __init__.py
│   └── settings.py               # All configurable parameters
├── core/                         # Business logic & orchestration
│   ├── synthesizer.py            # Main coordinator
│   ├── state.py                  # Type-safe state management
│   ├── workflow_manager.py        # LangGraph workflow orchestration
│   ├── reasoning_controller.py    # Dual-LLM reasoning stage
│   ├── reasoning_validator.py     # Reasoning output validation
│   ├── synthesis_validator.py     # Synthesis output validation
│   └── report_manager.py          # Report persistence
├── llm/                          # Language model operations
│   ├── reasoning.py              # Deep analysis LLM (Stage 1)
│   ├── synthesis.py              # Report generation LLM (Stage 2)
│   └── query_optimizer.py         # Query refinement (optional)
├── retrieval/                    # Document management & search
│   ├── loader.py                 # Document loading and chunking
│   ├── vectorstore.py            # FAISS vector database operations
│   ├── reranker.py               # Cross-encoder reranking
│   ├── indexing_manager.py        # Document indexing coordination
│   ├── retrieval_manager.py       # Multi-source retrieval
│   ├── embeddings_cache.py        # Embeddings cache management
│   └── source_classifier.py       # Source metadata handling
├── ui/                           # User interface
│   ├── app.py                    # Gradio UI construction
│   ├── handlers.py               # Event handlers & async operations
│   ├── utils.py                  # UI utilities
│   └── img/                      # UI images
├── utils/                        # Shared utilities
│   ├── file.py                   # File operations and signatures
│   ├── text.py                   # Text processing
│   └── references.py             # Citation handling
├── main.py                       # Application entry point
├── requirements.txt              # Python dependencies
└── readme.md                     # This file
```

---

## 🔧 Installation & Configuration

### Requirements
- Python 3.9+
- PyTorch with or without CUDA
- ~8GB RAM minimum (more for larger models)
- ~2GB disk space for vector database

### Step-by-Step Setup

1. **Clone and install**
   ```bash
   git clone <repository-url>
   cd research-synthesizer-v1
   pip install -r requirements.txt
   ```

2. **Configure [config/settings.py](config/settings.py)**
   ```python
   CONFIG_BALANCED = {
       # === Models ===
       "reasoning_model": "deepseek-r1:14b",
       "synthesis_model": "qwen2.5:7b-instruct",
       "embedding_model": "BAAI/bge-base-en-v1.5",
       "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
       
       # === Temperature (lower = focused, higher = creative) ===
       "reasoning_temperature": 0.15,
       "synthesis_temperature": 0.3,
       
       # === Token Limits ===
       "context_window": 32768,
       "reasoning_max_tokens": 10240,
       "synthesis_max_tokens": 6144,
       
       # === Retrieval ===
       "top_k_initial": 50,
       "top_k_final": 15,
       "use_reranking": True,
       
       # === Features ===
       "use_two_stage_synthesis": True,
       "use_fast_scan": True,
       "use_llm_query_optimization": True,
       
       # === API Keys (optional) ===
       # "tavily_api_key": "your-key-here",
   }
   ```

3. **Set environment variables** (optional)
   ```bash
   export TAVILY_API_KEY="your-api-key"        # For web search
   export HF_HUB_OFFLINE="1"                   # Run completely offline
   export HF_HUB_DISABLE_TELEMETRY="1"        # Disable telemetry
   ```

4. **Create directories** (auto-created, but you can pre-create)
   ```
   ./research_docs/     # Your documents here
   ./vector_db/         # Vector database (auto-created)
   ./_reports/          # Generated reports
   ./.embeddings_cache/ # Cached embeddings
   ```

5. **Launch the application**
   ```bash
   python main.py
   ```
   
   Application appears at: `http://127.0.0.1:7860`

---

## 💻 Programmatic Usage

Use Research Synthesizer as a Python library:

```python
from core.synthesizer import ResearchSynthesizer
from config.settings import CONFIG_BALANCED

# Initialize
assistant = ResearchSynthesizer(**CONFIG_BALANCED)

# Index documents
assistant.scan_and_load_documents()

# Run research query
report = assistant.research(
    query="What are recent advances in quantum computing?",
    use_docs=True,      # Search local documents
    use_arxiv=True,     # Search ArXiv
    use_web=True,       # Search web (requires API key)
    save_report=True    # Save to reports directory
)

print(report)

# Advanced usage
search_results = assistant.retrieval_manager.search_documents(
    query="quantum computing",
    top_k=10
)

cache_stats = assistant.get_cache_stats()
assistant.optimize_cache(keep_recent=7)

duplicates = assistant.check_for_duplicates()
if duplicates:
    assistant.clear_duplicates_and_rebuild()
```

---

## 🚀 Advanced Features

### Fast Document Scanning

By default, uses **stat-based signatures** (file size + modification time):
- **Speed**: 500-1000x faster than MD5 hashing
- **Automatic migration**: Converts legacy hash-based signatures on first scan
- **Fallback**: Set `use_fast_scan: False` in config for cryptographic verification

```python
from retrieval.loader import DocumentIndexer
indexer = DocumentIndexer(config)
indexer.scan_documents("./research_docs/")
```

### Embeddings Cache Management

Cached embeddings stored in `./.embeddings_cache/` directory:

```python
assistant.get_cache_stats()           # View cache usage
assistant.optimize_cache(keep_recent=7)  # Keep only recent embeddings
assistant.clear_embeddings_cache()     # Remove all cached embeddings
```

### Multi-Source Result Deduplication

The system handles deduplication across sources:
- **Local documents**: Exact match detection via FAISS
- **ArXiv papers**: Metadata-based deduplication
- **Web results**: URL-based deduplication
- **Cross-source**: Semantic similarity handling

### Async Background Operations

All time-consuming operations run in the background without blocking the UI:

```
UI remains responsive while:
├─ Scanning and indexing documents
├─ Checking for embeddings duplicates
├─ Rebuilding vector database
├─ Optimizing cache
└─ Running full database reset (with automatic backup)
```

Status updates display in the Task Status section when operations complete.

---

## 🏗️ System Architecture

### Component Responsibilities

**[config/](config/)** - Centralized configuration
- Single `settings.py` with all configurable parameters
- Easy to modify for different workflows or hardware
- Model paths, batch sizes, retrieval settings, feature toggles

**[core/](core/)** - Business logic orchestration
- `synthesizer.py`: Main coordinator class that initializes and orchestrates all subsystems
- `state.py`: Type-safe state management using LangGraph State objects
- `reasoning_controller.py`: Manages dual-LLM reasoning stage execution
- `reasoning_validator.py`: Validates reasoning output quality and extracts sources
- `synthesis_validator.py`: Validates final synthesis quality
- `report_manager.py`: Saves and loads research reports with metadata
- `workflow_manager.py`: LangGraph workflow orchestration for multi-source research

**[retrieval/](retrieval/)** - Document management and search
- `loader.py`: Fast document scanning, loading, and chunking
  - Stat-based signatures (500-1000x faster than hash-based)
  - Supports PDF, TXT, MD formats
  - Smart migration from legacy hash signatures
- `vectorstore.py`: FAISS vector database operations and persistence
- `reranker.py`: Cross-encoder reranking for improved result quality
- `indexing_manager.py`: Coordinates document indexing and embedding caching
- `embeddings_cache.py`: Manages cached embeddings for performance
- `retrieval_manager.py`: Multi-source retrieval (documents, ArXiv, web)
- `source_classifier.py`: Classifies and normalizes source metadata

**[llm/](llm/)** - Language model operations
- `reasoning.py`: Deep analysis stage (Stage 1 of Dual-LLM)
- `synthesis.py`: Report generation stage (Stage 2 or single-stage)
- `query_optimizer.py`: Optional LLM-based query refinement before search

**[ui/](ui/)** - User interface and async task handling
- `app.py`: Gradio UI construction with theme system and responsive layout
- `handlers.py`: Event handlers and async background task management
- `utils.py`: UI helper functions for status displays and config info

**[utils/](utils/)** - Shared utilities
- `file.py`: Fast file signatures, legacy hash support, document loading
- `text.py`: Text processing, sanitization, chunk preparation
- `references.py`: Citation and reference extraction from documents

### Initialization Order

The `ResearchSynthesizer` class initializes components in this order:

1. Configuration loading
2. Device detection (CUDA/CPU)
3. Embeddings model (with caching)
4. FAISS vector store
5. Document reranker
6. Document indexing manager
7. Retrieval manager
8. LLM engines (lazy loaded on first use)
9. Controllers and validators
10. Report manager
11. Workflow manager

This order is critical because dependencies must be initialized before consumers.

---

## 📊 System Flow Diagram

```mermaid
graph TD
    A["👤 User Input<br/>Research Query"] -->|Enter Question| B["🖥️ UI: Gradio App<br/>handlers.py"]
    
    B -->|Check Mode| C{Single-LLM<br/>or<br/>Dual-LLM?}
    
    C -->|Single-Stage| D0["⚡ Query Optimization<br/>query_optimizer.py<br/>(optional)"]
    C -->|Dual-Stage| D0
    
    D0 -->|Optimized Query| E["📚 Multi-Source Retrieval"]
    
    E --> E1["🔍 Local: Vector Search<br/>vectorstore.py"]
    E --> E2["🌐 ArXiv: API Search<br/>retrieval_manager.py"]
    E --> E3["🌍 Web: Tavily Search<br/>Tavily API"]
    
    E1 --> F["✂️ Merge & Deduplicate<br/>retrieval_manager.py"]
    E2 --> F
    E3 --> F
    
    F -->|Top-K Initial| G["📊 Rerank Results<br/>reranker.py"]
    G -->|Top-K Final| H["📄 Chunk Processing<br/>loader.py"]
    
    H -->|Context| I{Single-LLM<br/>or<br/>Dual-LLM?}
    
    I -->|Single-Stage| J["⚡ Direct Synthesis<br/>synthesis.py"]
    I -->|Dual-Stage| K["🧠 Stage 1: Reasoning<br/>reasoning.py"]
    
    K -->|Deep Analysis| L["✅ Validate & Extract<br/>reasoning_validator.py"]
    L -->|Analysis + Sources| J
    
    J -->|Report + Extract<br/>References| M["📋 Final Report<br/>with Citations"]
    
    M -->|Validate Quality| N["✅ Validate Synthesis<br/>synthesis_validator.py"]
    N -->|Validated Report| O["💾 Save Report<br/>report_manager.py"]
    
    O -->|Show Results| B
    B -->|Display| A
    
    P["⚙️ Config<br/>settings.py"] -.->|Parameters| B
    P -.->|Model Paths| K
    P -.->|Model Paths| J
    P -.->|Retrieval Settings| E
    
    style A fill:#e1f5ff
    style B fill:#fff3e0
    style D0 fill:#f0f4ff
    style E fill:#e8f5e9
    style F fill:#e8f5e9
    style G fill:#e8f5e9
    style H fill:#e8f5e9
    style J fill:#f3e5f5
    style K fill:#f3e5f5
    style L fill:#fce4ec
    style M fill:#fce4ec
    style N fill:#fce4ec
    style O fill:#e0f2f1
    style P fill:#ede7f6
```

### Data Flow Summary

**Query → Optimization → Multi-Source Retrieval → Processing → Validation → Storage → Display**

1. **Input**: User enters research query in Gradio UI
2. **Query Optimization** (optional):
   - LLM-based query refinement via `query_optimizer.py` (if enabled)
   - Improves retrieval accuracy before search
3. **Multi-Source Retrieval** (parallel):
   - Local documents: Vector similarity search (FAISS)
   - ArXiv: Direct API search
   - Web: Tavily Search API
   - Results deduplicated and merged by `retrieval_manager.py`
4. **Result Refinement**:
   - Cross-encoder reranking for quality ranking
   - Top-K filtering (top_k_initial → top_k_final)
   - Chunk processing and contextualization
5. **Processing Pipeline**:
   - **Dual-LLM Mode**: 
     - Stage 1: Deep reasoning analysis via `reasoning.py` (with retry loop)
     - Validation and source extraction via `reasoning_validator.py`
     - Stage 2: Synthesis into polished report via `synthesis.py`
   - **Single-LLM Mode**: Direct synthesis via `synthesis.py` (faster)
6. **Reference Handling**:
   - Sources extracted during reasoning validation (Dual-LLM) or synthesis (both modes)
   - References embedded in final report with citations
7. **Validation**:
   - Final report quality validated via `synthesis_validator.py`
   - Ensures accuracy and source grounding
8. **Storage & Display**: 
   - Report saved to `./reports/` with metadata via `report_manager.py`
   - JSON format with query, results, and references
   - Results displayed in UI chat interface
   - Previous reports accessible in Reports tab

---

## 🛠️ Development & Extension

### Project Architecture Principles

1. **Separation of Concerns**: Each module has a single responsibility
2. **Lazy Loading**: Models initialize only when needed for better UX
3. **Async Operations**: All I/O and long-running tasks run in background
4. **Type Safety**: Extensive use of type hints and LangGraph StateGraph
5. **Configuration-Driven**: Behavior controlled via `config/settings.py`
6. **Modular Retrieval**: Easy to add new search sources (local, web, APIs)

### Adding New Features

**New retrieval source** (e.g., Google Scholar, database API):
1. Create new method in `retrieval/retrieval_manager.py`
2. Add to `WorkflowManager` nodes in `core/workflow_manager.py`
3. Include in state update flow
4. Add configuration option in `config/settings.py`

**New LLM provider** (e.g., Claude, GPT-4, custom model):
1. Create module in `llm/` (e.g., `llm/claude_integration.py`)
2. Inherit from base LLM interface or implement compatible API
3. Add model initialization in `synthesizer.py`
4. Update configuration in `settings.py`

**New UI component or tab**:
1. Add Gradio elements in `ui/app.py`
2. Create handler in `ui/handlers.py` if async operation needed
3. Style with CSS in `assets/style.css`
4. Update layout/structure in appropriate section

### Code Style

- **PEP 8**: Follow Python style guidelines
- **Type Hints**: Always use for function parameters and returns
- **Docstrings**: Module, class, and function level
- **Comments**: Explain "why", not "what"
- **Imports**: Group standard library, third-party, local (alphabetically)
- **Max line length**: 120 characters

### Testing

**Quick validation**:
```bash
# Test document loading
python -c "from core.synthesizer import ResearchSynthesizer; from config.settings import CONFIG_BALANCED; assistant = ResearchSynthesizer(**CONFIG_BALANCED); print('✓ Core system ready')"
```

**Comprehensive testing**:
```bash
pip install pytest flake8 black mypy
python -m pytest tests/ -v
flake8 . --max-line-length=120
mypy . --ignore-missing-imports
```

---

## 🔍 Troubleshooting

### Common Issues and Solutions

**Documents Not Indexing**
- Verify file formats: Only PDF, TXT, MD supported
- Check file permissions in `./research_docs/`
- Check console for errors during rescan
- Rescan runs async - check status panel after moment
- Try "Reset Database" if stuck

**Model Loading Takes Too Long**
- First model load: 30-60 seconds depending on size
- Subsequent queries use cached models (much faster)
- Use smaller models for faster loading
- Check available RAM and GPU memory

**CUDA/GPU Errors**
- Set `"embedding_device": "cpu"` to disable GPU
- Check GPU memory: Reduce `embedding_batch_size`
- Verify PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`

**Duplicate Embeddings**
- Use "Check Duplicates" tool in Tools tab
- Click "Rebuild Index" if duplicates found
- Last resort: "Reset Database" button (creates backup)

**Web Search Not Working**
- Set `TAVILY_API_KEY` environment variable
- Get key from https://tavily.com
- Verify API key has quota

**Out of Memory Errors**
- Reduce batch sizes: `embedding_batch_size: 32`
- Reduce `top_k_initial` and `top_k_final`
- Use smaller models
- Switch to CPU: `embedding_device: "cpu"`

### Performance Optimization

| Issue | Solution |
|-------|----------|
| Slow document scanning | `use_fast_scan: True` (500-1000x faster) |
| Slow embeddings | Increase `embedding_batch_size` or use GPU |
| Slow reranking | Reduce `top_k_initial` or increase batch size |
| High memory | Reduce `chunk_size` or batch sizes |
| Slow model loading | Use smaller models |

---

## 📚 References

### Key Technologies
- **LangChain**: Orchestration and LLM integration
- **LangGraph**: Workflow state management and graph-based execution
- **FAISS**: Vector similarity search at scale
- **Gradio**: Web UI framework
- **Transformers**: Hugging Face models for embeddings and cross-encoding
- **PyTorch**: Deep learning backend
- **ArXiv API**: Academic paper search
- **Tavily Search API**: Web search integration

### Important Entry Points
- [main.py](main.py) - Application entry point
- [config/settings.py](config/settings.py) - All configuration options
- [core/synthesizer.py](core/synthesizer.py) - Main coordinator class
- [ui/app.py](ui/app.py) - Gradio UI and layout
- [core/workflow_manager.py](core/workflow_manager.py) - LangGraph workflow

---

## License

This repository is for portfolio and showcase purposes only.

The source code is private and not licensed for use, modification, or redistribution.
All rights reserved.


## Support

For questions or general feedback, please open an issue on this repository.
