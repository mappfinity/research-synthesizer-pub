# Research Synthesizer

**Enterprise-grade research synthesis powered by dual-LLM reasoning + synthesis architecture**

A lightweight, extensible research-assistant framework that combines deep analytical reasoning with polished report generation. Ingests local and remote documents, builds semantic vector indexes, and produces source-grounded technical outputs using configurable LLM pipelines—with optional text-to-speech playback for listening to generated reports and summaries.

> **Perfect for:** Research teams, knowledge synthesis, competitive analysis, literature reviews, technical due diligence, rapid prototyping, and hands-free consumption of research outputs.

---
### ☀️ **UI (Light theme)**
![My Picture](img/ui.png)

### 🌙 **UI (Dark theme)**
![My Picture](img/ui_dark.png)

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
## Simplified High-Level System Flow

```mermaid
graph TB
    START["🚀 ResearchSynthesizer<br/>query | use_docs | use_arxiv | use_web"] --> UI["UI Layer<br/>Gradio Interface"]
    
    UI --> HANDLERS["Request Handlers<br/>research | model | validation"]
    HANDLERS --> STATE_INIT["State Initialization<br/>ResearchState()"]
    STATE_INIT --> WF["WorkflowManager<br/>Main Orchestration"]
    WF --> CONFIG["Load Configuration<br/>Settings | Models | Logging"]
    CONFIG --> ROUTE{{"Source<br/>Selection"}}
    
    %% DOCUMENT PATH
    ROUTE -->|use_docs| DOC_CHECK{{"Existing<br/>Index?"}}
    DOC_CHECK -->|No| DOC_SCAN["DocumentIndexer<br/>scan_for_new_documents()"]
    DOC_CHECK -->|Yes| DOC_LOAD["Load Existing Index<br/>From disk cache"]
    
    DOC_SCAN --> DOC_CLASSIFY["SourceTypeClassifier<br/>classify() → BOOK | PAPER"]
    DOC_CLASSIFY --> DOC_CHUNK["Document Chunking<br/>size:1000 | overlap:200"]
    DOC_CHUNK --> DOC_CACHE_CHK{{"Cache<br/>Hit?"}}
    
    DOC_CACHE_CHK -->|"Hit ~1ms"| DOC_CACHE_LOAD["EmbeddingsCacheManager<br/>load_batch()"]
    DOC_CACHE_CHK -->|"Miss ~100ms"| DOC_EMBED["Generate Embeddings<br/>HuggingFace | batch:32"]
    
    DOC_EMBED --> DOC_CACHE_SAVE["Cache Embeddings<br/>save_batch()"]
    DOC_CACHE_SAVE --> DOC_DUAL
    DOC_CACHE_LOAD --> DOC_DUAL
    DOC_LOAD --> DOC_DUAL
    
    DOC_DUAL["DualVectorStoreManager<br/>books_index | papers_index<br/>FAISS"]
    DOC_DUAL --> DOC_SAVE["Save Indices<br/>books_path | papers_path"]
    
    %% ARXIV PATH
    ROUTE -->|use_arxiv| ARXIV_OPT["QueryOptimizer<br/>LLM-powered expansion"]
    ARXIV_OPT --> ARXIV_API["ArXiv API Search<br/>max:50 | sort:relevance"]
    ARXIV_API --> ARXIV_FILTER["Filter Results<br/>relevance | dedup"]
    ARXIV_FILTER --> ARXIV_PDF_CHK{{"Fetch<br/>PDFs?"}}
    
    ARXIV_PDF_CHK -->|Yes| ARXIV_PDF["AsyncPDFTitleEnhancer<br/>process_queue() | async download"]
    ARXIV_PDF_CHK -->|No| ARXIV_EMBED
    ARXIV_PDF --> ARXIV_PARSE["PDF Parsing<br/>Extract text | ParsedSource()"]
    ARXIV_PARSE --> ARXIV_EMBED["Embed ArXiv Results<br/>HuggingFace | temp store"]
    
    %% WEB PATH
    ROUTE -->|use_web| WEB_OPT["QueryOptimizer<br/>LLM-powered expansion"]
    WEB_OPT --> TAVILY["Tavily API Search<br/>depth:advanced | max:20"]
    TAVILY --> WEB_FILTER["WebPDFEmbedder<br/>filter relevance > 0.5"]
    WEB_FILTER --> WEB_PDF_CHECK{{"PDF<br/>Content?"}}
    
    WEB_PDF_CHECK -->|Yes| PDF_MGR["PDFManager<br/>cache | download<br/>max:50MB | timeout:30s"]
    WEB_PDF_CHECK -->|No| WEB_PARSE["Parse HTML<br/>extract text | clean"]
    
    PDF_MGR --> PDF_PARSE["PDF Processing<br/>extract | ParsedSource()"]
    PDF_PARSE --> WEB_EMBED
    WEB_PARSE --> WEB_EMBED
    WEB_EMBED["Embed Web Results<br/>batch download | max_concurrent:5"]
    
    %% SEARCH & RETRIEVAL
    DOC_SAVE --> SEARCH["RetrievalManager<br/>search_all() k=100<br/>Books | Papers | ArXiv | Web"]
    ARXIV_EMBED --> SEARCH
    WEB_EMBED --> SEARCH
    
    SEARCH --> COMBINE["Combine & Deduplicate<br/>Merge sources | normalize metadata"]
    COMBINE --> WEIGHT_ADJ["Source Weight Adjustment<br/>get_query_aware_weights()"]
    WEIGHT_ADJ --> RERANK["DocumentReranker<br/>CrossEncoder<br/>book:1.0 | paper:0.95 | web:0.7<br/>alpha:0.5 | batch:32"]
    
    RERANK --> TOP_K["Top-K Selection<br/>max_docs:50 | max_chars:100k"]
    
    %% REASONING PHASE
    TOP_K --> REASON_CTRL["ReasoningController<br/>max_retries:3"]
    REASON_CTRL --> REASON_MODE{{"Reasoning<br/>Mode?"}}
    
    REASON_MODE -->|Structured| REASON_STRUCT["ReasoningEngine<br/>LLM:ChatOllama | temp:0.7<br/>Output:JSON<br/>claims | evidence | comparisons"]
    REASON_MODE -->|Prose| REASON_PROSE["ReasoningEngine<br/>LLM:ChatOllama | temp:0.7<br/>Output:Prose | natural language"]
    
    REASON_STRUCT --> VAL_STRUCT["StructuredReasoningValidator<br/>schema | sources | evidence quality"]
    REASON_PROSE --> VAL_PROSE["ReasoningOutputValidator<br/>citations | sections | source counts"]
    
    VAL_STRUCT --> VAL_CHECK_R{{"Valid?"}}
    VAL_PROSE --> VAL_CHECK_R
    
    VAL_CHECK_R -->|Invalid| RETRY_R{{"Retry?"}}
    RETRY_R -->|"< max"| FEEDBACK_R["Generate Feedback<br/>error analysis | suggestions"]
    FEEDBACK_R --> REASON_CTRL
    RETRY_R -->|">= max"| REASON_FAIL["⚠️ Reasoning Failed<br/>log | use best attempt"]
    VAL_CHECK_R -->|Valid| REASON_OK["✅ Reasoning Complete<br/>reasoning_output | source_index"]
    REASON_FAIL --> REASON_OK
    
    %% SYNTHESIS PHASE
    REASON_OK --> SYNTH_CTRL["SynthesisController<br/>max_retries:3"]
    SYNTH_CTRL --> SOURCE_VAL["Source Validation<br/>validate_sources_before_synthesis()"]
    SOURCE_VAL --> SOURCE_VAL_CHK{{"Valid?"}}
    
    SOURCE_VAL_CHK -->|No| SOURCE_FIX["Attempt Fix<br/>remove invalid refs"]
    SOURCE_VAL_CHK -->|Yes| SYNTH_MODE
    SOURCE_FIX --> SYNTH_MODE
    
    SYNTH_MODE{{"Synthesis<br/>Strategy?"}}
    SYNTH_MODE -->|"2-stage JSON"| SYNTH_2S_JSON["SynthesisEngine<br/>from JSON<br/>Stage 1:structure | Stage 2:prose<br/>temp:0.3"]
    SYNTH_MODE -->|"2-stage prose"| SYNTH_2S_PROSE["SynthesisEngine<br/>from prose<br/>Stage 1:extract | Stage 2:enhance<br/>temp:0.3"]
    SYNTH_MODE -->|single| SYNTH_SINGLE["SynthesisEngine<br/>direct generation<br/>temp:0.3"]
    
    SYNTH_2S_JSON --> SYNTH_FORMAT
    SYNTH_2S_PROSE --> SYNTH_FORMAT
    SYNTH_SINGLE --> SYNTH_FORMAT
    
    SYNTH_FORMAT["Format Application<br/>style | document type"]
    SYNTH_FORMAT --> VAL_SYNTH["SynthesisOutputValidator<br/>sections | citations<br/>sources | completeness"]
    
    VAL_SYNTH --> VAL_CHECK_S{{"Valid?"}}
    VAL_CHECK_S -->|Invalid| RETRY_S{{"Retry?"}}
    RETRY_S -->|"< max"| FEEDBACK_S["Generate Feedback<br/>validation issues"]
    FEEDBACK_S --> SYNTH_CTRL
    
    RETRY_S -->|">= max"| SYNTH_FAIL_CHK{{"hard_reject?"}}
    SYNTH_FAIL_CHK -->|Yes| SYNTH_REJECT["❌ Synthesis Rejected<br/>return error"]
    SYNTH_FAIL_CHK -->|No| SYNTH_WARN["⚠️ Use Best Attempt<br/>log validation issues"]
    
    VAL_CHECK_S -->|Valid| SYNTH_OK["✅ Synthesis Complete<br/>final_report generated"]
    SYNTH_WARN --> SYNTH_OK
    
    %% POST-PROCESSING
    SYNTH_OK --> CLEANUP["Post-Synthesis Cleanup<br/>format | citations | metadata"]
    CLEANUP --> SAVE_CHECK{{"save_report?"}}
    
    SAVE_CHECK -->|Yes| FORMAT_SEL{{"Format?"}}
    FORMAT_SEL -->|md| SAVE_MD["Save Markdown"]
    FORMAT_SEL -->|html| SAVE_HTML["Save HTML"]
    FORMAT_SEL -->|docx| SAVE_DOCX["Save DOCX"]
    FORMAT_SEL -->|pdf| SAVE_PDF["Save PDF"]
    
    SAVE_MD --> REPORT_MGR
    SAVE_HTML --> REPORT_MGR
    SAVE_DOCX --> REPORT_MGR
    SAVE_PDF --> REPORT_MGR
    
    REPORT_MGR["ReportManager<br/>reports/YYYY-MM-DD/HH-MM-SS/<br/>metadata JSON"]
    
    SAVE_CHECK -->|No| VIZ_CHECK
    REPORT_MGR --> VIZ_CHECK
    
    VIZ_CHECK{{"visualize?"}}
    VIZ_CHECK -->|Yes| VIZ_GEN["Generate Visualization<br/>network | citations | sources"]
    VIZ_CHECK -->|No| TTS_CHECK
    VIZ_GEN --> TTS_CHECK
    
    TTS_CHECK{{"TTS?"}}
    TTS_CHECK -->|Yes| TTS_PROCESS["TTS Processing<br/>LatexMath | Citations | Insights"]
    TTS_CHECK -->|No| METRICS
    TTS_PROCESS --> METRICS
    
    METRICS["Log Metrics<br/>time:10-30s | sources<br/>tokens | cache hits | retries"]
    METRICS --> DONE["✅ Process Complete<br/>report | metadata | viz | audio"]
    
    SYNTH_REJECT --> ERROR_HANDLER["🔴 Error Handler<br/>log error"]
    ERROR_HANDLER --> END_ERROR["❌ Process Failed"]
    
    %% STYLING
    classDef entry fill:#1A365D,stroke:#0F2942,stroke-width:4px,color:#FFF,font-weight:bold
    classDef orchestrator fill:#2C5282,stroke:#1A365D,stroke-width:3px,color:#FFF,font-weight:bold
    classDef processor fill:#4A90C7,stroke:#2C5282,stroke-width:2px,color:#FFF
    classDef llm fill:#5B9FCF,stroke:#4A90C7,stroke-width:2px,color:#FFF
    classDef storage fill:#64748B,stroke:#475569,stroke-width:2px,color:#FFF
    classDef api fill:#7CB8E0,stroke:#5B9FCF,stroke-width:2px,color:#1A202C
    classDef validator fill:#2F855A,stroke:#22543D,stroke-width:2px,color:#FFF
    classDef decision fill:#718096,stroke:#4A5568,stroke-width:2px,color:#FFF
    classDef success fill:#38A169,stroke:#2F855A,stroke-width:3px,color:#FFF,font-weight:bold
    classDef warning fill:#DD6B20,stroke:#C05621,stroke-width:3px,color:#FFF,font-weight:bold
    classDef error fill:#E53E3E,stroke:#C53030,stroke-width:3px,color:#FFF,font-weight:bold
    classDef io fill:#E2E8F0,stroke:#A0AEC0,stroke-width:2px,color:#2D3748
    classDef cache fill:#90CDF4,stroke:#63B3ED,stroke-width:2px,color:#1A202C
    
    class START entry
    class WF,REASON_CTRL,SYNTH_CTRL orchestrator
    class DOC_SCAN,DOC_CLASSIFY,DOC_CHUNK,COMBINE,WEIGHT_ADJ,RERANK,TOP_K,CLEANUP processor
    class REASON_STRUCT,REASON_PROSE,SYNTH_2S_JSON,SYNTH_2S_PROSE,SYNTH_SINGLE,ARXIV_OPT,WEB_OPT llm
    class DOC_DUAL,DOC_SAVE,ARXIV_EMBED,WEB_EMBED,SEARCH storage
    class ARXIV_API,TAVILY,ARXIV_PDF,PDF_MGR,WEB_FILTER api
    class VAL_STRUCT,VAL_PROSE,VAL_SYNTH,SOURCE_VAL validator
    class ROUTE,DOC_CHECK,DOC_CACHE_CHK,ARXIV_PDF_CHK,WEB_PDF_CHECK,REASON_MODE,VAL_CHECK_R,RETRY_R,SOURCE_VAL_CHK,SYNTH_MODE,VAL_CHECK_S,RETRY_S,SYNTH_FAIL_CHK,SAVE_CHECK,FORMAT_SEL,VIZ_CHECK,TTS_CHECK decision
    class REASON_OK,SYNTH_OK,DONE success
    class REASON_FAIL,SYNTH_WARN warning
    class SYNTH_REJECT,ERROR_HANDLER,END_ERROR error
    class UI,HANDLERS,CONFIG,STATE_INIT,SAVE_MD,SAVE_HTML,SAVE_DOCX,SAVE_PDF,REPORT_MGR,VIZ_GEN,TTS_PROCESS,METRICS io
    class DOC_CACHE_LOAD,DOC_CACHE_SAVE,DOC_EMBED,DOC_LOAD cache
```
---

## Key Class Relationships

### Core Orchestrators
- **ResearchSynthesizer**: Main entry point, coordinates all components
- **WorkflowManager**: Executes research pipeline with parallel retrieval
- **ReasoningController**: Manages reasoning phase with retry logic
- **SynthesisController/Engine**: Generates final reports (single or two-stage)

### Retrieval & Indexing
- **IndexingManager**: Coordinates document loading and embedding
- **DocumentIndexer**: Scans, chunks, and classifies documents
- **RetrievalManager**: Manages ArXiv and web searches
- **VectorStoreManager** + **DualVectorStoreManager**: FAISS indices (books/papers)
- **EmbeddingsCacheManager**: SQLite cache with 60-80% hit rate

### Source Management
- **SourceTypeClassifier**: Classifies docs as book/paper/web
- **PDFManager**: Caches and manages PDF downloads
- **WebPDFEmbedder**: Filters and embeds web PDFs
- **AsyncPDFTitleEnhancer**: Enhances PDF metadata

### Search & Ranking
- **DocumentReranker**: Cross-encoder scoring and weighting
- **QueryOptimizer**: Refines queries (optional LLM)

### Validation & Output
- **ReasoningOutputValidator**: Validates prose reasoning
- **SourceValidator**: Checks citation integrity
- **SynthesisOutputValidator**: Validates report structure
- **ReportManager**: Saves in markdown/PDF/JSON formats

### Utilities
- **LoggerConfig**: Centralized logging
- **MinimalCitationProcessor**: Citation formatting
- **ReasoningCapabilityProbe**: Model capability detection
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
- ~16GB RAM minimum (more for larger models)
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
- **Piper TTS**: Text-to-speech playback of the reports

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
