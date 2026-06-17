# MultiRAG-Doc

Harness-guided Agentic RAG for scientific papers, automatic mathematical
modeling, and optimization-code generation.

This repository is a research prototype. Its main focus is not only paper QA,
but **evidence-grounded LLM mathematical modeling**: the system retrieves
modeling evidence from papers, plans the formulation, constrains generation with
a harness, verifies the result, and can emit a PlatEMO/MATLAB problem skeleton.

## Highlights

- **Harness-guided mathematical modeling**: build structured optimization drafts
  through component selection, symbol planning, formula rendering, verification,
  and quality scoring.
- **Lightweight multi-stage modeling workflow**: planning, evidence retrieval,
  harness drafting, formula generation, critic/verifier, polishing, and optional
  code generation.
- **Agentic RAG over scientific papers**: a LangGraph evidence-search loop with
  controlled tool calls, evidence budget, termination logic, and citation checks.
- **Multimodal paper ingestion**: parse PDFs into text, figures, tables,
  equations, captions, and modeling regions.
- **Optimization-code generation**: generate PlatEMO-style MATLAB problem class
  skeletons from structured model drafts.

## Core Idea

Directly asking an LLM to write a complete mathematical formulation is flexible
but unstable: it may invent undefined symbols, over-generate constraints, copy a
paper template too closely, or ignore components that the user explicitly
excluded.

MultiRAG-Doc uses a different route:

```text
User problem
  -> domain skill selection
  -> paper evidence retrieval
  -> modeling plan
  -> harness component selection
  -> symbol and operator planning
  -> formula rendering / LLM completion
  -> verifier and quality rubric
  -> structured model + optional PlatEMO code
```

The LLM still makes modeling decisions, but the harness keeps the formulation
inside a controlled modeling boundary.

## System Architecture

```text
                +--------------------+
PDF papers ---> | Multimodal Ingest  |
                | text/figure/table  |
                | equation/modeling  |
                +---------+----------+
                          |
                          v
                +--------------------+
                | FAISS Indexes      |
                | text/image/caption |
                +---------+----------+
                          |
          +---------------+----------------+
          |                                |
          v                                v
+--------------------+          +-------------------------+
| Agentic Paper QA   |          | Harness Modeling Stack  |
| standard/decompose |          | plan -> harness ->      |
| LangGraph agent    |          | verify -> codegen       |
+--------------------+          +-------------------------+
```

## What Is Implemented

| Area | Implementation |
|---|---|
| PDF ingestion | Docling/PyMuPDF parsing, multimodal chunking, figure/table/equation extraction |
| Retrieval | FAISS vector stores, text/image/caption retrieval, fusion, optional reranking |
| RAG generation | grounded prompt building, citation parsing, citation validation, guardrails |
| Agentic RAG | LangGraph loop, controlled `search_evidence` tool, evidence store, termination policy |
| Modeling | domain skills, model-region retrieval, harness draft, formula renderer, verifier, quality rubric |
| Code generation | deterministic PlatEMO/MATLAB problem skeleton generation |
| Interface | CLI and FastAPI + single-page Web UI |
| Evaluation | retrieval metrics and HHC modeling regression evaluation |

## Query Modes

| Mode | Description |
|---|---|
| `standard` | retrieve evidence once, then optionally generate a grounded answer |
| `decompose` | decompose a complex question into subqueries, retrieve in parallel, merge evidence |
| `agent` | use a LangGraph loop to iteratively search evidence, compact/select evidence, and answer |

## Modeling Modes

| Mode | Description |
|---|---|
| harness draft | fast deterministic component and symbol planning |
| harness formulas | render a controlled model without full free-form LLM generation |
| full model generation | retrieve evidence, plan, generate, verify, revise, and polish |
| PlatEMO codegen | compile a structured model draft into MATLAB problem skeleton code |

## Repository Layout

```text
MultiRAG-Doc/
├── src/
│   ├── agent/          # LangGraph evidence-search agent
│   ├── evaluation/     # citation, guardrails, retrieval/HHC evaluation
│   ├── generator/      # LLM client, prompts, answer formatting
│   ├── index/          # FAISS and metadata stores
│   ├── ingestion/      # PDF parsing, chunking, embedding, captioning
│   ├── modeling/       # harness, formula renderer, verifier, codegen
│   ├── pipeline/       # ingest/query orchestration
│   ├── query/          # standard, decompose, agent query modes
│   └── retrieval/      # text/image retrieval, fusion, reranking
├── web/                # FastAPI backend and browser UI
├── database/           # local runtime data layout, mostly ignored by Git
├── config.yml          # default API-based runtime config
└── environment_gpu.yml # conda environment
```

## Quick Start

Create the environment:

```bash
conda env create -f environment_gpu.yml
conda activate multirag-doc-gpu
```

Configure API keys:

```bash
cp .env.example .env
```

Then edit `.env`:

```text
LLM_API_KEY=your_llm_key_here
EMBEDDING_API_KEY=your_embedding_key_here
OPENAI_API_KEY=your_openai_compatible_key_here
```

Put PDFs into `database/pdf/`, then build the index:

```bash
python -m src.cli ingest-all --pdf-dir database/pdf --multimodal --staged
```

Start the Web UI:

```bash
uvicorn web.main:app --reload --port 8000
```

Open `http://127.0.0.1:8000`.

## CLI Examples

Standard grounded QA:

```bash
python -m src.cli query \
  --question "What optimization model is proposed in this paper?" \
  --mode standard \
  --generate
```

Agentic evidence search:

```bash
python -m src.cli query \
  --question "Explain the objective function and major constraints." \
  --mode agent \
  --generate
```

Harness-only modeling:

```bash
python -m src.cli generate-model \
  --problem "Build a compact home health care routing and scheduling model that minimizes travel time and patient waiting time. Do not include outsourcing or VIP classes." \
  --harness-formulas
```

Modeling plus PlatEMO code:

```bash
python -m src.cli generate-model \
  --problem "Build a home health care routing and scheduling model with assignment, routing, and time windows." \
  --harness-formulas \
  --platemo-code \
  --no-platemo-write
```

If you want to write directly into a local PlatEMO checkout, pass:

```bash
--platemo-root /path/to/PlatEMO --platemo-code
```

## Web UI

The Web UI includes:

- paper ingestion
- standard RAG query
- decomposed query
- agent query
- automatic mathematical modeling
- harness draft inspection
- PlatEMO code generation
- HHC modeling evaluation

## Configuration

The default `config.yml` uses API-based embedding and generation settings.
You can switch to local or hash/demo modes by editing `embedder.mode`,
`embedder.image_mode`, and the model endpoint fields in `config.yml`.

Important environment variables:

- `LLM_API_KEY`: caption/modeling helper calls
- `EMBEDDING_API_KEY`: text/image embedding calls
- `OPENAI_API_KEY`: answer generation and modeling generation

## Runtime Data

The GitHub release does not include local PDFs, FAISS indexes, extracted
figures, model weights, `.env`, logs, or generated evaluation outputs.

The `database/` folder only preserves the expected layout and a small curated
HHC modeling testset:

```text
database/
├── pdf/
├── chunks/
├── index/
│   └── figures/
├── staging/
├── model_cards/
├── eval_results/
└── testset/
    └── hhc_modeling_testset.json
```

## Limitations

- This is a research prototype, not a hardened production service.
- Running the full multimodal pipeline may require GPU memory or external model
  APIs, depending on your config.
- The PlatEMO generator currently targets an HHC-style optimization skeleton and
  should be manually checked before experiments.
- The multi-agent modeling workflow is implemented as a lightweight staged
  pipeline, not as a fully autonomous swarm of independent agents.

## Suggested Citation / Project Pitch

MultiRAG-Doc is a harness-guided Agentic RAG system for evidence-grounded
automatic mathematical modeling from scientific papers. It combines multimodal
paper retrieval, controlled agentic evidence search, structured modeling
harnesses, verification, and optimization-code generation.
