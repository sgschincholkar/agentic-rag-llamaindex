# Agentic RAG with LlamaIndex

An agent that answers questions about the McKinsey "State of AI" (March 2025) report. It combines three sources:

- **The report (PDF)**: a router picks between a summary index (high-level questions) and a vector index (specific facts and stats).
- **ArXiv**: recent academic papers.
- **Brave Search** (optional): current web results.

Built with LlamaIndex `AgentWorkflow` and OpenAI.

## Project structure

```
config.py        env vars, paths, global LLM + embedding settings
ingestion.py     load PDF, chunk (SentenceSplitter), build SummaryIndex + VectorStoreIndex
tools.py         router query engine, ArXiv and Brave tools
agent.py         AgentWorkflow and system prompt
main.py          entry point
data/            put state.pdf here (not committed)
```

## Prerequisites

- Python 3.10+
- An OpenAI API key
- (Optional) a Brave Search API key: https://brave.com/search/api/
- The report PDF, saved as `state.pdf`

## Setup

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your keys
cp .env.example .env
# edit .env and fill in OPENAI_API_KEY (and BRAVE_API_KEY if you have one)

# 4. Add the PDF
cp /path/to/state.pdf data/state.pdf
```

`.env` is gitignored. Never commit it.

## Run the agent

From the project folder, with the virtual environment active:

```bash
python main.py                          # runs a built-in sample question
python main.py "Who is Lareina Yee?"    # runs your own question
```

### In VS Code

1. Cmd+Shift+P (Ctrl+Shift+P on Windows/Linux), then **Python: Select Interpreter**, then choose `.venv`.
2. Open `main.py`.
3. Click the **Run** button at the top right.

### What happens on a run

1. The PDF is loaded and split into chunks.
2. Both indexes are built (this calls the OpenAI embeddings API, so the first step costs a few API calls on every run).
3. The agent decides which tools to use, and the router logs which index it picked.
4. The answer is printed.

## Configuration

Optional overrides in `.env`:

| Variable | Default | Purpose |
|---|---|---|
| `LLM_MODEL` | `gpt-3.5-turbo` | OpenAI chat model |
| `EMBED_MODEL` | `text-embedding-ada-002` | OpenAI embedding model |
| `CHUNK_SIZE` | `1024` | Chunk size for the splitter |
| `PDF_PATH` | `data/state.pdf` | Path to the PDF, relative to the project root |

If `BRAVE_API_KEY` is empty, web search is disabled and the agent uses the report and ArXiv only.

## Troubleshooting

| Error | Fix |
|---|---|
| `OPENAI_API_KEY is not set` | Fill in `.env` and check the variable name has no typos, quotes, or spaces around `=`. |
| `PDF not found at ...` | Put the file at `data/state.pdf`. |
| `ModuleNotFoundError` | Activate `.venv` and rerun `pip install -r requirements.txt`. In VS Code, select the `.venv` interpreter. |
| `BRAVE_API_KEY not set: web search tool disabled.` | Not an error. Add a Brave key to enable web search. |

## Notes

- The PDF is gitignored because the report is copyrighted. Each user supplies their own copy.
- The index is rebuilt on every run and not saved to disk.
