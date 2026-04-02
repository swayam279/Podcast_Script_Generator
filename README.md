# ASL Podcast Script Generator

An AI-powered podcast script producer that conducts a **conversational interview** to learn about your podcast, then generates a professional dialogue-only script — segment by segment — with human-in-the-loop review at every step.

## How It Works

```
 Describe your        Review & approve                       Review each
    podcast           the generated plan                   generated segment
       ↓                    ↓                                    ↓
  [Chat Phase] ─────► [Plan Review] ─────► [Segment 1 ⇄ Review] ⇄ [Segment 2 ⇄ Review] ...
                                                                                          ↓
                                                                                    [Final Script]
```

The agent asks you questions about your podcast concept, host/guest personas, and preferred segments. Once it has enough information it drafts a plan for your review. When you approve, it generates the script one segment at a time, stopping after each so you can **accept** it or request a **rewrite with specific feedback**. Progress is checkpointed in SQLite so you can quit at any time and resume later.

## Features

- **Conversational requirements gathering** — an agentic LLM interviews you to collect podcast name, platform, topic, duration, and detailed host/guest personas (at least 4 traits each).
- **Structured plan review** — the parsed plan is presented for approval before any script is generated. You can send natural-language feedback to revise it.
- **Segment-by-segment generation** — one segment at a time, with automatic continuity between segments for consistent tone and flow.
- **Human-in-the-loop review** — the graph pauses after each segment so you can accept or request a rewrite with specific feedback.
- **Checkpointed state** — SQLite checkpointer persists graph state; `quit` at any time and resume with the same thread ID.
- **Two interfaces**
  - **Streamlit web UI** — session sidebar, live chat bubbles, expandable plan/segment views.
  - **CLI** — terminal-only mode with minimal dependencies.
- **Strict script formatting** — dialogue-only output (`HOST:` / `GUEST:` lines). No stage directions, parentheses, asterisks, or bracketed actions.

## Tech Stack

| Layer | Technology |
|---|---|
| **Agent framework** | [LangGraph](https://github.com/langchain-ai/langgraph) (state graph with human-in-the-loop interrupts) |
| **LLM** | [ChatNVIDIA](https://python.langchain.com/docs/integrations/chat/nvidia_ai_endpoints/) — `kimi-k2-instruct-0905` |
| **Language** | Python 3.12+ |
| **Web UI** | [Streamlit](https://streamlit.io/) |
| **State persistence** | SQLite (`langgraph-checkpoint-sqlite`) |
| **Structured output** | Pydantic v2 models |
| **Package manager** | [uv](https://github.com/astral-sh/uv) |

## Installation

### Prerequisites

- **Python 3.12+**
- **uv** package manager (install: `pip install uv`)
- **NVIDIA API key** — get one at [build.nvidia.com](https://build.nvidia.com/)

### Setup

```bash
git clone https://github.com/your-username/ASL-Podcast-Script-Generator.git
cd ASL-Podcast-Script-Generator

uv sync
```

Create a `.env` file in the project root with your API keys:

```env
NVIDIA_API_KEY=nvapi-xxxxxxxxxxxxxxxx
```

## Usage

### Streamlit Web UI (Recommended)

```bash
streamlit run src/frontend.py
```

This launches a browser with a sidebar session manager, live chat interface, and interactive plan/segment review.

### CLI Mode

```bash
python src/backend.py
```

**Example flow:**
```
YOU: I want to make a podcast about machine learning for beginners
AGENT: That sounds like a great concept! Let's start with a few questions...
...
PODCAST PLAN FOR YOUR REVIEW
============================================================
Podcast Name : ML for Humans
Platform     : Spotify
Topic        : Machine learning for beginners
Est. Duration: 30 minutes
...
YOUR DECISION: (press Enter to confirm)
```

## Project Structure

```
ASL-Podcast-Script-Generator/
├── src/
│   ├── backend.py          # LangGraph state machine, all agent/script logic
│   └── frontend.py         # Streamlit web interface
|   └── script.db           # SQLite checkpoint store (created on first run)         
├── .env                    # API keys (copy from .env.example)
├── pyproject.toml          # Project metadata and dependencies
└── main.py                 # Entry point placeholder
```

## Architecture

The LangGraph state machine consists of **8 nodes**:

### Phase 1 — Requirements Gathering

| Node | Purpose |
|---|---|
| `agent_chat` | Converses with the user using a system prompt that guides it to collect all required fields. Calls the `finalize_requirements` tool when ready. |
| `parse_input` | Invokes the LLM with structured output to extract a `PodcastDetails` Pydantic model from the conversation history. |
| `confirm_requirements` | Passthrough node that sets `requirements_confirmed = True`. The graph **pauses here** for plan review. |

### Phase 2 — Script Generation

| Node | Purpose |
|---|---|
| `generate_segment` | Generates a single segment (or rewrites the last segment when human feedback exists). Maintains continuity with previously generated segments. |
| `human_interrupt` | Passthrough node. The graph **pauses here** after every segment for human review. |
| `check_completion` | Advances `current_segment_index`. Sets `is_complete = True` when all segments are done. |

### Graph topology

```
  START
    │
    ▼
 agent_chat ◄──────────────────── agent_chat (revision loop)
    │  │                              ▲
    │  ▼ (tool called)                │
    │ tools                           │
    │  │                              │
    ▼  ▼                              │
 parse_input                          │
    │                                 │
    ▼                                 │
 confirm_requirements ──(reject)─────┘
    │ (confirmed)
    ▼
 generate_segment ──► human_interrupt ──► (feedback? → generate_segment)
    │                                        │ (accepted)
    ▼                                        ▼
 check_completion ──► generate_segment    check_completion
    │ (complete)
    ▼
   END
```

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `NVIDIA_API_KEY` | Yes | Your NVIDIA NIM API key from [build.nvidia.com](https://build.nvidia.com/) |

## State Persistence

The graph uses `SqliteSaver` with a SQLite database (`script.db`). Each conversation gets its own `thread_id`, enabling:

- Pause and resume script generation mid-session.
- Multiple concurrent sessions (in the Streamlit UI).
- Recovery after server restarts.
