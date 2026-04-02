# backend.py

import sqlite3
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

load_dotenv()

# ─────────────────────────────────────────────
# LLM Setup
# ─────────────────────────────────────────────

model = ChatNVIDIA(model="moonshotai/kimi-k2-thinking")


# ─────────────────────────────────────────────
# Pydantic Models for Structured Output
# ─────────────────────────────────────────────


class Segment(BaseModel):
    name: str = Field(description="The name/identifier of this segment (e.g., 'welcome', 'intro', 'discussion', 'outro')")
    description: str = Field(description="Detailed guidelines for writing this segment, including tone, content requirements, and approximate length")
    estimated_minutes: int = Field(description="Estimated duration of this segment in minutes. All segment durations should sum to the total podcast duration.")


class PodcastDetails(BaseModel):
    topic: str = Field(description="The main topic of the podcast episode")
    host_persona: dict = Field(description=("Details about the host in JSON format that would help build a unique persona with at least 4 traits (e.g. vocabulary level, humor style, catchphrases, energy level)"))
    guest_persona: dict = Field(description=("Details about the guest in JSON format that would help build a unique persona with at least 4 traits (e.g. expertise, speaking style, personality quirks, communication approach)"))
    podcast_name: str = Field(description="Name or Title of the podcast")
    platform_name: str = Field(description="Name or Title of the platform hosting the podcast")
    estimated_duration_minutes: int = Field(description="The estimated total duration of the podcast episode in minutes")
    segments: list[Segment] = Field(
        description=(
            "A list of segments that make up the podcast structure, in order. Each segment should have a name, description, and estimated duration in minutes. "
            "The sum of all segment durations should equal the total estimated podcast duration. "
            "The following segments should be minimally included: welcome (host opens the show), intro (guest introduction), discussion (main topic exploration), outro (wrap-up and sign-off). "
            "You can add, remove, or rename segments based on what makes sense for this specific podcast. Try to include a mix of natural and scripted segments with more than the minimally required segments."
        )
    )


detail_model = model.with_structured_output(PodcastDetails)


# ─────────────────────────────────────────────
# Agent Tool: Finalize Requirements
# ─────────────────────────────────────────────


class FinalizeRequirements(BaseModel):
    """Call this tool ONLY when you have gathered ALL the required information from the user."""

    summary: str = Field(description=("A comprehensive summary of everything gathered from the conversation, including: podcast name, platform name, episode topic, estimated duration, host persona details (at least 4 traits), guest persona details (at least 4 traits), and any specific segment preferences."))


@tool("finalize_requirements", args_schema=FinalizeRequirements)
def finalize_requirements(summary: str) -> str:
    """Signal that enough information has been gathered to produce the podcast script."""
    return f"Requirements finalized. Summary: {summary}"


agent_tools = [finalize_requirements]
agent_model = model.bind_tools(agent_tools)

AGENT_SYSTEM_PROMPT = """You are a podcast production assistant. Your job is to have a conversation \
                        with the user to gather enough information to produce a professional podcast script.

                        You need to collect ALL of the following before calling finalize_requirements:

                        1. The podcast name
                        2. The hosting platform name
                        3. The episode topic (with enough detail to write about)
                        4. Estimated podcast duration (in minutes — e.g., 15 min, 30 min, 60 min). \
                           This is important because it determines how many segments to create and how long each one should be.
                        5. Host persona — at least 4 traits. Ask about: vocabulary level, humor style, catchphrases, energy level, speaking patterns
                        6. Guest persona — at least 4 traits. Ask about: area of expertise, speaking style, personality quirks, communication approach
                        7. (Optional) Any specific segment preferences or special requests

                        Guidelines:
                        - Be conversational and friendly.
                        - Ask follow-up questions if the user gives vague answers (e.g., "a funny host" → ask what KIND of funny).
                        - When asking about duration, give examples of what different durations look like \
                          (e.g., "15 min is a quick chat, 30 min allows deeper discussion, 60 min is a full deep-dive").
                        - Summarize what you've gathered before calling finalize_requirements so the user can confirm.
                        - Do NOT call finalize_requirements until you have sufficient detail for ALL required fields.
                        - If the user provides everything in one message, you can summarize and call finalize_requirements immediately.

                        IMPORTANT — When to finalize:
                        - If the user explicitly says "that's all", "go ahead", "finalize", or similar, call finalize_requirements \
                        IMMEDIATELY. Use the information you already have and fill in reasonable creative defaults for anything missing. \
                        Do NOT ask more questions after the user signals they are done.
                        - Always include a brief text response acknowledging you are finalizing ALONG WITH the tool call. \
                        Example: "Great, I have everything I need! Let me put this together for you." + call finalize_requirements.
                        """


# ─────────────────────────────────────────────
# Shared State
# ─────────────────────────────────────────────


class FullState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

    podcast_name: str
    platform_name: str
    topic: str
    estimated_duration_minutes: int
    host_persona: dict
    guest_persona: dict
    segment_structure: list[dict]

    requirements_confirmed: bool

    current_segment_index: int
    segments: list[dict]
    is_complete: bool
    human_feedback: str


# ─────────────────────────────────────────────
# Common Prompt Rules
# ─────────────────────────────────────────────

SCRIPT_FORMAT_RULES = """Rules:
                    • Format every line as HOST: <dialogue> or GUEST: <dialogue>.
                    • Each HOST: or GUEST: line MUST start on its own new line.
                    • The host's language, vocabulary, and energy MUST reflect their persona.
                    • The guest's language, vocabulary, and energy MUST reflect their persona.
                    • The dialogue must sound natural — include filler words, reactions, interruptions where appropriate.
                    • Do NOT include stage directions, action descriptions, or parentheticals.
                    • Do NOT wrap any text in asterisks (*), brackets ([]), or parentheses for actions/sounds/emotions.
                    • Do NOT write things like *laughs*, [sighs], (pauses), *claps*, [field audio], etc.
                    • Express emotions and actions ONLY through the dialogue words themselves.
                    • Output ONLY the script dialogue lines.
                    • Do NOT include any preamble, introduction, summary of changes, or commentary.
                    • Your response MUST start with HOST: or GUEST: and end with the final line of dialogue."""


# ─────────────────────────────────────────────
# Script Generation Nodes
# ─────────────────────────────────────────────


def generate_segment_node(state: FullState) -> FullState:
    """Generate the next segment, or regenerate the last one if feedback exists."""

    segment_structure = state["segment_structure"]
    current_index = state["current_segment_index"]
    topic = state["topic"]
    podcast_name = state["podcast_name"]
    platform_name = state["platform_name"]
    estimated_duration = state["estimated_duration_minutes"]
    host = state["host_persona"]
    guest = state["guest_persona"]
    segments = list(state.get("segments", []))
    feedback = state.get("human_feedback", "").strip()

    current_segment = segment_structure[current_index]
    segment_name = current_segment["name"]
    segment_guideline = current_segment["description"]
    segment_minutes = current_segment.get("estimated_minutes", "N/A")

    # ── Case 1: Re-generate the last segment with human feedback ──
    if feedback and segments:
        last = segments[-1]
        prompt = f"""You are a podcast script writer. Rewrite the following \
                "{last["type"]}" segment based on the reviewer's feedback.

                --- ORIGINAL SEGMENT ---
                {last["content"]}

                --- REVIEWER FEEDBACK ---
                {feedback}

                --- CONTEXT ---
                Podcast Name : {podcast_name}
                Platform Name : {platform_name}
                Topic : {topic}
                Total Podcast Duration : {estimated_duration} minutes
                This Segment Duration  : {segment_minutes} minutes
                Host : {host}
                Guest : {guest}

                {SCRIPT_FORMAT_RULES}

                Additional rewrite rules:
                • Apply the feedback precisely; do not change anything the reviewer didn't mention.
                • Keep the same overall structure and flow of the segment.
                • Keep the dialogue length appropriate for approximately {segment_minutes} minutes of spoken audio.
                • Only provide the updated script without any additional text."""

        response = model.invoke(prompt)
        content = response.content

        segments[-1] = {
            "type": last["type"],
            "content": content,
        }
        return {"segments": segments, "human_feedback": ""}

    # ── Case 2: Generate a brand-new segment ──
    previous_context = ""
    if segments:
        prev_text = "\n\n".join(f"=== {s['type'].upper()} ===\n{s['content']}" for s in segments)
        previous_context = f"Here are the segments generated so far (for continuity):\n{prev_text}\n\n"

    prompt = f"""You are a podcast script writer. Write the \
            "{segment_name}" segment for a podcast episode.

            Podcast Name : {podcast_name}
            Platform Name : {platform_name}
            Topic : {topic}
            Total Podcast Duration : {estimated_duration} minutes
            This Segment Duration  : {segment_minutes} minutes
            Host persona : {host}
            Guest persona : {guest}

            Segment guidelines: {segment_guideline}

            IMPORTANT: This segment should contain enough dialogue to fill approximately \
            {segment_minutes} minutes of spoken audio. As a rough guide, 1 minute of natural \
            conversation is about 120-150 words. So aim for roughly {int(segment_minutes) * 135 if isinstance(segment_minutes, int) else 'an appropriate number of'} words.

            {previous_context}
            {SCRIPT_FORMAT_RULES}"""

    response = model.invoke(prompt)
    content = response.content

    segments.append(
        {
            "type": segment_name,
            "content": content,
        }
    )
    return {"segments": segments, "human_feedback": ""}


def human_interrupt_node(state: FullState) -> FullState:
    """Passthrough node. Graph pauses BEFORE this node for human review."""
    return {}


def check_completion_node(state: FullState) -> FullState:
    """Advance current_segment_index or flag completion."""
    current_index = state["current_segment_index"]
    segment_structure = state["segment_structure"]

    if current_index >= len(segment_structure) - 1:
        return {"is_complete": True}

    return {
        "current_segment_index": current_index + 1,
        "is_complete": False,
    }


def after_human_review(state: FullState) -> Literal["generate_segment", "check_completion"]:
    if state.get("human_feedback", "").strip():
        return "generate_segment"
    return "check_completion"


def should_continue(state: FullState) -> Literal["generate_segment", "end"]:
    if state.get("is_complete", False):
        return "end"
    return "generate_segment"


# ─────────────────────────────────────────────
# Agent Nodes
# ─────────────────────────────────────────────


def agent_chat_node(state: FullState) -> FullState:
    """Run the agent conversation to gather podcast requirements."""
    messages = state["messages"]

    has_system = any(isinstance(m, SystemMessage) for m in messages)
    if not has_system:
        messages = [SystemMessage(content=AGENT_SYSTEM_PROMPT)] + list(messages)

    response = agent_model.invoke(messages)
    return {"messages": [response]}


def should_parse_or_continue_chat(state: FullState) -> Literal["tools", "end"]:
    last_message = state["messages"][-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        for tc in last_message.tool_calls:
            if tc["name"] == "finalize_requirements":
                return "tools"
    return "end"


def parse_input_node(state: FullState) -> FullState:
    """Extract structured podcast details from the agent conversation."""
    conversation = "\n".join(f"{msg.type}: {msg.content}" for msg in state["messages"] if hasattr(msg, "content") and msg.content)

    details = detail_model.invoke(conversation)

    segment_structure = [
        {
            "name": seg.name,
            "description": seg.description,
            "estimated_minutes": seg.estimated_minutes,
        }
        for seg in details.segments
    ]

    return {
        "topic": details.topic,
        "host_persona": details.host_persona,
        "guest_persona": details.guest_persona,
        "podcast_name": details.podcast_name,
        "platform_name": details.platform_name,
        "estimated_duration_minutes": details.estimated_duration_minutes,
        "segment_structure": segment_structure,
        "requirements_confirmed": False,
        "current_segment_index": 0,
        "segments": [],
        "is_complete": False,
        "human_feedback": "",
    }


def confirm_requirements_node(state: FullState) -> FullState:
    """Passthrough — sets confirmed=True when resumed normally."""
    return {"requirements_confirmed": True}


def after_confirmation(state: FullState) -> Literal["agent_chat", "generate_segment"]:
    if state.get("requirements_confirmed", False):
        return "generate_segment"
    return "agent_chat"


# ─────────────────────────────────────────────
# Tool Node
# ─────────────────────────────────────────────

tool_node = ToolNode(agent_tools)


# ─────────────────────────────────────────────
# Build Graph
# ─────────────────────────────────────────────

graph = StateGraph(FullState)

graph.add_node("agent_chat", agent_chat_node)
graph.add_node("tools", tool_node)
graph.add_node("parse_input", parse_input_node)
graph.add_node("confirm_requirements", confirm_requirements_node)
graph.add_node("generate_segment", generate_segment_node)
graph.add_node("human_interrupt", human_interrupt_node)
graph.add_node("check_completion", check_completion_node)

graph.add_edge(START, "agent_chat")

graph.add_conditional_edges(
    "agent_chat",
    should_parse_or_continue_chat,
    {
        "tools": "tools",
        "end": END,
    },
)

graph.add_edge("tools", "parse_input")
graph.add_edge("parse_input", "confirm_requirements")

graph.add_conditional_edges(
    "confirm_requirements",
    after_confirmation,
    {
        "agent_chat": "agent_chat",
        "generate_segment": "generate_segment",
    },
)

graph.add_edge("generate_segment", "human_interrupt")

graph.add_conditional_edges(
    "human_interrupt",
    after_human_review,
    {
        "generate_segment": "generate_segment",
        "check_completion": "check_completion",
    },
)

graph.add_conditional_edges(
    "check_completion",
    should_continue,
    {
        "generate_segment": "generate_segment",
        "end": END,
    },
)

conn = sqlite3.connect("script.db", check_same_thread=False)
memory = SqliteSaver(conn=conn)


# ─────────────────────────────────────────────
# Thread Management Helpers (Persistent Metadata)
# ─────────────────────────────────────────────


def _init_thread_metadata_table():
    """Create the thread_metadata table if it doesn't exist."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS thread_metadata (
            thread_id TEXT PRIMARY KEY,
            display_name TEXT NOT NULL
        )
        """
    )
    conn.commit()


_init_thread_metadata_table()


def rename_thread(thread_id: str, display_name: str):
    """Set or update the display name for a thread."""
    conn.execute(
        "INSERT OR REPLACE INTO thread_metadata (thread_id, display_name) VALUES (?, ?)",
        (thread_id, display_name),
    )
    conn.commit()


def get_thread_name(thread_id: str) -> str | None:
    """Get the display name for a thread, or None if not set."""
    cursor = conn.execute(
        "SELECT display_name FROM thread_metadata WHERE thread_id = ?",
        (thread_id,),
    )
    row = cursor.fetchone()
    return row[0] if row else None


def get_all_thread_metadata() -> dict[str, str]:
    """Get all thread_id → display_name mappings."""
    cursor = conn.execute("SELECT thread_id, display_name FROM thread_metadata")
    return {row[0]: row[1] for row in cursor.fetchall()}


def delete_thread(thread_id: str):
    """Delete all checkpoint data and metadata for a thread."""
    # Discover checkpoint tables dynamically to handle different SqliteSaver versions
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    all_tables = {row[0] for row in cursor.fetchall()}

    for table in ["checkpoints", "checkpoint_writes", "checkpoint_blobs"]:
        if table in all_tables:
            conn.execute(f"DELETE FROM {table} WHERE thread_id = ?", (thread_id,))

    conn.execute("DELETE FROM thread_metadata WHERE thread_id = ?", (thread_id,))
    conn.commit()


# ─────────────────────────────────────────────
# Compile Graph
# ─────────────────────────────────────────────

app = graph.compile(
    interrupt_before=["confirm_requirements", "human_interrupt"],
    checkpointer=memory,
)


# ─────────────────────────────────────────────
# Display Utilities
# ─────────────────────────────────────────────


def print_separator():
    print("\n" + "=" * 60)


def print_state_info(snapshot):
    print_separator()
    print(f"Next node      : {snapshot.next}")
    vals = snapshot.values
    if vals.get("segment_structure"):
        print(f"Segments planned: {len(vals['segment_structure'])}")
        print(f"Segment names   : {[s['name'] for s in vals['segment_structure']]}")
    if vals.get("segments"):
        print(f"Segments done   : {len(vals['segments'])}")
        idx = vals.get("current_segment_index", 0)
        if idx < len(vals.get("segment_structure", [])):
            print(f"Current segment : {vals['segment_structure'][idx]['name']}")
    print(f"Is complete     : {vals.get('is_complete', False)}")
    print_separator()


def print_latest_segment(snapshot):
    segments = snapshot.values.get("segments", [])
    if segments:
        latest = segments[-1]
        print(f"\n--- {latest['type'].upper()} ---\n")
        print(latest["content"])
        print()


def print_podcast_plan(snapshot):
    vals = snapshot.values
    print_separator()
    print("PODCAST PLAN FOR YOUR REVIEW")
    print_separator()
    print(f"Podcast Name : {vals.get('podcast_name', 'N/A')}")
    print(f"Platform     : {vals.get('platform_name', 'N/A')}")
    print(f"Topic        : {vals.get('topic', 'N/A')}")
    print(f"Est. Duration: {vals.get('estimated_duration_minutes', 'N/A')} minutes")
    print("\nHost Persona:")
    for k, v in vals.get("host_persona", {}).items():
        print(f"  {k}: {v}")
    print("\nGuest Persona:")
    for k, v in vals.get("guest_persona", {}).items():
        print(f"  {k}: {v}")
    segment_structure = vals.get("segment_structure", [])
    total_seg_minutes = sum(s.get("estimated_minutes", 0) for s in segment_structure)
    print(f"\nSegment Structure ({len(segment_structure)} segments, ~{total_seg_minutes} min total):")
    for i, seg in enumerate(segment_structure, 1):
        desc = seg["description"]
        mins = seg.get("estimated_minutes", "?")
        print(f"  {i}. {seg['name']} (~{mins} min): {desc[:70]}{'...' if len(desc) > 70 else ''}")
    print_separator()


def print_final_script(snapshot):
    vals = snapshot.values
    print_separator()
    print(f"FINAL SCRIPT: {vals.get('podcast_name', 'Podcast')}")
    print(f"Platform: {vals.get('platform_name', '')} | Topic: {vals.get('topic', '')} | Duration: ~{vals.get('estimated_duration_minutes', '?')} min")
    print_separator()
    for seg in vals.get("segments", []):
        print(f"\n{'=' * 50}")
        print(f"  {seg['type'].upper()}")
        print(f"{'=' * 50}")
        print(seg["content"])
    print_separator()
    print("SCRIPT COMPLETE")
    print_separator()


def get_agent_response_text(result: dict) -> tuple[str, bool]:
    """Extract the last AI message text and whether a tool call was made.
    Returns (text, has_tool_call)."""
    messages = result.get("messages", [])
    text = ""
    has_tool_call = False

    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            if msg.content:
                text = msg.content
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                has_tool_call = True
            break

    return text, has_tool_call


def get_user_input(prompt_text: str = "YOU: ") -> str:
    """Get input from user, handling KeyboardInterrupt gracefully."""
    try:
        return input(prompt_text).strip()
    except (KeyboardInterrupt, EOFError):
        print("\n\n👋 Session ended by user.")
        exit(0)


def print_segment_for_review(snapshot, max_segments):
    segments = snapshot.values.get("segments", [])
    if not segments:
        return
    latest = segments[-1]
    num = len(segments)
    segment_structure = snapshot.values.get("segment_structure", [])
    seg_minutes = "?"
    if num - 1 < len(segment_structure):
        seg_minutes = segment_structure[num - 1].get("estimated_minutes", "?")
    print(f"\n{'─' * 50}")
    print(f"  📝 SEGMENT {num}/{max_segments}: {latest['type'].upper()} (~{seg_minutes} min)")
    print(f"{'─' * 50}\n")
    print(latest["content"])
    print()
    
# ─────────────────────────────────────────────
# Sample Test Run
# ─────────────────────────────────────────────

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "interactive-session-1"}}

    print("\n🎙️  PODCAST SCRIPT PRODUCER")
    print("=" * 40)
    print("Describe the podcast you want to create.")
    print("The assistant will ask questions to gather details.")
    print("Type 'quit' at any time to exit.\n")

    # ──────────────────────────────────────────
    # PHASE 1: Agent Conversation
    # ──────────────────────────────────────────

    while True:
        user_input = get_user_input()

        if user_input.lower() in ("quit", "exit", "q"):
            print("\n👋 Goodbye!")
            exit(0)

        if not user_input:
            print("   (empty input — type something or 'quit' to exit)")
            continue

        # Send user message to the graph
        result = app.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config,
        )

        # Check where we landed
        snapshot = app.get_state(config)

        # If we hit confirm_requirements interrupt, break to Phase 2
        if snapshot.next and "confirm_requirements" in snapshot.next:
            ai_text, tool_called = get_agent_response_text(result)
            if ai_text:
                print(f"\n🤖 AGENT: {ai_text}")
            if tool_called:
                print("\n   🔧 Finalizing requirements...\n")
            break

        # Otherwise, print agent response and loop for next user message
        ai_text, tool_called = get_agent_response_text(result)
        if ai_text:
            print(f"\n🤖 AGENT: {ai_text}\n")
        if tool_called:
            print("   🔧 Finalizing requirements...\n")
            snapshot = app.get_state(config)
            if snapshot.next and "confirm_requirements" in snapshot.next:
                break

    # ──────────────────────────────────────────
    # PHASE 2: Review Parsed Requirements
    # ──────────────────────────────────────────

    snapshot = app.get_state(config)

    if not (snapshot.next and "confirm_requirements" in snapshot.next):
        print("❌ Could not reach the confirmation stage. Please try again.")
        exit(1)

    print_podcast_plan(snapshot)
    max_segments = len(snapshot.values.get("segment_structure", []))

    while True:
        print("\nOptions:")
        print("  [enter] — Confirm and start generating the script")
        print("  [text]  — Send a message to revise the plan")
        print("  quit    — Exit\n")

        user_input = get_user_input("YOUR DECISION: ")

        if user_input.lower() in ("quit", "exit", "q"):
            print("\n👋 Goodbye!")
            exit(0)

        if not user_input:
            print("\n✅ Plan confirmed. Generating script...\n")
            result = app.invoke(None, config)
            break
        else:
            print("\n🔄 Sending your feedback to the assistant...\n")
            app.update_state(
                config,
                {
                    "requirements_confirmed": False,
                    "messages": [HumanMessage(content=user_input)],
                },
                as_node="confirm_requirements",
            )
            result = app.invoke(None, config)

            while True:
                snapshot = app.get_state(config)

                if snapshot.next and "confirm_requirements" in snapshot.next:
                    print_podcast_plan(snapshot)
                    max_segments = len(snapshot.values.get("segment_structure", []))
                    break

                if not snapshot.next:
                    ai_text, tool_called = get_agent_response_text(result)
                    if ai_text:
                        print(f"🤖 AGENT: {ai_text}\n")
                    if tool_called:
                        print("   🔧 Re-finalizing requirements...\n")

                    user_input = get_user_input()
                    if user_input.lower() in ("quit", "exit", "q"):
                        print("\n👋 Goodbye!")
                        exit(0)
                    result = app.invoke(
                        {"messages": [HumanMessage(content=user_input)]},
                        config,
                    )
                else:
                    result = app.invoke(None, config)

    # ──────────────────────────────────────────
    # PHASE 3: Segment-by-Segment Review
    # ──────────────────────────────────────────

    snapshot = app.get_state(config)

    while True:
        snapshot = app.get_state(config)

        if not snapshot.next:
            break

        if "human_interrupt" not in snapshot.next:
            result = app.invoke(None, config)
            continue

        print_segment_for_review(snapshot, max_segments)

        current_segments = snapshot.values.get("segments", [])
        num_done = len(current_segments)
        seg_name = current_segments[-1]["type"] if current_segments else "?"

        while True:
            print("Options:")
            print("  [enter] — Accept this segment and continue")
            print("  [text]  — Provide feedback to regenerate this segment")
            print("  quit    — Exit (progress is saved)\n")

            user_input = get_user_input(f"SEGMENT {num_done}/{max_segments} DECISION: ")

            if user_input.lower() in ("quit", "exit", "q"):
                print("\n💾 Progress saved. Resume later with the same thread_id.")
                print(f"   Thread ID: {config['configurable']['thread_id']}")
                print(f"   Segments completed: {num_done}/{max_segments}")
                print("\n👋 Goodbye!")
                exit(0)

            if not user_input:
                print(f"\n   ✅ Accepted: {seg_name}\n")
                result = app.invoke(None, config)
                break
            else:
                print("\n   🔄 Regenerating with feedback...\n")
                app.update_state(
                    config,
                    {"human_feedback": user_input},
                    as_node="human_interrupt",
                )
                result = app.invoke(None, config)

                snapshot = app.get_state(config)
                if snapshot.next and "human_interrupt" in snapshot.next:
                    print_segment_for_review(snapshot, max_segments)
                    continue
                else:
                    break

    # ──────────────────────────────────────────
    # PHASE 4: Final Output
    # ──────────────────────────────────────────

    snapshot = app.get_state(config)
    print_final_script(snapshot)

    total = len(snapshot.values.get("segments", []))
    print(f"\n📊 Total segments: {total}/{max_segments}")
    print(f"⏱️  Target duration: {snapshot.values.get('estimated_duration_minutes', '?')} minutes")
    print(f"🏁 Complete: {snapshot.values.get('is_complete', False)}")