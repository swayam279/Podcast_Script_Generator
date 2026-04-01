# podcast_agent.py

import re
import sqlite3
import time
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.checkpoint.memory import MemorySaver  # noqa: F401
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

load_dotenv()

# ─────────────────────────────────────────────
# LLM Setup
# ─────────────────────────────────────────────

model = ChatNVIDIA(model="moonshotai/kimi-k2-instruct")


# ─────────────────────────────────────────────
# Retry Wrapper
# ─────────────────────────────────────────────


def invoke_with_retry(llm, prompt, max_retries=5, initial_wait=5):
    """Invoke the LLM with exponential backoff on rate limit errors."""
    for attempt in range(max_retries):
        try:
            return llm.invoke(prompt)
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "capacity" in error_str.lower() or "rate" in error_str.lower():
                wait_time = initial_wait * (2**attempt)
                print(f"   ⏳ Rate limited (attempt {attempt + 1}/{max_retries}). Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise
    return llm.invoke(prompt)


# ─────────────────────────────────────────────
# Script Cleaning Utility
# ─────────────────────────────────────────────


def clean_script_content(raw: str) -> str:
    """Strip LLM preamble, postamble, stage directions, and normalize formatting."""

    text = raw

    # 1. Normalize markdown speaker labels: **HOST:** → HOST:
    text = re.sub(r"\*\*(HOST|GUEST):\*\*", r"\1:", text)

    # 2. Remove stage directions in brackets: [laughs], [soft field audio: ...]
    text = re.sub(r"$$.*?$$", "", text)

    # 3. Remove asterisk-wrapped actions: *laughs*, *chuckling*, *finger-snaps*
    text = re.sub(r"\*[^*\n]+\*", "", text)

    # 4. Fix compound speaker labels like "HOST and GUEST together:"
    #    Convert to two separate lines
    text = re.sub(
        r"^HOST\s+and\s+GUEST\s*(?:together)?\s*:\s*(.+)$",
        r"HOST: \1\nGUEST: \1",
        text,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    # Handle reverse order too
    text = re.sub(
        r"^GUEST\s+and\s+HOST\s*(?:together)?\s*:\s*(.+)$",
        r"GUEST: \1\nHOST: \1",
        text,
        flags=re.MULTILINE | re.IGNORECASE,
    )

    # 5. Ensure each HOST: / GUEST: starts on its own line
    text = re.sub(r"(?<!\n)\s*(HOST:|GUEST:)", r"\n\1", text)

    # 6. Cut preamble: everything before the first HOST: or GUEST: line
    match = re.search(r"^(HOST|GUEST):", text, re.MULTILINE)
    if not match:
        return raw.strip()
    text = text[match.start() :]

    # 7. Cut postamble at horizontal rules, markdown headings, or meta-commentary
    text = re.split(r"\n\s*---\s*\n", text)[0]
    text = re.split(r"\n\s*#{1,4}\s+", text)[0]
    text = re.split(
        r"\n\s*(?:Key (?:Adjustments|Changes)|Changes [Mm]ade|Summary of [Cc]hanges|"
        r"Note:|Here (?:are|is) |I (?:have|made|kept)|The (?:above|script|rewrite))",
        text,
        maxsplit=1,
    )[0]

    # 8. Clean up leftover whitespace from removed stage directions
    text = re.sub(r"(HOST:|GUEST:)\s{2,}", r"\1 ", text)

    # 9. Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def is_segment_truncated(content: str) -> bool:
    """Detect if a generated segment was cut off mid-sentence."""
    stripped = content.strip()
    if not stripped:
        return True

    last_line = stripped.split("\n")[-1].strip()

    # Ends with just the speaker label and no/minimal content
    if re.match(r"^(HOST|GUEST):\s*\S{0,3}$", last_line):
        return True

    # Doesn't end with sentence-ending punctuation
    if not re.search(r"[.!?\"')\u2019]$", stripped):
        return True

    return False


# ─────────────────────────────────────────────
# Pydantic Models for Structured Output
# ─────────────────────────────────────────────


class Segment(BaseModel):
    name: str = Field(description="The name/identifier of this segment (e.g., 'welcome', 'intro', 'discussion', 'outro')")
    description: str = Field(description="Detailed guidelines for writing this segment, including tone, content requirements, and approximate length")


class PodcastDetails(BaseModel):
    topic: str = Field(description="The main topic of the podcast episode")
    host_persona: dict = Field(description=("Details about the host in JSON format that would help build a unique persona with at least 4 traits (e.g. vocabulary level, humor style, catchphrases, energy level)"))
    guest_persona: dict = Field(description=("Details about the guest in JSON format that would help build a unique persona with at least 4 traits (e.g. expertise, speaking style, personality quirks, communication approach)"))
    podcast_name: str = Field(description="Name or Title of the podcast")
    platform_name: str = Field(description="Name or Title of the platform hosting the podcast")
    segments: list[Segment] = Field(
        description=(
            "A list of segments that make up the podcast structure, in order. Each segment should have a "
            "name and description. The following segments should be minimally included: welcome (host opens "
            "the show), intro (guest introduction), discussion (main topic exploration), outro (wrap-up and "
            "sign-off). You can add, remove, or rename segments based on what makes sense for this specific "
            "podcast. Try to include a mix of natural and scripted segments with more than the minimally "
            "required segments."
        )
    )


detail_model = model.with_structured_output(PodcastDetails)


# ─────────────────────────────────────────────
# Agent Tool: Finalize Requirements
# ─────────────────────────────────────────────


class FinalizeRequirements(BaseModel):
    """Call this tool ONLY when you have gathered ALL the required information from the user."""

    summary: str = Field(description=("A comprehensive summary of everything gathered from the conversation, including: podcast name, platform name, episode topic, host persona details (at least 4 traits), guest persona details (at least 4 traits), and any specific segment preferences."))


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
4. Host persona — at least 4 traits. Ask about: vocabulary level, humor style, catchphrases, energy level, speaking patterns
5. Guest persona — at least 4 traits. Ask about: area of expertise, speaking style, personality quirks, communication approach
6. (Optional) Any specific segment preferences or special requests

Guidelines:
- Be conversational and friendly.
- Ask follow-up questions if the user gives vague answers (e.g., "a funny host" → ask what KIND of funny).
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
    host = state["host_persona"]
    guest = state["guest_persona"]
    segments = list(state.get("segments", []))
    feedback = state.get("human_feedback", "").strip()

    current_segment = segment_structure[current_index]
    segment_name = current_segment["name"]
    segment_guideline = current_segment["description"]

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
Podcast Name  : {podcast_name}
Platform Name : {platform_name}
Topic         : {topic}
Host          : {host}
Guest         : {guest}

{SCRIPT_FORMAT_RULES}

Additional rewrite rules:
• Apply the feedback precisely; do not change anything the reviewer didn't mention.
• Keep the same overall structure and flow of the segment."""

        response = invoke_with_retry(model, prompt)
        content = clean_script_content(response.content)

        # Handle truncation in rewrite
        max_continuations = 3
        for _ in range(max_continuations):
            if not is_segment_truncated(content):
                break
            continuation_prompt = f"""You are a podcast script writer. Continue the following \
"{last["type"]}" segment EXACTLY where it left off. Pick up mid-sentence if needed.

--- SEGMENT SO FAR ---
{content}

--- CONTEXT ---
Podcast Name  : {podcast_name}
Platform Name : {platform_name}
Topic         : {topic}
Host          : {host}
Guest         : {guest}

{SCRIPT_FORMAT_RULES}

Additional rules:
• Start IMMEDIATELY where the text above ended — do not repeat any lines.
• Complete the segment naturally and bring it to a proper conclusion."""

            cont_response = invoke_with_retry(model, continuation_prompt)
            cont_clean = clean_script_content(cont_response.content)
            content = content + "\n" + cont_clean

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

Podcast Name  : {podcast_name}
Platform Name : {platform_name}
Topic         : {topic}
Host persona  : {host}
Guest persona : {guest}

Segment guidelines: {segment_guideline}

{previous_context}
{SCRIPT_FORMAT_RULES}"""

    response = invoke_with_retry(model, prompt)
    content = clean_script_content(response.content)

    # Handle truncation — continue generating until complete
    max_continuations = 3
    for _ in range(max_continuations):
        if not is_segment_truncated(content):
            break
        continuation_prompt = f"""You are a podcast script writer. Continue the following \
"{segment_name}" segment EXACTLY where it left off. Pick up mid-sentence if needed.

--- SEGMENT SO FAR ---
{content}

--- CONTEXT ---
Podcast Name  : {podcast_name}
Platform Name : {platform_name}
Topic         : {topic}
Host          : {host}
Guest         : {guest}

{SCRIPT_FORMAT_RULES}

Additional rules:
• Start IMMEDIATELY where the text above ended — do not repeat any lines.
• Complete the segment naturally and bring it to a proper conclusion."""

        cont_response = invoke_with_retry(model, continuation_prompt)
        cont_clean = clean_script_content(cont_response.content)
        content = content + "\n" + cont_clean

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

    response = invoke_with_retry(agent_model, messages)
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

    details = invoke_with_retry(detail_model, conversation)

    segment_structure = [{"name": seg.name, "description": seg.description} for seg in details.segments]

    return {
        "topic": details.topic,
        "host_persona": details.host_persona,
        "guest_persona": details.guest_persona,
        "podcast_name": details.podcast_name,
        "platform_name": details.platform_name,
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
    print("\nHost Persona:")
    for k, v in vals.get("host_persona", {}).items():
        print(f"  {k}: {v}")
    print("\nGuest Persona:")
    for k, v in vals.get("guest_persona", {}).items():
        print(f"  {k}: {v}")
    print(f"\nSegment Structure ({len(vals.get('segment_structure', []))} segments):")
    for i, seg in enumerate(vals.get("segment_structure", []), 1):
        desc = seg["description"]
        print(f"  {i}. {seg['name']}: {desc[:80]}{'...' if len(desc) > 80 else ''}")
    print_separator()


def print_final_script(snapshot):
    vals = snapshot.values
    print_separator()
    print(f"FINAL SCRIPT: {vals.get('podcast_name', 'Podcast')}")
    print(f"Platform: {vals.get('platform_name', '')} | Topic: {vals.get('topic', '')}")
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
    print(f"\n{'─' * 50}")
    print(f"  📝 SEGMENT {num}/{max_segments}: {latest['type'].upper()}")
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
            # Tool was called but we need to check if graph continued past tools→parse_input
            # and landed at confirm_requirements
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
            # User confirmed — resume the graph
            print("\n✅ Plan confirmed. Generating script...\n")
            result = app.invoke(None, config)
            break
        else:
            # User wants changes — reject confirmation and send back to agent
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

            # This routes to agent_chat → agent responds → may finalize again
            # Loop until we're back at confirm_requirements
            while True:
                snapshot = app.get_state(config)

                if snapshot.next and "confirm_requirements" in snapshot.next:
                    print_podcast_plan(snapshot)
                    max_segments = len(snapshot.values.get("segment_structure", []))
                    break

                if not snapshot.next:
                    # Agent responded with text, waiting for user
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

        # Graph complete
        if not snapshot.next:
            break

        # Not at human_interrupt — resume automatically
        if "human_interrupt" not in snapshot.next:
            result = app.invoke(None, config)
            continue

        # Display the generated segment
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
                # Accept — resume graph (human_interrupt passthrough → check_completion → next)
                print(f"\n   ✅ Accepted: {seg_name}\n")
                result = app.invoke(None, config)
                break
            else:
                # Feedback — inject and resume (routes back to generate_segment → regenerates)
                print("\n   🔄 Regenerating with feedback...\n")
                app.update_state(
                    config,
                    {"human_feedback": user_input},
                    as_node="human_interrupt",
                )
                result = app.invoke(None, config)

                # After regeneration, we're back at human_interrupt with the new version
                snapshot = app.get_state(config)
                if snapshot.next and "human_interrupt" in snapshot.next:
                    print_segment_for_review(snapshot, max_segments)
                    # Loop back to ask for decision on the regenerated version
                    continue
                else:
                    # Unexpected — break inner loop, outer loop will handle
                    break

    # ──────────────────────────────────────────
    # PHASE 4: Final Output
    # ──────────────────────────────────────────

    snapshot = app.get_state(config)
    print_final_script(snapshot)

    total = len(snapshot.values.get("segments", []))
    print(f"\n📊 Total segments: {total}/{max_segments}")
    print(f"🏁 Complete: {snapshot.values.get('is_complete', False)}")
