# frontend.py

import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from backend import (
    app,
    delete_thread,
    get_all_thread_metadata,
    get_thread_name,
    memory,
    rename_thread,
)

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Podcast Script Producer",
    page_icon="🎙️",
    layout="wide",
)


# ─────────────────────────────────────────────
# Utility Functions
# ─────────────────────────────────────────────


def generate_thread_id():
    return str(uuid.uuid4())


def get_config():
    return {"configurable": {"thread_id": st.session_state["session_id"]}}


def retrieve_all_threads():
    """Retrieve all thread IDs from the checkpoint store and their display names."""
    all_threads = set()
    for cpt in memory.list(None):
        tid = cpt.config.get("configurable", {}).get("thread_id")
        if tid:
            all_threads.add(tid)

    # Load display names from metadata table
    metadata = get_all_thread_metadata()

    # Build ordered list — threads with metadata first, then any orphan checkpoint threads
    thread_list = []
    for tid in all_threads:
        thread_list.append(tid)
        # Ensure all checkpoint threads have a metadata entry
        if tid not in metadata:
            rename_thread(tid, tid[:8])

    return thread_list


def load_session_state(thread_id):
    """Load full graph state for a thread and reconstruct UI state."""
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = app.get_state(config)

    if not snapshot or not snapshot.values:
        st.session_state["phase"] = "chat"
        st.session_state["chat_history"] = []
        return

    vals = snapshot.values

    # Rebuild chat history
    chat_history = []
    for msg in vals.get("messages", []):
        if isinstance(msg, HumanMessage):
            chat_history.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage) and msg.content:
            chat_history.append({"role": "assistant", "content": msg.content})

    st.session_state["chat_history"] = chat_history

    # Determine phase
    next_nodes = snapshot.next if snapshot.next else ()

    if "confirm_requirements" in next_nodes:
        st.session_state["phase"] = "review_plan"
    elif "human_interrupt" in next_nodes:
        st.session_state["phase"] = "review_segments"
    elif vals.get("is_complete", False):
        st.session_state["phase"] = "complete"
    elif vals.get("segment_structure"):
        if vals.get("segments"):
            st.session_state["phase"] = "complete" if vals.get("is_complete") else "review_segments"
        else:
            st.session_state["phase"] = "review_plan"
    else:
        st.session_state["phase"] = "chat"


def reset_session():
    thread_id = generate_thread_id()
    st.session_state["session_id"] = thread_id
    st.session_state["chat_history"] = []
    st.session_state["phase"] = "chat"
    st.session_state["plan_reviewed"] = False
    # Save default display name
    default_name = f"New Podcast ({thread_id[:8]})"
    rename_thread(thread_id, default_name)
    add_thread(thread_id)


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def get_snapshot():
    return app.get_state(get_config())


def get_display_name(thread_id):
    """Get the display name for a thread, falling back to truncated ID."""
    name = get_thread_name(thread_id)
    return name if name else thread_id[:8]


def get_plan_display(snapshot):
    vals = snapshot.values
    lines = []
    lines.append(f"**Podcast Name:** {vals.get('podcast_name', 'N/A')}")
    lines.append(f"**Platform:** {vals.get('platform_name', 'N/A')}")
    lines.append(f"**Topic:** {vals.get('topic', 'N/A')}")
    lines.append(f"**Estimated Duration:** {vals.get('estimated_duration_minutes', 'N/A')} minutes")
    lines.append("")
    lines.append("**Host Persona:**")
    for k, v in vals.get("host_persona", {}).items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("**Guest Persona:**")
    for k, v in vals.get("guest_persona", {}).items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    segments = vals.get("segment_structure", [])
    total_seg_minutes = sum(s.get("estimated_minutes", 0) for s in segments)
    lines.append(f"**Segments ({len(segments)} segments, ~{total_seg_minutes} min total):**")
    for i, seg in enumerate(segments, 1):
        desc = seg["description"]
        mins = seg.get("estimated_minutes", "?")
        lines.append(f"{i}. **{seg['name']}** (~{mins} min): {desc[:100]}{'...' if len(desc) > 100 else ''}")
    return "\n".join(lines)


# ─────────────────────────────────────────────
# Session State Initialization
# ─────────────────────────────────────────────

if "session_id" not in st.session_state:
    st.session_state["session_id"] = generate_thread_id()

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

if "phase" not in st.session_state:
    st.session_state["phase"] = "chat"

if "plan_reviewed" not in st.session_state:
    st.session_state["plan_reviewed"] = False

if "renaming_thread" not in st.session_state:
    st.session_state["renaming_thread"] = None

add_thread(st.session_state["session_id"])

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────

st.sidebar.title("🎙️ Podcast Producer")
st.sidebar.caption("AI-powered podcast script generator")

if st.sidebar.button("➕ New Podcast", use_container_width=True):
    reset_session()
    st.rerun()

st.sidebar.divider()
st.sidebar.subheader("Sessions")

for thread_id in st.session_state["chat_threads"][::-1]:
    is_active = thread_id == st.session_state["session_id"]
    display_name = get_display_name(thread_id)

    # ── Rename mode for this thread ──
    if st.session_state["renaming_thread"] == thread_id:
        with st.sidebar.container():
            new_name = st.text_input(
                "New name:",
                value=display_name,
                key=f"rename_input_{thread_id}",
                label_visibility="collapsed",
            )
            rcol1, rcol2 = st.columns(2)
            with rcol1:
                if st.button("✅", key=f"rename_save_{thread_id}", use_container_width=True):
                    if new_name.strip():
                        rename_thread(thread_id, new_name.strip())
                    st.session_state["renaming_thread"] = None
                    st.rerun()
            with rcol2:
                if st.button("❌", key=f"rename_cancel_{thread_id}", use_container_width=True):
                    st.session_state["renaming_thread"] = None
                    st.rerun()
        continue

    # ── Normal display ──
    label_prefix = "▶ " if is_active else "🧵 "
    label_suffix = " (active)" if is_active else ""
    label = f"{label_prefix}{display_name}{label_suffix}"

    col_main, col_rename, col_delete = st.sidebar.columns([6, 1, 1])

    with col_main:
        if st.button(label, key=f"thread_{thread_id}", use_container_width=True):
            st.session_state["session_id"] = thread_id
            load_session_state(thread_id)
            st.rerun()

    with col_rename:
        if st.button("✏️", key=f"rename_btn_{thread_id}", use_container_width=True):
            st.session_state["renaming_thread"] = thread_id
            st.rerun()

    with col_delete:
        if st.button("🗑️", key=f"delete_btn_{thread_id}", use_container_width=True):
            # Delete the thread from DB and local state
            delete_thread(thread_id)
            st.session_state["chat_threads"].remove(thread_id)

            # If we deleted the active session, create a new one
            if thread_id == st.session_state["session_id"]:
                reset_session()

            # Clear rename state if it was targeting the deleted thread
            if st.session_state["renaming_thread"] == thread_id:
                st.session_state["renaming_thread"] = None

            st.rerun()

st.sidebar.divider()
phase_labels = {
    "chat": "💬 Gathering Requirements",
    "review_plan": "📋 Reviewing Plan",
    "review_segments": "📝 Reviewing Segments",
    "complete": "✅ Script Complete",
}
st.sidebar.info(f"**Phase:** {phase_labels.get(st.session_state['phase'], 'Unknown')}")

# ─────────────────────────────────────────────
# Main Content
# ─────────────────────────────────────────────

st.title("🎙️ Podcast Script Producer")

config = get_config()

# ─────────────────────────────────────────────
# PHASE: Chat
# ─────────────────────────────────────────────

if st.session_state["phase"] == "chat":
    st.caption("Describe the podcast you want to create. The assistant will ask questions to gather details.")

    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Describe your podcast...")

    if user_input:
        st.session_state["chat_history"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = app.invoke(
                    {"messages": [HumanMessage(content=user_input)]},
                    config,
                )

        snapshot = get_snapshot()

        if snapshot and snapshot.next and "confirm_requirements" in snapshot.next:
            ai_text = ""
            for msg in reversed(result.get("messages", [])):
                if isinstance(msg, AIMessage) and msg.content:
                    ai_text = msg.content
                    break
            if ai_text:
                st.session_state["chat_history"].append({"role": "assistant", "content": ai_text})

            st.session_state["chat_history"].append({"role": "assistant", "content": "🔧 I've gathered enough information! Let me put together a plan for your review."})
            st.session_state["phase"] = "review_plan"
            st.rerun()
        else:
            ai_text = ""
            for msg in reversed(result.get("messages", [])):
                if isinstance(msg, AIMessage) and msg.content:
                    ai_text = msg.content
                    break
            if ai_text:
                st.session_state["chat_history"].append({"role": "assistant", "content": ai_text})
                st.rerun()

# ─────────────────────────────────────────────
# PHASE: Review Plan
# ─────────────────────────────────────────────

elif st.session_state["phase"] == "review_plan":
    with st.expander("💬 Conversation History", expanded=False):
        for msg in st.session_state["chat_history"]:
            role_icon = "🧑" if msg["role"] == "user" else "🤖"
            st.markdown(f"{role_icon} **{msg['role'].title()}:** {msg['content']}")

    st.subheader("📋 Review Your Podcast Plan")

    snapshot = get_snapshot()
    if snapshot and snapshot.values:
        plan_text = get_plan_display(snapshot)
        st.markdown(plan_text)

        # Auto-rename the session to the podcast name if still using default
        vals = snapshot.values
        podcast_name = vals.get("podcast_name", "")
        if podcast_name:
            current_name = get_display_name(st.session_state["session_id"])
            if current_name.startswith("New Podcast (") or current_name == st.session_state["session_id"][:8]:
                rename_thread(st.session_state["session_id"], podcast_name)

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            if st.button("✅ Confirm & Start Generating", type="primary", use_container_width=True):
                with st.spinner("Generating first segment..."):
                    result = app.invoke(None, config)
                st.session_state["phase"] = "review_segments"
                st.rerun()

        with col2:
            feedback = st.text_area(
                "Want changes? Describe what to adjust:",
                placeholder="e.g., 'Make it 45 minutes instead' or 'Add a rapid-fire Q&A segment'",
                key="plan_feedback",
            )
            if st.button("🔄 Revise Plan", use_container_width=True):
                if feedback:
                    with st.spinner("Revising plan..."):
                        st.session_state["chat_history"].append({"role": "user", "content": feedback})

                        app.update_state(
                            config,
                            {
                                "requirements_confirmed": False,
                                "messages": [HumanMessage(content=feedback)],
                            },
                            as_node="confirm_requirements",
                        )
                        result = app.invoke(None, config)

                        snapshot = app.get_state(config)

                        if snapshot.next and "confirm_requirements" in snapshot.next:
                            for msg in reversed(result.get("messages", [])):
                                if isinstance(msg, AIMessage) and msg.content:
                                    st.session_state["chat_history"].append({"role": "assistant", "content": msg.content})
                                    break
                            st.rerun()
                        else:
                            for msg in reversed(result.get("messages", [])):
                                if isinstance(msg, AIMessage) and msg.content:
                                    st.session_state["chat_history"].append({"role": "assistant", "content": msg.content})
                                    break
                            st.session_state["phase"] = "chat"
                            st.rerun()
                else:
                    st.warning("Please enter your feedback before clicking Revise.")
    else:
        st.error("Could not load plan. Please start a new session.")
        if st.button("Start Over"):
            reset_session()
            st.rerun()

# ─────────────────────────────────────────────
# PHASE: Review Segments
# ─────────────────────────────────────────────

elif st.session_state["phase"] == "review_segments":
    snapshot = get_snapshot()

    if not snapshot or not snapshot.values:
        st.error("Could not load session. Please start a new session.")
        if st.button("Start Over"):
            reset_session()
            st.rerun()
    else:
        vals = snapshot.values
        max_segments = len(vals.get("segment_structure", []))
        current_segments = vals.get("segments", [])
        num_done = len(current_segments)

        with st.expander("📋 Podcast Plan", expanded=False):
            st.markdown(get_plan_display(snapshot))

        if num_done > 1:
            with st.expander(f"✅ Accepted Segments ({num_done - 1}/{max_segments})", expanded=False):
                for i, seg in enumerate(current_segments[:-1], 1):
                    seg_structure = vals.get("segment_structure", [])
                    seg_mins = seg_structure[i - 1].get("estimated_minutes", "?") if i - 1 < len(seg_structure) else "?"
                    st.markdown(f"**{i}. {seg['type']}** (~{seg_mins} min)")
                    st.code(seg["content"], language=None)
                    st.divider()

        if not snapshot.next:
            st.session_state["phase"] = "complete"
            st.rerun()

        if "human_interrupt" not in snapshot.next:
            with st.spinner("Processing..."):
                result = app.invoke(None, config)
            st.rerun()

        if current_segments:
            latest = current_segments[-1]
            seg_structure = vals.get("segment_structure", [])
            seg_mins = seg_structure[num_done - 1].get("estimated_minutes", "?") if num_done - 1 < len(seg_structure) else "?"
            st.subheader(f"📝 Segment {num_done}/{max_segments}: {latest['type']} (~{seg_mins} min)")
            st.progress(num_done / max_segments, text=f"Progress: {num_done}/{max_segments} segments")
            st.code(latest["content"], language=None)

            st.divider()

            col1, col2 = st.columns(2)

            with col1:
                if st.button("✅ Accept Segment", type="primary", use_container_width=True):
                    with st.spinner("Generating next segment..." if num_done < max_segments else "Finishing up..."):
                        result = app.invoke(None, config)
                    snapshot = app.get_state(config)
                    if not snapshot.next:
                        st.session_state["phase"] = "complete"
                    st.rerun()

            with col2:
                feedback = st.text_area(
                    "Feedback to regenerate this segment:",
                    placeholder="e.g., 'Make it shorter' or 'Make the host less aggressive'",
                    key=f"segment_feedback_{num_done}",
                )
                if st.button("🔄 Regenerate", use_container_width=True):
                    if feedback:
                        with st.spinner("Regenerating segment..."):
                            app.update_state(
                                config,
                                {"human_feedback": feedback},
                                as_node="human_interrupt",
                            )
                            result = app.invoke(None, config)
                        st.rerun()
                    else:
                        st.warning("Please enter feedback before clicking Regenerate.")

# ─────────────────────────────────────────────
# PHASE: Complete
# ─────────────────────────────────────────────

elif st.session_state["phase"] == "complete":
    snapshot = get_snapshot()

    if not snapshot or not snapshot.values:
        st.error("Could not load session.")
        if st.button("Start Over"):
            reset_session()
            st.rerun()
    else:
        vals = snapshot.values
        total = len(vals.get("segments", []))
        max_segments = len(vals.get("segment_structure", []))
        duration = vals.get("estimated_duration_minutes", "?")

        st.balloons()

        st.subheader(f"🎬 {vals.get('podcast_name', 'Your Podcast')} — Complete Script")
        st.caption(f"Platform: {vals.get('platform_name', '')} | Topic: {vals.get('topic', '')} | Duration: ~{duration} min")
        st.caption(f"✅ {total}/{max_segments} segments generated")

        with st.expander("📋 Podcast Plan", expanded=False):
            st.markdown(get_plan_display(snapshot))

        for i, seg in enumerate(vals.get("segments", []), 1):
            seg_structure = vals.get("segment_structure", [])
            seg_mins = seg_structure[i - 1].get("estimated_minutes", "?") if i - 1 < len(seg_structure) else "?"
            with st.expander(f"📝 {i}. {seg['type']} (~{seg_mins} min)", expanded=True):
                st.code(seg["content"], language=None)

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            seg_structure = vals.get("segment_structure", [])
            script_text = f"PODCAST: {vals.get('podcast_name', 'Podcast')}\n"
            script_text += f"PLATFORM: {vals.get('platform_name', '')}\n"
            script_text += f"TOPIC: {vals.get('topic', '')}\n"
            script_text += f"ESTIMATED DURATION: ~{duration} minutes\n"
            script_text += "=" * 60 + "\n\n"
            for i, seg in enumerate(vals.get("segments", []), 1):
                seg_mins = seg_structure[i - 1].get("estimated_minutes", "?") if i - 1 < len(seg_structure) else "?"
                script_text += f"\n{'=' * 40}\n"
                script_text += f"  {seg['type'].upper()} (~{seg_mins} min)\n"
                script_text += f"{'=' * 40}\n\n"
                script_text += seg["content"] + "\n"

            st.download_button(
                label="📥 Download Script (.txt)",
                data=script_text,
                file_name=f"{vals.get('podcast_name', 'podcast').replace(' ', '_')}_script.txt",
                mime="text/plain",
                use_container_width=True,
            )

        with col2:
            if st.button("➕ Create Another Podcast", use_container_width=True):
                reset_session()
                st.rerun()