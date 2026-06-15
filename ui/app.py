import streamlit as st
import requests
import json
import sseclient  # pip install sseclient-py

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Sheares Hall Policy Copilot",
    page_icon="🏛️",
    layout="wide"
)

st.title("🏛️ Sheares Hall Policy Copilot")
st.caption("Ask about hall policies, check your points eligibility, or calculate subsidies.")

# Sidebar: info + example questions
with st.sidebar:
    st.header("💡 Example Questions")
    examples = [
        "What is the minimum points needed for room retention?",
        "How do I submit a finance claim?",
        "I have 45 points — what tier am I in?",
        "Calculate my subsidy for 10 days stay",
        "What are the social media posting rules?",
        "How long do I have to submit a receipt after an event?",
        "What is the quorum for a JCRC meeting?",
        "Can I swap rooms in the first week?",
    ]
    for ex in examples:
        if st.button(ex, key=ex, use_container_width=True):
            st.session_state.pending_message = ex

    st.divider()
    st.caption("v2.0 · Adaptive RAG + Agentic AI")
    st.caption("Built by Isaac Chin")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            with st.expander(f"📚 {len(msg['citations'])} sources"):
                for c in msg["citations"]:
                    st.caption(f"• {c.get('source_doc', 'Policy document')} — {c.get('section', '')}")
        if msg.get("route"):
            route_icon = "⚡" if msg["route"] == "fast" else "🤖"
            st.caption(f"{route_icon} {'Fast RAG path' if msg['route'] == 'fast' else 'Agentic path'}")

# Handle pending message from sidebar buttons
prompt = st.session_state.pop("pending_message", None)

# Chat input
user_input = st.chat_input("Ask a hall policy question...") or prompt

if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Stream response
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_text = ""
        metadata = {}

        try:
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]
            ]

            response = requests.post(
                f"{API_URL}/chat",
                json={"message": user_input, "history": history, "stream": True},
                stream=True,
                timeout=30,
            )

            client = sseclient.SSEClient(response)
            for event in client.events():
                data = json.loads(event.data)
                if data["type"] == "token":
                    full_text += data["content"]
                    placeholder.markdown(full_text + "▌")
                elif data["type"] == "done":
                    metadata = data
                    placeholder.markdown(full_text)
        except Exception as e:
            full_text = f"⚠️ Could not reach the API: {e}\n\nMake sure `uvicorn api.main:app` is running."
            placeholder.markdown(full_text)

        # Show citations
        if metadata.get("citations"):
            with st.expander(f"📚 {len(metadata['citations'])} sources"):
                for c in metadata["citations"]:
                    st.caption(f"• {c.get('source_doc', 'Policy document')} — {c.get('section', '')}")

        # Show routing info
        if metadata.get("route"):
            route = metadata["route"]
            col1, col2, col3 = st.columns(3)
            col1.caption(f"{'⚡ Fast path' if route == 'fast' else '🤖 Agentic path'}")
            if metadata.get("tool_results"):
                col2.caption(f"🔧 {len(metadata['tool_results'])} tool(s) used")
            if not metadata.get("is_grounded", True):
                col3.caption("⚠️ Low confidence")

        # Feedback buttons
        cid = metadata.get("correlation_id", "")
        col1, col2, _ = st.columns([1, 1, 8])
        if col1.button("👍", key=f"up_{cid}"):
            requests.post(f"{API_URL}/feedback", json={"correlation_id": cid, "rating": 1})
        if col2.button("👎", key=f"dn_{cid}"):
            requests.post(f"{API_URL}/feedback", json={"correlation_id": cid, "rating": -1})

    # Store message with metadata
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_text,
        "citations": metadata.get("citations", []),
        "route": metadata.get("route", ""),
    })
