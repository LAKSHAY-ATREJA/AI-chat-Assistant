import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from datetime import datetime

load_dotenv()

st.set_page_config(
    page_title="AI Chat Assistant",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1a5276; }
    .persona-card {
        background: #f0f3f4;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin: 0.3rem 0;
        cursor: pointer;
        border-left: 3px solid #2980b9;
    }
    .stat-box {
        background: #eaf4fb;
        border-radius: 8px;
        padding: 0.8rem;
        text-align: center;
    }
    .error-box {
        background: #fdf2f2;
        border-left: 4px solid #e74c3c;
        padding: 0.8rem 1rem;
        border-radius: 4px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Personas ──────────────────────────────────────────────────
PERSONAS = {
    "General Assistant": "You are a helpful, friendly, and knowledgeable AI assistant. Answer clearly and concisely.",
    "Python Tutor": "You are an expert Python programming tutor. Explain concepts clearly with code examples. Always include working code snippets.",
    "Writing Coach": "You are a professional writing coach. Help improve writing, suggest better phrasing, and provide constructive feedback.",
    "Research Assistant": "You are a thorough research assistant. Provide well-structured, fact-based answers with clear explanations.",
    "Career Advisor": "You are an experienced career advisor specialising in tech roles. Give practical, actionable career advice."
}

DEFAULT_PERSONA = "General Assistant"

# ── Session state ─────────────────────────────────────────────
def _init_session_state():
    """Initialise all session state keys with safe defaults."""
    defaults = {
        "messages": [],
        "memory": None,
        "total_messages": 0,
        "current_persona": DEFAULT_PERSONA,
        "_last_api_key": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

_init_session_state()


def validate_api_key(api_key: str) -> bool:
    """Return True if the key looks structurally valid (starts with gsk_)."""
    return bool(api_key) and api_key.strip().startswith("gsk_")


def create_conversation(api_key: str, persona_prompt: str) -> ConversationChain:
    """Create a LangChain ConversationChain backed by Groq Llama3."""
    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0.7,
        max_tokens=1024,
        groq_api_key=api_key.strip()
    )

    memory = ConversationBufferWindowMemory(
        k=15,
        human_prefix="User",
        ai_prefix="Assistant"
    )

    prompt_template = persona_prompt + """

Current conversation:
{history}
User: {input}
Assistant:"""

    prompt = PromptTemplate(
        input_variables=["history", "input"],
        template=prompt_template
    )

    return ConversationChain(llm=llm, memory=memory, prompt=prompt, verbose=False)


def get_or_create_conversation(api_key: str, persona_name: str) -> ConversationChain:
    """Return the cached ConversationChain for the current persona/key combination,
    creating a new one when either changes."""
    conv_key = f"conversation_{persona_name}"
    api_key_changed = st.session_state.get("_last_api_key") != api_key

    if conv_key not in st.session_state or api_key_changed:
        st.session_state[conv_key] = create_conversation(
            api_key, PERSONAS[persona_name]
        )
        st.session_state["_last_api_key"] = api_key
        st.session_state.memory = conv_key  # track which chain is active

    return st.session_state[conv_key]


def build_export_text(messages: list) -> str:
    """Format chat history as plain text for download."""
    lines = [f"AI Chat Assistant — Export ({datetime.now().strftime('%Y-%m-%d %H:%M')})", ""]
    for msg in messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        ts = msg.get("timestamp", "")
        lines.append(f"[{ts}] {role}:")
        lines.append(msg["content"])
        lines.append("")
    return "\n".join(lines)


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")

    _env_key = os.getenv("GROQ_API_KEY", "")
    api_key = st.text_input(
        "Groq API Key",
        value=_env_key,
        type="password",
        placeholder="gsk_...",
        help="Get a free key at console.groq.com. Can also be set via the GROQ_API_KEY environment variable.",
    )

    if api_key and not validate_api_key(api_key):
        st.warning("Key format looks unexpected — Groq keys usually start with 'gsk_'.")

    st.divider()
    st.header("Select Persona")

    for persona_name in PERSONAS:
        is_active = persona_name == st.session_state.current_persona
        if st.button(
            persona_name,
            use_container_width=True,
            type="primary" if is_active else "secondary",
            key=f"persona_btn_{persona_name}"
        ):
            if not is_active:
                st.session_state.current_persona = persona_name
                st.session_state.messages = []
                st.session_state.memory = None
                st.rerun()

    st.divider()
    st.header("Session Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Messages", len(st.session_state.messages))
    with col2:
        st.metric("Total Sent", st.session_state.total_messages)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.memory = None
            # Remove cached chain so a fresh one is created on next message
            conv_key = f"conversation_{st.session_state.current_persona}"
            st.session_state.pop(conv_key, None)
            st.rerun()
    with col2:
        if st.session_state.messages:
            export_text = build_export_text(st.session_state.messages)
            st.download_button(
                "Export",
                data=export_text,
                file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain",
                use_container_width=True,
            )
        else:
            st.button("Export", disabled=True, use_container_width=True)


# ── Main area ─────────────────────────────────────────────────
st.markdown('<p class="main-header">AI Chat Assistant</p>', unsafe_allow_html=True)
st.markdown("Intelligent conversation with memory — remembers the last 15 exchanges.")

if not api_key:
    st.info("Enter your Groq API key in the sidebar to start chatting.")
    st.markdown(
        "You can get a free API key at [console.groq.com](https://console.groq.com). "
        "No credit card is required."
    )
    st.stop()

# Active persona label
st.info(f"Active Persona: {st.session_state.current_persona} — "
        f"{PERSONAS[st.session_state.current_persona][:80]}...")

# Load or create the conversation chain
try:
    chain = get_or_create_conversation(api_key, st.session_state.current_persona)
except Exception as e:
    st.error(f"Failed to initialise the AI model: {e}")
    st.stop()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "timestamp" in message:
            st.caption(message["timestamp"])

# Chat input
user_input = st.chat_input("Type your message...")
if user_input:
    user_input = user_input.strip()
    if not user_input:
        st.warning("Please enter a message before sending.")
        st.stop()

    timestamp = datetime.now().strftime("%H:%M")
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "timestamp": timestamp
    })

    with st.chat_message("user"):
        st.write(user_input)
        st.caption(timestamp)

    response = ""
    reply_ts = datetime.now().strftime("%H:%M")
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = chain.predict(input=user_input)
                if not response or not response.strip():
                    response = "I didn't receive a response. Please try again."
            except Exception as exc:
                error_msg = str(exc)
                if "401" in error_msg or "invalid_api_key" in error_msg.lower():
                    response = "Authentication failed. Please check that your Groq API key is correct."
                elif "429" in error_msg or "rate_limit" in error_msg.lower():
                    response = "Rate limit reached. Please wait a moment before sending another message."
                elif "503" in error_msg or "unavailable" in error_msg.lower():
                    response = "The AI service is temporarily unavailable. Please try again shortly."
                else:
                    response = f"An error occurred: {error_msg}"
                st.error(response)

        reply_ts = datetime.now().strftime("%H:%M")
        st.write(response)
        st.caption(reply_ts)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "timestamp": reply_ts
    })
    st.session_state.total_messages += 1
