import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from datetime import datetime

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
</style>
""", unsafe_allow_html=True)

# ── Personas ──────────────────────────────────────────────────
PERSONAS = {
    "🧠 General Assistant": "You are a helpful, friendly, and knowledgeable AI assistant. Answer clearly and concisely.",
    "💻 Python Tutor": "You are an expert Python programming tutor. Explain concepts clearly with code examples. Always include working code snippets.",
    "📝 Writing Coach": "You are a professional writing coach. Help improve writing, suggest better phrasing, and provide constructive feedback.",
    "🔬 Research Assistant": "You are a thorough research assistant. Provide well-structured, fact-based answers with clear explanations.",
    "💼 Career Advisor": "You are an experienced career advisor specialising in tech roles. Give practical, actionable career advice."
}

# ── Session state ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = None
if "total_messages" not in st.session_state:
    st.session_state.total_messages = 0
if "current_persona" not in st.session_state:
    st.session_state.current_persona = "🧠 General Assistant"


def create_conversation(api_key, persona_prompt):
    llm = ChatGroq(
        model_name="llama3-8b-8192",
        temperature=0.7,
        max_tokens=1024,
        groq_api_key=api_key
    )

    memory = ConversationBufferWindowMemory(k=15, human_prefix="User", ai_prefix="Assistant")

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


# ── UI ────────────────────────────────────────────────────────
st.markdown('<p class="main-header">🤖 AI Chat Assistant</p>', unsafe_allow_html=True)
st.markdown("Intelligent conversation with memory — remembers the last 15 exchanges.")

with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")

    st.divider()
    st.header("🎭 Select Persona")
    for persona_name in PERSONAS:
        if st.button(persona_name, use_container_width=True,
                     type="primary" if persona_name == st.session_state.current_persona else "secondary"):
            st.session_state.current_persona = persona_name
            st.session_state.messages = []
            st.session_state.memory = None
            st.rerun()

    st.divider()
    st.header("📊 Session Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Messages", len(st.session_state.messages))
    with col2:
        st.metric("Total Sent", st.session_state.total_messages)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.messages = []
            st.session_state.memory = None
            st.rerun()
    with col2:
        if st.button("💾 Export", use_container_width=True):
            if st.session_state.messages:
                export_text = "\n\n".join([
                    f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
                    for m in st.session_state.messages
                ])
                st.download_button(
                    "Download",
                    export_text,
                    file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"
                )

if not api_key:
    st.info("👈 Enter your Groq API key to start chatting.")
else:
    # Active persona display
    persona_desc = PERSONAS[st.session_state.current_persona]
    st.info(f"**Active Persona:** {st.session_state.current_persona}")

    # Initialise conversation
    if st.session_state.memory is None:
        st.session_state.conversation = create_conversation(api_key, persona_desc)

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "timestamp" in message:
                st.caption(message["timestamp"])

    # Input
    user_input = st.chat_input("Type your message...")
    if user_input:
        timestamp = datetime.now().strftime("%H:%M")
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": timestamp
        })

        with st.chat_message("user"):
            st.write(user_input)
            st.caption(timestamp)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.conversation.predict(input=user_input)
            st.write(response)
            st.caption(datetime.now().strftime("%H:%M"))

        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now().strftime("%H:%M")
        })

        st.session_state.total_messages += 1
