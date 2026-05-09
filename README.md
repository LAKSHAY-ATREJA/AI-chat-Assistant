# 🤖 AI Chat Assistant with Memory

A feature-rich conversational AI assistant with persistent memory, multiple personas, and conversation export.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-latest-red)
![LangChain](https://img.shields.io/badge/LangChain-latest-green)

## 🚀 Features

- **Conversation Memory** — Remembers last 15 exchanges for context-aware responses
- **5 AI Personas** — General Assistant, Python Tutor, Writing Coach, Research Assistant, Career Advisor
- **Session Statistics** — Track messages sent per session
- **Export Chat** — Download full conversation as a text file
- **Instant Persona Switch** — Change AI behaviour with one click

## 🏗️ Architecture

```
User Input → ConversationChain
    → ConversationBufferWindowMemory (k=15)
    → Persona-specific PromptTemplate
    → Groq LLM (Llama3-8b) → Response
    → Memory Updated → Display
```

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| UI | Streamlit |
| LLM | Groq (Llama3-8b-8192) |
| Memory | LangChain ConversationBufferWindowMemory |
| Orchestration | LangChain ConversationChain |

## ⚡ Quick Start

```bash
git clone https://github.com/LAKSHAY-ATREJA/ai-chat-assistant
cd ai-chat-assistant

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

streamlit run app.py
```

## 🎭 Available Personas

| Persona | Best For |
|---|---|
| 🧠 General Assistant | Everyday questions and tasks |
| 💻 Python Tutor | Learning Python with code examples |
| 📝 Writing Coach | Improving your writing |
| 🔬 Research Assistant | Detailed research questions |
| 💼 Career Advisor | Tech career guidance |
