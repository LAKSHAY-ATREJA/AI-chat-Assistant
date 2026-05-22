"""
demo.py — Command-line demonstration of the AI Chat Assistant.

This script exercises the same LangChain + Groq pipeline that the Streamlit UI
uses, but runs entirely in the terminal so you can verify the setup without a
browser.

Usage:
    python demo.py

The script reads GROQ_API_KEY from the environment (or from a .env file) and
runs a short scripted conversation under each of the five available personas.
"""

import os
import sys
import textwrap
from dotenv import load_dotenv

load_dotenv()

# ── Dependency check ──────────────────────────────────────────
missing = []
try:
    from langchain_groq import ChatGroq
except ImportError:
    missing.append("langchain-groq")

try:
    from langchain.memory import ConversationBufferWindowMemory
    from langchain.chains import ConversationChain
    from langchain.prompts import PromptTemplate
except ImportError:
    missing.append("langchain")

if missing:
    print(f"[ERROR] Missing packages: {', '.join(missing)}")
    print("Install them with:  pip install -r requirements.txt")
    sys.exit(1)

# ── Personas (mirrors app.py) ─────────────────────────────────
PERSONAS = {
    "General Assistant": (
        "You are a helpful, friendly, and knowledgeable AI assistant. "
        "Answer clearly and concisely."
    ),
    "Python Tutor": (
        "You are an expert Python programming tutor. Explain concepts clearly "
        "with code examples. Always include working code snippets."
    ),
    "Writing Coach": (
        "You are a professional writing coach. Help improve writing, suggest "
        "better phrasing, and provide constructive feedback."
    ),
    "Research Assistant": (
        "You are a thorough research assistant. Provide well-structured, "
        "fact-based answers with clear explanations."
    ),
    "Career Advisor": (
        "You are an experienced career advisor specialising in tech roles. "
        "Give practical, actionable career advice."
    ),
}

# ── Demo scenarios ────────────────────────────────────────────
# Each tuple is (persona_name, list_of_user_messages)
DEMO_SCENARIOS = [
    (
        "General Assistant",
        [
            "What is machine learning in one sentence?",
            "Can you give me a real-world example of that?",
        ],
    ),
    (
        "Python Tutor",
        [
            "How do I read a CSV file in Python?",
            "How would I filter rows where the value in column 'age' is greater than 30?",
        ],
    ),
    (
        "Writing Coach",
        [
            "Please improve this sentence: 'The results was very good and people liked it a lot.'",
        ],
    ),
]


def create_chain(api_key: str, persona_prompt: str) -> ConversationChain:
    """Build a ConversationChain with windowed memory."""
    llm = ChatGroq(
        model_name="llama3-8b-8192",
        temperature=0.7,
        max_tokens=512,
        groq_api_key=api_key,
    )
    memory = ConversationBufferWindowMemory(
        k=10,
        human_prefix="User",
        ai_prefix="Assistant",
    )
    template = persona_prompt + "\n\nCurrent conversation:\n{history}\nUser: {input}\nAssistant:"
    prompt = PromptTemplate(input_variables=["history", "input"], template=template)
    return ConversationChain(llm=llm, memory=memory, prompt=prompt, verbose=False)


def wrap(text: str, width: int = 88, indent: str = "    ") -> str:
    """Word-wrap text and indent each line."""
    return textwrap.fill(text, width=width, initial_indent=indent, subsequent_indent=indent)


def separator(char: str = "-", width: int = 72) -> str:
    return char * width


def run_demo(api_key: str) -> None:
    print(separator("="))
    print("  AI Chat Assistant — Demo")
    print("  Model: Groq Llama3-8b-8192   |   Memory window: 10 turns")
    print(separator("="))

    total_turns = 0
    errors = 0

    for persona_name, messages in DEMO_SCENARIOS:
        print(f"\nPersona: {persona_name}")
        print(f"System: {PERSONAS[persona_name][:80]}...")
        print(separator())

        try:
            chain = create_chain(api_key, PERSONAS[persona_name])
        except Exception as exc:
            print(f"[ERROR] Could not initialise chain for '{persona_name}': {exc}")
            errors += 1
            continue

        for user_msg in messages:
            print(f"\nUser:\n{wrap(user_msg)}")
            try:
                response = chain.predict(input=user_msg)
                print(f"\nAssistant:\n{wrap(response)}")
                total_turns += 1
            except Exception as exc:
                print(f"\n[ERROR] {exc}")
                errors += 1

        print(f"\n{separator()}")

    print(f"\nDemo complete — {total_turns} turn(s) run, {errors} error(s).")

    if errors:
        print(
            "\nIf you see authentication errors, make sure GROQ_API_KEY is set to a "
            "valid key from console.groq.com."
        )


def main() -> None:
    api_key = os.getenv("GROQ_API_KEY", "").strip()

    if not api_key:
        print("[ERROR] GROQ_API_KEY is not set.")
        print(
            "Set it in a .env file (copy .env.example) or export it in your shell:\n"
            "    export GROQ_API_KEY=gsk_..."
        )
        sys.exit(1)

    if not api_key.startswith("gsk_"):
        print(f"[WARNING] GROQ_API_KEY does not start with 'gsk_' — it may be invalid.")

    run_demo(api_key)


if __name__ == "__main__":
    main()
