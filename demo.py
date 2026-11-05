"""Terminal demonstration of the same LangChain/Groq pipeline as the Streamlit app.

Requires GROQ_API_KEY. Conversations retain the latest ten non-system messages.
"""
from __future__ import annotations

import os
import sys
import textwrap

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()
MODEL_NAME = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b").strip() or "openai/gpt-oss-20b"

PERSONAS = {
    "General Assistant": "You are a helpful, friendly, and knowledgeable AI assistant. Answer clearly and concisely.",
    "Python Tutor": "You are an expert Python programming tutor. Explain concepts clearly with code examples. Always include working code snippets.",
    "Writing Coach": "You are a professional writing coach. Help improve writing, suggest better phrasing, and provide constructive feedback.",
    "Research Assistant": "You are a thorough research assistant. Provide well-structured, fact-based answers with clear explanations.",
    "Career Advisor": "You are an experienced career advisor specialising in tech roles. Give practical, actionable career advice.",
}

DEMO_SCENARIOS = [
    ("General Assistant", [
        "What is machine learning in one sentence?",
        "Can you give me a real-world example of that?",
    ]),
    ("Python Tutor", [
        "How do I read a CSV file in Python?",
        "How would I filter rows where the value in column 'age' is greater than 30?",
    ]),
    ("Writing Coach", [
        "Please improve this sentence: 'The results was very good and people liked it a lot.'",
    ]),
]


def wrap(message: str) -> str:
    return textwrap.fill(str(message), width=88, initial_indent="    ", subsequent_indent="    ")


def run_demo(api_key: str) -> int:
    llm = ChatGroq(
        model_name=MODEL_NAME,
        temperature=0.7,
        max_tokens=512,
        groq_api_key=api_key,
    )
    print("=" * 72)
    print(f" AI Chat Assistant | Groq model: {MODEL_NAME} | Recent context: 10 messages")
    print("=" * 72)
    errors = 0
    for persona_name, user_prompts in DEMO_SCENARIOS:
        print(f"\nPersona: {persona_name}\n" + "-" * 72)
        system = SystemMessage(content=PERSONAS[persona_name])
        recent_messages = []
        for prompt in user_prompts:
            print("\nUser:\n" + wrap(prompt))
            user_message = HumanMessage(content=prompt)
            try:
                response = llm.invoke([system, *recent_messages[-10:], user_message])
                answer = str(response.content)
                print("\nAssistant:\n" + wrap(answer))
                recent_messages.extend((user_message, response))
            except Exception as exc:
                # Keep failed exchanges out of conversation history.
                errors += 1
                if "model_not_found" in str(exc) or "does not exist" in str(exc):
                    print("[ERROR] Model unavailable. Set GROQ_MODEL to a current Groq model.")
                else:
                    print(f"[ERROR] Request failed: {type(exc).__name__}")
        print("-" * 72)
    print(f"\nDemo finished with {errors} error(s).")
    return errors


def main() -> None:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        print("[ERROR] GROQ_API_KEY is missing. Add it to a .env file or your environment.")
        sys.exit(1)
    sys.exit(1 if run_demo(api_key) else 0)


if __name__ == "__main__":
    main()
