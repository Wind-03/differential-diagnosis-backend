"""CLI entry point for the conversational assistant."""
from __future__ import annotations

import uuid

from app.agent.factory import create_conversational_agent


def run_chat_session() -> None:
    session_id = str(uuid.uuid4())[:8]
    agent = create_conversational_agent()

    print("=" * 60)
    print("🏥 AI Conversational Clinical Assistant")
    print(f"Session ID: {session_id}")
    print("Type 'quit' or 'exit' to end the session.")
    print("=" * 60)
    print("\nAI: Hello! How are you feeling today? Please describe any symptoms you are experiencing.")

    while True:
        try:
            user_input = input("\nUser: ").strip()
            if not user_input:
                continue
            if user_input.lower() in {"quit", "exit", "q"}:
                print("\nAI: Take care. Please consult a physician if your symptoms persist.")
                return

            response = agent.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}},
            )
            print(f"\nAI: {response.get('output', response)}\n")
        except KeyboardInterrupt:
            print("\nExiting session...")
            return
        except Exception as exc:
            print(f"\n❌ Error: {exc}")


if __name__ == "__main__":
    run_chat_session()
