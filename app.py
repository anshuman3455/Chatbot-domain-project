

from chatbot_logic import get_response, clear_memory, get_memory_snapshot

BANNER = """
╔══════════════════════════════════════════════════════════╗
║          Domain-Aware Agentic Chatbot (Groq)             ║
║  Domains: travel · food · fitness · general              ║
║  Commands: 'exit' · 'clear' (reset memory) · 'memory'    ║
╚══════════════════════════════════════════════════════════╝
"""

def main():
    print(BANNER)

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue

        if query.lower() == "exit":
            print("Goodbye!")
            break

        if query.lower() == "clear":
            clear_memory()
            print("[Memory cleared. Fresh conversation started.]\n")
            continue

        if query.lower() == "memory":
            snapshot = get_memory_snapshot()
            if not snapshot:
                print("[No conversation history yet.]\n")
            else:
                print("\n── Conversation Memory ──")
                for msg in snapshot:
                    role = "You" if msg["role"] == "user" else "Bot"
                    print(f"  {role}: {msg['content'][:120]}{'...' if len(msg['content']) > 120 else ''}")
                print()
            continue

        try:
            result = get_response(query)
            print(f"\n[Domain: {result['domain'].upper()}]")
            print(f"Bot: {result['response']}\n")
        except Exception as e:
            print(f"[Error calling Groq API: {e}]\n")


if __name__ == "__main__":
    main()