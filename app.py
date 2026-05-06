# app.py
from chatbot_logic import get_response

def main():
    print("--- Domain-Aware Chatbot Loaded ---")
    print("(Type 'exit' to quit)")
    
    while True:
        query = input("\nUser: ")
        if query.lower() == "exit":
            print("Goodbye!")
            break

        result = get_response(query)

        print(f"Domain Identified: {result['domain']}")
        print(f"Bot: {result['response']}")

if __name__ == "__main__":
    main()