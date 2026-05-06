import os
from groq import Groq
from data import history   


GROQ_API_KEY = "Api key from groq"   
client = Groq(api_key=GROQ_API_KEY)
MODEL = "llama-3.1-8b-instant"            


def parse_history(raw: list) -> list:
    """
    Converts your history strings like:
      "User: Hi there! | Bot: Hello!"
    into LLM message dicts:
      [{"role": "user", "content": "Hi there!"},
       {"role": "assistant", "content": "Hello!"}]
    These become few-shot examples the LLM learns tone/style from.
    """
    parsed = []
    for item in raw:
        try:
            user_part, bot_part = item.split("|")
            user_text = user_part.replace("User:", "").strip()
            bot_text  = bot_part.replace("Bot:", "").strip()
            parsed.append({"role": "user",      "content": user_text})
            parsed.append({"role": "assistant", "content": bot_text})
        except ValueError:
            continue  
    return parsed

FEW_SHOT_EXAMPLES = parse_history(history)


DOMAIN_SYSTEM_PROMPTS = {
    "travel": (
        "You are an expert travel advisor agent. "
        "You help users plan trips, suggest destinations, give packing tips, "
        "recommend itineraries, and share travel hacks. "
        "Be enthusiastic, practical, and specific with your advice. "
        "The example conversation below shows the tone and style you should follow."
    ),
    "food": (
        "You are a professional chef and food consultant agent. "
        "You suggest recipes, meal plans, cooking techniques, restaurant picks, "
        "and dietary advice. Be creative, descriptive, and helpful. "
        "The example conversation below shows the tone and style you should follow."
    ),
    "fitness": (
        "You are a certified personal trainer and fitness coach agent. "
        "You create workout plans, explain exercises, give nutrition advice for training, "
        "and motivate users. Be encouraging and precise. "
        "The example conversation below shows the tone and style you should follow."
    ),
    "general": (
        "You are a helpful, friendly, and knowledgeable AI assistant. "
        "Answer questions clearly and conversationally across any topic. "
        "The example conversation below shows the tone and style you should follow."
    ),
}


DOMAIN_KEYWORDS = {
    "travel":  ["travel", "trip", "destination", "packing", "flight", "hotel",
                 "itinerary", "vacation", "tour", "visa", "passport", "sightseeing"],
    "food":    ["food", "lunch", "dinner", "breakfast", "recipe", "dessert",
                 "cook", "eat", "restaurant", "meal", "ingredient", "cuisine", "snack"],
    "fitness": ["exercise", "fitness", "cardio", "training", "workout", "gym",
                 "run", "weight", "muscle", "yoga", "sport", "calories", "strength"],
}

def classify_domain(text):
    """Fast keyword-based domain classifier."""
    text_lower = text.lower()
    scores = {domain: 0 for domain in DOMAIN_KEYWORDS}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                scores[domain] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


conversation_memory = []
MAX_MEMORY_TURNS = 10   


def build_messages(domain, user_input):
    """
    Full message list sent to Groq:

      [system prompt]          <- domain agent persona
      [few-shot examples]      <- parsed from your data.py history
      [live conversation]      <- what was said this session (memory)
      [current user message]   <- the new query
    """
    messages = [{"role": "system", "content": DOMAIN_SYSTEM_PROMPTS[domain]}]

    messages.extend(FEW_SHOT_EXAMPLES)

    trimmed_memory = conversation_memory[-(MAX_MEMORY_TURNS * 2):]
    messages.extend(trimmed_memory)

    messages.append({"role": "user", "content": user_input})
    return messages


def get_response(user_input):
    """
    Main agent entry point.
    Returns {"domain": str, "response": str}
    """
    domain   = classify_domain(user_input)
    messages = build_messages(domain, user_input)

    chat_completion = client.chat.completions.create(
        messages=messages,
        model=MODEL,
        temperature=0.7,
        max_tokens=512,
    )
    response_text = chat_completion.choices[0].message.content.strip()

    conversation_memory.append({"role": "user",      "content": user_input})
    conversation_memory.append({"role": "assistant", "content": response_text})

    return {"domain": domain, "response": response_text}


def clear_memory():
    """Reset live conversation history (few-shot examples are kept)."""
    conversation_memory.clear()


def get_memory_snapshot():
    """Return current live memory (for debugging)."""
    return list(conversation_memory)