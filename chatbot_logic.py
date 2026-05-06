# chatbot_logic.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from data import history

def prepare_data():
    user_queries = []
    bot_responses = []
    for item in history:
        parts = item.split("|")
        user_queries.append(parts[0].replace("User:", "").strip())
        bot_responses.append(parts[1].replace("Bot:", "").strip())
    return user_queries, bot_responses

def classify_domain(text):
    text = text.lower()
    if any(word in text for word in ["travel", "trip", "destination", "packing"]):
        return "travel"
    elif any(word in text for word in ["food", "lunch", "dinner", "recipe", "dessert"]):
        return "food"
    elif any(word in text for word in ["exercise", "fitness", "cardio", "training"]):
        return "fitness"
    return "general"

# Initialize global data
user_queries, bot_responses = prepare_data()
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(user_queries)

def get_response(user_input):
    domain = classify_domain(user_input)
    input_vec = vectorizer.transform([user_input])
    similarity = cosine_similarity(input_vec, X)
    index = similarity.argmax()
    
    return {
        "domain": domain,
        "response": bot_responses[index]
    }