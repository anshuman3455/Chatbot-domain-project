# Domain-Specific Chatbot (Statistical Retrieval)

This is a lightweight, high-performance chatbot that identifies user intent (domain) and provides relevant responses using statistical NLP techniques.

### ⚠️ Note: No LLM Used
Unlike modern bots that rely on GPT or other Large Language Models, this project uses **Traditional Machine Learning (TF-IDF)**. 
* **Zero API Costs:** Runs entirely locally with no external dependencies.
* **Deterministic:** High control over responses based on historical data.
* **Efficient:** Extremely low memory and CPU footprint.

---

## 🚀 Features
1. **Domain Classification:** Uses keyword-based heuristics to categorize queries into *Travel*, *Food*, *Fitness*, or *General*.
2. **Similarity Engine:** Uses `TfidfVectorizer` and `Cosine Similarity` to find the most mathematically relevant answer from a local knowledge base.
3. **Modular Architecture:** Separated into Data, Logic, and Application layers.