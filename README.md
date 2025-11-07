# 🤖 AI Support Agent - Streamlit + NLP (Enhanced Version)

## Description
Smart AI Agent that provides knowledge-based support with features like:

- Custom Knowledge Base upload (CSV)
- Persona Detection (Customer, Technical, Developer, User)
- AI-generated responses using TF-IDF & Cosine Similarity
- Chat History Storage & Download
- Tone & Intent Display for responses

Built using **Python**, **Streamlit**, **pandas**, **NumPy**, and **scikit-learn**.

---

## Features

1. **Knowledge Base Upload**
   - Upload your own CSV file for a custom knowledge base.
   - Default built-in dataset is used if no file is uploaded.

2. **TF-IDF & Cosine Similarity**
   - Converts questions into vectorized form.
   - Finds the most relevant response based on similarity.

3. **Persona Detection**
   - Automatically detects the type of user query:
     - **Customer**: Empathetic & polite tone
     - **Technical**: Concise & clear tone
     - **Developer**: Precise & data-based tone
     - **User**: General helpful tone

4. **Chat History**
   - Stores all user queries and AI responses.
   - Display in a table and download as CSV.

5. **Confidence Score**
   - Displays how confident the AI is about the suggested response.

---


