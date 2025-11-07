

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
import os
from io import BytesIO

# PAGE CONFIGURATION
st.set_page_config(page_title="AI Support Agent", page_icon="🤖", layout="wide")
st.title("🤖 AI Knowledge-Based Support Agent")
st.caption("Empowered by NLP & Machine Learning")

# FILE UPLOADER (Knowledge Base)
uploaded_file = st.file_uploader("📂 Upload your Knowledge Base (CSV)", type=["csv"])

@st.cache_data
def load_knowledge_base(file):
    try:
        if file is not None:
            kb = pd.read_csv(file)
            st.success("✅ Custom Knowledge Base uploaded successfully!")
        elif os.path.exists("knowledge_base.csv"):
            kb = pd.read_csv("knowledge_base.csv")
            st.success("✅ Default Knowledge Base loaded successfully!")
        else:
            raise FileNotFoundError
    except Exception:
        st.warning("⚠️ No KB file found. Using built-in dataset instead.")
        kb = pd.DataFrame({
            "Intent": ["Payment Issue", "App Crash", "Feedback"],
            "Question": [
                "My payment failed but money got deducted",
                "The app keeps crashing on my phone",
                "How can I share feedback?"
            ],
            "Response": [
                "We’re sorry for the issue. Refund will be processed within 5–7 business days.",
                "Please clear cache, update the app, or reinstall it to fix the crash issue.",
                "You can share your feedback in the Help & Feedback section of the app."
            ],
            "Persona": ["Customer", "Technical", "Customer"],
            "Tone": ["Professional", "Empathetic", "Appreciative"]
        })
    return kb

kb = load_knowledge_base(uploaded_file)

# TEXT VECTORIZATION USING TF-IDF
@st.cache_resource
def train_vectorizer(data):
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform(data)
    return vectorizer, vectors

vectorizer, kb_vectors = train_vectorizer(kb["Question"])

# PERSONA DETECTION
def detect_persona(text):
    text = text.lower()
    if any(word in text for word in ["bug", "crash", "error", "technical", "slow"]):
        return "Technical"
    elif any(word in text for word in ["refund", "payment", "order", "discount", "delay"]):
        return "Customer"
    elif any(word in text for word in ["developer", "api", "backend", "code"]):
        return "Developer"
    else:
        return "User"

# RESPONSE RETRIEVAL
def get_best_response(user_query):
    user_vec = vectorizer.transform([user_query])
    similarity = cosine_similarity(user_vec, kb_vectors)
    best_idx = np.argmax(similarity)
    confidence = similarity[0][best_idx]
    return kb.iloc[best_idx], confidence

# CHAT HISTORY STORAGE
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# MAIN APP LOGIC
st.divider()
user_query = st.text_input("💬 Ask your question:", placeholder="e.g., My payment failed but money was deducted")

if user_query:
    with st.spinner("Analyzing your query... ⏳"):
        time.sleep(1.5)
        persona = detect_persona(user_query)
        result, confidence = get_best_response(user_query)

    st.divider()
    col1, col2 = st.columns([2, 1])

    # LEFT COLUMN - RESPONSE
    with col1:
        st.subheader("🎯 AI Agent Response")
        st.success(result["Response"])
        st.write("**Intent:**", result["Intent"])
        st.write("**Tone:**", result["Tone"])
        st.write(f"**Confidence Score:** {confidence*100:.2f}%")

    # RIGHT COLUMN - PERSONA
    with col2:
        st.subheader("🧠 Detected Persona")
        if persona == "Customer":
            st.info("Persona: Customer 🧍‍♀️ — Use empathetic and polite tone.")
        elif persona == "Technical":
            st.warning("Persona: Technical 👨‍💻 — Use concise and clear tone.")
        elif persona == "Developer":
            st.success("Persona: Developer 🧑‍💻 — Use precise and data-based tone.")
        else:
            st.write("Persona: User 🙂 — General helpful tone.")

    # Save chat to history
    st.session_state.chat_history.append({
        "User Query": user_query,
        "Detected Persona": persona,
        "Intent": result["Intent"],
        "Tone": result["Tone"],
        "Response": result["Response"],
        "Confidence": round(confidence*100, 2)
    })

# CHAT HISTORY TABLE & DOWNLOAD OPTION

if st.session_state.chat_history:
    st.divider()
    st.subheader("📜 Chat History")
    chat_df = pd.DataFrame(st.session_state.chat_history)
    st.dataframe(chat_df, use_container_width=True)

    # Convert to CSV for download
    csv = chat_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download Chat History (CSV)",
        data=csv,
        file_name="chat_history.csv",
        mime="text/csv"
    )

# FOOTER
st.divider()
st.write("🚀 Built with Streamlit + Python NLP")
st.write("👨‍💻 : **Vijul Tyagi** | Company Assignment Project")
