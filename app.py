import streamlit as st
import pandas as pd
import os
import re
import datetime

# Import models
from models.tfidf import TFIDFModel
from models.embedding import EmbeddingModel

# -----------------
# Utility Functions
# -----------------
def is_amharic(text):
    # Determine if text has Amharic characters (Ethiopic script range)
    ethiopian_chars = re.findall(r'[\u1200-\u137F]', text)
    if ethiopian_chars:
        return True
    return False

def save_feedback(query, model_used, feedback):
    with open("evaluation/feedback.csv", "a", encoding="utf-8") as f:
        f.write(f'"{datetime.datetime.now().isoformat()}","{query}","{model_used}","{feedback}"\n')

def log_evaluation(query, tfidf_res, embed_res):
    file_path = "evaluation/results.csv"
    if not os.path.exists(file_path):
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("timestamp,query,tfidf_answer,tfidf_score,embedding_answer,embedding_score\n")
            
    with open(file_path, "a", encoding="utf-8") as f:
        tfidf_ans = tfidf_res[0]['answer'].replace('"', '""') if tfidf_res else ""
        tfidf_score = tfidf_res[0]['score'] if tfidf_res else 0
        embed_ans = embed_res[0]['answer'].replace('"', '""') if embed_res else ""
        embed_score = embed_res[0]['score'] if embed_res else 0
        query_clean = query.replace('"', '""')
        f.write(f'"{datetime.datetime.now().isoformat()}","{query_clean}","{tfidf_ans}",{tfidf_score},"{embed_ans}",{embed_score}\n')

# -----------------
# State Initialization
# -----------------
st.set_page_config(page_title="Amharic AI Student Assistant", page_icon="🌍", layout="wide")

@st.cache_resource(show_spinner="Loading Models (this may take a minute)...")
def load_models():
    # Load both models heavily utilizing caching
    return TFIDFModel(), EmbeddingModel()

# Custom CSS for aesthetics
st.markdown("""
<style>
div.stButton > button:first-child {
    border-radius: 8px;
    background-color: #262730;
    color: white;
    transition: all 0.2s ease-in-out;
}
div.stButton > button:first-child:hover {
    border-color: #4CAF50;
    color: #4CAF50;
}
</style>
""", unsafe_allow_html=True)

tfidf_model, embedding_model = load_models()

if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------
# Sidebar Design
# -----------------
with st.sidebar:
    st.title("⚙️ Model Controls")
    
    st.markdown("### Settings")
    eval_mode = st.toggle("📊 Evaluation Mode", help="Log queries and results to CSV")
    
    st.markdown("### Selection")
    model_choice = st.radio("Choose Model:", ["TF-IDF", "Embedding", "Compare Both"])
    
    st.markdown("---")
    st.markdown("### 💡 Example Queries")
    
    # Store clicked example in session state to be processed in main
    example_clicked = None
    examples = [
        "Who is considered an international student for university admission purposes?",
        "Do universities offer application fee waivers for international students?",
        "How can I apply for scholarships?",
        "የአለም አቀፍ ተማሪ ማን ነው?" # "Who is an international student?" in Amharic
    ]
    for ex in examples:
        if st.button(ex):
            example_clicked = ex
            
    st.markdown("---")
    st.info("🌍 **Amharic AI Student Assistant**\n\nSupports English & Amharic educational questions.")

# -----------------
# Main Interface
# -----------------
st.title("Amharic AI Student Assistant 💬")
st.markdown("*Compare baseline and modern NLP retrieval methods for educational queries.*")

# Display chat history
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)
        if msg["role"] == "assistant" and "feedback_done" not in msg:
            cols = st.columns([1, 1, 8])
            with cols[0]:
                if st.button("👍", key=f"up_{idx}"):
                    save_feedback(st.session_state.messages[idx-1]["content"] if idx>0 else "", model_choice, "Helpful")
                    st.session_state.messages[idx]["feedback_done"] = True
                    st.rerun()
            with cols[1]:
                if st.button("👎", key=f"down_{idx}"):
                    save_feedback(st.session_state.messages[idx-1]["content"] if idx>0 else "", model_choice, "Not helpful")
                    st.session_state.messages[idx]["feedback_done"] = True
                    st.rerun()

# Handle input
user_input = st.chat_input("Ask a question about admissions, scholarships, etc...")

# If an example was clicked, use it as input
if example_clicked:
    user_input = example_clicked

if user_input:
    # 1. Output the user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
        
    detected_lang = "Amharic 🇪🇹" if is_amharic(user_input) else "English 🇬🇧"
    
    # 2. Retrieval
    with st.spinner("Searching for the best answers..."):
        tfidf_res = tfidf_model.retrieve(user_input)
        embed_res = embedding_model.retrieve(user_input)
    
    if eval_mode:
        log_evaluation(user_input, tfidf_res, embed_res)

    def format_single_result(res_list, name):
        top = res_list[0]
        content = f"*(Detected Language: **{detected_lang}**)*\n\n"
        if top['confidence'] == "Low":
            content += "⚠️ *I'm not fully confident. Here are the closest answers.*\n\n"
        
        content += f"**Answer:** {top['answer']}\n\n"
        content += f"**Match Score:** `{top['score']:.4f}` | **Confidence:** `{top['confidence']}`\n\n"
        
        # Expandable top k
        content += f"<details><summary>🔍 View Top 3 matches ({name})</summary>\n\n<ol>"
        for r in res_list:
            content += f"<li>{r['answer']} <i>(Score: {r['score']:.4f})</i></li>"
        content += "</ol></details>"
        return content

    # 3. Formulate output based on model choice
    response_content = ""
    if model_choice == "TF-IDF":
        response_content = format_single_result(tfidf_res, "TF-IDF")
    elif model_choice == "Embedding":
        response_content = format_single_result(embed_res, "Embedding")
    elif model_choice == "Compare Both":
        response_content += f"*(Detected Language: **{detected_lang}**)*\n\n"
        
        t_top = tfidf_res[0]
        e_top = embed_res[0]
        
        if t_top['confidence'] == "Low" and e_top['confidence'] == "Low":
            response_content += "⚠️ *I'm not fully confident with either model. Here are the closest answers.* \n\n"
        
        # Two columns for comparison
        response_content += "<div style='display: flex; gap: 20px;'>"
        
        # TF-IDF Col
        response_content += "<div style='flex: 1; padding: 15px; border-radius: 10px; background: rgba(255, 255, 255, 0.05);'>"
        response_content += "<h4>📊 TF-IDF Baseline</h4>"
        response_content += f"<p><b>Answer:</b> {t_top['answer']}</p>"
        response_content += f"<p><b>Score:</b> <code>{t_top['score']:.4f}</code> | <b>Confidence:</b> <code>{t_top['confidence']}</code></p>"
        response_content += f"<details><summary>Top 3 Matches</summary><ul>"
        for r in tfidf_res:
            response_content += f"<li>{r['answer']} <i>({r['score']:.4f})</i></li>"
        response_content += "</ul></details></div>"
        
        # Embedding Col
        response_content += "<div style='flex: 1; padding: 15px; border-radius: 10px; background: rgba(255, 255, 255, 0.05);'>"
        response_content += "<h4>🧠 Sentence Embedding</h4>"
        response_content += f"<p><b>Answer:</b> {e_top['answer']}</p>"
        response_content += f"<p><b>Score:</b> <code>{e_top['score']:.4f}</code> | <b>Confidence:</b> <code>{e_top['confidence']}</code></p>"
        response_content += f"<details><summary>Top 3 Matches</summary><ul>"
        for r in embed_res:
            response_content += f"<li>{r['answer']} <i>({r['score']:.4f})</i></li>"
        response_content += "</ul></details></div>"
        
        response_content += "</div>"

    st.session_state.messages.append({"role": "assistant", "content": response_content})
    st.rerun()

