"""
Streamlit Chat UI — RAG over Excel (OpenAI + LangChain)

Run:
streamlit run app.py
"""

import os
import streamlit as st
from dotenv import load_dotenv
from rag_pipeline import build_pipeline

# Load environment variables
load_dotenv()

# Get API key from .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("❌ OPENAI_API_KEY not found. Please add it to your .env file.")
    st.stop()

# ─────────────────────────────
# Page config
# ─────────────────────────────

st.set_page_config(
    page_title="Excel RAG Chat",
    page_icon="📊",
    layout="centered",
)

# ─────────────────────────────
# Custom CSS
# ─────────────────────────────

st.markdown("""
<style>

#MainMenu {visibility:hidden;}
footer {visibility:hidden;}

.stChatMessage {
    border-radius: 10px;
}

section[data-testid="stSidebar"] {
    background-color: #0f1117;
}

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────
# Sidebar
# ─────────────────────────────

with st.sidebar:

    st.title("📊 Excel RAG")
    st.caption("Powered by OpenAI + LangChain")

    st.divider()

    uploaded_file = st.file_uploader(
        "Upload Excel File",
        type=["xlsx", "xls"]
    )

    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.session_state.chain = None
        st.rerun()

    st.divider()
    st.caption("LangChain · FAISS · GPT-4o-mini")

# ─────────────────────────────
# Session State
# ─────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chain" not in st.session_state:
    st.session_state.chain = None

if "last_file" not in st.session_state:
    st.session_state.last_file = None

# ─────────────────────────────
# Main UI
# ─────────────────────────────

st.title("Chat with your Excel 📊")

# Build pipeline when file uploaded
if uploaded_file:

    file_key = f"{uploaded_file.name}_{uploaded_file.size}"

    if st.session_state.last_file != file_key:

        temp_path = f"/tmp/{uploaded_file.name}"

        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        with st.spinner("Indexing Excel file..."):

            try:
                st.session_state.chain = build_pipeline(
                    temp_path,
                    OPENAI_API_KEY
                )

                st.session_state.last_file = file_key
                st.session_state.messages = []

                st.success("Excel indexed successfully!")

            except Exception as e:
                st.error(f"Error building index: {e}")

else:
    st.info("Upload an Excel file to start chatting.")

# ─────────────────────────────
# Chat history
# ─────────────────────────────

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ─────────────────────────────
# Chat Input
# ─────────────────────────────

if prompt := st.chat_input(
    "Ask questions about your Excel data...",
    disabled=st.session_state.chain is None
):

    # user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    # assistant response
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                session_cfg = {
                    "configurable": {
                        "session_id": "streamlit-session"
                    }
                }

                answer = st.session_state.chain.invoke(
                    {"question": prompt},
                    config=session_cfg
                )

            except Exception as e:
                answer = f"⚠️ Error: {e}"

        st.markdown(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })