# app.py
import os
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
load_dotenv()

# Streamlit page config MUST be first Streamlit call
st.set_page_config(page_title="Conversational RAG ChatBot", page_icon="📄", layout="wide")


os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
      .stButton > button {border-radius: 12px; padding: 0.6rem 1rem; font-weight: 600;}
      .stTextInput > div > div > input {border-radius: 10px;}
      .stFileUploader label {font-weight: 600;}
      footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)


# Hero section
st.markdown(
    """
    <div style='text-align: center;'>
        <h1 style='color:#4A90E2; margin-bottom: 0;'>📄 Conversational RAG ChatBot</h1>
        <p style='font-size:18px; color:#334155; margin-top: 8px;'>
            Upload PDFs and chat intelligently with their content — powered by LLM + RAG.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# Sidebar settings
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("Groq API Key", type="password", help="Paste your GROQ API key here")
    session_id = st.text_input("Session ID", value="default_session")

    with st.expander("🔧 Advanced Settings"):
        chunk_size = st.number_input("Chunk size", min_value=500, max_value=8000, value=2000, step=100)
        chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=1000, value=200, step=50)
        top_k = st.slider("Retriever top-k", min_value=1, max_value=10, value=4)
        clear = st.button("🧹 Clear Chat History")


# Session state (chat store)
if "store" not in st.session_state:
    st.session_state.store = {}

if clear:
    # Clear only the current session's history
    st.session_state.store.pop(session_id, None)
    st.success("Chat history cleared for this session.")


# LLM init guard
if not api_key:
    st.warning("Please enter the GROQ API Key in the sidebar to begin.")
    st.stop()

llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.3-70b-versatile")


# File upload & indexing
st.subheader("📂 Upload PDFs")
uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

docs = []
if uploaded_files:
    os.makedirs("tmp", exist_ok=True)
    with st.spinner("Processing documents..."):
        for uf in uploaded_files:
            tmp_path = os.path.join("tmp", f"{uuid4()}.pdf")
            with open(tmp_path, "wb") as f:
                f.write(uf.getvalue())
            loader = PyPDFLoader(tmp_path)
            docs.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = text_splitter.split_documents(docs)

    with st.spinner("Creating vector store..."):
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    st.success(f"Indexed {len(splits)} chunks from {len(uploaded_files)} file(s).")
else:
    st.info("Upload one or more PDF files to start chatting.")

# If there are no docs yet, don't build chains
if not docs:
    st.stop()


# RAG prompts & chains
contextualize_system_prompt = (
    "Given a chat history and the latest user question which might reference context in the chat history, "
    "formulate a standalone question that can be understood without the chat history. "
    "Do NOT answer the question; only rewrite it if needed."
)
contextualize_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_prompt)

answer_system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the retrieved context to answer the question. "
    "If the answer isn't in the context, say you don't know. "
    "Keep the answer to at most three sentences.\n\n{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", answer_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


def get_session_history(session: str) -> BaseChatMessageHistory:
    if session not in st.session_state.store:
        st.session_state.store[session] = ChatMessageHistory()
    return st.session_state.store[session]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)


# Chat UI (chat bubbles)
st.subheader("💬 Chat")

session_history = get_session_history(session_id)

# Render past messages
for m in session_history.messages:
    role = "user" if m.type in ("human", "user") else "assistant"
    with st.chat_message(role):
        st.markdown(m.content)

# Chat input
user_input = st.chat_input("Ask about your PDFs...")
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("Thinking..."):
        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}},
        )

    ai_text = response["answer"]
    with st.chat_message("assistant"):
        st.markdown(ai_text)


# Footer
st.markdown(
    """
    <hr style="border:1px solid #e5e7eb">
    <div style="text-align:center; font-size:14px; color:#475569;">
      Built with ❤️ using <b>Streamlit</b> + <b>LangChain</b> + <b>Groq</b> + <b>Chroma</b>
    </div>
    """,
    unsafe_allow_html=True,
)
