# 🧠 **Conversational RAG PDF Chatbot**  
*A Streamlit app that lets you upload PDFs and chat with them using a Retrieval-Augmented Generation (RAG) pipeline — with full chat history memory.*


## ✨ Features

- 📄 **Upload Multiple PDFs** – Ask questions across one or more documents at once.
- 🔍 **Retrieval-Augmented Generation (RAG)** – Combines local document search with LLM reasoning.
- 💬 **Persistent Chat History** – Keeps track of past conversation context by session ID.
- ⚡ **Plug-and-Play LLM** – Uses [Groq’s](https://groq.com/) ultra-fast `llama-3.3-70b-versatile` model.
- 🧩 **Custom Embeddings** – Uses Hugging Face `all-MiniLM-L6-v2` embeddings.
- 🖥️ **Simple UI** – Streamlit app with clean user inputs and outputs.

---

## 🏗️ Tech Stack

- [**Streamlit**](https://streamlit.io/) – web app framework  
- [**LangChain**](https://www.langchain.com/) – RAG pipeline and message history  
- [**Chroma**](https://www.trychroma.com/) – vector database for document retrieval  
- [**Hugging Face**](https://huggingface.co/) – embeddings (`all-MiniLM-L6-v2`)  
- [**Groq API**](https://groq.com/) – LLM inference  
- Python 3.8+  

---

## 🚀 Quick Start

### 1️⃣ Clone the Repository
- git clone https://github.com/your-username/conversational-rag-pdf-chatbot.git
- cd conversational-rag-pdf-chatbot


### 2️⃣ Create & Activate Virtual Environment
- python3 -m venv venv
- source venv/bin/activate   # On Mac/Linux
- venv\Scripts\activate      # On Windows


### 3️⃣ Install Requirements
- pip install -r requirements.txt


### 4️⃣ Set Up Environment Variables
- HF_TOKEN=your_huggingface_api_token
- GROQ_API_KEY=your_groq_api_key


### 5️⃣ Run the App
- streamlit run app.py


## 🧩 How It Works

- PDF Loading → Uses PyPDFLoader to extract text.
- Text Splitting → Chunks documents with RecursiveCharacterTextSplitter.
- Vector Store Creation → Stores embeddings in Chroma.
- History-Aware Retriever → Reformulates user questions using previous chat context.
- LLM Answering → Sends retrieved context + question to Groq’s llama-3.3-70b-versatile.
- Session Memory → ChatMessageHistory keeps per-session conversations.

## 🖥️ Demo
- ![App Screenshot](/Users/pahul17/Documents/End-End-Projects/4 - RAG Q&A With chat History/images/demo.png)

## ⚡ Future Enhancements

- Support for multiple LLM providers (OpenAI, Anthropic).
- Add document type detection (Word, CSV, HTML).
- Option to download chat history as PDF/CSV.
- Deploy on Streamlit Cloud / Hugging Face Spaces.


## 💡 Tips to Customize
- Replace `Conversational RAG PDF Chatbot` with your final repo name.  
- Add a **demo GIF** or Streamlit app link if you deploy it.  
- Update the **Future Enhancements** list with your personal roadmap.  
- Add badges (deployment, Python version) if you want a more polished look.

