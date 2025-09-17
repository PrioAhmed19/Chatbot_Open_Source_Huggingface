
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.18%2B-ff69b4)
![LangChain](https://img.shields.io/badge/LangChain-0.0.184-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
![HuggingFace](https://img.shields.io/badge/Models-HuggingFace-orange)

**Last Updated:** September 14, 2025

A Streamlit app that can **chat with multiple PDF documents** using open-source HuggingFace models. Load PDFs, ask questions, and get context-aware answers backed by vector search.

---

## ✨ Features

- 📂 Upload and process multiple PDF files at once  
- 🧩 Automatic text splitting into overlapping chunks  
- 🔍 Semantic search with FAISS and HuggingFace embeddings  
- 🤖 Q&A powered by open LLMs (e.g. Flan-T5)  
- 💬 Persistent conversation memory  

---

## 🛠️ Tech Stack

| Component          | Tool |
|--------------------|------|
| UI                 | [Streamlit](https://streamlit.io) |
| PDF parsing        | [PyPDF2](https://pypi.org/project/pypdf2/) |
| Text splitting     | LangChain `CharacterTextSplitter` |
| Embeddings         | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store       | [FAISS](https://github.com/facebookresearch/faiss) |
| LLM (open source)  | `google/flan-t5-small` (or any HuggingFace model) |
| Memory             | LangChain `ConversationBufferMemory` |

---

## Installation

1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/pdf-chatbot.git
cd pdf-chatbot

```

2.**Create a virtual environment**

```bash

python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows

```

3.**Install dependencies:**

```bash
pip install -r requirements.txt

```

4.**Set environment variable:**

-GOOGLE_API_KEY=<your_google_api_key>
-GOOGLE_APPLICATION_CREDENTIALS=<path_to_your_gemini_key.json>
-Create .env file


## Usage

### 1. Run the streamlit app

```bash

streamlit run app.py

```

### 2. Upload PDFs in the sidebar and click Process.
### 3. Ask questions in the chat input.
### 4. The chatbot will provide answers based on the PDF contents and maintain conversation history.




## Acknowledgement

This project is built using an updated structure, updated methods as the day stands [14/09/2025], and the open-source HuggingFace models. We would like to give a special shoutout to the tutorial by **Alejandro AO - Software & Ai** ([YouTube link](https://www.youtube.com/watch?v=dXxQ0LR-3Hg&t=3924s)), which inspired the original workflow and provided valuable guidance on PDF handling and conversational AI integration.
