# Standard libraries
import os
import asyncio
import threading
from dotenv import load_dotenv

# Streamlit
import streamlit as st

# PDF handling
from PyPDF2 import PdfReader

# HuggingFace Transformers
from transformers import pipeline

# LangChain core
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# LangChain community modules
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings

# LangChain HuggingFace integrations
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline, ChatHuggingFace, HuggingFaceEndpoint

# LangChain Google GenAI integrations
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

#Design
from htmlTemplates import css, bot_template, user_template








#loading api key
load_dotenv()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"E:\Langchain_updated\ask-multiple-pdfs-main\gemini_key.json"
api_key = os.environ.get("GOOGLE_API_KEY")



def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

    

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    # embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", transport="rest")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore, api_key):
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        max_new_tokens=512,
        model_kwargs={"temperature": 0.7, "top_k": 50}
    )


    llm = HuggingFacePipeline(pipeline=pipe)



    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        output_key ='answer'
    )

    retriever = vectorstore.as_retriever()
    
    # IMPORTANT: The prompt for this chain's QA step must accept "context" and "question"
    qa_prompt_template = """Use the following pieces of context to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    ----------------
    CONTEXT: {context}
    ----------------
    CHAT HISTORY: {chat_history}
    ----------------
    QUESTION: {question}
    ----------------
    Helpful Answer:"""
    
    QA_PROMPT = PromptTemplate(
        template=qa_prompt_template, input_variables=["context", "question", "chat_history"]
    )

    # Final conversation chain
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        # THIS IS THE FIX: Pass the prompt inside this dictionary
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True
    )

    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content),unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content),unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(
    page_title="Chatbot",
    page_icon="⚡🐭"
)
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Hi, How can I help you today")
    user_question = st.text_input("Drop your socks and grab your Crocs, we're about to get wet on this ride")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore, api_key)


if __name__ == '__main__':
    main()
