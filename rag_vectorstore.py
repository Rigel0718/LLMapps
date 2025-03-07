import streamlit as st

from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from typing import List
import time

def load_documents_chroma_vectorstore(docs : List[Document]) -> Chroma :
    embedding = OpenAIEmbeddings(api_key=st.session_state.openai_api_key)
    vector_db = Chroma.from_documents(
        documents=docs,
        embedding=embedding,
        persist_directory="./chroma_db",
        collection_name=f"{str(time.time()).replace('.', '')[:14]}_" + st.session_state['session_id']
    )
    
    return vector_db

def load_documents_faiss_vectorsotre(docs : List[Document]) -> FAISS :
    embedding = OpenAIEmbeddings(api_key=st.session_state.openai_api_key)
    vector_db = FAISS.from_documents(
        documents=docs,
        embedding=embedding
    )
    return vector_db