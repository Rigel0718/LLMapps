import streamlit as st

from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from typing import List

def load_documents_chroma_vectorstore(docs : List[Document]) -> Chroma :
    embedding = OpenAIEmbeddings(api_key=st.session_state.openai_api_key)
    vector_db = Chroma.from_documents(
        documents=docs,
        embedding=embedding,
        persist_directory=None, # default로 None이지만, 세션 종료하면 초기화 되는 것을 코드상 확실히 하기 위해서
    )
    
    return vector_db