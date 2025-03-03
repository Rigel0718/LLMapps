import streamlit as st

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from typing import List
import time

def load_documents_chroma_vectorstore(docs : List[Document]) -> Chroma :
    embedding = OpenAIEmbeddings(api_key=st.session_state.openai_api_key)
    vector_db = Chroma.from_documents(
        documents=docs,
        embedding=embedding,
        persist_directory="./chroma_db", # default로 None이지만, 세션 종료하면 초기화 되는 것을 코드상 확실히 하기 위해서
        collection_name=f"{str(time.time()).replace('.', '')[:14]}_" + st.session_state['session_id']
    )
    
    return vector_db