from typing import Optional, List, Any, Union
from langchain_core.language_models import BaseLanguageModel
from langchain_core.documents import Document
from langchain_text_splitters.base import TextSplitter
from langchain_core.runnables import Runnable
from .vectorstore.vectorstore import load_documents_faiss_vectorsotre
from .retriever.retriever import get_retrievered_documents
from langchain_community.document_loaders.base import BaseLoader
from langchain_core.vectorstores.base import VectorStore
from dataloader import FlexibleFileLoader, web_loader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

class Custom_RAGPipeline:
    def __init__(
        self,
        data_loader : Union[BaseLoader, FlexibleFileLoader],
        web_url_loader : BaseLoader,
        rag_llm: BaseLanguageModel,
        vectorstore : VectorStore,
        splitter: TextSplitter=None,
    ):
        self.rag_llm = rag_llm
        self.splitter = splitter
        self.dataloader = data_loader
        self.web_url_loader = web_url_loader
        self.vectorstore = vectorstore

    def run(
        self,
        uploaded_files: Optional[List] = None,
        upload_url: Optional[str] = None
    )-> Runnable[Any, List[Document]]:
        documents = []

        if uploaded_files:
            documents.extend(self.dataloader.load_multiple_streamlit_files(uploaded_files))

        if upload_url:
            documents.extend(self.web_url_loader.load_and_split(upload_url, text_splitter=self.splitter))

        vectorstore = load_documents_faiss_vectorsotre(documents)
        retriever_chain = get_retrievered_documents(vectorstore, self.rag_llm)
        return retriever_chain
    

    
    def custom_rag_pipeline(documents):
        return Custom_RAGPipeline(
            FlexibleFileLoader(

            ),
            web_loader(),
            OpenAIEmbeddings(),
            FAISS.from_documents(documents)
            )
        