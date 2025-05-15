from typing import Optional, List, Any
from langchain_core.language_models import BaseLanguageModel
from langchain_core.documents import Document
from langchain_text_splitters.base import TextSplitter
from langchain_core.runnables import Runnable
from .dataloader import Custom_Streamlit_FileLoader, web_loader
from .vectorstore.vectorstore import load_documents_faiss_vectorsotre
from .retriever import get_retrievered_documents
from langchain_community.document_loaders.base import BaseLoader
from langchain_core.vectorstores.base import VectorStore
# TODO splitter, vectorstore(embeddingmodel), retriever 모두 객체화로 customizing할 수 있게 
class Custom_RAGPipeline:
    def __init__(
        self,
        data_loader : BaseLoader,
        rag_llm: BaseLanguageModel,
        splitter: TextSplitter,
        vectorstore : VectorStore,
        chunk_size: int = 900,
        chunk_overlap: int = 200,
    ):
        self.rag_llm = rag_llm
        self.splitter = splitter
        self.dataloader = data_loader
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
            documents.extend(web_loader(upload_url))

        vectorstore = load_documents_faiss_vectorsotre(documents)
        retriever_chain = get_retrievered_documents(vectorstore, self.rag_llm)
        return retriever_chain