from typing import Optional, List, Any, Union
from langchain_core.language_models import BaseLanguageModel
from langchain_core.documents import Document
from langchain_text_splitters.base import TextSplitter
from langchain_core.runnables import Runnable
from .vectorstore.vectorstore import load_documents_faiss_vectorsotre
from .retriever.retriever import get_retrievered_documents
from langchain_community.document_loaders.base import BaseLoader
from langchain_core.vectorstores.base import VectorStore
from dataloader import FlexibleFileLoader, FileInput

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
        file_inputs: Optional[List[FileInput]] = None,
        url_inputs: Optional[List[str]] = None,
    ) -> Runnable[Any, List[Document]]:
        documents = []

        if file_inputs:
            documents.extend(self.dataloader.load(file_inputs, splitter=self.splitter))

        if url_inputs:
            for url in url_inputs:
                documents.extend(
                    self.web_url_loader.load_and_split(url, text_splitter=self.splitter)
                )

        vectorstore = load_documents_faiss_vectorsotre(documents)
        return get_retrievered_documents(vectorstore, self.rag_llm)