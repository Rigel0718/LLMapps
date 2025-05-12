from typing import Optional, List
from langchain.text_splitter import RecursiveCharacterTextSplitter

from .dataloader import Custom_Streamlit_FileLoader, web_loader
from .vectorstore import load_documents_faiss_vectorsotre


class Custom_RAGPipeline:
    def __init__(
        self,
        chunk_size: int = 900,
        chunk_overlap: int = 200,
    ):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.dataloader = Custom_Streamlit_FileLoader(
            splitter=self.splitter
        )

    def run(
        self,
        uploaded_files: Optional[List] = None,
        upload_url: Optional[str] = None
    ):
        documents = []

        if uploaded_files:
            documents.extend(self.dataloader.load_multiple_streamlit_files(uploaded_files))

        if upload_url:
            documents.extend(web_loader(upload_url))

        vectorstore = load_documents_faiss_vectorsotre(documents)
        return vectorstore