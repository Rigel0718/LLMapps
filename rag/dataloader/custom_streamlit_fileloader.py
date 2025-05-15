from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from streamlit.runtime.uploaded_file_manager import UploadedFile
from langchain.docstore.document import Document
from langchain_community.document_loaders.base import BaseLoader
import tempfile
import os
from typing import List, Optional
from pathlib import Path

class Custom_Streamlit_FileLoader:
    def __init__(self, splitter=None):
        self.loader_map = {
            ".docx": Docx2txtLoader,
            ".pdf": PyPDFLoader,
            ".pptx": UnstructuredPowerPointLoader,
        }
        self.splitter = splitter  # 기본값 없음

    def get_loader_class(self, file_name: str):
        ext = Path(file_name).suffix.lower()
        return self.loader_map.get(ext), ext

    def load_single_streamlit_file(self, uploaded_file: UploadedFile) -> Optional[List[Document]]:
        '''
        streamlit의 UploadedFile class만 getvalue, name가지고 있음.
        '''
        file_bytes = uploaded_file.getvalue() # 바이트 데이터 추출 (UploadedFile객체는 BytesIO 객체를 상속한다 즉 file_like 객체)
        file_name = uploaded_file.name

        loader_class, ext = self.get_loader_class(file_name)
        if loader_class is None:
            print(f"⚠️ Unsupported file type: {file_name}")
            return None

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
            tmp_file.write(file_bytes)
            tmp_path = tmp_file.name

        try:
            loader : BaseLoader = loader_class(tmp_path)
            if self.splitter:
                documents = loader.load_and_split(text_splitter=self.splitter)
            else:
                documents = loader.load()
            return documents
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def load_multiple_streamlit_files(self, files: List[UploadedFile]) -> List[Document]:
        all_docs = []
        for f in files:
            docs = self.load_single_streamlit_file(f)
            if docs:
                all_docs.extend(docs)
        return all_docs