from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from streamlit.runtime.uploaded_file_manager import UploadedFile
from langchain.docstore.document import Document
import streamlit as st
import tempfile
import os
from typing import List, Optional
from pathlib import Path

def get_documents(docs: List[UploadedFile]) -> List[Document]:
    '''
    TODO
    splitter 유연성 높이기
    '''
    doc_list = []

    for doc in docs:
        file_bytes = doc.getvalue()  # 바이트 데이터 추출 (UploadedFile객체는 BytesIO 객체를 상속한다 즉 file_like 객체)
        file_name = doc.name  # UploadedFile 객체라서 `.name` 사용 가능
        # 확장자 체크
        if file_name.endswith(".docx"):
            suffix = ".docx"
            loader_class = Docx2txtLoader
        elif file_name.endswith(".pdf"):
            suffix = ".pdf"
            loader_class = PyPDFLoader
        elif file_name.endswith(".pptx"):
            suffix = ".pptx"
            loader_class = UnstructuredPowerPointLoader
        else:
            print(f"⚠️ Unsupported file type: {file_name}")
            continue  # 지원되지 않는 파일 확장자는 건너뛰기

        tmp_file_path = None  # finally에서 참조 가능하도록 초기화

        #임시 파일 생성 및 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(file_bytes)
            tmp_file_path = tmp_file.name

        try:
            # 적절한 로더 사용
            loader = loader_class(tmp_file_path)
            # splitter 사용 
            text_splitter = RecursiveCharacterTextSplitter(chunk_size = 900, chunk_overlap = 200)
            documents = loader.load_and_split(text_splitter=text_splitter)
            doc_list.extend(documents)
        finally:
            # 임시 파일 삭제 (try-finally로 보장)
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)

    return doc_list


def get_url_documents(url) -> List[Document]:

    doc_list=[]

    try:
        loader = WebBaseLoader(url)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 900, chunk_overlap = 200)
        documents = loader.load_and_split(text_splitter=text_splitter)
        doc_list.extend(documents)
    except Exception as e:
        st.error(f'Error loading url document from {url} : {e}')

    return doc_list



class CustomLoader:
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

    def load_single_file(self, uploaded_file: UploadedFile) -> Optional[List[Document]]:
        file_bytes = uploaded_file.getvalue()
        file_name = uploaded_file.name

        loader_class, ext = self.get_loader_class(file_name)
        if loader_class is None:
            print(f"⚠️ Unsupported file type: {file_name}")
            return None

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
            tmp_file.write(file_bytes)
            tmp_path = tmp_file.name

        try:
            loader = loader_class(tmp_path)
            if self.splitter:
                documents = loader.load_and_split(text_splitter=self.splitter)
            else:
                documents = loader.load()
            return documents
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def load_multiple_files(self, files: List[UploadedFile]) -> List[Document]:
        all_docs = []
        for f in files:
            docs = self.load_single_file(f)
            if docs:
                all_docs.extend(docs)
        return all_docs