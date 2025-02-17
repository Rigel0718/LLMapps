from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from streamlit.runtime.uploaded_file_manager import UploadedFile
from langchain.docstore.document import Document

import tempfile
import os
from typing import List
from io import BytesIO
from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
)

def get_text(docs: List[UploadedFile]) -> List[Document]:
    doc_list = []

    for doc in docs:
        file_bytes = doc.getvalue()  # 바이트 데이터 추출
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
            continue  # 지원되지 않는 파일은 건너뛰기

        # 임시 파일 생성 및 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(file_bytes)
            tmp_file_path = tmp_file.name  # 파일 경로 저장

        try:
            # 적절한 로더 사용
            loader = loader_class(tmp_file_path)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size = 900, chunk_overlap = 200)
            documents = loader.load_and_split(text_splitter=text_splitter)
            doc_list.extend(documents)
        finally:
            # 임시 파일 삭제 (try-finally로 보장)
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)

    return doc_list