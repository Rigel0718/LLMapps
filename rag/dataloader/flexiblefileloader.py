from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain_community.document_loaders.base import BaseLoader
from typing import Optional, Callable, List, Type, Tuple
from pathlib import Path
from .temp_file_wrapper import with_temp_file
from langchain.docstore.document import Document
from langchain_text_splitters.base import TextSplitter
from streamlit.runtime.uploaded_file_manager import UploadedFile
from pydantic import BaseModel, Field

class FileInput(BaseModel):
    content: bytes
    filename: str = Field(..., description="Original filename including extension")



class FlexibleFileLoader:
    def __init__(
        self,
        preprocess_fn: Optional[Callable[[bytes], bytes]] = None
    ):
        self.preprocess_fn = preprocess_fn
        self.loader_map: dict[str, Type[BaseLoader]] = {
            ".pdf": PyPDFLoader,
            ".docx": Docx2txtLoader,
            ".pptx": UnstructuredPowerPointLoader,
        }

    def get_loader_class(self, file_path: str) -> Optional[Type[BaseLoader]]:
        suffix = Path(file_path).suffix.lower()
        return self.loader_map.get(suffix)

    @with_temp_file(preprocess_fn=None)
    def get_documents(self, temp_path: str, splitter: Optional[TextSplitter] = None) -> List[Document]:
        loader_class = self.get_loader_class(temp_path)
        if not loader_class:
            raise ValueError(f"Unsupported file type: {Path(temp_path).suffix}")
        loader = loader_class(temp_path)
        return loader.load_and_split(text_splitter=splitter) if splitter else loader.load()

    def load(
        self,
        files: List[FileInput],
        splitter: Optional[TextSplitter] = None
        ) -> List[Document]:
        documents = []
        for file in files:
            documents.extend(self.get_documents(file.content, file.filename, splitter))
        return documents
    

def extract_streamlit_file_info(uploaded_files: List[UploadedFile]) -> List[FileInput]:
    return [FileInput(content=f.getvalue(), filename=f.name) for f in uploaded_files]

def extract_file_paths(file_paths: List[str]) -> List[FileInput]:
    return [FileInput(content=Path(p).read_bytes(), filename=Path(p).name) for p in file_paths] 