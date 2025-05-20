from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain_community.document_loaders.base import BaseLoader
from typing import Optional, Callable, List, Type
from pathlib import Path
from .temp_file_wrapper import with_temp_file
from langchain.docstore.document import Document

class FlexibleFileLoader:
    def __init__(
        self,
        splitter=None,
        preprocess_fn: Optional[Callable[[bytes], bytes]] = None
    ):
        self.splitter = splitter
        self.preprocess_fn = preprocess_fn
        self.loader_map: dict[str, Type[BaseLoader]] = {
            ".pdf": PyPDFLoader,
            ".docx": Docx2txtLoader,
            ".pptx": UnstructuredPowerPointLoader,
        }

    def get_loader_class(self, file_path: str) -> Optional[Type[BaseLoader]]:
        suffix = Path(file_path).suffix.lower()
        return self.loader_map.get(suffix)

    @with_temp_file()
    def get_documents(self, temp_path: str) -> List[Document]:
        loader_class = self.get_loader_class(temp_path)
        if not loader_class:
            raise ValueError(f"Unsupported file type: {Path(temp_path).suffix}")
        
        loader = loader_class(temp_path)
        if self.splitter:
            return loader.load_and_split(text_splitter=self.splitter)
        else:
            return loader.load()