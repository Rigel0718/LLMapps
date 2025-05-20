from functools import wraps
from typing import Callable, Optional
from pathlib import Path
import tempfile
import os

def with_temp_file(preprocess_fn : Optional[Callable[[bytes, str], bytes]] = None):
    '''
    file_bytes의 전처리가 필요한 경우 preprocess_fn에 함수형태로 입력 (ex 이미지 전처리, text encoding .. etc)

    ex)
    @with_temp_file(preprocess_fn=None)  
    def load_pdf(tmp_path: str, file_name: str) -> List[Document]:
        loader = PyPDFLoader(tmp_path)
        return loader.load()
    '''
    def decorator(func):
        @wraps(func)
        def wrapper(file_bytes: bytes, file_name: str, *args, **kwargs):
            suffix = Path(file_name).suffix
            if not suffix:
                raise ValueError(f"File {file_name} has no extension")
            processed = preprocess_fn(file_bytes, file_name) if preprocess_fn else file_bytes

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(processed)
                tmp_path = tmp_file.name

            try:
                return func(tmp_path, file_name, *args, *kwargs)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        
        return wrapper
    return decorator


from contextlib import contextmanager
from typing import Generator

@contextmanager
def temp_file_from_bytes(file_bytes: bytes, file_name: str, preprocess_fn=None) -> Generator[str, None, None]:
    """
    ex)

    with temp_file_from_bytes(file_bytes, file_name, preprocess_fn=my_preprocess) as tmp_path:
    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    
    """
    suffix = Path(file_name).suffix
    if not suffix:
        raise ValueError(f"File '{file_name}' has no extension.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        processed_bytes = preprocess_fn(file_bytes, file_name) if preprocess_fn else file_bytes
        tmp_file.write(processed_bytes)
        tmp_path = tmp_file.name

    try:
        yield tmp_path
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


