from .dataloader import get_documents, get_url_documents, Custom_Streamlit_FileLoader
from .web_dataloader import bs4_web_loader, web_loader

__all__ = [
    "get_documents",
    "get_url_documents",
    "bs4_web_loader",
    "web_loader",
    "Custom_Streamlit_FileLoader",
]