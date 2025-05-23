from .flexiblefileloader import FlexibleFileLoader, FileInput
from .web_dataloader import bs4_web_loader, web_loader

__all__ = [
    "bs4_web_loader",
    "web_loader",
    "FlexibleFileLoader",
    "FileInput"
]