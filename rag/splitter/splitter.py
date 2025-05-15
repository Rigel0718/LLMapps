from langchain.text_splitter import RecursiveCharacterTextSplitter, HTMLHeaderTextSplitter
from typing import List
from bs4 import BeautifulSoup


def get_html_list_tags(soup : BeautifulSoup) -> List[str]:
    list_items = soup.find_all("li")
    list_texts = [li.get_text(strip=True) for li in list_items]
    return list_texts
