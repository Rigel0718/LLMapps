import requests
from bs4 import BeautifulSoup, SoupStrainer
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from typing import Iterator, Any


headers = {
    "User-Agent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
}



def bs4_web_loader(url, attrs : dict={}):
    response = requests.get(url, headers=headers)
    response.encoding = 'utf-8' 
    soup = BeautifulSoup(response.text, 'html.parser', parse_only=SoupStrainer('div', attrs=attrs))
    return soup.get_text()


def web_loader(url, attrs : dict={}) -> BaseLoader:
    loader = WebBaseLoader(
    web_path=url,
    encoding='utf-8',
    bs_kwargs=dict(
        parse_only=SoupStrainer(
            'div',
            attrs=attrs
        )
    ),
    header_template=headers
    )
    return loader


def _build_metadata(soup: Any, url: str) -> dict:
    """Build metadata from BeautifulSoup output."""
    metadata = {"source": url}
    if title := soup.find("title"):
        metadata["title"] = title.get_text()
    if description := soup.find("meta", attrs={"name": "description"}):
        metadata["description"] = description.get("content", "No description found.")
    if html := soup.find("html"):
        metadata["language"] = html.get("lang", "No language found.")
    return metadata


class CustomHTMLTagLoader(WebBaseLoader):
    def __init__(self, web_path: str, tags_to_extract: list[str] = None, **kwargs):
        super().__init__(web_path=web_path, **kwargs)
        self.tags_to_extract = tags_to_extract or ["h1", "h2", "h3", "p", "li"]

    def lazy_load(self) -> Iterator[Document]:
        for path in self.web_paths:
            soup = self._scrape(path, bs_kwargs=self.bs_kwargs)

            for tag in soup.find_all(self.tags_to_extract):
                text = tag.get_text(strip=True)
                if text:
                    metadata = _build_metadata(soup, path)
                    metadata.update({
                        "tag": tag.name,
                        "html": str(tag),  # 태그 전체를 HTML 형태로 저장
                    })
                    yield Document(page_content=text, metadata=metadata)


if __name__ == '__main__':
    url = 'https://turingpost.co.kr/p/8-rag-master-course'
    attrs = {'id' : ['web-header', 'content-blocks']}
    loader = web_loader(url,attrs)
    docs = loader.load()
    print(docs)
    