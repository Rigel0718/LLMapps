import requests
from bs4 import BeautifulSoup, SoupStrainer
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders.base import BaseLoader

url = ""
headers = {
    "User-Agent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
}



def bs4_web_loader(url):
    response = requests.get(url, headers=headers)
    response.encoding = 'utf-8' 
    soup = BeautifulSoup(response.text, 'html.parser', parse_only=SoupStrainer('div', attrs={'id': ['web-header', 'content-blocks']}))
    return soup.get_text()


def web_loader(url) -> BaseLoader:
    loader = WebBaseLoader(
    web_path=url,
    encoding='utf-8',
    bs_kwargs=dict(
        parse_only=SoupStrainer(
            'div',
            attrs={'id' : ['web-header', 'content-blocks']}
        )
    ),
    header_template=headers
    )
    return loader
    