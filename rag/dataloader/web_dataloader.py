import requests
from bs4 import BeautifulSoup, SoupStrainer
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders.base import BaseLoader

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

if __name__ == '__main__':
    url = 'https://turingpost.co.kr/p/8-rag-master-course'
    attrs = {'id' : ['web-header', 'content-blocks']}
    loader = web_loader(url,attrs)
    docs = loader.load()
    print(docs)
    