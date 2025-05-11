import requests
from bs4 import BeautifulSoup, SoupStrainer

url = "https://turingpost.co.kr/p/8-rag-master-course"
headers = {
    "User-Agent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
}


def bs4_web_loader(url):
    response = requests.get(url, headers=headers)
    response.encoding = 'utf-8' 
    soup = BeautifulSoup(response.text, 'html.parser', parse_only=SoupStrainer('div', attrs={'id': ['web-header', 'content-blocks']}))
    return soup.get_text()