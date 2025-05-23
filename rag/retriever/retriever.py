from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.language_models import LanguageModelLike
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.base import Runnable
from langchain_core.vectorstores.base import VectorStore
from langchain.docstore.document import Document
from typing import Any, List
from .utils import load_prompt_template

'''
대화의 흐름과 vectorstore의 자료, 두 가지 모두 참고하여 대화하는 chat bot chain.
자연스로운 대화를 위해 기존 대화 기록 'messages'를 두 번넣어 두 개의 llm chain을 사용했다.
1. 대화의 흐름을 고려하여 쿼리를 작성하는 chain (vectorstore를 참고하지만, 더 나은 퀄리티의 쿼리를 생성하기 위해 참고)
2. 생성된 쿼리를 이용해서 vectorstore에서 관련 문서 검색 + 기존 대화를 활용해서 자연스러운 답변 생성.
'''



def get_retrievered_documents(vectorstore : VectorStore, llm : LanguageModelLike) -> Runnable[Any, List[Document]] :
    '''
    vectorstore의 retriever를 이용하여, 기존 대화 기록을 반영하는 검색 쿼리를 생성하는 체인 구성
    즉, 대화의 흐름을 고려하여 어떤 내용을 검색해야하는지 결정하는 '체인'을 생성하는 함수.
    '''

    # vectorstore의 retriever 설정
    retriever = vectorstore.as_retriever()
    prompt_filepath = 'chat_query_rewrite_prompt.yaml'
    prompt = load_prompt_template(prompt_filepath)
    # 체인 생성
    retriever_chain : List[Document] = create_history_aware_retriever(llm, retriever, prompt)

    return retriever_chain


if __name__ == '__main__':
    # from pathlib import Path
    # base_path = Path(__file__).parent
    # prompt_filepath = base_path / 'retriever_prompt' / 'chat_query_rewrite_prompt.yaml'
    prompt_filepath = 'chat_query_rewrite_prompt.yaml'
    print(prompt_filepath)
    prompt = load_prompt_template(prompt_filepath)
    print(prompt)