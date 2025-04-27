# from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.language_models import LanguageModelLike
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.base import Runnable
from langchain_core.vectorstores.base import VectorStore
from langchain.docstore.document import Document
from typing import Any, List
from utils import get_chat_prompt_yaml

'''
대화의 흐름과 vectorstore의 자료, 두 가지 모두 참고하여 대화하는 chat bot chain.
자연스로운 대화를 위해 기존 대화 기록 'messages'를 두 번넣어 두 개의 llm chain을 사용했다.
1. 대화의 흐름을 고려하여 쿼리를 작성하는 chain (vectorstore를 참고하지만, 더 나은 퀄리티의 쿼리를 생성하기 위해 참고)
2. 생성된 쿼리를 이용해서 vectorstore에서 관련 문서 검색 + 기존 대화를 활용해서 자연스러운 답변 생성.
'''



def _get_retriever_chain(vectorstore : VectorStore, llm : LanguageModelLike) -> Runnable[Any, List[Document]] :
    '''
    vectorstore의 retriever를 이용하여, 기존 대화 기록을 반영하는 검색 쿼리를 생성하는 체인 구성
    즉, 대화의 흐름을 고려하여 어떤 내용을 검색해야하는지 결정하는 '체인'을 생성하는 함수.

    TODO 
    ConversationBufferMemory 사용해서 MessagePlaceholder대체
    '''

    # vectorstore의 retriever 설정
    retriever = vectorstore.as_retriever()
    prompt_filepath = 'prompts/retrieve_prompt.yaml'
    prompt = ChatPromptTemplate.from_messages(get_chat_prompt_yaml(prompt_filepath))
    # 체인 생성
    retriever_chain : List[Document] = create_history_aware_retriever(llm, retriever, prompt)

    return retriever_chain


def get_conversational_rag_chain(vectorstore : VectorStore, llm : LanguageModelLike):
    retriever_chain = _get_retriever_chain(vectorstore, llm)
    '''
    prompt 
    system : LLM의 역할 정의
    context : 검색된 문서  (context인 이유는 create_stuff_documents_chain 함수의 document variable parameter의 기본값이 context이기 때문이다.)
    messages : 기존 대화의 흐름 정보
    input : user의 쿼리
    '''
    prompt_filepath = 'prompts/basic_prompt.yaml'
    prompt = ChatPromptTemplate.from_messages(get_chat_prompt_yaml(prompt_filepath))
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)   
    
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)