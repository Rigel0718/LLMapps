# from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.vectorstores import Chroma
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.base import Runnable

def _get_retriever_chain(vectorstore : Chroma, llm) -> Runnable :
    '''
    vectorstore의 retriever를 이용하여, 기존 대화 기록을 반영하는 검색 쿼리를 생성하는 체인 구성
    즉, 대화의 흐름을 고려하여 어떤 내용을 검색해야하는지 결정하는 '체인'을 생성하는 함수.
    '''

    # vectorstore의 retriever 설정
    retriever = vectorstore.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="messages"), # 기존 대화 messages 참고
        ("user", "{input}"), # user 입력한 쿼리
        ("user", "Given the above conversation, generate a search query to look up in order to get inforamtion relevant to the conversation, focusing on the most recent messages."),
    ])
    # 체인 생성
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    return retriever_chain


def get_conversational_rag_chain(vectorstore, llm):
    retriever_chain = _get_retriever_chain(vectorstore, llm)
    '''
    prompt 
    system : LLM의 역할 정의
    context : 검색된 문서 
    messages : 기존 대화의 흐름 정보
    input : user의 쿼리
    '''
    prompt = ChatPromptTemplate.from_messages([
        ("system",
        """You are a helpful assistant. You will have to answer to user's queries.
        You will have some context to help with your answers, but now always would be completely related or helpful.
        You can also use your knowledge to assist answering the user's queries.\n 
        {context}"""),
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever_chain, stuff_documents_chain)