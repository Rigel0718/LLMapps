import streamlit as st
import time
from langchain.schema import AIMessage, HumanMessage


def convert_chat_history(chat_messages):
    '''
    Streamlit의 세션 상태에서 저장된 채팅 기록을 LangChain 메시지 형식으로 변환
    '''
    
    return [
        HumanMessage(content=msg["content"]) if msg["role"] == "user" 
        else AIMessage(content=msg["content"])
        for msg in chat_messages
    ]



from langchain.chains.base import Chain


def _pick_chain_output_(chain : Chain, messages, query=None):
    '''
    chain의 stream 출력을 따르면서, chatbot에 필요한 text형태의 key 값을 추출해주는 함수
    '''
    
    if rag_available():
        return chain.pick("answer").stream({'messages': messages, 'input' : query})
    else:
        return chain.stream(messages)
    
    

def stream_response(chain, messages, query=None):
    '''
    assistant가 stream형태의 답변을 구사할 수 있도록 만들어주는 함수

    * chunk와 UI의 충돌 방지를 위한 time.sleep(0.05) 구현
    '''
    with st.chat_message("assistant"):
        
        for chunk in _pick_chain_output_(chain,messages,query):
            yield chunk
            time.sleep(0.05)

        

def rag_available() -> bool:
    '''
    st.session_state.upload_files의 default -> []  empty list
    st.session_state.upload_url 의 default -> ''   empty string
    따라서 이 메서드는 둘 중 하나라도 검색가능한 document가 있다면, rag_available -> True
    '''
    return st.session_state.upload_files  or st.session_state.upload_url 