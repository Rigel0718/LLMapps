import streamlit as st
import time

def rag_available() -> bool:
    return st.session_state.upload_files  or st.session_state.upload_url 


from langchain.schema import AIMessage, HumanMessage

def convert_chat_history(chat_messages):
    """Streamlit의 세션 상태에서 저장된 채팅 기록을 LangChain 메시지 형식으로 변환"""
    return [
        HumanMessage(content=msg["content"]) if msg["role"] == "user" 
        else AIMessage(content=msg["content"])
        for msg in chat_messages
    ]

from langchain.chains.base import Chain


def _pick_chain_output_(chain : Chain, messages, query=None):
    
    if rag_available():
        return chain.pick("answer").stream({'messages': messages, 'input' : query})
    else:
        return chain.pick("text").stream(messages)
    
    

def stream_response(chain, messages, query=None):
    with st.chat_message("assistant"):
        
        for chunk in _pick_chain_output_(chain,messages,query):
            yield chunk
            time.sleep(0.05)

        