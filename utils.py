import streamlit as st
import time
from langchain.schema import AIMessage, HumanMessage
import yaml
from langchain_core.runnables import Runnable
from typing import List


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


def _pick_chain_output_(chain : Chain, messages, query):
    '''
    chain의 stream 출력을 따르면서, chatbot에 필요한 text형태의 key 값을 추출해주는 함수
    '''
    
    if rag_available():
        return chain.pick("answer").stream({'messages': messages, 'input' : query})
    else:
        return chain.stream({'messages': messages, 'input' : query})
    
    

def stream_response(chain, messages, query):
    '''
    assistant가 stream형태의 답변을 구사할 수 있도록 만들어주는 함수

    * chunk와 UI의 충돌 방지를 위한 time.sleep(0.05) 구현
    '''
    with st.chat_message("assistant"):
        
        for chunk in _pick_chain_output_(chain,messages,query):
            yield chunk
            time.sleep(0.05)

def multiturn_stream_response(chain, query, config):
    with st.chat_message("assistant"):
        
        for chunk in chain.stream({'input' : query}, config):
            yield chunk
            time.sleep(0.05)


        

def rag_available() -> bool:
    '''
    st.session_state.upload_files의 default -> []  empty list
    st.session_state.upload_url 의 default -> ''   empty string
    따라서 이 메서드는 둘 중 하나라도 검색가능한 document가 있다면, rag_available -> True
    '''
    return st.session_state.upload_files  or st.session_state.upload_url 

def get_chat_prompt_yaml(file_path):
    with open(file_path, "r", encoding='utf8') as f:
        yaml_content = yaml.safe_load(f)
        return [(message['role'], message['content']) for message in yaml_content['messages']]
    
def get_next_conversation_num(conversation_list: List[str]) -> str:
    nums = [int(c) for c in conversation_list if c.isdigit()]
    return str(max(nums) + 1) if nums else "0"


# Helper function for printing docs


def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )