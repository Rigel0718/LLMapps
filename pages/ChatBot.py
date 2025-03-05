import streamlit as st
from langchain_openai import ChatOpenAI
from rag_retriever import get_conversational_rag_chain
from rag_vectorstore import load_documents_chroma_vectorstore
from rag_loader import get_documents
import uuid
import time
MODEL = ['gpt-4o-mini', 'o3-mini']

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
    
    if st.session_state.upload_files:
        return chain.pick("answer").stream({'messages': messages, 'input' : query})
    else:
        return chain.pick("text").stream(messages)
    
    

def stream_response(chain, messages, query=None):
    with st.chat_message("assistant"):
        
        for chunk in _pick_chain_output_(chain,messages,query):
            yield chunk
            time.sleep(0.05)
        


def main():
    st.set_page_config(
        page_icon=':books:',
        page_title='ChatQA'
    )
    st.title("💬 Chatbot")

    if 'upload_files' not in st.session_state:
        st.session_state.upload_files = None

    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    if 'llm' not in st.session_state:
        st.session_state.llm = None

    with st.sidebar:
        st.session_state.upload_files = st.file_uploader('upload your files', type=['pdf', 'docx', 'pptx'], accept_multiple_files=True)
        st.text_input("OpenAI API Key", key="openai_api_key", type="password")
        "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
        st.selectbox("🤖 Select a Model", options=MODEL, key = 'model')
        rag_process = st.button('RAG PROCESS')
        

    if rag_process:
        if not st.session_state.upload_files:
            st.sidebar.error('⚠️ No file uploaded. Please upload a file first.')
        else :
            documents = get_documents(st.session_state.upload_files)
            st.session_state.vectorstore = load_documents_chroma_vectorstore(documents)
            st.sidebar.success('✅ Upload to vector store completed!')


    # session state initialize
    if 'messages' not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello, How can I help you?"}]



    for message in st.session_state.messages:
        st.chat_message(message["role"]).write(message["content"])

    st.session_state.llm = ChatOpenAI(
        api_key=st.session_state.openai_api_key, 
        model=st.session_state.model,
        temperature=0.3,
        streaming=True )


    if query := st.chat_input('Please enter your Question'):
        if not st.session_state.openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
        st.chat_message('user').write(query)
        st.session_state.messages.append({"role": "user", "content": query})
        messages = convert_chat_history(st.session_state.messages)

        if st.session_state.upload_files:
            rag_chain = get_conversational_rag_chain(st.session_state.vectorstore, st.session_state.llm)
            response = st.write_stream(stream_response(rag_chain, messages, query))   
            
        else:
            response = st.write_stream(stream_response(st.session_state.llm, messages))   

        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == '__main__':
    main()
