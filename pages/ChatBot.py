import streamlit as st
from langchain_openai import ChatOpenAI
from rag_retriever import get_conversational_rag_chain
from rag_vectorstore import load_documents_chroma_vectorstore, load_documents_faiss_vectorsotre
from rag_loader import get_documents, get_url_documents
from utils import rag_available, convert_chat_history, stream_response
import uuid

MODEL = ['gpt-4o-mini', 'o3-mini']



def main():
    st.set_page_config(
        page_icon=':books:',
        page_title='ChatQA'
    )
    st.title("💬 Chatbot")

    if 'upload_files' not in st.session_state:
        st.session_state.upload_files = None

    if 'upload_url' not in st.session_state:
        st.session_state.upload_url = None

    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    if 'llm' not in st.session_state:
        st.session_state.llm = None

    if 'rag_process'not in st.session_state:
        st.session_state.rag_process = False
    
    with st.sidebar:
        st.session_state.upload_files = st.file_uploader('upload your files', type=['pdf', 'docx', 'pptx'], accept_multiple_files=True)
        st.session_state.upload_url = st.text_input("🌐 upload URL")
        st.text_input("OpenAI API Key", key="openai_api_key", type="password")
        "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
        st.selectbox("🤖 Select a Model", options=MODEL, key = 'model')
        if rag_available():
            st.session_state.rag_process = st.button('RAG PROCESS')
        

    if st.session_state.rag_process:
        if not rag_available():
            st.sidebar.error('⚠️ No file uploaded. Please upload a file first.')
        else :
            documents = []
            if st.session_state.upload_files:
                file_documents = get_documents(st.session_state.upload_files)
                documents.extend(file_documents)
            if st.session_state.upload_url:
                url_documents = get_url_documents(st.session_state.upload_url)
                documents.extend(url_documents)
            st.session_state.vectorstore = load_documents_faiss_vectorsotre(documents)
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

        if rag_available():
            rag_chain = get_conversational_rag_chain(st.session_state.vectorstore, st.session_state.llm)
            response = st.write_stream(stream_response(rag_chain, messages, query))   
            
        else:
            response = st.write_stream(stream_response(st.session_state.llm, messages))   

        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == '__main__':
    main()
