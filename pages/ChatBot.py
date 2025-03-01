import streamlit as st
from util import get_text, get_text_chunks, get_vectorstore, get_conversation_chain
from langchain_openai import ChatOpenAI
from rag_retriever import get_conversational_rag_chain
from rag_vectorstore import load_documents_chroma_vectorstore

MODEL = ['model1', 'model2', 'model3']

llm = ChatOpenAI(
    api_key=st.session_state.openai_api_key, 
    model=st.session_state.model,
     temperature=0.3,
      streaming=True )

def main():
    st.set_page_config(
        page_icon=':books:',
        page_title='ChatQA'
    )
    st.title("💬 Chatbot")

    st.write("Hello, world!")

    if 'conversation_chain' not in st.session_state:
        st.session_state.conversation_chain = None

    if 'chat_memory' not in st.session_state:
        st.session_state.chat_memory = None

    with st.sidebar:
        st.file_uploader('upload your files', type=['pdf', 'docx', 'pptx'], accept_multiple_files=True, key='upload_files')
        st.text_input("OpenAI API Key", key="openai_api_key", type="password")
        "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
        st.selectbox("🤖 Select a Model", options=MODEL, key = 'model')
        process = st.button('process')
        

    if process:
        vectorstore = load_documents_chroma_vectorstore(st.session_state.upload_files)


    # session state initialize
    if 'messages' not in st.session_state:
        # st.session_state.messages = [{"role": "assistant", "content": "Please, input your OpenAI key on sidebar and upload your own files?"}]
        # st.chat_message('assistant').write('Please, input your OpenAI key on sidebar and upload your own files?')
        st.session_state.messages = [{"role": "assistant", "content": "Hello, How can I help you?"}]


    
    # st.session_state.messages = [{"role": "assistant", "content": "Hello, How can I help you?"}]

    for message in st.session_state.messages:
        st.chat_message(message["role"]).write(message["content"])

    if query := st.chat_input('Please enter your Question'):
        if not st.session_state.openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
        st.session_state.messages.append({"role": "user", "content": query})
        st.chat_message('user').write(query)
        rag_chain = get_conversational_rag_chain(vectorstore, llm)
        respons = rag_chain.invoke({'input' : query})
        st.session_state.messages.append({"role": "assistant", "content": respons})
        st.chat_message("assistant").write(respons)

if __name__ == '__main__':
    main()
