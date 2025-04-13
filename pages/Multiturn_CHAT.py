import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory


MODEL = ['gpt-4o-mini', 'o3-mini']


def main():
    st.set_page_config(page_title="Persist-CHAT")

    if 'llm' not in st.session_state:
        st.session_state.llm = None

    if 'openai_api_key' not in st.session_state:
          st.session_state.openai_api_key = None

    if 'client_id' not in st.session_state:
          st.session_state.client_id = None

    if 'conversation_num' not in st.session_state:
          st.session_state.conversation_num = None

    

    with st.sidebar:
        st.text_input("OpenAI API Key", key="openai_api_key", type="password")
        "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
        st.selectbox("🤖 Select a Model", options=MODEL, key = 'model')



    if 'messages' not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "Hello, How can I help you?"}]

    for message in st.session_state.messages:
            st.chat_message(message["role"]).write(message["content"])

    st.session_state.llm = ChatOpenAI(
            api_key=st.session_state.openai_api_key, 
            model=st.session_state.model,
            temperature=0.3,
            streaming=True )
    

if __name__ == '__main__':
    main()

