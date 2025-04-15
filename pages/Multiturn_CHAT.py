import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from utils import get_chat_prompt_yaml
from langchain_core.output_parsers import StrOutputParser
from message_history import get_message_history_sqlitedb, configs

MODEL = ['gpt-4o-mini', 'o3-mini']


def main():
    st.set_page_config(page_title="Multiturn-CHAT")

    if 'llm' not in st.session_state:
        st.session_state.llm = None

    if 'openai_api_key' not in st.session_state:
          st.session_state.openai_api_key = None

    if 'client_id' not in st.session_state:
          st.session_state.client_id = None

    if 'call_user_chathistory' not in st.session_state:
          st.session_state.call_user_chathistory = None

    if 'conversation_num' not in st.session_state:
          st.session_state.conversation_num = None

    

    with st.sidebar:
        st.text_input("USER_ID", key='cliend_id')
        st.session_state.call_user_chathistory = st.button("exist_ID")
        st.text_input("OpenAI API Key", key="openai_api_key", type="password")
        "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
        st.selectbox("🤖 Select a Model", options=MODEL, key = 'model')


    if 'messages' not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "Hello, How can I help you?"}]

    for message in st.session_state.messages:
            st.chat_message(message["role"]).write(message["content"])

    prompt_filepath = 'prompts/basic_prompt.yaml'
    prompt = ChatPromptTemplate.from_messages(get_chat_prompt_yaml(prompt_filepath))

    st.session_state.llm = ChatOpenAI(
            api_key=st.session_state.openai_api_key, 
            model=st.session_state.model,
            temperature=0.3,
            streaming=True )
    
    chain = prompt | st.session_state.llm | StrOutputParser()

if __name__ == '__main__':
    main()

