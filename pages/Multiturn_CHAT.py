import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from utils import get_chat_prompt_yaml, multiturn_stream_response
from langchain_core.output_parsers import StrOutputParser
from message_history import get_message_history_sqlitedb, configs_fields, load_messages_from_sqlite

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
        st.session_state.client_id = st.text_input("USER_ID", key='cliend_id')
        st.session_state.call_user_chathistory = st.button("MAKE_NEW_CONVERSATION")
        st.text_input("OpenAI API Key", key="openai_api_key", type="password")
        "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
        st.selectbox("🤖 Select a Model", options=MODEL, key = 'model')




    prompt_filepath = 'prompts/normal_prompt.yaml'
    prompt = ChatPromptTemplate.from_messages(get_chat_prompt_yaml(prompt_filepath))
    if st.session_state.client_id:
        loaded_messages = load_messages_from_sqlite(st.session_state.client_id, st.session_state.conversation_num)
    # Streamlit에 보여주기
        for message in loaded_messages:
            st.chat_message(message["role"]).write(message["content"])

    st.session_state.llm = ChatOpenAI(
            api_key=st.session_state.openai_api_key, 
            model=st.session_state.model,
            temperature=0.3,
            streaming=True )
    
    chain = prompt | st.session_state.llm | StrOutputParser()

    chat_message_history_chain = RunnableWithMessageHistory(
          chain,
          get_message_history_sqlitedb,
          input_messages_key= 'input',
          history_messages_key='messages',
          history_factory_config=configs_fields
    )

    config = {'configurable' : {'client_id' : st.session_state.client_id, 'conversation_num' : st.session_state.conversation_num}}

    if query := st.chat_input('Please enter your Question'):
        if not st.session_state.openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
        st.chat_message("user").write(query)
        st.write_stream(multiturn_stream_response(chat_message_history_chain, query, config))   

if __name__ == '__main__':
    main()

