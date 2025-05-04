import streamlit as st
from langchain_core.runnables.history import RunnableWithMessageHistory
from utils import multiturn_stream_response
from message_history import (get_message_history_sqlitedb, configs_fields, load_messages_from_sqlite, check_user_exists, get_conversation_nums)
from chains.chains import get_vanilla_chain

MODEL = ['gpt-4o-mini', 'o3-mini']


def main():
    st.set_page_config(page_title="Multiturn-CHAT")

    # Session state 초기화
    for key in ['client_id', 'conversation_num', 'openai_api_key', 'llm']:
        if key not in st.session_state:
            st.session_state[key] = None

    

    with st.sidebar:
        st.header("🔐 User & Key 설정")
        input_user_id = st.text_input("USER_ID", value=st.session_state.client_id or "")
        user_check = st.button("🔍 Check USER_ID")
        st.session_state.openai_api_key = st.text_input("OpenAI API Key", type="password")
        "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
        st.selectbox("🤖 Select a Model", options=MODEL, key='model')
    
    if user_check:
        if check_user_exists(input_user_id):
            st.session_state.client_id = input_user_id
            st.session_state.conversation_list = get_conversation_nums(input_user_id)
        else:
            st.session_state.client_id = None
            st.warning("❗ 등록된 user_id가 없습니다.")

    
    if st.session_state.client_id:
        loaded_messages = load_messages_from_sqlite(st.session_state.client_id, st.session_state.conversation_num)

        for message in loaded_messages:
            st.chat_message(message["role"]).write(message["content"])

    
    chain = get_vanilla_chain(st.session_state.openai_api_key, st.session_state.model)

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

