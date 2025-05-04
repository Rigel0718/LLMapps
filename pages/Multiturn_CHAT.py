import streamlit as st
from langchain_core.runnables.history import RunnableWithMessageHistory
from utils import multiturn_stream_response
from message_history import get_message_history_sqlitedb, configs_fields, load_messages_from_sqlite
from chains.chains import get_vanilla_chain

MODEL = ['gpt-4o-mini', 'o3-mini']


def main():
    st.set_page_config(page_title="Multiturn-CHAT")

    # Session state 초기화
    for key in ['client_id', 'conversation_num', 'openai_api_key', 'llm']:
        if key not in st.session_state:
            st.session_state[key] = None

    

    with st.sidebar:
        st.session_state.client_id = st.text_input("USER_ID", key='cliend_id')
        st.session_state.call_user_chathistory = st.button("MAKE_NEW_CONVERSATION")
        st.session_state.conversation_num = st.text_input('Conversation_Num')
        st.text_input("OpenAI API Key", key="openai_api_key", type="password")
        "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
        st.selectbox("🤖 Select a Model", options=MODEL, key = 'model')


    
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

