import streamlit as st
from langchain_core.runnables.history import RunnableWithMessageHistory
from utils import multiturn_stream_response
from message_history import (get_message_history_sqlitedb, configs_fields, load_messages_from_sqlite, 
                             check_user_exists, get_conversation_nums, create_user_table_if_not_exists,
                             load_conversation_title_list)
from chains.chains import get_vanilla_chain
import time

MODEL = ['gpt-4o-mini', 'o3-mini']
DBURL = 'sqlite:///customdb/custom.db'


def main():
    st.set_page_config(page_title="Multiturn-CHAT")

    # Session state 초기화
    for key in ['user_id', 'conversation_num', 'openai_api_key', 'llm']:
        if key not in st.session_state:
            st.session_state[key] = None

    if "user_check_failed" not in st.session_state:
        st.session_state.user_check_failed = False
    if "ready_to_register" not in st.session_state:
        st.session_state.ready_to_register = False

    # ✅ 기본 conversation_num = "0"
    if st.session_state.conversation_num is None:
        st.session_state.conversation_num = "0"

    with st.sidebar:
        st.header("🔐 User & Key 설정")

        if st.session_state.user_id:
            st.success(f"✅ 로그인: {st.session_state.user_id}")
            if st.button("🚪 로그아웃"):
                for key in ['user_id', 'conversation_num', 'conversation_list', 'openai_api_key',
                            'llm', 'user_check_failed', 'ready_to_register', 'chat_history']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
        else:
            input_user_id = st.text_input("USER_ID", value="")
            user_check = st.button("🔍 Check USER_ID")

            if user_check:
                if input_user_id.strip() == "":
                    st.warning("❗ USER_ID를 입력해주세요.")
                    st.stop()

                if check_user_exists(input_user_id):
                    st.session_state.user_id = input_user_id
                    st.session_state.conversation_list = get_conversation_nums(input_user_id)
                    if "0" not in st.session_state.conversation_list:
                        st.session_state.conversation_list.append("0")
                    st.session_state.user_check_failed = False
                    st.session_state.ready_to_register = False
                else:
                    st.session_state.user_id = None
                    st.session_state.user_check_failed = True
                    st.session_state.ready_to_register = True
                st.rerun()
        st.session_state.openai_api_key = st.text_input("OpenAI API Key", type="password")
        "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
        st.selectbox("🤖 Select a Model", options=MODEL, key='model')

    if st.session_state.user_check_failed:
        st.warning("❗ 등록된 user_id가 없습니다.")
        if st.session_state.ready_to_register:
            if st.button("🆕 새로운 유저로 등록하시겠습니까?"):
                create_user_table_if_not_exists(input_user_id)
                # st.success(f"✅ '{input_user_id}' 계정이 등록되었습니다. 로그인 버튼을 눌러주세요.")
                st.session_state.user_id = input_user_id
                st.session_state.conversation_list = ["0"]
                st.session_state.conversation_num = "0"
                st.session_state.user_check_failed = False
                st.session_state.ready_to_register = False
                st.success(f"✅ '{input_user_id}' 계정이 등록되었습니다.")
                time.sleep(2)
                st.rerun()

    # ✅ 로그인이 되었을 경우
    if not st.session_state.user_check_failed and st.session_state.user_id:
        # 계정의 db 정보 호출
        st.session_state.chat_history = get_message_history_sqlitedb(st.session_state.user_id, st.session_state.conversation_num)
        
        # 계정의 db에서 conversation_title 추출
        conv_list = load_conversation_title_list(st.session_state.chat_history)
        selected_conv = st.selectbox("🗂️ 선택할 conversation_num", conv_list, key="conversation_selector")

        new_conv = st.text_input("🆕 새 conversation_num 생성", key="new_conv")
        if st.button("➕ Create New Conversation"):
            st.session_state.conversation_list.append(new_conv)
            st.session_state.conversation_num = new_conv
            st.toast("✅ 새로운 대화가 생성되었습니다!", icon="🎉")
            time.sleep(3)
            st.rerun()
            
            
        # conversation_num 선택
        st.session_state.conversation_num = selected_conv or st.session_state.conversation_num
        # 메시지 불러오기
        if st.session_state.user_id and st.session_state.conversation_num and st.session_state.conversation_num.strip():
            
            loaded_messages = load_messages_from_sqlite(st.session_state.chat_history)

            for message in loaded_messages:
                st.chat_message(message["role"]).write(message["content"])

            # 모델 세팅 및 Runnable 구성
            if st.session_state.openai_api_key:
                

                chain = get_vanilla_chain(st.session_state.openai_api_key, st.session_state.model)
                chat_message_history_chain = RunnableWithMessageHistory(
                    chain,
                    get_session_history= lambda config : st.session_state.chat_history,
                    input_messages_key='input',
                    history_messages_key='messages',
                    history_factory_config=configs_fields
                )

                config = {
                    'configurable': {
                        'client_id': st.session_state.user_id,
                        'conversation_num': st.session_state.conversation_num
                    }
                }
                

                # 채팅 입력 (API 키가 있을 때만)
                if query := st.chat_input("🗨️ 질문을 입력하세요"):
                    st.chat_message("user").write(query)
                    st.write_stream(multiturn_stream_response(chat_message_history_chain, query, config))
            else :
                st.chat_input("🗨️ OpenAI 키를 입력하면 채팅이 가능합니다", disabled=True)



if __name__ == '__main__':
    main()
