from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables.utils import ConfigurableFieldSpec
from langchain_core.chat_history import BaseChatMessageHistory
from sql_db import CustomSQLChatMessageHistory, CustomMessageConverter, create_message_model
from sqlalchemy import create_engine, inspect, text
from typing import List
from utils import get_next_conversation_num
engine = create_engine('sqlite:///customdb/custom.db')

def check_user_exists(client_id: str) -> bool:
    with engine.connect() as conn:
        inspector = inspect(conn)
        table_names = inspector.get_table_names()
        print(f"[DEBUG] All tables: {table_names}")
        return client_id in table_names
    
def get_conversation_nums(user_id: str) -> list[str]:
    with engine.connect() as conn:
        query = text(f'SELECT DISTINCT session_id FROM "{user_id}"')
        result = conn.execute(query)
        return [row[0] for row in result.fetchall()]
    
def create_user_table_if_not_exists(user_id: str):
    model = create_message_model(user_id)
    # 모델 메타데이터에서 테이블을 명시적으로 생성
    with engine.begin() as conn: 
        model.__table__.create(bind=conn, checkfirst=True)


def get_message_history_sqlitedb(client_id, conversation_num) -> BaseChatMessageHistory:
    return CustomSQLChatMessageHistory(
        table_name=client_id,
        session_id=conversation_num,
        connection='sqlite:///customdb/custom.db',
    )


def load_messages_from_sqlite(chat_history : CustomSQLChatMessageHistory):
    
    messages = chat_history.messages
    
    return [
        {"role": "user" if m.type == "human" else "assistant", "content": m.content}
        for m in messages
    ]

def load_conversation_title_list(chat_history : CustomSQLChatMessageHistory) -> List[str]:

    title_map = chat_history.title_map

    return [title_map[i] for i in range(len(title_map))]

def create_new_conversation(chat_history : CustomSQLChatMessageHistory, conversation_list=None) -> str:
    _conversation_list = conversation_list or load_conversation_title_list(chat_history)
    next_conv = get_next_conversation_num(_conversation_list)
    # store_conversation_title(
    #     user_id=user_id,
    #     conversation_num=next_conv,
    #     title=f"untitled_{next_conv}"
    # )
    return next_conv

configs_fields =[
    ConfigurableFieldSpec(
        id='client_id',
        annotation=str,
        name='Client ID',
        default='',
        is_shared=True
    ),
    ConfigurableFieldSpec(
        id='conversation_num',
        annotation=str,
        name='Conversation_NUM',
        default='',
        is_shared=True
    )
]
