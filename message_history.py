from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables.utils import ConfigurableFieldSpec
from langchain_core.chat_history import BaseChatMessageHistory
from sql_db import CustomSQLChatMessageHistory, CustomMessageConverter, create_message_model
from sqlalchemy import create_engine, inspect, text

engine = create_engine('sqlite:///customdb/custom.db')

def check_user_exists(client_id: str) -> bool:
    inspector = inspect(engine)
    return client_id in inspector.get_table_names()

def get_conversation_nums(user_id: str) -> list[str]:
    with engine.connect() as conn:
        query = text(f'SELECT DISTINCT session_id FROM "{user_id}"')
        result = conn.execute(query)
        return [row[0] for row in result.fetchall()]
    
def create_user_table_if_not_exists(user_id: str):
    model = create_message_model(user_id)
    model.__table__.create(bind=engine, checkfirst=True)

# def get_message_history_sqlitedb(client_id, conversation_num) -> BaseChatMessageHistory:
#     return SQLChatMessageHistory(
#         table_name=client_id,
#         session_id=conversation_num,
#         connection='sqlite:///customdb/custom.db',
#         custom_message_converter=CustomMessageConverter(client_id)
#     )


# def load_messages_from_sqlite(client_id: str, conversation_num: str):
#     history = SQLChatMessageHistory(
#         table_name=client_id,
#         session_id=conversation_num,
#         connection='sqlite:///customdb/custom.db',
#         custom_message_converter=CustomMessageConverter(client_id)
#     )
#     messages = history.messages
    
#     return [
#         {"role": "user" if m.type == "human" else "assistant", "content": m.content}
#         for m in messages
#     ]

def get_message_history_sqlitedb(client_id, conversation_num) -> BaseChatMessageHistory:
    return CustomSQLChatMessageHistory(
        table_name=client_id,
        session_id=conversation_num,
        connection='sqlite:///customdb/custom.db',
    )


def load_messages_from_sqlite(client_id: str, conversation_num: str):
    history = CustomSQLChatMessageHistory(
        table_name=client_id,
        session_id=conversation_num,
        connection='sqlite:///customdb/custom.db',
    )
    messages = history.messages
    
    return [
        {"role": "user" if m.type == "human" else "assistant", "content": m.content}
        for m in messages
    ]


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
