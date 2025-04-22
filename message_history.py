from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables.utils import ConfigurableFieldSpec
from sql_db import CustomMessageConverter

def get_message_history_sqlitedb(client_id, conversation_num):
    return SQLChatMessageHistory(
        table_name=client_id,
        session_id=conversation_num,
        connection='sqlite:///custom.db',
        custom_message_converter=CustomMessageConverter(client_id)
    )

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


def load_messages_from_sqlite(client_id: str, conversation_num: str):
    history = SQLChatMessageHistory(
        table_name=client_id,
        session_id=conversation_num,
        connection='sqlite:///custom.db' 
    )
    messages = history.messages
    
    return [
        {"role": "user" if m.type == "human" else "assistant", "content": m.content}
        for m in messages
    ]