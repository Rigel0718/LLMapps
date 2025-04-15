from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables.utils import ConfigurableFieldSpec
from langchain_core.runnables.history import RunnableWithMessageHistory

def get_message_history_sqlitedb(client_id, conversation_num):
    return SQLChatMessageHistory(
        table_name=client_id,
        session_id=conversation_num,
        connection='sqlite:///.db'
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