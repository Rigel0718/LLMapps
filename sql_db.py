from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, Text

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

Base = declarative_base()

def create_message_model(table_name):

    class Messages(Base):
        __table_name__ = table_name
        id = Column(Integer, primary_key=True)
        session_id = Column(Text)
        conversation_title = Column(Text)
        message = Column(Text)

    return Messages

