from typing import Generator
from sqlalchemy.orm import declarative_base, Mapped, mapped_column, sessionmaker, Session
from sqlalchemy import Integer, Text
import json
from contextlib import contextmanager
from sqlalchemy import create_engine
from langchain_core.messages import messages_from_dict

Base = declarative_base()

def create_message_model(table_name):

    class Messages(Base):
        __table_name__ = table_name
        id : Mapped[int] = mapped_column(Integer, primary_key=True)
        session_id : Mapped[str] = mapped_column(Text)
        conversation_title : Mapped[str] = mapped_column(Text, nullable=True)
        message : Mapped[str] = mapped_column(Text, nullable=True)

    return Messages

engine = create_engine(url='sqlite:///.db', echo=True)
Session_local = sessionmaker(autoflush=True, bind=engine)

message_model_class = create_message_model('table_sample')
Base.metadata.create_all(engine)

@contextmanager
def get_db() -> Generator[Session, None, None]:
    with Session_local() as session:
        yield session

def messages(session_id):
    with get_db() as session:
        result = session.query(message_model_class).where(
            getattr(message_model_class, 'session_id') == session_id
        ).order_by(message_model_class.id.asc())
    
    chat = []
    for msg in result:
        chat.append(messages_from_dict([json.loads(msg.message)])[0])
    return chat