from typing import Generator
from sqlalchemy.orm import declarative_base, Mapped, mapped_column, sessionmaker, Session
from sqlalchemy import Integer, Text

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

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

def get_db() -> Generator[Session, None, None]:
    local_session = Session_local()
    try:
        yield local_session
    finally:
        local_session.close()
