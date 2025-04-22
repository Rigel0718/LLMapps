from typing import Generator, List, Optional, Any
from sqlalchemy.orm import declarative_base, Mapped, mapped_column, sessionmaker, Session
from sqlalchemy import Integer, Text 
import json
from contextlib import contextmanager
from sqlalchemy import create_engine
from langchain_core.messages import messages_from_dict, BaseMessage
from abc import ABC, abstractmethod


Base = declarative_base()



engine = create_engine(url='sqlite:///.db', echo=True)
Session_local = sessionmaker(autoflush=True, bind=engine)

message_model_class = create_message_model('KK')
Base.metadata.create_all(engine)

@contextmanager
def get_db() -> Generator[Session, None, None]:
    with Session_local() as session:
        yield session

def messages(session_id) -> List[BaseMessage]: 
    with get_db() as session:
        result = session.query(message_model_class).where(
            getattr(message_model_class, 'session_id') == session_id
        ).order_by(message_model_class.id.asc())
    
    chat = []
    for msg in result:
        chat.append(messages_from_dict([json.loads(msg.message)])[0])
    return chat



# BaseMessageConverter is not packaged in LangChain, so I copied and pasted it directly 
class BaseMessageConverter(ABC):
    """Convert BaseMessage to the SQLAlchemy model."""

    @abstractmethod
    def from_sql_model(self, sql_message: Any) -> BaseMessage:
        """Convert a SQLAlchemy model to a BaseMessage instance."""
        raise NotImplementedError

    @abstractmethod
    def to_sql_model(self, message: BaseMessage, session_id: str) -> Any:
        """Convert a BaseMessage instance to a SQLAlchemy model."""
        raise NotImplementedError

    @abstractmethod
    def get_sql_model_class(self) -> Any:
        """Get the SQLAlchemy model class."""
        raise NotImplementedError
    def create_message_model(table_name):

        class Messages(Base):
            __tablename__ = table_name
            id : Mapped[int] = mapped_column(Integer, primary_key=True)
            session_id : Mapped[Optional[str]] = mapped_column(Text, nullable=True)
            conversation_title : Mapped[str] = mapped_column(Text, nullable=True)
            message : Mapped[str] = mapped_column(Text, nullable=True)

        return Messages



if __name__ == '__main__':
    table_name = 'KK'
    session_id = '0'

    user_message_list = messages(session_id=session_id)
    print(user_message_list)
