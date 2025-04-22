from typing import Optional, Any
from sqlalchemy.orm import declarative_base, Mapped, mapped_column, sessionmaker, Session
from sqlalchemy import Integer, Text 
import json
from langchain_core.messages import messages_from_dict, BaseMessage, message_to_dict
from abc import ABC, abstractmethod





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
    
Base = declarative_base()


def create_message_model(table_name):

    class Custom_Messages(Base):
        __tablename__ = table_name
        id : Mapped[int] = mapped_column(Integer, primary_key=True)
        session_id : Mapped[Optional[str]] = mapped_column(Text, nullable=True)
        conversation_title : Mapped[str] = mapped_column(Text, nullable=True)
        message : Mapped[str] = mapped_column(Text, nullable=True)

    return Custom_Messages

    
class CustomMessageConverter(BaseMessageConverter):
    """The default message converter for SQLChatMessageHistory."""

    def __init__(self, table_name: str):
        self.model_class = create_message_model(table_name)

    def from_sql_model(self, sql_message: Any) -> BaseMessage:
        return messages_from_dict([json.loads(sql_message.message)])[0]

    def to_sql_model(self, message: BaseMessage, session_id: str) -> Any:
        return self.model_class(
            session_id=session_id, message=json.dumps(message_to_dict(message))
        )

    def get_sql_model_class(self) -> Any:
        return self.model_class