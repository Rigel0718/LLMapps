from typing import Any, Generator, List, Optional, Sequence, cast, Union, Dict
from sqlalchemy.orm import declarative_base, Mapped, mapped_column, sessionmaker, scoped_session, Session as SQLSession
from sqlalchemy import Integer, Text, create_engine, func
import json
from langchain_core.messages import messages_from_dict, BaseMessage, message_to_dict
from abc import ABC, abstractmethod
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeMeta
import contextlib
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core._api import deprecated

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
    


def create_message_model(table_name, base=declarative_base()):

    class Custom_Messages(base):
        __tablename__ = table_name
        __table_args__ = {'extend_existing': True}
        id : Mapped[int] = mapped_column(Integer, primary_key=True)
        session_id : Mapped[Optional[str]] = mapped_column(Text, nullable=True)
        conversation_title : Mapped[str] = mapped_column(Text, nullable=True)
        message : Mapped[str] = mapped_column(Text, nullable=True)

    return Custom_Messages

    
class CustomMessageConverter(BaseMessageConverter):
    """The custom message converter for SQLChatMessageHistory."""

    def __init__(self, table_name: str):
        self.model_class = create_message_model(table_name)

    def from_sql_model(self, sql_message: Any) -> BaseMessage:
        return messages_from_dict([json.loads(sql_message.message)])[0]

    def to_sql_model(self, message: BaseMessage, session_id: str, conversation_title) -> Any:
        return self.model_class(
            session_id=session_id, conversation_title=conversation_title, message=json.dumps(message_to_dict(message))
        )

    def get_sql_model_class(self) -> Any:
        return self.model_class


class CustomSQLChatMessageHistory(BaseChatMessageHistory):
    '''
    SQLChatMessageHistory의 기본 셋에서 async 제거,
    conversation_title의 column추가, 필요한 메소드 추가.
    '''
    @property
    @deprecated("0.2.2", removal="1.0", alternative="session_maker")
    def Session(self):
        return self.session_maker

    def __init__(
        self,
        session_id: str,
        conversation_title: str = None,
        connection: Union[Engine, str] = None,
        table_name: str = "message_store",
        session_id_field_name: str = "session_id",
        custom_message_converter: Optional[Any] = None,
        engine_args: Optional[Dict[str, Any]] = None,
    ):
        self.engine = create_engine(url=connection, **(engine_args or {}))
        self.session_maker = scoped_session(sessionmaker(bind=self.engine))

        self.session_id_field_name = session_id_field_name
        self.converter = custom_message_converter or CustomMessageConverter(table_name)
        self.sql_model_class = self.converter.get_sql_model_class()
        if not hasattr(self.sql_model_class, session_id_field_name):
            raise ValueError("SQL model class must have session_id column")

        self.sql_model_class.metadata.create_all(self.engine)
        self.session_id = session_id
        self.conversation_title = conversation_title or f"untitled_{session_id}"

    @property
    def messages(self) -> List[BaseMessage]:  # type: ignore
        """Retrieve all messages from db"""
        with self._make_sync_session() as session:
            result = (
                session.query(self.sql_model_class)
                .where(
                    getattr(self.sql_model_class, self.session_id_field_name)
                    == self.session_id
                )
                .order_by(self.sql_model_class.id.asc())
            )
            messages = []
            for record in result:
                messages.append(self.converter.from_sql_model(record))
            return messages

    
    def get_messages(self) -> List[BaseMessage]:
        return self.messages

    def add_message(self, message: BaseMessage) -> None:
        with self._make_sync_session() as session:
            session.add(self.converter.to_sql_model(message, self.session_id))
            session.commit()

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        with self._make_sync_session() as session:
            for message in messages:
                session.add(self.converter.to_sql_model(message, self.session_id))
            session.commit()

    def clear(self) -> None:
        with self._make_sync_session() as session:
            session.query(self.sql_model_class).filter(
                getattr(self.sql_model_class, self.session_id_field_name) == self.session_id
            ).delete()
            session.commit()

    @property
    def title_map(self) -> dict[str, str]:
        """
        SQLite에서 session_id 기준으로 하나의 conversation_title을 가져오는 매핑 딕셔너리 생성
        """
        with self._make_sync_session() as session:
            session_id_column = getattr(self.sql_model_class, self.session_id_field_name)
            rows = (
                session.query(
                    session_id_column,
                    func.min(self.sql_model_class.conversation_title)  # 가장 빠른 title 하나만
                )
                .filter(session_id_column.isnot(None))
                .group_by(session_id_column)
                .all()
            )
            return {row[0]: row[1] for row in rows}

    def get_title_map(self) -> dict[str, str]:
        return self.title_map

    def store_conversation_title(self, title: str) -> None:
        """처음 대화 생성 시, 빈 메시지와 함께 title을 저장"""
        with self._make_sync_session() as session:
            existing = session.query(self.sql_model_class).filter(
                getattr(self.sql_model_class, self.session_id_field_name) == self.session_id
            ).first()

            if not existing:
                row = self.sql_model_class(
                    session_id=self.session_id,
                    conversation_title=title,
                    message=json.dumps(message_to_dict({"type": "system", "data": ""}))
                )
                session.add(row)
                session.commit()

    def update_conversation_title(self, new_title: str) -> None:
        """기존 대화의 모든 메시지에 conversation_title을 일괄 업데이트"""
        with self._make_sync_session() as session:
            rows = session.query(self.sql_model_class).filter(
                getattr(self.sql_model_class, self.session_id_field_name) == self.session_id
            ).all()
            for row in rows:
                row.conversation_title = new_title
            session.commit()

    def get_conversation_title(self) -> Optional[str]:
        """현재 session_id에 해당하는 title을 하나 반환"""
        with self._make_sync_session() as session:
            row = session.query(self.sql_model_class).filter(
                getattr(self.sql_model_class, self.session_id_field_name) == self.session_id
            ).first()
            return row.conversation_title if row else None

    @contextlib.contextmanager
    def _make_sync_session(self) -> Generator[SQLSession, None, None]:
        with self.session_maker() as session:
            yield cast(SQLSession, session)