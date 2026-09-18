"""PostgreSQL-backed LangChain message history scoped to the authenticated user."""
from __future__ import annotations

import contextvars
import uuid

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from sqlalchemy import select

from app.db import get_session_factory
from app.models import ChatMessage, ChatSession

_current_user_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("current_user_id", default=None)


def set_current_user(user_id: str):
    return _current_user_id.set(user_id)


def reset_current_user(token) -> None:
    _current_user_id.reset(token)


class PostgresChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id: str, user_id: str):
        self.session_id = session_id
        self.user_id = user_id

    @property
    def messages(self) -> list[BaseMessage]:
        with get_session_factory()() as db:
            session = db.scalar(
                select(ChatSession).where(ChatSession.id == self.session_id, ChatSession.user_id == self.user_id)
            )
            if session is None:
                return []
            rows = db.scalars(
                select(ChatMessage).where(ChatMessage.session_id == self.session_id).order_by(ChatMessage.created_at)
            ).all()
            result: list[BaseMessage] = []
            for row in rows:
                if row.role == "user":
                    result.append(HumanMessage(content=row.content))
                elif row.role == "assistant":
                    result.append(AIMessage(content=row.content))
                elif row.role == "system":
                    result.append(SystemMessage(content=row.content))
                elif row.role == "tool":
                    result.append(ToolMessage(content=row.content, tool_call_id="persisted"))
            return result

    def add_message(self, message: BaseMessage) -> None:
        with get_session_factory()() as db:
            session = db.scalar(
                select(ChatSession).where(ChatSession.id == self.session_id, ChatSession.user_id == self.user_id)
            )
            if session is None:
                raise ValueError("Chat session does not belong to the authenticated user")
            role = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}.get(
                message.type, message.type
            )
            db.add(ChatMessage(id=str(uuid.uuid4()), session_id=self.session_id, role=role, content=str(message.content)))
            db.commit()

    def add_messages(self, messages: list[BaseMessage]) -> None:
        if not messages:
            return
        with get_session_factory()() as db:
            session = db.scalar(
                select(ChatSession).where(ChatSession.id == self.session_id, ChatSession.user_id == self.user_id)
            )
            if session is None:
                raise ValueError("Chat session does not belong to the authenticated user")
            role_map = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}
            for message in messages:
                db.add(
                    ChatMessage(
                        id=str(uuid.uuid4()),
                        session_id=self.session_id,
                        role=role_map.get(message.type, message.type),
                        content=str(message.content),
                    )
                )
            db.commit()

    def clear(self) -> None:
        with get_session_factory()() as db:
            session = db.scalar(
                select(ChatSession).where(ChatSession.id == self.session_id, ChatSession.user_id == self.user_id)
            )
            if session is None:
                return
            db.query(ChatMessage).filter(ChatMessage.session_id == self.session_id).delete()
            db.commit()


def get_session_history(session_id: str) -> PostgresChatMessageHistory:
    user_id = _current_user_id.get()
    if not user_id:
        raise RuntimeError("Authenticated user context is required for chat history")
    return PostgresChatMessageHistory(session_id, user_id)


def session_exists(session_id: str, user_id: str) -> bool:
    with get_session_factory()() as db:
        return db.scalar(select(ChatSession.id).where(ChatSession.id == session_id, ChatSession.user_id == user_id)) is not None


def serialize_history(session_id: str, user_id: str) -> list[dict[str, str]]:
    history = PostgresChatMessageHistory(session_id, user_id)
    return [{"role": m.type if m.type != "human" else "user", "content": str(m.content)} for m in history.messages]
