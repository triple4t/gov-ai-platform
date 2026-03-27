"""Chat API: send message, list/delete conversations, Azure OpenAI."""

import uuid
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, delete
from pydantic import BaseModel

from app.core.config import settings
from app.conversational.database import get_db, User, Conversation, Message
from app.conversational.auth import get_current_user, get_current_user_optional

router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    language: Optional[str] = None
    session_id: Optional[str] = None
    message_type: str = "text"


class ChatResponse(BaseModel):
    response: str
    language: str
    session_id: str
    sources: List[str] = []
    suggestions: List[str] = []
    translated_response: Optional[str] = None


class ConversationHistory(BaseModel):
    id: int
    session_id: str
    language: str
    messages: List[dict]
    created_at: datetime


async def get_or_create_conversation(
    db: AsyncSession, user_id: int, session_id: Optional[str], language: str
) -> Conversation:
    if session_id:
        r = await db.execute(
            select(Conversation).where(
                Conversation.user_id == user_id,
                Conversation.session_id == session_id,
            )
        )
        conv = r.scalar_one_or_none()
        if conv:
            return conv
    conv = Conversation(
        user_id=user_id,
        session_id=session_id or str(uuid.uuid4()),
        language=language,
    )
    db.add(conv)
    await db.commit()
    await db.refresh(conv)
    return conv


def _get_azure_chat_client():
    from openai import AzureOpenAI
    if not settings.AZURE_GPT_KEY or not settings.AZURE_GPT_ENDPOINT:
        return None
    return AzureOpenAI(
        azure_endpoint=settings.AZURE_GPT_ENDPOINT.rstrip("/"),
        api_key=settings.AZURE_GPT_KEY,
        api_version=settings.AZURE_GPT_API_VERSION,
    )


# Anonymous user id when no auth (session-only conversations)
ANONYMOUS_USER_ID = 0


@router.post("/send", response_model=ChatResponse)
async def send_message(
    request: ChatRequest,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: AsyncSession = Depends(get_db),
):
    user_id = current_user.id if current_user else ANONYMOUS_USER_ID
    language = (
        request.language
        or (getattr(current_user, "language_preference", None) if current_user else None)
        or "en"
    )
    if language not in settings.SUPPORTED_LANGUAGES:
        language = "en"

    conversation = await get_or_create_conversation(
        db, user_id, request.session_id, language
    )

    user_message = Message(
        conversation_id=conversation.id,
        role="user",
        content=request.message,
        language=language,
        message_type=request.message_type,
    )
    db.add(user_message)

    r = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation.id)
        .order_by(Message.created_at)
        .limit(10)
    )
    history = r.scalars().all()

    messages = [
        {
            "role": "system", 
            "content": (
                f"You are a helpful government assistant. Reply in {language}. "
                "IMPORTANT RULES: "
                "1. Structure your answer point-wise so it is easy to read. "
                "2. When one major component or section is finished, draw a straight line using '─────────────────────────'. "
                "3. Use the '🔹' symbol for main section titles or main headers. "
                "4. Use the '•' symbol or numbers (1., 2.) for lists and sub-points. "
                "5. DO NOT use raw Markdown formatting (strictly NO **, NO ###, NO ---). "
                "6. Keep the language extremely simple and easy to understand for a common citizen."
            )
        }
    ]
    for msg in history:
        messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": request.message})

    client = _get_azure_chat_client()
    if not client:
        raise HTTPException(status_code=503, detail="Azure OpenAI not configured")

    try:
        response = client.chat.completions.create(
            model=settings.AZURE_GPT_DEPLOYMENT,
            messages=messages,
        )
        content = (response.choices[0].message.content or "").strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

    ai_message = Message(
        conversation_id=conversation.id,
        role="assistant",
        content=content,
        language=language,
        message_type="text",
        message_metadata={},
    )
    db.add(ai_message)
    await db.commit()

    return ChatResponse(
        response=content,
        language=language,
        session_id=conversation.session_id,
        sources=[],
        suggestions=[],
        translated_response=None,
    )


@router.get("/history")
async def get_conversation_history(
    session_id: str,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: AsyncSession = Depends(get_db),
):
    """Get messages for a conversation by session_id (no auth required for anonymous session)."""
    user_id = current_user.id if current_user else ANONYMOUS_USER_ID
    r = await db.execute(
        select(Conversation).where(
            Conversation.user_id == user_id,
            Conversation.session_id == session_id,
        )
    )
    conv = r.scalar_one_or_none()
    if not conv:
        return {"session_id": session_id, "messages": [], "language": "hi"}
    r2 = await db.execute(
        select(Message).where(Message.conversation_id == conv.id).order_by(Message.created_at)
    )
    messages = r2.scalars().all()
    return {
        "session_id": conv.session_id,
        "language": conv.language,
        "messages": [
            {"role": m.role, "content": m.content}
            for m in messages
        ],
    }


@router.get("/conversations", response_model=List[ConversationHistory])
async def get_conversations(
    limit: int = 10,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    r = await db.execute(
        select(Conversation)
        .where(Conversation.user_id == current_user.id)
        .order_by(desc(Conversation.updated_at))
        .limit(limit)
    )
    conversations = r.scalars().all()
    result = []
    for conv in conversations:
        r2 = await db.execute(
            select(Message).where(Message.conversation_id == conv.id).order_by(Message.created_at)
        )
        messages = r2.scalars().all()
        result.append(
            ConversationHistory(
                id=conv.id,
                session_id=conv.session_id,
                language=conv.language,
                messages=[
                    {
                        "id": m.id,
                        "role": m.role,
                        "content": m.content,
                        "language": m.language,
                        "message_type": m.message_type,
                        "created_at": m.created_at.isoformat(),
                    }
                    for m in messages
                ],
                created_at=conv.created_at,
            )
        )
    return result


@router.delete("/conversation/{session_id}")
async def delete_conversation(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    r = await db.execute(
        select(Conversation).where(
            Conversation.user_id == current_user.id,
            Conversation.session_id == session_id,
        )
    )
    conv = r.scalar_one_or_none()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    await db.execute(delete(Message).where(Message.conversation_id == conv.id))
    await db.delete(conv)
    await db.commit()
    return {"message": "Conversation deleted"}


@router.get("/languages")
async def get_languages():
    return {"languages": settings.SUPPORTED_LANGUAGES, "default": "hi"}
