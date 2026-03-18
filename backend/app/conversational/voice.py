"""Voice: Sarvam STT/TTS and recording history."""

import base64
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
import httpx

from typing import Optional

from app.core.config import settings
from app.conversational.database import get_db, User, VoiceRecording
from app.conversational.auth import get_current_user_optional

router = APIRouter()

ANONYMOUS_USER_ID = 0

LANG_TO_SARVAM = {
    "hi": "hi-IN", "en": "en-IN", "bn": "bn-IN", "te": "te-IN", "mr": "mr-IN",
    "ta": "ta-IN", "ur": "ur-IN", "gu": "gu-IN", "kn": "kn-IN", "or": "od-IN",
    "pa": "pa-IN", "as": "as-IN", "ml": "ml-IN",
}
SARVAM_API_BASE = "https://api.sarvam.ai"


async def sarvam_speech_to_text(audio_bytes: bytes, language: str = "hi") -> str:
    if not settings.SARVAM_API_KEY:
        raise HTTPException(status_code=503, detail="Sarvam API key not configured")
    lang_code = LANG_TO_SARVAM.get(language, "hi-IN")
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            f"{SARVAM_API_BASE}/speech-to-text",
            headers={"api-subscription-key": settings.SARVAM_API_KEY},
            files={"file": ("audio.webm", audio_bytes, "audio/webm")},
            data={
                "model": "saaras:v3",
                "language_code": lang_code,
                "with_timestamps": "false",
                "with_disfluencies": "false",
            },
        )
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=f"Sarvam STT error: {response.text}")
    return response.json().get("transcript", "")


async def sarvam_text_to_speech(text: str, language: str = "hi") -> bytes:
    if not settings.SARVAM_API_KEY:
        raise HTTPException(status_code=503, detail="Sarvam API key not configured")
    lang_code = LANG_TO_SARVAM.get(language, "hi-IN")
    # Sarvam TTS: only bulbul:v2 and bulbul:v3 are valid; v3 has 30+ voices (default speaker: shubh)
    payload = {
        "text": text[:2500],
        "target_language_code": lang_code,
        "model": "bulbul:v3",
        "speaker": "shubh",
        "pace": 1.0,
        "speech_sample_rate": "24000",
        "output_audio_codec": "mp3",
    }
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            f"{SARVAM_API_BASE}/text-to-speech",
            headers={
                "api-subscription-key": settings.SARVAM_API_KEY,
                "Content-Type": "application/json",
            },
            json=payload,
        )
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=f"Sarvam TTS error: {response.text}")
    data = response.json()
    audio_b64 = data.get("audios", [None])[0]
    if not audio_b64:
        raise HTTPException(status_code=500, detail="No audio returned from Sarvam TTS")
    return base64.b64decode(audio_b64)


@router.post("/speech-to-text")
async def convert_speech_to_text(
    file: UploadFile = File(...),
    language: str = Form(default="hi"),
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: AsyncSession = Depends(get_db),
):
    audio_bytes = await file.read()
    try:
        text = await sarvam_speech_to_text(audio_bytes, language)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STT failed: {e}")
    user_id = current_user.id if current_user else ANONYMOUS_USER_ID
    recording = VoiceRecording(
        user_id=user_id,
        filename=file.filename or "recording.webm",
        file_path="",
        transcription=text,
        language=language,
    )
    db.add(recording)
    await db.commit()
    await db.refresh(recording)
    return {"text": text, "language": language, "recording_id": recording.id}


@router.post("/text-to-speech")
async def convert_text_to_speech(
    text: str = Form(...),
    language: str = Form(default="hi"),
    current_user: Optional[User] = Depends(get_current_user_optional),
):
    try:
        audio_bytes = await sarvam_text_to_speech(text, language)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS failed: {e}")
    return Response(content=audio_bytes, media_type="audio/mpeg")


@router.get("/recordings")
async def get_voice_recordings(
    limit: int = 10,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: AsyncSession = Depends(get_db),
):
    user_id = current_user.id if current_user else ANONYMOUS_USER_ID
    r = await db.execute(
        select(VoiceRecording)
        .where(VoiceRecording.user_id == user_id)
        .order_by(desc(VoiceRecording.created_at))
        .limit(limit)
    )
    recordings = r.scalars().all()
    return [
        {
            "id": rec.id,
            "filename": rec.filename,
            "transcription": rec.transcription,
            "language": rec.language,
            "created_at": rec.created_at.isoformat(),
        }
        for rec in recordings
    ]
