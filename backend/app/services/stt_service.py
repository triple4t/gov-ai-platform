"""
Speech-to-Text (STT) Service.
Handles audio file transcription using Deepgram API for blazing fast processing.
"""

import os
import logging
import asyncio
from deepgram import (
    DeepgramClient,
    PrerecordedOptions,
    FileSource,
)
from app.core.config import settings

logger = logging.getLogger(__name__)


class STTService:
    """Service to transcribe audio files to text using Deepgram API."""

    def __init__(self):
        # Initialize Deepgram client with API key from settings
        self.api_key = settings.DEEPGRAM_API_KEY
        if not self.api_key:
            logger.warning("DEEPGRAM_API_KEY is not set. STT will fail.")
        else:
            self.deepgram = DeepgramClient(self.api_key)

    async def transcribe_audio(self, audio_bytes: bytes, file_extension: str = "webm", language: str = "hi-IN") -> str:
        """
        Transcribe audio bytes to text using Deepgram.

        Args:
            audio_bytes: Raw audio file bytes
            file_extension: The format of the audio file (webm, mp3, wav, etc.)
            language: BCP-47 language code (hi-IN for Hindi, mr-IN for Marathi, en-IN for Indian English)

        Returns:
            Transcribed text string
        """
        if not self.api_key:
            raise ValueError("Deepgram API key is missing. Check .env file.")

        # Map frontend language codes to Deepgram supported language codes
        dg_language = "hi" # Default to Hindi
        if "hi" in language.lower():
            dg_language = "hi"
        elif "mr" in language.lower():
            dg_language = "mr"
        elif "en" in language.lower():
            dg_language = "en-IN"

        try:
            # Prepare the audio payload
            # Deepgram handles webm natively, so no more heavy PyDub conversions needed!
            payload: FileSource = {
                "buffer": audio_bytes,
                "mimetype": f"audio/{file_extension.lower()}" if file_extension.lower() != "m4a" else "audio/mp4",
            }

            # Configure transcription options
            options = PrerecordedOptions(
                model="nova-3",
                language=dg_language,
                smart_format=True,
                punctuate=True,
                paragraphs=False,
            )

            # Call Deepgram API (running synchronously inside a thread pool to avoid blocking)
            logger.info(f"Sending audio to Deepgram API (lang: {dg_language})...")
            
            # Use asyncio.to_thread because the deepgram SDK's transcribe_file is synchronous
            response = await asyncio.to_thread(
                self.deepgram.listen.rest.v("1").transcribe_file,
                payload, 
                options,
                timeout=10 # Fail fast if network hangs
            )
            
            # Extract transcript from the JSON response
            transcript = response["results"]["channels"][0]["alternatives"][0]["transcript"]
            
            logger.info(f"Deepgram STT Transcription ({dg_language}): '{transcript}'")
            
            if not transcript or transcript.strip() == "":
                 logger.warning("Deepgram returned an empty transcript. User might not have spoken.")
                 raise ValueError("Could not hear anything. Please speak clearly.")
                 
            return transcript.strip()

        except Exception as e:
            logger.error(f"Deepgram STT processing error: {e}")
            raise RuntimeError(f"Audio transcription failed: {e}")


stt_service = STTService()
