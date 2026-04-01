"""
Translation Service using Ollama with translategemma:4b.
Translates Indic language text (Hindi, Marathi) into English
for downstream NLP entity extraction.
"""

import logging
import re
import ollama as ollama_client
from openai import AzureOpenAI
from app.core.config import settings
from app.core.llm_routing import llm_provider_is_local

logger = logging.getLogger(__name__)


class TranslationService:
    """
    Service to translate Hindi/Marathi text to English
    using the translategemma:4b model via Ollama.
    """

    def __init__(self):
        self.model = settings.OLLAMA_TRANSLATE_MODEL
        self.base_url = settings.OLLAMA_BASE_URL
        
        if (
            not llm_provider_is_local()
            and settings.AZURE_GPT_KEY
            and settings.AZURE_GPT_ENDPOINT
        ):
            self.azure_client = AzureOpenAI(
                azure_endpoint=settings.AZURE_GPT_ENDPOINT,
                api_key=settings.AZURE_GPT_KEY,
                api_version=settings.AZURE_GPT_API_VERSION
            )
            self.deployment_name = settings.AZURE_GPT_DEPLOYMENT
            logger.info("TranslationService initialized with Azure OpenAI (Fallback)")
        else:
            self.azure_client = None
            if llm_provider_is_local():
                logger.info("TranslationService: Ollama + local GGUF only (Azure disabled).")
            else:
                logger.warning("TranslationService: Azure OpenAI credentials missing. Fallback disabled.")

    async def translate_to_english(self, text: str, source_lang: str = "hi") -> str:
        """
        Translate Indic text to English using translategemma:4b.

        Args:
            text: Input text in Hindi/Marathi/English
            source_lang: Source language code ('hi' for Hindi, 'mr' for Marathi, 'en' for English)

        Returns:
            English translated text
        """
        # If already English, skip translation
        if source_lang == "en":
            logger.info("Input is already in English, skipping translation.")
            return text

        # If the text is primarily digits, letters, and basic punctuation,
        # it doesn't need translation (e.g., PAN card "A B C D...", Phone "987 654...").
        # Translating simple numbers often causes translation models to hallucinate options.
        import re
        if re.match(r'^[\d\sA-Za-z\.,\-]*$', text.strip()):
            logger.info("Input appears to be primarily alphanumeric. Skipping translation to avoid hallucination.")
            return text

        try:
            # 1. Try Local Ollama first
            if await self.check_model_available():
                # Build the translation prompt for translategemma
                # translategemma expects a specific prompt format
                prompt = f"Translate the following {self._lang_name(source_lang)} text to English:\n\n{text}"

                logger.info(f"Translating with LOCAL {self.model}: '{text[:80]}...'")

                # Call Ollama API
                client = ollama_client.Client(host=self.base_url)
                response = client.generate(
                    model=self.model,
                    prompt=prompt,
                    options={
                        "temperature": 0.1,  # Low temperature for accurate translation
                        "num_predict": 512,
                    }
                )

                translated = response["response"].strip()
                if translated:
                    logger.info("Local translation successful.")
                    return self._clean_translation(translated)
            
            logger.warning(f"Local model {self.model} not available or empty response.")

        except Exception as e:
            logger.error(f"Local Translation error with {self.model}: {e}")

        # 2. Local GGUF (Qwen)
        if llm_provider_is_local():
            try:
                from app.core.local_llm import local_chat_complete, local_chat_gguf_configured

                if local_chat_gguf_configured():
                    prompt = (
                        f"Translate the following {self._lang_name(source_lang)} text to English. "
                        f"Return ONLY the translated English text, no labels or quotes.\n\n{text}"
                    )
                    out = local_chat_complete(
                        system="You are a professional translator. Output only the translation.",
                        user=prompt,
                        max_tokens=512,
                    ).strip()
                    if out:
                        logger.info("Local GGUF translation successful.")
                        return self._clean_translation(out)
            except Exception as ge:
                logger.warning(f"Local GGUF translation failed: {ge}")

        # 3. Azure (cloud mode only)
        if llm_provider_is_local():
            logger.warning("Translation: local backends failed; returning original text.")
            return text

        if not self.azure_client:
            logger.error("Azure translation fallback not available. Returning original text.")
            return text

        try:
            logger.info(f"Translating with AZURE ({self.deployment_name}): '{text[:80]}...'")
            prompt = f"Translate the following {self._lang_name(source_lang)} text to English. Return ONLY the translated string.\n\nText: {text}"
            
            response = self.azure_client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are a professional translator. Translate Indic text to English precisely."},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=512
            )
            
            translated = response.choices[0].message.content.strip()
            logger.info("Azure translation successful.")
            return self._clean_translation(translated)

        except Exception as e:
            logger.error(f"Azure translation error: {e}")
            return text

    def _clean_translation(self, translated: str) -> str:
        """Helper to clean LLM conversational filler and formatting."""
        import re

        prefixes_to_strip = [
            r"^here is the translation:\s*",
            r"^sure, here is the translation:\s*",
            r"^translation:\s*",
            r"^translated text:\s*",
            r"^english:\s*",
            r"^\*?\*?here are a few options.*?\*?\*?:\s*",
            r"^here are a few.*?\s*",
        ]
        for prefix in prefixes_to_strip:
            translated = re.sub(prefix, "", translated, flags=re.IGNORECASE)

        options_match = re.search(
            r"here are a few.*?(?:options|translations|context).*?:\s*(.*?)$",
            translated,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if options_match:
            translated = options_match.group(1).strip()

        bullet_match = re.search(
            r"^\s*(?:\*|-|\d+\.)\s*(?:\*\*)?(.*?)(?:\*\*)?(?:\n|$)",
            translated,
        )
        if bullet_match:
            translated = bullet_match.group(1)

        translated = re.sub(r"\s*\(.*?\)\s*$", "", translated).strip()
        return translated.strip()

    def _lang_name(self, code: str) -> str:
        """Convert language code to human-readable name."""
        lang_map = {
            "hi": "Hindi",
            "mr": "Marathi",
            "en": "English",
            "ta": "Tamil",
            "te": "Telugu",
            "bn": "Bengali",
        }
        return lang_map.get(code, "Hindi")

    async def check_model_available(self) -> bool:
        """Check if the translategemma:4b model is available in Ollama."""
        try:
            client = ollama_client.Client(host=self.base_url)
            models = client.list()
            available = [m["model"] for m in models.get("models", [])]
            return self.model in available or any(self.model.split(":")[0] in m for m in available)
        except Exception as e:
            logger.error(f"Ollama connection check failed: {e}")
            return False


translation_service = TranslationService()
