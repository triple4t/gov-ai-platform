"""
Entity Extraction Service.
Uses an LLM (via Ollama) to extract the 6 target form fields
from translated English text.
"""

import json
import logging
import re
import ollama
import mlflow
from openai import AzureOpenAI
from app.services.prompt_registry import get_extraction_prompt
from app.core.config import settings
from app.core.llm_routing import llm_provider_is_local
from app.models.schemas import FormEntities

logger = logging.getLogger(__name__)




class ExtractionService:
    """
    Service to extract structured form entities from English text
    using Azure OpenAI ChatGPT.
    """

    def __init__(self):
        # 1. Azure OpenAI (only when LLM_PROVIDER=azure)
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
            logger.info("ExtractionService initialized with Azure OpenAI (Fallback)")
        else:
            self.azure_client = None
            if llm_provider_is_local():
                logger.info("ExtractionService: using Ollama + local GGUF only (Azure disabled).")
            else:
                logger.warning("Azure OpenAI credentials missing. Fallback disabled.")

        # 2. Check Local Ollama (Primary for Speed)
        try:
            self.ollama_model = settings.OLLAMA_EXTRACT_MODEL
            # Ping to check if running
            ollama.list()
            self.has_ollama = True
            logger.info(f"ExtractionService enabled Local GPU (Ollama: {self.ollama_model})")
            # Pre-warm the model
            self._prewarm_model()
        except Exception as e:
            self.has_ollama = False
            if llm_provider_is_local():
                logger.warning(
                    "Ollama local service not found: %s. Entity extraction will use local GGUF when possible.",
                    e,
                )
            else:
                logger.warning(f"Ollama local service not found: {e}. All requests will go to Azure.")

    def _prewarm_model(self):
        """Send a tiny request to load the model into GPU memory."""
        try:
            logger.info(f"Warming up local model {self.ollama_model}...")
            ollama.chat(model=self.ollama_model, messages=[{"role":"user","content":"hi"}], keep_alive="5m")
            logger.info("Local model warmed up and ready.")
        except Exception as e:
             logger.debug(f"Warmup failed: {e}")


    @mlflow.trace(name="extract_entities")
    async def extract_entities(self, source_text: str) -> FormEntities:
        """
        Extract the 6 form entities.
        Tries Local GPU first, then falls back to Azure if local fails or finds nothing.
        """
        prompt = get_extraction_prompt(source_text)
        raw_output = None
        entities = FormEntities()

        # --- STEP 1: Try Local Llama (GPU) ---
        if self.has_ollama:
            try:
                logger.info(f"Extracting entities with LOCAL GPU ({self.ollama_model})...")
                response = ollama.chat(
                    model=self.ollama_model,
                    messages=[
                        {"role": "system", "content": "You are a precise data extraction assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    format="json",
                    options={"temperature": 0},
                    keep_alive="5m"
                )
                raw_output = response['message']['content'].strip()
                entities_dict = self._parse_llm_json(raw_output)
                entities = FormEntities(**entities_dict)
                
                # If we found at least one field, we're good
                if self.count_filled_fields(entities) > 0:
                    logger.info("Local GPU extraction successful (found data).")
                    return entities
                else:
                    logger.info("Local GPU returned empty data. Falling back to Azure...")
            except Exception as e:
                logger.warning(f"Local GPU extraction failed: {e}. Trying other backends...")

        # --- STEP 2: Local GGUF (Qwen) ---
        if llm_provider_is_local():
            try:
                from app.core.local_llm import local_chat_complete, local_chat_gguf_configured

                if local_chat_gguf_configured():
                    logger.info("Extracting entities with local GGUF (Qwen)...")
                    raw_output = local_chat_complete(
                        system="You are a helpful data extraction assistant that only outputs valid JSON.",
                        user=prompt,
                        max_tokens=1024,
                    )
                    entities_dict = self._parse_llm_json(raw_output)
                    entities = FormEntities(**entities_dict)
                    if self.count_filled_fields(entities) > 0:
                        logger.info("Local GGUF extraction successful (found data).")
                        return entities
            except Exception as e:
                logger.warning(f"Local GGUF extraction failed: {e}")

        # --- STEP 3: Azure (cloud mode only) ---
        if llm_provider_is_local():
            if self.count_filled_fields(entities) == 0:
                logger.warning("Entity extraction: local backends produced no filled fields.")
            return entities

        if not self.azure_client:
            logger.error("Azure fallback not available and no other extractor returned entities.")
            return entities

        try:
            logger.info(f"Extracting entities with AZURE ({self.deployment_name})...")
            response = self.azure_client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are a helpful data extraction assistant that only outputs valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=512,
                response_format={ "type": "json_object" }
            )
            raw_output = response.choices[0].message.content.strip()
            entities_dict = self._parse_llm_json(raw_output)
            entities = FormEntities(**entities_dict)
            return entities
        except Exception as e:
            logger.error(f"Azure extraction error: {e}")
            return entities

    def _parse_llm_json(self, raw_output: str) -> dict:
        """
        Robustly parse JSON output from the LLM.
        Handles cases where the LLM wraps JSON in markdown code blocks.
        """
        if not raw_output:
            return {}
            
        # Try direct JSON parse first
        try:
            return json.loads(raw_output)
        except json.JSONDecodeError:
            pass

        # Try extracting JSON from markdown code block
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_output)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try extracting anything that looks like a JSON object
        json_match = re.search(r"\{[\s\S]*\}", raw_output)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        logger.warning(f"Could not parse LLM output as JSON: {raw_output}")
        return {}

    def count_filled_fields(self, entities: FormEntities) -> int:
        """Count how many of the 6 fields have been successfully extracted."""
        count = 0
        data = entities.model_dump()
        for key, value in data.items():
            if value is not None and str(value).strip() != "":
                count += 1
        return count

    # =========================================================================
    # TRANSLATION & LOCAL EXTRACTION ENGINE
    # =========================================================================

    def _translate_number_words(self, text: str) -> str:
        """Convert ALL Marathi/Hindi/English number words, months, and Devanagari digits to standard digits."""
        mapping = {
            # --- Marathi numbers ---
            'शून्य': '0', 'एक': '1', 'दोन': '2', 'तीन': '3', 'चार': '4',
            'पाच': '5', 'सहा': '6', 'सात': '7', 'आठ': '8', 'नऊ': '9', 'दहा': '10',
            'अकरा': '11', 'बारा': '12', 'तेरा': '13', 'चौदा': '14', 'पंधरा': '15',
            'सोळा': '16', 'सतरा': '17', 'अठरा': '18', 'एकोणीस': '19', 'वीस': '20',
            'तीस': '30', 'चाळीस': '40', 'पन्नास': '50', 'साठ': '60', 'सत्तर': '70',
            'ऐंशी': '80', 'नव्वद': '90', 'शंभर': '100',
            # --- Hindi numbers ---
            'दो': '2', 'पाँच': '5', 'पांच': '5', 'छह': '6', 'छः': '6', 'नौ': '9', 'दस': '10',
            'ग्यारह': '11', 'बारह': '12', 'तेरह': '13', 'चौदह': '14', 'पंद्रह': '15', 'सोलह': '16', 
            'सत्रह': '17', 'अठारह': '18', 'उन्नीस': '19', 'बीस': '20',
            'इक्कीस': '21', 'बाईस': '22', 'तेईस': '23', 'चौबीस': '24', 'पच्चीस': '25',
            'छब्बीस': '26', 'सत्ताईस': '27', 'अट्ठाईस': '28', 'उनतीस': '29',
            'इकतीस': '31', 'बत्तीस': '32', 'तैंतीस': '33', 'चौंतीस': '34', 'पैंतीस': '35',
            'छत्तीस': '36', 'सैंतीस': '37', 'अड़तीस': '38', 'उनतालीस': '39',
            'चालीस': '40', 'इकतालीस': '41', 'बयालीस': '42', 'तैंतालीस': '43', 'चवालीस': '44',
            'पैंतालीस': '45', 'छियालीस': '46', 'सैंतालीस': '47', 'अड़तालीस': '48', 'उनचास': '49',
            'पचास': '50', 'इक्यावन': '51', 'बावन': '52', 'तिरपन': '53', 'चौवन': '54',
            'पचपन': '55', 'छप्पन': '56', 'सत्तावन': '57', 'अट्ठावन': '58', 'उनसठ': '59',
            'साठ': '60', 'इकसठ': '61', 'बासठ': '62', 'तिरसठ': '63', 'चौंसठ': '64',
            'पैंसठ': '65', 'छियासठ': '66', 'सड़सठ': '67', 'अड़सठ': '68', 'उनहत्तर': '69',
            'सत्तर': '70', 'इकहत्तर': '71', 'बहत्तर': '72', 'तिहत्तर': '73', 'चौहत्तर': '74',
            'पचहत्तर': '75', 'छिहत्तर': '76', 'सतहत्तर': '77', 'अठहत्तर': '78', 'उनासी': '79',
            'अस्सी': '80', 'इक्यासी': '81', 'बयासी': '82', 'तिरासी': '83', 'चौरासी': '84',
            'पचासी': '85', 'छियासी': '86', 'सतासी': '87', 'अठासी': '88', 'नवासी': '89',
            'नब्बे': '90', 'इक्यानवे': '91', 'बानवे': '92', 'तिरानवे': '93', 'चौरानवे': '94',
            'पचानवे': '95', 'छियानवे': '96', 'सत्तानवे': '97', 'अट्ठानवे': '98', 'निन्यानवे': '99',
            # --- Big numbers ---
            'सौ': '100', 'हज़ार': '1000', 'हजार': '1000',
            # --- Devanagari Digits ---
            '०': '0', '१': '1', '२': '2', '३': '3', '४': '4', '५': '5', '६': '6', '७': '7', '८': '8', '९': '9',
            # --- Months (Marathi/Hindi/English) ---
            'जानेवारी': '01', 'फेब्रुवारी': '02', 'मार्च': '03', 'एप्रिल': '04', 'मे': '05', 'जून': '06', 
            'जुलै': '07', 'ऑगस्ट': '08', 'सप्टेंबर': '09', 'ऑक्टोबर': '10', 'नोव्हेंबर': '11', 'डिसेंबर': '12',
            'जनवरी': '01', 'फरवरी': '02', 'अप्रैल': '04', 'मई': '05', 'जुलाई': '07', 'अगस्त': '08', 
            'सितंबर': '09', 'अक्टूबर': '10', 'नवंबर': '11', 'दिसंबर': '12',
            'january': '01', 'february': '02', 'march': '03', 'april': '04', 'may': '05', 'june': '06', 
            'july': '07', 'august': '08', 'september': '09', 'october': '10', 'november': '11', 'december': '12',
            # --- English number words (for age: "twenty one" -> digits) ---
            'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5', 'six': '6',
            'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10', 'eleven': '11', 'twelve': '12',
            'thirteen': '13', 'fourteen': '14', 'fifteen': '15', 'sixteen': '16', 'seventeen': '17',
            'eighteen': '18', 'nineteen': '19', 'twenty': '20', 'thirty': '30', 'forty': '40', 'fifty': '50',
            'sixty': '60', 'seventy': '70', 'eighty': '80', 'ninety': '90', 'hundred': '100',
            # --- Phonetic English Letters (A-Z) ---
            'ए': 'A', 'बी': 'B', 'सी': 'C', 'डी': 'D', 'ई': 'E', 'एफ': 'F', 'जी': 'G', 'एच': 'H', 'आय': 'I', 'जे': 'J', 
            'के': 'K', 'एल': 'L', 'एम': 'M', 'एन': 'N', 'ओ': 'O', 'पी': 'P', 'क्यू': 'Q', 'आर': 'R', 'एस': 'S', 'टी': 'T', 
            'यू': 'U', 'व्ही': 'V', 'डब्ल्यू': 'W', 'एक्स': 'X', 'वाय': 'Y', 'झेड': 'Z',
        }
        
        cleaned = text.lower()
        # Sort keys by length descending to replace longest patterns first
        for word in sorted(mapping.keys(), key=len, reverse=True):
            if word in cleaned:
                cleaned = cleaned.replace(word, mapping[word])
            
        # Stitch adjacent digit groups: "11 12 13 14" -> "11121314"
        # Combine multiple spaces/dots/commas into a single space
        cleaned = re.sub(r'[,\.\s]+', ' ', cleaned)
        # Iteratively join adjacent digits
        for _ in range(10):
            new_cleaned = re.sub(r'(\d)\s+(\d)', r'\1\2', cleaned)
            if new_cleaned == cleaned:
                break
            cleaned = new_cleaned
        
        return cleaned.strip()


    def _local_extract_pan(self, text: str) -> str:
        """Extract PAN card from pre-translated text. PAN = 5 letters + 4 digits + 1 optional letter."""
        # Cleaned should now have letters (from phonetic) and digits
        cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
        logger.info(f"[PAN] Cleaned string for regex: '{cleaned}'")
        
        # Try full PAN first (5 letters + 4 digits + 1 letter)
        match = re.search(r'[A-Z]{5}\d{4}[A-Z]', cleaned)
        if match:
            return match.group(0)
        # Try partial PAN (5 letters + 4 digits, missing last letter)
        match = re.search(r'[A-Z]{5}\d{4}', cleaned)
        if match:
            return match.group(0)
        # Try even more partial (some letters + some digits)
        match = re.search(r'([A-Z]*)\d{4}([A-Z]*)', cleaned)
        if match:
            return match.group(0)
        match = re.search(r'[A-Z]{3,5}\d{2,4}[A-Z]?', cleaned)
        if match and len(match.group(0)) >= 4:
            return match.group(0)
        return ""

    def _parse_hindi_number(self, text: str) -> int:
        """Convert a Marathi/Hindi/English number phrase to integer."""
        # Mapping now includes Marathi words
        num_map = {
            'शून्य': 0, 'एक': 1, 'दोन': 2, 'दो': 2, 'तीन': 3, 'चार': 4,
            'पाच': 5, 'पाँच': 5, 'पांच': 5, 'सहा': 6, 'छह': 6, 'छः': 6,
            'सात': 7, 'आठ': 8, 'नऊ': 9, 'नौ': 9, 'दहा': 10, 'दस': 10,
            'अकरा': 11, 'ग्यारह': 11, 'बारा': 12, 'तेरा': 13, 'चौदा': 14, 'चौदह': 14,
            'पंधरा': 15, 'पंद्रह': 15, 'सोलह': 16, 'सोळा': 16, 'सतरा': 17, 'सत्रह': 17,
            'अठरा': 18, 'अठारह': 18, 'एकोणीस': 19, 'उन्नीस': 19, 'वीस': 20, 'बीस': 20,
        }
        cleaned = text.strip().rstrip('.')
        
        # If it's already digits (or includes Devanagari digits normalized), parse it
        processed = self._translate_number_words(cleaned)
        digits = ''.join(filter(str.isdigit, processed))
        if digits:
            return int(digits)
        
        # Simple word lookup
        for word, val in sorted(num_map.items(), key=lambda x: len(x[0]), reverse=True):
            if word in cleaned:
                return val
        
        return 0

    def _local_extract_dob(self, source_text: str) -> str:
        """Extract DOB from voice text. Handles '15 ऑगस्ट 2000' or 'पंद्रह August 2003'."""
        # Use translated text to get standard numbers/months
        processed = self._translate_number_words(source_text)
        logger.info(f"[DOB] Processed for parsing: '{processed}'")
        
        # Look for 3 groups of digits (DD MM YYYY or YYYY MM DD)
        digit_groups = re.findall(r'\d+', processed)
        
        if len(digit_groups) >= 3:
            # Assume DD MM YYYY or similar
            d = digit_groups[0]
            m = digit_groups[1]
            y = digit_groups[2]
            
            # Simple validation/adjustment
            if len(y) == 2: y = "20" + y # 00 -> 2000
            if int(m) > 12 and int(d) <= 12: # Swap if likely MM DD YYYY
                d, m = m, d
            
            return f"{int(d):02d}/{int(m):02d}/{y}"
        
        # Fallback to older word-based detection if specifically needed
        # but _translate_number_words now handles most words -> digits
        
        # If it's just one group of digits, it's likely an age
        if len(digit_groups) == 1:
            age = int(digit_groups[0])
            if 0 < age < 150:
                return str(age)
        # Two groups can be "20" + "1" -> age 21 (twenty one)
        if len(digit_groups) == 2:
            a, b = int(digit_groups[0]), int(digit_groups[1])
            if 10 <= a <= 90 and a % 10 == 0 and 0 <= b <= 9:
                age = a + b
                if 0 < age < 150:
                    return str(age)
        
        return ""

    def _normalize_age_to_number(self, value: str) -> str:
        """Ensure age is always returned as digits (e.g. 'Twenty one.' -> '21')."""
        if not value or not value.strip():
            return value
        processed = self._translate_number_words(value)
        digit_groups = re.findall(r"\d+", processed)
        if len(digit_groups) == 1:
            age = int(digit_groups[0])
            if 0 < age < 150:
                return str(age)
        if len(digit_groups) == 2:
            a, b = int(digit_groups[0]), int(digit_groups[1])
            if 10 <= a <= 90 and a % 10 == 0 and 0 <= b <= 9:
                age = a + b
                if 0 < age < 150:
                    return str(age)
        # Already a valid DD/MM/YYYY from _local_extract_dob
        if re.match(r"\d{1,2}/\d{1,2}/\d{2,4}", value.strip()):
            return value.strip()
        return value.strip()

    # =========================================================================
    # MAIN SINGLE-FIELD EXTRACTION
    # =========================================================================

    @mlflow.trace(name="extract_single_field")
    async def extract_single_field(self, source_text: str, field_name: str, field_description: str) -> str:
        """
        Extract a single field with intelligent routing:
        - Phone/Aadhaar: Local regex (instant)  
        - PAN Card: Local Regex -> Azure LLM Fallback
        - Age/DOB: Local date parser
        - Name: Direct transcript
        - Address: Translated transcript
        """
        value = ""

        # Step 0: Translate Hindi number words to digits
        processed_text = self._translate_number_words(source_text)
        logger.info(f"[{field_name}] Translated: '{source_text}' -> '{processed_text}'")

        # Step 1: Name - use raw transcript directly
        if field_name == "full_name":
            value = source_text.strip().rstrip('.')
            logger.info(f"[{field_name}] Direct transcript: {value}")
            return value

        # Step 2: Address - use translated transcript
        if field_name == "address":
            value = processed_text.strip().rstrip('.')
            logger.info(f"[{field_name}] Translated transcript: {value}")
            return value

        # Step 3: Phone/Aadhaar - local digit extraction
        if field_name in ["phone_number", "aadhaar_number"]:
            digits = "".join(filter(str.isdigit, processed_text))
            if digits:
                logger.info(f"[{field_name}] Local digits: {digits}")
                return digits

        # Step 4: PAN Card - local regex first, then LLM fallback
        elif field_name == "pan_card":
            # 4.1 Try Local Regex on processed text (fast & avoids Azure filters)
            value = self._local_extract_pan(processed_text)
            if value:
                logger.info(f"[{field_name}] Local PAN found: {value}")
                return value
            
            # 4.2 Fallback: local GGUF or Azure LLM
            raw_content = None
            if llm_provider_is_local():
                try:
                    from app.core.local_llm import local_chat_complete, local_chat_gguf_configured

                    if local_chat_gguf_configured():
                        logger.info(f"[{field_name}] No local match. Trying local GGUF...")
                        raw_content = local_chat_complete(
                            system='You extract PAN card numbers. Return ONLY JSON: {"value": "ABCDE1234F" or null}',
                            user=f"Extract the PAN (5 letters + 4 digits + 1 letter) from: '{source_text}'.",
                            max_tokens=150,
                        )
                except Exception as e:
                    logger.error(f"[{field_name}] Local GGUF error: {e}")
            elif self.azure_client:
                logger.info(f"[{field_name}] No local match. Trying AZURE LLM...")
                try:
                    response = self.azure_client.chat.completions.create(
                        model=self.deployment_name,
                        messages=[
                            {"role": "system", "content": "You extract PAN card numbers. Return ONLY a JSON object with a 'value' key. If not found, return {\"value\": null}."},
                            {"role": "user", "content": f"Extract the PAN card (exactly 5 letters + 4 digits + 1 letter) from: '{source_text}'. Transcription might be 'ABCDE 1 2 3 4 F'. Return JSON: {{\"value\": \"PAN\"}}"}
                        ],
                        max_completion_tokens=100
                    )
                    choice = response.choices[0]
                    raw_content = choice.message.content
                    finish_reason = getattr(choice, 'finish_reason', 'unknown')
                    if not raw_content:
                        logger.warning(f"[{field_name}] Azure returned empty content. Finish Reason: {finish_reason}")
                except Exception as e:
                    logger.error(f"[{field_name}] Azure error: {e}")

            if raw_content:
                logger.info(f"[{field_name}] LLM raw response: {raw_content.strip()}")
                parsed = self._parse_llm_json(raw_content)
                value = str(parsed.get("value", "")).strip().upper()
                if not value or value == "NULL":
                    match = re.search(r'[A-Z]{5}\d{4}[A-Z]', raw_content.upper())
                    if match:
                        value = match.group(0)

        # Step 5: Age/DOB - local date parser (always return numerical format)
        elif field_name == "age":
            value = self._local_extract_dob(source_text)
            if value:
                logger.info(f"[{field_name}] Local DOB/Age: {value}")
                return value

        # Final Fallback: never return empty
        if not value or value.lower() == "none" or value.lower() == "null":
            value = source_text.strip()
            # If the source text looks like letters and numbers, clean it
            if field_name == "pan_card":
                cleaned = re.sub(r'[^A-Z0-9]', '', value.upper())
                if len(cleaned) >= 5:
                    value = cleaned

        # Age field: always normalize to numerical form (e.g. "Twenty one." -> "21")
        if field_name == "age" and value:
            value = self._normalize_age_to_number(value)
            
        return value




extraction_service = ExtractionService()
