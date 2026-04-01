import json
import logging
import ollama
from openai import OpenAI
from app.core.config import settings
from app.core.llm_routing import llm_provider_is_local

logger = logging.getLogger(__name__)


def _cloud_openai_client() -> OpenAI:
    return OpenAI(
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL,
    )


def extract_structured_data_with_llm(raw_text: str) -> dict:
    """
    Uses OpenAI LLM to parse OCR text into structured format.
    """

    prompt = f"""
            You are an advanced AI system for document understanding and information extraction.

            INPUT:
            Below is OCR text extracted from a scanned document. The document may be any type such as:
            - Government ID (Aadhaar, PAN, Passport, Driving License)
            - Invoice or Receipt
            - Application Form
            - Bank Statement
            - Medical Document
            - Utility Bill
            - Contract
            - Certificate
            - Handwritten Form
            - Any unknown or unstructured document

            OCR TEXT:
            {raw_text[:4000]}

            TASK:

            1. Carefully analyze the OCR text.
            2. Identify the most likely document type.
            3. Extract all meaningful structured information from the document.
            4. Normalize field names into clear machine-readable keys.
            5. If the document type is unknown, still extract important key-value information from the text.

            IMPORTANT RULES:

            - If the document type is known (Aadhaar, Invoice, Passport, etc.), extract standard fields.
            - For CONTRACTS, OFFER LETTERS, or EMPLOYMENT documents: extract the person's full name (e.g. "To: Tejas Gadhe" or "I, Tejas Gadhe"); put work location, office city, or "Work-from-home" into the "address" field when no street address exists (e.g. "Bangalore office" -> address: "Bangalore"; "Work-from-home for Month 1, then Bangalore office" -> address: "Work-from-home; Bangalore"); put company/organization name into organization_name or company_name; put role/title, stipend, start date, and other terms in other_fields.
            - If the document type is UNKNOWN, extract any meaningful information such as names, numbers, dates, addresses, emails, IDs, or labeled fields.
            - Do NOT skip extraction even if the document structure is unclear.
            - Always populate "address" when any location, city, office, or work place is mentioned (e.g. "Bangalore office", "Mumbai", "Remote").

            FIELDS TO EXTRACT IF PRESENT:

            Identity fields:
            - name
            - id_number
            - aadhaar_number
            - pan_number
            - passport_number
            - license_number

            Personal information:
            - date_of_birth
            - gender
            - nationality
            - address
            - phone_number
            - email

            Document metadata:
            - document_type
            - issue_date
            - expiry_date
            - issuing_authority

            Financial information:
            - invoice_number
            - bill_number
            - transaction_id
            - total_amount
            - tax_amount
            - currency

            Location information:
            - city
            - state
            - country
            - postal_code

            Other information:
            - organization_name
            - company_name
            - reference_number
            - additional_notes

            IF DOCUMENT TYPE IS UNKNOWN:

            Extract any useful fields found in the text and include them inside "other_fields" as key-value pairs.

            OUTPUT FORMAT (IMPORTANT):

            Return ONLY valid JSON in this structure:

            {{
            "document_type": "detected_document_type_or_unknown_document",
            "confidence": "high | medium | low",
            "fields": {{
                "name": "...",
                "id_number": "...",
                "date_of_birth": "...",
                "address": "... (or work location, office city, e.g. Bangalore, Work-from-home)",
                "phone_number": "...",
                "email": "...",
                "city": "...",
                "work_location": "...",
                "issue_date": "...",
                "expiry_date": "...",
                "invoice_number": "...",
                "total_amount": "...",
                "organization_name": "...",
                "reference_number": "...",
                "additional_notes": "..."
            }},
            "other_fields": {{
                "field_name": "value"
            }}
            }}

            RULES:

            - Return ONLY JSON.
            - Do NOT add explanations.
            - If a field does not exist, return null.
            - Do NOT invent information.
            - Preserve numbers exactly as they appear.
            - Ensure JSON format is valid.
            """

    raw_output = None

    # --- STEP 1: Local Ollama (primary) ---
    try:
        model_name = settings.OLLAMA_EXTRACT_MODEL
        logger.info(f"Attempting document extraction with Ollama ({model_name})...")
        ollama.list()
        response = ollama.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a professional document extraction assistant. Output valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            format="json",
            options={"temperature": 0},
            keep_alive="5m",
        )
        raw_output = response["message"]["content"].strip()
        logger.info("Ollama document extraction successful.")
    except Exception as e:
        logger.warning(f"Ollama document extraction failed: {e}")

    # --- STEP 2: Local GGUF (Qwen via llama-cpp-python) ---
    if raw_output is None and llm_provider_is_local():
        try:
            from app.core.local_llm import local_chat_complete, local_chat_gguf_configured

            if local_chat_gguf_configured():
                logger.info("Attempting document extraction with local GGUF...")
                raw_output = local_chat_complete(
                    system="You are a professional document extraction assistant. Output valid JSON only, no markdown.",
                    user=prompt,
                    max_tokens=8192,
                ).strip()
                logger.info("Local GGUF document extraction successful.")
        except Exception as le:
            logger.warning(f"Local GGUF document extraction failed: {le}")

    # --- STEP 3: OpenAI (cloud mode only, when key is set) ---
    if raw_output is None and not llm_provider_is_local() and settings.OPENAI_API_KEY:
        try:
            logger.info(f"Extracting with OpenAI ({settings.OPENAI_MODEL})...")
            response = _cloud_openai_client().chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            raw_output = response.choices[0].message.content or "{}"
            logger.info("OpenAI document extraction successful.")
        except Exception as cloud_err:
            logger.error(f"OpenAI document extraction failed: {cloud_err}")

    if raw_output is None:
        logger.error("All document extraction backends failed.")
        return {}

    if not raw_output:
        return {}
        
    try:
        # Super-robust JSON extraction
        json_str = raw_output.strip()
        extracted_data = {}
        try:
            # Try parsing directly first
            extracted_data = json.loads(json_str)
        except json.JSONDecodeError:
            # Use regex to find the first '{' and the last '}'
            import re
            match = re.search(r'(\{.*\})', json_str, re.DOTALL)
            if match:
                try:
                    extracted_data = json.loads(match.group(1))
                except Exception as je:
                    extracted_data = {}
            else:
                extracted_data = {}
        
        fields = extracted_data.get("fields") or {}
        other_fields = extracted_data.get("other_fields") or {}
        if not isinstance(fields, dict):
            fields = {}
        if not isinstance(other_fields, dict):
            other_fields = {}

        def _of(*keys: str):
            for k in keys:
                v = fields.get(k)
                if v is not None and str(v).strip():
                    return v
                v = other_fields.get(k)
                if v is not None and str(v).strip():
                    return v
            return None

        # Build address from address, city, state, work_location, or other_fields when main address is empty
        address = (
            _of("address", "work_location", "office_location")
            or other_fields.get("address")
            or other_fields.get("work_location")
        )
        if not address and (fields.get("city") or other_fields.get("city")):
            parts = [fields.get("city") or other_fields.get("city")]
            if fields.get("state") or other_fields.get("state"):
                parts.append(fields.get("state") or other_fields.get("state"))
            address = ", ".join(p for p in parts if p)

        name = _of("name", "full_name", "holder_name", "employee_name", "applicant_name")
        id_number = _of(
            "id_number",
            "aadhaar_number",
            "uid",
            "pan_number",
            "passport_number",
            "license_number",
        )
        date_of_birth = _of("date_of_birth", "dob", "birth_date")
        phone_number = _of("phone_number", "mobile", "mobile_number", "contact_number")

        normalized_data = {
            "document_type": extracted_data.get("document_type"),
            "confidence": extracted_data.get("confidence"),
            "name": name,
            "id_number": id_number,
            "date_of_birth": date_of_birth,
            "address": address,
            "phone_number": phone_number,
            "fields": fields,
            "other_fields": other_fields,
        }

        logger.info(
            "Document structured extraction: doc_type=%r name=%r id=%r dob=%r phone=%r address_len=%d other_keys=%s",
            normalized_data.get("document_type"),
            normalized_data.get("name"),
            (normalized_data.get("id_number") or "")[:32] if normalized_data.get("id_number") else None,
            normalized_data.get("date_of_birth"),
            (normalized_data.get("phone_number") or "")[:24] if normalized_data.get("phone_number") else None,
            len(normalized_data.get("address") or "") if normalized_data.get("address") else 0,
            list(other_fields.keys())[:20],
        )

        return normalized_data

    except Exception as e:
        return {}
