import json
import logging
import ollama
from openai import OpenAI
from app.core.config import settings

logger = logging.getLogger(__name__)

# Initialize OpenAI/Azure Client
client = OpenAI(
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
    
    # --- STEP 1: Try Local Ollama (Primary) ---
    try:
        model_name = settings.OLLAMA_EXTRACT_MODEL
        logger.info(f"Attempting document extraction with LOCAL {model_name}...")
        
        # Check if Ollama service is reachable and has the model
        ollama.list() 
        
        response = ollama.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a professional document extraction assistant. Output valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            format="json",
            options={"temperature": 0},
            keep_alive="5m"
        )
        raw_output = response['message']['content'].strip()
        logger.info("Local Ollama document extraction successful.")
        
    except Exception as e:
        logger.warning(f"Local Ollama extraction failed: {e}. Falling back to Azure...")
        
        # --- STEP 2: Fallback to Azure OpenAI ---
        try:
            logger.info(f"Extracting with AZURE ({settings.OPENAI_MODEL})...")
            response = client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            raw_output = response.choices[0].message.content or "{}"
            logger.info("Azure document extraction successful.")
        except Exception as azure_err:
            logger.error(f"Final fallback to Azure failed: {azure_err}")
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
        
        # Normalize keys to lowercase for consistency
        fields = extracted_data.get("fields", {})
        other_fields = extracted_data.get("other_fields", {})

        # Build address from address, city, state, work_location, or other_fields when main address is empty
        address = (
            fields.get("address")
            or fields.get("work_location")
            or other_fields.get("address")
            or other_fields.get("work_location")
            or other_fields.get("office_location")
        )
        if not address and (fields.get("city") or other_fields.get("city")):
            parts = [fields.get("city") or other_fields.get("city")]
            if fields.get("state") or other_fields.get("state"):
                parts.append(fields.get("state") or other_fields.get("state"))
            address = ", ".join(p for p in parts if p)

        normalized_data = {
            "name": fields.get("name"),
            "id_number": (
                fields.get("id_number")
                or fields.get("aadhaar_number")
                or fields.get("pan_number")
                or fields.get("passport_number")
                or other_fields.get("pan_number")
                or other_fields.get("aadhaar_number")
            ),
            "date_of_birth": fields.get("date_of_birth"),
            "address": address,
            "phone_number": fields.get("phone_number"),
            "other_fields": other_fields
        }

        return normalized_data

    except Exception as e:
        return {}
