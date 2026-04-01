"""
Pydantic schemas for request/response validation.
Defines the exact 6-field entity structure for Indian document extraction.
"""

from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Optional, List, Dict, Any
import re


class FormEntities(BaseModel):
    """
    The 6 core entities extracted from citizen voice input.
    Each field includes regex-based validation for Indian document formats.
    """
    full_name: Optional[str] = Field(None, description="Full name of the citizen (पूरा नाम)")
    phone_number: Optional[str] = Field(None, description="10-digit Indian phone number (फोन नंबर)")
    pan_card: Optional[str] = Field(None, description="PAN Card number e.g. ABCDE1234F (पैन कार्ड)")
    age: Optional[str] = Field(None, description="Age or Date of Birth (आयु / जन्म तिथि)")
    aadhaar_number: Optional[str] = Field(None, description="12-digit Aadhaar number (आधार संख्या)")
    address: Optional[str] = Field(None, description="Residential address (पता)")

    @field_validator("phone_number")
    @classmethod
    def validate_phone(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            cleaned = re.sub(r"[\s\-]", "", v)
            if not re.match(r"^[6-9]\d{9}$", cleaned):
                return v  # Return as-is, flag as unvalidated
            return cleaned
        return v

    @field_validator("pan_card")
    @classmethod
    def validate_pan(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            cleaned = v.upper().strip()
            if not re.match(r"^[A-Z]{5}[0-9]{4}[A-Z]$", cleaned):
                return v  # Return as-is, flag as unvalidated
            return cleaned
        return v

    @field_validator("aadhaar_number")
    @classmethod
    def validate_aadhaar(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            cleaned = re.sub(r"[\s\-]", "", v)
            if not re.match(r"^\d{12}$", cleaned):
                return v  # Return as-is, flag as unvalidated
            return cleaned
        return v


class VoiceFormResponse(BaseModel):
    """Response schema returned by the /extract-voice-form endpoint."""
    success: bool
    transcript: Optional[str] = Field(None, description="Raw transcribed text from audio")
    translated_text: Optional[str] = Field(None, description="English translation of the transcript")
    entities: Optional[FormEntities] = None
    fields_filled: int = Field(0, description="Number of fields successfully extracted (out of 6)")
    error: Optional[str] = None



class HealthResponse(BaseModel):
    """Response schema for the health check endpoint."""
    status: str
    service: str
    deepgram_status: str
    azure_status: str


# --- Document OCR Specific Schemas ---

class ExtractedData(BaseModel):
    """Structured OCR output: flat keys for the UI plus nested field bags."""

    model_config = ConfigDict(extra="allow")

    document_type: Optional[str] = Field(
        None, description="Detected document type (invoice, aadhaar, form, receipt, etc.)"
    )

    confidence: Optional[str] = Field(
        None, description="Confidence level of extraction"
    )

    name: Optional[str] = Field(None, description="Full name from document")
    id_number: Optional[str] = Field(None, description="Primary ID (Aadhaar, PAN, etc.)")
    date_of_birth: Optional[str] = Field(None, description="Date of birth")
    address: Optional[str] = Field(None, description="Address or work location")
    phone_number: Optional[str] = Field(None, description="Phone / mobile")

    fields: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="All extracted fields from the document as key-value pairs"
    )

    other_fields: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional extracted information not covered in fields"
    )


class BarcodeData(BaseModel):
    data: str
    type: str
    engine: str
    strategy: str
    rect: Optional[Dict[str, int]] = None


class DocumentOCRResponse(BaseModel):
    success: bool
    message: str

    raw_text: Optional[str] = Field(
        None, description="Raw OCR extracted text from the document"
    )

    extracted_data: Optional[ExtractedData] = None

    barcodes: Optional[List[BarcodeData]] = Field(
        default_factory=list, description="List of detected barcodes and QR codes"
    )


class SummarizeResponse(BaseModel):
    """Response schema for the Document Summarization endpoint."""
    success: bool
    message: str
    summary: Optional[str] = Field(None, description="Concise executive summary of the document")
    key_points: Optional[List[str]] = Field(default_factory=list, description="Bullet list of key points")
    raw_text_preview: Optional[str] = Field(None, description="First 500 chars of extracted text (debug)")


class FaceVerificationResponse(BaseModel):
    success: bool
    status: Optional[str] = None
    verification_result: Optional[str] = None
    match_score: Optional[float] = None
    liveness_status: Optional[str] = None
    liveness_details: Optional[Dict[str, Any]] = None
    match_details: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
