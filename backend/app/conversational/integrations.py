"""Government integrations: UMANG and DigiLocker (optional)."""

from typing import Any, Dict, Optional
from fastapi import APIRouter, Depends, HTTPException
import httpx
from pydantic import BaseModel

from app.core.config import settings
from app.conversational.auth import get_current_user
from app.conversational.database import User

router = APIRouter()


class UMANGRequest(BaseModel):
    service_code: str
    parameters: Dict[str, Any]


class DigiLockerRequest(BaseModel):
    request_type: str
    parameters: Dict[str, Any]


class IntegrationResponse(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]] = None
    message: str


@router.post("/umang/service", response_model=IntegrationResponse)
async def call_umang_service(
    request: UMANGRequest,
    current_user: User = Depends(get_current_user),
):
    if not settings.UMANG_API_KEY:
        raise HTTPException(status_code=500, detail="UMANG API key not configured")
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                "https://api.umang.gov.in/api/service",
                headers={
                    "Authorization": f"Bearer {settings.UMANG_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={"service_code": request.service_code, "parameters": request.parameters},
            )
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        return IntegrationResponse(success=True, data=response.json(), message="UMANG service called successfully")
    except httpx.RequestError as e:
        return IntegrationResponse(success=False, message=f"UMANG service unavailable: {e}")


@router.post("/digilocker/request", response_model=IntegrationResponse)
async def digilocker_request(
    request: DigiLockerRequest,
    current_user: User = Depends(get_current_user),
):
    if not settings.DIGILOCKER_API_KEY or not settings.DIGILOCKER_CLIENT_ID:
        raise HTTPException(status_code=500, detail="DigiLocker credentials not configured")
    uid = str(current_user.id)
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            headers = {
                "Authorization": f"Bearer {settings.DIGILOCKER_API_KEY}",
                "Client-Id": settings.DIGILOCKER_CLIENT_ID,
                "Content-Type": "application/json",
            }
            if request.request_type == "get_issued_docs":
                r = await client.get(
                    f"https://api.digilocker.gov.in/api/user/{request.parameters.get('user_id', uid)}/issued",
                    headers=headers,
                )
            elif request.request_type == "get_e_docs":
                r = await client.get(
                    f"https://api.digilocker.gov.in/api/user/{request.parameters.get('user_id', uid)}/e-docs",
                    headers=headers,
                )
            elif request.request_type == "verify_document":
                r = await client.post(
                    "https://api.digilocker.gov.in/api/verify",
                    headers=headers,
                    json={
                        "doc_id": request.parameters["doc_id"],
                        "user_id": request.parameters.get("user_id", uid),
                    },
                )
            else:
                raise HTTPException(status_code=400, detail="Invalid request type")
        if r.status_code != 200:
            raise HTTPException(status_code=r.status_code, detail=r.text)
        return IntegrationResponse(success=True, data=r.json(), message="DigiLocker request processed")
    except httpx.RequestError as e:
        return IntegrationResponse(success=False, message=f"DigiLocker unavailable: {e}")


@router.get("/services/list")
async def list_services():
    return {
        "umang": {
            "name": "UMANG",
            "description": "Unified Mobile Application for New-age Governance",
            "services": [
                {"code": "aadhar_status", "name": "Aadhaar Status Check"},
                {"code": "pan_status", "name": "PAN Card Status"},
                {"code": "passport_status", "name": "Passport Status"},
                {"code": "driving_license", "name": "Driving License Services"},
                {"code": "complaint_filing", "name": "File Complaint"},
            ],
        },
        "digilocker": {
            "name": "DigiLocker",
            "description": "Digital document storage and verification",
            "services": [
                {"code": "issued_docs", "name": "Issued Documents"},
                {"code": "e_docs", "name": "E-Documents"},
                {"code": "doc_verification", "name": "Document Verification"},
            ],
        },
    }


@router.get("/status")
async def integration_status():
    return {
        "umang": {
            "configured": bool(settings.UMANG_API_KEY),
            "status": "available" if settings.UMANG_API_KEY else "not_configured",
        },
        "digilocker": {
            "configured": bool(settings.DIGILOCKER_API_KEY and settings.DIGILOCKER_CLIENT_ID),
            "status": "available" if (settings.DIGILOCKER_API_KEY and settings.DIGILOCKER_CLIENT_ID) else "not_configured",
        },
    }
