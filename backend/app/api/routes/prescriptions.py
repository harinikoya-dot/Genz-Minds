
import os
import tempfile

from typing import List

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    UploadFile,
    File,
    status,
)
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.all_models import (
    User,
    Prescription,
    PrescriptionMedicine,
)
from app.schemas.schemas import (
    PrescriptionResponse,
    PrescriptionConfirmRequest,
)
from app.api.dependencies import get_current_active_user
from app.core.permissions import verify_resource_ownership
from app.services.prescription_service import prescription_service
from app.services.ocr_service import ocr_service


router = APIRouter(
    prefix="/prescriptions",
    tags=["Prescriptions"],
)


# --------------------------------------------------
# 1. LIST THE CURRENT USER'S PRESCRIPTIONS
# --------------------------------------------------
@router.get("/", response_model=List[PrescriptionResponse])
def list_prescriptions(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    return (
        db.query(Prescription)
        .filter(Prescription.user_id == current_user.id)
        .order_by(Prescription.created_at.desc())
        .all()
    )


# --------------------------------------------------
# 2. GET ONE PRESCRIPTION
# --------------------------------------------------
@router.get(
    "/{prescription_id}",
    response_model=PrescriptionResponse,
)
def get_prescription(
    prescription_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    prescription = (
        db.query(Prescription)
        .filter(Prescription.id == prescription_id)
        .first()
    )

    if not prescription:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prescription not found.",
        )

    verify_resource_ownership(
        prescription.user_id,
        current_user.id,
        "Prescription",
    )

    return prescription


# --------------------------------------------------
# 3. SCAN A PRESCRIPTION IMAGE OR PDF
# --------------------------------------------------
@router.post("/scan")
def scan_prescription(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user),
):
    """
    Extract text from a prescription and return possible medicine
    entries. This endpoint does not save medicines or schedules.
    """

    allowed_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".pdf",
        ".txt",
    }

    allowed_content_types = {
        "image/jpeg",
        "image/png",
        "application/pdf",
        "text/plain",
    }

    filename = file.filename or ""
    extension = os.path.splitext(filename)[1].lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Upload a JPG, JPEG, PNG, PDF, or TXT file.",
        )

    if file.content_type not in allowed_content_types:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file type is not supported.",
        )

    temp_path = None

    try:
        # Read the upload and enforce a 10 MB limit.
        content = file.file.read()

        if not content:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty.",
            )

        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="File size must not exceed 10 MB.",
            )

        # Save temporarily for the OCR service.
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension,
        ) as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name

        # Extract actual text from the uploaded file.
        ocr_result = ocr_service.extract_text_from_file(
            temp_path,
            file.content_type,
        )

        extracted_text = ocr_result.get("text", "").strip()

        if not extracted_text:
            raise HTTPException(
                status_code=422,
                detail=(
                    "No readable text was extracted. "
                    "Try a clearer prescription image."
                ),
            )

        # Parse only entries recognized by the existing parser.
        medicines = prescription_service.extract_prescription_items(
            extracted_text
        )

        warnings = []

        if not medicines:
            warnings.append(
                "No medicine entries were recognized. "
                "Please inspect the extracted text and enter details "
                "with help from your pharmacist or prescribing doctor."
            )

        confidence = ocr_result.get("confidence")

        if confidence is not None and confidence < 0.70:
            warnings.append(
                "OCR confidence is low. Check the extracted text carefully."
            )

        for medicine in medicines:
            if medicine.get("is_unclear", True):
                warnings.append(
                    "Please verify the dose and instructions for "
                    f"{medicine.get('medicine_name', 'this medicine')}."
                )

        return {
            "status": "success",
            "filename": filename,
            "extracted_text": extracted_text,
            "page_count": ocr_result.get("page_count", 1),
            "ocr_confidence": confidence,
            "extraction_method": ocr_result.get(
                "extraction_method"
            ),
            "medicines": medicines,
            "simple_explanation": (
                "The prescription text has been extracted. "
                "Review the medicine details and warnings below. "
                "Any missing or unclear medical instructions must "
                "be verified before use."
            ),
            "warnings": warnings,
            "requires_review": True,
        }

    except HTTPException:
        raise

    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        # Full traceback remains in the backend terminal.
        import logging

        logging.exception("Prescription scan failed")

        raise HTTPException(
            status_code=500,
            detail="Prescription scanning failed. Check backend logs.",
        ) from exc

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

        file.file.close()


# --------------------------------------------------
# 4. CONFIRM REVIEWED PRESCRIPTION MEDICINES
# --------------------------------------------------
@router.post("/confirm")
def confirm_prescription(
    data: PrescriptionConfirmRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    prescription = (
        db.query(Prescription)
        .filter(Prescription.id == data.prescription_id)
        .first()
    )

    if not prescription:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prescription not found.",
        )

    verify_resource_ownership(
        prescription.user_id,
        current_user.id,
        "Prescription",
    )

    items_dicts = [
        item.model_dump()
        for item in data.medicines
    ]

    try:
        created_meds = (
            prescription_service.confirm_prescription_medicines(
                db=db,
                prescription_id=data.prescription_id,
                user_id=current_user.id,
                confirmed_items=items_dicts,
            )
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    return {
        "status": "success",
        "message": (
            f"Confirmed {len(created_meds)} medicines. "
            "Schedules were created for the supplied reminder times."
        ),
        "medicines_created": len(created_meds),
    }
