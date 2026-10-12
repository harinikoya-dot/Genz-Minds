
import re
from typing import List, Dict, Any

from sqlalchemy.orm import Session

from app.models.all_models import (
    Prescription,
    PrescriptionMedicine,
    Medicine,
    MedicationSchedule,
)


class PrescriptionService:

    @staticmethod
    def extract_prescription_items(
        raw_text: str,
    ) -> List[Dict[str, Any]]:
        """
        Extract possible medicine entries from OCR text.

        This is heuristic parsing only. It does not verify medical
        correctness. Uncertain fields must be checked before confirmation.
        """
        if not raw_text or not raw_text.strip():
            return []

        items: List[Dict[str, Any]] = []
        seen_names = set()

        lines = raw_text.splitlines()

        # Recognizes patterns such as:
        # 1. ExampleMed 500mg - twice daily after food
        # ExampleMed 250 mg - once daily
        pattern = re.compile(
            r"^\s*(?:\d+[\.\)]\s*)?"
            r"([A-Za-z][A-Za-z0-9]*(?:[\s\-][A-Za-z][A-Za-z0-9]*){0,2})"
            r"\s+(\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|IU|units?)?)"
            r"\s*[-:]\s*(.+?)\s*$",
            re.IGNORECASE,
        )

        for line in lines:
            line_clean = line.strip()

            if not line_clean:
                continue

            if line_clean.lower().startswith(
                (
                    "doctor:",
                    "date:",
                    "patient:",
                    "special instructions:",
                    "page ",
                    "medinova",
                )
            ):
                continue

            match = pattern.match(line_clean)

            if not match:
                continue

            medicine_name = match.group(1).strip()
            strength = match.group(2).strip()
            instructions = match.group(3).strip()
            normalized_name = medicine_name.casefold()

            if normalized_name in seen_names:
                continue

            seen_names.add(normalized_name)

            # Only infer frequency when a recognizable phrase is present.
            lower_instructions = instructions.lower()
            frequency = "Not specified"

            if (
                "three times daily" in lower_instructions
                or "thrice daily" in lower_instructions
                or re.search(r"\btid\b", lower_instructions)
            ):
                frequency = "Three times daily"
            elif (
                "twice daily" in lower_instructions
                or "two times daily" in lower_instructions
                or re.search(r"\bbid\b", lower_instructions)
            ):
                frequency = "Twice daily"
            elif (
                "once daily" in lower_instructions
                or "once a day" in lower_instructions
                or re.search(r"\bqd\b", lower_instructions)
            ):
                frequency = "Once daily"

            food_instructions = "unspecified"

            if "empty stomach" in lower_instructions or "before food" in lower_instructions:
                food_instructions = "before_food"
            elif "after food" in lower_instructions:
                food_instructions = "after_food"
            elif "with food" in lower_instructions:
                food_instructions = "with_food"

            duration_match = re.search(
                r"\b(\d+\s*(?:days?|weeks?|months?))\b",
                lower_instructions,
            )
            duration = (
                duration_match.group(1)
                if duration_match
                else "Not specified"
            )

            # Do not invent a dose, route, dosage form, or reminder time.
            items.append(
                {
                    "medicine_name": medicine_name,
                    "generic_name": None,
                    "strength": strength,
                    "dosage_form": "Not specified",
                    "dose": "Not specified",
                    "route": "Not specified",
                    "frequency": frequency,
                    "duration": duration,
                    "food_instructions": food_instructions,
                    "scheduled_timing": [],
                    "confidence": 0.50,
                    "is_unclear": True,
                    "clinical_notes": (
                        "OCR-extracted instructions require verification: "
                        + instructions
                    ),
                }
            )

        return items

    @staticmethod
    def confirm_prescription_medicines(
        db: Session,
        prescription_id: str,
        user_id: str,
        confirmed_items: List[Dict[str, Any]],
    ) -> List[Medicine]:
        """
        Save user-reviewed prescription medicines and their reminder
        schedules. The prescription must belong to the current user.
        """
        prescription = (
            db.query(Prescription)
            .filter(
                Prescription.id == prescription_id,
                Prescription.user_id == user_id,
            )
            .first()
        )

        if not prescription:
            raise ValueError(
                "Prescription not found for this user."
            )

        if not confirmed_items:
            raise ValueError(
                "No confirmed medicines were provided."
            )

        created_medicines: List[Medicine] = []

        try:
            for item in confirmed_items:
                medicine_name = str(
                    item.get("medicine_name", "")
                ).strip()

                if not medicine_name:
                    raise ValueError(
                        "Every medicine must have a name."
                    )

                # These fields must come from user/clinician review.
                dose = str(
                    item.get("dose", "")
                ).strip()
                route = str(
                    item.get("route", "")
                ).strip()
                frequency = str(
                    item.get("frequency", "")
                ).strip()
                duration = str(
                    item.get("duration", "")
                ).strip()

                if not all((dose, route, frequency, duration)):
                    raise ValueError(
                        f"Review dose, route, frequency and duration "
                        f"for {medicine_name} before confirming."
                    )

                timings = item.get("scheduled_timing") or []

                if not isinstance(timings, list):
                    raise ValueError(
                        f"Reminder times for {medicine_name} "
                        "must be a list."
                    )

                for slot in timings:
                    if not re.fullmatch(
                        r"(?:[01]\d|2[0-3]):[0-5]\d",
                        str(slot),
                    ):
                        raise ValueError(
                            f"Invalid reminder time for {medicine_name}: "
                            f"{slot}. Use HH:MM format."
                        )

                # Save the reviewed medicine against the prescription.
                prescription_medicine = PrescriptionMedicine(
                    prescription_id=prescription.id,
                    user_id=user_id,
                    medicine_name=medicine_name,
                    generic_name=item.get("generic_name"),
                    strength=item.get("strength"),
                    dosage_form=item.get(
                        "dosage_form", "Not specified"
                    ),
                    dose=dose,
                    route=route,
                    frequency=frequency,
                    duration=duration,
                    quantity=item.get("quantity"),
                    food_instructions=item.get(
                        "food_instructions", "unspecified"
                    ),
                    scheduled_timing=timings,
                    confidence=item.get("confidence", 0.50),
                    is_unclear=bool(item.get("is_unclear", True)),
                    is_confirmed=True,
                    clinical_notes=item.get("clinical_notes"),
                )

                db.add(prescription_medicine)
                db.flush()

                medicine = Medicine(
                    user_id=user_id,
                    prescription_medicine_id=prescription_medicine.id,
                    name=medicine_name,
                    generic_name=item.get("generic_name"),
                    strength=item.get("strength"),
                    dosage_form=item.get(
                        "dosage_form", "Not specified"
                    ),
                    prescribed_dose=dose,
                    route=route,
                    frequency=frequency,
                    time_slots=timings,
                    food_instructions=item.get(
                        "food_instructions", "unspecified"
                    ),
                    purpose=item.get("clinical_notes"),
                    prescriber=prescription.doctor_name,
                    status="active",
                    source="pres_scan",
                )

                db.add(medicine)
                db.flush()

                # Create schedules only for explicitly reviewed times.
                for slot in timings:
                    schedule = MedicationSchedule(
                        user_id=user_id,
                        medicine_id=medicine.id,
                        dose_time=slot,
                        dosage=dose,
                        food_instruction=item.get(
                            "food_instructions", "unspecified"
                        ),
                        is_active=True,
                    )
                    db.add(schedule)

                created_medicines.append(medicine)

            prescription.is_verified = True
            db.commit()

            return created_medicines

        except Exception:
            db.rollback()
            raise


prescription_service = PrescriptionService()
