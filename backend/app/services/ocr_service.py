
import os
import re
from typing import Dict, Any, List, Optional

import pytesseract
from PIL import Image, ImageOps, ImageFilter
from pypdf import PdfReader
from pytesseract import Output


# Windows Tesseract installation
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

if os.path.isfile(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


class OCRService:
    """Extract actual text from images and PDF medical documents."""

    @staticmethod
    def _prepare_image(image: Image.Image) -> Image.Image:
        """Prepare an image to improve OCR readability."""
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        image = ImageOps.grayscale(image)

        # Enlarge small images for better recognition.
        if image.width < 1200:
            scale = 1200 / image.width
            image = image.resize(
                (1200, int(image.height * scale))
            )

        image = ImageOps.autocontrast(image)
        return image.filter(ImageFilter.SHARPEN)

    @staticmethod
    def _ocr_image(image: Image.Image) -> Dict[str, Any]:
        """Extract image text and calculate confidence from OCR output."""
        image = OCRService._prepare_image(image)

        data = pytesseract.image_to_data(
            image,
            output_type=Output.DICT
        )

        words = []
        confidences = []

        for word, raw_conf in zip(data["text"], data["conf"]):
            word = word.strip()

            try:
                conf = float(raw_conf)
            except (TypeError, ValueError):
                conf = -1

            if word and conf >= 0:
                words.append(word)
                confidences.append(conf)

        # Keep word spacing and line breaks from Tesseract.
        text = pytesseract.image_to_string(
            image,
            config="--psm 3"
        ).strip()

        confidence = (
            sum(confidences) / len(confidences) / 100
            if confidences
            else None
        )

        return {
            "text": text,
            "confidence": confidence,
        }

    @staticmethod
    def extract_text_from_file(
        file_path: str,
        mime_type: str
    ) -> Dict[str, Any]:
        """
        Extract real text from TXT, PDF, PNG, JPG and JPEG files.

        Scanned PDFs are processed page by page using Tesseract.
        No sample medical results are generated.
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(
                f"Uploaded file not found: {file_path}"
            )

        file_size = os.path.getsize(file_path)

        if file_size == 0:
            raise ValueError("The uploaded file is empty.")

        extension = os.path.splitext(file_path)[1].lower()
        mime = (mime_type or "").lower().split(";")[0].strip()

        pages: List[Dict[str, Any]] = []
        ocr_confidences = []

        # Plain text files
        if extension == ".txt" or mime == "text/plain":
            with open(
                file_path, "r", encoding="utf-8-sig", errors="replace"
            ) as file:
                text = file.read().strip()

            if not text:
                raise ValueError("No readable text was found.")

            pages.append({
                "page_number": 1,
                "text": text,
                "confidence": None,
            })

            method = "text"

        # Image files
        elif extension in {".png", ".jpg", ".jpeg"} or mime.startswith("image/"):
            try:
                with Image.open(file_path) as image:
                    result = OCRService._ocr_image(image.copy())
            except pytesseract.TesseractNotFoundError:
                raise RuntimeError(
                    "Tesseract was not found. Check TESSERACT_PATH."
                )
            except Exception as exc:
                raise ValueError(
                    f"Could not process the uploaded image: {exc}"
                ) from exc

            if result["confidence"] is not None:
                ocr_confidences.append(result["confidence"])

            pages.append({
                "page_number": 1,
                "text": result["text"],
                "confidence": result["confidence"],
            })

            method = "tesseract_ocr"

        # PDF files: extract embedded text first; OCR scanned pages.
        elif extension == ".pdf" or mime == "application/pdf":
            try:
                reader = PdfReader(file_path)

                if reader.is_encrypted:
                    raise ValueError(
                        "This PDF is password-protected. "
                        "Please upload an unlocked PDF."
                    )

                if not reader.pages:
                    raise ValueError("The PDF contains no pages.")

                for index, pdf_page in enumerate(reader.pages):
                    page_text = (pdf_page.extract_text() or "").strip()
                    confidence = None

                    # Scanned page or image-only page.
                    if len(page_text) < 10:
                        try:
                            from pdf2image import convert_from_path

                            images = convert_from_path(
                                file_path,
                                first_page=index + 1,
                                last_page=index + 1,
                                dpi=250,
                            )

                            if not images:
                                raise ValueError(
                                    f"Could not render PDF page {index + 1}."
                                )

                            result = OCRService._ocr_image(images[0])
                            page_text = result["text"]
                            confidence = result["confidence"]

                            if confidence is not None:
                                ocr_confidences.append(confidence)

                        except ImportError as exc:
                            raise RuntimeError(
                                "Install pdf2image to process scanned PDFs."
                            ) from exc
                        except Exception as exc:
                            raise RuntimeError(
                                f"Could not OCR PDF page {index + 1}. "
                                "For scanned PDFs, install Poppler for "
                                f"Windows and add its bin folder to PATH. Details: {exc}"
                            ) from exc

                    pages.append({
                        "page_number": index + 1,
                        "text": page_text,
                        "confidence": confidence,
                    })

            except (ValueError, RuntimeError):
                raise
            except Exception as exc:
                raise ValueError(
                    f"Could not read the PDF: {exc}"
                ) from exc

            method = "pdf_text_and_ocr"

        else:
            raise ValueError(
                f"Unsupported file type: {extension or mime}. "
                "Supported types: TXT, PDF, PNG, JPG, JPEG."
            )

        extracted_text = "\n\n".join(
            f"[Page {page['page_number']}]\n{page['text']}"
            for page in pages
            if page["text"].strip()
        ).strip()

        if not extracted_text:
            raise ValueError(
                "No readable text was extracted. Try a clearer image "
                "or a higher-quality scan."
            )

        overall_confidence: Optional[float] = (
            sum(ocr_confidences) / len(ocr_confidences)
            if ocr_confidences
            else None
        )

        return {
            "text": extracted_text,
            "pages": pages,
            "page_count": len(pages),
            "confidence": overall_confidence,
            "file_size": file_size,
            "extraction_method": method,
        }

    @staticmethod
    def chunk_document(
        text: str,
        chunk_size: int = 400,
        overlap: int = 80
    ) -> List[Dict[str, Any]]:
        """Split document text into overlapping chunks for RAG."""
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")

        if overlap < 0 or overlap >= chunk_size:
            raise ValueError(
                "overlap must be non-negative and smaller than chunk_size."
            )

        text = text.strip()
        if not text:
            return []

        # Prefer splitting at paragraph or sentence boundaries.
        paragraphs = re.split(r"\n+|(?<=[.!?])\s+", text)
        chunks = []
        current = ""

        def add_chunk(value: str) -> None:
            chunks.append({
                "chunk_index": len(chunks),
                "text": value,
                "page_number": 1,
            })

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            # Handle an individual paragraph longer than chunk_size.
            if len(paragraph) > chunk_size:
                if current:
                    add_chunk(current)
                    current = ""

                start = 0
                while start < len(paragraph):
                    end = min(start + chunk_size, len(paragraph))
                    piece = paragraph[start:end].strip()

                    if piece:
                        add_chunk(piece)

                    if end == len(paragraph):
                        break

                    start = end - overlap

                continue

            candidate = (
                f"{current} {paragraph}".strip()
                if current else paragraph
            )

            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    add_chunk(current)

                # Carry over a small amount of previous context.
                prefix = current[-overlap:] if overlap and current else ""
                current = f"{prefix} {paragraph}".strip()

                if len(current) > chunk_size:
                    current = paragraph

        if current:
            add_chunk(current)

        return chunks


ocr_service = OCRService()
