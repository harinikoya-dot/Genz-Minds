
from pathlib import Path
from app.services.ocr_service import ocr_service

# Create a real image containing test text.
from PIL import Image, ImageDraw

test_path = Path("ocr_sample.png")

image = Image.new("RGB", (900, 200), "white")
draw = ImageDraw.Draw(image)
draw.text(
    (30, 60),
    "MEDINOVA AI - Prescription OCR Test",
    fill="black",
    font_size=32,
)
image.save(test_path)

try:
    result = ocr_service.extract_text_from_file(
        str(test_path.resolve()),
        "image/png",
    )

    print("\nExtracted text:")
    print(result["text"])

    print("\nConfidence:", result["confidence"])
    print("Pages:", result["page_count"])
    print("Method:", result["extraction_method"])

    chunks = ocr_service.chunk_document(result["text"])
    print("\nChunks:", len(chunks))
    for chunk in chunks:
        print(chunk)

finally:
    if test_path.exists():
        test_path.unlink()
