import pytesseract
from PIL import Image, ImageDraw

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

image = Image.new("RGB", (700, 150), "white")
draw = ImageDraw.Draw(image)
draw.text((20, 40), "MEDINOVA AI - OCR TEST", fill="black")

text = pytesseract.image_to_string(image)

print("Extracted text:")
print(text)