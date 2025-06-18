from pdf2image import convert_from_path, convert_from_bytes
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)
import pytesseract


def get_text_from_pdf(file_path: str) -> str:
    """
    Extract text from a PDF file using OCR.
    
    Args:
        file_path (str): Path to the PDF file.
        
    Returns:
        str: Extracted text from the PDF.
    """
    try:
        images = convert_from_path(file_path, dpi=300)
        text: str = ''
        for i, image in enumerate(images):
            text += pytesseract.image_to_string(image)
        return text
    except (PDFInfoNotInstalledError, PDFPageCountError, PDFSyntaxError) as e:
        raise RuntimeError(f"Error processing PDF: {e}") from e