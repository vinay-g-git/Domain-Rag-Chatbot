"""
document_loader.py
-------------------
Module 1 (Document Upload) + Module 2 (Text Extraction).

Reads one or more PDFs with pypdf, keeps document name + page number as
metadata for every page, skips empty pages safely, and validates file
type/size before processing.
"""

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

MAX_FILE_SIZE_MB = 50
ALLOWED_EXTENSIONS = {".pdf"}


@dataclass
class PageRecord:
    """One non-empty page of text plus its source metadata."""
    text: str
    source_document: str
    page_number: int  # 1-indexed, human-friendly


class DocumentValidationError(ValueError):
    """Raised when an uploaded file fails type/size validation."""


def validate_file(filename: str, size_bytes: int) -> None:
    """Validate file extension and size before we try to parse it."""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise DocumentValidationError(
            f"'{filename}' is not a supported file type. Only PDF files are allowed."
        )
    size_mb = size_bytes / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise DocumentValidationError(
            f"'{filename}' is {size_mb:.1f} MB, which exceeds the "
            f"{MAX_FILE_SIZE_MB} MB limit."
        )


def extract_pages_from_pdf(file_path: str, source_name: str | None = None) -> list[PageRecord]:
    """
    Extract text from every page of a single PDF.

    Empty / whitespace-only pages are skipped safely (no crash on scanned
    pages with no extractable text -- those are left for the optional OCR
    extension mentioned in the project spec).
    """
    source_name = source_name or Path(file_path).name
    reader = PdfReader(file_path)

    records: list[PageRecord] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            # Corrupt or unreadable page -- skip rather than crash the run.
            continue

        text = text.strip()
        if not text:
            continue  # likely a scanned/image-only page; needs OCR extension

        records.append(PageRecord(text=text, source_document=source_name, page_number=i))

    return records


def extract_pages_from_multiple_pdfs(file_paths: list[str]) -> list[PageRecord]:
    """Extract pages from several PDFs, tagging each with its own filename."""
    all_records: list[PageRecord] = []
    for path in file_paths:
        all_records.extend(extract_pages_from_pdf(path))
    return all_records
