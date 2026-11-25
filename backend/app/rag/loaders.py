import io
from pathlib import PurePath

from pypdf import PdfReader
from pypdf.errors import PyPdfError

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


class UnsupportedFileTypeError(ValueError):
    pass


class DocumentParseError(ValueError):
    """The file could not be read, or contained no extractable text."""


def extract_text(filename: str, data: bytes) -> str:
    ext = PurePath(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"unsupported file type {ext or '(none)'}; use one of {sorted(SUPPORTED_EXTENSIONS)}"
        )
    if ext == ".pdf":
        try:
            reader = PdfReader(io.BytesIO(data))
            text = "\n\n".join((page.extract_text() or "") for page in reader.pages)
        except (PyPdfError, ValueError, KeyError) as exc:
            raise DocumentParseError(f"could not read PDF: {exc}") from exc
    else:
        text = data.decode("utf-8", errors="replace")
    text = text.replace("\x00", "")  # Postgres text columns reject NUL bytes
    if not text.strip():
        raise DocumentParseError("no extractable text found in document")
    return text
