# utils/cv_text_extractor.py
import io
import re
import os
import httpx
from typing import Tuple, Dict, Any, Optional

# Optional deps:
# pip install pypdf beautifulsoup4 lxml
from pypdf import PdfReader
from bs4 import BeautifulSoup

DEFAULT_CV_TIMEOUT = float(os.getenv("CV_FETCH_TIMEOUT_SECONDS", "25"))
DEFAULT_MAX_BYTES = int(os.getenv("CV_FETCH_MAX_BYTES", str(25 * 1024 * 1024)))  # 25MB

def _is_pdf(url: str, content_type: Optional[str]) -> bool:
    if content_type and "pdf" in content_type.lower():
        return True
    return url.lower().endswith(".pdf")

def _clean_text(text: str) -> str:
    # collapse whitespace, normalize bullets, trim long runs
    text = text.replace("\x00", " ").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"•", "-", text)
    return text.strip()

def _extract_pdf_text(pdf_bytes: bytes) -> Tuple[str, Dict[str, Any]]:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = []
    for i, p in enumerate(reader.pages):
        t = p.extract_text() or ""
        # keep page breaks to help later chunking
        pages.append(t.strip())
    text = "\n\n=== PAGE BREAK ===\n\n".join(pages)
    return _clean_text(text), {"mime": "application/pdf", "pages": len(reader.pages), "bytes": len(pdf_bytes)}

def _extract_html_text(html: str) -> Tuple[str, Dict[str, Any]]:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    return _clean_text(text), {"mime": "text/html", "bytes": len(html.encode("utf-8", "ignore"))}

async def fetch_and_extract_cv_text(cv_url: str) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Fetches the CV from a public URL and returns (text, meta).
    Safeguards: timeout, max size, and tolerant content-type handling.
    """
    if not cv_url:
        return None, {"error": "empty_url"}

    headers = {"User-Agent": "RecruiterBot/1.0"}
    async with httpx.AsyncClient(follow_redirects=True, headers=headers) as client:
        # HEAD (best effort) to check size/content-type
        try:
            head = await client.head(cv_url, timeout=DEFAULT_CV_TIMEOUT)
            ctype = head.headers.get("Content-Type", "")
            clen = head.headers.get("Content-Length")
            if clen and int(clen) > DEFAULT_MAX_BYTES:
                return None, {"error": "file_too_large", "content_length": int(clen)}
        except Exception:
            ctype = None  # proceed with GET anyway

        # GET (stream + size guard)
        try:
            r = await client.get(cv_url, timeout=DEFAULT_CV_TIMEOUT)
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            return None, {"error": "http_status", "status": e.response.status_code}
        except Exception as e:
            return None, {"error": "fetch_failure", "detail": str(e)}

        content_type = r.headers.get("Content-Type", ctype) or ""
        content = r.content
        if len(content) > DEFAULT_MAX_BYTES:
            return None, {"error": "file_too_large", "bytes": len(content)}

        try:
            if _is_pdf(cv_url, content_type):
                text, meta = _extract_pdf_text(content)
                return (text or None), meta
            else:
                # treat as html/text fallback
                try:
                    html = r.text  # will decode per headers
                except Exception:
                    html = content.decode("utf-8", errors="replace")
                text, meta = _extract_html_text(html)
                return (text or None), meta
        except Exception as e:
            # last-chance: try to decode as utf-8 text and return raw
            try:
                raw = content.decode("utf-8", errors="replace")
                return _clean_text(raw), {"mime": content_type or "application/octet-stream", "bytes": len(content), "warning": "raw_decode_fallback", "detail": str(e)}
            except Exception:
                return None, {"error": "extract_failure", "detail": str(e), "mime": content_type}
