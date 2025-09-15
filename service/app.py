from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Iterable, Tuple
import os, json, textwrap, re, uuid, base64, logging, io, requests
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
from google.auth.exceptions import RefreshError # For user impersonation
from pydantic import BaseModel, Field
from starlette.datastructures import UploadFile as StarletteUploadFile  # type hinting only
from difflib import SequenceMatcher


# =========================
# Fuzzy & Parse Helpers. Move canddiates from stage to stage
# =========================

_NAME_SCORE_THRESHOLD = int(os.environ.get("NAME_SCORE_THRESHOLD", "70"))
_STAGE_SCORE_THRESHOLD = int(os.environ.get("STAGE_SCORE_THRESHOLD", "70"))

_AND_SPLIT_RE = re.compile(r"\s*(?:,| and )\s*", re.IGNORECASE)
_TO_CLAUSE_RE = re.compile(r"\bto\b", re.IGNORECASE)

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()

def _token_set_ratio(a: str, b: str) -> int:
    """
    Simple token-set similarity (0..100). No external deps.
    """
    ta = set(_norm(a).split())
    tb = set(_norm(b).split())
    if not ta or not tb:
        return 0
    inter = ta & tb
    if not inter:
        # fallback to SequenceMatcher if token sets do not intersect
        return int(SequenceMatcher(None, _norm(a), _norm(b)).ratio() * 100)
    # Jaccard-style scaled to 100
    score = 100 * len(inter) / len(ta | tb)
    # Blend with SequenceMatcher for better partials
    sm = int(SequenceMatcher(None, _norm(a), _norm(b)).ratio() * 100)
    return int((score * 0.6) + (sm * 0.4))

def _best_match(query: str, candidates: Iterable[Tuple[str, Any]]) -> Tuple[int, Tuple[str, Any] | None]:
    """
    candidates: iterable of (display_name, payload)
    Returns (score, (display_name, payload) or (0, None))
    """
    best = (0, None)
    for name, payload in candidates:
        sc = _token_set_ratio(query, name)
        if sc > best[0]:
            best = (sc, (name, payload))
    return best

def _parse_move_prompt(prompt: str) -> list[dict]:
    """
    Parse strings like:
      "move alice and bob to onsite, and charlie to offer accepted"
    into:
      [ { "candidateQueries": ["alice", "bob"], "stageQuery": "onsite" },
        { "candidateQueries": ["charlie"], "stageQuery": "offer accepted" } ]
    Very forgiving with commas/ands.
    """
    text = prompt.strip()
    # split into "... to STAGE" clauses while retaining stage parts
    # We'll greedily split by ' to ' boundaries.
    parts = re.split(r"\s*\bto\b\s*", text, flags=re.IGNORECASE)
    groups = []
    if len(parts) < 2:
        # no 'to' -> interpret whole thing as candidates with missing stage (handled later)
        return []

    # Rebuild pairs: [candidates_part, stage_part] [+ any next candidates part consumed already]
    # Example: ["move alice and bob ", " onsite, and charlie ", " offer accepted"]
    # We walk in steps of 2.
    head = parts[0]
    tail = parts[1:]
    # The first "candidates" segment is inside head's trailing text.
    current_candidates_text = head

    for i, seg in enumerate(tail):
        # seg starts with stage (possibly followed by ", and NAME(S)" that actually belong to the next clause)
        # Try to split stage from the next " , and ... " that signals a new candidate group
        # We'll look for a ", and " followed by a name+ " to " later; but to keep it robust:
        # take everything up to the last comma/and chunk as stage, unless another ' to ' exists (already split).
        # Simpler: take whole seg as stage for this group; any subsequent group comes from subsequent iterations.
        stage_text = seg

        # Build candidate queries from the previous chunk
        cand_text = re.sub(r"^\s*move\s+", "", current_candidates_text, flags=re.IGNORECASE)
        cand_text = cand_text.strip(" ,.")
        if cand_text:
            cand_list = [c for c in _AND_SPLIT_RE.split(cand_text) if c]
            groups.append({
                "candidateQueries": cand_list,
                "stageQuery": stage_text.strip(" ,.")
            })

        # Prepare for next candidates chunk: by default it's empty unless next segment exists
        current_candidates_text = ""

    return groups


# =========================
# Drive Scanners for a Position
# =========================

def _load_pipeline_stages(drive, role_id: str) -> list[dict]:
    """
    Given a role folder ID, return list of stage dicts under its 'Hiring Pipeline'.
    """
    pipeline = _find_child_folder_by_name(drive, role_id, "Hiring Pipeline")
    if not pipeline:
        return []
    stages = list(_iter_child_folders(drive, pipeline["id"]))
    return [{"id": s["id"], "name": s.get("name", "")} for s in stages]

def _scan_stage_files(drive, stage_id: str) -> list[dict]:
    """
    Return files (id, name, parents, mimeType) for a stage.
    """
    q = (
        "trashed=false "
        "and mimeType!='application/vnd.google-apps.folder' "
        f"and '{stage_id}' in parents"
    )
    # include parents to allow move operation computation
    items = []
    for f in _drive_list(
        drive,
        q=q,
        fields="nextPageToken, files(id,name,mimeType,parents)"
    ):
        items.append({"id": f["id"], "name": f.get("name", ""), "parents": f.get("parents", []), "mimeType": f.get("mimeType", "")})
    return items

def _build_candidate_index(drive, role_id: str) -> tuple[list[dict], dict]:
    """
    Build an index of all files inside every stage under 'Hiring Pipeline' for a role.
    Returns (stages, {fileId: {..., stageId, stageName}}).
    """
    stages = _load_pipeline_stages(drive, role_id)
    file_index = {}
    for st in stages:
        for f in _scan_stage_files(drive, st["id"]):
            file_index[f["id"]] = {
                "id": f["id"],
                "name": f["name"],
                "stageId": st["id"],
                "stageName": st["name"],
                "mimeType": f["mimeType"],
            }
    return stages, file_index

def _list_roles_under_department(drive, dept_folder_id: str) -> list[dict]:
    return [{"id": r["id"], "name": r.get("name", "")} for r in _iter_child_folders(drive, dept_folder_id)]


# =========================
# Resolution & Move Engine
# =========================

def _resolve_best_stage(stage_query: str, stages: list[dict]) -> tuple[int, dict | None]:
    return _best_match(stage_query, [(s["name"], s) for s in stages])

def _resolve_best_candidate_file(cand_query: str, file_index: dict) -> tuple[int, dict | None]:
    return _best_match(cand_query, [(meta["name"], meta) for meta in file_index.values()])

def _move_file_between_stages(drive, file_id: str, from_stage_id: str, to_stage_id: str) -> dict:
    return drive.files().update(
        fileId=file_id,
        addParents=to_stage_id,
        removeParents=from_stage_id,
        fields="id, parents",
        supportsAllDrives=True
    ).execute()



# Size/time limits
MAX_FILE_BYTES = int(os.environ.get("MAX_FILE_BYTES", str(25 * 1024 * 1024)))  # 25 MB

# Configure logging once (top of file)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 
BOLD_RE = re.compile(r"\*\*(.+?)\*\*")

# ----- Branding -----
LOGO_URI = "https://drive.google.com/uc?id=1tGh_4cmuRhLOX4ZYcQsaX_F1bcP0x6L4"
LOGO_WIDTH_PT = 200   # ~1.67 in (adjust as you like)
LOGO_HEIGHT_PT = 72   # ~0.5 in  (keep aspect ratio-ish)
# End of Logo branding

IMPERSONATE_HEADER = "x-user-email"  # or "x-impersonate-user" # Choose a header name you’ll set from your app / gateway
DEFAULT_IMPERSONATION_SUBJECT = os.environ.get("DEFAULT_IMPERSONATION_SUBJECT")  # optional fallback

API_KEY = os.environ.get("API_KEY")  # set in Railway "Variables"

app = FastAPI(title="Recruiting Sheet Insights")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend.example.com", "http://localhost:3000"],  # or ["*"] while testing
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # or ["*"]
    allow_headers=["*"],  # or explicitly ["x-api-key", "x-user-email", "content-type", "authorization"]
)

# Below are helper functions

PDF_MAGIC = b"%PDF-"

#Start of Helpers for Candidate Summary end point

# ======== ADD: new helper to pull plain text from a Google Doc ========
def _doc_text_from_google_doc(docs, doc_id: str) -> str:
    """
    Read a Google Doc and return its plain text by walking the structural elements.
    """
    doc = docs.documents().get(documentId=doc_id).execute()
    elements = doc.get("body", {}).get("content", [])
    out_lines: list[str] = []

    def read_elements(elems) -> None:
        for el in elems:
            # Paragraphs
            para = el.get("paragraph")
            if para:
                text_chunks = []
                for el2 in para.get("elements", []):
                    tr = el2.get("textRun")
                    if tr and "content" in tr:
                        text_chunks.append(tr["content"])
                out_lines.append("".join(text_chunks).rstrip("\n"))
            # Tables
            table = el.get("table")
            if table:
                for row in table.get("tableRows", []):
                    for cell in row.get("tableCells", []):
                        read_elements(cell.get("content", []))
            # Table of contents
            toc = el.get("tableOfContents")
            if toc:
                read_elements(toc.get("content", []))

    read_elements(elements)
    return "\n".join(out_lines).strip()


# ======== ADD: text extraction wrapper for stage files ========
_DOC_EXT_RE = re.compile(r"\.(docx?|pdf)$", re.IGNORECASE)

def _is_doc_or_pdf(file_name: str, mime_type: str) -> bool:
    """
    Decide whether to attempt extraction. Prefer filename (robust to Drive MIME quirks).
    """
    if _DOC_EXT_RE.search(file_name or ""):
        return True
    # Also cover native Google Docs
    if mime_type == "application/vnd.google-apps.document":
        return True
    return False

def _extract_text_from_file(drive, docs, file_obj: dict) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract text from a Drive file.

    Strategy:
      * If file is a Google Doc -> read via Docs API.
      * If file looks like .doc/.docx/.pdf -> copy/convert to Google Doc, read, then delete temp.
      * Returns (text, error). Only one of them will be non-empty.
    """
    fid = file_obj.get("id")
    fname = file_obj.get("name", "")
    mime = file_obj.get("mimeType", "")

    # Case 1: Already a Google Doc
    if mime == "application/vnd.google-apps.document":
        try:
            return _doc_text_from_google_doc(docs, fid), None
        except Exception as e:
            logger.exception("Failed reading Google Doc %s (%s)", fname, fid)
            return None, f"Failed reading Google Doc: {e}"

    # Case 2: Try to convert .doc/.docx/.pdf to Google Doc via copy()
    if _is_doc_or_pdf(fname, mime):
        temp_id = None
        try:
            # Copy with target mimeType = Google Doc (converts supported formats, incl. docx/pdf)
            temp = drive.files().copy(
                fileId=fid,
                body={"mimeType": "application/vnd.google-apps.document"},
                supportsAllDrives=True,
                fields="id"
            ).execute()
            temp_id = temp.get("id")
            text = _doc_text_from_google_doc(docs, temp_id)
            return text, None
        except Exception as e:
            logger.exception("Failed converting/extracting %s (%s)", fname, fid)
            return None, f"Failed converting/extracting: {e}"
        finally:
            # Best-effort cleanup
            if temp_id:
                try:
                    drive.files().delete(fileId=temp_id, supportsAllDrives=True).execute()
                except Exception as del_err:
                    logger.warning("Could not delete temp converted doc %s: %s", temp_id, del_err)

    # Fallback: unsupported type
    return None, "Unsupported type for extraction"


def _drive_list(drive, **kwargs) -> Iterable[dict]:
    """
    Wrapper for drive.files().list with pagination.
    Always sets includeItemsFromAllDrives / supportsAllDrives and a lean fields set.
    Yields items across all pages.
    """
    params = {
        "includeItemsFromAllDrives": True,
        "supportsAllDrives": True,
        "fields": "nextPageToken, files(id,name,mimeType,parents)",
    }
    params.update(kwargs)
    while True:
        resp = drive.files().list(**params).execute()
        for f in resp.get("files", []):
            yield f
        token = resp.get("nextPageToken")
        if not token:
            break
        params["pageToken"] = token


def _iter_child_folders(drive, parent_id: str) -> Iterable[dict]:
    q = (
        "mimeType='application/vnd.google-apps.folder' "
        "and trashed=false "
        f"and '{parent_id}' in parents"
    )
    yield from _drive_list(drive, q=q)


def _iter_child_files(drive, parent_id: str) -> Iterable[dict]:
    q = (
        "trashed=false "
        "and mimeType!='application/vnd.google-apps.folder' "
        f"and '{parent_id}' in parents"
    )
    yield from _drive_list(drive, q=q)

def _find_child_folder_by_name(drive, parent_id: str, name: str) -> dict | None:
    q = (
        "mimeType='application/vnd.google-apps.folder' "
        "and trashed=false "
        f"and name='{name}' "
        f"and '{parent_id}' in parents"
    )
    for f in _drive_list(drive, q=q):
        return f
    return None


def _iter_files_recursive(drive, folder_id: str) -> Iterable[dict]:
    # files directly here
    for f in _iter_child_files(drive, folder_id):
        yield f
    # and recurse into subfolders
    for sub in _iter_child_folders(drive, folder_id):
        yield from _iter_files_recursive(drive, sub["id"])

#End of Helpers for Candidates Summary


def _is_valid_pdf(raw: bytes) -> bool:
    """
    Lightweight validation to check if a file is a plausible PDF.
    Logs reasons when rejected.
    """
    # Too small to be a useful PDF (adjust threshold if needed)
    if len(raw) < 1000:  # ~1 KB
        logger.warning("PDF rejected: file size too small (%d bytes)", len(raw))
        return False

    # Check PDF magic header
    if not raw.startswith(PDF_MAGIC):
        logger.warning("PDF rejected: missing %%PDF- header")
        return False

    # Check EOF marker (must be present, even if whitespace follows)
    if not raw.rstrip().endswith(b"%%EOF"):
        logger.warning("PDF rejected: missing %%EOF marker")
        return False

    return True

        

def _extract_subject_from_request(req: Request) -> Optional[str]:
    """
    Determine which user to impersonate.
    Priority: request header -> query param -> env default -> None (falls back to raw SA).
    """
    # Header first (recommended)
    hed = req.headers.get(IMPERSONATE_HEADER)
    if hed and hed.strip():
        return hed.strip()

    # Optional: allow a query param for testing (remove if you don't want this)
    qp = req.query_params.get("userEmail")
    if qp and qp.strip():
        return qp.strip()

    # Optional fallback configured via env
    if DEFAULT_IMPERSONATION_SUBJECT:
        return DEFAULT_IMPERSONATION_SUBJECT.strip()

    return None

def strip_heading_markers(s: str):
    # returns (clean_text, kind_override)
    if s.startswith("### "):
        return s[4:], "H3"
    if s.startswith("## "):
        return s[3:], "H2"
    if s.startswith("# "):
        return s[2:], "H1"
    return s, None

def strip_inline_bold_and_spans(s: str):
    """Return (clean_text, spans) where spans are (start,end) in CLEAN text."""
    spans = []
    out = []
    idx = 0
    for m in BOLD_RE.finditer(s):
        out.append(s[idx:m.start()])           # text before
        bold_text = m.group(1)                 # without ** **
        start_in_out = sum(len(x) for x in out)
        out.append(bold_text)
        end_in_out = start_in_out + len(bold_text)
        spans.append((start_in_out, end_in_out))
        idx = m.end()
    out.append(s[idx:])
    return "".join(out), spans

def get_clients(subject: Optional[str] = None): # client builder accepts now optional subjects for impersonation
    """
    Build Sheets/Drive/Docs clients.
    If `subject` is provided and DWD is configured, we impersonate that user,
    so created files/folders are owned by them (not by the service account).
    """
    info = json.loads(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"])
    base_creds = Credentials.from_service_account_info(info, scopes=[
        "https://www.googleapis.com/auth/drive",
        "https://www.googleapis.com/auth/documents",
    ])

    creds = base_creds.with_subject(subject) if subject else base_creds

    try:
        sheets = build("sheets", "v4", credentials=creds)
        drive  = build("drive",  "v3", credentials=creds)
        docs   = build("docs",   "v1", credentials=creds)
    except RefreshError as e:
        # Helpful error if DWD isn’t configured, or subject not allowed for the scopes
        raise HTTPException(
            500,
            f"Failed to obtain delegated credentials. "
            f"Check DWD configuration/scopes for subject='{subject}'. Details: {e}"
        )

    return sheets, drive, docs


def require_api_key(req: Request):
    if not API_KEY or req.headers.get("x-api-key") != API_KEY:
        raise HTTPException(403, "Forbidden")


def create_folder(drive, name: str, parent_id: str) -> str:
    """Create a folder and return its Google Drive file ID"""
    metadata = {
        "name": name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_id]
    }
    folder = drive.files().create(body=metadata, fields="id, owners", supportsAllDrives=True ).execute()
    print("create_folder owners:", folder.get("owners"))  # 👈 logs impersonated owner or shared drive
    return folder["id"]

def create_named_subfolder(drive, parent_id: str, subfolder_name: str) -> str: 
    """Always create a new subfolder inside parent folder."""
    return create_folder(drive, subfolder_name, parent_id)



def create_google_doc(docs, drive, folder_id: str, title: str, content: str) -> str:
    # 1) Create the Google Doc in the target folder
    file_metadata = {"name": title, "mimeType": "application/vnd.google-apps.document", "parents": [folder_id]}
    file = drive.files().create(body=file_metadata, fields="id", supportsAllDrives=True).execute()
    doc_id = file["id"]

    # 2) Fetch doc length (to clear)
    doc = docs.documents().get(documentId=doc_id).execute()
    doc_length = doc.get("body").get("content")[-1]["endIndex"]

    # 3) Clear existing text
    requests = []
    if doc_length > 2:
        requests.append({
            "deleteContentRange": {"range": {"startIndex": 1, "endIndex": doc_length - 1}}
        })
    
    # Apply the delete (if any), then reset 'requests'
    if requests:
        docs.documents().batchUpdate(documentId=doc_id, body={"requests": requests}).execute()
    requests = []
    
    # ----------------------------------------------------------------
    # HEADER: same on all pages — always run this, not only when doc had content
    # ----------------------------------------------------------------
    phase1 = docs.documents().batchUpdate(documentId=doc_id, body={"requests": [
        {"updateDocumentStyle": {
            "documentStyle": {"useFirstPageHeaderFooter": False},
            "fields": "useFirstPageHeaderFooter"
        }},
        {"createHeader": {"type": "DEFAULT"}}
    ]}).execute()
    
    # put the fixed snippet here
    header_id = None
    for r in phase1.get("replies", []):
        ch = r.get("createHeader")
        if ch:
            header_id = ch.get("headerId")
            break
    
    # Insert the logo into the DEFAULT header
    if header_id and LOGO_URI:
        docs.documents().batchUpdate(documentId=doc_id, body={"requests": [
            {"insertText": {"location": {"segmentId": header_id, "index": 0}, "text": "\n"}},
            {"insertInlineImage": {
                "location": {"segmentId": header_id, "index": 1},
                "uri": LOGO_URI,
                "objectSize": {
                    "width":  {"magnitude": LOGO_WIDTH_PT,  "unit": "PT"},
                    "height": {"magnitude": LOGO_HEIGHT_PT, "unit": "PT"}
                }
            }},
            {"updateParagraphStyle": {
                "range": {"segmentId": header_id, "startIndex": 0, "endIndex": 2},
                "paragraphStyle": {"alignment": "START"},
                "fields": "alignment"
            }},
        ]}).execute()


    # --- Normalize content ---
    # Remove leading indentation from the triple-quoted template, drop leading/trailing blank lines
    norm_txt = textwrap.dedent(content).strip("\n")
    raw_lines = norm_txt.splitlines()

    # --- Preprocess: insert a blank line BEFORE any "### ..." heading line ---
    prepped_lines = []
    for line in raw_lines:
        trimmed = line.strip()
        if trimmed.startswith("###"):
            prepped_lines.append("")   # ensures a blank paragraph above the section
        prepped_lines.append(line)

    insert_index = 1
    requests = []  # <- restart requests for body content now
    para_ranges = []   # tuples: (start, end, kind) where kind in {"H1","H2","H3","NORMAL","BULLET"}
    list_groups = []   # tuples: (group_start, group_end)
    inline_bold_spans = []  # collect absolute (start,end) ranges for bold  <-- added
    in_list = False
    group_start = None

    def insert_line(txt: str):
        nonlocal insert_index, requests
        # Always end with newline so ranges include it
        if not txt.endswith("\n"):
            txt += "\n"
        start = insert_index
        end = start + len(txt)
        requests.append({"insertText": {"location": {"index": start}, "text": txt}})
        insert_index = end
        return start, end

    for i, line in enumerate(prepped_lines):
        trimmed = line.strip()

        # Handle blank lines (including the ones we injected)
        if not trimmed:
            # Close any open list group
            if in_list:
                list_groups.append((group_start, insert_index))
                in_list = False
                group_start = None
            # Actually insert the blank paragraph
            insert_line("")  # writes just "\n"
            continue

        # --------- REPLACED LOOP BODY (Markdown → Docs) ----------
        is_bullet = trimmed.startswith("- ")

        # handle Markdown headings first
        clean, kind_override = strip_heading_markers(trimmed)

        # first non-blank line becomes H1 if no explicit # heading
        is_h1_by_position = (i == 0 and kind_override is None)
        is_h2_by_colon = clean.endswith(":")

        # bullets: remove '- ' before inserting; otherwise use cleaned text
        text_for_insert = clean[2:] if is_bullet else clean

        # inline bold: strip ** and collect spans relative to the cleaned line
        text_for_insert, bold_spans_rel = strip_inline_bold_and_spans(text_for_insert)

        start, end = insert_line(text_for_insert)

        # record absolute bold ranges for later updateTextStyle
        for bstart, bend in bold_spans_rel:
            inline_bold_spans.append((start + bstart, start + bend))

        if is_bullet:
            if not in_list:
                in_list = True
                group_start = start
            para_ranges.append((start, end, "BULLET"))
        else:
            if in_list:
                list_groups.append((group_start, insert_index))
                in_list = False
                group_start = None

            if kind_override:              # H1/H2/H3 from #,##,###
                para_ranges.append((start, end, kind_override))
            elif is_h1_by_position:        # fallback: first line is H1
                para_ranges.append((start, end, "H1"))
            elif is_h2_by_colon:
                para_ranges.append((start, end, "H2"))
            else:
                para_ranges.append((start, end, "NORMAL"))
        # --------- END REPLACED LOOP BODY ----------

    if in_list:
        list_groups.append((group_start, insert_index))

    # --- Apply paragraph styles ---
    for start, end, kind in para_ranges:
        if kind == "H1":
            requests.append({
                "updateParagraphStyle": {
                    "range": {"startIndex": start, "endIndex": end},
                    "paragraphStyle": {"namedStyleType": "HEADING_1"},
                    "fields": "namedStyleType"
                }
            })
        elif kind == "H2":
            requests.append({
                "updateParagraphStyle": {
                    "range": {"startIndex": start, "endIndex": end},
                    "paragraphStyle": {"namedStyleType": "HEADING_2"},
                    "fields": "namedStyleType"
                }
            })
        elif kind == "H3":
            requests.append({
                "updateParagraphStyle": {
                    "range": {"startIndex": start, "endIndex": end},
                    "paragraphStyle": {"namedStyleType": "HEADING_3"},
                    "fields": "namedStyleType"
                }
            })
        elif kind == "NORMAL":
            requests.append({
                "updateParagraphStyle": {
                    "range": {"startIndex": start, "endIndex": end},
                    "paragraphStyle": {"namedStyleType": "NORMAL_TEXT"},
                    "fields": "namedStyleType"
                }
            })

    # --- Apply bullets once per contiguous group ---
    for gs, ge in list_groups:
        requests.append({
            "createParagraphBullets": {
                "range": {"startIndex": gs, "endIndex": ge},
                "bulletPreset": "BULLET_DISC_CIRCLE_SQUARE"
            }
        })

    # --- Apply inline bold (added) ---
    for bs, be in inline_bold_spans:
        requests.append({
            "updateTextStyle": {
                "range": {"startIndex": bs, "endIndex": be},
                "textStyle": {"bold": True},
                "fields": "bold"
            }
        })

    # 5) Execute body insertions & styles
    if requests:
        docs.documents().batchUpdate(documentId=doc_id, body={"requests": requests}).execute()

    return doc_id


#End of Helper Functions



class PositionRequest(BaseModel):
    name: str
    department: str = "Software Engineering"
    dryRun: bool = False
    userEmail: Optional[str] = None  # <-- add this

@app.post("/positions/create")
def create_position(request: Request, body: PositionRequest):
    require_api_key(request)
    # prefer body.userEmail; then header/query/env; else SA
    subject = body.userEmail or _extract_subject_from_request(request)
    _, drive, _ = get_clients(subject)

    name = body.name
    department = body.department

    if body.dryRun:
        return {
            "message": f"[dryRun] Would create role '{name}' in {department}",
            "positionId": "",
            "departmentFolderId": "",
            "created": False
        }

    DEPARTMENTS_FOLDER_ID = os.environ.get("DEPARTMENTS_FOLDER_ID")
    if not DEPARTMENTS_FOLDER_ID:
        raise HTTPException(500, "DEPARTMENTS_FOLDER_ID env var not set")

    # Step 0: Ensure department folder exists
    query = (
        f"mimeType='application/vnd.google-apps.folder' "
        f"and trashed=false and name='{department}' "
        f"and '{DEPARTMENTS_FOLDER_ID}' in parents"
    )
    results = drive.files().list(q=query, fields="files(id,name)", includeItemsFromAllDrives=True, supportsAllDrives=True).execute()
    items = results.get("files", [])

    if items:
        department_folder_id = items[0]["id"]
    else:
        department_folder_id = create_folder(drive, department, DEPARTMENTS_FOLDER_ID)

    # Step 1: Check if role folder already exists
    query = (
        f"mimeType='application/vnd.google-apps.folder' "
        f"and trashed=false and name='{name}' "
        f"and '{department_folder_id}' in parents"
    )
    results = drive.files().list(q=query, fields="files(id,name)", includeItemsFromAllDrives=True, supportsAllDrives=True).execute()
    if results.get("files"):
        return {
            "message": f"Role '{name}' already exists in {department}",
            "positionId": results["files"][0]["id"],
            "departmentFolderId": department_folder_id,
            "created": False
        }

    # Step 2: Create role folder
    position_id = create_folder(drive, name, department_folder_id)

    
    return {
        "message": f"Role '{name}' created successfully in {department}",
        "positionId": position_id,
        "departmentFolderId": department_folder_id,
        "created": True
    }


@app.get("/positions/list") # End point that understands what department folders already exist under Hiring folder
def list_positions(request: Request, department: Optional[str] = None):
    require_api_key(request)
    subject = _extract_subject_from_request(request)
    _, drive, _ = get_clients(subject)

    DEPARTMENTS_FOLDER_ID = os.environ.get("DEPARTMENTS_FOLDER_ID")
    if not DEPARTMENTS_FOLDER_ID:
        raise HTTPException(500, "DEPARTMENTS_FOLDER_ID env var not set")

    # If department specified, check inside it
    parent_id = DEPARTMENTS_FOLDER_ID
    if department:
        query = (
            f"mimeType='application/vnd.google-apps.folder' "
            f"and trashed=false and name='{department}' "
            f"and '{DEPARTMENTS_FOLDER_ID}' in parents"
        )
        results = drive.files().list(q=query, fields="files(id,name)", includeItemsFromAllDrives=True, supportsAllDrives=True).execute()
        items = results.get("files", [])
        if not items:
            return {"roles": [], "department": department, "exists": False}
        parent_id = items[0]["id"]

    # List folders under parent (roles or departments)
    query = f"mimeType='application/vnd.google-apps.folder' and trashed=false and '{parent_id}' in parents"
    results = drive.files().list(q=query, fields="files(id,name)", includeItemsFromAllDrives=True, supportsAllDrives=True).execute()
    roles = [{"id": f["id"], "name": f["name"]} for f in results.get("files", [])]

    return {
        "department": department or "Hiring",
        "roles": roles,
        "exists": True
    }



class CreateJDRequest(BaseModel):
    positionId: str
    roleName: str
    content: Optional[str] = None
    userEmail: Optional[str] = None  # <-- add this

@app.post("/positions/createJD")
def create_jd(request: Request, body: CreateJDRequest):
    require_api_key(request)
    subject = body.userEmail or _extract_subject_from_request(request)
    _, drive, docs = get_clients(subject)

    # ✅ Always create a fresh subfolder for JD
    jd_folder_id = create_named_subfolder(drive, body.positionId, "Job Descriptions")
    
    # ✅ Default polished template
    content = body.content or f"""
        Job Description – {body.roleName}
        
        Role Summary:
        We are seeking a strategic and results-oriented {body.roleName} to drive our initiatives and ensure measurable business impact. 
        This role requires strong leadership, cross-functional collaboration, and a proven ability to deliver results in fast-paced environments.
        
        Key Responsibilities:
        - Develop and execute the company’s {body.roleName} strategy.
        - Collaborate with Product, Sales, and Engineering teams to align goals.
        - Lead and mentor a high-performing team to deliver against objectives.
        - Define, measure, and report on KPIs such as ROI, retention, and conversion.
        - Partner with leadership to shape the company’s strategic direction.
        
        Requirements:
        - 7+ years of professional experience, including at least 3 years in a leadership role.
        - Proven success in executing scalable strategies in dynamic environments.
        - Strong analytical and decision-making skills, with attention to detail.
        - Excellent communication, presentation, and stakeholder management abilities.
        
        """

    file_id = create_google_doc(docs, drive, jd_folder_id, f"JD - {body.roleName}", content)
    return {
        "message": f"JD created for {body.roleName}",
        "fileId": file_id,
        "folderId": jd_folder_id,
        "docLink": f"https://docs.google.com/document/d/{file_id}/edit"
    }

class CreateScreeningRequest(BaseModel):
    positionId: str
    roleName: str
    content: Optional[str] = None
    userEmail: Optional[str] = None  # for impersonation

@app.post("/positions/createScreeningTemplate")
def create_screening(request: Request, body: CreateScreeningRequest):
    require_api_key(request)
    subject = body.userEmail or _extract_subject_from_request(request)
    _, drive, docs = get_clients(subject)

    # ✅ Always create a fresh subfolder for Screening
    screening_folder_id = create_named_subfolder(drive, body.positionId, "Screening Templates")

    # ✅ Default polished template
    content = body.content or f"""
        Screening Template – {body.roleName}
        
        Candidate Information:
        - Name: ______________________________________
        - Date: ______________________________________
        
        Screening Questions:
        1. Why are you interested in the role of {body.roleName}?
        2. Please describe your most relevant skills and experience for this position.
        3. What achievements best demonstrate your impact in previous roles?
        4. How do you typically approach problem-solving in your area of expertise?
        5. What motivates you to join our company at this stage of growth?
        
        Evaluator Notes:
        __________________________________________________________
        __________________________________________________________
        __________________________________________________________
        
        Recommendation:
        - Progress to next stage
        - Hold for review
        - Reject
        
                """

    file_id = create_google_doc(docs, drive, screening_folder_id, f"Screening Template - {body.roleName}", content)
    return {
        "message": f"Screening template created for {body.roleName}",
        "fileId": file_id,
        "folderId": screening_folder_id,
        "docLink": f"https://docs.google.com/document/d/{file_id}/edit"
    }


class CreateScoringRequest(BaseModel):
    positionId: str
    roleName: str
    content: Optional[str] = None
    userEmail: Optional[str] = None  # for impersonation

@app.post("/positions/createScoringModel")
def create_scoring(request: Request, body: CreateScoringRequest):
    require_api_key(request)
    subject = body.userEmail or _extract_subject_from_request(request)
    _, drive, docs = get_clients(subject)

    # ✅ Always create a fresh subfolder for Scoring
    scoring_folder_id = create_named_subfolder(drive, body.positionId, "Scoring Rubrics")

    # ✅ Default polished template
    content = body.content or f"""
        Interview Scoring Rubric – {body.roleName}
        
        Scoring Guidelines:
        Rate each category on a scale of 1 (Poor) to 5 (Excellent). 
        Provide written feedback to support your score.
        
        Evaluation Categories:
        - Role Expertise (1–5): Depth of knowledge and skills relevant to {body.roleName}.
        - Problem-Solving (1–5): Ability to analyze challenges and propose solutions.
        - Communication (1–5): Clarity, articulation, and ability to collaborate effectively.
        - Culture Fit (1–5): Alignment with company values, adaptability, and teamwork.
        - Leadership & Initiative (1–5): Ability to inspire, mentor, and take ownership.
        
        Overall Score:
        /25
        
        Evaluator Comments:
        __________________________________________________________
        __________________________________________________________
        __________________________________________________________
        """


    file_id = create_google_doc(docs, drive, scoring_folder_id, f"Scoring Rubric - {body.roleName}", content)
    return {
        "message": f"Scoring rubric created for {body.roleName}",
        "fileId": file_id,
        "folderId": scoring_folder_id,
        "docLink": f"https://docs.google.com/document/d/{file_id}/edit"
    }


class CreateDepartmentsRequest(BaseModel):
    names: List[str]
    userEmail: Optional[str] = None  # impersonation

@app.post("/departments/create")
def create_departments(request: Request, body: CreateDepartmentsRequest):
    require_api_key(request)
    subject = body.userEmail or _extract_subject_from_request(request)
    _, drive, _ = get_clients(subject)

    HIRING_FOLDER_ID = os.environ.get("HIRING_FOLDER_ID")
    if not HIRING_FOLDER_ID:
        raise HTTPException(500, "HIRING_FOLDER_ID env var not set")

    # 1. Look ONLY for "Departments" folder directly under Hiring
    query = (
        "mimeType='application/vnd.google-apps.folder' "
        "and trashed=false and name='Departments' "
        f"and '{HIRING_FOLDER_ID}' in parents"
    )
    results = drive.files().list(
        q=query,
        fields="files(id,name,parents)",
        includeItemsFromAllDrives=True,
        supportsAllDrives=True
    ).execute()

    departments_folder_id = None
    created_root = False

    if results.get("files"):
        # Always pick the one that is directly inside Hiring
        for f in results["files"]:
            if HIRING_FOLDER_ID in f.get("parents", []):
                departments_folder_id = f["id"]
                break

    if not departments_folder_id:
        # Create "Departments" folder if not found directly under Hiring
        departments_folder_id = create_folder(drive, "Departments", HIRING_FOLDER_ID)
        created_root = True

    created_departments = []
    for dept in body.names:
        # Check if department folder already exists inside "Departments"
        query = (
            "mimeType='application/vnd.google-apps.folder' "
            f"and trashed=false and name='{dept}' "
            f"and '{departments_folder_id}' in parents"
        )
        existing = drive.files().list(
            q=query,
            fields="files(id,name)",
            includeItemsFromAllDrives=True,
            supportsAllDrives=True
        ).execute()

        if existing.get("files"):
            created_departments.append({
                "name": dept,
                "id": existing["files"][0]["id"],
                "created": False
            })
        else:
            dept_id = create_folder(drive, dept, departments_folder_id)
            created_departments.append({
                "name": dept,
                "id": dept_id,
                "created": True
            })

    return {
        "message": "Departments processed successfully",
        "departmentsFolderId": departments_folder_id,
        "createdRootFolder": created_root,
        "departments": created_departments
    }


class HiringFlow(BaseModel):
    flowName: str
    stages: List[str]

class CreateHiringFlowsRequest(BaseModel):
    flows: List[HiringFlow]
    userEmail: Optional[str] = None  # impersonation

@app.post("/HiringFlows/create")
def create_hiring_flows(request: Request, body: CreateHiringFlowsRequest):
    require_api_key(request)
    subject = body.userEmail or _extract_subject_from_request(request)
    _, drive, _ = get_clients(subject)

    HIRING_FOLDER_ID = os.environ.get("HIRING_FOLDER_ID")
    if not HIRING_FOLDER_ID:
        raise HTTPException(500, "HIRING_FOLDER_ID env var not set")

    # 1. Check (or create) "Hiring Flows" folder under Hiring
    query = (
        "mimeType='application/vnd.google-apps.folder' "
        "and trashed=false and name='Hiring Flows' "
        f"and '{HIRING_FOLDER_ID}' in parents"
    )
    results = drive.files().list(
        q=query,
        fields="files(id,name,parents)",
        includeItemsFromAllDrives=True,
        supportsAllDrives=True
    ).execute()

    if results.get("files"):
        flows_folder_id = results["files"][0]["id"]
        created_root = False
    else:
        flows_folder_id = create_folder(drive, "Hiring Flows", HIRING_FOLDER_ID)
        created_root = True

    created_flows = []
    for flow in body.flows:
        # 2. Check (or create) flow subfolder
        query = (
            "mimeType='application/vnd.google-apps.folder' "
            f"and trashed=false and name='{flow.flowName}' "
            f"and '{flows_folder_id}' in parents"
        )
        existing_flow = drive.files().list(
            q=query,
            fields="files(id,name)",
            includeItemsFromAllDrives=True,
            supportsAllDrives=True
        ).execute()

        if existing_flow.get("files"):
            flow_folder_id = existing_flow["files"][0]["id"]
            flow_created = False
        else:
            flow_folder_id = create_folder(drive, flow.flowName, flows_folder_id)
            flow_created = True

        # 3. Create stage subfolders
        created_stages = []
        for stage in flow.stages:
            query = (
                "mimeType='application/vnd.google-apps.folder' "
                f"and trashed=false and name='{stage}' "
                f"and '{flow_folder_id}' in parents"
            )
            existing_stage = drive.files().list(
                q=query,
                fields="files(id,name)",
                includeItemsFromAllDrives=True,
                supportsAllDrives=True
            ).execute()

            if existing_stage.get("files"):
                stage_id = existing_stage["files"][0]["id"]
                stage_created = False
            else:
                stage_id = create_folder(drive, stage, flow_folder_id)
                stage_created = True

            created_stages.append({
                "name": stage,
                "id": stage_id,
                "created": stage_created
            })

        created_flows.append({
            "flowName": flow.flowName,
            "id": flow_folder_id,
            "created": flow_created,
            "stages": created_stages
        })

    return {
        "message": "Hiring flows processed successfully",
        "flowsFolderId": flows_folder_id,
        "createdRootFolder": created_root,
        "flows": created_flows
    }
    

@app.get("/HiringFlows/read")
def read_hiring_flows(request: Request):
    require_api_key(request)
    subject = _extract_subject_from_request(request)
    _, drive, _ = get_clients(subject)

    HIRING_FOLDER_ID = os.environ.get("HIRING_FOLDER_ID")
    if not HIRING_FOLDER_ID:
        raise HTTPException(500, "HIRING_FOLDER_ID env var not set")

    # Locate "Hiring Flows" folder
    query = (
        "mimeType='application/vnd.google-apps.folder' "
        "and trashed=false and name='Hiring Flows' "
        f"and '{HIRING_FOLDER_ID}' in parents"
    )
    results = drive.files().list(
        q=query,
        fields="files(id,name)",
        includeItemsFromAllDrives=True,
        supportsAllDrives=True
    ).execute()

    if not results.get("files"):
        return {"message": "No Hiring Flows found", "flows": []}

    flows_folder_id = results["files"][0]["id"]

    # Get flows inside
    flows = drive.files().list(
        q=f"mimeType='application/vnd.google-apps.folder' and trashed=false and '{flows_folder_id}' in parents",
        fields="files(id,name)",
        includeItemsFromAllDrives=True,
        supportsAllDrives=True
    ).execute().get("files", [])

    flows_data = []
    for flow in flows:
        stages = drive.files().list(
            q=f"mimeType='application/vnd.google-apps.folder' and trashed=false and '{flow['id']}' in parents",
            fields="files(id,name)",
            includeItemsFromAllDrives=True,
            supportsAllDrives=True
        ).execute().get("files", [])
        flows_data.append({
            "flowName": flow["name"],
            "id": flow["id"],
            "stages": [s["name"] for s in stages]
        })

    return {
        "message": "Hiring Flows read successfully",
        "flowsFolderId": flows_folder_id,
        "flows": flows_data
    }


class CreateHiringPipelineRequest(BaseModel):
    hiringFlowName: str
    positionId: str
    userEmail: Optional[str] = None  # for impersonation

@app.post("/HiringPipeline/create")
def create_hiring_pipeline(request: Request, body: CreateHiringPipelineRequest):
    require_api_key(request)
    subject = body.userEmail or _extract_subject_from_request(request)
    _, drive, _ = get_clients(subject)

    HIRING_FOLDER_ID = os.environ.get("HIRING_FOLDER_ID")
    if not HIRING_FOLDER_ID:
        raise HTTPException(500, "HIRING_FOLDER_ID env var not set")

    # Locate "Hiring Flows" folder
    query = (
        "mimeType='application/vnd.google-apps.folder' "
        "and trashed=false and name='Hiring Flows' "
        f"and '{HIRING_FOLDER_ID}' in parents"
    )
    results = drive.files().list(
        q=query,
        fields="files(id,name)",
        includeItemsFromAllDrives=True,
        supportsAllDrives=True
    ).execute()

    if not results.get("files"):
        raise HTTPException(404, "Hiring Flows folder not found")

    flows_folder_id = results["files"][0]["id"]

    # Find requested flow
    query = (
        "mimeType='application/vnd.google-apps.folder' "
        f"and trashed=false and name='{body.hiringFlowName}' "
        f"and '{flows_folder_id}' in parents"
    )
    flow_results = drive.files().list(
        q=query,
        fields="files(id,name)",
        includeItemsFromAllDrives=True,
        supportsAllDrives=True
    ).execute()

    if not flow_results.get("files"):
        raise HTTPException(404, f"Hiring Flow '{body.hiringFlowName}' not found")

    flow_id = flow_results["files"][0]["id"]

    # Get stages inside the flow
    stages = drive.files().list(
        q=f"mimeType='application/vnd.google-apps.folder' and trashed=false and '{flow_id}' in parents",
        fields="files(id,name)",
        includeItemsFromAllDrives=True,
        supportsAllDrives=True
    ).execute().get("files", [])

    # Create "Hiring Pipeline" inside job position
    pipeline_id = create_named_subfolder(drive, body.positionId, "Hiring Pipeline")

    created_stages = []
    for stage in stages:
        sid = create_named_subfolder(drive, pipeline_id, stage["name"])
        created_stages.append({"name": stage["name"], "id": sid})

    return {
        "message": f"Hiring Pipeline created for position {body.positionId} using flow {body.hiringFlowName}",
        "pipelineFolderId": pipeline_id,
        "stages": created_stages
    }


class CandidateFile(BaseModel):
    filename: str
    content: str  # base64-encoded PDF


class UploadJsonRequest(BaseModel):
    candidateNames: List[str]
    departments: List[str]
    roles: List[str]
    hiringStages: List[str]
    files: List[CandidateFile]
    userEmail: Optional[str] = None  # for impersonation


@app.post("/candidates/uploadJson")
def upload_candidates_json(request: Request, body: UploadJsonRequest):
    """
    JSON wrapper for candidate upload (Base64 instead of multipart).
    """

    require_api_key(request)

    # Impersonation
    subject = body.userEmail or _extract_subject_from_request(request)
    _, drive, _ = get_clients(subject)

    n = len(body.candidateNames)
    if not (len(body.departments) == len(body.roles) == len(body.hiringStages) == len(body.files) == n):
        raise HTTPException(400, "Mismatched number of fields")

    processed = []
    DEPARTMENTS_FOLDER_ID = os.environ.get("DEPARTMENTS_FOLDER_ID")
    if not DEPARTMENTS_FOLDER_ID:
        raise HTTPException(500, "DEPARTMENTS_FOLDER_ID env var not set")

    for i in range(n):
        cand_name = body.candidateNames[i].strip()
        dept = body.departments[i].strip()
        role = body.roles[i].strip()
        stage = body.hiringStages[i].strip()
        file = body.files[i]

        # ---- Find department
        query = (
            f"mimeType='application/vnd.google-apps.folder' "
            f"and trashed=false and name='{dept}' "
            f"and '{DEPARTMENTS_FOLDER_ID}' in parents"
        )
        dept_results = drive.files().list(
            q=query, fields="files(id,name,parents)",
            includeItemsFromAllDrives=True, supportsAllDrives=True
        ).execute()
        if not dept_results.get("files"):
            raise HTTPException(404, f"Department '{dept}' not found")
        dept_id = dept_results["files"][0]["id"]

        # ---- Find role under department
        query = (
            f"mimeType='application/vnd.google-apps.folder' "
            f"and trashed=false and name='{role}' "
            f"and '{dept_id}' in parents"
        )
        role_results = drive.files().list(
            q=query, fields="files(id,name)",
            includeItemsFromAllDrives=True, supportsAllDrives=True
        ).execute()
        if not role_results.get("files"):
            raise HTTPException(404, f"Role '{role}' not found in Department '{dept}'")
        role_id = role_results["files"][0]["id"]

        # ---- Find Hiring Pipeline
        query = (
            f"mimeType='application/vnd.google-apps.folder' and trashed=false "
            f"and name='Hiring Pipeline' and '{role_id}' in parents"
        )
        pipeline = drive.files().list(
            q=query, fields="files(id,name)",
            includeItemsFromAllDrives=True, supportsAllDrives=True
        ).execute()
        if not pipeline.get("files"):
            raise HTTPException(404, f"Hiring Pipeline not found for role '{role}' in Department '{dept}'")
        pipeline_id = pipeline["files"][0]["id"]

        # ---- Find stage under pipeline
        query = (
            f"mimeType='application/vnd.google-apps.folder' and trashed=false "
            f"and name='{stage}' and '{pipeline_id}' in parents"
        )
        stage_result = drive.files().list(
            q=query, fields="files(id,name)",
            includeItemsFromAllDrives=True, supportsAllDrives=True
        ).execute()
        if not stage_result.get("files"):
            raise HTTPException(404, f"Stage '{stage}' not found under Hiring Pipeline for '{role}'")
        stage_id = stage_result["files"][0]["id"]

        # ---- Decode Base64 file
        try:
            raw = base64.b64decode(file.content)
        except Exception:
            raise HTTPException(400, f"File '{file.filename}' is not valid base64")

        if not _is_valid_pdf(raw):
            raise HTTPException(400, f"File '{file.filename}' is not a valid/complete PDF")

        # ---- Upload to Drive
        safe_name = _safe_pdf_name(cand_name, file.filename)
        media = MediaIoBaseUpload(io.BytesIO(raw), mimetype="application/pdf", resumable=False)
        file_obj = drive.files().create(
            body={"name": safe_name, "parents": [stage_id]},
            media_body=media,
            fields="id,parents",
            supportsAllDrives=True
        ).execute()
        uploaded_file_id = file_obj["id"]

        processed.append({
            "candidateName": cand_name,
            "department": dept,
            "role": role,
            "stage": stage,
            "uploadedFileId": uploaded_file_id,
            "uploadedTo": stage_id
        })

    return {"message": "Candidates uploaded successfully", "processed": processed}

class StageFileExtract(BaseModel):
    id: str
    name: str
    mimeType: str
    text: Optional[str] = None
    error: Optional[str] = None

class StageLite(BaseModel):
    id: str
    name: str
    files: List[StageFileExtract] = Field(default_factory=list)

class RoleWithStages(BaseModel):
    id: str
    name: str
    stages: List[StageLite] = Field(default_factory=list)

class DepartmentWithRolesStages(BaseModel):
    id: str
    name: str
    roles: List[RoleWithStages] = Field(default_factory=list)

class DepartmentsRolesStagesResponse(BaseModel):
    message: str
    updatedAt: str  # ISO 8601 string
    scope: Dict[str, Any]
    departments: List[DepartmentWithRolesStages]
    

@app.get("/candidates/summary", response_model=DepartmentsRolesStagesResponse)
def candidates_summary_raw(
    request: Request,
    userEmail: Optional[str] = Query(None, description="Impersonate this Workspace user"),
):
    """
    Returns Departments → Roles → Stages (without inline text extraction).
    Only file metadata is included; text must be fetched separately via /candidates/fileText.
    """
    require_api_key(request)
    subject = userEmail or _extract_subject_from_request(request)
    _, drive, _ = get_clients(subject)

    DEPARTMENTS_FOLDER_ID = os.environ.get("DEPARTMENTS_FOLDER_ID")
    if not DEPARTMENTS_FOLDER_ID:
        raise HTTPException(500, "DEPARTMENTS_FOLDER_ID env var not set")

    departments_out: List[DepartmentWithRolesStages] = []

    for dept in _iter_child_folders(drive, DEPARTMENTS_FOLDER_ID):
        dept_id = dept.get("id")
        dept_name = dept.get("name") or "(unnamed)"
        roles_out: List[RoleWithStages] = []

        for role in _iter_child_folders(drive, dept_id):
            role_id = role.get("id")
            role_name = role.get("name") or "(unnamed)"
            stages_out: List[StageLite] = []

            pipeline = _find_child_folder_by_name(drive, role_id, "Hiring Pipeline")
            if pipeline:
                for stage in _iter_child_folders(drive, pipeline["id"]):
                    stage_id = stage.get("id")
                    stage_name = stage.get("name") or "(unnamed)"

                    stage_files: List[StageFileExtract] = []
                    try:
                        for f in _iter_child_files(drive, stage_id):
                            stage_files.append(StageFileExtract(
                                id=f.get("id"),
                                name=f.get("name", ""),
                                mimeType=f.get("mimeType", ""),
                                text=None,   # ✅ no inline extraction
                                error=None
                            ))
                    except Exception as e:
                        logger.exception("Failed listing files for stage %s (%s)", stage_name, stage_id)
                        stage_files.append(StageFileExtract(
                            id="",
                            name="(stage scan error)",
                            mimeType="",
                            text=None,
                            error=f"Failed scanning stage: {e}"
                        ))

                    stages_out.append(StageLite(
                        id=stage_id,
                        name=stage_name,
                        files=stage_files
                    ))

            roles_out.append(RoleWithStages(
                id=role_id,
                name=role_name,
                stages=stages_out
            ))

        departments_out.append(DepartmentWithRolesStages(
            id=dept_id,
            name=dept_name,
            roles=roles_out
        ))

    return DepartmentsRolesStagesResponse(
        message="Departments, roles, and stages collected successfully (metadata only)",
        updatedAt=datetime.now(timezone.utc).isoformat(),
        scope={
            "departmentsFolderId": DEPARTMENTS_FOLDER_ID,
            "impersonating": subject or None,
        },
        departments=departments_out,
    )

@app.get("/candidates/fileText", response_model=StageFileExtract)
def get_file_text(
    request: Request,
    fileId: str = Query(..., description="Google Drive file ID"),
    userEmail: Optional[str] = Query(None, description="Impersonate this Workspace user"),
):
    """
    Extract plain text from a specific candidate file (Google Doc, Docx, or PDF).
    """
    require_api_key(request)
    subject = userEmail or _extract_subject_from_request(request)
    _, drive, docs = get_clients(subject)

    try:
        f = drive.files().get(fileId=fileId, fields="id,name,mimeType").execute()
    except Exception as e:
        raise HTTPException(404, f"File '{fileId}' not found: {e}")

    fname = f.get("name", "")
    mime = f.get("mimeType", "")

    text, err = (None, None)
    if _is_doc_or_pdf(fname, mime):
        text, err = _extract_text_from_file(drive, docs, f)

    return StageFileExtract(
        id=f["id"],
        name=fname,
        mimeType=mime,
        text=text,
        error=err
    )

# =========================
# Pydantic models for Move API
# =========================

class MoveGroup(BaseModel):
    candidateQueries: List[str] = Field(..., description="Names/partials/typos allowed")
    stageQuery: str = Field(..., description="Target stage name (fuzzy)")

class MoveByPromptRequest(BaseModel):
    positionId: str = Field(..., description="Role folder ID")
    prompt: str = Field(..., description="Free-text instruction like 'move alice and bob to onsite'")
    dryRun: bool = False
    userEmail: Optional[str] = None

class MoveStructuredRequest(BaseModel):
    positionId: str = Field(..., description="Role folder ID")
    moves: List[MoveGroup]
    dryRun: bool = False
    userEmail: Optional[str] = None

class MoveDecision(BaseModel):
    candidateQuery: str
    candidateMatchedName: Optional[str] = None
    candidateFileId: Optional[str] = None
    candidateScore: int
    fromStageId: Optional[str] = None
    fromStageName: Optional[str] = None
    toStageQuery: str
    toStageId: Optional[str] = None
    toStageName: Optional[str] = None
    stageScore: int
    moved: bool
    error: Optional[str] = None

class MoveResponse(BaseModel):
    message: str
    positionId: str
    dryRun: bool
    thresholds: Dict[str, int]
    decisions: List[MoveDecision]

# =========================
# POST /candidates/moveByPrompt
# =========================
@app.post("/candidates/moveByPrompt", response_model=MoveResponse)
def move_candidates_by_prompt(request: Request, body: MoveByPromptRequest):
    require_api_key(request)
    subject = body.userEmail or _extract_subject_from_request(request)
    _, drive, _ = get_clients(subject)

    # Build stage + file indexes for the role (position)
    stages, file_index = _build_candidate_index(drive, body.positionId)
    if not stages:
        raise HTTPException(404, "Hiring Pipeline / stages not found under the specified role")

    groups = _parse_move_prompt(body.prompt)
    if not groups:
        raise HTTPException(400, "Could not parse any 'candidates → stage' group from the prompt")

    decisions: List[MoveDecision] = []

    for g in groups:
        # Resolve stage once per group
        stage_score, stage_match = _resolve_best_stage(g["stageQuery"], stages)
        for cand_q in g["candidateQueries"]:
            cand_score, cand_match = _resolve_best_candidate_file(cand_q, file_index)
            decision = MoveDecision(
                candidateQuery=cand_q,
                candidateMatchedName=cand_match["name"] if cand_match else None,
                candidateFileId=cand_match["id"] if cand_match else None,
                candidateScore=cand_score,
                fromStageId=cand_match["stageId"] if cand_match else None,
                fromStageName=cand_match["stageName"] if cand_match else None,
                toStageQuery=g["stageQuery"],
                toStageId=stage_match["id"] if stage_match else None,
                toStageName=stage_match["name"] if stage_match else None,
                stageScore=stage_score,
                moved=False,
                error=None
            )

            # Validation / thresholds
            if not cand_match:
                decision.error = "No candidate file matched"
            elif cand_score < _NAME_SCORE_THRESHOLD:
                decision.error = f"Low candidate match score ({cand_score}<{_NAME_SCORE_THRESHOLD})"
            elif not stage_match:
                decision.error = "No target stage matched"
            elif stage_score < _STAGE_SCORE_THRESHOLD:
                decision.error = f"Low stage match score ({stage_score}<{_STAGE_SCORE_THRESHOLD})"
            elif cand_match["stageId"] == stage_match["id"]:
                decision.error = "Already in target stage"
            else:
                if not body.dryRun:
                    try:
                        _move_file_between_stages(drive, cand_match["id"], cand_match["stageId"], stage_match["id"])
                        decision.moved = True
                        # Update local index so subsequent groups see the new stage
                        file_index[cand_match["id"]]["stageId"] = stage_match["id"]
                        file_index[cand_match["id"]]["stageName"] = stage_match["name"]
                    except Exception as e:
                        decision.error = f"Move failed: {e}"
                else:
                    decision.moved = False  # dry run

            decisions.append(decision)

    return MoveResponse(
        message="Processed moveByPrompt",
        positionId=body.positionId,
        dryRun=body.dryRun,
        thresholds={"name": _NAME_SCORE_THRESHOLD, "stage": _STAGE_SCORE_THRESHOLD},
        decisions=decisions
    )

# =========================
# POST /candidates/move (structured)
# =========================
@app.post("/candidates/move", response_model=MoveResponse)
def move_candidates_structured(request: Request, body: MoveStructuredRequest):
    require_api_key(request)
    subject = body.userEmail or _extract_subject_from_request(request)
    _, drive, _ = get_clients(subject)

    stages, file_index = _build_candidate_index(drive, body.positionId)
    if not stages:
        raise HTTPException(404, "Hiring Pipeline / stages not found under the specified role")

    decisions: List[MoveDecision] = []

    for grp in body.moves:
        stage_score, stage_match = _resolve_best_stage(grp.stageQuery, stages)
        for cand_q in grp.candidateQueries:
            cand_score, cand_match = _resolve_best_candidate_file(cand_q, file_index)
            decision = MoveDecision(
                candidateQuery=cand_q,
                candidateMatchedName=cand_match["name"] if cand_match else None,
                candidateFileId=cand_match["id"] if cand_match else None,
                candidateScore=cand_score,
                fromStageId=cand_match["stageId"] if cand_match else None,
                fromStageName=cand_match["stageName"] if cand_match else None,
                toStageQuery=grp.stageQuery,
                toStageId=stage_match["id"] if stage_match else None,
                toStageName=stage_match["name"] if stage_match else None,
                stageScore=stage_score,
                moved=False,
                error=None
            )

            if not cand_match:
                decision.error = "No candidate file matched"
            elif cand_score < _NAME_SCORE_THRESHOLD:
                decision.error = f"Low candidate match score ({cand_score}<{_NAME_SCORE_THRESHOLD})"
            elif not stage_match:
                decision.error = "No target stage matched"
            elif stage_score < _STAGE_SCORE_THRESHOLD:
                decision.error = f"Low stage match score ({stage_score}<{_STAGE_SCORE_THRESHOLD})"
            elif cand_match["stageId"] == stage_match["id"]:
                decision.error = "Already in target stage"
            else:
                if not body.dryRun:
                    try:
                        _move_file_between_stages(drive, cand_match["id"], cand_match["stageId"], stage_match["id"])
                        decision.moved = True
                        file_index[cand_match["id"]]["stageId"] = stage_match["id"]
                        file_index[cand_match["id"]]["stageName"] = stage_match["name"]
                    except Exception as e:
                        decision.error = f"Move failed: {e}"
                else:
                    decision.moved = False

            decisions.append(decision)

    return MoveResponse(
        message="Processed structured move",
        positionId=body.positionId,
        dryRun=body.dryRun,
        thresholds={"name": _NAME_SCORE_THRESHOLD, "stage": _STAGE_SCORE_THRESHOLD},
        decisions=decisions
    )




@app.get("/whoami") # Verify who the api is acting as when user impersonation
def whoami(request: Request):
    require_api_key(request)
    subject = _extract_subject_from_request(request)
    _, drive, _ = get_clients(subject)
    about = drive.about().get(fields="user(emailAddress,displayName),storageQuota").execute()
    return {"subject_param": subject, "drive_user": about.get("user")}
