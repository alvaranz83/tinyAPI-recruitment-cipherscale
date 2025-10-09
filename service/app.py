from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Iterable, Tuple, Union
import os, json, textwrap, re, uuid, logging, io, httpx, time, random
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload 
from google.auth.exceptions import RefreshError # For user impersonation
from pydantic import BaseModel, Field
from difflib import SequenceMatcher
from openai import AsyncOpenAI
from googleapiclient.errors import HttpError
from google.oauth2 import service_account
from sqlalchemy import create_engine, MetaData, insert
from databases import Database
from db import database
from uuid import UUID

#

app = FastAPI(title="Recruiting Sheet Insights")

@app.on_event("startup")
async def startup():
    await database.connect()

@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend.example.com", "http://localhost:3000"],  # or ["*"] while testing
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # or ["*"]
    allow_headers=["*"],  # or explicitly ["x-api-key", "x-user-email", "content-type", "authorization"]
)

## DB Related - Dev DB
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_NAME = os.getenv("DB_NAME")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

# Path to your service account key
SERVICE_ACCOUNT_FILE = os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"]
SCOPES = ["https://www.googleapis.com/auth/drive"]

# Initialize OpenAI once at top-level
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
#end

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_API_URL = "https://slack.com/api/chat.postMessage"

# ==========================
# Fuzzy & Parse Helpers..
# =========================

_STATUS_RE = re.compile(r"\b(open|opened|opening|close|closed|closing)\b", re.IGNORECASE)
_NAME_SCORE_THRESHOLD = int(os.environ.get("NAME_SCORE_THRESHOLD", "40"))
_STAGE_SCORE_THRESHOLD = int(os.environ.get("STAGE_SCORE_THRESHOLD", "40"))
_ROLE_SCORE_THRESHOLD = int(os.environ.get("ROLE_SCORE_THRESHOLD", "20"))
_FOR_SPLIT_RE = re.compile(r"\bfor\b", re.IGNORECASE)

_AND_SPLIT_RE = re.compile(r"\s*(?:,| and )\s*", re.IGNORECASE)
_TO_CLAUSE_RE = re.compile(r"\bto\b", re.IGNORECASE)

## Helper to use OpenAI API to talk to agent from external services
async def process_with_gpt(prompt: str) -> str:
    """
    Call the custom Cipherscale HR/Recruitment Ops Agent via Assistants API.
    """
    try:
        # 1. Create a new thread
        thread = await openai_client.beta.threads.create()

        # 2. Add the user message
        await openai_client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=prompt
        )

        # 3. Run the assistant. Custom GPT ID
        run = await openai_client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=os.environ["ASSISTANT_ID"]  # put your assistant_id in env
        )

        # 4. Poll until run completes
        while True:
            run_status = await openai_client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id
            )
            if run_status.status in ["completed", "failed", "cancelled"]:
                break
            await asyncio.sleep(1)

        # 5. Fetch the last message
        messages = await openai_client.beta.threads.messages.list(thread_id=thread.id)
        last = messages.data[0]
        return last.content[0].text.value if last.content else "No response."

    except Exception as e:
        return f"❌ Error calling custom GPT Agent: {str(e)}"


## end

## For importing candidates into database
def _split_name(full_name: str):
    parts = full_name.strip().split()
    first = parts[0] if parts else ""
    last = " ".join(parts[1:]) if len(parts) > 1 else ""
    return first, last
## end


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

def _parse_status_prompt(prompt: str, drive, departments_root_id: str) -> list[dict]:
    prompt = prompt.strip()
    out: list[dict] = []

    # Handle "all roles" case
    if re.search(r"\ball\b.*\broles?\b", prompt, re.IGNORECASE):
        status_match = _STATUS_RE.search(prompt)
        if status_match:
            status = status_match.group(1)
            all_roles = _list_all_roles(drive, departments_root_id)
            role_names = [r["name"] for r in all_roles]
            out.append({"roleQueries": role_names, "statusQuery": status})
        return out

    # Normal case: split by status keywords
    parts = _STATUS_RE.split(prompt)
    if len(parts) < 2:
        return []

    it = iter(parts)
    _ = next(it)  # skip prefix
    for status, roles_chunk in zip(it, it):
        role_text = roles_chunk.strip(" ,.")
        if not role_text:
            continue
        role_list = [r for r in _AND_SPLIT_RE.split(role_text) if r]
        out.append({
            "roleQueries": role_list,
            "statusQuery": status
        })

    return out


def _normalize_status(s: str) -> Optional[str]:
    s = (s or "").strip().lower()
    if s in ["open", "opened", "opening"]:
        return "open"
    if s in ["close", "closed", "closing"]:
        return "close"
    return None


def _best_match(query: str, candidates: Iterable[Tuple[str, Any]]) -> Tuple[int, Any | None, str | None]:
    """
    candidates: iterable of (display_name, payload)
    Returns (score, payload_or_None, matched_display_or_None)
    """
    best_score = 0
    best_payload = None
    best_display = None
    for display, payload in candidates:
        sc = _token_set_ratio(query, display)
        if sc > best_score:
            best_score = sc
            best_payload = payload
            best_display = display
    return best_score, best_payload, best_display

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

def _resolve_best_stage(stage_query: str, stages: list[dict]) -> tuple[int, dict | None, str | None]:
    # stages elements are dicts with "name", "id"
    return _best_match(stage_query, [(s["name"], s) for s in stages])

def _resolve_best_candidate_file(cand_query: str, file_index: dict) -> tuple[int, dict | None, str | None]:
    # file_index values are dicts with "name", "id", "stageId", "stageName"
    return _best_match(cand_query, [(meta["name"], meta) for meta in file_index.values()])

def _move_file_between_stages(drive, file_id: str, from_stage_id: str, to_stage_id: str) -> dict:
    return drive.files().update(
        fileId=file_id,
        addParents=to_stage_id,
        removeParents=from_stage_id,
        fields="id, parents",
        supportsAllDrives=True
    ).execute()

def _strip_ext(name: str) -> str:
    return re.sub(r"\.[A-Za-z0-9]+$", "", name or "")

def _resolve_best_role_by_name(drive, departments_root_id: str, role_query: str) -> tuple[int, dict | None, str | None]:
    """
    Walk all department folders under DEPARTMENTS_FOLDER_ID, gather role folders,
    fuzzy match by name. Returns (score, {'id','name','deptId','deptName'}|None, display_name).
    """
    candidates = []
    for dept in _iter_child_folders(drive, departments_root_id):
        dept_id = dept.get("id")
        dept_name = dept.get("name", "")
        for role in _iter_child_folders(drive, dept_id):
            r = {"id": role["id"], "name": role.get("name", ""), "deptId": dept_id, "deptName": dept_name}
            candidates.append((f"{role.get('name','')} ({dept_name})", r))
    return _best_match(role_query, candidates)

def _list_all_roles(drive, departments_root_id: str) -> list[dict]:
    roles = []
    for dept in _iter_child_folders(drive, departments_root_id):
        dept_id = dept.get("id")
        dept_name = dept.get("name", "")
        for role in _iter_child_folders(drive, dept_id):
            roles.append({"id": role["id"], "name": role.get("name",""), "deptId": dept_id, "deptName": dept_name})
    return roles

def _parse_upload_prompt(prompt: str) -> list[dict]:
    """
    Parse strings like:
      "upload alice and bob to onsite for Senior Backend Engineer,
       and charlie to offer accepted for Sales AE"
    ->
      [
        { "candidateQueries": ["alice", "bob"], "stageQuery": "onsite", "roleQuery": "Senior Backend Engineer" },
        { "candidateQueries": ["charlie"], "stageQuery": "offer accepted", "roleQuery": "Sales AE" }
      ]

    Also tolerates prompt variants like:
      "upload vadim to technical interview"
      (roleQuery omitted)
    """
    # Reuse move-style parsing to get candidate + "stage for role" blobs
    groups_basic = _parse_move_prompt(prompt)
    out = []
    for g in groups_basic:
        stage = g.get("stageQuery","")
        role_query = None
        # Split "... to STAGE for ROLE"
        parts = _FOR_SPLIT_RE.split(stage)
        if len(parts) >= 2:
            stage_query = parts[0].strip(" ,.")
            role_query  = parts[1].strip(" ,.")
        else:
            stage_query = stage.strip(" ,.")
        out.append({
            "candidateQueries": g.get("candidateQueries", []),
            "stageQuery": stage_query,
            "roleQuery": role_query
        })
    return out

def _assign_files_to_candidates(files: list[UploadFile], candidate_names: list[str]) -> dict[str, UploadFile | None]:
    """
    Greedy assignment: match files to candidate names by filename similarity.
    Returns map: candidate_display_name -> UploadFile (or None if not found).
    """
    remaining = list(files)
    assignments = {}
    for cand in candidate_names:
        best = None
        best_score = -1
        for uf in remaining:
            fname = _strip_ext(uf.filename)
            sc = _token_set_ratio(cand, fname)
            if sc > best_score:
                best_score = sc
                best = uf
        assignments[cand] = best
        if best in remaining:
            remaining.remove(best)
    return assignments

def _upload_as_google_doc(drive, docs, parent_id: str, doc_name: str, file: UploadFile) -> str:
    """
    Convert uploaded file directly to a Google Doc inside parent_id.
    No base64. Uses multipart media upload.
    """
    # Fallback content-type if missing
    ctype = file.content_type or "application/octet-stream"
    buf = io.BytesIO(file.file.read())
    media = MediaIoBaseUpload(buf, mimetype=ctype, resumable=False)

    # Let Drive convert to Google Docs by setting metadata mimeType to Docs
    meta = {
        "name": doc_name,
        "mimeType": "application/vnd.google-apps.document",
        "parents": [parent_id],
    }
    created = drive.files().create(
        body=meta,
        media_body=media,
        fields="id",
        supportsAllDrives=True
    ).execute()
    return created["id"]
    

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



def create_google_doc(docs, drive, folder_id: str, title: str, content: str, raw_mode: bool = False) -> str:
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
    if requests:
        docs.documents().batchUpdate(documentId=doc_id, body={"requests": requests}).execute()

    # --- ✅ NEW: raw_mode branch ---
    if raw_mode:
        requests = [{
            "insertText": {
                "location": {"index": 1},
                "text": content
            }
        }]
        docs.documents().batchUpdate(documentId=doc_id, body={"requests": requests}).execute()
        return doc_id

    
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

def _fetch_all_drive_items(drive):
    """
    Fetch ALL items in Drive (not just scoped to Departments).
    This ensures we capture the entire folder tree: Departments → Roles → Hiring Pipeline → Stages → Candidate files.
    """
    query = "trashed=false"
    results = []
    page_token = None

    while True:
        response = drive.files().list(
            q=query,
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
            fields="nextPageToken, files(id, name, mimeType, parents)",
            pageToken=page_token
        ).execute()

        results.extend(response.get("files", []))
        page_token = response.get("nextPageToken")
        if page_token is None:
            break

    return results

def _get_drive_client(userEmail: Optional[str] = None):
    """
    Returns a Drive API client.
    If userEmail is provided, impersonates that user (requires domain-wide delegation).
    """
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES
    )

    if userEmail:
        creds = creds.with_subject(userEmail)

    return build("drive", "v3", credentials=creds)

#End of Helper Functions



class PositionRequest(BaseModel):
    name: str
    department: str = "Software Engineering"
    dryRun: bool = False
    userEmail: Optional[str] = None  # <-- add this

@app.post("/positions/create")
async def create_position(request: Request, body: PositionRequest):
    require_api_key(request)
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
    results = drive.files().list(
        q=query, fields="files(id,name)",
        includeItemsFromAllDrives=True, supportsAllDrives=True
    ).execute()
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
    results = drive.files().list(
        q=query, fields="files(id,name)",
        includeItemsFromAllDrives=True, supportsAllDrives=True
    ).execute()

    created = False
    if results.get("files"):
        position_id = results["files"][0]["id"]
    else:
        # Step 2: Create role folder
        position_id = create_folder(drive, name, department_folder_id)
        created = True

        # Step 3: Ensure default subfolders exist
        default_subfolders = [
            "Job Description",
            "CVs Assessment",
            "TA/HR Interview Template",
            "TA/HR Interviews (Assessments)",
            "1st Technical Interview Template",
            "1st Technical Interviews (Assessments)",
            "2nd Technical Interview Template",
            "2nd Technical Interviews (Assessments)",
            "Hiring Pipeline"
        ]

        for sub in default_subfolders:
            existing = _find_child_folder_by_name(drive, position_id, sub)
            if not existing:
                create_named_subfolder(drive, position_id, sub)

    # ✅ Step 4: Insert into DB always
    try:
        dept_uuid = await database.fetch_val(
            "SELECT id FROM departments WHERE drive_id = :drive_id",
            {"drive_id": department_folder_id}
        )
        if not dept_uuid:
            raise HTTPException(404, f"Department '{department}' not found in DB")

        query = """
            INSERT INTO roles (
                drive_id, role_name, department_id, department_name,
                created_by, created_at, status, job_description_url
            )
            VALUES (
                :drive_id, :role_name, :department_id, :department_name,
                :created_by, :created_at, :status, :job_description_url
            )
            ON CONFLICT (drive_id) DO NOTHING
            RETURNING id
        """
        values = {
            "drive_id": position_id,
            "role_name": name,
            "department_id": dept_uuid,
            "department_name": department,
            "created_by": subject or "system",
            "created_at": datetime.now(timezone.utc),
            "status": "open",
            "job_description_url": None  # JD created later
        }

        role_id = await database.execute(query=query, values=values)

        if not role_id:
            # If already exists, fetch its ID
            role_id = await database.fetch_val(
                "SELECT id FROM roles WHERE drive_id = :drive_id",
                {"drive_id": position_id}
            )

    except Exception as e:
        logger.error(f"❌ Failed to insert role into DB: {e}")
        role_id = None

    return {
        "message": f"Role '{name}' {'created' if created else 'ensured'} successfully in {department}",
        "positionId": position_id,
        "departmentFolderId": department_folder_id,
        "created": created
    }



@app.get("/positions/list")
def list_positions(request: Request, department: Optional[str] = None):
    require_api_key(request)
    subject = _extract_subject_from_request(request)
    _, drive, _ = get_clients(subject)

    DEPARTMENTS_FOLDER_ID = os.environ.get("DEPARTMENTS_FOLDER_ID")
    if not DEPARTMENTS_FOLDER_ID:
        raise HTTPException(500, "DEPARTMENTS_FOLDER_ID env var not set")

    FOLDER_MIME = "application/vnd.google-apps.folder"

    if department:
        # If department specified, check inside it
        query = (
            f"mimeType='{FOLDER_MIME}' "
            f"and trashed=false and name='{department}' "
            f"and '{DEPARTMENTS_FOLDER_ID}' in parents"
        )
        results = drive.files().list(
            q=query, fields="files(id,name)",
            includeItemsFromAllDrives=True, supportsAllDrives=True
        ).execute()
        items = results.get("files", [])
        if not items:
            return {"roles": [], "department": department, "exists": False}
        parent_id = items[0]["id"]

        # List roles under department (request properties too)
        query = f"mimeType='{FOLDER_MIME}' and trashed=false and '{parent_id}' in parents"
        results = drive.files().list(
            q=query, fields="files(id,name,properties)",
            includeItemsFromAllDrives=True, supportsAllDrives=True
        ).execute()
        roles = [
            {
                "id": f["id"],
                "name": f["name"],
                "roleStatus": f.get("properties", {}).get("roleStatus")
            }
            for f in results.get("files", [])
        ]

        return {
            "department": department,
            "roles": roles,
            "exists": True
        }

    else:
        # No department specified → list all departments AND their roles
        query = f"mimeType='{FOLDER_MIME}' and trashed=false and '{DEPARTMENTS_FOLDER_ID}' in parents"
        results = drive.files().list(
            q=query, fields="files(id,name)",
            includeItemsFromAllDrives=True, supportsAllDrives=True
        ).execute()
        departments = results.get("files", [])

        out = []
        for dept in departments:
            dept_id = dept["id"]
            dept_name = dept["name"]

            # Find roles under each department (request properties too)
            query = f"mimeType='{FOLDER_MIME}' and trashed=false and '{dept_id}' in parents"
            roles_results = drive.files().list(
                q=query, fields="files(id,name,properties)",
                includeItemsFromAllDrives=True, supportsAllDrives=True
            ).execute()
            roles = [
                {
                    "id": f["id"],
                    "name": f["name"],
                    "roleStatus": f.get("properties", {}).get("roleStatus")
                }
                for f in roles_results.get("files", [])
            ]

            out.append({
                "department": dept_name,
                "roles": roles
            })

        return {
            "message": "Departments and their roles listed successfully",
            "departments": out,
            "exists": True
        }


class CreateJDRequest(BaseModel):
    positionId: str           # Google Drive folder ID of the role
    roleName: str
    content: Optional[str] = None
    userEmail: Optional[str] = None  # for impersonation


@app.post("/positions/createJD")
async def create_jd(request: Request, body: CreateJDRequest):
    require_api_key(request)
    subject = body.userEmail or _extract_subject_from_request(request)
    _, drive, docs = get_clients(subject)

    # ✅ Ensure Job Description folder exists
    jd_folder = _find_child_folder_by_name(drive, body.positionId, "Job Description")
    if jd_folder:
        jd_folder_id = jd_folder["id"]
    else:
        jd_folder_id = create_named_subfolder(drive, body.positionId, "Job Description")

    # ✅ Ensure CVs Assessment folder exists (future usage)
    cv_assessment_folder = _find_child_folder_by_name(drive, body.positionId, "CVs Assessment")
    if cv_assessment_folder:
        cv_assessment_folder_id = cv_assessment_folder["id"]
    else:
        cv_assessment_folder_id = create_named_subfolder(drive, body.positionId, "CVs Assessment")

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

    # ✅ Create Google Doc
    file_id = create_google_doc(docs, drive, jd_folder_id, f"JD - {body.roleName}", content)
    doc_link = f"https://docs.google.com/document/d/{file_id}/edit"

    try:
        # ✅ Fetch the Role UUID from DB
        role_uuid = await database.fetch_val(
            "SELECT id FROM roles WHERE drive_id = :drive_id",
            {"drive_id": body.positionId}
        )
        if not role_uuid:
            raise HTTPException(404, f"No role found in DB with drive_id={body.positionId}")

        # ✅ Insert into job_descriptions table
        jd_id = await database.execute(
            """
            INSERT INTO job_descriptions (
                template_name, template_url, drive_id,
                created_by, created_at, role_name, role
            )
            VALUES (:template_name, :template_url, :drive_id,
                    :created_by, :created_at, :role_name, :role)
            RETURNING id
            """,
            {
                "template_name": f"JD - {body.roleName}",
                "template_url": doc_link,
                "drive_id": file_id,              # Google Drive File ID (TEXT)
                "created_by": subject or "system",
                "created_at": datetime.now(timezone.utc),
                "role_name": body.roleName,
                "role": role_uuid                 # UUID from roles table ✅
            }
        )

        # ✅ Update roles table with JD link
        await database.execute(
            """
            UPDATE roles
            SET job_description_url = :doc_link
            WHERE id = :role_id
            """,
            {"doc_link": doc_link, "role_id": role_uuid}
        )

    except Exception as e:
        logger.error(f"❌ Failed to persist Job Description or update Role: {e}")
        raise HTTPException(500, f"Failed to persist JD or update role: {e}")

    return {
        "message": f"JD created for {body.roleName}",
        "fileId": file_id,
        "folderId": jd_folder_id,
        "docLink": doc_link,
        "jobDescriptionId": jd_id
    }




class CreateScreeningRequest(BaseModel):
    positionId: str
    roleName: str
    content: Optional[str] = None
    userEmail: Optional[str] = None  # for impersonation

@app.post("/positions/createScreeningTemplate")
async def create_screening(request: Request, body: CreateScreeningRequest):
    require_api_key(request)
    subject = body.userEmail or _extract_subject_from_request(request)
    _, drive, docs = get_clients(subject)

    # ✅ Ensure TA/HR Interview Template folder exists
    screening_folder = _find_child_folder_by_name(drive, body.positionId, "TA/HR Interview Template")
    if screening_folder:
        screening_folder_id = screening_folder["id"]
    else:
        screening_folder_id = create_named_subfolder(drive, body.positionId, "TA/HR Interview Template")

    # ✅ Ensure TA/HR Interviews (Assessments) folder exists
    assessment_folder = _find_child_folder_by_name(drive, body.positionId, "TA/HR Interviews (Assessments)")
    if assessment_folder:
        assessment_folder_id = assessment_folder["id"]
    else:
        assessment_folder_id = create_named_subfolder(drive, body.positionId, "TA/HR Interviews (Assessments)")

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

    # ✅ Create Google Doc
    file_id = create_google_doc(
        docs, drive, screening_folder_id,
        f"Screening Template - {body.roleName}", content
    )
    doc_link = f"https://docs.google.com/document/d/{file_id}/edit"

    # ✅ Persist into DB with conflict handling
    try:
        await database.execute(
            """
            INSERT INTO ta_hr_interview_templates (drive_id, template_name, created_by, template_url, created_at)
            VALUES (:drive_id, :template_name, :created_by, :template_url, :created_at)
            ON CONFLICT (drive_id) DO NOTHING
            """,
            {
                "drive_id": file_id,
                "template_name": f"Screening Template - {body.roleName}",
                "created_by": subject or "system",
                "template_url": doc_link,
                "created_at": datetime.now(timezone.utc)
            }
        )
         # 🔥 New part: also update the corresponding role
        await database.execute(
            """
            UPDATE roles
            SET ta_hr_interview_template_url = :template_url
            WHERE drive_id = :position_id
            """,
            {
                "template_url": doc_link,
                "position_id": body.positionId
            }
        )


    except Exception as e:
        logger.error(f"❌ Failed to persist TA/HR Interview Template: {e}")

    return {
        "message": f"Screening template created for {body.roleName}",
        "fileId": file_id,
        "folderId": screening_folder_id,
        "docLink": doc_link
    }


class CreateFirstTechInterviewRequest(BaseModel):
    positionId: str          # Google Drive folder ID of the role
    candidateName: str
    content: Optional[str] = None
    userEmail: Optional[str] = None  # for impersonation


@app.post("/candidates/createFirstTechnicalInterview")
async def create_first_tech_interview(request: Request, body: CreateFirstTechInterviewRequest):
    require_api_key(request)
    subject = body.userEmail or _extract_subject_from_request(request)
    _, drive, docs = get_clients(subject)

    # ✅ Ensure 1st Technical Interview Template folder exists
    screening_folder = _find_child_folder_by_name(drive, body.positionId, "1st Technical Interview Template")
    if screening_folder:
        screening_folder_id = screening_folder["id"]
    else:
        screening_folder_id = create_named_subfolder(drive, body.positionId, "1st Technical Interview Template")

    # ✅ Ensure 1st Technical Interviews (Assessments) folder exists
    tech_assessment_folder = _find_child_folder_by_name(drive, body.positionId, "1st Technical Interviews (Assessments)")
    if tech_assessment_folder:
        tech_assessment_folder_id = tech_assessment_folder["id"]
    else:
        tech_assessment_folder_id = create_named_subfolder(drive, body.positionId, "1st Technical Interviews (Assessments)")

    # ✅ Check if the template already exists
    query = (
        "mimeType='application/vnd.google-apps.document' "
        "and trashed=false "
        f"and name='{body.candidateName} - 1st Technical Interview' "
        f"and '{screening_folder_id}' in parents"
    )
    results = drive.files().list(
        q=query,
        fields="files(id,name)",
        includeItemsFromAllDrives=True,
        supportsAllDrives=True
    ).execute()

    created = False
    if results.get("files"):
        existing = results["files"][0]
        file_id = existing["id"]
        file_name = existing["name"]
    else:
        # ✅ Default polished template
        content = body.content or f"""
            1st Technical Interview – {body.candidateName}

            Candidate: {body.candidateName}
            Role: [Specify Role Here]
            Date: ___________________________
            Interviewer(s): ___________________________

            Technical Questions:
            - Q1: ______________________________________
            - Q2: ______________________________________
            - Q3: ______________________________________
            - Q4: ______________________________________
            - Q5: ______________________________________

            Feedback:
            - Strengths:
            ____________________________________________

            - Weaknesses:
            ____________________________________________

            Scorecard (1–5 for each dimension):
            - Technical Knowledge: ___
            - Problem Solving: ___
            - Communication: ___
            - Culture Fit: ___

            Overall Recommendation:
            - Strong Hire / Hire / Neutral / No Hire
        """

        file_name = f"{body.candidateName} - 1st Technical Interview"
        file_id = create_google_doc(docs, drive, screening_folder_id, file_name, content)
        created = True

    doc_link = f"https://docs.google.com/document/d/{file_id}/edit"

    # ✅ Persist template record into DB (in first_tech_interview_templates)
    try:
        await database.execute(
            """
            INSERT INTO first_tech_interview_templates 
                (template_name, created_by, created_at, template_url, drive_id)
            VALUES (:template_name, :created_by, :created_at, :template_url, :drive_id)
            ON CONFLICT (drive_id) DO NOTHING
            """,
            {
                "template_name": file_name,
                "created_by": subject or "system",   # ✅ fixed column name
                "created_at": datetime.now(timezone.utc),
                "template_url": doc_link,
                "drive_id": file_id
            }
        )
    
        # ✅ Resolve role UUID
        role_uuid = await database.fetch_val(
            "SELECT id FROM roles WHERE drive_id = :drive_id",
            {"drive_id": body.positionId}
        )
        if not role_uuid:
            raise HTTPException(404, f"No role found in DB with drive_id={body.positionId}")
        
        # ✅ Update roles table with first technical interview template URL
        await database.execute(
            """
            UPDATE roles
            SET first_tech_interview_template_url = :template_url
            WHERE id = :id
            """,
            {"template_url": doc_link, "id": role_uuid}
        )


    except Exception as e:
        logger.error(f"❌ Failed to persist 1st Technical Interview Template or update role: {e}")
        raise HTTPException(500, f"DB insert/update failed: {e}")


    return {
        "message": f"1st Technical Interview template {'created' if created else 'already existed'} for {body.candidateName}",
        "fileId": file_id,
        "folderId": screening_folder_id,
        "docLink": doc_link,
        "created": created
    }



class CreateDepartmentsRequest(BaseModel):
    names: List[str]
    userEmail: Optional[str] = None  # impersonation


@app.post("/departments/create")
async def create_departments(request: Request, body: CreateDepartmentsRequest):
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
        for f in results["files"]:
            if HIRING_FOLDER_ID in f.get("parents", []):
                departments_folder_id = f["id"]
                break

    if not departments_folder_id:
        departments_folder_id = create_folder(drive, "Departments", HIRING_FOLDER_ID)
        created_root = True

    created_departments = []
    for dept in body.names:
        # Check if department folder already exists
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
            dept_id = existing["files"][0]["id"]
            dept_created = False
        else:
            dept_id = create_folder(drive, dept, departments_folder_id)
            dept_created = True

        # ✅ Insert or update department in DB with drive_id, created_by, created_at
        query = """
            INSERT INTO departments (department_name, drive_id, created_by, created_at)
            VALUES (:department_name, :drive_id, :created_by, :created_at)
            ON CONFLICT (department_name) DO NOTHING
            RETURNING id
        """
        values = {
            "department_name": dept,
            "drive_id": dept_id,
            "created_by": subject or "system",
            "created_at": datetime.now(timezone.utc)
        }
        dept_db_id = await database.execute(query=query, values=values)

        # if already exists, fetch its id
        if not dept_db_id:
            dept_db_id = await database.fetch_val(
                "SELECT id FROM departments WHERE department_name = :department_name",
                {"department_name": dept}
            )

        created_departments.append({
            "name": dept,
            "id": dept_id,
            "created": dept_created,
            "db_id": dept_db_id
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

from sqlalchemy import insert

@app.post("/HiringFlows/create")
async def create_hiring_flows(request: Request, body: CreateHiringFlowsRequest):
    require_api_key(request)
    subject = body.userEmail or _extract_subject_from_request(request)
    _, drive, _ = get_clients(subject)

    HIRING_FOLDER_ID = os.environ.get("HIRING_FOLDER_ID")
    if not HIRING_FOLDER_ID:
        raise HTTPException(500, "HIRING_FOLDER_ID env var not set")

    # 1. Ensure "Hiring Flows" folder exists in Drive
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
        # 2. Check (or create) flow subfolder in Drive
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

        # ✅ Insert flow into DB (store stages also in flowstage column)
        stages_serialized = json.dumps(flow.stages)  # store as JSON string
        
        query = """
            INSERT INTO hiring_flows (flowName, createdByUser, flowstage)
            VALUES (:flowName, :createdByUser, :flowstage)
            ON CONFLICT (flowName) DO NOTHING
            RETURNING id
        """
        values = {
            "flowName": flow.flowName,
            "createdByUser": subject or "system",
            "flowstage": stages_serialized
        }
        flow_id = await database.execute(query=query, values=values)
        
        # If already exists, update flowstage column too
        if not flow_id:
            await database.execute(
                "UPDATE hiring_flows SET flowstage = :flowstage WHERE flowName = :flowName",
                {"flowName": flow.flowName, "flowstage": stages_serialized}
            )
            flow_id = await database.fetch_val(
                "SELECT id FROM hiring_flows WHERE flowName = :flowName",
                {"flowName": flow.flowName}
            )

        created_stages = []
        for stage in flow.stages:
            # 3. Create stage subfolder in Drive
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

            # ✅ Insert stage into DB
            query = """INSERT INTO flow_stages (stageName, flow_id)
                       VALUES (:stageName, :flow_id)
                       ON CONFLICT DO NOTHING"""
            await database.execute(query=query, values={"stageName": stage, "flow_id": flow_id})

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

    # Create "Hiring Pipeline" inside job position if not exists
    pipeline = _find_child_folder_by_name(drive, body.positionId, "Hiring Pipeline")
    if pipeline:
        pipeline_id = pipeline["id"]
    else:
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

# ===== Pydantic models for Candidates summary/fileText =====
# --- Response Models ---

class CandidateSummary(BaseModel):
    id: UUID
    full_name: str

class StageSummary(BaseModel):
    name: str
    candidates: List[CandidateSummary]

class RoleSummary(BaseModel):
    name: str
    count: int
    stages: List[StageSummary]

class DepartmentSummary(BaseModel):
    name: str
    count: int
    roles: List[RoleSummary]

class CandidatesSummaryResponse(BaseModel):
    message: str
    total_candidates: int
    departments: List[DepartmentSummary]


# --- Endpoint ---
@app.get("/candidates/summary", response_model=CandidatesSummaryResponse)
async def get_candidates_summary(request: Request, userEmail: Optional[str] = Query(None)):
    require_api_key(request)

    rows = await database.fetch_all("""
        SELECT id, full_name, current_role_name, current_stage_name, current_department_name
        FROM candidates
    """)

    if not rows:
        return CandidatesSummaryResponse(
            message="No candidates found",
            total_candidates=0,
            departments=[]
        )

    # Build hierarchy: Department → Role → Stage → Candidates
    departments_dict: Dict[str, Dict[str, Dict[str, List[dict]]]] = {}
    for row in rows:
        dept_name = row["current_department_name"] or "Unassigned Department"
        role_name = row["current_role_name"] or "Unknown Role"
        stage_name = row["current_stage_name"] or "Unassigned"

        dept = departments_dict.setdefault(dept_name, {})
        role = dept.setdefault(role_name, {})
        stage = role.setdefault(stage_name, [])
        stage.append({"id": row["id"], "full_name": row["full_name"]})

    # Convert to Pydantic response
    departments_out = []
    total_candidates = len(rows)

    for dept_name, roles in departments_dict.items():
        dept_total = 0
        roles_out = []

        for role_name, stages in roles.items():
            role_total = sum(len(cands) for cands in stages.values())
            dept_total += role_total

            stages_out = [
                StageSummary(name=stage_name, candidates=[CandidateSummary(**c) for c in cands])
                for stage_name, cands in stages.items()
            ]

            roles_out.append(RoleSummary(name=role_name, count=role_total, stages=stages_out))

        departments_out.append(DepartmentSummary(name=dept_name, count=dept_total, roles=roles_out))

    return CandidatesSummaryResponse(
        message="Candidate summary fetched successfully",
        total_candidates=total_candidates,
        departments=departments_out
    )


# =========================
# Updated Data Models
# =========================

class MoveByPromptItem(BaseModel):
    candidateName: str = Field(..., description="Candidate name (fuzzy matched if needed)")
    stageQuery: str = Field(..., description="Target hiring stage where the candidate should be moved")
    roleQuery: str = Field(..., description="Role or position name for which the candidate should be moved")


class MoveByPromptRequest(BaseModel):
    moves: List[MoveByPromptItem] = Field(..., description="List of candidate → stage → role mappings")
    dryRun: bool = False
    userEmail: Optional[str] = None


class MoveDecision(BaseModel):
    candidateName: str
    candidateMatchedName: Optional[str] = None
    candidateFileId: Optional[str] = None
    candidateScore: int = 0
    fromStageName: Optional[str] = None
    toStageName: Optional[str] = None
    stageScore: int = 0
    moved: bool = False
    error: Optional[str] = None
    roleName: Optional[str] = None


class MoveResponse(BaseModel):
    message: str
    dryRun: bool
    decisions: List[MoveDecision]


# =========================
# POST /candidates/moveByPrompt
# =========================

@app.post("/candidates/moveByPrompt", response_model=MoveResponse)
async def move_candidates_by_prompt(request: Request, body: MoveByPromptRequest):
    """
    New version of moveByPrompt — receives structured move instructions:
    Each includes candidate name, stage, and role.
    """
    require_api_key(request)
    subject = body.userEmail or _extract_subject_from_request(request)

    decisions: List[MoveDecision] = []

    for move in body.moves:
        try:
            # Resolve role folder and pipeline automatically (fuzzy match)
            role_id, role_info = _resolve_role(move.roleQuery)
            stages, file_index = _build_candidate_index(drive=None, position_id=role_id)

            # Resolve target stage
            stage_score, stage_match, stage_display = _resolve_best_stage(move.stageQuery, stages)

            # Resolve candidate
            cand_score, cand_match, cand_display = _resolve_best_candidate_file(move.candidateName, file_index)

            decision = MoveDecision(
                candidateName=move.candidateName,
                candidateMatchedName=cand_display if cand_match else None,
                candidateFileId=cand_match["id"] if cand_match else None,
                candidateScore=cand_score,
                fromStageName=cand_match["stageName"] if cand_match else None,
                toStageName=stage_display,
                stageScore=stage_score,
                roleName=move.roleQuery,
                moved=False,
                error=None
            )

            # Validate and move if thresholds are met
            if (
                cand_match
                and stage_match
                and cand_score >= _NAME_SCORE_THRESHOLD
                and stage_score >= _STAGE_SCORE_THRESHOLD
                and cand_match["stageId"] != stage_match["id"]
            ):
                if not body.dryRun:
                    _move_file_between_stages(None, cand_match["id"], cand_match["stageId"], stage_match["id"])
                    decision.moved = True

                    await database.execute(
                        """
                        UPDATE candidates
                        SET current_stage_name = :new_stage
                        WHERE cv_name = :cv_name
                        """,
                        {"new_stage": stage_match["name"], "cv_name": cand_match["name"]}
                    )

            decisions.append(decision)

        except Exception as e:
            decisions.append(
                MoveDecision(
                    candidateName=move.candidateName,
                    roleName=move.roleQuery,
                    toStageName=move.stageQuery,
                    moved=False,
                    error=str(e)
                )
            )

    return MoveResponse(
        message="Processed moveByPrompt (new structure)",
        dryRun=body.dryRun,
        decisions=decisions
    )


# =========================
# Pydantic models for UploadCVs
# =========================

class UploadCVItem(BaseModel):
    candidateName: str
    positionId: Optional[str] = None  # role folder ID, optional
    stageQuery: str                   # stage name (e.g., "HR Screening")
    roleQuery: Optional[str] = None   # fuzzy role name if no positionId
    content: str                      # extracted CV text provided by GPT


class UploadCVsRequest(BaseModel):
    items: List[UploadCVItem]
    dryRun: bool = False
    userEmail: Optional[str] = None


class UploadCVItemDecision(BaseModel):
    candidateName: str
    positionId: Optional[str] = None
    stageQuery: str
    roleQuery: Optional[str] = None
    createdFileId: Optional[str] = None
    createdDocLink: Optional[str] = None
    error: Optional[str] = None
    moved: bool = False
    cvTextPreview: Optional[str] = None  # Optional preview of what was saved


class UploadCVsResponse(BaseModel):
    message: str
    dryRun: bool
    decisions: List[UploadCVItemDecision]


# =========================
# POST /candidates/uploadCVs
# =========================

@app.post("/candidates/uploadCVs", response_model=UploadCVsResponse)
async def upload_cvs(request: Request, body: UploadCVsRequest):
    """
    Upload CVs into Hiring Pipeline stages using extracted text (provided by GPT).
    If positionId is missing, we try to resolve it from roleQuery.
    If stageQuery is fuzzy, we resolve it against the role’s Hiring Pipeline.
    """
    require_api_key(request)
    subject = body.userEmail or _extract_subject_from_request(request)
    _, drive, docs = get_clients(subject)

    DEPARTMENTS_FOLDER_ID = os.environ.get("DEPARTMENTS_FOLDER_ID")
    if not DEPARTMENTS_FOLDER_ID:
        raise HTTPException(500, "DEPARTMENTS_FOLDER_ID env var not set")

    logger.info("🚀 uploadCVs called with %d items (dryRun=%s, user=%s)", len(body.items), body.dryRun, subject)

    decisions: List[UploadCVItemDecision] = []

    for item in body.items:
        logger.info("📝 Processing candidate: %s (stageQuery=%s, roleQuery=%s, positionId=%s)",
                    item.candidateName, item.stageQuery, item.roleQuery, item.positionId)
    
        # ✅ Log the full extracted CV text
        if item.content:
            logger.info("📄 Full extracted CV text for %s:\n%s", item.candidateName, item.content)
        else:
            logger.warning("⚠️ No extracted CV text provided for %s", item.candidateName)
    
        dec = UploadCVItemDecision(
            candidateName=item.candidateName,
            positionId=item.positionId,
            stageQuery=item.stageQuery,
            roleQuery=item.roleQuery,
            cvTextPreview=item.content[:200] + "..." if item.content else None,
        )


        # Ensure we have a role folder ID
        role_id = item.positionId
        role_name_display = None
        if not role_id:
            if not item.roleQuery:
                dec.error = "No positionId or roleQuery provided"
                logger.warning("⚠️ No roleQuery or positionId provided for candidate %s", item.candidateName)
                decisions.append(dec)
                continue

            logger.info("🔍 Resolving role by name: %s", item.roleQuery)
            score, match, role_name_display = _resolve_best_role_by_name(
                drive, DEPARTMENTS_FOLDER_ID, item.roleQuery
            )
            logger.info("   → Role resolution score=%s, match=%s", score, role_name_display)
            if not match or score < _ROLE_SCORE_THRESHOLD:
                dec.error = f"Could not resolve role '{item.roleQuery}' (score={score})"
                logger.error("❌ Failed to resolve role for candidate %s (score=%s)", item.candidateName, score)
                decisions.append(dec)
                continue

            role_id = match["id"]
            dec.positionId = role_id
            dec.roleQuery = match["name"]

        logger.info("📂 Loading pipeline stages for roleId=%s (%s)", role_id, dec.roleQuery or role_name_display)
        stages = _load_pipeline_stages(drive, role_id)
        if not stages:
            dec.error = f"No Hiring Pipeline found under role {dec.roleQuery or role_name_display}"
            logger.error("❌ No Hiring Pipeline found for roleId=%s", role_id)
            decisions.append(dec)
            continue

        logger.info("✅ Found %d stages: %s", len(stages), [s['name'] for s in stages])
        stage_score, stage_match, stage_display = _resolve_best_stage(item.stageQuery, stages)
        logger.info("🔍 Stage resolution: query='%s' → score=%s, match=%s", item.stageQuery, stage_score, stage_display)

        if not stage_match or stage_score < _STAGE_SCORE_THRESHOLD:
            dec.error = f"Could not resolve stage '{item.stageQuery}' (score={stage_score})"
            logger.error("❌ Failed to resolve stage for candidate %s (score=%s)", item.candidateName, stage_score)
            decisions.append(dec)
            continue

        target_stage_id = stage_match["id"]
        dec.stageQuery = stage_match["name"]

        if not item.content.strip():
            dec.error = "No CV text provided"
            logger.error("❌ No CV text provided for candidate %s", item.candidateName)
            decisions.append(dec)
            continue

        if not body.dryRun:
            try:
                logger.info("📄 Creating Google Doc for candidate %s in stage %s", item.candidateName, dec.stageQuery)
                doc_name = f"{item.candidateName} - CV"
                new_id = create_google_doc(docs, drive, target_stage_id, doc_name, item.content)
                dec.createdFileId = new_id
                dec.createdDocLink = f"https://docs.google.com/document/d/{new_id}/edit"
                dec.moved = True
                logger.info("✅ Created Google Doc for %s: %s", item.candidateName, dec.createdDocLink)
            except Exception as e:
                dec.error = f"Failed to create doc: {e}"
                logger.exception("❌ Exception creating Google Doc for candidate %s", item.candidateName)
        else:
            logger.info("🔎 Dry-run enabled, not creating Google Doc for %s", item.candidateName)
            dec.moved = False

        decisions.append(dec)

    logger.info("🏁 Completed uploadCVs for %d candidates", len(body.items))
    return UploadCVsResponse(
        message="Processed uploadCVs with GPT-extracted CV text",
        dryRun=body.dryRun,
        decisions=decisions
    )


class CreateTAHRAssessmentRequest(BaseModel):
    positionId: Optional[str] = None   # direct role folder ID if known
    roleQuery: str                     # fuzzy role name if no ID provided
    candidateName: str                 # candidate name string (always used for doc naming)
    assessmentContent: str             # ✅ mandatory
    userEmail: Optional[str] = None    # impersonation
    dryRun: bool = False


class CreateTAHRAssessmentResponse(BaseModel):
    message: str
    roleId: str
    roleName: str
    candidateName: str
    createdDocs: Dict[str, str]   # { "assessment": link, "transcript": link, "geminiNotes": link }
    errors: Optional[List[str]] = None


@app.post("/candidates/createTAHRAssessment", response_model=CreateTAHRAssessmentResponse)
async def create_tahr_assessment(request: Request, body: CreateTAHRAssessmentRequest):
    require_api_key(request)
    subject = body.userEmail or _extract_subject_from_request(request)
    _, drive, docs = get_clients(subject)

    created_docs = {}
    errors = []

    DEPARTMENTS_FOLDER_ID = os.environ.get("DEPARTMENTS_FOLDER_ID")
    if not DEPARTMENTS_FOLDER_ID:
        raise HTTPException(500, "DEPARTMENTS_FOLDER_ID env var not set")

    # 🔍 Resolve Role
    role_id = body.positionId
    role_display = body.roleQuery
    if not role_id:
        score, match, role_display = _resolve_best_role_by_name(drive, DEPARTMENTS_FOLDER_ID, body.roleQuery)
        if not match or score < _ROLE_SCORE_THRESHOLD:
            raise HTTPException(404, f"Could not resolve role '{body.roleQuery}' (score={score})")
        role_id = match["id"]
        role_display = match["name"]

    # ✅ Ensure Assessment folder
    def _get_or_create_assessment_folder(drive, role_id: str, folder_name: str) -> str:
        for folder in _iter_child_folders(drive, role_id):
            if folder["name"] == folder_name:
                return folder["id"]
        return create_named_subfolder(drive, role_id, folder_name)

    assessment_folder_id = _get_or_create_assessment_folder(
        drive, role_id, "TA/HR Interviews (Assessments)"
    )

    def _save_doc(doc_name: str, content: str) -> str:
        if body.dryRun:
            return f"[DryRun] Would create: {doc_name}"
        try:
            new_id = create_google_doc(docs, drive, assessment_folder_id, doc_name, content)
            return f"https://docs.google.com/document/d/{new_id}/edit"
        except Exception as e:
            errors.append(f"Failed to create {doc_name}: {e}")
            return None

    # ✅ Save only assessment
    created_docs["assessment"] = _save_doc(
        f"{body.candidateName} - TA/HR Interview Assessment", body.assessmentContent
    )

    # ✅ Persist into DB
    try:
        # Find candidate_id from candidates table
        candidate_id = await database.fetch_val(
            "SELECT id FROM candidates WHERE full_name = :full_name",
            {"full_name": body.candidateName}
        )
    
        # Insert into ta_hr_interview_assessments
        query = """
            INSERT INTO ta_hr_interview_assessments (
                template_name, drive_id, candidate_name, candidate,
                score, role_name, department_name, created_by, created_at
            )
            VALUES (
                :template_name, :drive_id, :candidate_name, :candidate,
                :score, :role_name, :department_name, :created_by, NOW()
            )
            RETURNING id
        """
    
        values = {
            "template_name": f"{body.candidateName} - TA/HR Interview Assessment",
            "drive_id": created_docs.get("assessment").split("/d/")[1].split("/")[0] if created_docs.get("assessment") else None,
            "candidate_name": body.candidateName,
            "candidate": candidate_id,
            "score": None,  # or parse from assessmentContent if structured
            "role_name": role_display,
            "department_name": None,  # optionally fetch from roles table if you store it there
            "created_by": subject or "system"
        }
    
        new_id = await database.execute(query=query, values=values)
        logger.info("✅ Inserted TA/HR assessment into DB with id=%s", new_id)
    
    except Exception as e:
        logger.error("❌ Failed to persist TA/HR assessment: %s", e)
        errors.append(f"DB insert failed: {e}")


    return CreateTAHRAssessmentResponse(
        message="TA/HR Interview Assessment saved successfully",
        roleId=role_id,
        roleName=role_display,
        candidateName=body.candidateName,
        createdDocs=created_docs,
        errors=errors or None
    )



class TemplateFile(BaseModel):
    id: str
    name: str
    text: Optional[str] = None
    error: Optional[str] = None

class GetTAHRTemplateResponse(BaseModel):
    message: str
    roleId: str
    roleName: str
    files: List[TemplateFile]


@app.get("/positions/getTAHRInterviewTemplate", response_model=GetTAHRTemplateResponse)
def get_tahr_interview_template(
    request: Request,
    positionId: Optional[str] = Query(None, description="Role folder ID"),
    roleQuery: Optional[str] = Query(None, description="Role name to fuzzy match if no ID"),
    userEmail: Optional[str] = Query(None, description="Impersonate this Workspace user"),
):
    """
    Fetch and extract the TA/HR Interview Template docs for a role.
    - Locate 'TA/HR Interview Template' subfolder under the role.
    - Extract full text of all documents inside it.
    """
    require_api_key(request)
    subject = userEmail or _extract_subject_from_request(request)
    _, drive, docs = get_clients(subject)

    DEPARTMENTS_FOLDER_ID = os.environ.get("DEPARTMENTS_FOLDER_ID")
    if not DEPARTMENTS_FOLDER_ID:
        raise HTTPException(500, "DEPARTMENTS_FOLDER_ID env var not set")

    # 🔍 Resolve Role
    role_id = positionId
    role_display = roleQuery
    if not role_id:
        if not roleQuery:
            raise HTTPException(400, "Must provide either positionId or roleQuery")
        score, match, role_display = _resolve_best_role_by_name(drive, DEPARTMENTS_FOLDER_ID, roleQuery)
        if not match or score < _ROLE_SCORE_THRESHOLD:
            raise HTTPException(404, f"Could not resolve role '{roleQuery}' (score={score})")
        role_id = match["id"]
        role_display = match["name"]

    # 🔍 Locate "TA/HR Interview Template" folder
    scoring_folder = _find_child_folder_by_name(drive, role_id, "TA/HR Interview Template")
    if not scoring_folder:
        raise HTTPException(404, f"No 'TA/HR Interview Template' folder found under role {role_display}")

    # 📂 Get all files in the folder
    files = _scan_stage_files(drive, scoring_folder["id"])
    out_files: List[TemplateFile] = []

    for f in files:
        text, err = (None, None)
        if _is_doc_or_pdf(f["name"], f["mimeType"]):
            text, err = _extract_text_from_file(drive, docs, f)
        out_files.append(TemplateFile(
            id=f["id"],
            name=f["name"],
            text=text,
            error=err
        ))

    return GetTAHRTemplateResponse(
        message=f"Fetched {len(out_files)} scoring model docs for role {role_display}",
        roleId=role_id,
        roleName=role_display,
        files=out_files
    )

class CreateFirstTechInterviewAssessmentRequest(BaseModel):
    positionId: Optional[str] = None   # direct role folder ID if known
    roleQuery: str                     # fuzzy role name if no ID provided
    candidateName: str                 # candidate name string (always used for doc naming)
    assessmentContent: str             # ✅ mandatory
    userEmail: Optional[str] = None    # impersonation
    dryRun: bool = False


class CreateFirstTechInterviewAssessmentResponse(BaseModel):
    message: str
    roleId: str
    roleName: str
    candidateName: str
    createdDocs: Dict[str, str]   # { "assessment": link, "transcript": link, "geminiNotes": link }
    errors: Optional[List[str]] = None


@app.post("/candidates/createFirstTechnicalInterviewAssessment", response_model=CreateFirstTechInterviewAssessmentResponse)
async def create_first_tech_interview_assessment(request: Request, body: CreateFirstTechInterviewAssessmentRequest):
    require_api_key(request)
    subject = body.userEmail or _extract_subject_from_request(request)
    _, drive, docs = get_clients(subject)

    created_docs = {}
    errors = []

    if not body.assessmentContent:
        raise HTTPException(400, "Must provide assessmentContent")

    DEPARTMENTS_FOLDER_ID = os.environ.get("DEPARTMENTS_FOLDER_ID")
    if not DEPARTMENTS_FOLDER_ID:
        raise HTTPException(500, "DEPARTMENTS_FOLDER_ID env var not set")

    # 🔍 Resolve Role
    role_id = body.positionId
    role_display = body.roleQuery
    if not role_id:
        score, match, role_display = _resolve_best_role_by_name(drive, DEPARTMENTS_FOLDER_ID, body.roleQuery)
        if not match or score < _ROLE_SCORE_THRESHOLD:
            raise HTTPException(404, f"Could not resolve role '{body.roleQuery}' (score={score})")
        role_id = match["id"]
        role_display = match["name"]

    # ✅ Ensure Assessment folder
    def _get_or_create_assessment_folder(drive, role_id: str, folder_name: str) -> str:
        for folder in _iter_child_folders(drive, role_id):
            if folder["name"] == folder_name:
                return folder["id"]
        return create_named_subfolder(drive, role_id, folder_name)

    assessment_folder_id = _get_or_create_assessment_folder(
        drive, role_id, "1st Technical Interviews (Assessments)"
    )

    def _save_doc(doc_name: str, content: str) -> str:
        if body.dryRun:
            return f"[DryRun] Would create: {doc_name}"
        try:
            new_id = create_google_doc(docs, drive, assessment_folder_id, doc_name, content)
            return f"https://docs.google.com/document/d/{new_id}/edit"
        except Exception as e:
            errors.append(f"Failed to create {doc_name}: {e}")
            return None

    # ✅ Save only assessment
    created_docs["assessment"] = _save_doc(
        f"{body.candidateName} - 1st Technical Interview Assessment", body.assessmentContent
    )


    # ✅ Persist into DB
    try:
        # Find candidate_id from candidates table
        candidate_id = await database.fetch_val(
            "SELECT id FROM candidates WHERE full_name = :full_name",
            {"full_name": body.candidateName}
        )
    
        # Insert into first_tech_interview_assessments
        query = """
            INSERT INTO first_tech_interview_assessments (
                template_name, drive_id, candidate_name, candidate,
                score, role_name, department_name, created_by, created_at
            )
            VALUES (
                :template_name, :drive_id, :candidate_name, :candidate,
                :score, :role_name, :department_name, :created_by, NOW()
            )
            RETURNING id
        """
    
        values = {
            "template_name": f"{body.candidateName} - 1st Technical Interview Assessment",
            "drive_id": created_docs.get("assessment").split("/d/")[1].split("/")[0] if created_docs.get("assessment") else None,
            "candidate_name": body.candidateName,
            "candidate": candidate_id,
            "score": None,  # You can parse actual score from body.assessmentContent later
            "role_name": role_display,
            "department_name": None,  # Optional — can fetch from roles table if needed
            "created_by": subject or "system"
        }
    
        new_id = await database.execute(query=query, values=values)
        logger.info("✅ Inserted 1st Technical Interview assessment into DB with id=%s", new_id)
    
    except Exception as e:
        logger.error("❌ Failed to persist 1st Tech Interview assessment: %s", e)
        errors.append(f"DB insert failed: {e}")


    
    return CreateFirstTechInterviewAssessmentResponse(
        message="1st Technical Interview Assessment saved successfully",
        roleId=role_id,
        roleName=role_display,
        candidateName=body.candidateName,
        createdDocs=created_docs,
        errors=errors or None
    )



class GetFirstTechTemplateRequest(BaseModel):
    interviewTranscript: str
    geminiMeetingNotes: str
    role: str
    candidate: str
    assessmentType: str
    userEmail: Optional[str] = None


class GetFirstTechTemplateResponse(BaseModel):
    message: str
    roleId: str
    roleName: str
    candidate: str
    assessmentType: str
    interviewTranscript: Optional[str] = None
    geminiMeetingNotes: Optional[str] = None
    files: List[TemplateFile]



@app.post("/positions/GetFirstTechnicalInterviewTemplate", response_model=GetFirstTechTemplateResponse)
def get_first_tech_interview_template(request: Request, body: GetFirstTechTemplateRequest):
    require_api_key(request)
    subject = body.userEmail or _extract_subject_from_request(request)
    _, drive, docs = get_clients(subject)

    DEPARTMENTS_FOLDER_ID = os.environ.get("DEPARTMENTS_FOLDER_ID")
    if not DEPARTMENTS_FOLDER_ID:
        raise HTTPException(500, "DEPARTMENTS_FOLDER_ID env var not set")

    # 🔍 Resolve role
    score, match, role_display = _resolve_best_role_by_name(drive, DEPARTMENTS_FOLDER_ID, body.role)
    if not match or score < _ROLE_SCORE_THRESHOLD:
        raise HTTPException(404, f"Could not resolve role '{body.role}' (score={score})")
    role_id = match["id"]
    role_display = match["name"]

    # 🔍 Locate scoring folder (fuzzy matching)
    scoring_folder = _find_child_folder_by_name(drive, role_id, "1st Technical Interview Template")
    if not scoring_folder:
        scoring_folder = _find_child_folder_by_name(drive, role_id, "1st Technical Interview Template")
    if not scoring_folder:
        raise HTTPException(404, f"No '1st Technical Interview Template' folder found for {role_display}")

    # 📂 Get all files inside
    files = _scan_stage_files(drive, scoring_folder["id"])
    out_files: List[TemplateFile] = []
    for f in files:
        text, err = (None, None)
        if _is_doc_or_pdf(f["name"], f["mimeType"]):
            text, err = _extract_text_from_file(drive, docs, f)
        out_files.append(TemplateFile(
            id=f["id"], name=f["name"], text=text, error=err
        ))

    return GetFirstTechTemplateResponse(
    message=f"Fetched {len(out_files)} scoring model docs for role {role_display}",
    roleId=role_id,
    roleName=role_display,
    candidate=body.candidate,
    assessmentType=body.assessmentType,
    interviewTranscript=body.interviewTranscript,
    geminiMeetingNotes=body.geminiMeetingNotes,
    files=out_files
)



class UpdateRoleStatusRequest(BaseModel):
    prompt: str = Field(..., description="Free text like 'Close role1, role2' or 'Mark all roles as open'")
    dryRun: bool = False
    userEmail: Optional[str] = None


class RoleStatusDecision(BaseModel):
    roleQuery: str
    matchedRoleName: Optional[str] = None
    roleId: Optional[str] = None
    statusQuery: str
    resolvedStatus: Optional[str] = None
    score: int
    updated: bool
    error: Optional[str] = None

class UpdateRoleStatusResponse(BaseModel):
    message: str
    dryRun: bool
    decisions: List[RoleStatusDecision]



@app.post("/roles/updateStatus", response_model=UpdateRoleStatusResponse)
def update_role_status(request: Request, body: UpdateRoleStatusRequest):
    require_api_key(request)
    subject = body.userEmail or _extract_subject_from_request(request)
    _, drive, _ = get_clients(subject)

    DEPARTMENTS_FOLDER_ID = os.environ.get("DEPARTMENTS_FOLDER_ID")
    if not DEPARTMENTS_FOLDER_ID:
        raise HTTPException(500, "DEPARTMENTS_FOLDER_ID env var not set")

    groups = _parse_status_prompt(body.prompt, drive, DEPARTMENTS_FOLDER_ID)
    if not groups:
        raise HTTPException(400, "Could not parse any 'role → status' groups from the prompt")

    # Reuse fuzzy matching + update logic
    decisions: List[RoleStatusDecision] = []
    for grp in groups:
        resolved_status = _normalize_status(grp["statusQuery"])
        if not resolved_status:
            for rq in grp["roleQueries"]:
                decisions.append(RoleStatusDecision(
                    roleQuery=rq,
                    statusQuery=grp["statusQuery"],
                    resolvedStatus=None,
                    score=0,
                    updated=False,
                    error=f"Unrecognized status '{grp['statusQuery']}'"
                ))
            continue

        for rq in grp["roleQueries"]:
            score, match, display = _resolve_best_role_by_name(drive, DEPARTMENTS_FOLDER_ID, rq)
            decision = RoleStatusDecision(
                roleQuery=rq,
                matchedRoleName=display if match else None,
                roleId=match["id"] if match else None,
                statusQuery=grp["statusQuery"],
                resolvedStatus=resolved_status,
                score=score,
                updated=False,
                error=None
            )

            if not match:
                decision.error = "No role matched"
            elif score < _ROLE_SCORE_THRESHOLD:
                decision.error = f"Low match score ({score}<{_ROLE_SCORE_THRESHOLD})"
            else:
                if not body.dryRun:
                    try:
                        drive.files().update(
                            fileId=match["id"],
                            body={"properties": {"roleStatus": resolved_status}},
                            fields="id,properties",
                            supportsAllDrives=True
                        ).execute()
                        decision.updated = True
                    except Exception as e:
                        decision.error = f"Update failed: {e}"
                else:
                    decision.updated = False
            decisions.append(decision)

    return UpdateRoleStatusResponse(
        message="Processed role status updates",
        dryRun=body.dryRun,
        decisions=decisions
    )

class UpdateDocumentRequest(BaseModel):
    fileId: str
    content: str
    userEmail: Optional[str] = None  # for impersonation
    rawMode: bool = False            # if true, inserts plain text without formatting


class UpdateDocumentResponse(BaseModel):
    message: str
    fileId: str
    docLink: str


@app.post("/documents/update", response_model=UpdateDocumentResponse)
def update_document(request: Request, body: UpdateDocumentRequest):
    """
    Overwrite an existing Google Doc with new content.
    Keeps the same fileId but clears and re-writes all content.
    """
    require_api_key(request)
    subject = body.userEmail or _extract_subject_from_request(request)
    _, drive, docs = get_clients(subject)

    # Fetch doc length to clear contents
    try:
        doc = docs.documents().get(documentId=body.fileId).execute()
    except Exception as e:
        raise HTTPException(404, f"Document {body.fileId} not found: {e}")

    doc_length = doc.get("body").get("content")[-1]["endIndex"]

    # Step 1: Clear old content
    requests_clear = []
    if doc_length > 2:
        requests_clear.append({
            "deleteContentRange": {
                "range": {"startIndex": 1, "endIndex": doc_length - 1}
            }
        })
    if requests_clear:
        docs.documents().batchUpdate(documentId=body.fileId, body={"requests": requests_clear}).execute()

    # Step 2: Insert new content
    if body.rawMode:
        # Insert plain text
        docs.documents().batchUpdate(documentId=body.fileId, body={
            "requests": [{"insertText": {"location": {"index": 1}, "text": body.content}}]
        }).execute()
    else:
        # Reuse existing formatter (so updates look like create_google_doc outputs)
        create_google_doc(
            docs, drive, folder_id="",  # folder_id not needed, we're updating
            title="",                   # not creating new file
            content=body.content,
            raw_mode=False
        )
        # ⚠️ You may want to refactor create_google_doc into two parts
        # (one for creating, one for writing content), so you can call the writing part here.

    return UpdateDocumentResponse(
        message="Document updated successfully",
        fileId=body.fileId,
        docLink=f"https://docs.google.com/document/d/{body.fileId}/edit"
    )

class CreateSecondTechInterviewRequest(BaseModel):
    positionId: str
    candidateName: str
    content: Optional[str] = None
    userEmail: Optional[str] = None  # for impersonation

@app.post("/candidates/createSecondTechnicalInterview")
async def create_second_tech_interview(request: Request, body: CreateSecondTechInterviewRequest):
    require_api_key(request)
    subject = body.userEmail or _extract_subject_from_request(request)
    _, drive, docs = get_clients(subject)

    # ✅ Ensure 2nd Technical Interview Template folder exists
    screening_folder = _find_child_folder_by_name(drive, body.positionId, "2nd Technical Interview Template")
    if screening_folder:
        screening_folder_id = screening_folder["id"]
    else:
        screening_folder_id = create_named_subfolder(drive, body.positionId, "2nd Technical Interview Template")

    # ✅ Ensure 2nd Technical Interviews (Assessments) folder exists
    tech_assessment_folder = _find_child_folder_by_name(drive, body.positionId, "2nd Technical Interviews (Assessments)")
    if tech_assessment_folder:
        tech_assessment_folder_id = tech_assessment_folder["id"]
    else:
        tech_assessment_folder_id = create_named_subfolder(drive, body.positionId, "2nd Technical Interviews (Assessments)")

    # ✅ Check if the candidate's interview doc already exists
    query = (
        "mimeType='application/vnd.google-apps.document' "
        "and trashed=false "
        f"and name='{body.candidateName} - 2nd Technical Interview' "
        f"and '{screening_folder_id}' in parents"
    )
    results = drive.files().list(
        q=query,
        fields="files(id,name)",
        includeItemsFromAllDrives=True,
        supportsAllDrives=True
    ).execute()

    created = False
    if results.get("files"):
        existing = results["files"][0]
        file_id = existing["id"]
        file_name = existing["name"]
    else:
        # ✅ Default polished template
        content = body.content or f"""
            2nd Technical Interview – {body.candidateName}

            Candidate: {body.candidateName}
            Role: [Specify Role Here]
            Date: ___________________________
            Interviewer(s): ___________________________

            Advanced Technical Questions:
            - Q1: ______________________________________
            - Q2: ______________________________________
            - Q3: ______________________________________
            - Q4: ______________________________________
            - Q5: ______________________________________

            Feedback:
            - Strengths:
            ____________________________________________

            - Areas for Improvement:
            ____________________________________________

            Scorecard (1–5 for each dimension):
            - System Design / Architecture: ___
            - Code Quality / Best Practices: ___
            - Problem Solving: ___
            - Collaboration & Communication: ___
            - Culture Fit: ___

            Overall Recommendation:
            - Strong Hire / Hire / Neutral / No Hire
        """

        file_name = f"{body.candidateName} - 2nd Technical Interview"
        file_id = create_google_doc(docs, drive, screening_folder_id, file_name, content)
        created = True

    doc_link = f"https://docs.google.com/document/d/{file_id}/edit"

    # ✅ Persist into second_tech_interview_templates table
    try:
        await database.execute(
            """
            INSERT INTO second_tech_interview_templates (
                template_name, role_name, created_by, template_url, created_at, drive_id
            )
            VALUES (:template_name, :role_name, :created_by, :template_url, :created_at, :drive_id)
            ON CONFLICT (drive_id) DO NOTHING
            """,
            {
                "template_name": file_name,
                "role_name": "[Specify Role Here]",  # You can resolve role name from roles table if needed
                "created_by": subject or "system",
                "template_url": doc_link,
                "created_at": datetime.now(timezone.utc),
                "drive_id": file_id
            }
        )

        # ✅ Update roles table with second tech interview template link
        role_uuid = await database.fetch_val(
            "SELECT id FROM roles WHERE drive_id = :drive_id",
            {"drive_id": body.positionId}
        )
        if role_uuid:
            await database.execute(
                """
                UPDATE roles
                SET second_tech_interview_template_url = :template_url
                WHERE id = :id
                """,
                {"template_url": doc_link, "id": role_uuid}
            )

    except Exception as e:
        logger.error(f"❌ Failed to persist 2nd Technical Interview Template or update role: {e}")
        raise HTTPException(500, f"DB insert/update failed: {e}")

    return {
        "message": f"2nd Technical Interview template {'created' if created else 'already existed'} for {body.candidateName}",
        "fileId": file_id,
        "folderId": screening_folder_id,
        "docLink": doc_link,
        "created": created
    }


class GetSecondTechTemplateRequest(BaseModel):
    interviewTranscript: str
    geminiMeetingNotes: str
    role: str
    candidate: str
    assessmentType: str
    userEmail: Optional[str] = None


class GetSecondTechTemplateResponse(BaseModel):
    message: str
    roleId: str
    roleName: str
    candidate: str
    assessmentType: str
    interviewTranscript: Optional[str] = None
    geminiMeetingNotes: Optional[str] = None
    files: List[TemplateFile]


@app.post("/positions/GetSecondTechnicalInterviewTemplate", response_model=GetSecondTechTemplateResponse)
def get_second_tech_interview_template(request: Request, body: GetSecondTechTemplateRequest):
    require_api_key(request)
    subject = body.userEmail or _extract_subject_from_request(request)
    _, drive, docs = get_clients(subject)

    DEPARTMENTS_FOLDER_ID = os.environ.get("DEPARTMENTS_FOLDER_ID")
    if not DEPARTMENTS_FOLDER_ID:
        raise HTTPException(500, "DEPARTMENTS_FOLDER_ID env var not set")

    # 🔍 Resolve role
    score, match, role_display = _resolve_best_role_by_name(drive, DEPARTMENTS_FOLDER_ID, body.role)
    if not match or score < _ROLE_SCORE_THRESHOLD:
        raise HTTPException(404, f"Could not resolve role '{body.role}' (score={score})")
    role_id = match["id"]
    role_display = match["name"]

    # 🔍 Locate 2nd Technical Interview Template folder
    scoring_folder = _find_child_folder_by_name(drive, role_id, "2nd Technical Interview Template")
    if not scoring_folder:
        raise HTTPException(404, f"No '2nd Technical Interview Template' folder found for {role_display}")

    # 📂 Get all files inside
    files = _scan_stage_files(drive, scoring_folder["id"])
    out_files: List[TemplateFile] = []
    for f in files:
        text, err = (None, None)
        if _is_doc_or_pdf(f["name"], f["mimeType"]):
            text, err = _extract_text_from_file(drive, docs, f)
        out_files.append(TemplateFile(
            id=f["id"], name=f["name"], text=text, error=err
        ))

    return GetSecondTechTemplateResponse(
        message=f"Fetched {len(out_files)} scoring model docs for role {role_display}",
        roleId=role_id,
        roleName=role_display,
        candidate=body.candidate,
        assessmentType=body.assessmentType,
        interviewTranscript=body.interviewTranscript,
        geminiMeetingNotes=body.geminiMeetingNotes,
        files=out_files
    )

class CreateSecondTechInterviewAssessmentRequest(BaseModel):
    positionId: Optional[str] = None   # direct role folder ID if known
    roleQuery: str                     # fuzzy role name if no ID provided
    candidateName: str                 # candidate name string (always used for doc naming)
    assessmentContent: str             # ✅ mandatory
    userEmail: Optional[str] = None    # impersonation
    dryRun: bool = False


class CreateSecondTechInterviewAssessmentResponse(BaseModel):
    message: str
    roleId: str
    roleName: str
    candidateName: str
    createdDocs: Dict[str, str]   # { "assessment": link, "transcript": link, "geminiNotes": link }
    errors: Optional[List[str]] = None


@app.post("/candidates/createSecondTechnicalInterviewAssessment", response_model=CreateSecondTechInterviewAssessmentResponse)
async def create_second_tech_interview_assessment(request: Request, body: CreateSecondTechInterviewAssessmentRequest):
    require_api_key(request)
    subject = body.userEmail or _extract_subject_from_request(request)
    _, drive, docs = get_clients(subject)

    created_docs = {}
    errors = []

    if not body.assessmentContent:
        raise HTTPException(400, "Must provide assessmentContent")

    DEPARTMENTS_FOLDER_ID = os.environ.get("DEPARTMENTS_FOLDER_ID")
    if not DEPARTMENTS_FOLDER_ID:
        raise HTTPException(500, "DEPARTMENTS_FOLDER_ID env var not set")

    # 🔍 Resolve Role
    role_id = body.positionId
    role_display = body.roleQuery
    if not role_id:
        score, match, role_display = _resolve_best_role_by_name(drive, DEPARTMENTS_FOLDER_ID, body.roleQuery)
        if not match or score < _ROLE_SCORE_THRESHOLD:
            raise HTTPException(404, f"Could not resolve role '{body.roleQuery}' (score={score})")
        role_id = match["id"]
        role_display = match["name"]

    # ✅ Ensure Assessment folder
    def _get_or_create_assessment_folder(drive, role_id: str, folder_name: str) -> str:
        for folder in _iter_child_folders(drive, role_id):
            if folder["name"] == folder_name:
                return folder["id"]
        return create_named_subfolder(drive, role_id, folder_name)

    assessment_folder_id = _get_or_create_assessment_folder(
        drive, role_id, "2nd Technical Interviews (Assessments)"
    )

    def _save_doc(doc_name: str, content: str) -> str:
        if body.dryRun:
            return f"[DryRun] Would create: {doc_name}"
        try:
            new_id = create_google_doc(docs, drive, assessment_folder_id, doc_name, content)
            return f"https://docs.google.com/document/d/{new_id}/edit"
        except Exception as e:
            errors.append(f"Failed to create {doc_name}: {e}")
            return None

    # ✅ Save only assessment
    created_docs["assessment"] = _save_doc(
        f"{body.candidateName} - 2nd Technical Interview Assessment", body.assessmentContent
    )

    # ✅ Persist into PostgreSQL
    try:
        # 1️⃣ Find candidate_id from candidates table
        candidate_id = await database.fetch_val(
            "SELECT id FROM candidates WHERE full_name = :full_name",
            {"full_name": body.candidateName}
        )

        # 2️⃣ Extract Drive file ID (if available)
        drive_id = None
        if created_docs.get("assessment") and "docs.google.com/document/d/" in created_docs["assessment"]:
            drive_id = created_docs["assessment"].split("/d/")[1].split("/")[0]

        # 3️⃣ Build SQL insert aligned with schema
        query = """
            INSERT INTO second_tech_interview_assessments (
                template_name, drive_id, candidate_name, candidate,
                score, role_name, department_name, created_by, created_at
            )
            VALUES (
                :template_name, :drive_id, :candidate_name, :candidate,
                :score, :role_name, :department_name, :created_by, NOW()
            )
            RETURNING id
        """

        # 4️⃣ Build values dict
        values = {
            "template_name": f"{body.candidateName} - 2nd Technical Interview Assessment",
            "drive_id": drive_id,
            "candidate_name": body.candidateName,
            "candidate": candidate_id,
            "score": None,  # Optional: parse from body.assessmentContent if structured
            "role_name": role_display,
            "department_name": None,  # Optional: fetch from roles if desired
            "created_by": subject or "system",
        }

        # 5️⃣ Execute DB insert
        new_id = await database.execute(query=query, values=values)
        logger.info("✅ Inserted 2nd Technical Interview assessment into DB with id=%s", new_id)

    except Exception as e:
        logger.error("❌ Failed to persist 2nd Technical Interview assessment: %s", e)
        errors.append(f"DB insert failed: {e}")

    return CreateSecondTechInterviewAssessmentResponse(
        message="2nd Technical Interview Assessment saved successfully",
        roleId=role_id,
        roleName=role_display,
        candidateName=body.candidateName,
        createdDocs=created_docs,
        errors=errors or None
    )



@app.post("/slack/events")
async def slack_events(request: Request):
    data = await request.json()

    if "challenge" in data:
        return {"challenge": data["challenge"]}

    if "event" in data:
        event = data["event"]
        if event.get("type") == "message" and not event.get("bot_id"):
            user_prompt = event.get("text")
            channel_id = event.get("channel")

            response_text = await process_with_gpt(user_prompt)

            async with httpx.AsyncClient() as client:
                await client.post(
                    SLACK_API_URL,
                    headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"},
                    json={"channel": channel_id, "text": response_text}
                )

    return {"ok": True}


## For importing candidates on the database
class ImportCandidatesResponse(BaseModel):
    message: str
    inserted: int
    skipped: int

@app.post("/candidates/import")
async def import_candidates(request: Request, userEmail: Optional[str] = None):
    require_api_key(request)
    subject = userEmail or _extract_subject_from_request(request)
    _, drive, _ = get_clients(subject)

    DEPARTMENTS_FOLDER_ID = os.environ.get("DEPARTMENTS_FOLDER_ID")
    if not DEPARTMENTS_FOLDER_ID:
        raise HTTPException(500, "DEPARTMENTS_FOLDER_ID env var not set")

    inserted, skipped = 0, 0

    # Step 1: Loop departments
    for dept in _iter_child_folders(drive, DEPARTMENTS_FOLDER_ID):
        dept_id, dept_name = dept["id"], dept["name"]

        # Step 2: Loop roles inside department
        for role in _iter_child_folders(drive, dept_id):
            role_id, role_name = role["id"], role["name"]

            # Step 3: Locate "Hiring Pipeline"
            pipeline = _find_child_folder_by_name(drive, role_id, "Hiring Pipeline")
            if not pipeline:
                continue

            # Step 4: Loop stages inside pipeline
            for stage in _iter_child_folders(drive, pipeline["id"]):
                stage_id, stage_name = stage["id"], stage["name"]

                # Step 5: Loop candidate files
                for cand in _iter_child_files(drive, stage_id):
                    file_id, file_name = cand["id"], cand["name"]

                    full_name = _strip_ext(file_name)
                    first, last = _split_name(full_name)
                    cv_name = file_name
                    cv_url = f"https://drive.google.com/file/d/{file_id}/view"

                    # Insert into DB
                    query = """
                        INSERT INTO candidates (
                            status, full_name, first_name, last_name,
                            cv_name, cv_url, source,
                            current_stage_name, current_role_name, current_department_name,
                            created_by_user, created_at
                        )
                        VALUES (
                            :status, :full_name, :first_name, :last_name,
                            :cv_name, :cv_url, :source,
                            :current_stage_name, :current_role_name, :current_department_name,
                            :created_by_user, NOW()
                        )
                        ON CONFLICT DO NOTHING
                    """
                    values = {
                        "status": "active",
                        "full_name": full_name,
                        "first_name": first,
                        "last_name": last,
                        "cv_name": cv_name,
                        "cv_url": cv_url,
                        "source": "google-drive",
                        "current_stage_name": stage_name,
                        "current_role_name": role_name,
                        "current_department_name": dept_name,  # 👈 add this
                        "created_by_user": "julio@cipherscale.com",
                    }
                    
                    try:
                        result = await database.execute(query=query, values=values)
                        if result:
                            inserted += 1
                        else:
                            skipped += 1
                    except Exception as e:
                        logging.error(f"❌ Failed to insert {full_name}: {e}")
                        skipped += 1

    return ImportCandidatesResponse(
        message="Candidate import completed",
        inserted=inserted,
        skipped=skipped
    )


class GetCandidateDocumentsRequest(BaseModel):
    candidate_name: str
    userEmail: Optional[str] = None


class GetCandidateDocumentsResponse(BaseModel):
    message: str
    filters: Dict[str, Any]
    results: Dict[str, Any]


@app.post("/candidates/getDocument", response_model=GetCandidateDocumentsResponse)
async def get_candidate_documents(request: Request, body: GetCandidateDocumentsRequest):
    """
    Fetch all relevant documents and records from the DB related to a candidate.
    Supports partial name matching (ILIKE).
    Returns structured JSON for GPT to interpret and find the relevant document.
    """
    require_api_key(request)

    candidate_name = body.candidate_name.strip()
    if not candidate_name:
        raise HTTPException(400, "candidate_name is required")

    # Log for debug
    logger.info(f"🔍 Fetching candidate documents for name ~ '{candidate_name}'")

    # Define all the queries
    queries = {
        "candidates": """
            SELECT * FROM candidates
            WHERE full_name ILIKE :pattern
               OR first_name ILIKE :pattern
               OR last_name ILIKE :pattern
        """,
        "ta_hr_interview_assessments": """
            SELECT * FROM ta_hr_interview_assessments
            WHERE candidate_name ILIKE :pattern
        """,
        "first_tech_interview_assessments": """
            SELECT * FROM first_tech_interview_assessments
            WHERE candidate_name ILIKE :pattern
        """,
        "second_tech_interview_assessments": """
            SELECT * FROM second_tech_interview_assessments
            WHERE candidate_name ILIKE :pattern
        """
    }

    pattern = f"%{candidate_name}%"
    results = {}

    # Execute each query and collect the results
    for table, sql in queries.items():
        try:
            rows = await database.fetch_all(sql, values={"pattern": pattern})
            # Convert to list of dicts
            results[table] = [dict(row) for row in rows]
            logger.info(f"✅ Found {len(rows)} rows in {table}")
        except Exception as e:
            logger.error(f"❌ Error fetching {table}: {e}")
            results[table] = {"error": str(e)}

    total_records = sum(len(v) if isinstance(v, list) else 0 for v in results.values())

    return GetCandidateDocumentsResponse(
        message=f"Fetched {total_records} records across candidate-related tables.",
        filters={"candidate_name": candidate_name},
        results=results
    )


@app.get("/whoami") # Verify who the api is acting as when user impersonation
def whoami(request: Request):
    require_api_key(request)
    subject = _extract_subject_from_request(request)
    _, drive, _ = get_clients(subject)
    about = drive.about().get(fields="user(emailAddress,displayName),storageQuota").execute()
    return {"subject_param": subject, "drive_user": about.get("user")}
