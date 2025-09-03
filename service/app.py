from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from datetime import datetime, timezone
from typing import Optional, List
import os, json, textwrap, re, uuid, base64, logging
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
from google.auth.exceptions import RefreshError # For user impersonation
from pydantic import BaseModel
from collections import Counter

# Configure logging once (top of file)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE64_PATTERN = re.compile(r'^[A-Za-z0-9+/=\r\n]+$')  # allow newlines too

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

# Below are helper functions

def prepare_candidate_file(file_ref: str) -> str:
    if file_ref.startswith("drive:"):
        return file_ref

    # Detect if it looks like base64 (long enough and only base64 chars)
    try:
        # Attempt decode → if works, it's base64
        base64.b64decode(file_ref, validate=True)
        return file_ref
    except Exception:
        pass

    if not os.path.exists(file_ref):
        raise FileNotFoundError(f"File not found: {file_ref}")

    with open(file_ref, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
        

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
        "https://www.googleapis.com/auth/spreadsheets.readonly",
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


def norm(s: Optional[str]) -> str:
    return (s or "").strip()


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

@app.get("/roles/unique")
def unique_roles(request: Request, fileId: str, sheetName: Optional[str] = None,
                 headerRow: int = 1, roleHeader: str = "Role"):
    require_api_key(request)
    subject = _extract_subject_from_request(request)
    sheets, drive, _ = get_clients(subject)

    meta = drive.files().get(fileId=fileId, fields="id,name,mimeType,modifiedTime", supportsAllDrives=True).execute()
    if meta.get("mimeType") != "application/vnd.google-apps.spreadsheet":
        raise HTTPException(400, "File is not a Google Sheet")

    ss = sheets.spreadsheets().get(spreadsheetId=fileId).execute()
    title = sheetName or ss["sheets"][0]["properties"]["title"]

    grid = sheets.spreadsheets().get(
        spreadsheetId=fileId,
        ranges=[f"{title}!A:ZZ"],
        includeGridData=True,
        fields="sheets(data(rowData(values(userEnteredValue,hyperlink))))"
    ).execute()

    rows = grid.get("sheets", [{}])[0].get("data", [{}])[0].get("rowData", [])
    if headerRow < 1 or headerRow > len(rows):
        raise HTTPException(400, "Header row out of range")

    header_cells = rows[headerRow - 1].get("values", [])
    headers = [norm((c.get("userEnteredValue", {}) or {}).get("stringValue")) for c in header_cells]
    try:
        role_idx = next(i for i,h in enumerate(headers) if h.lower() == roleHeader.lower())
    except StopIteration:
        raise HTTPException(400, f"Header '{roleHeader}' not found")

    roles: List[str] = []
    for r in rows[headerRow:]:
        cells = r.get("values", [])
        if role_idx < len(cells):
            cell = cells[role_idx]
            text = norm((cell.get("userEnteredValue", {}) or {}).get("stringValue"))
            link = norm(cell.get("hyperlink"))
            val = text or link
            if val:
                roles.append(val)

    unique = sorted(set(roles))
    return {
        "count": len(unique),
        "roles": unique,
        "source": {
            "fileId": fileId,
            "sheetName": title,
            "detectedColumnIndex": role_idx,
            "updatedAt": datetime.now(timezone.utc).isoformat()
        }
    }



@app.get("/stages/summary")
def stages_summary(
    request: Request,
    fileId: str,
    sheetName: Optional[str] = None,
    headerRow: int = 1,
    stageHeader: str = "Candidate Stage Helper",
):
    # Reuse your existing API key check & Google clients
    require_api_key(request)
    subject = _extract_subject_from_request(request)
    sheets, drive, _ = get_clients(subject)

    # Make sure it's a Google Sheet we can read
    meta = drive.files().get(fileId=fileId, fields="mimeType", supportsAllDrives=True).execute()
    if meta.get("mimeType") != "application/vnd.google-apps.spreadsheet":
        raise HTTPException(400, "File is not a Google Sheet")

    # Resolve tab name
    ss = sheets.spreadsheets().get(spreadsheetId=fileId).execute()
    title = sheetName or ss["sheets"][0]["properties"]["title"]

    # Pull cells (values only is enough here)
    grid = sheets.spreadsheets().get(
        spreadsheetId=fileId,
        ranges=[f"{title}!A:ZZ"],
        includeGridData=True,
        fields="sheets(data(rowData(values(userEnteredValue))))"
    ).execute()

    rows = grid.get("sheets", [{}])[0].get("data", [{}])[0].get("rowData", [])
    if headerRow < 1 or headerRow > len(rows):
        raise HTTPException(400, "Header row out of range")

    # Locate the "Candidate Stage" column by header
    header_cells = rows[headerRow - 1].get("values", [])
    headers = [norm((c.get("userEnteredValue", {}) or {}).get("stringValue")) for c in header_cells]
    try:
        stage_idx = next(i for i, h in enumerate(headers) if h.lower() == stageHeader.lower())
    except StopIteration:
        raise HTTPException(400, f"Header '{stageHeader}' not found")

    # Collect stages
    stages: List[str] = []
    for r in rows[headerRow:]:
        cells = r.get("values", [])
        if stage_idx < len(cells):
            v = norm((cells[stage_idx].get("userEnteredValue", {}) or {}).get("stringValue"))
            if v:
                stages.append(v)

    total = len(stages)
    if total == 0:
        return {
            "total": 0,
            "distinctStages": 0,
            "byStage": [],
            "source": {"fileId": fileId, "sheetName": title, "columnIndex": stage_idx, "updatedAt": datetime.now(timezone.utc).isoformat()}
        }

    # Case-insensitive grouping with nice labels
    counts = Counter(s.lower() for s in stages)
    def label(k: str) -> str:
        # choose a readable label; title-case is fine for most HR stages
        return k.title()

    by_stage = [
        {
            "stage": label(k),
            "count": c,
            "percentage": round(c * 100.0 / total, 1),
        }
        for k, c in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    ]

    return {
        "total": total,
        "distinctStages": len(counts),
        "byStage": by_stage,
        "source": {
            "fileId": fileId,
            "sheetName": title,
            "columnIndex": stage_idx,
            "updatedAt": datetime.now(timezone.utc).isoformat(),
        },
    }



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


class CandidateUpload(BaseModel):
    candidateNames: List[str]
    departments: List[str]
    roles: List[str]
    hiringStages: List[str]
    files: List[str]  # maybe Drive file IDs or base64 if you want
    userEmails: Optional[List[str]] = None


@app.post("/candidates/uploadManually")
async def upload_candidates_json(request: Request, body: CandidateUpload):
    """
    Upload candidate CVs into:
    Departments/{Department}/{Role}/Hiring Pipeline/{Stage}
    Using JSON payload (instead of multipart/form-data).
    """

    # 🔹 Log raw request
    raw_body = await request.body()
    logger.info(f"📥 Raw request body: {raw_body.decode('utf-8', errors='ignore')}")

    # 🔹 Log parsed fields
    logger.info(
        "✅ Parsed CandidateUpload: candidateNames=%s, departments=%s, roles=%s, hiringStages=%s, files_count=%d",
        body.candidateNames,
        body.departments,
        body.roles,
        body.hiringStages,
        len(body.files) if body.files else 0
    )

    if body.userEmails and len(body.userEmails) > 0:
        subject = body.userEmails[0]  # use first provided user
    else:
        subject = _extract_subject_from_request(request)  # fallback to header/query/env
    _, drive, _ = get_clients(subject)

    # validate lengths
    if not (len(body.candidateNames) == len(body.roles) == len(body.departments) == len(body.hiringStages) == len(body.files)):
        logger.error("❌ Mismatched number of fields: names=%d roles=%d depts=%d stages=%d files=%d",
                     len(body.candidateNames), len(body.roles), len(body.departments), len(body.hiringStages), len(body.files))
        raise HTTPException(400, "Mismatched number of fields")

    processed = []

    for i in range(len(body.candidateNames)):
        cand_name = body.candidateNames[i]
        role = body.roles[i]
        dept = body.departments[i]
        stage = body.hiringStages[i]

        logger.info("➡️ Candidate %d: name=%s, dept=%s, role=%s, stage=%s", i+1, cand_name, dept, role, stage)

        # 🔹 Use the helper here
        file_ref = prepare_candidate_file(body.files[i])

        # 1. Locate department
        DEPARTMENTS_FOLDER_ID = os.environ.get("DEPARTMENTS_FOLDER_ID")
        query = (
            f"mimeType='application/vnd.google-apps.folder' "
            f"and trashed=false and name='{dept}' "
            f"and '{DEPARTMENTS_FOLDER_ID}' in parents"
        )
        logger.debug(f"🔎 Dept query: {query}")
        dept_results = drive.files().list(q=query, fields="files(id,name)", supportsAllDrives=True).execute()
        if not dept_results.get("files"):
            logger.error("❌ Department not found: %s", dept)
            raise HTTPException(404, f"Department '{dept}' not found")
        dept_id = dept_results["files"][0]["id"]

        # 2. Locate role inside department
        query = (
            f"mimeType='application/vnd.google-apps.folder' "
            f"and trashed=false and name='{role}' "
            f"and '{dept_id}' in parents"
        )
        logger.debug(f"🔎 Role query: {query}")
        role_results = drive.files().list(q=query, fields="files(id,name)", supportsAllDrives=True).execute()
        if not role_results.get("files"):
            logger.error("❌ Role not found: %s in %s", role, dept)
            raise HTTPException(404, f"Role '{role}' not found in Department '{dept}'")
        role_id = role_results.get("files")[0]["id"]

        # 3. Locate Hiring Pipeline inside role
        query = f"mimeType='application/vnd.google-apps.folder' and trashed=false and name='Hiring Pipeline' and '{role_id}' in parents"
        logger.debug(f"🔎 Pipeline query: {query}")
        pipeline = drive.files().list(q=query, fields="files(id,name)", supportsAllDrives=True).execute()
        if not pipeline.get("files"):
            logger.error("❌ Hiring Pipeline not found for role=%s dept=%s", role, dept)
            raise HTTPException(404, f"Hiring Pipeline not found for role '{role}' in Department '{dept}'")
        pipeline_id = pipeline["files"][0]["id"]

        # 4. Locate stage inside pipeline
        query = f"mimeType='application/vnd.google-apps.folder' and trashed=false and name='{stage}' and '{pipeline_id}' in parents"
        logger.debug(f"🔎 Stage query: {query}")
        stage_result = drive.files().list(q=query, fields="files(id,name)", supportsAllDrives=True).execute()
        if not stage_result.get("files"):
            logger.error("❌ Stage not found: %s under role=%s dept=%s", stage, role, dept)
            raise HTTPException(404, f"Stage '{stage}' not found under Hiring Pipeline for '{role}'")
        stage_id = stage_result["files"][0]["id"]

        # 5. Attach CV
        if file_ref.startswith("drive:"):
            uploaded_file_id = file_ref.replace("drive:", "")
            logger.info("📂 Using existing Drive file for %s: %s", cand_name, uploaded_file_id)
        else:
            decoded = base64.b64decode(file_ref)
            file_metadata = {"name": f"{cand_name} - CV.pdf", "parents": [stage_id]}
            media = MediaIoBaseUpload(io.BytesIO(decoded), mimetype="application/pdf")
            file_obj = drive.files().create(
                body=file_metadata,
                media_body=media,
                fields="id,parents",
                supportsAllDrives=True
            ).execute()
            uploaded_file_id = file_obj["id"]
            logger.info("📤 Uploaded CV for %s -> %s", cand_name, uploaded_file_id)

        processed.append({
            "candidateName": cand_name,
            "department": dept,
            "role": role,
            "stage": stage,
            "uploadedFileId": uploaded_file_id,
            "uploadedTo": stage_id
        })

    logger.info("✅ Finished upload. Processed candidates: %s", processed)
    return {"message": "Candidates uploaded successfully", "processed": processed}


@app.get("/whoami") # Verify who the api is acting as when user impersonation
def whoami(request: Request):
    require_api_key(request)
    subject = _extract_subject_from_request(request)
    _, drive, _ = get_clients(subject)
    about = drive.about().get(fields="user(emailAddress,displayName),storageQuota").execute()
    return {"subject_param": subject, "drive_user": about.get("user")}
