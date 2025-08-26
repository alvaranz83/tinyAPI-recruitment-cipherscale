from fastapi import FastAPI, HTTPException, Request
from datetime import datetime, timezone
from typing import Optional, List
import os, json
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from google.auth.exceptions import RefreshError # For user impersonation
from pydantic import BaseModel
from collections import Counter
import textwrap


IMPERSONATE_HEADER = "x-user-email"  # or "x-impersonate-user" # Choose a header name you’ll set from your app / gateway
DEFAULT_IMPERSONATION_SUBJECT = os.environ.get("DEFAULT_IMPERSONATION_SUBJECT")  # optional fallback

API_KEY = os.environ.get("API_KEY")  # set in Railway "Variables"

app = FastAPI(title="Recruiting Sheet Insights")

# Below are helper functions

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
    # Step 1: Create the Google Doc in the target folder
    file_metadata = {
        "name": title,
        "mimeType": "application/vnd.google-apps.document",
        "parents": [folder_id]
    }
    file = drive.files().create(body=file_metadata, fields="id", supportsAllDrives=True).execute()
    doc_id = file["id"]

    # Step 2: Fetch doc length (to clear)
    doc = docs.documents().get(documentId=doc_id).execute()
    doc_length = doc.get("body").get("content")[-1]["endIndex"]

    # Step 3: Clear existing text
    requests = []
    if doc_length > 2:
        requests.append({
            "deleteContentRange": {
                "range": {"startIndex": 1, "endIndex": doc_length - 1}
            }
        })

    # --- IMPORTANT: normalize/dedent the template before processing ---
    norm = textwrap.dedent(content).strip("\n")
    raw_lines = norm.splitlines()

    insert_index = 1
    para_ranges = []   # (start, end, kind)
    list_groups = []   # (group_start, group_end)
    in_list = False
    group_start = None

    def insert_line(txt: str):
        nonlocal insert_index, requests
        if not txt.endswith("\n"):
            txt += "\n"
        start = insert_index
        end = start + len(txt)
        requests.append({"insertText": {"location": {"index": start}, "text": txt}})
        insert_index = end
        return start, end

    for i, line in enumerate(raw_lines):
        # Skip truly blank lines but close any open list
        if not line.strip():
            if in_list:
                list_groups.append((group_start, insert_index))
                in_list = False
                group_start = None
            continue

        trimmed = line.strip()
        is_h1 = (i == 0)
        is_h2 = trimmed.endswith(":")
        is_bullet = trimmed.startswith("- ")

        # For bullets, strip the "- " before inserting to avoid "• - Item"
        text_to_insert = trimmed[2:] if is_bullet else trimmed
        start, end = insert_line(text_to_insert)

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

            if is_h1:
                para_ranges.append((start, end, "H1"))
            elif is_h2:
                para_ranges.append((start, end, "H2"))
            else:
                para_ranges.append((start, end, "NORMAL"))

    if in_list:
        list_groups.append((group_start, insert_index))

    # Styles
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
        elif kind == "NORMAL":
            requests.append({
                "updateParagraphStyle": {
                    "range": {"startIndex": start, "endIndex": end},
                    "paragraphStyle": {"namedStyleType": "NORMAL_TEXT"},
                    "fields": "namedStyleType"
                }
            })

    # Apply bullets once per contiguous group
    for gs, ge in list_groups:
        requests.append({
            "createParagraphBullets": {
                "range": {"startIndex": gs, "endIndex": ge},
                "bulletPreset": "BULLET_DISC_CIRCLE_SQUARE"
            }
        })

    # Step 5: Execute
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

    HIRING_FOLDER_ID = os.environ.get("HIRING_FOLDER_ID")
    if not HIRING_FOLDER_ID:
        raise HTTPException(500, "HIRING_FOLDER_ID env var not set")

    # Step 0: Ensure department folder exists
    query = (
        f"mimeType='application/vnd.google-apps.folder' "
        f"and trashed=false and name='{department}' "
        f"and '{HIRING_FOLDER_ID}' in parents"
    )
    results = drive.files().list(q=query, fields="files(id,name)", includeItemsFromAllDrives=True, supportsAllDrives=True).execute()
    items = results.get("files", [])

    if items:
        department_folder_id = items[0]["id"]
    else:
        department_folder_id = create_folder(drive, department, HIRING_FOLDER_ID)

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
        "created": True,
        "nextAction": "Would you like me to also create the Job Description (JD), TA Screening Template, and Interview Scoring Rubric for this role?"
    }


@app.get("/positions/list") # End point that understands what department folders already exist under Hiring folder
def list_positions(request: Request, department: Optional[str] = None):
    require_api_key(request)
    subject = _extract_subject_from_request(request)
    _, drive, _ = get_clients(subject)

    HIRING_FOLDER_ID = os.environ.get("HIRING_FOLDER_ID")
    if not HIRING_FOLDER_ID:
        raise HTTPException(500, "HIRING_FOLDER_ID env var not set")

    # If department specified, check inside it
    parent_id = HIRING_FOLDER_ID
    if department:
        query = (
            f"mimeType='application/vnd.google-apps.folder' "
            f"and trashed=false and name='{department}' "
            f"and '{HIRING_FOLDER_ID}' in parents"
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



@app.get("/whoami") # Verify who the api is acting as when user impersonation
def whoami(request: Request):
    require_api_key(request)
    subject = _extract_subject_from_request(request)
    _, drive, _ = get_clients(subject)
    about = drive.about().get(fields="user(emailAddress,displayName),storageQuota").execute()
    return {"subject_param": subject, "drive_user": about.get("user")}
