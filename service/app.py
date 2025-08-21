from fastapi import FastAPI, HTTPException, Request
from datetime import datetime, timezone
from typing import Optional, List
import os, json
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

API_KEY = os.environ.get("API_KEY")  # set in Railway "Variables"

app = FastAPI(title="Recruiting Sheet Insights")

# Below are helper functions
def get_clients():
    info = json.loads(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"])
    creds = Credentials.from_service_account_info(info, scopes=[
        "https://www.googleapis.com/auth/spreadsheets.readonly",
        "https://www.googleapis.com/auth/drive",
        "https://www.googleapis.com/auth/documents"  # add Docs scope
    ])
    sheets = build("sheets", "v4", credentials=creds)
    drive  = build("drive",  "v3", credentials=creds)
    docs   = build("docs",   "v1", credentials=creds)
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
    folder = drive.files().create(body=metadata, fields="id").execute()
    return folder["id"]

def upload_doc_to_drive(drive, folder_id: str, name: str, content: str) -> str:
    """Upload a text file as Google Doc into the given folder"""
    file_metadata = {
        "name": name,
        "mimeType": "application/vnd.google-apps.document",
        "parents": [folder_id]
    }
    media = {
        "mimeType": "text/plain",
        "body": content
    }
    file = drive.files().create(body=file_metadata, media_body=media, fields="id").execute()
    return file["id"]

def create_google_doc(docs, drive, folder_id: str, title: str, content: str) -> str:
    # Step 1: Create empty Google Doc
    doc = docs.documents().create(body={"title": title}).execute()
    doc_id = doc.get("documentId")

    # Step 2: Insert content at beginning
    requests = [
        {"insertText": {"location": {"index": 1}, "text": content}}
    ]
    docs.documents().batchUpdate(documentId=doc_id, body={"requests": requests}).execute()

    # Step 3: Move into the correct Drive folder
    file = drive.files().get(fileId=doc_id, fields="parents").execute()
    prev_parents = ",".join(file.get("parents"))
    drive.files().update(
        fileId=doc_id,
        addParents=folder_id,
        removeParents=prev_parents,
        fields="id, parents"
    ).execute()

    return doc_id

#End of Helper Functions

@app.get("/roles/unique")
def unique_roles(request: Request, fileId: str, sheetName: Optional[str] = None,
                 headerRow: int = 1, roleHeader: str = "Role"):
    require_api_key(request)
    sheets, drive = get_clients()

    meta = drive.files().get(fileId=fileId, fields="id,name,mimeType,modifiedTime").execute()
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

from collections import Counter
from typing import Optional, List
from fastapi import HTTPException, Request
from datetime import datetime, timezone

def norm(s: Optional[str]) -> str:
    return (s or "").strip()

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
    sheets, drive = get_clients()

    # Make sure it's a Google Sheet we can read
    meta = drive.files().get(fileId=fileId, fields="mimeType").execute()
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

@app.post("/positions/create")
def create_position(request: Request, name: str, department: str = "Software Engineering"):
    require_api_key(request)
    _, drive = get_clients()

    HIRING_FOLDER_ID = os.environ.get("HIRING_FOLDER_ID")
    if not HIRING_FOLDER_ID:
        raise HTTPException(500, "HIRING_FOLDER_ID env var not set")

    # Step 0: Ensure department folder exists
    query = (
        f"mimeType='application/vnd.google-apps.folder' "
        f"and trashed=false and name='{department}' "
        f"and '{HIRING_FOLDER_ID}' in parents"
    )
    results = drive.files().list(q=query, fields="files(id,name)").execute()
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
    results = drive.files().list(q=query, fields="files(id,name)").execute()
    if results.get("files"):
        return {
            "message": f"Role '{name}' already exists in {department}",
            "positionId": results["files"][0]["id"],
            "departmentFolderId": department_folder_id,
            "created": False
        }

    # Step 2: Create role folder
    position_id = create_folder(drive, name, department_folder_id)

    # Step 3: Subfolders (Role Description, Pre-screening, Candidate Stages)
    role_desc_id = create_folder(drive, "Role Description", position_id)
    prescreen_id = create_folder(drive, "Pre-screening questionnaire", position_id)
    stages_id = create_folder(drive, "Candidate Stages", position_id)

    # Step 4: Candidate stage subfolders per department
    department_stages = {
        "Software Engineering": [
            "Manually Applied", "TA Sourced", "TA pre-screening scheduled",
            "TA pre-screening completed", "1st technical interview scheduled",
            "1st technical interview completed", "2nd technical interview scheduled",
            "2nd technical interview completed", "Leadership-CEO chat",
            "Candidate dropped", "Candidate rejected", "Offer made",
            "Offer accepted", "Contract sent", "Contract signed off"
        ],
        "Marketing": [
            "Applied", "Recruiter screen", "Marketing skills assessment / portfolio review",
            "Hiring Manager interview", "Team interview", "Leadership interview",
            "Candidate dropped", "Candidate rejected", "Offer made",
            "Offer accepted", "Contract sent", "Contract signed off"
        ],
        "Sales": [
            "Applied", "Recruiter screen", "Hiring Manager interview",
            "Role play / sales pitch simulation", "Panel interview", "Leadership interview",
            "Candidate dropped", "Candidate rejected", "Offer made",
            "Offer accepted", "Contract sent", "Contract signed off"
        ],
        "Product & Project Management": [
            "Applied", "Recruiter screen", "Product case study / take-home assignment",
            "Hiring Manager interview", "Cross-functional panel interview",
            "Leadership interview", "Candidate dropped", "Candidate rejected",
            "Offer made", "Offer accepted", "Contract sent", "Contract signed off"
        ],
        "HR & Recruitment": [
            "Applied", "Recruiter screen", "HR knowledge/assessment",
            "Hiring Manager interview", "Panel interview", "Leadership interview",
            "Candidate dropped", "Candidate rejected", "Offer made",
            "Offer accepted", "Contract sent", "Contract signed off"
        ],
        "Finance & Accounting": [
            "Applied", "Recruiter screen", "Finance/Accounting technical test",
            "Hiring Manager interview", "Team/peer interview", "Leadership interview",
            "Candidate dropped", "Candidate rejected", "Offer made",
            "Offer accepted", "Contract sent", "Contract signed off"
        ]
    }

    stages = department_stages.get(department, department_stages["Software Engineering"])
    for stage in stages:
        create_folder(drive, stage, stages_id)

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
    _, drive = get_clients()

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
        results = drive.files().list(q=query, fields="files(id,name)").execute()
        items = results.get("files", [])
        if not items:
            return {"roles": [], "department": department, "exists": False}
        parent_id = items[0]["id"]

    # List folders under parent (roles or departments)
    query = f"mimeType='application/vnd.google-apps.folder' and trashed=false and '{parent_id}' in parents"
    results = drive.files().list(q=query, fields="files(id,name)").execute()
    roles = [{"id": f["id"], "name": f["name"]} for f in results.get("files", [])]

    return {
        "department": department or "Hiring",
        "roles": roles,
        "exists": True
    }

@app.post("/positions/createJD")
def create_jd(request: Request, positionId: str, roleName: str):
    require_api_key(request)
    _, drive, docs = get_clients()

    content = f"""Job Description for {roleName}

Responsibilities:
- Define and execute {roleName} strategies
- Collaborate with cross-functional teams
- Deliver measurable outcomes

Qualifications:
- Proven experience in {roleName}
- Strong analytical, communication, and leadership skills
"""
    file_id = create_google_doc(docs, drive, positionId, f"JD - {roleName}", content)
    return {
        "message": f"JD created for {roleName}",
        "fileId": file_id,
        "docLink": f"https://docs.google.com/document/d/{file_id}/edit"
    }

@app.post("/positions/createScreeningTemplate")
def create_screening(request: Request, positionId: str, roleName: str):
    require_api_key(request)
    _, drive, docs = get_clients()

    content = f"""Screening Template for {roleName}

Candidate Name:
Date:

Questions:
1. Why are you interested in this role?
2. Describe your relevant skills and experience.
3. What achievements best demonstrate your impact?

Evaluator Notes:
"""
    file_id = create_google_doc(docs, drive, positionId, f"Screening Template - {roleName}", content)
    return {
        "message": f"Screening template created for {roleName}",
        "fileId": file_id,
        "docLink": f"https://docs.google.com/document/d/{file_id}/edit"
    }

@app.post("/positions/createScoringModel")
def create_scoring(request: Request, positionId: str, roleName: str):
    require_api_key(request)
    _, drive, docs = get_clients()

    content = f"""Scoring Rubric for {roleName}

Criteria (1-5 each):
- Role Expertise
- Problem Solving
- Communication
- Culture Fit

Total Score: ___ / 20
"""
    file_id = create_google_doc(docs, drive, positionId, f"Scoring Rubric - {roleName}", content)
    return {
        "message": f"Scoring rubric created for {roleName}",
        "fileId": file_id,
        "docLink": f"https://docs.google.com/document/d/{file_id}/edit"
    }

