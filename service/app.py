from fastapi import FastAPI, HTTPException, Request
from datetime import datetime, timezone
from typing import Optional, List
import os, json
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

API_KEY = os.environ.get("API_KEY")  # set in Railway "Variables"

app = FastAPI(title="Recruiting Sheet Insights")

#def get_clients():
 #   info = json.loads(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"])  # paste SA JSON into this env var
  #  creds = Credentials.from_service_account_info(info, scopes=[
   #     "https://www.googleapis.com/auth/spreadsheets.readonly",
    #    "https://www.googleapis.com/auth/drive.readonly",
    #])
    #sheets = build("sheets", "v4", credentials=creds)
    #drive  = build("drive",  "v3", credentials=creds)
    #return sheets, drive

def get_clients():
    info = json.loads(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"]) # paste SA JSON into this env var
    creds = Credentials.from_service_account_info(info, scopes=[
        "https://www.googleapis.com/auth/spreadsheets.readonly",
        "https://www.googleapis.com/auth/drive"  # full drive access
    ])
    sheets = build("sheets", "v4", credentials=creds)
    drive  = build("drive",  "v3", credentials=creds)
    return sheets, drive


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
    require_api_key(request)  # API key check
    _, drive = get_clients()

    HIRING_FOLDER_ID = os.environ.get("HIRING_FOLDER_ID")
    if not HIRING_FOLDER_ID:
        raise HTTPException(500, "HIRING_FOLDER_ID env var not set")

    # Step 0: Ensure department folder exists under Hiring
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

    # Step 1: Create position folder inside department
    position_id = create_folder(drive, name, department_folder_id)

    # Step 2: Create base subfolders
    role_desc_id = create_folder(drive, "Role Description", position_id)
    prescreen_id = create_folder(drive, "Pre-screening questionnaire", position_id)
    stages_id    = create_folder(drive, "Candidate Stages", position_id)

    # Step 3: Candidate stage subfolders per department
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
        "message": f"Folder structure for '{name}' created successfully in {department}",
        "positionId": position_id,
        "departmentFolderId": department_folder_id
    }


