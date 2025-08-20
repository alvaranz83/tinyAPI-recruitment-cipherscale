from fastapi import FastAPI, HTTPException, Request
from datetime import datetime, timezone
from typing import Optional, List
import os, json
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

API_KEY = os.environ.get("API_KEY")  # set in Railway "Variables"

app = FastAPI(title="Recruiting Sheet Insights")

def get_clients():
    info = json.loads(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"])  # paste SA JSON into this env var
    creds = Credentials.from_service_account_info(info, scopes=[
        "https://www.googleapis.com/auth/spreadsheets.readonly",
        "https://www.googleapis.com/auth/drive.readonly",
    ])
    sheets = build("sheets", "v4", credentials=creds)
    drive  = build("drive",  "v3", credentials=creds)
    return sheets, drive

def require_api_key(req: Request):
    if not API_KEY or req.headers.get("x-api-key") != API_KEY:
        raise HTTPException(403, "Forbidden")

def norm(s: Optional[str]) -> str:
    return (s or "").strip()

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
