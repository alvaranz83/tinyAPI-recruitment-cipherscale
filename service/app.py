from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request, Query, Body, Path, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Iterable, Tuple, Union
import os, json, textwrap, re, uuid, logging, io, httpx, time, random, requests, subprocess, asyncio
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload 
from google.auth.exceptions import RefreshError # For user impersonation
from pydantic import BaseModel, Field, field_validator
from difflib import SequenceMatcher
from openai import AsyncOpenAI
from googleapiclient.errors import HttpError
from google.oauth2 import service_account
from sqlalchemy import create_engine, MetaData, insert
from databases import Database
from .db import database
from uuid import UUID
import logging
from pydantic.config import ConfigDict  # Pydantic v2
import urllib.parse, httpx
from logging.handlers import RotatingFileHandler
from utils.cv_text_extractor import fetch_and_extract_cv_text


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


# =======================
# SLACK Env Variables
# =======================

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_API_URL = "https://slack.com/api/chat.postMessage"


# ========================
# OpenAI Env Variables
# =========================

OPENAI_URL = os.getenv("OPENAI_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY")) # Initialize OpenAI once at top-level


# ========================
# Recruitee Env VAriables
# ==========================

RECRUITEE_COMPANY_ID = os.getenv("RECRUITEE_COMPANY_ID")  # e.g. "123456" or subdomain
RECRUITEE_API_TOKEN = os.getenv("RECRUITEE_API_TOKEN")   # Bearer token
RECRUITEE_BASE = os.getenv("RECRUITEE_BASE") # Recruitee Company name base url
RECRUITEE_API_URL = os.getenv("RECRUITEE_API_URL") # Recruitee API URL

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


# ============================
# Helper Functions
# ============================


# Helpers for evaluating canddiates that apply and adding Scoring on Recriutee

async def call_recruitee_candidate(candidate_id: int, company_id: int):
    """Fetch candidate details from Recruitee."""
    url = f"{RECRUITEE_API_URL}/c/{company_id}/candidates/{candidate_id}"
    headers = {"Authorization": f"Bearer {RECRUITEE_API_TOKEN}"}
    async with httpx.AsyncClient() as client:
        r = await client.get(url, headers=headers, timeout=30)
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=f"Recruitee API error: {r.text}")
    return r.json()

async def call_openai_evaluation(payload: dict) -> dict:
    """Send candidate/job data to OpenAI LLM and return scoring + explanation."""
    prompt = f"""
You are an expert recruiter scoring candidates (A–E). 
Use this grading scale:

🔹 A — Outstanding Fit (Top 5–10%) Profile: CV and LinkedIn show direct, high-level alignment with all core job requirements. Demonstrated success in similar roles or industries (e.g., proven performance in relevant companies or projects). Possesses all mandatory skills and most optional/nice-to-have ones. Evidence of leadership, innovation, or measurable impact (e.g., "reduced costs by 30%", "scaled API to millions of users"). Cultural and communication fit appears excellent based on writing style, tone, and profile. Action: Fast-track to TA/HR Interview or 1st Technical Interview.
🔹 B — Strong Fit (Top 20–30%) Profile: Meets all must-have qualifications and most preferred ones. Has relevant experience, even if from smaller or less-known organizations. Some areas (e.g., tooling or domain) may require light training. Shows solid problem-solving and learning potential. LinkedIn reflects professional growth and continuous learning. Action: Advance to interview. High potential for success. 
🔹 C — Moderate Fit (Average, 40–50%) Profile: Meets some core requirements but lacks depth or years of experience. Career path may be adjacent (e.g., backend developer applying for DevOps). Missing 1–2 important technical skills or certifications. Limited evidence of impact or ownership. LinkedIn profile somewhat generic or incomplete. Action: Keep in pipeline as “potential”; interview only if pipeline is thin or role flexibility exists. 
🔹 D — Weak Fit (Bottom 20–30%) Profile: Lacks several mandatory skills or relevant experience. Experience unrelated to the job’s technical or functional domain. Career progression unclear or inconsistent with role expectations. CV generic, lacking outcomes or clarity. LinkedIn incomplete or outdated. Action: Disqualify or park in “Talent Pool”. 
🔹 E — Poor Fit / Reject (Bottom 10%) Profile: No alignment with the job’s scope or industry. Missing essential qualifications, education, or technical baseline. CV poorly written or indicates career mismatch. No evidence of adaptability or learning potential. Possibly AI-generated or spam submissions. Action: Reject.

Evaluate the candidate below against the job description. 
Return ONLY JSON with keys: "Scoring" and "Score_Explanation".

Candidate data:
{json.dumps(payload, indent=2)}
    """
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post(OPENAI_URL, headers=headers, json=body, timeout=60)

    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {resp.text}")

    data = resp.json()
    text = data["choices"][0]["message"]["content"]

    # ✅ Fix: Strip Markdown code fences if present
    cleaned_text = text.strip()
    if cleaned_text.startswith("```"):
        cleaned_text = cleaned_text.strip("`")
        cleaned_text = re.sub(r"^json\s*", "", cleaned_text).strip()

    try:
        result = json.loads(cleaned_text)
        # ✅ Log the parsed result neatly
        logger.info(
            "🤖 OpenAI Scoring Response:\n%s",
            json.dumps(result, indent=2, ensure_ascii=False),
        )
    except Exception:
        logger.warning("⚠️ OpenAI returned non-JSON text:\n%s", text)
        result = {"Scoring": "N/A", "Score_Explanation": text}

    return result


async def update_recruitee_custom_fields(company_id: int, candidate_id: int, score: str, explanation: str):
    """Create custom Recruitee fields Contact Priority (AI-GPT) and Explanation."""
    base_url = f"{RECRUITEE_API_URL}/c/{company_id}/custom_fields/candidates/{candidate_id}/fields"
    headers = {"Authorization": f"Bearer {RECRUITEE_API_TOKEN}", "Content-Type": "application/json"}
    async with httpx.AsyncClient() as client:
        # 1️⃣ Contact Priority (AI-GPT)
        field1 = {
            "field": {
                "name": "Contact Priority (AI-GPT)",
                "values": [{"text": score}],
                "kind": "single_line",
            }
        }
        await client.post(base_url, headers=headers, json=field1, timeout=30)

        # 2️⃣ Contact Priority Explanation
        field2 = {
            "field": {
                "name": "Contact Priority Explanation",
                "values": [{"text": explanation}],
                "kind": "multi_line",
            }
        }
        await client.post(base_url, headers=headers, json=field2, timeout=30)


async def get_existing_custom_fields(company_id: int, candidate_id: int) -> list:
    """Check existing Recruitee fields to avoid duplicates."""
    url = f"{RECRUITEE_API_URL}/c/{company_id}/candidates/{candidate_id}/custom_fields"
    headers = {"Authorization": f"Bearer {RECRUITEE_API_TOKEN}"}
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url, headers=headers)
    if r.status_code != 200:
        logger.warning(f"⚠️ Failed to get existing fields for candidate {candidate_id}: {r.status_code}")
        return []
    return [f["name"] for f in r.json().get("custom_fields", []) if "name" in f]



# Helpers for the LinkedIn Profile DOM Scrapper

def _get_attr(node, key, default=None):
    return (node.get("attributes") or {}).get(key, default)

def _classlist(node):
    cls = _get_attr(node, "class", "") or ""
    # classes may be space-separated string
    return set(c for c in re.split(r"\s+", cls.strip()) if c)

def _class_contains(node, needle):
    return any(needle in c for c in _classlist(node))

def _tag_is(node, tag):
    return isinstance(node, dict) and node.get("tag", "").upper() == tag.upper()

def _iter_children(node):
    return (node.get("children") or []) if isinstance(node, dict) else []

def _walk(node):
    if isinstance(node, dict):
        yield node
        for ch in _iter_children(node):
            yield from _walk(ch)
    elif isinstance(node, list):
        for item in node:
            yield from _walk(item)

def _find_div_by_id(dom, id_value):
    for n in _walk(dom):
        if _tag_is(n, "DIV") and (_get_attr(n, "id") or "").strip().lower() == id_value.strip().lower():
            return n
    return None

def _collect_span_text(node, into):
    # Collect plain text from span nodes under node
    if not isinstance(node, (dict, list)):
        return
    if isinstance(node, dict):
        if _tag_is(node, "SPAN"):
            txt = node.get("text")
            if isinstance(txt, str) and txt.strip():
                into.append(txt.strip())
        for ch in _iter_children(node):
            _collect_span_text(ch, into)
    else:
        for item in node:
            _collect_span_text(item, into)

def _collect_text_under(node):
    buf = []
    _collect_span_text(node, buf)
    # collapse whitespace & dedupe short repeats
    text = " ".join(buf)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None

def _find_descendants_by_class_contains(node, substr):
    for n in _walk(node):
        if _class_contains(n, substr):
            yield n

def _find_items(node):
    # Find list items marked with class containing 'artdeco-list__item'
    for n in _walk(node):
        if _class_contains(n, "artdeco-list__item"):
            yield n

def _extract_header_and_desc(item_node):
    # Header block: class includes 'display-flex flex-row justify-space-between'
    header = None
    for n in _walk(item_node):
        cls = " ".join(_classlist(n))
        if all(part in cls for part in ["display-flex", "flex-row", "justify-space-between"]):
            header = n
            break
    header_text = _collect_text_under(header) if header else None

    # Description block: class contains 'pvs-entity__sub-component' or 'pvs-entity__sub-components'
    desc = None
    for n in _walk(item_node):
        if _class_contains(n, "pvs-entity__sub-component") or _class_contains(n, "pvs-entity__sub-components"):
            desc = n
            break
    desc_text = _collect_text_under(desc) if desc else None

    return header_text, desc_text

# Main Extfractor of LinkedIn info

def extract_linkedin_sections(dom):
    """
    dom: your LinkedIn DOM JSON (dict/list form coming from the scraper)
    returns a dict with the requested sections
    """
    if dom is None:
        return {}

    out = {}

    # --- Simple single-block sections (About, Services, Featured) ---
    about_div = _find_div_by_id(dom, "about")
    if about_div:
        out["About"] = _collect_text_under(about_div)

    services_div = _find_div_by_id(dom, "services")
    if services_div:
        out["Services"] = _collect_text_under(services_div)

    featured_div = _find_div_by_id(dom, "featured")
    if featured_div:
        out["Featured"] = _collect_text_under(featured_div)

    # --- Experience (div id="Experience") ---
    exp_div = _find_div_by_id(dom, "Experience")
    experiences = []
    if exp_div:
        for item in _find_items(exp_div):
            header_text, desc_text = _extract_header_and_desc(item)
            experiences.append({
                "Header": header_text,       # company, role, work type, dates
                "Description": desc_text,    # role description
            })
    if experiences:
        out["Experiences"] = experiences

    # --- Education (div id="education") ---
    edu_div = _find_div_by_id(dom, "education")
    educations = []
    if edu_div:
        for item in _find_items(edu_div):
            header_text, desc_text = _extract_header_and_desc(item)
            educations.append({
                "Header": header_text,        # institution, degree/type, dates
                "Description": desc_text,     # details
            })
    if educations:
        out["Education"] = educations

    # --- Licenses & Certifications (div id="licenses_and_certifications") ---
    lic_div = _find_div_by_id(dom, "licenses_and_certifications")
    licenses = []
    if lic_div:
        for item in _find_items(lic_div):
            # certification name/company/issued/id noted in attributes around image field;
            # header still contains useful text summary.
            header_text, desc_text = _extract_header_and_desc(item)
            # Look for data-field="entity_image_licenses_and_certifications" (optional extra)
            meta_text = None
            for n in _walk(item):
                if (_get_attr(n, "data-field") or "") == "entity_image_licenses_and_certifications":
                    meta_text = _collect_text_under(n)
                    break
            licenses.append({
                "Header": header_text,
                "Meta": meta_text,            # issuer, issued date, cert id (if surfaced)
                "Description": desc_text,
            })
    if licenses:
        out["Licenses_and_Certifications"] = licenses

    # --- Projects (div id="projects") ---
    proj_div = _find_div_by_id(dom, "projects")
    projects = []
    if proj_div:
        for item in _find_items(proj_div):
            header_text, desc_text = _extract_header_and_desc(item)
            projects.append({
                "Header": header_text,        # project name + date
                "Description": desc_text,     # project description
            })
    if projects:
        out["Projects"] = projects

    # --- Volunteering (div id="volunteering_experience") ---
    vol_div = _find_div_by_id(dom, "volunteering_experience")
    volunteering = []
    if vol_div:
        for item in _find_items(vol_div):
            header_text, desc_text = _extract_header_and_desc(item)
            volunteering.append({
                "Header": header_text,        # role, org, dates
                "Description": desc_text,     # details
            })
    if volunteering:
        out["Volunteering"] = volunteering

    # --- Skills (div id="skills") ---
    skills_div = _find_div_by_id(dom, "skills")
    skills = []
    if skills_div:
        for item in _find_items(skills_div):
            header_text, desc_text = _extract_header_and_desc(item)
            skills.append({
                "Skill": header_text,         # skill name
                "Applied_In": desc_text,      # where it was used (if provided)
            })
    if skills:
        out["Skills"] = skills

    # Trim empty strings/lists
    for k, v in list(out.items()):
        if v in ("", None) or (isinstance(v, (list, dict)) and not v):
            out.pop(k, None)

    return out

# End of LinkedIn Helper Functions ----

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


# =========================
# Recruitee Helper Function 
# =========================

def _rb_headers() -> dict:
    if not RECRUITEE_API_TOKEN:
        raise HTTPException(500, "RECRUITEE_API_TOKEN not configured")
    return {
        "accept": "application/json",
        "authorization": f"Bearer {RECRUITEE_API_TOKEN}",
    }


def _bool_to_str(b: bool | None) -> str | None:
    if b is None:
        return None
    # Recruitee expects 'true' or '1' (either is fine); use 'true'/'false'
    return "true" if b else "false"


def _iso_no_microseconds(dt: datetime) -> str:
    # API accepts 'yyyy-mm-ddThh:mm:ss' (seconds optional). Use seconds, no micros, always UTC if tz-aware.
    if dt.tzinfo:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt.replace(microsecond=0).isoformat()

def require_api_key(request: Request):
    if "x-api-key" not in request.headers:
        raise HTTPException(401, "Missing x-api-key header")


def _unix_timestamp(dt: datetime) -> int:
    return int(dt.timestamp())


# ==========================
# End of recruitee helper functions
# =========================


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
    require_api_key(request)
    subject = body.userEmail or _extract_subject_from_request(request)
    _, drive, _ = get_clients(subject)

    DEPARTMENTS_FOLDER_ID = os.environ.get("DEPARTMENTS_FOLDER_ID")
    if not DEPARTMENTS_FOLDER_ID:
        raise HTTPException(500, "DEPARTMENTS_FOLDER_ID env var not set")

    decisions: List[MoveDecision] = []

    for move in body.moves:
        try:
            # ✅ Resolve the role first
            score, match, role_display = _resolve_best_role_by_name(drive, DEPARTMENTS_FOLDER_ID, move.roleQuery)
            if not match or score < _ROLE_SCORE_THRESHOLD:
                raise Exception(f"Could not resolve role '{move.roleQuery}' (score={score})")
            role_id = match["id"]
            role_info = match

            # ✅ Build stage + file index
            stages, file_index = _build_candidate_index(drive, role_id)

            # ✅ Resolve stage & candidate
            stage_score, stage_match, stage_display = _resolve_best_stage(move.stageQuery, stages)
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

            if (
                cand_match
                and stage_match
                and cand_score >= _NAME_SCORE_THRESHOLD
                and stage_score >= _STAGE_SCORE_THRESHOLD
                and cand_match["stageId"] != stage_match["id"]
            ):
                if not body.dryRun:
                    _move_file_between_stages(drive, cand_match["id"], cand_match["stageId"], stage_match["id"])
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
        message="Processed moveByPrompt (fixed resolver)",
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


# ----------------------------
# Recruitee Webhook Models (final, tolerant)
# ----------------------------

class RecruiteeBaseModel(BaseModel):
    model_config = ConfigDict(extra="allow")  # ✅ v2 style

# --- Submodels ---
class Stage(RecruiteeBaseModel):
    id: int | None = None
    name: str | None = None
    category: str | None = None

class Department(RecruiteeBaseModel):
    id: int | None = None
    name: str | None = None

class Location(RecruiteeBaseModel):
    id: int | None = None
    country_code: str | None = None
    state_code: str | None = None
    full_address: str | None = None

class Tag(RecruiteeBaseModel):
    id: int | None = None
    name: str | None = None

class Offer(RecruiteeBaseModel):
    id: int | None = None
    title: str | None = None
    kind: str | None = None
    slug: str | None = None
    department: Department | None = None
    locations: List[Location] | None = None
    created_at: str | None = None
    updated_at: str | None = None
    tags: List[Tag] | None = None
    status: str | None = None

class Candidate(RecruiteeBaseModel):
    id: int | None = None
    name: str | None = None
    emails: List[str] | None = None
    phones: List[str] | None = None
    photo_thumb_url: str | None = None
    referrer: str | None = None
    source: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    has_avatar: bool | None = None
    initials: str | None = None

class CandidateMovedDetails(RecruiteeBaseModel):
    from_stage: Stage | None = None
    to_stage: Stage | None = None
    disqualify_reason: Dict[str, Any] | None = None

class Company(RecruiteeBaseModel):
    id: int | None = None
    name: str | None = None

class RecruiteeWebhookPayload(RecruiteeBaseModel):
    attempt_count: int | None = None
    created_at: str | None = None
    candidate: Candidate | None = None
    details: CandidateMovedDetails | None = None
    offers: List[Offer] | None = None          # plural
    offer: Offer | None = None                 # singular (other events)
    company: Company | None = None
    placement_locations: List[Location] | None = None
    talent_pools: List[Any] | None = None      # ← your events include this

class RecruiteeWebhookAttributes(RecruiteeBaseModel):
    attempt_count: int | None = None
    created_at: str | None = None
    id: int | None = None
    level: str | None = None
    event_type: str | None = None
    event_subtype: str | None = None
    payload: RecruiteeWebhookPayload | None = None
    test: bool | None = None

class RecruiteeWebhookRequest(BaseModel):
    # Wrapped shape
    message: str | None = None
    attributes: RecruiteeWebhookAttributes | None = None

    # Flat shape (optional & harmless)
    id: int | None = None
    event_type: str | None = None
    event_subtype: str | None = None
    attempt_count: int | None = None
    created_at: str | None = None
    level: str | None = None
    payload: RecruiteeWebhookPayload | None = None

    tags: Dict[str, Any] | None = None
    timestamp: str | None = None

    model_config = ConfigDict(extra="allow")


# -------------------------------
# Endpoint
# -------------------------------

@app.post("/candidates/moveByRecruiteeWebhook")
async def move_by_recruitee_webhook(request: Request):
    """
    Handles Recruitee 'candidate_moved' webhook events.
    ✅ Logs raw payload
    ✅ Uses full Pydantic validation for Recruitee JSON
    ✅ Handles test events gracefully
    """

    # Step 1️⃣ — Read and log the raw body BEFORE parsing
    try:
        raw_body = await request.body()
        raw_text = raw_body.decode("utf-8")
        logger.info("🔍 RAW WEBHOOK PAYLOAD (Recruitee):\n%s", raw_text)
    except Exception as e:
        logger.exception("❌ Failed to read webhook body: %s", e)
        raise HTTPException(status_code=400, detail=f"Cannot read webhook body: {e}")

    # Step 2️⃣ — Parse JSON manually first to catch bad payloads early
    try:
        json_data = json.loads(raw_text)
    except Exception as e:
        logger.exception("❌ Invalid JSON format: %s", e)
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    # Step 3️⃣ — Validate against Pydantic model
    try:
        body = RecruiteeWebhookRequest(**json_data)
        logger.info("✅ Webhook validated successfully via Pydantic model.")
    except Exception as e:
        logger.exception("❌ Validation error while parsing webhook payload: %s", e)
        raise HTTPException(status_code=422, detail=f"Webhook validation failed: {e}")
    
    # Step 4️⃣ — Safely extract attributes, whether dict or model
    attrs = body.attributes or {}
    
    # Normalize for both dict and BaseModel cases
    if isinstance(attrs, dict):
        event_type = attrs.get("event_type")
        event_subtype = attrs.get("event_subtype")
        test_flag = attrs.get("test", False)
    else:
        event_type = getattr(attrs, "event_type", None)
        event_subtype = getattr(attrs, "event_subtype", None)
        test_flag = getattr(attrs, "test", False)
    
    logger.info("📦 Event type: %s | Subtype: %s | Test: %s", event_type, event_subtype, test_flag)
    
    # Step 5️⃣ — Handle test webhooks gracefully
    if test_flag:
        logger.info("🧪 Test webhook received — responding 200 OK.")
        return {"message": "Recruitee webhook verified successfully (test event)."}


    # Step 6️⃣ — Extract key information from validated model
    try:
        payload = None
    
        # Handle dict-based attributes (test or flexible payloads)
        if isinstance(attrs, dict):
            payload = attrs.get("payload")
    
        # Handle Pydantic BaseModel attributes
        elif hasattr(attrs, "payload"):
            payload = getattr(attrs, "payload")
    
        if not payload:
            logger.info("⚙️ No payload found in webhook — likely a test or diagnostic event.")
            return {"message": "No actionable payload found (test or minimal webhook)."}
    
        # Extract inner elements safely
        candidate = getattr(payload, "candidate", None)
        offer = getattr(payload, "offer", None)
        details = getattr(payload, "details", None)
    
        candidate_name = getattr(candidate, "name", "Unknown")
        role_name = getattr(offer, "title", None) or getattr(offer, "slug", "Unknown Role")
        from_stage = getattr(getattr(details, "from_stage", None), "name", "Unknown")
        to_stage = getattr(getattr(details, "to_stage", None), "name", "Unknown")
    
        logger.info(
            "🎯 Candidate '%s' moving from '%s' → '%s' in role '%s'",
            candidate_name, from_stage, to_stage, role_name
        )

    except Exception as e:
        logger.exception("❌ Failed to extract candidate movement details: %s", e)
        raise HTTPException(status_code=400, detail=f"Invalid payload structure: {e}")


    # Step 7️⃣ — Execute your main logic (role/stage resolution + Drive update)
    try:
        require_api_key(request)
        subject = _extract_subject_from_request(request)
        _, drive, _ = get_clients(subject)

        DEPARTMENTS_FOLDER_ID = os.environ.get("DEPARTMENTS_FOLDER_ID")
        if not DEPARTMENTS_FOLDER_ID:
            raise HTTPException(500, "DEPARTMENTS_FOLDER_ID not set")

        # --- Role resolution ---
        role_score, role_match, _ = _resolve_best_role_by_name(drive, DEPARTMENTS_FOLDER_ID, role_name)
        if not role_match or role_score < _ROLE_SCORE_THRESHOLD:
            raise Exception(f"Could not resolve role '{role_name}' (score={role_score})")
        role_id = role_match["id"]

        # --- Candidate & stage resolution ---
        stages, file_index = _build_candidate_index(drive, role_id)
        cand_score, cand_match, _ = _resolve_best_candidate_file(candidate_name, file_index)
        stage_score, stage_match, _ = _resolve_best_stage(to_stage, stages)

        if not cand_match or not stage_match:
            raise Exception("Candidate or stage not found")

        moved = False
        if cand_match["stageId"] != stage_match["id"]:
            _move_file_between_stages(
                drive, cand_match["id"], cand_match["stageId"], stage_match["id"]
            )
            moved = True

            await database.execute(
                """
                UPDATE candidates
                SET current_stage_name = :new_stage
                WHERE cv_name = :cv_name OR full_name = :full_name
                """,
                {
                    "new_stage": stage_match["name"],
                    "cv_name": cand_match["name"],
                    "full_name": candidate_name,
                },
            )

        logger.info("✅ Candidate '%s' successfully moved to '%s'", candidate_name, to_stage)

        return {
            "message": "Candidate moved successfully",
            "candidate_name": candidate_name,
            "role_name": role_name,
            "from_stage": from_stage,
            "to_stage": to_stage,
            "moved": moved,
        }

    except Exception as e:
        logger.exception("❌ Error while processing webhook event: %s", e)
        return {
            "message": "Failed to process Recruitee webhook",
            "error": str(e),
        }

# -----------------------------------------
# /candidates/newCandidateRecruiteeWebhook
# -----------------------------------------

@app.post("/candidates/newCandidateRecruiteeWebhook")
async def new_candidate_recruitee_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    Fast webhook for Recruitee → ACK immediately, then process in background.
    """
    try:
        payload = await request.json()
    except Exception as e:
        logger.error(f"❌ Invalid JSON body: {e}")
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    # ✅ Instant response to Recruitee
    background_tasks.add_task(process_new_candidate, payload)
    return {"status": "accepted"}  # Recruitee sees 200 OK in <100 ms

# BACKGROUND TASK — FULL LOGIC

async def process_new_candidate(payload: dict):
    """Runs the full candidate evaluation workflow asynchronously."""
    try:
        logger.info("🚀 Starting background candidate processing...")

        # Normalize payload structure
        if "attributes" not in payload:
            payload = {"attributes": {"payload": payload.get("payload", {}), **payload}}

        attrs = payload["attributes"]
        data = attrs.get("payload", {})
        candidate = data.get("candidate", {})
        company = data.get("company", {})
        candidate_id = candidate.get("id")
        company_id = company.get("id")

        if not candidate_id or not company_id:
            raise ValueError("Missing candidate_id or company_id")

        logger.info(f"🎯 Processing candidate_id={candidate_id} company_id={company_id}")

        # Step 1️⃣ — Fetch candidate info from Recruitee
        candidate_data = await call_recruitee_candidate(candidate_id, company_id)

        # Step 2️⃣ — CV extraction and preview logging
        cv_url = candidate_data.get("candidate", {}).get("cv_url")
        cv_text, cv_meta = None, {}
        if cv_url:
            try:
                cv_text, cv_meta = await fetch_and_extract_cv_text(cv_url)
                max_chars = int(os.getenv("CV_TEXT_MAX_CHARS", "120000"))
                if cv_text and len(cv_text) > max_chars:
                    cv_meta = {**cv_meta, "truncated": True, "orig_len": len(cv_text)}
                    cv_text = cv_text[:max_chars]
                preview = (cv_text or "")[:1500].replace("\n", " ")
                logger.info(
                    "📄 CV extraction: mime=%s pages=%s bytes=%s text_len=%s preview=%s",
                    cv_meta.get("mime"),
                    cv_meta.get("pages"),
                    cv_meta.get("bytes"),
                    len(cv_text or ""),
                    preview,
                )
            except Exception as e:
                logger.warning(f"⚠️ CV text extraction failed: {e}")
        else:
            logger.info("ℹ️ Candidate has no cv_url")

        # Step 3️⃣ — LinkedIn scraping (run local Puppeteer script and stream logs)
        linkedin_sections = {}
        social_links = candidate_data["candidate"].get("social_links", [])
        linkedin_url = next((link for link in social_links if "linkedin.com" in link.lower()), None)

        if linkedin_url:
            logger.info(f"🔍 Found LinkedIn URL: {linkedin_url}")
            try:
                process = await asyncio.create_subprocess_exec(
                    "node", "scripts/linkedin_automation.js", linkedin_url,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env={
                        **os.environ,
                        "LINKEDIN_EMAIL": os.getenv("LINKEDIN_EMAIL"),
                        "LINKEDIN_PASSWORD": os.getenv("LINKEDIN_PASSWORD"),
                    },
                )

                stdout_lines, stderr_lines = [], []

                async def read_stream(stream, buffer, level=logging.INFO):
                    async for line in stream:
                        decoded = line.decode().rstrip()
                        buffer.append(decoded)
                        logger.log(level, f"🧩 Puppeteer: {decoded}")

                # Stream logs live from Puppeteer
                await asyncio.gather(
                    read_stream(process.stdout, stdout_lines, logging.INFO),
                    read_stream(process.stderr, stderr_lines, logging.ERROR),
                )
                await process.wait()

                full_stdout = "\n".join(stdout_lines)
                if process.returncode != 0:
                    logger.warning(f"⚠️ Puppeteer returned non-zero exit code: {process.returncode}")

                # Extract JSON from Puppeteer output
                match = re.search(r"###DOM_JSON###(.*)", full_stdout, re.DOTALL)
                if match:
                    try:
                        dom_json = json.loads(match.group(1))
                        linkedin_sections = extract_linkedin_sections(dom_json)
                        logger.info("✅ Successfully extracted LinkedIn sections")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to parse LinkedIn DOM JSON: {e}")
                else:
                    logger.warning("⚠️ No ###DOM_JSON### marker found in Puppeteer output")

            except Exception as e:
                logger.error(f"❌ Error running Puppeteer scraper: {e}")
        else:
            logger.info("⚠️ No LinkedIn URL found for this candidate")

        # Step 4️⃣ — Build structured AI payload
        applied_contact_priority = {
            "Candidate": {
                "Id": candidate_data["candidate"].get("id"),
                "Name": candidate_data["candidate"].get("name"),
                "cv_url": cv_url,
                "cv_text": cv_text,
                "cv_meta": cv_meta,
                "Source": candidate_data["candidate"].get("source"),
                "Sources": candidate_data["candidate"].get("sources", []),
                "Social_links": social_links,
                "Links": candidate_data["candidate"].get("links", []),
                "Open_question_answers": candidate_data["candidate"].get("open_question_answers", []),
                "Grouped_open_questions_answers": candidate_data["candidate"].get("grouped_open_question_answers", []),
                "LinkedIn": linkedin_sections,
            },
            "Job_Description": {},
        }

        offers = [r for r in candidate_data.get("references", []) if r.get("type") == "Offer"]
        if offers:
            offer = offers[0]
            applied_contact_priority["Job_Description"] = {
                "id": offer.get("id"),
                "Title": offer.get("title"),
                "Description": offer.get("description"),
                "Requirements": offer.get("requirements"),
                "Department": offer.get("department"),
                "Url": offer.get("url"),
                "Remote": offer.get("remote"),
                "Hybrid": offer.get("hybrid"),
                "On_site": offer.get("on_site"),
                "Highlight_html": offer.get("highlight_html"),
            }

        # === LOG: Applied Contact Priority JSON ===
        try:
            applied_contact_priority_json = json.dumps(applied_contact_priority, ensure_ascii=False, indent=2)
            preview_limit = int(os.getenv("CONTACT_PRIORITY_PREVIEW_CHARS", "20000"))
            logger.info(
                "📦 Applied Contact Priority object (len=%d, showing first %d chars):\n%s",
                len(applied_contact_priority_json),
                min(len(applied_contact_priority_json), preview_limit),
                applied_contact_priority_json[:preview_limit],
            )
            if os.getenv("LOG_CONTACT_PRIORITY") == "1":
                contact_logger = logging.getLogger("contact_priority")
                if not contact_logger.handlers:
                    os.makedirs("logs", exist_ok=True)
                    handler = RotatingFileHandler("logs/contact_priority.log", maxBytes=50_000_000, backupCount=3)
                    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
                    contact_logger.addHandler(handler)
                    contact_logger.setLevel(logging.DEBUG)
                contact_logger.debug(
                    "candidate_id=%s company_id=%s len=%d\n%s",
                    candidate_id,
                    company_id,
                    len(applied_contact_priority_json),
                    applied_contact_priority_json,
                )
        except Exception as e:
            logger.warning(f"⚠️ Failed to log applied_contact_priority: {e}")

        # Step 5️⃣ — AI evaluation (reduced timeout)
        ai_result = await call_openai_evaluation(applied_contact_priority)
        score = ai_result.get("Scoring", "N/A")
        explanation = ai_result.get("Score_Explanation", "No explanation returned.")

        # Step 6️⃣ — Prevent duplicate updates
        existing = await get_existing_custom_fields(company_id, candidate_id)
        if "Contact Priority (AI-GPT)" in existing:
            logger.info(f"ℹ️ Candidate {candidate_id} already has AI-GPT field — skipping update.")
        else:
            await update_recruitee_custom_fields(company_id, candidate_id, score, explanation)
            logger.info(f"✅ Updated Recruitee custom fields for candidate {candidate_id}")

        logger.info(f"🎉 Finished processing candidate {candidate_id} — Score: {score}")

    except Exception as e:
        logger.exception(f"❌ Error in background candidate processing: {e}")





# ---- Pydantic query model for /recruitee/search/new/candidates ----
class RecruiteeSearchCandidatesQuery(BaseModel):
    limit: int | None = 50
    page: int | None = 1
    filters_json: list[dict] | None = None
    sort_by: str | None = "created_at_desc"

    @field_validator("sort_by")
    @classmethod
    def _validate_sort_by(cls, v):
        if v is None:
            return v
        allowed = {
            "relevance_asc", "relevance_desc",
            "created_at_asc", "created_at_desc",
            "candidate_name_asc", "candidate_name_desc",
            "candidate_rating_asc", "candidate_rating_desc",
            "candidate_positive_ratings_asc", "candidate_positive_ratings_desc",
            "candidate_job_title_asc", "candidate_job_title_desc",
            "candidate_stage_name_asc", "candidate_stage_name_desc",
            "gdpr_expires_at_asc", "gdpr_expires_at_desc",
        }
        if v not in allowed:
            raise ValueError(f"sort_by must be one of {allowed}")
        return v

    def to_recruitee_params(self) -> dict[str, str]:
        params: dict[str, str] = {}
        if self.limit is not None:
            params["limit"] = str(min(self.limit, 10000))
        if self.page is not None:
            params["page"] = str(self.page)
        if self.filters_json:
            params["filters_json"] = json.dumps(self.filters_json)
        if self.sort_by:
            params["sort_by"] = self.sort_by
        return params



# ---- GET /recruitee/search/new/candidates ----
@app.get("/recruitee/search/new/candidates")
async def list_recruitee_candidates_new(
    request: Request,
    limit: int | None = Query(50, ge=1, le=10000),
    page: int | None = Query(1, ge=1),
    sort_by: str | None = Query("created_at_desc"),
    has_cv: bool | None = Query(None, description="Filter only candidates with CVs"),
    created_after: datetime | None = Query(None, description="Filter candidates created after this date"),
    created_before: datetime | None = Query(None, description="Filter candidates created before this date"),
    source: str | None = Query(None, description="Candidate source, e.g., career_site, email, manual"),
):
    """
    Proxy to Recruitee: GET /c/{company_id}/search/new/candidates
    Uses JSON-based filters for performance and flexibility.
    """

    require_api_key(request)
    if not RECRUITEE_COMPANY_ID:
        raise HTTPException(500, "RECRUITEE_COMPANY_ID not configured")

    # Build filter list dynamically
    filters = []

    if has_cv is not None:
        filters.append({"field": "has_cv", "eq": has_cv})

    if created_after or created_before:
        filter_time = {"field": "created_at"}
        if created_after:
            filter_time["gte"] = _unix_timestamp(created_after)
        if created_before:
            filter_time["lte"] = _unix_timestamp(created_before)
        filters.append(filter_time)

    if source:
        filters.append({"field": "source", "in": [source]})

    qmodel = RecruiteeSearchCandidatesQuery(
        limit=limit,
        page=page,
        filters_json=filters if filters else None,
        sort_by=sort_by,
    )

    params = qmodel.to_recruitee_params()
    url = f"{RECRUITEE_API_URL}/c/{urllib.parse.quote(RECRUITEE_COMPANY_ID)}/search/new/candidates"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url, headers=_rb_headers(), params=params)
        if resp.status_code >= 400:
            logger.error("Recruitee error %s: %s", resp.status_code, resp.text)
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        data = resp.json()
        return {
            "message": "OK",
            "company_id": RECRUITEE_COMPANY_ID,
            "params": params,
            "filters_applied": filters,
            "result_count": len(data.get("candidates", [])),
            "result": data,
        }
    except httpx.TimeoutException:
        raise HTTPException(504, "Recruitee API timed out")
    except Exception as e:
        logger.exception("Recruitee call failed")
        raise HTTPException(502, f"Upstream error: {e}")



# GET /recruitee/candidates/{candidate_id} — fetch single candidate
@app.get("/recruitee/candidates/{candidate_id}")
async def get_recruitee_candidate(
    request: Request,
    candidate_id: int,
):
    """
    Proxy to Recruitee: GET /c/{company_id}/candidates/{candidate_id}
    """
    require_api_key(request)
    if not RECRUITEE_COMPANY_ID:
        raise HTTPException(500, "RECRUITEE_COMPANY_ID not configured")

    url = f"{RECRUITEE_API_URL}/c/{urllib.parse.quote(RECRUITEE_COMPANY_ID)}/candidates/{candidate_id}"
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url, headers=_rb_headers())

        if resp.status_code >= 400:
            logger.error("Recruitee error %s: %s", resp.status_code, resp.text)
            raise HTTPException(status_code=resp.status_code, detail=resp.text)

        # Log the full JSON response (pretty-printed)
        try:
            data = resp.json()
            logger.info("📦 Recruitee candidate %s full JSON:\n%s",
                        candidate_id,
                        json.dumps(data, ensure_ascii=False, indent=2))
            return data
        except ValueError:
            # Fallback: not JSON (unexpected), log raw text
            logger.warning("⚠️ Recruitee candidate %s returned non-JSON. Raw body:\n%s",
                           candidate_id, resp.text)
            return resp.text

    except httpx.TimeoutException:
        raise HTTPException(504, "Recruitee API timed out")
    except Exception as e:
        logger.exception("Recruitee call failed")
        raise HTTPException(502, f"Upstream error: {e}")



# ---- Pydantic query model for /positions/get ----
class PositionsQuery(BaseModel):
    scope: str | None = Field(
        None,
        description=(
            "Offer scope filter: 'archived', 'active', or 'not_archived'. "
            "If omitted, lists all job offers."
        ),
    )
    view_mode: str | None = Field(
        "default",
        description="Controls level of details: 'default' (detailed) or 'brief' (minimal).",
    )

    def to_recruitee_params(self) -> dict[str, str]:
        params: dict[str, str] = {}
        if self.scope:
            params["scope"] = self.scope
        if self.view_mode:
            params["view_mode"] = self.view_mode
        return params


@app.get("/positions/get")
async def get_positions(
    request: Request,
    scope: str | None = Query(
        None,
        description="Offer scope filter: 'archived', 'active', or 'not_archived'.",
    ),
    view_mode: str | None = Query(
        "default",
        description="Level of detail for the response: 'default' or 'brief'.",
    ),
):
    """
    Fetches job positions (offers) from Recruitee.
    Mirrors Recruitee's GET /c/{company_id}/offers endpoint,
    returning only offers with 'status' == 'published'.
    """

    require_api_key(request)

    if not RECRUITEE_COMPANY_ID:
        raise HTTPException(500, "RECRUITEE_COMPANY_ID not configured")

    # ✅ Correct endpoint URL
    url = f"{RECRUITEE_API_URL}/c/{RECRUITEE_COMPANY_ID}/offers"

    qmodel = PositionsQuery(scope=scope, view_mode=view_mode)
    params = qmodel.to_recruitee_params()

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url, headers=_rb_headers(), params=params)

        if resp.status_code >= 400:
            logger.error("Recruitee /offers error %s: %s", resp.status_code, resp.text)
            raise HTTPException(status_code=resp.status_code, detail=resp.text)

        data = resp.json()
        offers = data.get("offers", [])

        # ✅ Keep only published offers
        published_offers = [offer for offer in offers if offer.get("status") == "published"]

        # Replace offers list with filtered version
        data["offers"] = published_offers

        return {
            "message": "OK",
            "url": url,
            "params": params,
            "total_published": len(published_offers),
            "result": data,
        }

    except httpx.TimeoutException:
        raise HTTPException(504, "Recruitee API timed out")

    except Exception as e:
        logger.exception("Recruitee /offers call failed")
        raise HTTPException(502, f"Upstream error: {e}")

# ---- Pydantic query model for /departments/get ----

class DepartmentsQuery(BaseModel):
    company_id: str

    def to_recruitee_url(self) -> str:
        return f"https://api.recruitee.com/c/{self.company_id}/departments"


@app.get("/departments/get")
async def get_departments(request: Request):
    """
    Fetches all company departments from Recruitee.
    Mirrors Recruitee's GET /departments endpoint.
    Automatically uses RECRUITEE_COMPANY_ID from environment.
    """

    require_api_key(request)

    # ✅ Get company_id from environment
    company_id = RECRUITEE_COMPANY_ID
    if not company_id:
        raise HTTPException(500, "RECRUITEE_COMPANY_ID not configured in environment")

    qmodel = DepartmentsQuery(company_id=company_id)
    url = qmodel.to_recruitee_url()

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url, headers=_rb_headers())

        if resp.status_code >= 400:
            logger.error("Recruitee /departments error %s: %s", resp.status_code, resp.text)
            raise HTTPException(status_code=resp.status_code, detail=resp.text)

        data = resp.json()

        # ✅ Normalize the Recruitee response
        departments = [
            {
                "name": d.get("name"),
                "offers_count": d.get("offers_count", 0),
                "status": d.get("status"),
            }
            for d in data.get("departments", [])
        ]

        return {
            "message": "OK",
            "company_id": company_id,
            "result": {
                "departments": departments,
            },
        }

    except httpx.TimeoutException:
        raise HTTPException(504, "Recruitee API timed out")

    except Exception as e:
        logger.exception("Recruitee /departments call failed")
        raise HTTPException(502, f"Upstream error: {e}")



# ---- GET /positions/get/{offer_id} ----
@app.get("/positions/get/{offer_id}")
async def get_position_by_id(
    request: Request,
    offer_id: str = Path(..., description="The ID of the Recruitee job offer to fetch"),
):
    """
    Fetch a single job offer (position) from Recruitee by its ID.
    Mirrors Recruitee's GET /c/{company_id}/offers/{id} endpoint.
    """

    require_api_key(request)

    if not RECRUITEE_COMPANY_ID:
        raise HTTPException(500, "RECRUITEE_COMPANY_ID not configured")

    # ✅ Build the Recruitee API URL
    url = f"{RECRUITEE_API_URL}/c/{urllib.parse.quote(RECRUITEE_COMPANY_ID)}/offers/{offer_id}"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url, headers=_rb_headers())

        # Handle HTTP errors gracefully
        if resp.status_code >= 400:
            logger.error("Recruitee /offers/{id} error %s: %s", resp.status_code, resp.text)
            raise HTTPException(status_code=resp.status_code, detail=resp.text)

        data = resp.json()

        # ✅ Normalize the response to match your API schema
        offer = data.get("offer") or data  # some versions wrap in "offer"
        return {
            "message": "OK",
            "company_id": RECRUITEE_COMPANY_ID,
            "url": url,
            "result": offer,
        }

    except httpx.TimeoutException:
        raise HTTPException(504, "Recruitee API timed out")
    except Exception as e:
        logger.exception("Recruitee /offers/{id} call failed")
        raise HTTPException(502, f"Upstream error: {e}")


# ======================================================
# Create Job Offer (Recruitee Integration)
# ======================================================

class CreateJobOfferRequest(BaseModel):
    title: str
    description: str
    requirements: str
    department_id: Optional[str] = None
    location_ids: List[int]
    kind: str = "job"  # job | talent_pool
    on_site: bool = False
    hybrid: bool = False
    remote: bool = True
    visibility_options: List[str] = ["locations_question"]
    locations_question: str = "What is your preferred work location?"
    locations_question_type: str = "multiple_choice"
    locations_question_required: bool = True
    userEmail: Optional[str] = None
    dryRun: bool = False


@app.post("/positions/createJobOffer")
async def create_job_offer(request: Request, body: CreateJobOfferRequest):
    """
    Creates a new Job Offer in Recruitee.
    No local DB insert — Recruitee is the system of record.
    """

    # 🔐 Security
    require_api_key(request)

    # ✅ Configuration validation
    if not RECRUITEE_COMPANY_ID or not RECRUITEE_API_TOKEN or not RECRUITEE_API_URL:
        raise HTTPException(status_code=500, detail="Recruitee credentials not configured properly.")

    subject = body.userEmail or "system"

    # 🧪 Dry Run Mode
    if body.dryRun:
        return {
            "message": f"[dryRun] Would create Job Offer '{body.title}' "
                       f"in department '{body.department_id or 'N/A'}' (by {subject})",
            "created": False,
            "offer_id": None,
            "offer_url": None
        }

    # ✅ Prepare payload
    payload = {
        "title": body.title,
        "kind": body.kind,
        "location_ids": body.location_ids,
        "description": body.description,
        "requirements": body.requirements,
        "department_id": body.department_id,
        "on_site": body.on_site,
        "hybrid": body.hybrid,
        "remote": body.remote,
        "visibility_options": body.visibility_options,
        "locations_question": body.locations_question,
        "locations_question_type": body.locations_question_type,
        "locations_question_required": body.locations_question_required,
    }

    headers = {
        "accept": "application/json",
        "authorization": f"Bearer {RECRUITEE_API_TOKEN}",
        "content-type": "application/json"
    }

    # ✅ Call Recruitee API
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{RECRUITEE_API_URL}/c/{RECRUITEE_COMPANY_ID}/offers",
                json=payload,
                headers=headers,
                timeout=30
            )

        if response.status_code != 201:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Failed to create Job Offer in Recruitee: {response.text}"
            )

        data = response.json()
        offer = data.get("offer", {})

    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"Network error calling Recruitee: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

    # ✅ Success Response
    return {
        "message": f"✅ Job Offer '{body.title}' created successfully in Recruitee",
        "offer_id": offer.get("id"),
        "offer_url": offer.get("careers_url"),
        "created": True
    }

# ---- Pydantic query model for /recruitee/candidates ----
class RecruiteeCandidatesQuery(BaseModel):
    limit: int | None = 50
    offset: int | None = 0
    created_after: datetime | None = None
    disqualified: bool | None = None
    qualified: bool | None = None
    ids: str | None = None
    offer_id: str | None = None
    query: str | None = None
    sort: str | None = "by_date"
    with_messages: bool | None = None
    with_my_messages: bool | None = None

    @field_validator("sort")
    @classmethod
    def _validate_sort(cls, v):
        if v is None:
            return v
        allowed = {"by_date", "by_last_message"}
        if v not in allowed:
            raise ValueError(f"sort must be one of {allowed}")
        return v

    def to_recruitee_params(self) -> dict[str, str]:
        params: dict[str, str] = {}
        if self.limit is not None:
            params["limit"] = str(min(self.limit, 10000))
        if self.offset is not None:
            params["offset"] = str(self.offset)
        if self.created_after:
            params["created_after"] = self.created_after.isoformat()
        if self.disqualified is not None:
            params["disqualified"] = "1" if self.disqualified else "0"
        if self.qualified is not None:
            params["qualified"] = "1" if self.qualified else "0"
        if self.ids:
            params["ids"] = self.ids
        if self.offer_id:
            params["offer_id"] = self.offer_id
        if self.query:
            params["query"] = self.query
        if self.sort:
            params["sort"] = self.sort
        if self.with_messages is not None:
            params["with_messages"] = "1" if self.with_messages else "0"
        if self.with_my_messages is not None:
            params["with_my_messages"] = "1" if self.with_my_messages else "0"
        return params


# ---- GET /recruitee/candidates ----
@app.get("/recruitee/candidates")
async def list_recruitee_candidates(
    request: Request,
    limit: int | None = Query(50, ge=1, le=10000),
    offset: int | None = Query(0, ge=0),
    created_after: datetime | None = Query(None),
    disqualified: bool | None = Query(None),
    qualified: bool | None = Query(None),
    ids: str | None = Query(None),
    offer_id: str | None = Query(None),
    query: str | None = Query(None),
    sort: str | None = Query("by_date"),
    with_messages: bool | None = Query(None),
    with_my_messages: bool | None = Query(None),
):
    """
    Proxy to Recruitee: GET /c/{company_id}/candidates
    Retrieves candidates from the Recruitee database using offset-based pagination.
    """

    require_api_key(request)
    if not RECRUITEE_COMPANY_ID:
        raise HTTPException(500, "RECRUITEE_COMPANY_ID not configured")

    qmodel = RecruiteeCandidatesQuery(
        limit=limit,
        offset=offset,
        created_after=created_after,
        disqualified=disqualified,
        qualified=qualified,
        ids=ids,
        offer_id=offer_id,
        query=query,
        sort=sort,
        with_messages=with_messages,
        with_my_messages=with_my_messages,
    )

    params = qmodel.to_recruitee_params()
    url = f"{RECRUITEE_API_URL}/c/{urllib.parse.quote(RECRUITEE_COMPANY_ID)}/candidates"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url, headers=_rb_headers(), params=params)
        if resp.status_code >= 400:
            logger.error("Recruitee error %s: %s", resp.status_code, resp.text)
            raise HTTPException(status_code=resp.status_code, detail=resp.text)

        data = resp.json()
        return {
            "message": "OK",
            "company_id": RECRUITEE_COMPANY_ID,
            "params": params,
            "result_count": len(data.get("candidates", [])),
            "result": data,
        }

    except httpx.TimeoutException:
        raise HTTPException(504, "Recruitee API timed out")
    except Exception as e:
        logger.exception("Recruitee call failed")
        raise HTTPException(502, f"Upstream error: {e}")



@app.post("/scrape-linkedin")
async def scrape_linkedin(url: str = Query(..., description="LinkedIn URL to visit")):
    """Runs Puppeteer LinkedIn scraper and streams logs to server."""
    try:
        logger.info(f"🌐 Starting Puppeteer scrape for URL: {url}")

        # Launch Puppeteer script as subprocess
        process = await asyncio.create_subprocess_exec(
            "node", "scripts/linkedin_automation.js", url,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={
                **os.environ,
                "LINKEDIN_EMAIL": os.getenv("LINKEDIN_EMAIL"),
                "LINKEDIN_PASSWORD": os.getenv("LINKEDIN_PASSWORD"),
            }
        )

        stdout_lines = []
        stderr_lines = []

        async def read_stream(stream, buffer, level=logging.INFO):
            async for line in stream:
                decoded = line.decode().rstrip()
                buffer.append(decoded)
                logger.log(level, f"🧩 Puppeteer: {decoded}")

        # Read both stdout and stderr concurrently
        await asyncio.gather(
            read_stream(process.stdout, stdout_lines, logging.INFO),
            read_stream(process.stderr, stderr_lines, logging.ERROR)
        )

        await process.wait()

        full_stdout = "\n".join(stdout_lines)
        if process.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Puppeteer failed: {process.returncode}")

        # Extract DOM JSON if present
        match = re.search(r"###DOM_JSON###(.*)", full_stdout, re.DOTALL)
        if match:
            try:
                dom_json = json.loads(match.group(1))
            except Exception as e:
                logger.warning(f"⚠️ Failed to parse DOM JSON: {e}")
                dom_json = {"raw_output": match.group(1)[:500]}
        else:
            dom_json = {"raw_output": full_stdout}

        logger.info("✅ Puppeteer finished successfully")
        return {"status": "success", "url": url, "dom": dom_json}

    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Puppeteer timed out")
    except Exception as e:
        logger.exception(f"❌ Error running Puppeteer: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/whoami") # Verify who the api is acting as when user impersonation
def whoami(request: Request):
    require_api_key(request)
    subject = _extract_subject_from_request(request)
    _, drive, _ = get_clients(subject)
    about = drive.about().get(fields="user(emailAddress,displayName),storageQuota").execute()
    return {"subject_param": subject, "drive_user": about.get("user")}
