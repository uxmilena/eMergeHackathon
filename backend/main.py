"""
Canon — AI-powered HR policy compliance backend.

Endpoints:
  POST /upload              PDF → extracted text
  GET  /regulations         Federal Register pull (DOL)
  POST /state-regulations   Hardcoded state labor law rules
  POST /analyze             Claude-powered gap analysis
  POST /generate            Claude-generated announcement/Slack/email/TLDR
"""
from dotenv import load_dotenv
load_dotenv()

import asyncio
import json
import os
from pathlib import Path
from typing import List, Optional, Union

import fitz  # PyMuPDF
import httpx
from anthropic import Anthropic
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------- App setup ----------

app = FastAPI(title="Canon API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten before prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

CLAUDE_MODEL = "claude-opus-4-7"


def get_claude() -> Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(500, "ANTHROPIC_API_KEY env var is not set")
    return Anthropic(api_key=api_key)


def extract_claude_text(resp) -> str:
    """Pull text out of an anthropic Messages response, ignoring non-text blocks."""
    return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")


def parse_json_response(raw: str) -> Union[dict, list]:
    """Parse Claude's JSON response, defensively stripping markdown fences."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        # Strip opening fence (```json or ```)
        first_nl = cleaned.find("\n")
        if first_nl != -1:
            cleaned = cleaned[first_nl + 1:]
        # Strip closing fence
        if cleaned.rstrip().endswith("```"):
            cleaned = cleaned.rstrip()[:-3].rstrip()
    return json.loads(cleaned)


# ---------- Federal regulation baseline ----------
# Stable statutory floor that always applies. Federal Register pull (below) adds freshness.

FEDERAL_REGULATIONS: dict = {
    "name": "Federal (FLSA baseline)",
    "rules": [
        {
            "topic": "Overtime (FLSA)",
            "rule": (
                "Non-exempt employees are entitled to overtime at 1.5× the regular rate "
                "for hours worked over 40 in a workweek. Applies regardless of work location."
            ),
            "authority": "29 U.S.C. § 207; 29 CFR Part 778",
        },
        {
            "topic": "Minimum Wage (FLSA)",
            "rule": (
                "Federal minimum wage is $7.25/hour. States may set higher; the higher of "
                "federal or state minimum applies. Remote employees are entitled to whichever is higher."
            ),
            "authority": "29 U.S.C. § 206",
        },
        {
            "topic": "Expense Kickback Rule",
            "rule": (
                "Employers cannot require employees to bear expenses that would reduce their "
                "effective wage below the federal (or applicable state) minimum. Unreimbursed "
                "remote work costs for non-exempt employees frequently trigger this rule."
            ),
            "authority": "29 CFR § 531.35",
        },
        {
            "topic": "Hours Worked / Off-the-Clock",
            "rule": (
                "All work time must be compensated for non-exempt employees, including work "
                "performed at home. Employers must have a reasonable means to track remote "
                "hours and cannot permit unreported off-the-clock work."
            ),
            "authority": "29 CFR Part 785; DOL Field Assistance Bulletin 2020-5",
        },
        {
            "topic": "Exempt Classification (FLSA)",
            "rule": (
                "Exempt employees must (1) be paid on a salary basis, (2) meet the federal "
                "salary threshold ($684/week minimum, subject to DOL rulemaking), and "
                "(3) satisfy duties tests for executive, administrative, professional, or outside sales."
            ),
            "authority": "29 CFR Part 541",
        },
        {
            "topic": "FMLA Leave",
            "rule": (
                "Covered employers (50+ employees within 75 miles) must provide eligible "
                "employees (worked 1,250 hours in prior 12 months) up to 12 weeks of unpaid, "
                "job-protected leave. A remote employee's worksite is generally the location "
                "from which their work is assigned."
            ),
            "authority": "29 U.S.C. § 2601; 29 CFR Part 825",
        },
        {
            "topic": "Recordkeeping",
            "rule": (
                "Employers must maintain accurate records of hours worked, wages, and payroll "
                "data for non-exempt employees — including remote workers."
            ),
            "authority": "29 CFR Part 516",
        },
        {
            "topic": "ADA Reasonable Accommodation",
            "rule": (
                "Remote work may itself be a reasonable accommodation under the ADA for "
                "qualified employees with disabilities. Employers must engage in the interactive "
                "process and cannot categorically refuse remote accommodations."
            ),
            "authority": "42 U.S.C. § 12111 et seq.; EEOC guidance",
        },
    ],
}

# Default topics for the live Federal Register pull during /analyze.
# Chosen for reliably returning DOL final rules relevant to remote work compliance.
DEFAULT_FEDERAL_TOPICS = ["overtime", "wage and hour", "family medical leave"]


# ---------- State regulation knowledge base ----------
# Curated labor law rules relevant to remote work for the 6 demo states.
# Swap for a live data source post-hackathon.

STATE_REGULATIONS: dict[str, dict] = {
    "CA": {
        "name": "California",
        "rules": [
            {
                "topic": "Overtime",
                "rule": (
                    "Daily overtime required after 8 hours in a workday; "
                    "double-time after 12 hours. A 40-hour weekly threshold alone is insufficient."
                ),
                "authority": "CA Labor Code § 510",
            },
            {
                "topic": "Expense Reimbursement",
                "rule": (
                    "Employers must reimburse all necessary expenses incurred by employees in "
                    "direct consequence of their duties — including a reasonable portion of home "
                    "internet, cell phone, and equipment costs for remote work."
                ),
                "authority": "CA Labor Code § 2802",
            },
            {
                "topic": "Meal & Rest Breaks",
                "rule": (
                    "30-minute unpaid meal break before the end of the 5th hour worked. "
                    "10-minute paid rest break for every 4 hours worked. Remote employees retain these rights."
                ),
                "authority": "CA Labor Code §§ 226.7, 512; IWC Wage Orders",
            },
            {
                "topic": "Paid Sick Leave",
                "rule": (
                    "Employees accrue 1 hour of paid sick leave per 30 hours worked — minimum "
                    "40 hours/year accrual with at least 40 hours usable."
                ),
                "authority": "CA Healthy Workplaces, Healthy Families Act",
            },
            {
                "topic": "Exempt Classification",
                "rule": (
                    "Exempt employees must earn at least 2× the state minimum wage for full-time work. "
                    "California's minimum salary threshold exceeds federal FLSA levels."
                ),
                "authority": "CA Labor Code § 515",
            },
        ],
    },
    "NY": {
        "name": "New York",
        "rules": [
            {
                "topic": "Expense Reimbursement",
                "rule": (
                    "Employers must reimburse necessary business expenses. Wage deductions are "
                    "heavily restricted under Labor Law § 193 — employers generally cannot deduct "
                    "unreimbursed expenses from pay."
                ),
                "authority": "NY Labor Law §§ 198-c, 193",
            },
            {
                "topic": "Workplace Safety (HERO Act)",
                "rule": (
                    "The NY HERO Act requires a written airborne infectious disease exposure "
                    "prevention plan that extends to any worksite where employees perform work, "
                    "including remote home offices."
                ),
                "authority": "NY Labor Law § 218-B",
            },
            {
                "topic": "Wage Notice",
                "rule": (
                    "The Wage Theft Prevention Act requires written notice of pay rate, overtime rate, "
                    "pay frequency, and employer information at hire and again for any material changes."
                ),
                "authority": "NY Labor Law § 195",
            },
            {
                "topic": "Paid Family Leave",
                "rule": (
                    "Employees are entitled to up to 12 weeks of paid family leave. Remote employees "
                    "whose work is performed in New York are covered."
                ),
                "authority": "NY Paid Family Leave Law",
            },
            {
                "topic": "Paid Sick Leave",
                "rule": (
                    "Employers with 5+ employees must provide paid sick leave; accrual of 1 hour per "
                    "30 hours worked, up to 40 or 56 hours/year depending on employer size."
                ),
                "authority": "NY Labor Law § 196-b",
            },
        ],
    },
    "TX": {
        "name": "Texas",
        "rules": [
            {
                "topic": "At-Will Employment",
                "rule": (
                    "Texas is a strong at-will state. Policy language that implies guaranteed processes "
                    "or progressive discipline can unintentionally modify at-will status and create "
                    "implied-contract exposure."
                ),
                "authority": "Texas common law",
            },
            {
                "topic": "Overtime",
                "rule": (
                    "No state overtime rule beyond federal FLSA. Overtime owed after 40 hours in a workweek; "
                    "no daily overtime requirement."
                ),
                "authority": "Federal FLSA applies",
            },
            {
                "topic": "Meal & Rest Breaks",
                "rule": (
                    "No state-mandated meal or rest breaks. Federal FLSA rules apply — breaks under "
                    "20 minutes must be paid."
                ),
                "authority": "Federal FLSA",
            },
            {
                "topic": "Expense Reimbursement",
                "rule": (
                    "No state-level mandate. Federal FLSA 'kickback rule' may require reimbursement if "
                    "unreimbursed work expenses drop effective pay below minimum wage."
                ),
                "authority": "Federal FLSA kickback rule",
            },
            {
                "topic": "Final Pay",
                "rule": (
                    "Final wages due within 6 days of involuntary termination; on the next regular "
                    "payday for voluntary resignation."
                ),
                "authority": "Texas Payday Law",
            },
        ],
    },
    "OR": {
        "name": "Oregon",
        "rules": [
            {
                "topic": "Meal & Rest Breaks",
                "rule": (
                    "30-minute unpaid meal break for shifts over 6 hours; 10-minute paid rest break "
                    "for every 4-hour segment. Applies to remote employees."
                ),
                "authority": "OAR 839-020-0050",
            },
            {
                "topic": "Paid Sick Leave",
                "rule": (
                    "Employers with 10+ employees (6+ in Portland) must provide paid sick leave; "
                    "accrual of 1 hour per 30 hours worked, up to 40 hours/year."
                ),
                "authority": "ORS 653.601",
            },
            {
                "topic": "Expense Reimbursement",
                "rule": (
                    "Employers must reimburse work-related expenses. Courts have applied this to "
                    "home office and equipment costs for remote employees."
                ),
                "authority": "ORS 652.610",
            },
            {
                "topic": "Family & Paid Leave",
                "rule": (
                    "Oregon Family Leave Act (OFLA) provides up to 12 weeks of protected leave. "
                    "Paid Leave Oregon adds up to 12 weeks of paid family, medical, or safe leave."
                ),
                "authority": "ORS 659A.150-186; ORS 657B",
            },
            {
                "topic": "Final Pay",
                "rule": (
                    "Final wages due on the next business day after involuntary termination; within "
                    "5 days or next payday (whichever first) for voluntary resignation without notice."
                ),
                "authority": "ORS 652.140",
            },
        ],
    },
    "FL": {
        "name": "Florida",
        "rules": [
            {
                "topic": "At-Will Employment",
                "rule": (
                    "Florida is an at-will state. No state overtime rule beyond federal FLSA."
                ),
                "authority": "Federal FLSA applies",
            },
            {
                "topic": "Meal & Rest Breaks",
                "rule": (
                    "No state-mandated meal or rest breaks for adult employees. Federal FLSA rules apply."
                ),
                "authority": "Federal FLSA",
            },
            {
                "topic": "Minimum Wage",
                "rule": (
                    "Florida's constitutional minimum wage exceeds federal and adjusts annually. "
                    "Unreimbursed remote work expenses cannot push effective pay below state minimum."
                ),
                "authority": "Fla. Const. Art. X, § 24",
            },
            {
                "topic": "Expense Reimbursement",
                "rule": (
                    "No general state mandate for expense reimbursement, but contractual obligations "
                    "and minimum-wage protections still apply."
                ),
                "authority": "No affirmative state mandate",
            },
            {
                "topic": "Final Pay",
                "rule": (
                    "No specific statute. Final wages typically paid on the next regular payday per "
                    "standard practice."
                ),
                "authority": "Florida common practice",
            },
        ],
    },
    "IL": {
        "name": "Illinois",
        "rules": [
            {
                "topic": "Expense Reimbursement",
                "rule": (
                    "Employers must reimburse all necessary expenditures incurred within the scope "
                    "of employment, including remote work expenses like internet, phone, and equipment "
                    "when primarily for the employer's benefit."
                ),
                "authority": "820 ILCS 115/9.5",
            },
            {
                "topic": "One Day Rest In Seven (ODRISA)",
                "rule": (
                    "Employees are entitled to at least 24 consecutive hours off in every consecutive "
                    "7-day period and a 20-minute meal break for shifts of 7.5+ hours."
                ),
                "authority": "820 ILCS 140",
            },
            {
                "topic": "Paid Leave for All Workers Act",
                "rule": (
                    "Employees accrue 1 hour of paid leave per 40 hours worked, up to 40 hours/year, "
                    "usable for any reason."
                ),
                "authority": "820 ILCS 192",
            },
            {
                "topic": "Wage Payment Timing",
                "rule": (
                    "Wages must be paid at least semi-monthly for most employees. Final pay due at "
                    "the next regular payday."
                ),
                "authority": "820 ILCS 115",
            },
            {
                "topic": "Local Ordinances",
                "rule": (
                    "Employees working remotely within Chicago or Cook County may trigger local "
                    "minimum wage and paid sick leave ordinances distinct from state law."
                ),
                "authority": "Chicago Minimum Wage Ordinance; Cook County Earned Sick Leave Ordinance",
            },
        ],
    },
}


# ---------- Pydantic models ----------

class StatesRequest(BaseModel):
    states: List[str]


class AnalyzeRequest(BaseModel):
    policy_text: str
    states: List[str]
    include_federal_register: bool = True
    federal_register_topics: Optional[List[str]] = None  # defaults to DEFAULT_FEDERAL_TOPICS


class GenerateRequest(BaseModel):
    policy_text: str
    company_name: str
    states: List[str]
    approved_findings: Optional[List[dict]] = None


# ---------- Endpoints ----------

@app.get("/")
def root():
    return {
        "service": "Canon API",
        "status": "ok",
        "federal_baseline": FEDERAL_REGULATIONS["name"],
        "federal_baseline_topics": [r["topic"] for r in FEDERAL_REGULATIONS["rules"]],
        "supported_states": list(STATE_REGULATIONS.keys()),
        "default_federal_register_topics": DEFAULT_FEDERAL_TOPICS,
    }


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Accept a PDF, extract text via PyMuPDF, return text and page breakdown."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "File must be a PDF")

    contents = await file.read()
    save_path = UPLOAD_DIR / file.filename
    save_path.write_bytes(contents)

    try:
        doc = fitz.open(stream=contents, filetype="pdf")
        pages = []
        full_text_parts = []
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()
            pages.append({"page": page_num, "text": text})
            full_text_parts.append(text)
        doc.close()
    except Exception as e:
        raise HTTPException(500, f"Failed to parse PDF: {e}")

    full_text = "\n\n".join(full_text_parts)
    return {
        "filename": file.filename,
        "num_pages": len(pages),
        "char_count": len(full_text),
        "full_text": full_text,
        "pages": pages,
    }


@app.get("/regulations")
async def get_federal_regulations(
    topic: str,
    limit: int = 3,
    include_proposed: bool = False,
):
    """Pull relevance-ranked DOL rules from the Federal Register.

    Returns final rules (RULE) by default — what actually governs compliance.
    Set include_proposed=True to also include proposed rules (PRORULE), useful
    for anticipating upcoming changes.
    """
    types = ["RULE", "PRORULE"] if include_proposed else ["RULE"]
    url = "https://www.federalregister.gov/api/v1/documents.json"
    params = {
        "conditions[term]": topic,
        "conditions[agencies][]": "labor-department",
        "conditions[type][]": types,
        "per_page": min(max(limit, 1), 10),
        # No `order` param → Federal Register defaults to relevance ranking.
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            data = r.json()
    except httpx.HTTPError as e:
        raise HTTPException(502, f"Federal Register API error: {e}")

    results = []
    for d in data.get("results", [])[:limit]:
        results.append({
            "title": d.get("title"),
            "date": d.get("publication_date"),
            "abstract": d.get("abstract"),
            "url": d.get("html_url"),
            "type": d.get("type"),
            "agency_names": d.get("agency_names", []),
        })

    return {
        "topic": topic,
        "doc_types": types,
        "count": len(results),
        "results": results,
    }

@app.get("/dashboard-sync")
async def dashboard_sync():
    """
    Live pull of recent DOL rules across all watched topics.
    Returns dashboard-ready cards sorted by publication date.
    """
    topics = DEFAULT_FEDERAL_TOPICS  # ["overtime", "wage and hour", "family medical leave"]
    
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            tasks = [_fetch_fr_topic(client, t) for t in topics]
            results = await asyncio.gather(*tasks, return_exceptions=True)
    except Exception as e:
        raise HTTPException(502, f"Federal Register sync failed: {e}")
    
    cards = []
    seen_urls = set()
    for topic, result in zip(topics, results):
        if isinstance(result, Exception):
            continue
        for doc in result:
            url = doc.get("html_url")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            
            title = doc.get("title") or "Untitled rule"
            abstract = (doc.get("abstract") or "").strip()
            # Abstracts can be long; truncate for card display
            if len(abstract) > 260:
                abstract = abstract[:257].rstrip() + "…"
            
            cards.append({
                "source": "Dept. of Labor",
                "source_type": "federal_register",
                "date": doc.get("publication_date"),
                "title": title,
                "abstract": abstract,
                "url": url,
                "topic_query": topic,
                "doc_type": doc.get("type", "Rule"),
            })
    
    # Sort newest first
    cards.sort(key=lambda c: c.get("date") or "", reverse=True)
    
    return {
        "synced_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "sources_checked": ["Federal Register (DOL)"],
        "topics_watched": topics,
        "card_count": len(cards),
        "cards": cards[:10],  # cap at 10 for clean UI
    }
    
@app.post("/state-regulations")
def get_state_regulations(req: StatesRequest):
    """Return curated labor-law rules for requested states."""
    out = {}
    unknown = []
    for code in req.states:
        key = code.upper()
        if key in STATE_REGULATIONS:
            out[key] = STATE_REGULATIONS[key]
        else:
            unknown.append(key)
    return {"states": out, "unknown_states": unknown}


async def _fetch_fr_topic(client: httpx.AsyncClient, topic: str) -> list:
    """Fetch top DOL final rules from Federal Register for a single topic."""
    url = "https://www.federalregister.gov/api/v1/documents.json"
    params = {
        "conditions[term]": topic,
        "conditions[agencies][]": "labor-department",
        "conditions[type][]": "RULE",
        "per_page": 2,
    }
    r = await client.get(url, params=params)
    r.raise_for_status()
    return r.json().get("results", [])


async def fetch_federal_register_supplement(topics: List[str]) -> List[dict]:
    """Pull DOL Federal Register rules for multiple topics in parallel. Degrades gracefully."""
    async with httpx.AsyncClient(timeout=20.0) as client:
        tasks = [_fetch_fr_topic(client, t) for t in topics]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    combined = []
    for topic, result in zip(topics, results):
        if isinstance(result, Exception):
            continue
        for doc in result[:2]:
            combined.append({
                "topic_query": topic,
                "title": doc.get("title"),
                "date": doc.get("publication_date"),
                "abstract": (doc.get("abstract") or "")[:400],
                "url": doc.get("html_url"),
            })
    return combined


@app.post("/analyze")
async def analyze_policy(req: AnalyzeRequest):
    """Claude compares policy against federal baseline + live DOL rules + state rules."""

    # ---- Federal baseline (always included) ----
    fed_rules_str = "\n".join(
        f"  - [{r['topic']}] {r['rule']} (Authority: {r['authority']})"
        for r in FEDERAL_REGULATIONS["rules"]
    )
    federal_baseline_block = (
        "FEDERAL BASELINE RULES (always apply — statutory floor):\n" + fed_rules_str
    )

    # ---- Live Federal Register supplement (optional, parallel fetch) ----
    fr_supplement_block = ""
    fr_docs: List[dict] = []
    if req.include_federal_register:
        topics = req.federal_register_topics or DEFAULT_FEDERAL_TOPICS
        try:
            fr_docs = await fetch_federal_register_supplement(topics)
        except Exception:
            fr_docs = []  # don't fail the analyze if Federal Register is down
        if fr_docs:
            fr_lines = []
            for d in fr_docs:
                fr_lines.append(
                    f"  - [query: {d['topic_query']}] {d['title']} ({d['date']})\n"
                    f"    Abstract: {d['abstract']}\n"
                    f"    URL: {d['url']}"
                )
            fr_supplement_block = (
                "RECENT DOL FEDERAL REGISTER RULES (current-awareness context — "
                "flag if policy is out of step with recent rulemaking):\n" + "\n".join(fr_lines)
            )

    # ---- State rules block ----
    state_context_parts = []
    unknown_states = []
    for code in req.states:
        key = code.upper()
        info = STATE_REGULATIONS.get(key)
        if not info:
            unknown_states.append(key)
            continue
        rules_str = "\n".join(
            f"  - [{r['topic']}] {r['rule']} (Authority: {r['authority']})"
            for r in info["rules"]
        )
        state_context_parts.append(f"{info['name']} ({key}):\n{rules_str}")
    state_context = "\n\n".join(state_context_parts)

    # ---- Assemble full prompt context ----
    context_blocks = [federal_baseline_block]
    if fr_supplement_block:
        context_blocks.append(fr_supplement_block)
    if state_context:
        context_blocks.append(f"STATE REQUIREMENTS:\n{state_context}")
    full_context = "\n\n".join(context_blocks)

    system_prompt = (
        "You are an employment law compliance analyst. You compare a company's remote work "
        "policy against federal and state labor law requirements and identify every meaningful "
        "compliance gap.\n\n"
        "Severity classification:\n"
        "- \"critical\": policy directly violates or materially conflicts with federal or state law\n"
        "- \"recommended\": policy is silent on a requirement or creates meaningful risk\n"
        "- \"compliant\": policy already aligns with the requirement (include for coverage confirmation)\n\n"
        "Use FEDERAL BASELINE RULES as the always-applicable statutory floor. Use RECENT DOL "
        "FEDERAL REGISTER RULES as current-awareness context — flag the policy if it is out of "
        "step with recent DOL rulemaking. Use STATE REQUIREMENTS to identify state-specific gaps.\n\n"
        "Return ONLY a JSON array. No preamble, no prose, no markdown fences."
    )

    user_prompt = f"""POLICY TEXT:
\"\"\"
{req.policy_text}
\"\"\"

{full_context}

Return a JSON array. Each element must have these exact keys:
- "state": the jurisdiction — use "Federal" for federal findings, or the full state name (e.g., "California")
- "section": topic area (e.g., "Overtime", "Expense Reimbursement")
- "severity": one of "critical", "recommended", "compliant"
- "issue": plain-English explanation — 1 to 3 sentences
- "suggestion": specific policy-ready replacement or addendum text
- "authority": the legal authority citation

Cover every jurisdiction's every topic. Be strict and specific — do not hedge."""

    client = get_claude()
    try:
        # Run sync Claude call in threadpool so it doesn't block the event loop
        resp = await asyncio.to_thread(
            client.messages.create,
            model=CLAUDE_MODEL,
            max_tokens=8192,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
    except Exception as e:
        raise HTTPException(502, f"Claude API error: {e}")

    raw = extract_claude_text(resp)
    try:
        findings = parse_json_response(raw)
    except json.JSONDecodeError as e:
        raise HTTPException(500, f"Claude returned non-JSON response: {e}\n---\n{raw[:800]}")

    if not isinstance(findings, list):
        raise HTTPException(500, f"Expected JSON array, got {type(findings).__name__}")

    counts = {"critical": 0, "recommended": 0, "compliant": 0}
    for f in findings:
        sev = str(f.get("severity", "")).lower()
        if sev in counts:
            counts[sev] += 1

    return {
        "findings": findings,
        "summary": counts,
        "jurisdictions_analyzed": ["Federal"] + [c.upper() for c in req.states if c.upper() in STATE_REGULATIONS],
        "unknown_states": unknown_states,
        "federal_register_context": fr_docs,  # surface the live pull to the frontend
    }


@app.post("/generate")
def generate_outputs(req: GenerateRequest):
    """Claude generates HTML announcement, Slack post, email draft, and TL;DR."""
    approved_block = ""
    if req.approved_findings:
        approved_block = (
            "\n\nAPPROVED COMPLIANCE CHANGES (incorporate these into the announcement):\n"
            + json.dumps(req.approved_findings, indent=2)
        )

    system_prompt = (
        "You generate polished employee communications for policy updates. You write clearly, "
        "warmly, and without corporate filler. Return ONLY a JSON object — no preamble, no markdown fences."
    )

    user_prompt = f"""Company: {req.company_name}
States covered: {", ".join(req.states)}

POLICY TEXT:
\"\"\"
{req.policy_text}
\"\"\"{approved_block}

Produce a JSON object with these exact keys:

1. "html_announcement": a complete, self-contained HTML page (inline CSS, no external deps) announcing the updated remote work policy. Include the company name, a clear header, 3–5 sections covering what changed and why, a state-by-state callout noting addendums where relevant, and a "Questions?" footer. Use readable typography, generous spacing, a neutral professional palette. No JavaScript.

2. "slack_post": a concise Slack announcement, 3–5 short lines, emoji-light, with a [LINK] placeholder to the full HTML policy. Should sound human.

3. "email_draft": a manager-distribution email with Subject, Greeting, Body (3 short paragraphs), and Sign-off. Plain text, no HTML. Managers will forward to their teams.

4. "tldr": one paragraph, plain English, 3–5 sentences, readable in 20 seconds — what changed and what it means for an employee.

Return ONLY the JSON object."""

    client = get_claude()
    try:
        resp = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
    except Exception as e:
        raise HTTPException(502, f"Claude API error: {e}")

    raw = extract_claude_text(resp)
    try:
        outputs = parse_json_response(raw)
    except json.JSONDecodeError as e:
        raise HTTPException(500, f"Claude returned non-JSON response: {e}\n---\n{raw[:800]}")

    return outputs
