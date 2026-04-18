from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import fitz  # PyMuPDF
import httpx
import anthropic
import os
import json

app = FastAPI(title="Canon API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# --- State labor law rules (hardcoded for demo) ---
STATE_RULES = {
    "CA": {
        "name": "California",
        "rules": [
            "Daily overtime required after 8 hours (not just 40hr weekly)",
            "Meal break required every 5 hours",
            "Employers must reimburse all remote work expenses (internet, phone, equipment)",
            "Final paycheck due same day if terminated",
            "Minimum wage: $16/hr (2024)"
        ]
    },
    "NY": {
        "name": "New York",
        "rules": [
            "Spread of hours pay required if shift exceeds 10 hours",
            "Employers must provide written notice of pay rate",
            "NYC: additional sick leave requirements",
            "Remote worker expense reimbursement required in NYC",
            "Minimum wage: $16/hr NYC, $15/hr elsewhere"
        ]
    },
    "TX": {
        "name": "Texas",
        "rules": [
            "No state income tax",
            "At-will employment — no additional remote work mandates beyond federal",
            "Final paycheck due within 6 days if fired, next payday if resigned"
        ]
    },
    "OR": {
        "name": "Oregon",
        "rules": [
            "Oregon Family Leave Act: up to 12 weeks (broader than FMLA)",
            "Paid Leave Oregon: up to 12 weeks paid leave for qualifying events",
            "Expense reimbursement required for remote workers",
            "Predictive scheduling laws apply in some cities",
            "Minimum wage varies by region: $14.20 - $15.45"
        ]
    },
    "FL": {
        "name": "Florida",
        "rules": [
            "No state income tax",
            "At-will employment state",
            "No state-specific remote work mandates beyond federal",
            "Minimum wage: $13/hr (2024)"
        ]
    },
    "IL": {
        "name": "Illinois",
        "rules": [
            "Illinois WARN Act: stricter than federal for layoff notices",
            "Expense reimbursement required for remote workers",
            "Chicago: additional paid sick leave requirements",
            "Biometric data privacy law (BIPA) — strict rules on data collection",
            "Minimum wage: $14/hr"
        ]
    }
}


# --- 1. PDF Upload + Parse ---
@app.post("/upload")
async def upload_policy(file: UploadFile = File(...)):
    contents = await file.read()
    upload_path = f"uploads/{file.filename}"
    with open(upload_path, "wb") as f:
        f.write(contents)

    doc = fitz.open(upload_path)
    text = ""
    for page in doc:
        text += page.get_text()

    return {
        "filename": file.filename,
        "page_count": len(doc),
        "text": text,
        "char_count": len(text)
    }


# --- 2. Federal Register API ---
@app.get("/regulations")
async def get_regulations(topic: str = "remote work"):
    url = "https://www.federalregister.gov/api/v1/documents.json"
    params = {
        "conditions[term]": topic,
        "conditions[agencies][]": "labor-department",
        "per_page": 3,
        "fields[]": ["title", "publication_date", "abstract", "html_url"]
    }
    async with httpx.AsyncClient() as client_http:
        response = await client_http.get(url, params=params)
        data = response.json()

    results = data.get("results", [])
    return {"regulations": results, "count": len(results)}


# --- 3. State Rules ---
@app.get("/states")
async def get_state_rules(states: str):
    state_list = [s.strip().upper() for s in states.split(",")]
    result = {}
    for state in state_list:
        if state in STATE_RULES:
            result[state] = STATE_RULES[state]
    return {"states": result}


# --- 4. AI Compliance Analysis ---
class AnalyzeRequest(BaseModel):
    policy_text: str
    states: list[str]
    company_name: str = "the company"


@app.post("/analyze")
async def analyze_policy(req: AnalyzeRequest):
    state_rules_text = ""
    for state in req.states:
        state_upper = state.upper()
        if state_upper in STATE_RULES:
            rules = STATE_RULES[state_upper]
            state_rules_text += f"\n{rules['name']}:\n"
            for rule in rules["rules"]:
                state_rules_text += f"  - {rule}\n"

    prompt = f"""You are a senior HR compliance attorney. 

Here is a company remote work policy:
---
{req.policy_text}
---

Here are the labor laws for states where their employees work:
{state_rules_text}

Compare the policy against each state's rules. Identify compliance gaps, missing clauses, or outdated language.

Return ONLY a JSON array with this exact structure, no other text:
[
  {{
    "state": "State Name",
    "state_code": "XX",
    "section": "Policy section name",
    "severity": "critical|recommended|compliant",
    "issue": "Plain English explanation of the problem",
    "suggestion": "Specific replacement or addendum text HR should add"
  }}
]

If a state is fully compliant, include it with severity "compliant" and a brief note.
"""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = message.content[0].text.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    flags = json.loads(raw)

    return {"flags": flags, "state_count": len(req.states)}


# --- 5. Generate Branded Outputs ---
class GenerateRequest(BaseModel):
    policy_text: str
    company_name: str
    states: list[str]
    change_summary: str


@app.post("/generate")
async def generate_outputs(req: GenerateRequest):
    prompt = f"""You are Canon, an AI that generates professional HR communications.

Company: {req.company_name}
Policy updated: Remote Work Policy
States affected: {", ".join(req.states)}
Summary of changes: {req.change_summary}

Generate four outputs. Return ONLY valid JSON, no other text:

{{
  "html_announcement": "A complete, self-contained HTML page announcing this policy update. Clean, professional design with dark background (#0a0a0f), accent color (#c8f060), font DM Sans. Include: company name, what changed, which states are affected, effective date (today), and who to contact with questions.",
  "slack_post": "A Slack announcement for #company-all. Use Slack markdown. 2-3 short paragraphs. Professional but human tone.",
  "email_draft": "A manager distribution email. Include subject line. Professional tone. Explain what changed, what managers need to communicate to their teams, and next steps.",
  "tldr": "One paragraph, plain English. Written for employees, not lawyers. What changed, who it affects, what they need to do."
}}
"""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = message.content[0].text.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    outputs = json.loads(raw)

    return outputs


# --- Health check ---
@app.get("/")
async def root():
    return {"status": "Canon API running", "version": "1.0.0"}
