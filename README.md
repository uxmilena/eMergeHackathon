# Canon — Policy OS

> One source. Every version. Always on brand.

AI-powered HR policy compliance tool. Upload your company policies, select your employee states, and Canon flags compliance gaps, suggests edits, routes approvals, and generates branded outputs — automatically.

---

## Team

| Role | Zone |
|------|------|
| Miu — Design & Frontend | `/frontend` |
| Teammate — Backend & API | `/backend` |

---

## Structure

```
canon/
├── frontend/       ← UI (HTML/CSS/JS)
│   └── index.html
├── backend/        ← API (Python/FastAPI)
│   ├── main.py
│   ├── requirements.txt
│   └── uploads/
└── README.md
```

---

## Getting Started

### Backend
```bash
cd backend
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here
uvicorn main:app --reload --port 8000
```

### Frontend
```bash
cd frontend
# open index.html in browser
# or use Live Server in VS Code / Cursor
```

---

## Demo Flow

1. HR uploads Remote Work Policy PDF
2. Selects states where employees are located
3. Canon pulls federal + state regulations
4. AI compares policy → flags compliance gaps
5. HR reviews suggestions → sends to Legal
6. Legal approves
7. Canon generates branded PDF, HTML page, Slack post, email
8. Version logged with full audit trail

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/upload` | Upload + parse PDF |
| GET | `/regulations` | Pull Federal Register data |
| POST | `/analyze` | AI compliance comparison |
| POST | `/generate` | Generate branded outputs |

---

## Tech Stack

- **Frontend:** HTML / CSS / JS
- **Backend:** Python 3.11 + FastAPI
- **AI:** Anthropic Claude API (claude-sonnet-4-20250514)
- **PDF:** PyMuPDF
- **HTTP:** httpx
