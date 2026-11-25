# slidenoteLM

Standalone, local-first web app that turns a lecture recording (one audio or video file) and its slide PDF into one information-dense English Markdown file optimized for **NotebookLM**.

See [PLAN.md](./PLAN.md) for the phased build plan and architectural guardrails.

## Prerequisites

- **Python 3.11+** (tested with 3.13)
- **Node.js 20+** and **npm**
- **ffmpeg** (and `ffprobe`) available on your system `PATH`:
  - Windows: `winget install Gyan.FFmpeg` (then reopen your terminal)
  - macOS: `brew install ffmpeg`
- API credentials (for live transcription & OCR):
  - **Groq API key** (`GROQ_API_KEY`) for Whisper transcription
  - **Google Gemini API key** (`GEMINI_API_KEY`) — or Vertex AI (`GOOGLE_GENAI_USE_VERTEXAI=true`, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`) — for structured slide OCR (no `pdfplumber` fallback)

## Setup

### 1. Environment variables

Copy `.env.example` to `.env` and fill in your API keys (or use Bitwarden Secrets Manager below so API keys never sit in plaintext on disk):

```bash
cp .env.example .env
```

#### Optional: Bitwarden Secrets Manager (`bws`)

`backend/app/config.py` loads `.env` with `override=False`, so secrets injected by `bws run` automatically take precedence over `.env`.

1. Install the CLI (Windows):
   ```powershell
   winget install -e --id Bitwarden.BWS
   ```
   *(If your Bitwarden account is hosted in the EU region, run `bws config server-base https://vault.bitwarden.eu` once.)*
2. In Bitwarden Secrets Manager, create a project (e.g., `slidenoteLM`), add secrets named `GROQ_API_KEY` and `GEMINI_API_KEY`, and create a Machine Account with read access to that project.
3. Save the Machine Account token as a Windows User environment variable (**not** in `.env`), then reopen your terminal:
   ```powershell
   [Environment]::SetEnvironmentVariable("BWS_ACCESS_TOKEN", "<YOUR_TOKEN>", "User")
   ```
4. Add the non-secret project UUID to `.env` as `BWS_PROJECT_ID=<YOUR_PROJECT_UUID>` (leave `GROQ_API_KEY` and `GEMINI_API_KEY` out of `.env`), then launch the backend:
   ```powershell
   .\scripts\run-backend.ps1
   ```
   *(Use `.\scripts\run-backend.ps1 -SkipBws` to run directly from `.env` / stub mode without `bws`.)*

### 2. Backend

```bash
python -m venv .venv
```

Activate the virtual environment and install dependencies:

- **Windows (PowerShell):**
  ```powershell
  .\.venv\Scripts\Activate.ps1
  pip install -r backend\requirements.txt
  ```
- **macOS / Linux:**
  ```bash
  source .venv/bin/activate
  pip install -r backend/requirements.txt
  ```

### 3. Frontend

```bash
cd frontend
npm install
```

## Status

Phases 0–4 are implemented (repo setup + `ffmpeg`/`ffprobe`, FastAPI + SQLite background job skeleton, FFmpeg audio normalization + chunked parallel Groq Whisper transcription, strict Google Gemini Slide OCR with PyMuPDF batching, and alignment quality checks + AI-first NotebookLM Markdown scaffolding with deterministic stub mode). Follow [PLAN.md](./PLAN.md) for Phases 5–6.
