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
3. Add the non-secret project UUID to `.env` as `BWS_PROJECT_ID=<YOUR_PROJECT_UUID>` (leave `GROQ_API_KEY`, `GEMINI_API_KEY`, and `BWS_ACCESS_TOKEN` out of `.env` — and do **not** save `BWS_ACCESS_TOKEN` as a persistent OS environment variable).
4. Launch the backend:
   ```powershell
   .\scripts\run-backend.ps1
   ```
   The script prompts for `BWS_ACCESS_TOKEN` (masked input) once per terminal session and keeps it only in that terminal's memory until you close the window.
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

## Deploying to Railway

The repository includes a multi-stage [`Dockerfile`](./Dockerfile) and [`railway.json`](./railway.json) for single-container deployment: Vite builds `frontend/dist`, and FastAPI serves both the SPA and `/api/*` with `ffmpeg` and the Bitwarden Secrets Manager (`bws`) CLI installed, honoring Railway's `$PORT` and healthchecking `/api/health`.

1. **Create a Railway service** from this GitHub repo (Railway automatically detects `railway.json` and `Dockerfile`).
2. **Attach a Persistent Volume** mounted at `/data` so the SQLite database and generated Markdown notes survive redeploys, and set `DATA_DIR=/data` in Railway Variables.
3. **Configure secrets & environment variables** in Railway:
   - **Recommended — Bitwarden Secrets Manager (`bws`)** (keeps API keys and app password solely in Bitwarden):
     - `DATA_DIR=/data`
     - `BWS_PROJECT_ID=<your-slidenoteLM-project-uuid>` *(non-secret Bitwarden project UUID)*
     - `BWS_ACCESS_TOKEN=<machine-account-access-token>` *(scoped read-only to the `slidenoteLM` Bitwarden project)*
     - `PIPELINE_LLM_MODE=stub` *(or your configured LLM mode)*
     - Store `GROQ_API_KEY`, `GEMINI_API_KEY` *(or Vertex AI vars)*, and `APP_PASSWORD` *(plus optional `APP_SESSION_SECRET`)* solely in your Bitwarden `slidenoteLM` project; the container starts via `bws run --project-id "$BWS_PROJECT_ID"` and injects them into `uvicorn` at runtime. *(Setting `BWS_ACCESS_TOKEN` without `BWS_PROJECT_ID` fails startup loudly instead of running without secrets.)*
   - **Alternative — Direct Railway Variables** (no Bitwarden):
     - Omit `BWS_ACCESS_TOKEN` and the container launches `uvicorn` directly using Railway Variables:
       - `DATA_DIR=/data`
       - `APP_PASSWORD=<choose-a-strong-password>` *(enables the built-in login gate; leave unset locally to disable auth)*
       - `APP_SESSION_SECRET=` *(optional; when empty, session cookie signing derives a stable key from `APP_PASSWORD` so restarts keep sessions valid)*
       - `GROQ_API_KEY=<your-groq-key>`
       - `GEMINI_API_KEY=<your-gemini-key>` *(or Vertex AI vars)*
       - `PIPELINE_LLM_MODE=stub` *(or your configured LLM mode)*
4. **Custom domain with Cloudflare (important for large recordings):**
   - If you route a custom domain through Cloudflare, set the DNS record to **DNS only (grey cloud)** instead of Proxied (orange cloud). Cloudflare Free enforces a **100 MB** upload body limit that would reject larger lecture videos, whereas DNS-only lets uploads go straight to Railway.

## Status

Phases 0–4 are implemented (repo setup + `ffmpeg`/`ffprobe`, FastAPI + SQLite background job skeleton, FFmpeg audio normalization + chunked parallel Groq Whisper transcription, strict Google Gemini Slide OCR with PyMuPDF batching, and alignment quality checks + AI-first NotebookLM Markdown scaffolding with deterministic stub mode). Follow [PLAN.md](./PLAN.md) for Phases 5–6.
