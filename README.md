# slidenoteLM

Standalone, local-first web app that turns a lecture recording (one audio or video file) and its slide PDF into one information-dense English Markdown file optimized for **NotebookLM**.

See [PLAN.md](./PLAN.md) for the phased build plan and architectural guardrails.

## Prerequisites

- **Python 3.11+** (tested with 3.13)
- **Node.js 20+** and **npm**
- **ffmpeg** (and `ffprobe`) available on your system `PATH`:
  - Windows: `winget install Gyan.FFmpeg` (then reopen your terminal)
  - macOS: `brew install ffmpeg`
- API credentials (for live transcription, OCR, alignment, and note generation):
  - **Groq API key** (`GROQ_API_KEY`) for Whisper transcription
  - **Google Gemini API key** (`GEMINI_API_KEY`) — or Vertex AI (`GOOGLE_GENAI_USE_VERTEXAI=true`, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`) — for **Gemini 4 Neon** (`gemini-4-neon`) across structured slide OCR (`MEDIUM` thinking; no `pdfplumber` fallback), alignment (`HIGH` thinking), section Markdown notes (`HIGH` thinking), and transcript coverage audits (`MEDIUM` thinking). *(Note: ensure your Gemini API billing tier / Google AI Studio plan has access to your configured Gemini model.)*

## Setup

### 1. Environment variables

Copy `.env.example` to `.env` and fill in your API keys (or use Bitwarden Secrets Manager below so API keys never sit in plaintext on disk):

```bash
cp .env.example .env
```

#### Optional: Bitwarden Secrets Manager (`bws`)

`backend/app/config.py` loads `.env` with `override=False`, so secrets injected by `bws run` automatically take precedence over `.env`.

1. Install the `bws` CLI:
   - **Windows:**
     ```powershell
     winget install -e --id Bitwarden.BWS
     ```
   - **macOS (Apple Silicon):**
     ```bash
     curl -fsSL "https://github.com/bitwarden/sdk-sm/releases/download/bws-v2.1.0/bws-aarch64-apple-darwin-2.1.0.zip" -o /tmp/bws.zip
     unzip -o /tmp/bws.zip -d /tmp/bws_extract
     install -m 755 /tmp/bws_extract/bws ~/.local/bin/bws
     rm -rf /tmp/bws.zip /tmp/bws_extract
     ```
   *(If your Bitwarden account is hosted in the EU region, run `bws config server-base https://vault.bitwarden.eu` once.)*
2. In Bitwarden Secrets Manager, create a project (e.g., `slidenoteLM`), add secrets named `GROQ_API_KEY` and `GEMINI_API_KEY`, and create a Machine Account with read access to that project.
3. Add the non-secret project UUID to `.env` as `BWS_PROJECT_ID=<YOUR_PROJECT_UUID>` (leave `GROQ_API_KEY`, `GEMINI_API_KEY`, and `BWS_ACCESS_TOKEN` out of `.env` — and do **not** save `BWS_ACCESS_TOKEN` in persistent OS environment variables or `~/.zshrc`).
4. Launch the backend:
   - **Windows (PowerShell):**
     ```powershell
     .\scripts\run-backend.ps1
     ```
     *(Use `.\scripts\run-backend.ps1 -SkipBws` to run directly from `.env` without `bws`.)*
   - **macOS / Linux:**
     ```bash
     ./scripts/run-backend.sh
     ```
     *(Use `./scripts/run-backend.sh --skip-bws` to run directly from `.env` without `bws`.)*
   Both scripts prompt for `BWS_ACCESS_TOKEN` with hidden input so the token is never written to disk or shell history.

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
       - `PIPELINE_LLM_MODE=gemini` *(uses `gemini-4-neon` with varied thinking levels; set `stub` for deterministic offline testing)*
4. **Custom domain with Cloudflare (important for large recordings):**
   - If you route a custom domain through Cloudflare, set the DNS record to **DNS only (grey cloud)** instead of Proxied (orange cloud). Cloudflare Free enforces a **100 MB** upload body limit that would reject larger lecture videos, whereas DNS-only lets uploads go straight to Railway.

## API Rate Limits

All `/api/*` endpoints use a built-in per-IP sliding-window rate limiter (emitting `X-RateLimit-*` and `Retry-After` headers), and outbound Groq Whisper / Google Gemini calls include configurable RPM pacing. Defaults (per 60s window) can be tuned or disabled (`0` or `RATE_LIMIT_ENABLED=false`) via environment variables documented in [`.env.example`](./.env.example):

- `POST /api/auth/login`: `10/min` (`RATE_LIMIT_AUTH_LOGIN_PER_MINUTE`)
- `/api/auth/*`: `60/min` (`RATE_LIMIT_AUTH_PER_MINUTE`)
- `POST /api/jobs`: `10/min` (`RATE_LIMIT_JOBS_CREATE_PER_MINUTE`)
- `GET /api/jobs/*` (status, preview, download): `120/min` (`RATE_LIMIT_JOBS_READ_PER_MINUTE`)
- `GET /api/health` & other `/api/*`: `120/min` (`RATE_LIMIT_HEALTH_PER_MINUTE`, `RATE_LIMIT_DEFAULT_PER_MINUTE`)
- Groq uses a 20 RPM limit. Gemini OCR uses 60 RPM and slide alignment, notes, and coverage share a 90 RPM limit (capped overall by `GEMINI_COMBINED_RATE_LIMIT_RPM=120`).

Multiple queued lectures process concurrently across 4 multi-slot stage lanes (`audio_prep`: 2, `transcription`: 2, `ocr`: 3, `llm`: 3; up to 4 lectures in flight at once). Slide OCR runs alongside audio preparation and transcription, and alignment starts once both are ready.

Gemini drafts up to nine lecture sections at once. Each worker receives a slide outline and the previous four transcript segments for continuity. Coverage checks start as drafts finish, and the final document follows lecture order. Short off-slide remarks between slides stay in the surrounding section with their own timestamp. Sections under 200 transcript characters use a local coverage check.

| Setting | Default |
| --- | ---: |
| `QUEUE_MAX_IN_FLIGHT_JOBS` | 4 |
| `QUEUE_LANE_AUDIO_PREP_CONCURRENCY` | 2 |
| `QUEUE_LANE_TRANSCRIPTION_CONCURRENCY` | 2 |
| `QUEUE_LANE_OCR_CONCURRENCY` | 3 |
| `QUEUE_LANE_LLM_CONCURRENCY` | 3 |
| `OCR_MAX_PAGES_PER_BATCH` | 8 |
| `OCR_PARALLEL_WORKERS` | 8 |
| `OCR_GLOBAL_MAX_CONCURRENT` | 16 |
| `OCR_RATE_LIMIT_RPM` | 60 |
| `LLM_NOTES_MAX_OUTPUT_TOKENS` | 49152 |
| `LLM_GLOBAL_MAX_CONCURRENT` | 24 |
| `LLM_RATE_LIMIT_RPM` | 90 |
| `GEMINI_COMBINED_RATE_LIMIT_RPM` | 120 |
| `LLM_NOTES_PARALLEL_WORKERS` | 9 |
| `LLM_COVERAGE_PARALLEL_WORKERS` | 4 |
| `NOTES_MAX_SLIDES_PER_SECTION` | 8 |
| `NOTES_MAX_SECTION_DURATION_SECONDS` | 900.0 |
| `NOTES_MAX_SECTION_TRANSCRIPT_CHARS` | 14000 |
| `NOTES_INLINE_OFF_SLIDE_MAX_SECONDS` | 35.0 |
| `NOTES_INLINE_OFF_SLIDE_MAX_CHARS` | 300 |
| `NOTES_MIN_AUDIT_TRANSCRIPT_CHARS` | 200 |


## Status

Phases 0–5 and Phase 6's Google Gemini 4 Neon wiring are completed (repo setup + `ffmpeg`/`ffprobe`, FastAPI + SQLite background job skeleton, FFmpeg audio normalization + chunked parallel Groq Whisper transcription, strict Google Gemini Slide OCR with PyMuPDF batching, live Gemini 4 Neon alignment + AI-first NotebookLM Markdown generation + coverage audit with varied `thinking_level`s and deterministic `stub` fallback, and the single-screen React + Vite UI with optional password gate, rate limiting, and Railway deployment support). Remaining in [PLAN.md](./PLAN.md): live end-to-end verification on a real lecture + 3-question speech-only content check.
