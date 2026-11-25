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
  - **Google Gemini API key** (`GEMINI_API_KEY`) for structured slide OCR (no `pdfplumber` fallback)

## Setup

### 1. Environment variables

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

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

Currently bootstrapped at **Phase 0** (folder structure, starter files, dependency manifests, and phased plan). Follow [PLAN.md](./PLAN.md) phase by phase to implement and verify each stage.
