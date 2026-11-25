# slidenoteLM — Phased Implementation Plan

Build **slidenoteLM**, a standalone, local-first web app that turns one lecture recording (audio or video) and its slide PDF into one information-dense English Markdown file optimized for uploading into **NotebookLM**.

Reference repo (read-only inspiration; **copy and simplify**, never import at runtime): `C:\Users\lukas\code\TeachersNote`.

---

## Locked Decisions & Guardrails

- **Scope boundary:** The app's job ends when the user downloads the `.md` file. No chat, no course management, no user accounts/login, no direct NotebookLM integration.
- **Standalone & cross-platform:** Local-first on Windows and Mac using **SQLite + local filesystem** job storage (no Postgres, Redis, or Supabase).
- **Inputs (v1):** One slide PDF + one audio or video file (audio extracted/normalized via `ffmpeg`).
- **Output language:** English.
- **Primary reader:** **NotebookLM (AI retrieval & Q&A)**, not human-polished textbook prose. Optimize for factual completeness, explicitness, self-contained chunks, and zero ambiguity over stylistic flow.
- **Fixed providers:**
  - **Transcription:** Groq Whisper (chunked/parallelized as needed).
  - **Slide extraction:** Google Gemini OCR **only** (capturing visible text, visual/diagram context, tables, code, and formulas). **Never fall back to `pdfplumber`.** If Gemini OCR fails or credentials are missing, fail clearly at the OCR stage with actionable next steps.
- **Deferred LLM provider:** Do **not** build a multi-provider LLM abstraction upfront. Keep alignment and Markdown note generation behind clean stage interfaces + deterministic stub mode until the single target LLM provider is chosen in Phase 6.
- **Do not reuse TeachersNote's enrichment rules:** TeachersNote's `scripts/enrich_prompt.py` / `enrich_policy.py` enforce bullet caps, strip/truncate off-slide discussion, and isolate slide batches. `slidenoteLM` uses a dedicated section-aware Markdown generator.

---

## Planned Repository Layout

```text
slidenoteLM/
├── PLAN.md                      # This phased plan
├── README.md                    # Local setup, prerequisites, run commands
├── .env.example                 # Environment variable template
├── .gitignore
├── backend/
│   ├── requirements.txt
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI routes (upload, job status, download)
│   │   ├── config.py            # Env loading & paths
│   │   ├── db.py                # SQLite schema & job persistence
│   │   ├── worker.py            # Background job runner & stage transitions
│   │   ├── schemas.py           # API response / artifact dataclasses
│   │   └── pipeline/
│   │       ├── __init__.py
│   │       ├── audio.py         # ffmpeg audio extraction, normalization, chunking
│   │       ├── transcribe.py    # Groq Whisper chunked transcription
│   │       ├── ocr.py           # Google Gemini structured slide OCR (no pdfplumber)
│   │       ├── align.py         # Alignment contract, quality checks, stub hook (LLM in Phase 6)
│   │       ├── notes.py         # Sectioning, context carry-over, AI-first prompts, coverage-check contract, stub hook (LLM in Phase 6)
│   │       └── prompts.py       # English NotebookLM-optimized prompt templates
│   └── tests/                   # unittest test suite
└── frontend/
    ├── package.json
    ├── index.html
    ├── vite.config.ts
    └── src/                     # Single-screen UI (upload, progress, private link, error, download)
```

### Persisted Job Artifacts (per job under `data/jobs/<job_token>/`)

- `input/slides.pdf` — uploaded slide deck
- `input/recording.<ext>` — uploaded audio or video recording
- `artifacts/audio_normalized.mp3` — 16 kHz mono MP3 prepared for Whisper
- `artifacts/transcript.json` — timestamped Whisper segments (`[{index, start, end, text}, ...]`)
- `artifacts/slides_ocr.json` — structured Gemini OCR per slide (`visible_text`, `visual_context`, `tables_code_formulas`)
- `artifacts/alignment.json` — slide-to-segment boundaries + structural quality/uncertainty flags
- `artifacts/notes.md` — final downloadable Markdown file

---

## Phased Build Plan

### Phase 0 — Repo Bootstrap & Dependencies *(Completed — pending `ffmpeg` on PATH)*
- [x] Create standalone folder `C:\Users\lukas\code\slidenoteLM` and initialize Git repo.
- [x] Add `.gitignore`, `.env.example`, `README.md`, and this `PLAN.md`.
- [x] Add minimal dependency manifests (`backend/requirements.txt`, `frontend/package.json`) and basic Vite/TypeScript/ESLint config.
- [x] Create local Python `.venv`, install backend dependencies (`fastapi`, `uvicorn`, `groq`, `google-genai`, `pymupdf`, `httpx`), and install frontend dependencies (`react`, `vite`, `typescript`, `eslint`, `vitest`).
- [ ] Ensure `ffmpeg` (and `ffprobe`) is installed and available on `PATH` before Phase 2.

### Phase 1 — Backend Skeleton, SQLite Job Store & Upload API *(Completed)*
**Goal:** Accept one PDF + one audio/video upload, persist a job with a private unguessable token, run stages asynchronously in the background, and serve job status and `.md` downloads.
- [x] Implement `backend/app/config.py` (data directory, env vars, limits) and `backend/app/db.py` (SQLite tables for jobs, stages, status, timestamps, error stage + actionable hint, artifact paths).
- [x] Generate high-entropy private job tokens (`secrets.token_urlsafe(24)`) used for `/jobs/<token>` result links.
- [x] Implement FastAPI endpoints in `backend/app/main.py`:
  - `GET /api/health`
  - `POST /api/jobs` (multipart upload: 1 PDF + 1 audio/video file, returns `{token, result_url, status}`)
  - `GET /api/jobs/{token}` (current stage, stage history/progress, error stage + next step if failed, download readiness)
  - `GET /api/jobs/{token}/download` (streams `notes.md` with clean filename)
- [x] Implement `backend/app/worker.py` so processing runs in the background after upload finishes and survives closing the browser tab. Mark interrupted in-progress jobs clearly on restart.
- [x] Add unit/API tests covering upload validation, token lookup, background state transitions, and failure reporting.

### Phase 2 — Audio Prep & Groq Whisper Transcription
**Goal:** Turn any uploaded audio or video recording into accurate, timestamped `transcript.json`.
- [ ] Inspect TeachersNote's `backend/services/transcription_audio.py` and `backend/pipeline.py` transcription chunking for reference.
- [ ] Implement `backend/app/pipeline/audio.py`:
  - Verify `ffmpeg`/`ffprobe` availability with a clear error message if missing.
  - Extract and normalize audio from audio or video inputs to 16 kHz mono MP3.
  - Split long recordings safely below Groq's upload byte limit with accurate timestamp offsets.
- [ ] Implement `backend/app/pipeline/transcribe.py`:
  - Parallel Groq Whisper transcription with bounded concurrency and retry/backoff.
  - Merge chunk segments with shifted timestamps into `artifacts/transcript.json`.
- [ ] Add unit tests (mocked `ffmpeg`/Groq + timestamp offset merging) and verify with a real sample clip if `GROQ_API_KEY` is configured.

### Phase 3 — Google Gemini Slide OCR (Strict OCR Only)
**Goal:** Extract rich slide text + visual diagram descriptions from the PDF into `slides_ocr.json` with **zero** `pdfplumber` fallback.
- [ ] Inspect TeachersNote's `backend/services/parse_slides_gemini.py` and `backend/services/ocr_sections.py` for reference.
- [ ] Implement `backend/app/pipeline/ocr.py`:
  - Use `pymupdf` (`fitz`) to inspect page count and slice/batch PDF pages cleanly for Gemini OCR.
  - Call `google-genai` to extract structured per-slide output (`VISIBLE_TEXT`, `VISUAL_CONTEXT`, `TABLES_CODE_FORMULAS`).
  - Strip repeating headers/footers/watermarks without losing structured markers.
  - **Strict failure policy:** If `GEMINI_API_KEY` (or Vertex config) is missing or Gemini OCR fails after retries, fail the job at the `ocr` stage with an explicit, actionable error message. Never fall back to `pdfplumber`.
- [ ] Add unit tests (structured marker parsing, boilerplate cleanup, strict OCR failure behavior) and verify against a real PDF if Google credentials are configured.

### Phase 4 — Alignment & AI-First Markdown Scaffolding (LLM-Deferred)
**Goal:** Lock in data contracts, sectioning/context carry-over logic, NotebookLM prompt specifications, and a deterministic stub mode so the full pipeline runs end-to-end before choosing the LLM provider.
- [ ] Implement `backend/app/pipeline/align.py`:
  - Port TeachersNote's structural alignment quality checks (`alignment_issues` / unresolved warnings) and define `alignment.json` schema.
  - Mark uncertain or out-of-order regions explicitly so downstream note generation falls back to timestamp-only anchors instead of dubious slide numbers.
  - Provide a clean hook (`align_with_llm(...)`) plus a deterministic stub mode (`PIPELINE_LLM_MODE=stub`) for end-to-end testing.
- [ ] Implement `backend/app/pipeline/prompts.py` and `backend/app/pipeline/notes.py`:
  - Implement section chunking (grouping contiguous aligned slides/transcript segments into manageable sections, plus explicit off-slide/demo/Q&A spans) and rolling context carry-over across section boundaries.
  - Encode the full **AI-first NotebookLM Markdown Specification** (see below) in `prompts.py`, plus a **transcript coverage-check** prompt/contract to catch silently dropped explanations or Q&A.
  - Provide a clean single hook (`generate_section_notes_with_llm(...)` + `check_coverage_with_llm(...)`) and deterministic stub mode that emits valid, structured `notes.md` from `slides_ocr.json` + `transcript.json`.
- [ ] Add unit tests for sectioning, boundary context handoff, uncertainty fallback (timestamp-only), and Markdown assembly.

### Phase 5 — One-Screen Frontend UI & End-to-End Verification
**Goal:** Deliver a clean, single-screen browser UI for Windows and Mac and verify the complete upload → background run → reopen link → download `.md` flow.
- [ ] Build the single-screen React + Vite UI (`frontend/src/`):
  - Drag-and-drop / file pickers for **one slide PDF** and **one audio or video recording**.
  - Start processing action; automatically navigate/update URL to `/jobs/<token>` once uploaded.
  - Copyable private result link banner so the user can close the tab and return anytime.
  - Clear stage-by-stage progress indicator (`Upload` → `Audio Prep` → `Transcription (Groq)` → `Slide OCR (Gemini)` → `Alignment` → `Markdown Notes` → `Ready`).
  - Actionable error panel naming the exact failed stage and what to do next (e.g., missing `ffmpeg`, invalid API key, Gemini OCR failure).
  - Download `.md` button (plus optional lightweight preview/stats: slide count, duration, word count).
- [ ] Verify end-to-end flow locally:
  - Upload fixture/real files → close tab → reopen `/jobs/<token>` → confirm background completion → download `.md`.
  - Verify stage-specific failure UI (including Gemini OCR failure with no `pdfplumber` fallback).
  - Run frontend `npm run lint` and `npm run build`, and backend `unittest` suite.

### Phase 6 — Wire Chosen LLM Provider & Live Content Verification
**Goal:** Once you pick your exact LLM provider for alignment and note generation, plug that single SDK cleanly into the Phase 4 hooks and verify note fidelity on a real lecture.
- [ ] Add the chosen provider SDK to `backend/requirements.txt` and its env var(s) to `.env.example`.
- [ ] Implement the single-provider calls inside `backend/app/pipeline/align.py` and `backend/app/pipeline/notes.py` (no multi-provider wrapper bloat).
- [ ] Run end-to-end on a real lecture (PDF + recording).
- [ ] **Speech-only content check:**
  - Pick **3 details found only in the speech** (not written on the slides), including **at least 1 requiring following the teacher's reasoning chain**.
  - Turn them into questions and verify that the exported `notes.md` clearly and accurately answers all 3.
  - Inspect a real exported section for visual grounding, timestamp accuracy, and absence of hallucinated steps or false exam-importance tags.

---

## AI-First Markdown Specification (for Phase 4 Prompts & Phase 6 LLM)

The primary consumer of `notes.md` is **NotebookLM (AI retrieval & Q&A)**. Prioritize **completeness, explicitness, retrieval clarity, and zero ambiguity** over literary prose or visual prettiness.

1. **Length driven by teaching:** No fixed sentence counts or per-slide word quotas. A title slide may take one line; a 12-minute diagram explanation may take dense, detailed multi-part coverage.
2. **Explicit over stylistic:** Prefer crisp, structured, information-dense statements, step-by-step sequences, and explicit causal links over flowing narrative. Replace ambiguous pronouns ("it", "this", "that method") with explicit concept names so isolated chunks stay unambiguous when retrieved by NotebookLM.
3. **Preserve reasoning in sequence:** If the teacher walks through three steps and explains why the third fails, record every step, what happened, and why. Never compress a worked example into "the teacher demonstrated a failure."
4. **Keep substantive spoken content:** Preserve concrete examples, analogies, definitions, exceptions, edge cases, corrections, and substantive student Q&A with answers. Strip spoken filler, false starts, operational chatter, and empty repetition while keeping any new nuance a second explanation adds.
5. **Connect speech to visuals:** Use Gemini OCR `VISUAL_CONTEXT` so references like "this arrow" or "the second curve" identify the specific visual element when supported by the slide and recording. Explicitly note uncertainty when ambiguous.
6. **Accurate exam/study guidance:** Tag exam hints only when the lecturer explicitly says so. Time spent on a topic alone is never evidence that it will be assessed.
7. **Strict grounding:** Preserve technical terms, equations, numbers, units, code, and qualifications verbatim/accurately. Never invent missing steps, silently resolve contradictions, or present a paraphrase as a direct quote.
8. **Chunk-friendly Markdown structure:**
   - Lecture title + descriptive topic headings naming the concrete concept explained.
   - Slide numbers and `[MM:SS–MM:SS]` timestamps where available; let topics span multiple slides when the explanation does.
   - Dedicated timestamped sections for substantive off-slide discussion, live demos, and student Q&A.
   - Distinguish slide content vs. spoken-only additions where helpful for grounding.
   - When slide alignment is uncertain (unresolved alignment warnings or out-of-order jumps), use the timestamp alone instead of forcing a dubious slide number.
9. **Long-lecture reliability:** Process in manageable sections, carry rolling context across section boundaries so multi-slide examples stay coherent, and run a transcript coverage check so substantive explanations or Q&A are not silently dropped.
