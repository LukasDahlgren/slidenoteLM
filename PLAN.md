# slidenoteLM — Phased Implementation Plan

Build **slidenoteLM**, a standalone, local-first web app that turns one lecture recording (audio or video) and its slide PDF into one English Markdown source preserving what a student could learn from the lecture recording and slide deck together for uploading into **NotebookLM**.

Reference repo (read-only inspiration; **copy and simplify**, never import at runtime): `C:\Users\lukas\code\TeachersNote`.

---

## Locked Decisions & Guardrails

- **Scope boundary:** The app's job ends when the user downloads the `.md` file. No chat, no course management, no user accounts/login, no direct NotebookLM integration.
- **Standalone & cross-platform:** Local-first on Windows and Mac using **SQLite + local filesystem** job storage (no Postgres, Redis, or Supabase).
- **Inputs (v1):** One slide PDF + one audio or video file (audio extracted/normalized via `ffmpeg`).
- **Output language:** English.
- **Primary deliverable & consumer:** One English Markdown source uploaded into **NotebookLM** that preserves what a student could learn from the lecture recording and slide deck together. NotebookLM handles source partitioning, retrieval, and generated study aids after upload, so optimize for source fidelity, factual completeness, explicit attribution where it matters, resolved deictic/visual references when supported by the slides, and natural structure rather than artificial self-contained chunks or repetitive boilerplate.
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
│   │       ├── notes.py         # Sectioning, context carry-over, lecture-source prompts, coverage-check contract, stub hook (LLM in Phase 6)
│   │       └── prompts.py       # English lecture-source Markdown prompt templates for NotebookLM
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

### Phase 0 — Repo Bootstrap & Dependencies *(Completed)*
- [x] Create standalone folder `C:\Users\lukas\code\slidenoteLM` and initialize Git repo.
- [x] Add `.gitignore`, `.env.example`, `README.md`, and this `PLAN.md`.
- [x] Add minimal dependency manifests (`backend/requirements.txt`, `frontend/package.json`) and basic Vite/TypeScript/ESLint config.
- [x] Create local Python `.venv`, install backend dependencies (`fastapi`, `uvicorn`, `groq`, `google-genai`, `pymupdf`, `httpx`), and install frontend dependencies (`react`, `vite`, `typescript`, `eslint`, `vitest`).
- [x] Ensure `ffmpeg` (and `ffprobe`) is installed and available on `PATH` before Phase 2.

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

### Phase 2 — Audio Prep & Groq Whisper Transcription *(Completed)*
**Goal:** Turn any uploaded audio or video recording into accurate, timestamped `transcript.json`.
- [x] Inspect TeachersNote's `backend/services/transcription_audio.py` and `backend/pipeline.py` transcription chunking for reference.
- [x] Implement `backend/app/pipeline/audio.py`:
  - Verify `ffmpeg`/`ffprobe` availability with a clear error message if missing.
  - Extract and normalize audio from audio or video inputs to 16 kHz mono MP3.
  - Split long recordings safely below Groq's upload byte limit with accurate timestamp offsets.
- [x] Implement `backend/app/pipeline/transcribe.py`:
  - Parallel Groq Whisper transcription with bounded concurrency and retry/backoff.
  - Merge chunk segments with shifted timestamps into `artifacts/transcript.json`.
- [x] Add unit tests (mocked `ffmpeg`/Groq + timestamp offset merging + real `ffmpeg`/`ffprobe` integration tests; live Groq smoke test ready once `GROQ_API_KEY` is added in `.env`).

### Phase 3 — Google Gemini Slide OCR (Strict OCR Only) *(Completed)*
**Goal:** Extract rich slide text + visual diagram descriptions from the PDF into `slides_ocr.json` with **zero** `pdfplumber` fallback.
- [x] Inspect TeachersNote's `backend/services/parse_slides_gemini.py` and `backend/services/ocr_sections.py` for reference.
- [x] Implement `backend/app/pipeline/ocr.py`:
  - Use `pymupdf` (`fitz`) to inspect page count and slice/batch PDF pages cleanly for Gemini OCR.
  - Call `google-genai` to extract structured per-slide output (`VISIBLE_TEXT`, `VISUAL_CONTEXT`, `TABLES_CODE_FORMULAS`).
  - Strip repeating headers/footers/watermarks without losing structured markers.
  - **Strict failure policy:** If `GEMINI_API_KEY` (or Vertex config) is missing or Gemini OCR fails after retries, fail the job at the `ocr` stage with an explicit, actionable error message. Never fall back to `pdfplumber`.
- [x] Add unit tests (structured marker parsing, boilerplate cleanup, strict OCR failure behavior) and verify against a real PDF if Google credentials are configured.

### Phase 4 — Alignment & AI-First Markdown Scaffolding (LLM-Deferred) *(Completed)*
**Goal:** Lock in data contracts, sectioning/context carry-over logic, NotebookLM prompt specifications, and a deterministic stub mode so the full pipeline runs end-to-end before choosing the LLM provider.
- [x] Implement `backend/app/pipeline/align.py`:
  - Port TeachersNote's structural alignment quality checks (`alignment_issues` / unresolved warnings) and define `alignment.json` schema.
  - Mark uncertain or out-of-order regions explicitly so downstream note generation falls back to timestamp-only anchors instead of dubious slide numbers.
  - Provide a clean hook (`align_with_llm(...)`) plus a deterministic stub mode (`PIPELINE_LLM_MODE=stub`) for end-to-end testing.
- [x] Implement `backend/app/pipeline/prompts.py` and `backend/app/pipeline/notes.py`:
  - Implement section chunking (grouping contiguous aligned slides/transcript segments into manageable sections, plus explicit off-slide/demo/Q&A spans) and rolling context carry-over across section boundaries.
  - Encode the full **Lecture Source Markdown Specification** (see below) in `prompts.py` (`AI_FIRST_MARKDOWN_RULES`), plus a **transcript coverage-check** prompt/contract to catch silently dropped explanations, lost meaning, or Q&A.
  - Provide a clean single hook (`generate_section_notes_with_llm(...)` + `check_coverage_with_llm(...)`) and deterministic stub mode that emits valid, structured `notes.md` from `slides_ocr.json` + `transcript.json`.
- [x] Add unit tests for sectioning, boundary context handoff, uncertainty fallback (timestamp-only), and Markdown assembly.

### Phase 5 — One-Screen Frontend UI & End-to-End Verification *(Completed)*
**Goal:** Deliver a clean, single-screen browser UI for Windows and Mac and verify the complete upload → background run → reopen link → download `.md` flow.
- [x] Build the single-screen React + Vite UI (`frontend/src/`):
  - Drag-and-drop / file pickers for **one slide PDF** and **one audio or video recording**.
  - Start processing action; automatically navigate/update URL to `/jobs/<token>` once uploaded.
  - Copyable private result link banner so the user can close the tab and return anytime.
  - Clear stage-by-stage progress indicator (`Upload` → `Audio Prep` → `Transcription (Groq)` → `Slide OCR (Gemini)` → `Alignment` → `Markdown Notes` → `Ready`).
  - Actionable error panel naming the exact failed stage and what to do next (e.g., missing `ffmpeg`, invalid API key, Gemini OCR failure).
  - Download `.md` button (plus optional lightweight preview/stats: slide count, duration, word count).
- [x] Verify end-to-end flow locally:
  - Upload fixture/real files → close tab → reopen `/jobs/<token>` → confirm background completion → download `.md`.
  - Verify stage-specific failure UI (including Gemini OCR failure with no `pdfplumber` fallback).
  - Run frontend `npm run lint` and `npm run build`, and backend `unittest` suite.

### Phase 6 — Wire Chosen LLM Provider & Live Content Verification
**Goal:** Plug Google Gemini 4 Neon (`gemini-4-neon`, varied `thinking_level` per stage) cleanly into the Phase 4 hooks via `google-genai` (keeping the app at 2 providers total: Groq Whisper + Google Gemini) and verify note fidelity on a real lecture.
- [x] Bump `google-genai>=2.34.0` in `backend/requirements.txt` and add per-stage Gemini 4 Neon model & `thinking_level` env vars (`MEDIUM` for OCR, `HIGH` for Alignment & Notes, `MEDIUM` for Coverage Audit) to `.env.example` and `backend/app/config.py`.
- [x] Implement the single-provider Gemini 4 calls inside `backend/app/pipeline/gemini_client.py`, `backend/app/pipeline/ocr.py`, `backend/app/pipeline/align.py`, and `backend/app/pipeline/notes.py` (no multi-provider wrapper bloat), with unit tests in `backend/tests/test_phase6_gemini_llm.py`.
- [ ] Run end-to-end on a real lecture (PDF + recording).
- [ ] **Speech-only content check:**
  - Pick **3 details found only in the speech** (not written on the slides), including **at least 1 requiring following the teacher's reasoning chain**.
  - Turn them into questions and verify that the exported `notes.md` clearly and accurately answers all 3.
  - Inspect a real exported section for visual grounding, timestamp accuracy, and absence of hallucinated steps or false exam-importance tags.

---

## Lecture Source Markdown Specification (for Phase 4 Prompts & Phase 6 LLM)

The deliverable (`notes.md`) is one English Markdown source uploaded to **NotebookLM** that preserves what a student could learn from the lecture recording and slide deck together. NotebookLM handles source partitioning, retrieval, and generated study aids after upload, so `backend/app/pipeline/prompts.py` (`AI_FIRST_MARKDOWN_RULES`, `SECTION_NOTES_SYSTEM_PROMPT`, and `build_section_notes_prompt`) prioritizes **source fidelity, natural structure, and teaching depth** over artificial chunking or repetitive boilerplate.

1. **Length driven by teaching:** No fixed sentence counts, bullet caps, or per-slide word quotas. Let the depth of the teaching determine length: a title or transition slide may take one concise line, whereas a 12-minute diagram walkthrough or derivation requires detailed, multi-part coverage.
2. **Clear exposition and resolved references:** Write clear, direct, information-dense English with explicit causal links rather than padded narrative. Resolve deictic and visual references ("this arrow", "that curve", "here", "the box on the right") to the concrete concept or visual element when the slide or surrounding explanation establishes their meaning, and note uncertainty when the intended target is ambiguous. Do not force every paragraph to repeat concept names or stand as an isolated self-contained chunk.
3. **Preserve reasoning chains in sequence:** Record multi-step derivations, proofs, debugging walkthroughs, and worked explanations in exact logical sequence, including steps that cross slide boundaries. If the lecturer walks through three steps and explains why the third step fails, capture every step, what happened, and why it failed. Never compress a worked example into a vague summary such as "the lecturer demonstrated a failure."
4. **Substantive spoken content and attribution:** Preserve every substantive spoken explanation: mechanisms, concrete examples, analogies, definitions, exceptions, edge cases, caveats, and substantive student Q&A together with the lecturer's answers. Strip only spoken filler, false starts, operational chatter, and empty repetition while retaining any new nuance or clarification added by a second explanation. Integrate slide content, visuals, and speech together rather than splitting them into separate blocks for every topic; identify the source explicitly whenever attribution matters—such as when a claim appears only on the slide, a student proposes an interpretation, or speech conflicts with a diagram—and preserve who asked, asserted, corrected, or speculated whenever that distinction affects meaning.
5. **Connect speech to visuals by what they teach:** Use Gemini OCR `VISUAL_CONTEXT` (together with `VISIBLE_TEXT` and `TABLES_CODE_FORMULAS`) to ground spoken references to figures, curves, arrows, tables, or diagrams when supported by the slide and recording, and explicitly note uncertainty when ambiguous or unconfirmed. Choose the Markdown representation by what the visual teaches: prose to capture a relationship, a Markdown table to preserve values and units, a numbered sequence to explain a flow or process, and an ASCII or Mermaid diagram only when the supplied evidence establishes its connections and a drawing conveys them more clearly.
6. **Assessment cues and scope boundaries:** Keep explicit assessment requirements, exam hints, and explicit scope limits (what is in scope, out of scope, or not required to memorize) beside the relevant concept with a `[MM:SS–MM:SS]` timestamp rather than moving them into a separate checklist. Preserve the lecturer's level of certainty accurately: "important to understand" alone is a study cue, and "probably on the exam" carries a different commitment from "will be on the exam." Time spent on a topic alone is never evidence that it will be assessed.
7. **Strict grounding, discrepancies, and self-corrections:** Preserve technical terms, formulas, source-supplied variable definitions, symbols, numbers, units, code snippets, and qualifications accurately. Never invent missing logical steps, never present a paraphrase as a direct quotation, and omit unknown metadata (use only source-supported course details). Apply a meaning-based rule to ASR discrepancies and self-corrections: use the slide's spelling for an obvious transcription error when the intended term is clear; retain any self-correction that changes the explanation; and when two plausible readings or a conflict between speech and a slide or diagram would teach different things, identify what the transcript and slide each support and leave the uncertainty visible, adding a short discrepancy note only when a reader needs it to interpret the teaching.
8. **Natural topic headings and source anchors:**
   - Use descriptive natural topic headings naming the concrete concept explained, with confirmed slide references (`Slide N` / `Slides A–B`) and `[MM:SS–MM:SS]` timestamps.
   - Let topics span multiple slides when several slides support one argument, and let substantive off-slide discussion, live demos, and student Q&A stand on their `[MM:SS–MM:SS]` timestamp.
   - Whenever slide alignment is uncertain (unresolved alignment warnings, out-of-order jumps, or unaligned spans), use the `[MM:SS–MM:SS]` timestamp alone instead of forcing a dubious slide number.
   - Do not repeat full heading hierarchies or fill out empty categories or fixed per-slide blocks.
   - Leave generated study aids (such as a formula glossary or exam checklist) to NotebookLM: keep formulas, source-supplied variable definitions, and assessment guidance where they were taught, and preserve a concluding recap only if the lecturer or deck actually provides one.
9. **Long-lecture continuity and reliability:** Process in manageable sections through `notes.py`, carry rolling context and immediate prior boundary transcript overlap across section boundaries so explanations crossing slide or section boundaries remain coherent without repeating already-documented material, and run a transcript coverage check so substantive explanations, steps, qualifications, attribution, scope limits, corrections, or student Q&A are not silently lost.

### Transcript Coverage Check (`COVERAGE_CHECK_SYSTEM_PROMPT` / `build_coverage_check_prompt`)

- Audits each section's chronological transcript against the generated Markdown to catch silently dropped substantive content or loss of meaning.
- Treats a missing step, a lost qualification, a change from tentative to definite, missing meaningful attribution (who asked, asserted, corrected, or speculated), omitted assessment cues/scope limits/certainty levels/timestamps, or omitted explanation-changing self-corrections or substantive slide/speech discrepancies as a loss of meaning.
- Does **not** flag a clearly explained point that appears once in integrated form even when supported by both slide and speech, obvious ASR transcription typos normalized to the clear slide spelling, spoken filler/operational chatter/empty repetition, or omitted generated study aids and unknown metadata.
