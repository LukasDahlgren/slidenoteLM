import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { App } from './App'
import { HISTORY_STORAGE_KEY } from './history'
import type {
  JobCreateResponse,
  JobStatusResponse,
  QueueItemResponse,
  QueueSnapshotResponse,
} from './types'

function mockJsonResponse(
  body: unknown,
  status = 200,
  extraHeaders: Record<string, string> = {},
): Response {
  return new Response(status === 204 ? null : JSON.stringify(body), {
    status,
    headers: status === 204 ? extraHeaders : { 'Content-Type': 'application/json', ...extraHeaders },
  })
}

const AUTH_DISABLED = { auth_required: false, authenticated: true }
const AUTH_LOCKED = { auth_required: true, authenticated: false }
const AUTH_UNLOCKED = { auth_required: true, authenticated: true }

function makeJobStatus(overrides: Partial<JobStatusResponse> = {}): JobStatusResponse {
  const token = overrides.token ?? 'tok_abc123'
  return {
    token,
    result_url: `/jobs/${token}`,
    status: 'processing',
    current_stage: 'transcription',
    current_stage_label: 'Transcription (Groq)',
    progress_pct: 42,
    slides_filename: 'lecture-01.pdf',
    recording_filename: 'lecture-01.mp4',
    created_at: '2026-11-25T20:00:00Z',
    updated_at: '2026-11-25T20:00:10Z',
    started_at: '2026-11-25T20:00:01Z',
    completed_at: null,
    download_ready: false,
    download_url: null,
    download_filename: null,
    stages: [
      {
        stage: 'upload',
        label: 'Upload',
        status: 'completed',
        started_at: '2026-11-25T20:00:00Z',
        completed_at: '2026-11-25T20:00:01Z',
        message: 'Files uploaded.',
      },
      {
        stage: 'audio_prep',
        label: 'Audio Prep',
        status: 'completed',
        started_at: '2026-11-25T20:00:01Z',
        completed_at: '2026-11-25T20:00:04Z',
        message: 'Normalized audio to 32k mono MP3.',
      },
      {
        stage: 'transcription',
        label: 'Transcription (Groq)',
        status: 'running',
        started_at: '2026-11-25T20:00:04Z',
        completed_at: null,
        message: 'Transcribing chunk 1/2...',
      },
      {
        stage: 'ocr',
        label: 'Slide OCR (Gemini)',
        status: 'pending',
        started_at: null,
        completed_at: null,
        message: null,
      },
      {
        stage: 'alignment',
        label: 'Alignment',
        status: 'pending',
        started_at: null,
        completed_at: null,
        message: null,
      },
      {
        stage: 'notes',
        label: 'Markdown Notes',
        status: 'pending',
        started_at: null,
        completed_at: null,
        message: null,
      },
    ],
    error: null,
    stats: {},
    warnings: [],
    pipeline_role: 'primary',
    queue_position: 1000,
    queue_index: null,
    waiting_reason: null,
    can_cancel: true,
    can_retry: false,
    ...overrides,
  }
}

function makeQueueItem(overrides: Partial<QueueItemResponse> = {}): QueueItemResponse {
  const token = overrides.token ?? 'tok_queued_1'
  return {
    token,
    result_url: `/jobs/${token}`,
    status: 'queued',
    current_stage: 'upload',
    current_stage_label: 'Upload',
    progress_pct: 0,
    slides_filename: 'lecture-03.pdf',
    recording_filename: 'lecture-03.mp4',
    queue_label: null,
    batch_id: null,
    queue_position: 3000,
    queue_index: 1,
    pipeline_role: null,
    waiting_reason: null,
    created_at: '2026-11-25T20:05:00Z',
    updated_at: '2026-11-25T20:05:00Z',
    started_at: null,
    completed_at: null,
    download_ready: false,
    download_url: null,
    download_filename: null,
    error: null,
    stats: {},
    can_cancel: true,
    can_retry: false,
    ...overrides,
  }
}

function makeQueueSnapshot(
  overrides: Partial<QueueSnapshotResponse> = {},
): QueueSnapshotResponse {
  const active = overrides.active ?? []
  const queued = overrides.queued ?? []
  const recent = overrides.recent ?? []
  return {
    paused: false,
    pause_reason: null,
    max_in_flight_jobs: 2,
    max_queue_depth: 50,
    counts: {
      processing: active.length,
      queued: queued.length,
      completed: recent.filter((item) => item.status === 'completed').length,
      failed: recent.filter((item) => item.status === 'failed').length,
      cancelled: recent.filter((item) => item.status === 'cancelled').length,
    },
    active,
    queued,
    recent,
    ...overrides,
  }
}

interface MockUploadResult {
  status: number
  body?: unknown
  headers?: Record<string, string>
  networkError?: boolean
  progressEvents?: Array<{ loaded: number; total: number }>
  deferred?: Promise<Omit<MockUploadResult, 'deferred'>>
}

type UploadHandler = (formData: FormData, xhr: MockXMLHttpRequest) => MockUploadResult

class MockXMLHttpRequest {
  static handler: UploadHandler = () => ({
    status: 202,
    body: {
      token: 'tok_default',
      result_url: '/jobs/tok_default',
      status: 'queued',
      current_stage: 'upload',
      created_at: '2026-11-25T20:00:00Z',
    } satisfies JobCreateResponse,
  })
  static calls: FormData[] = []

  status = 0
  responseText = ''
  private responseHeaders: Record<string, string> = {}
  private listeners: Record<string, Array<(event?: unknown) => void>> = {}
  private uploadListeners: Record<string, Array<(event: ProgressEvent) => void>> = {}
  aborted = false

  upload = {
    addEventListener: (type: string, cb: (event: ProgressEvent) => void) => {
      if (!this.uploadListeners[type]) this.uploadListeners[type] = []
      this.uploadListeners[type].push(cb)
    },
  }

  open() {
    // no-op
  }

  setRequestHeader() {
    // no-op
  }

  addEventListener(type: string, cb: (event?: unknown) => void) {
    if (!this.listeners[type]) this.listeners[type] = []
    this.listeners[type].push(cb)
  }

  getResponseHeader(name: string): string | null {
    const key = Object.keys(this.responseHeaders).find(
      (header) => header.toLowerCase() === name.toLowerCase(),
    )
    return key ? this.responseHeaders[key] : null
  }

  abort() {
    this.aborted = true
    this.emit('abort')
  }

  send(body: FormData) {
    MockXMLHttpRequest.calls.push(body)
    const result = MockXMLHttpRequest.handler(body, this)

    const finalize = (resolved: Omit<MockUploadResult, 'deferred'>) => {
      if (this.aborted) return
      if (resolved.progressEvents) {
        for (const evt of resolved.progressEvents) {
          for (const cb of this.uploadListeners.progress ?? []) {
            cb({
              lengthComputable: true,
              loaded: evt.loaded,
              total: evt.total,
            } as ProgressEvent)
          }
        }
      }
      if (resolved.networkError) {
        this.emit('error')
        return
      }
      this.status = resolved.status
      this.responseHeaders = {
        'content-type': 'application/json',
        ...(resolved.headers ?? {}),
      }
      this.responseText = resolved.body !== undefined ? JSON.stringify(resolved.body) : ''
      this.emit('load')
    }

    if (result.deferred) {
      void result.deferred.then((resolved) => finalize(resolved))
    } else {
      queueMicrotask(() => finalize(result))
    }
  }

  private emit(type: string) {
    for (const cb of this.listeners[type] ?? []) {
      cb()
    }
  }
}

describe('slidenoteLM App', () => {
  const fetchMock = vi.fn<typeof fetch>()

  beforeEach(() => {
    fetchMock.mockReset()
    MockXMLHttpRequest.calls = []
    MockXMLHttpRequest.handler = () => ({
      status: 202,
      body: {
        token: 'tok_default',
        result_url: '/jobs/tok_default',
        status: 'queued',
        current_stage: 'upload',
        created_at: '2026-11-25T20:00:00Z',
      },
    })
    vi.stubGlobal('fetch', fetchMock)
    vi.stubGlobal('XMLHttpRequest', MockXMLHttpRequest as unknown as typeof XMLHttpRequest)
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.unstubAllGlobals()
  })

  it('renders 2 upload boxes and enables Generate Notes only when both valid files are chosen', async () => {
    fetchMock.mockImplementation(async (input) => {
      const url = String(input)
      if (url === '/api/auth/status') return mockJsonResponse(AUTH_DISABLED)
      if (url.startsWith('/api/queue')) return mockJsonResponse(makeQueueSnapshot())
      return mockJsonResponse({}, 404)
    })

    render(<App />)

    const recordingInput = await screen.findByLabelText('Video / Audio file input')
    const slidesInput = screen.getByLabelText('Slides PDF file input')
    const submitButton = screen.getByRole('button', { name: 'Generate Notes' })

    expect(submitButton).toBeDisabled()
    expect(screen.queryByRole('button', { name: 'Sign out' })).not.toBeInTheDocument()

    // Reject wrong file type in slides box
    const badSlides = new File(['hello'], 'notes.txt', { type: 'text/plain' })
    fireEvent.change(slidesInput, { target: { files: [badSlides] } })
    expect(screen.getByRole('alert')).toHaveTextContent(/valid \.pdf/i)
    expect(submitButton).toBeDisabled()

    // Choose valid recording file
    const recording = new File(['video-bytes'], 'week1.mp4', { type: 'video/mp4' })
    fireEvent.change(recordingInput, { target: { files: [recording] } })
    expect(screen.getByText('week1.mp4')).toBeInTheDocument()
    expect(submitButton).toBeDisabled()

    // Choose valid PDF slides file
    const slides = new File(['%PDF-1.4'], 'week1.pdf', { type: 'application/pdf' })
    fireEvent.change(slidesInput, { target: { files: [slides] } })
    expect(screen.getByText('week1.pdf')).toBeInTheDocument()
    expect(submitButton).toBeEnabled()
  })

  it('uploads files, polls queue & live console steps, shows Download .md when complete, and stops polling when idle', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true })

    const processingStatus = makeJobStatus()
    const completedStatus = makeJobStatus({
      status: 'completed',
      current_stage: 'notes',
      current_stage_label: 'Markdown Notes',
      progress_pct: 100,
      completed_at: '2026-11-25T20:00:20Z',
      download_ready: true,
      download_url: '/api/jobs/tok_abc123/download',
      download_filename: 'lecture-01-notes.md',
      stages: processingStatus.stages.map((stage) => ({
        ...stage,
        status: 'completed',
        message: `${stage.label} finished.`,
      })),
    })

    let uploaded = false
    let queuePollsAfterUpload = 0

    MockXMLHttpRequest.handler = () => {
      uploaded = true
      return {
        status: 202,
        progressEvents: [{ loaded: 50, total: 100 }],
        body: {
          token: 'tok_abc123',
          result_url: '/jobs/tok_abc123',
          status: 'queued',
          current_stage: 'upload',
          created_at: '2026-11-25T20:00:00Z',
        } satisfies JobCreateResponse,
      }
    }

    fetchMock.mockImplementation(async (input) => {
      const url = String(input)
      if (url === '/api/auth/status') return mockJsonResponse(AUTH_DISABLED)
      if (url.startsWith('/api/queue')) {
        if (!uploaded) return mockJsonResponse(makeQueueSnapshot())
        queuePollsAfterUpload += 1
        if (queuePollsAfterUpload === 1) {
          return mockJsonResponse(makeQueueSnapshot({ active: [processingStatus] }))
        }
        return mockJsonResponse(
          makeQueueSnapshot({
            active: [],
            recent: [
              makeQueueItem({
                token: 'tok_abc123',
                status: 'completed',
                slides_filename: 'lecture-01.pdf',
                recording_filename: 'lecture-01.mp4',
                progress_pct: 100,
                download_ready: true,
                download_url: '/api/jobs/tok_abc123/download',
                download_filename: 'lecture-01-notes.md',
                updated_at: '2026-11-25T20:00:20Z',
              }),
            ],
          }),
        )
      }
      if (url === '/api/jobs/tok_abc123') {
        return mockJsonResponse(completedStatus)
      }
      return mockJsonResponse({}, 404)
    })

    render(<App />)

    fireEvent.change(await screen.findByLabelText('Video / Audio file input'), {
      target: { files: [new File(['audio'], 'lecture-01.mp4', { type: 'video/mp4' })] },
    })
    fireEvent.change(screen.getByLabelText('Slides PDF file input'), {
      target: { files: [new File(['%PDF-1.4'], 'lecture-01.pdf', { type: 'application/pdf' })] },
    })

    fireEvent.click(screen.getByRole('button', { name: 'Generate Notes' }))

    // Console details start collapsed by default and can be expanded
    const toggleConsoleButton = await screen.findByRole('button', { name: /Show console/i })
    expect(toggleConsoleButton).toHaveAttribute('aria-expanded', 'false')
    expect(screen.queryByLabelText('Live log output')).not.toBeInTheDocument()

    fireEvent.click(toggleConsoleButton)
    expect(screen.getByRole('button', { name: /Hide console/i })).toHaveAttribute(
      'aria-expanded',
      'true',
    )

    await waitFor(() => {
      expect(screen.getAllByText(/Transcribing chunk 1\/2/i).length).toBeGreaterThan(0)
    })

    // Queue uploads stay cleanly on `/` without spamming browser history
    expect(window.location.pathname).toBe('/')
    expect(screen.queryByRole('link', { name: 'Download .md' })).not.toBeInTheDocument()

    await act(async () => {
      await vi.advanceTimersByTimeAsync(2100)
    })

    const downloadLink = await screen.findByRole('link', { name: 'Download .md' })
    expect(downloadLink).toHaveAttribute('href', '/api/jobs/tok_abc123/download')
    expect(downloadLink).toHaveAttribute('download', 'lecture-01-notes.md')
    expect(screen.getByText('lecture-01-notes.md')).toBeInTheDocument()

    // Ensure polling stops completely once idle
    const callCountAfterComplete = fetchMock.mock.calls.length
    await act(async () => {
      await vi.advanceTimersByTimeAsync(6000)
    })
    expect(fetchMock.mock.calls.length).toBe(callCountAfterComplete)
  })

  it('displays failed stage, error message, and actionable hint when a job fails', async () => {
    window.history.replaceState({}, '', '/jobs/tok_fail')

    const failedStatus = makeJobStatus({
      token: 'tok_fail',
      result_url: '/jobs/tok_fail',
      status: 'failed',
      current_stage: 'ocr',
      current_stage_label: 'Slide OCR (Gemini)',
      progress_pct: 50,
      can_retry: true,
      stages: makeJobStatus().stages.map((stage) =>
        stage.stage === 'ocr'
          ? { ...stage, status: 'failed', message: 'Gemini API key missing.' }
          : stage,
      ),
      error: {
        stage: 'ocr',
        stage_label: 'Slide OCR (Gemini)',
        message: 'GEMINI_API_KEY is not configured on the server.',
        hint: 'Set GEMINI_API_KEY in .env and restart the backend.',
      },
    })

    fetchMock.mockImplementation(async (input) => {
      const url = String(input)
      if (url === '/api/auth/status') return mockJsonResponse(AUTH_DISABLED)
      if (url.startsWith('/api/queue')) return mockJsonResponse(makeQueueSnapshot())
      if (url === '/api/jobs/tok_fail') return mockJsonResponse(failedStatus)
      return mockJsonResponse({}, 404)
    })

    render(<App />)

    await waitFor(() => {
      expect(screen.getByText(/Failed at Slide OCR \(Gemini\)/i)).toBeInTheDocument()
    })

    expect(
      screen.getByText('GEMINI_API_KEY is not configured on the server.'),
    ).toBeInTheDocument()
    expect(
      screen.getByText(/Set GEMINI_API_KEY in \.env and restart the backend\./i),
    ).toBeInTheDocument()
    expect(screen.queryByRole('link', { name: 'Download .md' })).not.toBeInTheDocument()
  })

  it('persists history in localStorage and loads a job when clicking a history entry', async () => {
    window.localStorage.setItem(
      HISTORY_STORAGE_KEY,
      JSON.stringify([
        {
          token: 'tok_saved',
          slidesFilename: 'algorithms-ch3.pdf',
          recordingFilename: 'algorithms-ch3.m4a',
          status: 'completed',
          createdAt: '2026-11-24T12:00:00Z',
          updatedAt: '2026-11-24T12:05:00Z',
          downloadReady: true,
          downloadUrl: '/api/jobs/tok_saved/download',
          downloadFilename: 'algorithms-ch3-notes.md',
        },
      ]),
    )

    const savedStatus = makeJobStatus({
      token: 'tok_saved',
      result_url: '/jobs/tok_saved',
      status: 'completed',
      current_stage: 'notes',
      current_stage_label: 'Markdown Notes',
      progress_pct: 100,
      slides_filename: 'algorithms-ch3.pdf',
      recording_filename: 'algorithms-ch3.m4a',
      download_ready: true,
      download_url: '/api/jobs/tok_saved/download',
      download_filename: 'algorithms-ch3-notes.md',
      stages: makeJobStatus().stages.map((s) => ({
        ...s,
        status: 'completed',
        message: 'Done.',
      })),
    })

    fetchMock.mockImplementation(async (input) => {
      const url = String(input)
      if (url === '/api/auth/status') return mockJsonResponse(AUTH_DISABLED)
      if (url.startsWith('/api/queue')) return mockJsonResponse(makeQueueSnapshot())
      if (url === '/api/jobs/tok_saved') return mockJsonResponse(savedStatus)
      return mockJsonResponse({}, 404)
    })

    render(<App />)

    const historyButton = await screen.findByRole('button', {
      name: /algorithms-ch3\.pdf.*algorithms-ch3\.m4a/i,
    })
    expect(historyButton).toBeInTheDocument()

    fireEvent.click(historyButton)

    await waitFor(() => {
      expect(window.location.pathname).toBe('/jobs/tok_saved')
    })

    const downloadLink = await screen.findByRole('link', { name: 'Download .md' })
    expect(downloadLink).toHaveAttribute('href', '/api/jobs/tok_saved/download')

    const persisted = JSON.parse(window.localStorage.getItem(HISTORY_STORAGE_KEY) ?? '[]')
    expect(persisted).toHaveLength(1)
    expect(persisted[0].token).toBe('tok_saved')
  })

  it('saves newly uploaded jobs into localStorage history and supports starting a new upload', async () => {
    const completedStatus = makeJobStatus({
      token: 'tok_new_history',
      result_url: '/jobs/tok_new_history',
      status: 'completed',
      current_stage: 'notes',
      current_stage_label: 'Markdown Notes',
      progress_pct: 100,
      slides_filename: 'systems-lec2.pdf',
      recording_filename: 'systems-lec2.mp3',
      download_ready: true,
      download_url: '/api/jobs/tok_new_history/download',
      download_filename: 'systems-lec2-notes.md',
    })

    let uploaded = false
    MockXMLHttpRequest.handler = () => {
      uploaded = true
      return {
        status: 202,
        body: {
          token: 'tok_new_history',
          result_url: '/jobs/tok_new_history',
          status: 'queued',
          current_stage: 'upload',
          created_at: '2026-11-25T21:00:00Z',
        } satisfies JobCreateResponse,
      }
    }

    fetchMock.mockImplementation(async (input) => {
      const url = String(input)
      if (url === '/api/auth/status') return mockJsonResponse(AUTH_DISABLED)
      if (url.startsWith('/api/queue')) {
        if (!uploaded) return mockJsonResponse(makeQueueSnapshot())
        return mockJsonResponse(
          makeQueueSnapshot({
            recent: [
              makeQueueItem({
                token: 'tok_new_history',
                status: 'completed',
                slides_filename: 'systems-lec2.pdf',
                recording_filename: 'systems-lec2.mp3',
                download_ready: true,
                download_url: '/api/jobs/tok_new_history/download',
                download_filename: 'systems-lec2-notes.md',
              }),
            ],
          }),
        )
      }
      if (url === '/api/jobs/tok_new_history') return mockJsonResponse(completedStatus)
      return mockJsonResponse({}, 404)
    })

    render(<App />)

    fireEvent.change(await screen.findByLabelText('Video / Audio file input'), {
      target: { files: [new File(['audio'], 'systems-lec2.mp3', { type: 'audio/mpeg' })] },
    })
    fireEvent.change(screen.getByLabelText('Slides PDF file input'), {
      target: { files: [new File(['%PDF-1.4'], 'systems-lec2.pdf', { type: 'application/pdf' })] },
    })

    fireEvent.click(screen.getByRole('button', { name: 'Generate Notes' }))

    await screen.findByRole('link', { name: 'Download .md' })

    const persisted = JSON.parse(window.localStorage.getItem(HISTORY_STORAGE_KEY) ?? '[]')
    expect(persisted).toHaveLength(1)
    expect(persisted[0]).toMatchObject({
      token: 'tok_new_history',
      slidesFilename: 'systems-lec2.pdf',
      recordingFilename: 'systems-lec2.mp3',
      status: 'completed',
      downloadReady: true,
    })

    fireEvent.click(screen.getByRole('button', { name: 'New upload' }))
    expect(window.location.pathname).toBe('/')
    expect(screen.queryByRole('region', { name: 'Live progress console' })).not.toBeInTheDocument()
  })

  it('shows password gate when auth is required, handles invalid password, unlocks on valid login, and signs out', async () => {
    let authed = false
    fetchMock.mockImplementation(async (input, init) => {
      const url = String(input)
      if (url === '/api/auth/status') {
        return mockJsonResponse(authed ? AUTH_UNLOCKED : AUTH_LOCKED)
      }
      if (url === '/api/auth/login') {
        const body = JSON.parse(String(init?.body ?? '{}')) as { password?: string }
        if (body.password === 'correct-pass') {
          authed = true
          return mockJsonResponse(AUTH_UNLOCKED)
        }
        return mockJsonResponse({ detail: 'Invalid password.' }, 401)
      }
      if (url === '/api/auth/logout') {
        authed = false
        return mockJsonResponse(AUTH_LOCKED)
      }
      if (url.startsWith('/api/queue')) {
        return mockJsonResponse(makeQueueSnapshot())
      }
      return mockJsonResponse({}, 404)
    })

    render(<App />)

    const passwordInput = await screen.findByLabelText('Password')
    const unlockButton = screen.getByRole('button', { name: 'Unlock' })
    expect(unlockButton).toBeDisabled()
    expect(screen.queryByLabelText('Video / Audio file input')).not.toBeInTheDocument()

    // Invalid password attempt
    fireEvent.change(passwordInput, { target: { value: 'wrong-pass' } })
    expect(unlockButton).toBeEnabled()
    fireEvent.click(unlockButton)

    expect(await screen.findByRole('alert')).toHaveTextContent('Invalid password.')
    expect(screen.queryByLabelText('Video / Audio file input')).not.toBeInTheDocument()

    // Valid password attempt
    fireEvent.change(passwordInput, { target: { value: 'correct-pass' } })
    fireEvent.click(unlockButton)

    expect(await screen.findByLabelText('Video / Audio file input')).toBeInTheDocument()
    const signOutButton = screen.getByRole('button', { name: 'Sign out' })
    expect(signOutButton).toBeInTheDocument()

    // Sign out returns to locked gate
    fireEvent.click(signOutButton)
    expect(await screen.findByLabelText('Password')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Sign out' })).not.toBeInTheDocument()
  })

  it('returns cleanly to the password gate when a protected API request returns 401', async () => {
    window.history.replaceState({}, '', '/jobs/tok_expired')

    fetchMock.mockImplementation(async (input) => {
      const url = String(input)
      if (url === '/api/auth/status') return mockJsonResponse(AUTH_UNLOCKED)
      return mockJsonResponse({ detail: 'Authentication required.' }, 401)
    })

    render(<App />)

    expect(await screen.findByLabelText('Password')).toBeInTheDocument()
    expect(screen.getByRole('alert')).toHaveTextContent(/session expired/i)
    expect(screen.queryByLabelText('Video / Audio file input')).not.toBeInTheDocument()
  })

  it('respects Retry-After on HTTP 429 during job polling and surfaces rate-limit messages', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true })
    window.history.replaceState({}, '', '/jobs/tok_rate_limited')

    const completedStatus = makeJobStatus({
      token: 'tok_rate_limited',
      result_url: '/jobs/tok_rate_limited',
      status: 'completed',
      current_stage: 'notes',
      current_stage_label: 'Markdown Notes',
      progress_pct: 100,
      download_ready: true,
      download_url: '/api/jobs/tok_rate_limited/download',
      download_filename: 'lecture-01-notes.md',
    })

    let jobCalls = 0
    fetchMock.mockImplementation(async (input) => {
      const url = String(input)
      if (url === '/api/auth/status') return mockJsonResponse(AUTH_DISABLED)
      if (url.startsWith('/api/queue')) return mockJsonResponse(makeQueueSnapshot())
      if (url === '/api/jobs/tok_rate_limited') {
        jobCalls += 1
        if (jobCalls === 1) {
          return mockJsonResponse(
            { detail: 'Rate limit exceeded. Please try again in 8 second(s).' },
            429,
            { 'Retry-After': '8' },
          )
        }
        return mockJsonResponse(completedStatus)
      }
      return mockJsonResponse({}, 404)
    })

    render(<App />)

    expect(await screen.findByRole('alert')).toHaveTextContent(/rate limit exceeded/i)

    // After 4s (< 8s Retry-After), polling must still wait
    await act(async () => {
      await vi.advanceTimersByTimeAsync(4000)
    })
    expect(jobCalls).toBe(1)

    // After the full 8s Retry-After elapses, polling resumes and succeeds
    await act(async () => {
      await vi.advanceTimersByTimeAsync(4500)
    })
    expect(await screen.findByRole('link', { name: 'Download .md' })).toBeInTheDocument()
  })

  it('shows rate-limit error message when login returns 429', async () => {
    fetchMock.mockImplementation(async (input) => {
      const url = String(input)
      if (url === '/api/auth/status') return mockJsonResponse(AUTH_LOCKED)
      if (url === '/api/auth/login') {
        return mockJsonResponse(
          { detail: 'Rate limit exceeded. Please try again in 15 second(s).' },
          429,
          { 'Retry-After': '15' },
        )
      }
      return mockJsonResponse({}, 404)
    })

    render(<App />)

    const passwordInput = await screen.findByLabelText('Password')
    fireEvent.change(passwordInput, { target: { value: 'guess' } })
    fireEvent.click(screen.getByRole('button', { name: 'Unlock' }))

    expect(await screen.findByRole('alert')).toHaveTextContent(/15 second/i)
  })

  it('stages multiple pairs, reorders them, and uploads sequentially one at a time', async () => {
    let resolveFirstUpload!: (val: Omit<MockUploadResult, 'deferred'>) => void
    const firstDeferred = new Promise<Omit<MockUploadResult, 'deferred'>>((resolve) => {
      resolveFirstUpload = resolve
    })

    const uploadedOrder: string[] = []
    MockXMLHttpRequest.handler = (formData) => {
      const slides = formData.get('slides') as File
      uploadedOrder.push(slides.name)
      if (uploadedOrder.length === 1) {
        return {
          status: 202,
          deferred: firstDeferred,
        }
      }
      return {
        status: 202,
        body: {
          token: 'tok_second',
          result_url: '/jobs/tok_second',
          status: 'queued',
          current_stage: 'upload',
          created_at: '2026-11-25T20:01:00Z',
        } satisfies JobCreateResponse,
      }
    }

    fetchMock.mockImplementation(async (input) => {
      const url = String(input)
      if (url === '/api/auth/status') return mockJsonResponse(AUTH_DISABLED)
      if (url.startsWith('/api/queue')) return mockJsonResponse(makeQueueSnapshot())
      return mockJsonResponse(makeJobStatus({ token: 'tok_first', status: 'queued' }))
    })

    render(<App />)

    const recInput = await screen.findByLabelText('Video / Audio file input')
    const slidesInput = screen.getByLabelText('Slides PDF file input')

    // Stage lecture 1
    fireEvent.change(recInput, {
      target: { files: [new File(['a'], 'lec1.mp4', { type: 'video/mp4' })] },
    })
    fireEvent.change(slidesInput, {
      target: { files: [new File(['%PDF'], 'lec1.pdf', { type: 'application/pdf' })] },
    })
    fireEvent.click(screen.getByRole('button', { name: /\+ Stage & Add Another/i }))

    // Stage lecture 2
    fireEvent.change(recInput, {
      target: { files: [new File(['b'], 'lec2.mp4', { type: 'video/mp4' })] },
    })
    fireEvent.change(slidesInput, {
      target: { files: [new File(['%PDF'], 'lec2.pdf', { type: 'application/pdf' })] },
    })
    fireEvent.click(screen.getByRole('button', { name: /\+ Stage & Add Another/i }))

    expect(screen.getByText(/Staged to Upload \(2 pairs/i)).toBeInTheDocument()

    // Move lec2 above lec1 before starting
    fireEvent.click(screen.getByRole('button', { name: 'Move staged pair lec2.pdf up' }))

    // Start queue upload
    fireEvent.click(screen.getByRole('button', { name: 'Upload & Start Queue (2)' }))

    // Only 1 XHR upload is in flight (lec2 first due to reorder)
    await waitFor(() => {
      expect(uploadedOrder).toEqual(['lec2.pdf'])
    })
    expect(screen.getByText(/Keep this tab open until uploads finish/i)).toBeInTheDocument()

    // Complete first upload -> second begins automatically
    await act(async () => {
      resolveFirstUpload({
        status: 202,
        body: {
          token: 'tok_first',
          result_url: '/jobs/tok_first',
          status: 'queued',
          current_stage: 'upload',
          created_at: '2026-11-25T20:00:00Z',
        } satisfies JobCreateResponse,
      })
    })

    await waitFor(() => {
      expect(uploadedOrder).toEqual(['lec2.pdf', 'lec1.pdf'])
    })

    expect(
      await screen.findByText(/All lectures uploaded — safe to close this tab/i),
    ).toBeInTheDocument()
  })

  it('auto-pairs bulk dropped files and lets user manually pair unmatched files in the Unpaired drawer', async () => {
    fetchMock.mockImplementation(async (input) => {
      const url = String(input)
      if (url === '/api/auth/status') return mockJsonResponse(AUTH_DISABLED)
      if (url.startsWith('/api/queue')) return mockJsonResponse(makeQueueSnapshot())
      return mockJsonResponse({}, 404)
    })

    render(<App />)

    const uploadSection = await screen.findByRole('region', { name: 'Upload files' })

    fireEvent.drop(uploadSection, {
      dataTransfer: {
        files: [
          new File(['v1'], '01_Intro_Recording.mp4'),
          new File(['p1'], '01_Intro_Slides.pdf'),
          new File(['v2'], 'guest_session.mp4'),
          new File(['p2'], 'bonus_deck.pdf'),
          new File(['bad'], 'notes.docx'),
        ],
      },
    })

    // Auto-paired 01_Intro goes straight to staging tray
    expect(screen.getByText(/Staged to Upload \(1 pair/i)).toBeInTheDocument()
    expect(screen.getByText('01_Intro_Recording.mp4')).toBeInTheDocument()

    // Rejected file alert shown
    expect(screen.getByText(/Unsupported files skipped/i)).toBeInTheDocument()
    expect(screen.getByText('notes.docx')).toBeInTheDocument()

    // Unpaired drawer shows remaining recording + PDF
    const unpairedDrawer = screen.getByLabelText('Unpaired files')
    expect(within(unpairedDrawer).getByText('guest_session.mp4')).toBeInTheDocument()
    expect(within(unpairedDrawer).getByText('bonus_deck.pdf')).toBeInTheDocument()

    // Manually pick and pair them
    fireEvent.click(within(unpairedDrawer).getByRole('button', { name: /^guest_session\.mp4/i }))
    fireEvent.click(within(unpairedDrawer).getByRole('button', { name: /^bonus_deck\.pdf/i }))
    fireEvent.click(within(unpairedDrawer).getByRole('button', { name: 'Pair selected' }))

    expect(screen.getByText(/Staged to Upload \(2 pairs/i)).toBeInTheDocument()
    expect(screen.queryByLabelText('Unpaired files')).not.toBeInTheDocument()
  })

  it('handles 429 Retry-After countdown in upload pump and retries automatically', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true })

    let attempts = 0
    MockXMLHttpRequest.handler = () => {
      attempts += 1
      if (attempts === 1) {
        return {
          status: 429,
          headers: { 'Retry-After': '3' },
          body: { detail: 'Upload rate limit exceeded.' },
        }
      }
      return {
        status: 202,
        body: {
          token: 'tok_after_429',
          result_url: '/jobs/tok_after_429',
          status: 'queued',
          current_stage: 'upload',
          created_at: '2026-11-25T20:00:00Z',
        } satisfies JobCreateResponse,
      }
    }

    fetchMock.mockImplementation(async (input) => {
      const url = String(input)
      if (url === '/api/auth/status') return mockJsonResponse(AUTH_DISABLED)
      if (url.startsWith('/api/queue')) return mockJsonResponse(makeQueueSnapshot())
      return mockJsonResponse(makeJobStatus({ token: 'tok_after_429', status: 'completed' }))
    })

    render(<App />)

    fireEvent.change(await screen.findByLabelText('Video / Audio file input'), {
      target: { files: [new File(['a'], 'lec1.mp4')] },
    })
    fireEvent.change(screen.getByLabelText('Slides PDF file input'), {
      target: { files: [new File(['p'], 'lec1.pdf')] },
    })
    fireEvent.click(screen.getByRole('button', { name: 'Generate Notes' }))

    expect(await screen.findByText(/Rate-limited, retrying in 3s/i)).toBeInTheDocument()

    await act(async () => {
      await vi.advanceTimersByTimeAsync(3200)
    })

    await waitFor(() => {
      expect(attempts).toBe(2)
    })
  })

  it('preserves staged pairs across 401 session expiry and resumes uploading after re-login', async () => {
    let authed = true
    let uploadAttempts = 0

    MockXMLHttpRequest.handler = () => {
      uploadAttempts += 1
      if (!authed) {
        return { status: 401, body: { detail: 'Session expired.' } }
      }
      if (uploadAttempts === 1) {
        authed = false
        return { status: 401, body: { detail: 'Session expired.' } }
      }
      return {
        status: 202,
        body: {
          token: 'tok_resumed',
          result_url: '/jobs/tok_resumed',
          status: 'queued',
          current_stage: 'upload',
          created_at: '2026-11-25T20:00:00Z',
        } satisfies JobCreateResponse,
      }
    }

    fetchMock.mockImplementation(async (input) => {
      const url = String(input)
      if (url === '/api/auth/status') return mockJsonResponse(AUTH_UNLOCKED)
      if (url === '/api/auth/login') {
        authed = true
        return mockJsonResponse(AUTH_UNLOCKED)
      }
      if (url.startsWith('/api/queue')) return mockJsonResponse(makeQueueSnapshot())
      return mockJsonResponse(makeJobStatus({ token: 'tok_resumed', status: 'completed' }))
    })

    render(<App />)

    fireEvent.change(await screen.findByLabelText('Video / Audio file input'), {
      target: { files: [new File(['a'], 'persist.mp4')] },
    })
    fireEvent.change(screen.getByLabelText('Slides PDF file input'), {
      target: { files: [new File(['p'], 'persist.pdf')] },
    })
    fireEvent.click(screen.getByRole('button', { name: 'Generate Notes' }))

    // 401 flips to password gate
    const passwordInput = await screen.findByLabelText('Password')
    expect(screen.getByRole('alert')).toHaveTextContent(/session expired/i)

    // Re-login automatically resumes the preserved upload pair
    fireEvent.change(passwordInput, { target: { value: 'secret' } })
    fireEvent.click(screen.getByRole('button', { name: 'Unlock' }))

    await waitFor(() => {
      expect(uploadAttempts).toBe(2)
    })
  })

  it('isolates single-pair upload failures and continues uploading the next pair', async () => {
    const seen: string[] = []
    MockXMLHttpRequest.handler = (formData) => {
      const slides = formData.get('slides') as File
      seen.push(slides.name)
      if (slides.name === 'bad.pdf') {
        return { status: 413, body: { detail: 'File too large.' } }
      }
      return {
        status: 202,
        body: {
          token: 'tok_good',
          result_url: '/jobs/tok_good',
          status: 'queued',
          current_stage: 'upload',
          created_at: '2026-11-25T20:00:00Z',
        } satisfies JobCreateResponse,
      }
    }

    fetchMock.mockImplementation(async (input) => {
      const url = String(input)
      if (url === '/api/auth/status') return mockJsonResponse(AUTH_DISABLED)
      if (url.startsWith('/api/queue')) return mockJsonResponse(makeQueueSnapshot())
      return mockJsonResponse(makeJobStatus({ token: 'tok_good', status: 'completed' }))
    })

    render(<App />)

    const uploadSection = await screen.findByRole('region', { name: 'Upload files' })
    fireEvent.drop(uploadSection, {
      dataTransfer: {
        files: [
          new File(['a'], 'bad.mp4'),
          new File(['b'], 'bad.pdf'),
          new File(['c'], 'good.mp4'),
          new File(['d'], 'good.pdf'),
        ],
      },
    })

    fireEvent.click(screen.getByRole('button', { name: 'Upload & Start Queue (2)' }))

    await waitFor(() => {
      expect(seen).toEqual(['bad.pdf', 'good.pdf'])
    })

    expect(screen.getByText(/Upload failed: File too large\./i)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Retry' })).toBeInTheDocument()
  })

  it('renders RUNNING, PREPARING NEXT, and QUEUED rows, supports reorder, pause/resume with auto-pause banner, cancel, retry, and auto-follow', async () => {
    const primaryJob = makeJobStatus({
      token: 'tok_j1',
      slides_filename: 'lec-01.pdf',
      recording_filename: 'lec-01.mp4',
      pipeline_role: 'primary',
      current_stage: 'notes',
      current_stage_label: 'Markdown Notes',
      progress_pct: 83,
    })
    const preparingJob = makeJobStatus({
      token: 'tok_j2',
      slides_filename: 'lec-02.pdf',
      recording_filename: 'lec-02.mp4',
      pipeline_role: 'preparing',
      current_stage: 'transcription',
      current_stage_label: 'Transcription (Groq)',
      progress_pct: 42,
      waiting_reason: 'Waiting for previous lecture to finish Markdown Notes',
    })
    let queuedList: QueueItemResponse[] = [
      makeQueueItem({
        token: 'tok_j3',
        slides_filename: 'lec-03.pdf',
        recording_filename: 'lec-03.mp4',
        queue_index: 1,
      }),
      makeQueueItem({
        token: 'tok_j4',
        slides_filename: 'lec-04.pdf',
        recording_filename: 'lec-04.mp4',
        queue_index: 2,
      }),
    ]
    let recentList: QueueItemResponse[] = [
      makeQueueItem({
        token: 'tok_failed_prev',
        status: 'failed',
        slides_filename: 'lec-00.pdf',
        recording_filename: 'lec-00.mp4',
        can_cancel: false,
        can_retry: true,
        error: {
          stage: 'ocr',
          stage_label: 'Slide OCR (Gemini)',
          message: 'Transient OCR error',
          hint: 'Retry this lecture.',
        },
      }),
    ]
    let paused = false
    let pauseReason: string | null = null
    let lastReorderedTokens: string[] = []
    let cancelledToken: string | null = null
    let retriedToken: string | null = null

    const currentSnapshot = (): QueueSnapshotResponse =>
      makeQueueSnapshot({
        paused,
        pause_reason: pauseReason,
        active: [primaryJob, preparingJob],
        queued: queuedList,
        recent: recentList,
      })

    fetchMock.mockImplementation(async (input, init) => {
      const url = String(input)
      const method = init?.method ?? 'GET'
      if (url === '/api/auth/status') return mockJsonResponse(AUTH_DISABLED)
      if (url.startsWith('/api/queue') && method === 'GET') {
        return mockJsonResponse(currentSnapshot())
      }
      if (url === '/api/queue/order' && method === 'PATCH') {
        const body = JSON.parse(String(init?.body ?? '{}')) as { tokens: string[] }
        lastReorderedTokens = body.tokens
        queuedList = body.tokens.map((tok, idx) => {
          const found = queuedList.find((q) => q.token === tok)!
          return { ...found, queue_index: idx + 1 }
        })
        return mockJsonResponse(currentSnapshot())
      }
      if (url === '/api/queue/pause' && method === 'POST') {
        paused = true
        pauseReason = 'GEMINI_API_KEY invalid'
        return mockJsonResponse(currentSnapshot())
      }
      if (url === '/api/queue/resume' && method === 'POST') {
        paused = false
        pauseReason = null
        return mockJsonResponse(currentSnapshot())
      }
      if (url === '/api/jobs/tok_j3/cancel' && method === 'POST') {
        cancelledToken = 'tok_j3'
        queuedList = queuedList
          .filter((q) => q.token !== 'tok_j3')
          .map((q, idx) => ({ ...q, queue_index: idx + 1 }))
        return mockJsonResponse(
          makeJobStatus({
            token: 'tok_j3',
            status: 'cancelled',
            slides_filename: 'lec-03.pdf',
            recording_filename: 'lec-03.mp4',
            can_cancel: false,
            can_retry: false,
          }),
        )
      }
      if (url === '/api/jobs/tok_failed_prev/retry' && method === 'POST') {
        retriedToken = 'tok_failed_prev'
        recentList = recentList.filter((r) => r.token !== 'tok_failed_prev')
        queuedList = [
          ...queuedList,
          makeQueueItem({
            token: 'tok_failed_prev',
            slides_filename: 'lec-00.pdf',
            recording_filename: 'lec-00.mp4',
            queue_index: queuedList.length + 1,
          }),
        ]
        return mockJsonResponse(
          makeJobStatus({
            token: 'tok_failed_prev',
            status: 'queued',
            slides_filename: 'lec-00.pdf',
            recording_filename: 'lec-00.mp4',
            queue_index: queuedList.length,
          }),
          202,
        )
      }
      return mockJsonResponse({}, 404)
    })

    render(<App />)

    // Verify QueuePanel badges & waiting reason
    const queueSection = await screen.findByRole('region', { name: 'Processing queue' })
    expect(within(queueSection).getByText('RUNNING')).toBeInTheDocument()
    expect(within(queueSection).getByText('PREPARING NEXT')).toBeInTheDocument()
    expect(
      within(queueSection).getByText('Waiting for previous lecture to finish Markdown Notes'),
    ).toBeInTheDocument()
    expect(within(queueSection).getByText('#1')).toBeInTheDocument()
    expect(within(queueSection).getByText('#2')).toBeInTheDocument()

    // LiveConsole auto-follows primary job (lec-01.pdf)
    const consoleSection = screen.getByRole('region', { name: 'Live progress console' })
    expect(within(consoleSection).getByText('Following active')).toBeInTheDocument()
    expect(within(consoleSection).getByText('lec-01.pdf')).toBeInTheDocument()

    // Click PREPARING NEXT row -> LiveConsole inspects lec-02 and shows Follow active job button
    fireEvent.click(within(queueSection).getByRole('button', { name: /PREPARING NEXT.*lec-02/i }))
    expect(within(consoleSection).getByText('lec-02.pdf')).toBeInTheDocument()
    const followBtn = within(consoleSection).getByRole('button', { name: 'Follow active job' })
    fireEvent.click(followBtn)
    expect(within(consoleSection).getByText('Following active')).toBeInTheDocument()
    expect(within(consoleSection).getByText('lec-01.pdf')).toBeInTheDocument()

    // Reorder #2 (lec-04) up above #1 (lec-03)
    fireEvent.click(within(queueSection).getByRole('button', { name: 'Move lec-04.pdf up' }))
    await waitFor(() => {
      expect(lastReorderedTokens).toEqual(['tok_j4', 'tok_j3'])
    })

    // Pause queue -> auto-pause banner with pause_reason appears, then Resume clears it
    fireEvent.click(within(queueSection).getByRole('button', { name: 'Pause queue' }))
    expect(await within(queueSection).findByText('GEMINI_API_KEY invalid')).toBeInTheDocument()

    const resumeButtons = within(queueSection).getAllByRole('button', { name: 'Resume queue' })
    fireEvent.click(resumeButtons[0])
    await waitFor(() => {
      expect(within(queueSection).queryByText('GEMINI_API_KEY invalid')).not.toBeInTheDocument()
    })

    // Cancel queued job lec-03
    fireEvent.click(within(queueSection).getByRole('button', { name: 'Cancel lec-03.pdf' }))
    await waitFor(() => {
      expect(cancelledToken).toBe('tok_j3')
    })

    // Retry failed job from HistoryList
    const historySection = screen.getByRole('region', { name: 'Job history' })
    fireEvent.click(within(historySection).getByRole('button', { name: 'Retry lec-00.pdf' }))
    await waitFor(() => {
      expect(retriedToken).toBe('tok_failed_prev')
    })
  })

  it('allows dropping multiple PDFs into the Slides PDF box, reordering them, and uploading them together in order', async () => {
    let uploadedSlidesNames: string[] = []
    MockXMLHttpRequest.handler = (formData) => {
      uploadedSlidesNames = formData.getAll('slides').map((entry) => (entry as File).name)
      return {
        status: 202,
        body: {
          token: 'tok_multi_pdf',
          result_url: '/jobs/tok_multi_pdf',
          status: 'queued',
          current_stage: 'upload',
          created_at: '2026-11-26T12:00:00Z',
        } satisfies JobCreateResponse,
      }
    }

    fetchMock.mockImplementation(async (input) => {
      const url = String(input)
      if (url === '/api/auth/status') return mockJsonResponse(AUTH_DISABLED)
      if (url.startsWith('/api/queue')) return mockJsonResponse(makeQueueSnapshot())
      return mockJsonResponse(
        makeJobStatus({
          token: 'tok_multi_pdf',
          slides_filename: 'part1 + part2.pdf',
          recording_filename: 'lecture.mp4',
          status: 'completed',
        }),
      )
    })

    render(<App />)

    const recordingInput = await screen.findByLabelText('Video / Audio file input')
    const slidesBox = screen.getByRole('button', { name: 'Slides PDF' })

    // Select a recording
    fireEvent.change(recordingInput, {
      target: { files: [new File(['video'], 'lecture.mp4', { type: 'video/mp4' })] },
    })

    // Drop first PDF onto the Slides PDF box
    fireEvent.drop(slidesBox, {
      dataTransfer: {
        files: [new File(['%PDF-1.4 a'], 'part2.pdf', { type: 'application/pdf' })],
      },
    })
    expect(screen.getByText('part2.pdf')).toBeInTheDocument()

    // Drop a second and third PDF onto the Slides PDF box -> appends them
    fireEvent.drop(slidesBox, {
      dataTransfer: {
        files: [
          new File(['%PDF-1.4 b'], 'part1.pdf', { type: 'application/pdf' }),
          new File(['%PDF-1.4 c'], 'extra.pdf', { type: 'application/pdf' }),
        ],
      },
    })

    expect(screen.getByText('3 PDFs · combined')).toBeInTheDocument()

    // Remove extra.pdf and move part1.pdf above part2.pdf
    fireEvent.click(screen.getByRole('button', { name: 'Remove extra.pdf' }))
    expect(screen.getByText('2 PDFs · combined')).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: 'Move part1.pdf up' }))

    // Submit and verify both PDFs are appended in the reordered sequence
    fireEvent.click(screen.getByRole('button', { name: 'Generate Notes' }))

    await waitFor(() => {
      expect(uploadedSlidesNames).toEqual(['part1.pdf', 'part2.pdf'])
    })
  })

  it('toggles between light and dark mode and persists theme in localStorage', async () => {
    fetchMock.mockImplementation(async (input) => {
      const url = String(input)
      if (url === '/api/auth/status') return mockJsonResponse(AUTH_DISABLED)
      if (url.startsWith('/api/queue')) return mockJsonResponse(makeQueueSnapshot())
      return mockJsonResponse({}, 404)
    })

    render(<App />)

    const toggleBtn = await screen.findByRole('button', { name: 'Switch to dark mode' })
    expect(document.documentElement.getAttribute('data-theme')).toBe('light')

    fireEvent.click(toggleBtn)
    expect(document.documentElement.getAttribute('data-theme')).toBe('dark')
    expect(window.localStorage.getItem('slidenotelm:theme')).toBe('dark')
    expect(screen.getByRole('button', { name: 'Switch to light mode' })).toHaveAttribute(
      'aria-pressed',
      'true',
    )

    fireEvent.click(screen.getByRole('button', { name: 'Switch to light mode' }))
    expect(document.documentElement.getAttribute('data-theme')).toBe('light')
    expect(window.localStorage.getItem('slidenotelm:theme')).toBe('light')
  })
})
