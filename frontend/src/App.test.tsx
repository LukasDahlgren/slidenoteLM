import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { App } from './App'
import { HISTORY_STORAGE_KEY } from './history'
import type { JobCreateResponse, JobStatusResponse } from './types'

function mockJsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

const AUTH_DISABLED = { auth_required: false, authenticated: true }
const AUTH_LOCKED = { auth_required: true, authenticated: false }
const AUTH_UNLOCKED = { auth_required: true, authenticated: true }

function makeJobStatus(overrides: Partial<JobStatusResponse> = {}): JobStatusResponse {
  return {
    token: 'tok_abc123',
    result_url: '/jobs/tok_abc123',
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
    ...overrides,
  }
}

describe('slidenoteLM App', () => {
  const fetchMock = vi.fn<typeof fetch>()

  beforeEach(() => {
    fetchMock.mockReset()
    vi.stubGlobal('fetch', fetchMock)
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.unstubAllGlobals()
  })

  it('renders 2 upload boxes and enables Generate Notes only when both valid files are chosen', async () => {
    fetchMock.mockResolvedValueOnce(mockJsonResponse(AUTH_DISABLED, 200))

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

  it('uploads files, updates URL to /jobs/<token>, polls live console steps, and shows Download .md when complete', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true })

    const createdResponse: JobCreateResponse = {
      token: 'tok_abc123',
      result_url: '/jobs/tok_abc123',
      status: 'queued',
      current_stage: 'upload',
      created_at: '2026-11-25T20:00:00Z',
    }

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

    fetchMock
      .mockResolvedValueOnce(mockJsonResponse(AUTH_DISABLED, 200))
      .mockResolvedValueOnce(mockJsonResponse(createdResponse, 202))
      .mockResolvedValueOnce(mockJsonResponse(processingStatus, 200))
      .mockResolvedValueOnce(mockJsonResponse(completedStatus, 200))

    render(<App />)

    fireEvent.change(await screen.findByLabelText('Video / Audio file input'), {
      target: { files: [new File(['audio'], 'lecture-01.mp4', { type: 'video/mp4' })] },
    })
    fireEvent.change(screen.getByLabelText('Slides PDF file input'), {
      target: { files: [new File(['%PDF-1.4'], 'lecture-01.pdf', { type: 'application/pdf' })] },
    })

    fireEvent.click(screen.getByRole('button', { name: 'Generate Notes' }))

    await waitFor(() => {
      expect(window.location.pathname).toBe('/jobs/tok_abc123')
    })

    await waitFor(() => {
      expect(screen.getAllByText(/Transcribing chunk 1\/2/i).length).toBeGreaterThan(0)
    })

    expect(screen.queryByRole('link', { name: 'Download .md' })).not.toBeInTheDocument()

    await act(async () => {
      await vi.advanceTimersByTimeAsync(1600)
    })

    const downloadLink = await screen.findByRole('link', { name: 'Download .md' })
    expect(downloadLink).toHaveAttribute('href', '/api/jobs/tok_abc123/download')
    expect(downloadLink).toHaveAttribute('download', 'lecture-01-notes.md')
    expect(screen.getByText('lecture-01-notes.md')).toBeInTheDocument()

    // Ensure polling stops once completed
    const callCountAfterComplete = fetchMock.mock.calls.length
    await act(async () => {
      await vi.advanceTimersByTimeAsync(5000)
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

    fetchMock
      .mockResolvedValueOnce(mockJsonResponse(AUTH_DISABLED, 200))
      .mockResolvedValueOnce(mockJsonResponse(failedStatus, 200))

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

    fetchMock
      .mockResolvedValueOnce(mockJsonResponse(AUTH_DISABLED, 200))
      .mockResolvedValueOnce(mockJsonResponse(savedStatus, 200))

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
    const createdResponse: JobCreateResponse = {
      token: 'tok_new_history',
      result_url: '/jobs/tok_new_history',
      status: 'queued',
      current_stage: 'upload',
      created_at: '2026-11-25T21:00:00Z',
    }

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

    fetchMock
      .mockResolvedValueOnce(mockJsonResponse(AUTH_DISABLED, 200))
      .mockResolvedValueOnce(mockJsonResponse(createdResponse, 202))
      .mockResolvedValueOnce(mockJsonResponse(completedStatus, 200))

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
    fetchMock
      .mockResolvedValueOnce(mockJsonResponse(AUTH_LOCKED, 200))
      .mockResolvedValueOnce(mockJsonResponse({ detail: 'Invalid password.' }, 401))
      .mockResolvedValueOnce(mockJsonResponse(AUTH_UNLOCKED, 200))
      .mockResolvedValueOnce(mockJsonResponse(AUTH_LOCKED, 200))

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

    fetchMock
      .mockResolvedValueOnce(mockJsonResponse(AUTH_UNLOCKED, 200))
      .mockResolvedValueOnce(mockJsonResponse({ detail: 'Authentication required.' }, 401))

    render(<App />)

    expect(await screen.findByLabelText('Password')).toBeInTheDocument()
    expect(screen.getByRole('alert')).toHaveTextContent(/session expired/i)
    expect(screen.queryByLabelText('Video / Audio file input')).not.toBeInTheDocument()
  })
})
