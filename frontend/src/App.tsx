import { type FormEvent, useEffect, useState } from 'react'
import {
  ApiError,
  createJob,
  getAuthStatus,
  getJobStatus,
  loginWithPassword,
  logout,
} from './api'
import { HistoryList } from './components/HistoryList'
import { LiveConsole } from './components/LiveConsole'
import { UploadBox } from './components/UploadBox'
import {
  POLL_INTERVAL_MS,
  RECORDING_ACCEPT,
  RECORDING_EXTENSIONS,
  SLIDES_ACCEPT,
  SLIDES_EXTENSIONS,
  formatLogTime,
} from './constants'
import {
  createEntryFromStatus,
  createEntryFromUpload,
  extractTokenFromPath,
  loadHistory,
  removeHistoryEntry,
  saveHistory,
  upsertHistoryEntry,
} from './history'
import type { ConsoleLogEntry, HistoryEntry, JobStatusResponse } from './types'

function mergeStageLogs(
  existing: ConsoleLogEntry[],
  job: JobStatusResponse,
): ConsoleLogEntry[] {
  const seen = new Set(existing.map((entry) => entry.key))
  const additions: ConsoleLogEntry[] = []

  for (const stage of job.stages) {
    if (stage.status === 'pending' && !stage.message) continue
    const message =
      stage.message?.trim() ||
      (stage.status === 'running'
        ? 'In progress...'
        : stage.status === 'completed'
          ? 'Completed.'
          : 'Failed.')
    const key = `${stage.stage}:${stage.status}:${message}`
    if (seen.has(key)) continue
    seen.add(key)
    additions.push({
      key,
      timestamp: formatLogTime(stage.completed_at ?? stage.started_at ?? job.updated_at),
      stage: stage.stage,
      label: stage.label,
      status: stage.status,
      message,
    })
  }

  return additions.length > 0 ? [...existing, ...additions] : existing
}

export function App() {
  const [authChecked, setAuthChecked] = useState(false)
  const [authRequired, setAuthRequired] = useState(false)
  const [authenticated, setAuthenticated] = useState(false)
  const [passwordInput, setPasswordInput] = useState('')
  const [authError, setAuthError] = useState<string | null>(null)
  const [isLoggingIn, setIsLoggingIn] = useState(false)
  const [isLoggingOut, setIsLoggingOut] = useState(false)

  const [recordingFile, setRecordingFile] = useState<File | null>(null)
  const [slidesFile, setSlidesFile] = useState<File | null>(null)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [uploadError, setUploadError] = useState<string | null>(null)

  const [activeToken, setActiveToken] = useState<string | null>(() =>
    extractTokenFromPath(window.location.pathname),
  )
  const [reloadNonce, setReloadNonce] = useState(0)
  const [job, setJob] = useState<JobStatusResponse | null>(null)
  const [isLoadingJob, setIsLoadingJob] = useState<boolean>(() =>
    Boolean(extractTokenFromPath(window.location.pathname)),
  )
  const [pollError, setPollError] = useState<string | null>(null)
  const [logsByToken, setLogsByToken] = useState<Record<string, ConsoleLogEntry[]>>({})
  const [history, setHistory] = useState<HistoryEntry[]>(() => loadHistory())

  const isLocked = authChecked && authRequired && !authenticated
  const canAccessApi = authChecked && (!authRequired || authenticated)

  useEffect(() => {
    let cancelled = false

    const checkAuth = async () => {
      try {
        const status = await getAuthStatus()
        if (cancelled) return
        setAuthRequired(status.auth_required)
        setAuthenticated(status.authenticated)
      } catch {
        if (cancelled) return
        setAuthRequired(false)
        setAuthenticated(true)
      } finally {
        if (!cancelled) {
          setAuthChecked(true)
        }
      }
    }

    void checkAuth()

    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    const handlePopState = () => {
      const token = extractTokenFromPath(window.location.pathname)
      setActiveToken(token)
      setJob(null)
      setPollError(null)
      setIsLoadingJob(Boolean(token))
    }

    window.addEventListener('popstate', handlePopState)
    return () => window.removeEventListener('popstate', handlePopState)
  }, [])

  useEffect(() => {
    if (!activeToken || !canAccessApi) return

    let cancelled = false
    let timerId: number | null = null

    const poll = async () => {
      try {
        const status = await getJobStatus(activeToken)
        if (cancelled) return

        setJob(status)
        setPollError(null)
        setIsLoadingJob(false)

        setLogsByToken((prev) => ({
          ...prev,
          [activeToken]: mergeStageLogs(prev[activeToken] ?? [], status),
        }))

        setHistory((prev) => {
          const updated = upsertHistoryEntry(prev, createEntryFromStatus(status))
          saveHistory(updated)
          return updated
        })

        if (status.status === 'queued' || status.status === 'processing') {
          timerId = window.setTimeout(() => {
            void poll()
          }, POLL_INTERVAL_MS)
        }
      } catch (err) {
        if (cancelled) return
        setIsLoadingJob(false)

        if (err instanceof ApiError && err.status === 401) {
          setAuthRequired(true)
          setAuthenticated(false)
          setAuthError('Session expired. Please enter the password to continue.')
          setPollError(null)
          return
        }

        const message =
          err instanceof ApiError
            ? err.message
            : err instanceof Error
              ? err.message
              : 'Network error while fetching job status.'
        setPollError(message)

        // Retry transient errors, but stop on 404 Not Found.
        if (!(err instanceof ApiError && err.status === 404)) {
          timerId = window.setTimeout(() => {
            void poll()
          }, POLL_INTERVAL_MS * 2)
        }
      }
    }

    void poll()

    return () => {
      cancelled = true
      if (timerId !== null) {
        window.clearTimeout(timerId)
      }
    }
  }, [activeToken, reloadNonce, canAccessApi])

  const canSubmit = Boolean(recordingFile && slidesFile) && !isSubmitting

  const handleLoginSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    if (!passwordInput || isLoggingIn) return

    setIsLoggingIn(true)
    setAuthError(null)

    try {
      const status = await loginWithPassword(passwordInput)
      setAuthRequired(status.auth_required)
      setAuthenticated(status.authenticated)
      setPasswordInput('')
      setAuthError(null)
      setPollError(null)
      setUploadError(null)
    } catch (err) {
      const message =
        err instanceof Error ? err.message : 'Invalid password. Please try again.'
      setAuthError(message)
    } finally {
      setIsLoggingIn(false)
    }
  }

  const handleLogout = async () => {
    if (isLoggingOut) return
    setIsLoggingOut(true)
    try {
      const status = await logout()
      setAuthRequired(status.auth_required)
      setAuthenticated(status.authenticated)
    } catch {
      setAuthRequired(true)
      setAuthenticated(false)
    } finally {
      setPasswordInput('')
      setAuthError(null)
      setUploadError(null)
      setPollError(null)
      setIsLoggingOut(false)
    }
  }

  const handleStartProcessing = async () => {
    if (!recordingFile || !slidesFile || isSubmitting) return

    setIsSubmitting(true)
    setUploadError(null)

    try {
      const created = await createJob(slidesFile, recordingFile)
      const nextUrl = created.result_url || `/jobs/${created.token}`

      window.history.pushState({ token: created.token }, '', nextUrl)

      setHistory((prev) => {
        const updated = upsertHistoryEntry(
          prev,
          createEntryFromUpload(created, slidesFile.name, recordingFile.name),
        )
        saveHistory(updated)
        return updated
      })

      setJob(null)
      setPollError(null)
      setIsLoadingJob(true)
      setActiveToken(created.token)
      setRecordingFile(null)
      setSlidesFile(null)
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) {
        setAuthRequired(true)
        setAuthenticated(false)
        setAuthError('Session expired. Please enter the password to continue.')
        setUploadError(null)
        return
      }
      const message =
        err instanceof Error ? err.message : 'Upload failed. Please try again.'
      setUploadError(message)
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleSelectHistory = (token: string) => {
    const nextUrl = `/jobs/${encodeURIComponent(token)}`
    if (window.location.pathname !== nextUrl) {
      window.history.pushState({ token }, '', nextUrl)
    }
    setUploadError(null)
    setPollError(null)
    setJob((prev) => (prev?.token === token ? prev : null))
    setIsLoadingJob(true)
    setActiveToken(token)
    setReloadNonce((n) => n + 1)
  }

  const handleRemoveHistory = (token: string) => {
    setHistory((prev) => {
      const updated = removeHistoryEntry(prev, token)
      saveHistory(updated)
      return updated
    })
  }

  const handleClearHistory = () => {
    setHistory([])
    saveHistory([])
  }

  const handleNewUpload = () => {
    setActiveToken(null)
    setJob(null)
    setPollError(null)
    setIsLoadingJob(false)
    setUploadError(null)
    setRecordingFile(null)
    setSlidesFile(null)
    if (window.location.pathname !== '/') {
      window.history.pushState({}, '', '/')
    }
  }

  const activeLogs = activeToken ? (logsByToken[activeToken] ?? []) : []

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="app-header__row">
          <div>
            <h1 className="app-header__title">slidenoteLM</h1>
            <p className="app-header__subtitle">
              Upload a lecture recording and slides PDF to generate aligned Markdown study notes.
            </p>
          </div>
          {authChecked && authRequired && authenticated ? (
            <button
              type="button"
              className="btn btn--ghost btn--sm app-header__signout"
              disabled={isLoggingOut}
              onClick={() => void handleLogout()}
            >
              {isLoggingOut ? 'Signing out...' : 'Sign out'}
            </button>
          ) : null}
        </div>
      </header>

      <main className="app-main">
        {!authChecked ? (
          <section className="panel auth-panel" aria-label="Checking session">
            <p className="auth-panel__hint" role="status">
              Checking session...
            </p>
          </section>
        ) : isLocked ? (
          <section className="panel auth-panel" aria-label="Password gate">
            <h2 className="panel__title">Password required</h2>
            <p className="auth-panel__hint">
              Enter the app password to upload lectures and access generated notes.
            </p>
            <form className="auth-form" onSubmit={(e) => void handleLoginSubmit(e)}>
              <label className="sr-only" htmlFor="app-password-input">
                Password
              </label>
              <input
                id="app-password-input"
                type="password"
                className="auth-input"
                placeholder="Enter password"
                autoComplete="current-password"
                autoFocus
                value={passwordInput}
                disabled={isLoggingIn}
                onChange={(e) => {
                  setAuthError(null)
                  setPasswordInput(e.target.value)
                }}
              />
              <button
                type="submit"
                className="btn btn--primary"
                disabled={isLoggingIn || passwordInput.length === 0}
              >
                {isLoggingIn ? 'Unlocking...' : 'Unlock'}
              </button>
            </form>
            {authError ? (
              <div className="alert alert--error" role="alert">
                {authError}
              </div>
            ) : null}
          </section>
        ) : (
          <>
            <section className="panel upload-panel" aria-label="Upload files">
              <div className="upload-grid">
                <UploadBox
                  title="Video / Audio"
                  subtitle="MP4, MOV, MKV, WEBM, MP3, WAV, M4A, and more"
                  accept={RECORDING_ACCEPT}
                  allowedExtensions={RECORDING_EXTENSIONS}
                  invalidTypeMessage="Please select a supported video or audio file (.mp4, .mov, .mkv, .webm, .avi, .m4v, .mp3, .wav, .m4a, .aac, .ogg, .flac, .opus)."
                  file={recordingFile}
                  disabled={isSubmitting}
                  onFileChange={(file) => {
                    setUploadError(null)
                    setRecordingFile(file)
                  }}
                />

                <UploadBox
                  title="Slides PDF"
                  subtitle="PDF document (.pdf)"
                  accept={SLIDES_ACCEPT}
                  allowedExtensions={SLIDES_EXTENSIONS}
                  invalidTypeMessage="Please select a valid .pdf slides file."
                  file={slidesFile}
                  disabled={isSubmitting}
                  onFileChange={(file) => {
                    setUploadError(null)
                    setSlidesFile(file)
                  }}
                />
              </div>

              {uploadError ? (
                <div className="alert alert--error" role="alert">
                  {uploadError}
                </div>
              ) : null}

              <div className="upload-panel__footer">
                <button
                  type="button"
                  className="btn btn--primary"
                  disabled={!canSubmit}
                  onClick={() => void handleStartProcessing()}
                >
                  {isSubmitting ? 'Uploading...' : 'Generate Notes'}
                </button>
              </div>
            </section>

            {activeToken ? (
              <LiveConsole
                token={activeToken}
                job={job}
                logs={activeLogs}
                isLoading={isLoadingJob}
                pollError={pollError}
                onNewUpload={handleNewUpload}
              />
            ) : null}

            <HistoryList
              entries={history}
              activeToken={activeToken}
              onSelect={handleSelectHistory}
              onRemove={handleRemoveHistory}
              onClearAll={handleClearHistory}
            />
          </>
        )}
      </main>
    </div>
  )
}

export default App
