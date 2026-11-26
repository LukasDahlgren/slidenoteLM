import { type FormEvent, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  ApiError,
  cancelJob,
  deleteJob,
  getAuthStatus,
  getJobStatus,
  getQueueSnapshot,
  loginWithPassword,
  logout,
  pauseQueue,
  reorderQueue,
  resumeQueue,
  retryJob,
} from './api'
import { HistoryList } from './components/HistoryList'
import { LiveConsole } from './components/LiveConsole'
import { QueueComposer } from './components/QueueComposer'
import { QueuePanel } from './components/QueuePanel'
import {
  MAX_LOGS_PER_JOB,
  MAX_LOG_TOKENS,
  POLL_INTERVAL_MS,
  QUEUE_POLL_HIDDEN_INTERVAL_MS,
  QUEUE_POLL_INTERVAL_MS,
  formatLogTime,
  formatSlidesDisplayName,
} from './constants'
import {
  createEntryFromStatus,
  createEntryFromUpload,
  extractTokenFromPath,
  loadHistory,
  mergeHistoryWithServerRecent,
  removeHistoryEntry,
  saveHistory,
  upsertHistoryEntry,
} from './history'
import { useUploadPump } from './hooks/useUploadPump'
import { autoPairFiles, pairInNaturalOrder } from './pairing'
import type {
  ConsoleLogEntry,
  HistoryEntry,
  JobCreateResponse,
  JobStatusResponse,
  QueueSnapshotResponse,
  StagedLecturePair,
  UnpairedFile,
} from './types'

let unpairedCounter = 0
const THEME_STORAGE_KEY = 'slidenotelm:theme'
type ThemeMode = 'light' | 'dark'

function getInitialTheme(): ThemeMode {
  try {
    const saved = window.localStorage.getItem(THEME_STORAGE_KEY)
    if (saved === 'dark' || saved === 'light') {
      return saved
    }
  } catch {
    // Ignore storage read errors
  }
  if (
    typeof window.matchMedia === 'function' &&
    window.matchMedia('(prefers-color-scheme: dark)').matches
  ) {
    return 'dark'
  }
  return 'light'
}

function makeUnpairedFile(file: File): UnpairedFile {
  unpairedCounter += 1
  return {
    id: `unpaired_${Date.now()}_${unpairedCounter}`,
    file,
  }
}

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
          : stage.status === 'cancelled'
            ? 'Cancelled.'
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

  if (additions.length === 0) return existing
  const combined = [...existing, ...additions]
  return combined.length > MAX_LOGS_PER_JOB
    ? combined.slice(combined.length - MAX_LOGS_PER_JOB)
    : combined
}

function updateLogsWithCap(
  prev: Record<string, ConsoleLogEntry[]>,
  token: string,
  nextEntries: ConsoleLogEntry[],
): Record<string, ConsoleLogEntry[]> {
  const cappedEntries =
    nextEntries.length > MAX_LOGS_PER_JOB
      ? nextEntries.slice(nextEntries.length - MAX_LOGS_PER_JOB)
      : nextEntries

  if (prev[token] === cappedEntries) return prev

  const next: Record<string, ConsoleLogEntry[]> = {}
  for (const [key, value] of Object.entries(prev)) {
    if (key !== token) {
      next[key] = value
    }
  }
  next[token] = cappedEntries

  const keys = Object.keys(next)
  if (keys.length > MAX_LOG_TOKENS) {
    const excess = keys.length - MAX_LOG_TOKENS
    for (let i = 0; i < excess; i += 1) {
      delete next[keys[i]]
    }
  }

  return next
}

export function App() {
  const [theme, setTheme] = useState<ThemeMode>(getInitialTheme)
  const [authChecked, setAuthChecked] = useState(false)
  const [authRequired, setAuthRequired] = useState(false)
  const [authenticated, setAuthenticated] = useState(false)
  const [passwordInput, setPasswordInput] = useState('')
  const [authError, setAuthError] = useState<string | null>(null)
  const [isLoggingIn, setIsLoggingIn] = useState(false)
  const [isLoggingOut, setIsLoggingOut] = useState(false)

  const [recordingFile, setRecordingFile] = useState<File | null>(null)
  const [slidesFiles, setSlidesFiles] = useState<File[]>([])
  const [unpairedRecordings, setUnpairedRecordings] = useState<UnpairedFile[]>([])
  const [unpairedSlides, setUnpairedSlides] = useState<UnpairedFile[]>([])
  const [rejectedFiles, setRejectedFiles] = useState<string[]>([])
  const [uploadError, setUploadError] = useState<string | null>(null)

  const [queue, setQueue] = useState<QueueSnapshotResponse | null>(null)
  const [queueRefreshNonce, setQueueRefreshNonce] = useState(0)
  const [queueControlBusy, setQueueControlBusy] = useState(false)
  const [queueControlError, setQueueControlError] = useState<string | null>(null)

  const [selectedToken, setSelectedToken] = useState<string | null>(() =>
    extractTokenFromPath(window.location.pathname),
  )
  const [autoFollowActive, setAutoFollowActive] = useState<boolean>(
    () => !extractTokenFromPath(window.location.pathname),
  )
  const [reloadNonce, setReloadNonce] = useState(0)
  const [inspectedJob, setInspectedJob] = useState<JobStatusResponse | null>(null)
  const [isLoadingJob, setIsLoadingJob] = useState<boolean>(() =>
    Boolean(extractTokenFromPath(window.location.pathname)),
  )
  const [pollError, setPollError] = useState<string | null>(null)
  const [logsByToken, setLogsByToken] = useState<Record<string, ConsoleLogEntry[]>>({})
  const [history, setHistory] = useState<HistoryEntry[]>(() => loadHistory())

  const lastPrimaryTokenRef = useRef<string | null>(null)

  const isLocked = authChecked && authRequired && !authenticated
  const canAccessApi = authChecked && (!authRequired || authenticated)

  const handleAuthExpired = useCallback(() => {
    setAuthRequired(true)
    setAuthenticated(false)
    setAuthError('Session expired. Please enter the password to continue.')
    setPollError(null)
    setQueueControlError(null)
  }, [])

  const applyQueueSnapshot = useCallback((snapshot: QueueSnapshotResponse) => {
    setQueue(snapshot)
    setQueueControlError(null)

    const primaryJob =
      snapshot.active.find((item) => item.pipeline_role === 'primary') ??
      snapshot.active[0] ??
      null

    if (primaryJob) {
      lastPrimaryTokenRef.current = primaryJob.token
    }

    if (snapshot.active.length > 0) {
      setLogsByToken((prev) => {
        let updated = prev
        for (const activeJob of snapshot.active) {
          const merged = mergeStageLogs(updated[activeJob.token] ?? [], activeJob)
          updated = updateLogsWithCap(updated, activeJob.token, merged)
        }
        return updated
      })
    }

    setHistory((prev) => {
      const merged = mergeHistoryWithServerRecent(
        prev,
        snapshot.recent,
        snapshot.active,
        snapshot.queued,
      )
      saveHistory(merged)
      return merged
    })
  }, [])

  const handleJobEnqueued = useCallback(
    (created: JobCreateResponse, pair: StagedLecturePair) => {
      const nowIso = new Date().toISOString()
      const slidesDisplayName = formatSlidesDisplayName(pair.slides)
      const uploadEntries: ConsoleLogEntry[] = [
        {
          key: `upload-complete:${created.token}`,
          timestamp: formatLogTime(nowIso),
          stage: 'upload',
          label: 'Upload',
          status: 'completed',
          message: `Uploaded ${pair.recording.name} and ${slidesDisplayName}. Queued for processing.`,
        },
      ]

      setLogsByToken((prev) => updateLogsWithCap(prev, created.token, uploadEntries))

      setHistory((prev) => {
        const updated = upsertHistoryEntry(
          prev,
          createEntryFromUpload(created, slidesDisplayName, pair.recording.name, pair.label),
        )
        saveHistory(updated)
        return updated
      })

      setSelectedToken((prev) => prev ?? created.token)
      setAutoFollowActive((prev) => prev || window.location.pathname === '/')
      setPollError(null)
      setQueueRefreshNonce((n) => n + 1)
    },
    [],
  )

  const {
    stagedPairs,
    hasActiveUploads,
    remainingUploadCount,
    allUploadedNotice,
    stagePairs,
    enqueueImmediatePair,
    startAllStagedUploads,
    moveStagedPair,
    removeStagedPair,
    retryStagedPair,
    clearStagedPairs,
    dismissAllUploadedNotice,
  } = useUploadPump({
    canAccessApi,
    onJobEnqueued: handleJobEnqueued,
    onAuthExpired: handleAuthExpired,
  })

  // Sync theme to document attribute and persist preference
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    try {
      window.localStorage.setItem(THEME_STORAGE_KEY, theme)
    } catch {
      // Ignore storage quota / private-mode errors
    }
  }, [theme])

  const handleToggleTheme = () => {
    setTheme((prev) => (prev === 'dark' ? 'light' : 'dark'))
  }

  // Check initial authentication status
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

  // Handle browser back/forward navigation
  useEffect(() => {
    const handlePopState = () => {
      const token = extractTokenFromPath(window.location.pathname)
      setSelectedToken(token)
      setAutoFollowActive(!token)
      setInspectedJob(null)
      setPollError(null)
      setIsLoadingJob(Boolean(token))
    }

    window.addEventListener('popstate', handlePopState)
    return () => window.removeEventListener('popstate', handlePopState)
  }, [])

  // Consolidated GET /api/queue polling (stops completely when idle)
  useEffect(() => {
    if (!canAccessApi) return

    let cancelled = false
    let timerId: number | null = null
    let shouldContinuePolling = false

    const clearTimer = () => {
      if (timerId !== null) {
        window.clearTimeout(timerId)
        timerId = null
      }
    }

    const scheduleNext = (delayMs: number) => {
      clearTimer()
      timerId = window.setTimeout(() => {
        void pollQueue()
      }, delayMs)
    }

    const pollQueue = async () => {
      clearTimer()
      try {
        const snapshot = await getQueueSnapshot({ recentLimit: 50 })
        if (cancelled) return

        const hadPrimary = lastPrimaryTokenRef.current
        applyQueueSnapshot(snapshot)

        // If the primary job just completed/failed and left `active`, keep it selected so
        // the user sees the final Download .md / error state in LiveConsole.
        if (hadPrimary && snapshot.active.length === 0) {
          setSelectedToken((prev) => prev ?? hadPrimary)
        }

        const hasServerWork = snapshot.active.length > 0 || snapshot.queued.length > 0
        shouldContinuePolling = hasServerWork

        if (hasServerWork) {
          const interval = document.hidden
            ? QUEUE_POLL_HIDDEN_INTERVAL_MS
            : QUEUE_POLL_INTERVAL_MS
          scheduleNext(interval)
        }
      } catch (err) {
        if (cancelled) return

        if (err instanceof ApiError && err.status === 401) {
          handleAuthExpired()
          return
        }

        if (shouldContinuePolling) {
          const retryDelayMs =
            err instanceof ApiError && err.status === 429
              ? Math.min(
                  60_000,
                  Math.max(QUEUE_POLL_INTERVAL_MS * 2, (err.retryAfterSeconds ?? 5) * 1000),
                )
              : QUEUE_POLL_INTERVAL_MS * 2
          scheduleNext(retryDelayMs)
        }
      }
    }

    const handleVisibilityChange = () => {
      if (!document.hidden && shouldContinuePolling) {
        void pollQueue()
      }
    }

    document.addEventListener('visibilitychange', handleVisibilityChange)
    void pollQueue()

    return () => {
      cancelled = true
      clearTimer()
      document.removeEventListener('visibilitychange', handleVisibilityChange)
    }
  }, [canAccessApi, queueRefreshNonce, applyQueueSnapshot, handleAuthExpired])

  const primaryActiveJob = useMemo(() => {
    if (!queue || queue.active.length === 0) return null
    return queue.active.find((item) => item.pipeline_role === 'primary') ?? queue.active[0] ?? null
  }, [queue])

  const effectiveToken = autoFollowActive
    ? (primaryActiveJob?.token ?? selectedToken)
    : selectedToken

  const activeJobFromQueue = useMemo(() => {
    if (!effectiveToken || !queue) return null
    return queue.active.find((item) => item.token === effectiveToken) ?? null
  }, [effectiveToken, queue])

  const queuedItemFromQueue = useMemo(() => {
    if (!effectiveToken || !queue) return null
    return queue.queued.find((item) => item.token === effectiveToken) ?? null
  }, [effectiveToken, queue])

  const recentItemFromQueue = useMemo(() => {
    if (!effectiveToken || !queue) return null
    return queue.recent.find((item) => item.token === effectiveToken) ?? null
  }, [effectiveToken, queue])

  const recentUpdatedAt = recentItemFromQueue?.updated_at ?? null
  const isKnownQueuedInServer = Boolean(queuedItemFromQueue)
  const isActiveInServer = Boolean(activeJobFromQueue)

  // Fetch GET /api/jobs/{token} only when inspecting a job not currently in queue.active
  useEffect(() => {
    if (!effectiveToken || !canAccessApi || isActiveInServer) return

    let cancelled = false
    let timerId: number | null = null

    const fetchJob = async () => {
      try {
        const status = await getJobStatus(effectiveToken)
        if (cancelled) return

        setInspectedJob(status)
        setPollError(null)
        setIsLoadingJob(false)

        setLogsByToken((prev) =>
          updateLogsWithCap(prev, effectiveToken, mergeStageLogs(prev[effectiveToken] ?? [], status)),
        )

        setHistory((prev) => {
          const updated = upsertHistoryEntry(prev, createEntryFromStatus(status))
          saveHistory(updated)
          return updated
        })

        // If the job is still queued/processing and not already tracked by queue polling,
        // continue polling GET /api/jobs/{token} cleanly.
        if (
          (status.status === 'queued' || status.status === 'processing') &&
          !isKnownQueuedInServer
        ) {
          timerId = window.setTimeout(() => {
            void fetchJob()
          }, POLL_INTERVAL_MS)
        }
      } catch (err) {
        if (cancelled) return
        setIsLoadingJob(false)

        if (err instanceof ApiError && err.status === 401) {
          handleAuthExpired()
          return
        }

        const message =
          err instanceof ApiError
            ? err.message
            : err instanceof Error
              ? err.message
              : 'Network error while fetching job status.'
        setPollError(message)

        if (!(err instanceof ApiError && err.status === 404)) {
          const retryDelayMs =
            err instanceof ApiError && err.status === 429
              ? Math.min(
                  60_000,
                  Math.max(POLL_INTERVAL_MS * 2, (err.retryAfterSeconds ?? 5) * 1000),
                )
              : POLL_INTERVAL_MS * 2
          timerId = window.setTimeout(() => {
            void fetchJob()
          }, retryDelayMs)
        }
      }
    }

    void fetchJob()

    return () => {
      cancelled = true
      if (timerId !== null) {
        window.clearTimeout(timerId)
      }
    }
  }, [
    effectiveToken,
    canAccessApi,
    isActiveInServer,
    isKnownQueuedInServer,
    recentUpdatedAt,
    reloadNonce,
    handleAuthExpired,
  ])

  const effectiveJob: JobStatusResponse | null = useMemo(() => {
    if (activeJobFromQueue) return activeJobFromQueue
    if (inspectedJob && inspectedJob.token === effectiveToken) {
      if (queuedItemFromQueue) {
        return {
          ...inspectedJob,
          status: queuedItemFromQueue.status,
          queue_index: queuedItemFromQueue.queue_index ?? inspectedJob.queue_index,
          queue_position: queuedItemFromQueue.queue_position ?? inspectedJob.queue_position,
          waiting_reason: queuedItemFromQueue.waiting_reason ?? inspectedJob.waiting_reason,
        }
      }
      return inspectedJob
    }
    return null
  }, [activeJobFromQueue, inspectedJob, effectiveToken, queuedItemFromQueue])

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
      setQueueControlError(null)
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
      setQueueControlError(null)
      setIsLoggingOut(false)
    }
  }

  const handleRecordingChange = (file: File | null) => {
    setUploadError(null)
    dismissAllUploadedNotice()
    setRecordingFile(file)
  }

  const handleSlidesChange = (files: File[]) => {
    setUploadError(null)
    dismissAllUploadedNotice()
    setSlidesFiles(files)
  }

  const handleStageCurrentPair = () => {
    if (!recordingFile || slidesFiles.length === 0) return
    setUploadError(null)
    stagePairs([{ recording: recordingFile, slides: slidesFiles }])
    setRecordingFile(null)
    setSlidesFiles([])
  }

  const handleSubmitImmediate = () => {
    if (!recordingFile || slidesFiles.length === 0) return
    setUploadError(null)
    setAutoFollowActive(true)
    enqueueImmediatePair(recordingFile, slidesFiles)
    setRecordingFile(null)
    setSlidesFiles([])
  }

  const handleStartStagedUploads = () => {
    setUploadError(null)
    setAutoFollowActive(true)
    const extraPair =
      recordingFile && slidesFiles.length > 0
        ? { recording: recordingFile, slides: slidesFiles }
        : null
    startAllStagedUploads(extraPair)
    if (extraPair) {
      setRecordingFile(null)
      setSlidesFiles([])
    }
  }

  const handleBulkFilesDrop = (files: File[]) => {
    setUploadError(null)
    dismissAllUploadedNotice()

    const existingRecs = [
      ...(recordingFile ? [recordingFile] : []),
      ...unpairedRecordings.map((item) => item.file),
    ]
    const existingSlides = [
      ...slidesFiles,
      ...unpairedSlides.map((item) => item.file),
    ]

    const result = autoPairFiles(files, existingRecs, existingSlides)

    if (result.rejected.length > 0) {
      setRejectedFiles(result.rejected.map((file) => file.name))
    }

    if (result.pairs.length > 0) {
      stagePairs(result.pairs.map((p) => ({ recording: p.recording, slides: [p.slides] })))
      setRecordingFile(null)
      setSlidesFiles([])
      setUnpairedRecordings(result.unpairedRecordings.map(makeUnpairedFile))
      setUnpairedSlides(result.unpairedSlides.map(makeUnpairedFile))
      return
    }

    // No new pairs yet: populate the 2 upload boxes when at most 1 of each exists.
    if (result.unpairedRecordings.length <= 1 && result.unpairedSlides.length <= 1) {
      setRecordingFile(result.unpairedRecordings[0] ?? null)
      setSlidesFiles(result.unpairedSlides)
      setUnpairedRecordings([])
      setUnpairedSlides([])
      return
    }

    setRecordingFile(null)
    setSlidesFiles([])
    setUnpairedRecordings(result.unpairedRecordings.map(makeUnpairedFile))
    setUnpairedSlides(result.unpairedSlides.map(makeUnpairedFile))
  }

  const handleManualPair = (recordingId: string, slideId: string) => {
    const recItem = unpairedRecordings.find((item) => item.id === recordingId)
    const slideItem = unpairedSlides.find((item) => item.id === slideId)
    if (!recItem || !slideItem) return

    stagePairs([{ recording: recItem.file, slides: slideItem.file }])
    setUnpairedRecordings((prev) => prev.filter((item) => item.id !== recordingId))
    setUnpairedSlides((prev) => prev.filter((item) => item.id !== slideId))
  }

  const handleRemoveUnpairedRecording = (id: string) => {
    setUnpairedRecordings((prev) => prev.filter((item) => item.id !== id))
  }

  const handleRemoveUnpairedSlide = (id: string) => {
    setUnpairedSlides((prev) => prev.filter((item) => item.id !== id))
  }

  const handlePairRemainingInOrder = () => {
    const result = pairInNaturalOrder(
      unpairedRecordings.map((item) => item.file),
      unpairedSlides.map((item) => item.file),
    )
    if (result.pairs.length > 0) {
      stagePairs(result.pairs)
    }
    setUnpairedRecordings(result.remainingRecordings.map(makeUnpairedFile))
    setUnpairedSlides(result.remainingSlides.map(makeUnpairedFile))
  }

  const handleSelectJob = (token: string) => {
    const nextUrl = `/jobs/${encodeURIComponent(token)}`
    if (window.location.pathname !== nextUrl) {
      window.history.pushState({ token }, '', nextUrl)
    }
    setUploadError(null)
    setPollError(null)
    setSelectedToken(token)
    setAutoFollowActive(primaryActiveJob?.token === token)
    const inActive = queue?.active.some((item) => item.token === token) ?? false
    if (!inActive) {
      setInspectedJob((prev) => (prev?.token === token ? prev : null))
      setIsLoadingJob(true)
      setReloadNonce((n) => n + 1)
    } else {
      setIsLoadingJob(false)
    }
  }

  const handleFollowActive = () => {
    setAutoFollowActive(true)
    if (primaryActiveJob) {
      setSelectedToken(primaryActiveJob.token)
    }
    setPollError(null)
    if (window.location.pathname !== '/') {
      window.history.pushState({}, '', '/')
    }
  }

  const handlePauseQueue = async () => {
    if (queueControlBusy) return
    setQueueControlBusy(true)
    setQueueControlError(null)
    try {
      const updated = await pauseQueue()
      applyQueueSnapshot(updated)
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) {
        handleAuthExpired()
        return
      }
      setQueueControlError(err instanceof Error ? err.message : 'Failed to pause queue.')
    } finally {
      setQueueControlBusy(false)
    }
  }

  const handleResumeQueue = async () => {
    if (queueControlBusy) return
    setQueueControlBusy(true)
    setQueueControlError(null)
    try {
      const updated = await resumeQueue()
      applyQueueSnapshot(updated)
      setQueueRefreshNonce((n) => n + 1)
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) {
        handleAuthExpired()
        return
      }
      setQueueControlError(err instanceof Error ? err.message : 'Failed to resume queue.')
    } finally {
      setQueueControlBusy(false)
    }
  }

  const handleMoveQueued = async (token: string, direction: -1 | 1) => {
    if (!queue || queueControlBusy) return
    const tokens = queue.queued.map((item) => item.token)
    const index = tokens.indexOf(token)
    const targetIndex = index + direction
    if (index === -1 || targetIndex < 0 || targetIndex >= tokens.length) return

    const nextTokens = tokens.slice()
    const [moved] = nextTokens.splice(index, 1)
    nextTokens.splice(targetIndex, 0, moved)

    setQueueControlBusy(true)
    setQueueControlError(null)
    try {
      const updated = await reorderQueue(nextTokens)
      applyQueueSnapshot(updated)
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) {
        handleAuthExpired()
        return
      }
      setQueueControlError(err instanceof Error ? err.message : 'Failed to reorder queue.')
    } finally {
      setQueueControlBusy(false)
    }
  }

  const handleCancelJob = async (token: string) => {
    if (queueControlBusy) return
    setQueueControlBusy(true)
    setQueueControlError(null)
    try {
      const updatedJob = await cancelJob(token)
      if (inspectedJob?.token === token || effectiveToken === token) {
        setInspectedJob(updatedJob)
      }
      setHistory((prev) => {
        const next = upsertHistoryEntry(prev, createEntryFromStatus(updatedJob))
        saveHistory(next)
        return next
      })
      setQueueRefreshNonce((n) => n + 1)
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) {
        handleAuthExpired()
        return
      }
      setQueueControlError(err instanceof Error ? err.message : 'Failed to cancel job.')
    } finally {
      setQueueControlBusy(false)
    }
  }

  const handleRetryJob = async (token: string) => {
    if (queueControlBusy) return
    setQueueControlBusy(true)
    setQueueControlError(null)
    try {
      const retriedJob = await retryJob(token, { fromScratch: false, position: 'end' })
      if (inspectedJob?.token === token || effectiveToken === token) {
        setInspectedJob(retriedJob)
      }
      setHistory((prev) => {
        const next = upsertHistoryEntry(prev, createEntryFromStatus(retriedJob))
        saveHistory(next)
        return next
      })
      setPollError(null)
      setQueueRefreshNonce((n) => n + 1)
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) {
        handleAuthExpired()
        return
      }
      setQueueControlError(err instanceof Error ? err.message : 'Failed to retry job.')
    } finally {
      setQueueControlBusy(false)
    }
  }

  const handleRemoveHistory = (token: string) => {
    const entry = history.find((item) => item.token === token)
    setHistory((prev) => {
      const updated = removeHistoryEntry(prev, token)
      saveHistory(updated)
      return updated
    })

    const isTerminal =
      entry &&
      (entry.status === 'completed' ||
        entry.status === 'failed' ||
        entry.status === 'cancelled')

    if (canAccessApi && isTerminal && entry.source === 'server') {
      void deleteJob(token)
        .then(() => {
          setQueueRefreshNonce((n) => n + 1)
        })
        .catch(() => {
          // Ignore 404/409 cleanup errors for already-removed or local-only jobs.
        })
    }
  }

  const handleClearHistory = () => {
    setHistory([])
    saveHistory([])
  }

  const handleNewUpload = () => {
    setSelectedToken(null)
    setAutoFollowActive(false)
    setInspectedJob(null)
    setPollError(null)
    setIsLoadingJob(false)
    setUploadError(null)
    setRecordingFile(null)
    setSlidesFiles([])
    if (window.location.pathname !== '/') {
      window.history.pushState({}, '', '/')
    }
  }

  const queueHasWork = Boolean(queue && (queue.active.length > 0 || queue.queued.length > 0))
  const showQueuePanel = Boolean(
    queue && (queue.active.length > 0 || queue.queued.length > 0 || queue.paused),
  )

  const uploadingPair = stagedPairs.find((pair) => pair.status === 'uploading') ?? null
  const showUploadingInConsole = Boolean(uploadingPair && !effectiveToken)
  const uploadConsoleLogs: ConsoleLogEntry[] = uploadingPair
    ? [
        {
          key: `uploading:${uploadingPair.id}`,
          timestamp: formatLogTime(new Date().toISOString()),
          stage: 'upload',
          label: 'Upload',
          status: 'running',
          message: `Uploading ${uploadingPair.recording.name} and ${formatSlidesDisplayName(uploadingPair.slides)} (${uploadingPair.uploadPct}%). Keep this tab open until the upload finishes.`,
        },
      ]
    : []

  const activeLogs = effectiveToken ? (logsByToken[effectiveToken] ?? []) : []
  const isFollowingThisJob = Boolean(
    primaryActiveJob && effectiveToken === primaryActiveJob.token && autoFollowActive,
  )

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="app-header__row">
          <div>
            <h1 className="app-header__title">slidenoteLM</h1>
            <p className="app-header__subtitle">
              Queue one or more lecture recordings and slides PDFs to generate aligned Markdown
              study notes.
            </p>
          </div>
          <div className="app-header__actions">
            <button
              type="button"
              className="btn btn--ghost btn--sm"
              aria-pressed={theme === 'dark'}
              aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
              onClick={handleToggleTheme}
            >
              {theme === 'dark' ? '☀ Light' : '☾ Dark'}
            </button>
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
            <QueueComposer
              recordingFile={recordingFile}
              slidesFiles={slidesFiles}
              stagedPairs={stagedPairs}
              unpairedRecordings={unpairedRecordings}
              unpairedSlides={unpairedSlides}
              rejectedFiles={rejectedFiles}
              queueHasWork={queueHasWork}
              hasActiveUploads={hasActiveUploads}
              remainingUploadCount={remainingUploadCount}
              uploadError={uploadError}
              allUploadedNotice={allUploadedNotice}
              onRecordingChange={handleRecordingChange}
              onSlidesChange={handleSlidesChange}
              onBulkFilesDrop={handleBulkFilesDrop}
              onStageCurrentPair={handleStageCurrentPair}
              onSubmitImmediate={handleSubmitImmediate}
              onStartStagedUploads={handleStartStagedUploads}
              onMoveStagedPair={moveStagedPair}
              onRemoveStagedPair={removeStagedPair}
              onRetryStagedPair={retryStagedPair}
              onClearStaged={clearStagedPairs}
              onManualPair={handleManualPair}
              onRemoveUnpairedRecording={handleRemoveUnpairedRecording}
              onRemoveUnpairedSlide={handleRemoveUnpairedSlide}
              onPairRemainingInOrder={handlePairRemainingInOrder}
              onDismissRejected={() => setRejectedFiles([])}
            />

            {showQueuePanel && queue ? (
              <QueuePanel
                queue={queue}
                selectedToken={effectiveToken}
                isControlBusy={queueControlBusy}
                controlError={queueControlError}
                onSelect={handleSelectJob}
                onPause={() => void handlePauseQueue()}
                onResume={() => void handleResumeQueue()}
                onMoveQueued={(token, dir) => void handleMoveQueued(token, dir)}
                onCancel={(token) => void handleCancelJob(token)}
              />
            ) : null}

            {showUploadingInConsole || effectiveToken ? (
              <LiveConsole
                token={showUploadingInConsole ? null : effectiveToken}
                isUploading={showUploadingInConsole}
                job={showUploadingInConsole ? null : effectiveJob}
                logs={showUploadingInConsole ? uploadConsoleLogs : activeLogs}
                isLoading={!showUploadingInConsole && !effectiveJob && isLoadingJob}
                pollError={showUploadingInConsole ? null : pollError}
                autoFollowActive={autoFollowActive}
                hasActiveJob={Boolean(primaryActiveJob)}
                isFollowingThisJob={isFollowingThisJob}
                totalQueued={queue?.counts.queued ?? 0}
                isActionBusy={queueControlBusy}
                onFollowActive={handleFollowActive}
                onCancel={(token) => void handleCancelJob(token)}
                onRetry={(token) => void handleRetryJob(token)}
                onNewUpload={handleNewUpload}
              />
            ) : null}

            <HistoryList
              entries={history}
              activeToken={effectiveToken}
              isActionBusy={queueControlBusy}
              onSelect={handleSelectJob}
              onRetry={(token) => void handleRetryJob(token)}
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
