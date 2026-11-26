import { useCallback, useEffect, useRef, useState } from 'react'
import { ApiError, createJobWithProgress } from '../api'
import type { JobCreateResponse, StagedLecturePair } from '../types'

interface UseUploadPumpOptions {
  canAccessApi: boolean
  onJobEnqueued: (created: JobCreateResponse, pair: StagedLecturePair) => void
  onAuthExpired: () => void
}

let pairIdCounter = 0

function nextPairId(): string {
  pairIdCounter += 1
  return `pair_${Date.now()}_${pairIdCounter}`
}

function generateBatchId(): string {
  const randomPart =
    typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function'
      ? crypto.randomUUID().slice(0, 8)
      : Math.random().toString(36).slice(2, 10)
  return `batch_${randomPart}`
}

function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === 'AbortError'
}

export function useUploadPump({
  canAccessApi,
  onJobEnqueued,
  onAuthExpired,
}: UseUploadPumpOptions) {
  const [stagedPairs, setStagedPairs] = useState<StagedLecturePair[]>([])
  const [allUploadedNotice, setAllUploadedNotice] = useState(false)

  const abortControllersRef = useRef<Map<string, AbortController>>(new Map())
  const retryTimersRef = useRef<Map<string, number>>(new Map())
  const unmountedRef = useRef(false)
  const hadActiveUploadsRef = useRef(false)
  const onJobEnqueuedRef = useRef(onJobEnqueued)
  const onAuthExpiredRef = useRef(onAuthExpired)

  useEffect(() => {
    onJobEnqueuedRef.current = onJobEnqueued
  }, [onJobEnqueued])

  useEffect(() => {
    onAuthExpiredRef.current = onAuthExpired
  }, [onAuthExpired])

  useEffect(() => {
    unmountedRef.current = false
    const controllers = abortControllersRef.current
    const timers = retryTimersRef.current

    return () => {
      unmountedRef.current = true
      for (const controller of controllers.values()) {
        controller.abort()
      }
      controllers.clear()
      for (const timer of timers.values()) {
        window.clearTimeout(timer)
      }
      timers.clear()
    }
  }, [])

  const hasActiveUploads = stagedPairs.some(
    (pair) =>
      pair.status === 'uploading' ||
      pair.status === 'waiting' ||
      pair.status === 'rate_limited',
  )

  const remainingUploadCount = stagedPairs.filter(
    (pair) =>
      pair.status === 'uploading' ||
      pair.status === 'waiting' ||
      pair.status === 'rate_limited',
  ).length

  useEffect(() => {
    if (!hasActiveUploads) return
    const handleBeforeUnload = (event: BeforeUnloadEvent) => {
      event.preventDefault()
      event.returnValue = ''
    }
    window.addEventListener('beforeunload', handleBeforeUnload)
    return () => window.removeEventListener('beforeunload', handleBeforeUnload)
  }, [hasActiveUploads])

  const scheduleRateLimitCountdown = useCallback((pairId: string, seconds: number) => {
    const existingTimer = retryTimersRef.current.get(pairId)
    if (existingTimer !== undefined) {
      window.clearTimeout(existingTimer)
    }

    const tick = (remaining: number) => {
      if (unmountedRef.current) return
      if (remaining <= 0) {
        retryTimersRef.current.delete(pairId)
        setStagedPairs((prev) =>
          prev.map((item) =>
            item.id === pairId
              ? {
                  ...item,
                  status: 'waiting',
                  retryAfterSeconds: null,
                  errorMessage: null,
                }
              : item,
          ),
        )
        return
      }

      setStagedPairs((prev) =>
        prev.map((item) =>
          item.id === pairId
            ? {
                ...item,
                status: 'rate_limited',
                retryAfterSeconds: remaining,
              }
            : item,
        ),
      )

      const timerId = window.setTimeout(() => tick(remaining - 1), 1000)
      retryTimersRef.current.set(pairId, timerId)
    }

    const firstTimer = window.setTimeout(() => tick(seconds - 1), 1000)
    retryTimersRef.current.set(pairId, firstTimer)
  }, [])

  const activeOrRateLimited = stagedPairs.some(
    (pair) => pair.status === 'uploading' || pair.status === 'rate_limited',
  )
  const nextWaitingPair =
    canAccessApi && !activeOrRateLimited
      ? (stagedPairs.find((pair) => pair.status === 'waiting') ?? null)
      : null

  useEffect(() => {
    if (!nextWaitingPair) return
    if (abortControllersRef.current.has(nextWaitingPair.id)) return

    const pair = nextWaitingPair
    const controller = new AbortController()
    abortControllersRef.current.set(pair.id, controller)
    hadActiveUploadsRef.current = true

    const runUpload = async () => {
      await Promise.resolve()
      if (unmountedRef.current || controller.signal.aborted) return

      setAllUploadedNotice(false)
      setStagedPairs((prev) =>
        prev.map((item) =>
          item.id === pair.id
            ? {
                ...item,
                status: 'uploading',
                uploadPct: 0,
                loadedBytes: 0,
                retryAfterSeconds: null,
                errorMessage: null,
              }
            : item,
        ),
      )

      try {
        const created = await createJobWithProgress(pair.slides, pair.recording, {
          batchId: pair.batchId,
          label: pair.label,
          signal: controller.signal,
          onProgress: (loaded, total, pct) => {
            if (unmountedRef.current || controller.signal.aborted) return
            setStagedPairs((prev) =>
              prev.map((item) =>
                item.id === pair.id
                  ? {
                      ...item,
                      loadedBytes: loaded,
                      totalBytes: total > 0 ? total : item.totalBytes,
                      uploadPct: pct,
                    }
                  : item,
              ),
            )
          },
        })

        if (unmountedRef.current || controller.signal.aborted) return

        setStagedPairs((prev) => {
          const next = prev.filter((item) => item.id !== pair.id)
          if (next.length === 0 && hadActiveUploadsRef.current) {
            hadActiveUploadsRef.current = false
            setAllUploadedNotice(true)
          }
          return next
        })
        onJobEnqueuedRef.current(created, pair)
      } catch (err) {
        if (unmountedRef.current || controller.signal.aborted || isAbortError(err)) {
          return
        }

        if (err instanceof ApiError && err.status === 401) {
          // Preserve all staged/waiting File objects across re-authentication.
          setStagedPairs((prev) =>
            prev.map((item) =>
              item.id === pair.id
                ? {
                    ...item,
                    status: 'waiting',
                    uploadPct: 0,
                    loadedBytes: 0,
                    errorMessage: null,
                  }
                : item,
            ),
          )
          onAuthExpiredRef.current()
          return
        }

        if (err instanceof ApiError && err.status === 429) {
          const waitSeconds = Math.max(1, err.retryAfterSeconds ?? 5)
          setStagedPairs((prev) =>
            prev.map((item) =>
              item.id === pair.id
                ? {
                    ...item,
                    status: 'rate_limited',
                    uploadPct: 0,
                    loadedBytes: 0,
                    retryAfterSeconds: waitSeconds,
                    errorMessage: err.message,
                  }
                : item,
            ),
          )
          scheduleRateLimitCountdown(pair.id, waitSeconds)
          return
        }

        const message =
          err instanceof Error ? err.message : 'Upload failed. Please try again.'
        setStagedPairs((prev) =>
          prev.map((item) =>
            item.id === pair.id
              ? {
                  ...item,
                  status: 'upload_error',
                  uploadPct: 0,
                  loadedBytes: 0,
                  retryAfterSeconds: null,
                  errorMessage: message,
                }
              : item,
          ),
        )
      } finally {
        abortControllersRef.current.delete(pair.id)
      }
    }

    void runUpload()
  }, [nextWaitingPair, scheduleRateLimitCountdown])

  const makeStagedPair = useCallback(
    (
      recording: File,
      slides: File | File[],
      status: 'staged' | 'waiting',
      batchId: string | null = null,
    ): StagedLecturePair => {
      const slidesList = Array.isArray(slides) ? slides : [slides]
      const slidesBytes = slidesList.reduce((sum, file) => sum + file.size, 0)
      return {
        id: nextPairId(),
        recording,
        slides: slidesList,
        label: null,
        batchId,
        status,
        uploadPct: 0,
        loadedBytes: 0,
        totalBytes: recording.size + slidesBytes,
        retryAfterSeconds: null,
        errorMessage: null,
      }
    },
    [],
  )

  const stagePairs = useCallback(
    (pairs: Array<{ recording: File; slides: File | File[] }>) => {
      if (pairs.length === 0) return
      setAllUploadedNotice(false)
      setStagedPairs((prev) => [
        ...prev,
        ...pairs.map((p) => makeStagedPair(p.recording, p.slides, 'staged')),
      ])
    },
    [makeStagedPair],
  )

  const enqueueImmediatePair = useCallback(
    (recording: File, slides: File | File[]) => {
      setAllUploadedNotice(false)
      setStagedPairs((prev) => [...prev, makeStagedPair(recording, slides, 'waiting')])
    },
    [makeStagedPair],
  )

  const startAllStagedUploads = useCallback(
    (extraPair?: { recording: File; slides: File | File[] } | null) => {
      setAllUploadedNotice(false)
      setStagedPairs((prev) => {
        const nextList = extraPair
          ? [...prev, makeStagedPair(extraPair.recording, extraPair.slides, 'staged')]
          : prev

        const toStartCount = nextList.filter(
          (item) => item.status === 'staged' || item.status === 'upload_error',
        ).length
        const batchId = toStartCount > 1 ? generateBatchId() : null

        return nextList.map((item) =>
          item.status === 'staged' || item.status === 'upload_error'
            ? {
                ...item,
                status: 'waiting',
                batchId: item.batchId ?? batchId,
                errorMessage: null,
                retryAfterSeconds: null,
                uploadPct: 0,
                loadedBytes: 0,
              }
            : item,
        )
      })
    },
    [makeStagedPair],
  )

  const moveStagedPair = useCallback((id: string, direction: -1 | 1) => {
    setStagedPairs((prev) => {
      const index = prev.findIndex((item) => item.id === id)
      if (index === -1) return prev
      const targetIndex = index + direction
      if (targetIndex < 0 || targetIndex >= prev.length) return prev
      if (prev[index].status === 'uploading' || prev[targetIndex].status === 'uploading') {
        return prev
      }
      const next = prev.slice()
      const [moved] = next.splice(index, 1)
      next.splice(targetIndex, 0, moved)
      return next
    })
  }, [])

  const removeStagedPair = useCallback((id: string) => {
    const controller = abortControllersRef.current.get(id)
    if (controller) {
      controller.abort()
      abortControllersRef.current.delete(id)
    }
    const timer = retryTimersRef.current.get(id)
    if (timer !== undefined) {
      window.clearTimeout(timer)
      retryTimersRef.current.delete(id)
    }
    setStagedPairs((prev) => {
      const next = prev.filter((item) => item.id !== id)
      if (next.length === 0) {
        hadActiveUploadsRef.current = false
      }
      return next
    })
  }, [])

  const retryStagedPair = useCallback((id: string) => {
    setAllUploadedNotice(false)
    setStagedPairs((prev) =>
      prev.map((item) =>
        item.id === id
          ? {
              ...item,
              status: 'waiting',
              errorMessage: null,
              retryAfterSeconds: null,
              uploadPct: 0,
              loadedBytes: 0,
            }
          : item,
      ),
    )
  }, [])

  const clearStagedPairs = useCallback(() => {
    for (const controller of abortControllersRef.current.values()) {
      controller.abort()
    }
    abortControllersRef.current.clear()
    for (const timer of retryTimersRef.current.values()) {
      window.clearTimeout(timer)
    }
    retryTimersRef.current.clear()
    hadActiveUploadsRef.current = false
    setStagedPairs([])
    setAllUploadedNotice(false)
  }, [])

  const dismissAllUploadedNotice = useCallback(() => {
    setAllUploadedNotice(false)
  }, [])

  return {
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
  }
}
