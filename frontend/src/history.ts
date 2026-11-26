import type {
  HistoryEntry,
  JobCreateResponse,
  JobStatus,
  JobStatusResponse,
  QueueItemResponse,
} from './types'

export const HISTORY_STORAGE_KEY = 'slidenotelm:job-history:v1'
const MAX_HISTORY_ITEMS = 50
const VALID_STATUSES: ReadonlySet<JobStatus> = new Set([
  'queued',
  'processing',
  'completed',
  'failed',
  'cancelled',
])

function isHistoryEntry(value: unknown): value is HistoryEntry {
  if (!value || typeof value !== 'object') return false
  const record = value as Record<string, unknown>
  return (
    typeof record.token === 'string' &&
    record.token.length > 0 &&
    typeof record.slidesFilename === 'string' &&
    typeof record.recordingFilename === 'string' &&
    typeof record.status === 'string' &&
    VALID_STATUSES.has(record.status as JobStatus) &&
    typeof record.createdAt === 'string'
  )
}

export function loadHistory(): HistoryEntry[] {
  try {
    const raw = window.localStorage.getItem(HISTORY_STORAGE_KEY)
    if (!raw) return []
    const parsed: unknown = JSON.parse(raw)
    if (!Array.isArray(parsed)) return []
    return parsed
      .filter(isHistoryEntry)
      .map((entry) => ({
        ...entry,
        updatedAt: typeof entry.updatedAt === 'string' ? entry.updatedAt : entry.createdAt,
        downloadReady: Boolean(entry.downloadReady),
        downloadUrl: typeof entry.downloadUrl === 'string' ? entry.downloadUrl : null,
        downloadFilename:
          typeof entry.downloadFilename === 'string' ? entry.downloadFilename : null,
        queueLabel: typeof entry.queueLabel === 'string' ? entry.queueLabel : null,
        canRetry: typeof entry.canRetry === 'boolean' ? entry.canRetry : undefined,
        errorSummary: typeof entry.errorSummary === 'string' ? entry.errorSummary : null,
        source: entry.source === 'server' ? ('server' as const) : ('local' as const),
      }))
      .slice(0, MAX_HISTORY_ITEMS)
  } catch {
    return []
  }
}

export function saveHistory(entries: HistoryEntry[]): void {
  try {
    window.localStorage.setItem(
      HISTORY_STORAGE_KEY,
      JSON.stringify(entries.slice(0, MAX_HISTORY_ITEMS)),
    )
  } catch {
    // Ignore storage quota / private-mode errors
  }
}

export function upsertHistoryEntry(
  entries: HistoryEntry[],
  next: HistoryEntry,
): HistoryEntry[] {
  const existingIndex = entries.findIndex((item) => item.token === next.token)
  if (existingIndex === -1) {
    return [next, ...entries].slice(0, MAX_HISTORY_ITEMS)
  }
  const updated = entries.slice()
  updated[existingIndex] = {
    ...entries[existingIndex],
    ...next,
  }
  return updated
}

export function removeHistoryEntry(entries: HistoryEntry[], token: string): HistoryEntry[] {
  return entries.filter((item) => item.token !== token)
}

export function createEntryFromUpload(
  created: JobCreateResponse,
  slidesFilename: string,
  recordingFilename: string,
  queueLabel?: string | null,
): HistoryEntry {
  return {
    token: created.token,
    slidesFilename,
    recordingFilename,
    status: created.status,
    createdAt: created.created_at,
    updatedAt: created.created_at,
    downloadReady: false,
    downloadUrl: null,
    downloadFilename: null,
    queueLabel: queueLabel ?? null,
    canRetry: false,
    errorSummary: null,
    source: 'server',
  }
}

export function createEntryFromStatus(job: JobStatusResponse): HistoryEntry {
  return {
    token: job.token,
    slidesFilename: job.slides_filename,
    recordingFilename: job.recording_filename,
    status: job.status,
    createdAt: job.created_at,
    updatedAt: job.updated_at,
    downloadReady: job.download_ready,
    downloadUrl: job.download_url,
    downloadFilename: job.download_filename,
    queueLabel: job.queue_label ?? null,
    canRetry: job.can_retry,
    errorSummary: job.error ? `${job.error.stage_label}: ${job.error.message}` : null,
    source: 'server',
  }
}

export function createEntryFromQueueItem(item: QueueItemResponse): HistoryEntry {
  return {
    token: item.token,
    slidesFilename: item.slides_filename,
    recordingFilename: item.recording_filename,
    status: item.status,
    createdAt: item.created_at,
    updatedAt: item.updated_at,
    downloadReady: item.download_ready,
    downloadUrl: item.download_url,
    downloadFilename: item.download_filename,
    queueLabel: item.queue_label ?? null,
    canRetry: item.can_retry,
    errorSummary: item.error ? `${item.error.stage_label}: ${item.error.message}` : null,
    source: 'server',
  }
}

export function mergeHistoryWithServerRecent(
  localEntries: HistoryEntry[],
  recentItems: QueueItemResponse[],
  activeItems: JobStatusResponse[] = [],
  queuedItems: QueueItemResponse[] = [],
): HistoryEntry[] {
  const byToken = new Map<string, HistoryEntry>()

  for (const local of localEntries) {
    byToken.set(local.token, local)
  }

  for (const item of [...recentItems, ...queuedItems]) {
    byToken.set(item.token, createEntryFromQueueItem(item))
  }

  for (const active of activeItems) {
    byToken.set(active.token, createEntryFromStatus(active))
  }

  return Array.from(byToken.values())
    .sort((a, b) => {
      const timeA = Date.parse(a.createdAt)
      const timeB = Date.parse(b.createdAt)
      if (!Number.isNaN(timeA) && !Number.isNaN(timeB) && timeA !== timeB) {
        return timeB - timeA
      }
      return b.createdAt.localeCompare(a.createdAt)
    })
    .slice(0, MAX_HISTORY_ITEMS)
}

export function extractTokenFromPath(pathname: string): string | null {
  const match = pathname.match(/^\/jobs\/([^/]+)\/?$/)
  if (!match) return null
  try {
    return decodeURIComponent(match[1])
  } catch {
    return match[1]
  }
}
