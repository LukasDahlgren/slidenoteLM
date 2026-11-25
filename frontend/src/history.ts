import type { HistoryEntry, JobCreateResponse, JobStatus, JobStatusResponse } from './types'

export const HISTORY_STORAGE_KEY = 'slidenotelm:job-history:v1'
const MAX_HISTORY_ITEMS = 50
const VALID_STATUSES: ReadonlySet<JobStatus> = new Set([
  'queued',
  'processing',
  'completed',
  'failed',
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
  updated[existingIndex] = next
  return updated
}

export function removeHistoryEntry(entries: HistoryEntry[], token: string): HistoryEntry[] {
  return entries.filter((item) => item.token !== token)
}

export function createEntryFromUpload(
  created: JobCreateResponse,
  slidesFilename: string,
  recordingFilename: string,
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
  }
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
