import type { PipelineStage } from './types'

export const RECORDING_EXTENSIONS = [
  '.mp4',
  '.mov',
  '.mkv',
  '.webm',
  '.avi',
  '.m4v',
  '.mp3',
  '.wav',
  '.m4a',
  '.aac',
  '.ogg',
  '.flac',
  '.opus',
] as const

export const SLIDES_EXTENSIONS = ['.pdf'] as const

export const RECORDING_ACCEPT = RECORDING_EXTENSIONS.join(',')
export const SLIDES_ACCEPT = SLIDES_EXTENSIONS.join(',')

export const POLL_INTERVAL_MS = 1500
export const QUEUE_POLL_INTERVAL_MS = 2000
export const QUEUE_POLL_HIDDEN_INTERVAL_MS = 8000
export const MAX_LOGS_PER_JOB = 200
export const MAX_LOG_TOKENS = 10

export const DEFAULT_STAGES: ReadonlyArray<{ stage: PipelineStage; label: string }> = [
  { stage: 'upload', label: 'Upload' },
  { stage: 'audio_prep', label: 'Audio Prep' },
  { stage: 'transcription', label: 'Transcription (Groq)' },
  { stage: 'ocr', label: 'Slide OCR (Gemini)' },
  { stage: 'alignment', label: 'Alignment' },
  { stage: 'notes', label: 'Markdown Notes' },
]

export function hasAllowedExtension(filename: string, allowed: readonly string[]): boolean {
  const lower = filename.trim().toLowerCase()
  return allowed.some((ext) => lower.endsWith(ext))
}

export function formatSlidesDisplayName(slides: readonly File[]): string {
  if (slides.length === 0) return ''
  if (slides.length === 1) return slides[0].name
  const stems = slides.map((file, idx) => {
    const trimmed = file.name.trim()
    const dotIndex = trimmed.lastIndexOf('.')
    const stem = dotIndex > 0 ? trimmed.slice(0, dotIndex).trim() : trimmed
    return stem || `slides-${idx + 1}`
  })
  return `${stems.join(' + ')}.pdf`
}

export function formatFileSize(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) return ''
  if (bytes < 1024) return `${bytes} B`
  const kb = bytes / 1024
  if (kb < 1024) return `${kb.toFixed(1)} KB`
  const mb = kb / 1024
  if (mb < 1024) return `${mb.toFixed(1)} MB`
  return `${(mb / 1024).toFixed(2)} GB`
}

export function formatTimestamp(isoString: string | null | undefined): string {
  if (!isoString) return ''
  const date = new Date(isoString)
  if (Number.isNaN(date.getTime())) return isoString
  return date.toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

export function formatLogTime(isoString: string | null | undefined): string {
  const date = isoString ? new Date(isoString) : new Date()
  if (Number.isNaN(date.getTime())) return '--:--:--'
  return date.toLocaleTimeString(undefined, {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  })
}
