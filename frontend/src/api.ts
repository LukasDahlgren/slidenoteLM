import type { AuthStatusResponse, JobCreateResponse, JobStatusResponse } from './types'

export class ApiError extends Error {
  status: number
  retryAfterSeconds: number | null

  constructor(message: string, status: number, retryAfterSeconds: number | null = null) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.retryAfterSeconds = retryAfterSeconds
  }
}

function parseRetryAfterSeconds(response: Response): number | null {
  const raw = response.headers.get('retry-after')
  if (!raw) return null
  const trimmed = raw.trim()
  if (!trimmed) return null

  const numeric = Number(trimmed)
  if (Number.isFinite(numeric) && numeric > 0) {
    return Math.ceil(numeric)
  }

  const parsedDate = Date.parse(trimmed)
  if (!Number.isNaN(parsedDate)) {
    const deltaSeconds = Math.ceil((parsedDate - Date.now()) / 1000)
    if (deltaSeconds > 0) return deltaSeconds
  }

  return null
}

function withRateLimitFallback(
  response: Response,
  fallback: string,
  retryAfterSeconds: number | null,
): string {
  if (response.status === 429) {
    return retryAfterSeconds
      ? `Too many requests. Please try again in ${retryAfterSeconds} second(s).`
      : 'Too many requests. Please wait a moment and try again.'
  }
  return fallback
}

async function parseErrorMessage(response: Response, fallback: string): Promise<string> {
  try {
    const contentType = response.headers.get('content-type') ?? ''
    if (contentType.includes('application/json')) {
      const data: unknown = await response.json()
      if (data && typeof data === 'object' && 'detail' in data) {
        const detail = (data as { detail: unknown }).detail
        if (typeof detail === 'string' && detail.trim()) {
          return detail.trim()
        }
        if (Array.isArray(detail) && detail.length > 0) {
          const messages = detail
            .map((item) =>
              item && typeof item === 'object' && 'msg' in item
                ? String((item as { msg: unknown }).msg)
                : '',
            )
            .filter(Boolean)
          if (messages.length > 0) return messages.join('; ')
        }
      }
    } else {
      const text = (await response.text()).trim()
      if (text) return text
    }
  } catch {
    // Ignore parse errors and return fallback below
  }
  return fallback
}

export async function getAuthStatus(): Promise<AuthStatusResponse> {
  const response = await fetch('/api/auth/status', {
    method: 'GET',
    credentials: 'same-origin',
    headers: { Accept: 'application/json' },
  })

  if (!response.ok) {
    const retryAfterSeconds = parseRetryAfterSeconds(response)
    const message = await parseErrorMessage(
      response,
      withRateLimitFallback(
        response,
        `Failed to check authentication status (HTTP ${response.status}).`,
        retryAfterSeconds,
      ),
    )
    throw new ApiError(message, response.status, retryAfterSeconds)
  }

  return (await response.json()) as AuthStatusResponse
}

export async function loginWithPassword(password: string): Promise<AuthStatusResponse> {
  const response = await fetch('/api/auth/login', {
    method: 'POST',
    credentials: 'same-origin',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'application/json',
    },
    body: JSON.stringify({ password }),
  })

  if (!response.ok) {
    const retryAfterSeconds = parseRetryAfterSeconds(response)
    const message = await parseErrorMessage(
      response,
      withRateLimitFallback(
        response,
        response.status === 401
          ? 'Invalid password.'
          : `Sign-in failed (HTTP ${response.status}).`,
        retryAfterSeconds,
      ),
    )
    throw new ApiError(message, response.status, retryAfterSeconds)
  }

  return (await response.json()) as AuthStatusResponse
}

export async function logout(): Promise<AuthStatusResponse> {
  const response = await fetch('/api/auth/logout', {
    method: 'POST',
    credentials: 'same-origin',
    headers: { Accept: 'application/json' },
  })

  if (!response.ok) {
    const retryAfterSeconds = parseRetryAfterSeconds(response)
    const message = await parseErrorMessage(
      response,
      withRateLimitFallback(
        response,
        `Sign-out failed (HTTP ${response.status}).`,
        retryAfterSeconds,
      ),
    )
    throw new ApiError(message, response.status, retryAfterSeconds)
  }

  return (await response.json()) as AuthStatusResponse
}

export async function createJob(slides: File, recording: File): Promise<JobCreateResponse> {
  const formData = new FormData()
  formData.append('slides', slides)
  formData.append('recording', recording)

  const response = await fetch('/api/jobs', {
    method: 'POST',
    credentials: 'same-origin',
    body: formData,
  })

  if (!response.ok) {
    const retryAfterSeconds = parseRetryAfterSeconds(response)
    const message = await parseErrorMessage(
      response,
      withRateLimitFallback(
        response,
        `Upload failed (HTTP ${response.status}). Please check your files and try again.`,
        retryAfterSeconds,
      ),
    )
    throw new ApiError(message, response.status, retryAfterSeconds)
  }

  return (await response.json()) as JobCreateResponse
}

export async function getJobStatus(token: string): Promise<JobStatusResponse> {
  const response = await fetch(`/api/jobs/${encodeURIComponent(token)}`, {
    method: 'GET',
    credentials: 'same-origin',
    headers: { Accept: 'application/json' },
  })

  if (!response.ok) {
    const retryAfterSeconds = parseRetryAfterSeconds(response)
    const message = await parseErrorMessage(
      response,
      withRateLimitFallback(
        response,
        response.status === 404
          ? 'Job not found. Check that the link is complete.'
          : `Failed to fetch job status (HTTP ${response.status}).`,
        retryAfterSeconds,
      ),
    )
    throw new ApiError(message, response.status, retryAfterSeconds)
  }

  return (await response.json()) as JobStatusResponse
}

export function getDownloadUrl(token: string): string {
  return `/api/jobs/${encodeURIComponent(token)}/download`
}
