import { useState } from 'react'
import { DEFAULT_STAGES } from '../constants'
import type { ConsoleLogEntry, JobStatusResponse, StageStateResponse } from '../types'

interface LiveConsoleProps {
  token: string
  job: JobStatusResponse | null
  logs: ConsoleLogEntry[]
  isLoading: boolean
  pollError: string | null
  onNewUpload: () => void
}

const STATUS_LABELS: Record<JobStatusResponse['status'], string> = {
  queued: 'Queued',
  processing: 'Processing',
  completed: 'Completed',
  failed: 'Failed',
}

const STAGE_SYMBOL: Record<StageStateResponse['status'], string> = {
  pending: '○',
  running: '●',
  completed: '✓',
  failed: '✕',
}

export function LiveConsole({
  token,
  job,
  logs,
  isLoading,
  pollError,
  onNewUpload,
}: LiveConsoleProps) {
  const [copied, setCopied] = useState(false)

  const stages: StageStateResponse[] =
    job?.stages && job.stages.length > 0
      ? job.stages
      : DEFAULT_STAGES.map(({ stage, label }) => ({
          stage,
          label,
          status: 'pending',
          started_at: null,
          completed_at: null,
          message: null,
        }))

  const sharePath = job?.result_url ?? `/jobs/${token}`

  const handleCopyLink = async () => {
    const fullUrl = `${window.location.origin}${sharePath}`
    try {
      await navigator.clipboard.writeText(fullUrl)
      setCopied(true)
      window.setTimeout(() => setCopied(false), 1800)
    } catch {
      setCopied(false)
    }
  }

  return (
    <section className="panel console-panel" aria-label="Live progress console">
      <div className="console-panel__top">
        <div className="console-panel__title-group">
          <div className="console-panel__heading-row">
            <h2 className="panel__title">Live Console</h2>
            {job ? (
              <span className={`status-badge status-badge--${job.status}`}>
                {STATUS_LABELS[job.status]}
              </span>
            ) : (
              <span className="status-badge status-badge--queued">
                {isLoading ? 'Loading' : 'Pending'}
              </span>
            )}
            {job ? <span className="console-panel__pct">{job.progress_pct}%</span> : null}
          </div>
          {job ? (
            <p className="console-panel__files">
              <span>{job.recording_filename}</span>
              <span className="divider-dot" aria-hidden="true">
                ·
              </span>
              <span>{job.slides_filename}</span>
            </p>
          ) : (
            <p className="console-panel__files">Job {token.slice(0, 12)}</p>
          )}
        </div>

        <div className="console-panel__actions">
          <button
            type="button"
            className="btn btn--ghost btn--sm"
            onClick={handleCopyLink}
            title={sharePath}
          >
            {copied ? 'Link copied' : 'Copy link'}
          </button>
          <button type="button" className="btn btn--ghost btn--sm" onClick={onNewUpload}>
            New upload
          </button>
        </div>
      </div>

      {pollError ? (
        <div className="alert alert--error" role="alert">
          <strong>Unable to load job status</strong>
          <p>{pollError}</p>
        </div>
      ) : null}

      <div className="console-box" aria-live="polite">
        <ul className="stage-list" aria-label="Pipeline stages">
          {stages.map((stage) => (
            <li
              key={stage.stage}
              className={`stage-row stage-row--${stage.status}`}
              data-stage={stage.stage}
              data-status={stage.status}
            >
              <span className="stage-row__icon" aria-hidden="true">
                {STAGE_SYMBOL[stage.status]}
              </span>
              <span className="stage-row__label">{stage.label}</span>
              <span className="stage-row__status">{stage.status}</span>
              {stage.message ? <span className="stage-row__message">{stage.message}</span> : null}
            </li>
          ))}
        </ul>

        <div className="console-log" aria-label="Live log output">
          {logs.length === 0 ? (
            <div className="console-log__line console-log__line--muted">
              Waiting for pipeline updates...
            </div>
          ) : (
            logs.map((entry) => (
              <div key={entry.key} className={`console-log__line console-log__line--${entry.status}`}>
                <span className="console-log__time">[{entry.timestamp}]</span>{' '}
                <span className="console-log__stage">{entry.label}:</span>{' '}
                <span>{entry.message}</span>
              </div>
            ))
          )}
        </div>
      </div>

      {job && job.warnings.length > 0 ? (
        <div className="alert alert--warning" role="status">
          <strong>Warnings</strong>
          <ul className="alert__list">
            {job.warnings.map((warning, index) => (
              <li key={`${index}-${warning}`}>{warning}</li>
            ))}
          </ul>
        </div>
      ) : null}

      {job?.error ? (
        <div className="alert alert--error" role="alert">
          <strong>Failed at {job.error.stage_label}</strong>
          <p className="alert__message">{job.error.message}</p>
          <p className="alert__hint">Hint: {job.error.hint}</p>
        </div>
      ) : null}

      {job?.download_ready && job.download_url ? (
        <div className="download-bar">
          <div className="download-bar__info">
            <span className="download-bar__ready">Markdown notes ready</span>
            {job.download_filename ? (
              <span className="download-bar__filename">{job.download_filename}</span>
            ) : null}
          </div>
          <a
            href={job.download_url}
            download={job.download_filename ?? undefined}
            className="btn btn--primary"
          >
            Download .md
          </a>
        </div>
      ) : null}
    </section>
  )
}
