import { getDownloadUrl } from '../api'
import { formatTimestamp } from '../constants'
import type { HistoryEntry } from '../types'

interface HistoryListProps {
  entries: HistoryEntry[]
  activeToken: string | null
  isActionBusy?: boolean
  onSelect: (token: string) => void
  onRetry?: (token: string) => void
  onRemove: (token: string) => void
  onClearAll: () => void
}

export function HistoryList({
  entries,
  activeToken,
  isActionBusy = false,
  onSelect,
  onRetry,
  onRemove,
  onClearAll,
}: HistoryListProps) {
  return (
    <section className="panel history-panel" aria-label="Job history">
      <div className="history-panel__header">
        <h2 className="panel__title">History</h2>
        {entries.length > 0 ? (
          <button type="button" className="btn btn--ghost btn--sm" onClick={onClearAll}>
            Clear history
          </button>
        ) : null}
      </div>

      {entries.length === 0 ? (
        <p className="history-panel__empty">
          No jobs yet. Upload a recording and slides above to generate Markdown notes.
        </p>
      ) : (
        <ul className="history-list">
          {entries.map((entry) => {
            const isActive = entry.token === activeToken
            const downloadHref = entry.downloadReady
              ? (entry.downloadUrl ?? getDownloadUrl(entry.token))
              : null
            const title = entry.queueLabel || entry.slidesFilename
            const canRetry =
              Boolean(onRetry) &&
              (entry.status === 'failed' || entry.status === 'cancelled') &&
              Boolean(entry.canRetry)

            return (
              <li
                key={entry.token}
                className={`history-item ${isActive ? 'history-item--active' : ''}`}
              >
                <button
                  type="button"
                  className="history-item__select"
                  onClick={() => onSelect(entry.token)}
                  aria-current={isActive ? 'true' : undefined}
                >
                  <div className="history-item__main">
                    <span className="history-item__title">{title}</span>
                    <span className="history-item__subtitle">{entry.recordingFilename}</span>
                    {entry.errorSummary && entry.status === 'failed' ? (
                      <span className="history-item__error">{entry.errorSummary}</span>
                    ) : null}
                  </div>
                  <div className="history-item__meta">
                    <span className={`status-badge status-badge--${entry.status}`}>
                      {entry.status}
                    </span>
                    <span className="history-item__time">{formatTimestamp(entry.createdAt)}</span>
                  </div>
                </button>

                <div className="history-item__actions">
                  {canRetry && onRetry ? (
                    <button
                      type="button"
                      className="btn btn--ghost btn--sm"
                      disabled={isActionBusy}
                      aria-label={`Retry ${title}`}
                      onClick={() => onRetry(entry.token)}
                    >
                      Retry
                    </button>
                  ) : null}
                  {downloadHref ? (
                    <a
                      href={downloadHref}
                      download={entry.downloadFilename ?? undefined}
                      className="btn btn--ghost btn--sm"
                      aria-label={`Download ${entry.downloadFilename ?? 'notes.md'}`}
                    >
                      .md
                    </a>
                  ) : null}
                  <button
                    type="button"
                    className="btn btn--ghost btn--sm history-item__remove"
                    aria-label={`Remove ${entry.slidesFilename} from history`}
                    onClick={() => onRemove(entry.token)}
                  >
                    ×
                  </button>
                </div>
              </li>
            )
          })}
        </ul>
      )}
    </section>
  )
}
