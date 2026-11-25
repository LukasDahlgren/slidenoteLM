import { getDownloadUrl } from '../api'
import { formatTimestamp } from '../constants'
import type { HistoryEntry } from '../types'

interface HistoryListProps {
  entries: HistoryEntry[]
  activeToken: string | null
  onSelect: (token: string) => void
  onRemove: (token: string) => void
  onClearAll: () => void
}

export function HistoryList({
  entries,
  activeToken,
  onSelect,
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
            const downloadHref =
              entry.downloadReady ? (entry.downloadUrl ?? getDownloadUrl(entry.token)) : null

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
                    <span className="history-item__title">{entry.slidesFilename}</span>
                    <span className="history-item__subtitle">{entry.recordingFilename}</span>
                  </div>
                  <div className="history-item__meta">
                    <span className={`status-badge status-badge--${entry.status}`}>
                      {entry.status}
                    </span>
                    <span className="history-item__time">{formatTimestamp(entry.createdAt)}</span>
                  </div>
                </button>

                <div className="history-item__actions">
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
