import type { QueueSnapshotResponse } from '../types'

interface QueuePanelProps {
  queue: QueueSnapshotResponse
  selectedToken: string | null
  isControlBusy: boolean
  controlError: string | null
  onSelect: (token: string) => void
  onPause: () => void
  onResume: () => void
  onMoveQueued: (token: string, direction: -1 | 1) => void
  onCancel: (token: string) => void
}

export function QueuePanel({
  queue,
  selectedToken,
  isControlBusy,
  controlError,
  onSelect,
  onPause,
  onResume,
  onMoveQueued,
  onCancel,
}: QueuePanelProps) {
  const primaryCount = queue.active.filter((job) => job.pipeline_role !== 'preparing').length
  const preparingCount = queue.active.filter((job) => job.pipeline_role === 'preparing').length

  return (
    <section className="panel queue-panel" aria-label="Processing queue">
      <div className="queue-panel__header">
        <div>
          <h2 className="panel__title">Processing Queue</h2>
          <p className="queue-panel__summary">
            <span>Running {primaryCount}</span>
            <span className="divider-dot" aria-hidden="true">
              ·
            </span>
            <span>Preparing {preparingCount}</span>
            <span className="divider-dot" aria-hidden="true">
              ·
            </span>
            <span>Queued {queue.counts.queued}</span>
            <span className="divider-dot" aria-hidden="true">
              ·
            </span>
            <span>Done {queue.counts.completed}</span>
            {queue.counts.failed > 0 ? (
              <>
                <span className="divider-dot" aria-hidden="true">
                  ·
                </span>
                <span className="queue-panel__failed-count">Failed {queue.counts.failed}</span>
              </>
            ) : null}
          </p>
        </div>

        <div className="queue-panel__controls">
          {queue.paused ? (
            <button
              type="button"
              className="btn btn--primary btn--sm"
              disabled={isControlBusy}
              onClick={onResume}
            >
              Resume queue
            </button>
          ) : (
            <button
              type="button"
              className="btn btn--ghost btn--sm"
              disabled={isControlBusy}
              onClick={onPause}
            >
              Pause queue
            </button>
          )}
        </div>
      </div>

      {queue.paused && queue.pause_reason ? (
        <div className="alert alert--warning queue-panel__banner" role="alert">
          <div className="alert__row">
            <div>
              <strong>Queue paused</strong>
              <p>{queue.pause_reason}</p>
            </div>
            <button
              type="button"
              className="btn btn--primary btn--sm"
              disabled={isControlBusy}
              onClick={onResume}
            >
              Resume queue
            </button>
          </div>
        </div>
      ) : queue.paused ? (
        <div className="alert alert--warning queue-panel__banner" role="status">
          Queue is paused. Active primary work will finish, and no new queued lectures will start
          until you resume.
        </div>
      ) : null}

      {controlError ? (
        <div className="alert alert--error queue-panel__banner" role="alert">
          {controlError}
        </div>
      ) : null}

      {queue.active.length === 0 && queue.queued.length === 0 ? (
        <p className="queue-panel__empty">No lectures currently running or queued.</p>
      ) : (
        <ul className="queue-list">
          {queue.active.map((job) => {
            const isSelected = job.token === selectedToken
            const isPreparing = job.pipeline_role === 'preparing'
            const title = job.queue_label || job.slides_filename

            return (
              <li
                key={job.token}
                className={`queue-item queue-item--active ${isSelected ? 'queue-item--selected' : ''}`}
              >
                <button
                  type="button"
                  className="queue-item__select"
                  aria-current={isSelected ? 'true' : undefined}
                  onClick={() => onSelect(job.token)}
                >
                  <div className="queue-item__top">
                    <span
                      className={`status-badge ${
                        isPreparing ? 'status-badge--preparing' : 'status-badge--processing'
                      }`}
                    >
                      {isPreparing ? 'PREPARING NEXT' : 'RUNNING'}
                    </span>
                    <span className="queue-item__title" title={title}>
                      {title}
                    </span>
                    <span className="queue-item__pct">{job.progress_pct}%</span>
                  </div>
                  <div className="queue-item__meta-row">
                    <span className="queue-item__subtitle" title={job.recording_filename}>
                      {job.recording_filename}
                    </span>
                    <span className="divider-dot" aria-hidden="true">
                      ·
                    </span>
                    <span className="queue-item__stage">{job.current_stage_label}</span>
                  </div>
                  {job.waiting_reason ? (
                    <div className="queue-item__waiting">{job.waiting_reason}</div>
                  ) : null}
                  <div className="progress-track progress-track--thin" aria-hidden="true">
                    <div
                      className={`progress-track__fill ${
                        isPreparing ? 'progress-track__fill--preparing' : ''
                      }`}
                      style={{ width: `${job.progress_pct}%` }}
                    />
                  </div>
                </button>

                <div className="queue-item__actions">
                  {job.can_cancel !== false ? (
                    <button
                      type="button"
                      className="btn btn--ghost btn--sm"
                      aria-label={`Stop ${title}`}
                      disabled={isControlBusy}
                      onClick={() => onCancel(job.token)}
                    >
                      Stop
                    </button>
                  ) : null}
                </div>
              </li>
            )
          })}

          {queue.queued.map((item, index) => {
            const isSelected = item.token === selectedToken
            const title = item.queue_label || item.slides_filename
            const displayPosition = item.queue_index ?? index + 1

            return (
              <li
                key={item.token}
                className={`queue-item queue-item--queued ${isSelected ? 'queue-item--selected' : ''}`}
              >
                <button
                  type="button"
                  className="queue-item__select"
                  aria-current={isSelected ? 'true' : undefined}
                  onClick={() => onSelect(item.token)}
                >
                  <div className="queue-item__top">
                    <span className="queue-item__pos">#{displayPosition}</span>
                    <span className="status-badge status-badge--queued">QUEUED</span>
                    <span className="queue-item__title" title={title}>
                      {title}
                    </span>
                  </div>
                  <div className="queue-item__meta-row">
                    <span className="queue-item__subtitle" title={item.recording_filename}>
                      {item.recording_filename}
                    </span>
                  </div>
                </button>

                <div className="queue-item__actions">
                  <button
                    type="button"
                    className="btn btn--ghost btn--sm icon-btn"
                    aria-label={`Move ${title} up`}
                    disabled={isControlBusy || index === 0}
                    onClick={() => onMoveQueued(item.token, -1)}
                  >
                    ↑
                  </button>
                  <button
                    type="button"
                    className="btn btn--ghost btn--sm icon-btn"
                    aria-label={`Move ${title} down`}
                    disabled={isControlBusy || index === queue.queued.length - 1}
                    onClick={() => onMoveQueued(item.token, 1)}
                  >
                    ↓
                  </button>
                  {item.can_cancel !== false ? (
                    <button
                      type="button"
                      className="btn btn--ghost btn--sm icon-btn"
                      aria-label={`Cancel ${title}`}
                      disabled={isControlBusy}
                      onClick={() => onCancel(item.token)}
                    >
                      ×
                    </button>
                  ) : null}
                </div>
              </li>
            )
          })}
        </ul>
      )}
    </section>
  )
}
