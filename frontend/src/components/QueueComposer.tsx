import { useState } from 'react'
import type { DragEvent } from 'react'
import {
  RECORDING_ACCEPT,
  RECORDING_EXTENSIONS,
  SLIDES_ACCEPT,
  SLIDES_EXTENSIONS,
  formatFileSize,
  formatSlidesDisplayName,
} from '../constants'
import type { StagedLecturePair, UnpairedFile } from '../types'
import { UploadBox } from './UploadBox'

interface QueueComposerProps {
  recordingFile: File | null
  slidesFiles: File[]
  stagedPairs: StagedLecturePair[]
  unpairedRecordings: UnpairedFile[]
  unpairedSlides: UnpairedFile[]
  rejectedFiles: string[]
  queueHasWork: boolean
  hasActiveUploads: boolean
  remainingUploadCount: number
  uploadError: string | null
  allUploadedNotice: boolean
  onRecordingChange: (file: File | null) => void
  onSlidesChange: (files: File[]) => void
  onBulkFilesDrop: (files: File[]) => void
  onStageCurrentPair: () => void
  onSubmitImmediate: () => void
  onStartStagedUploads: () => void
  onMoveStagedPair: (id: string, direction: -1 | 1) => void
  onRemoveStagedPair: (id: string) => void
  onRetryStagedPair: (id: string) => void
  onClearStaged: () => void
  onManualPair: (recordingId: string, slideId: string) => void
  onRemoveUnpairedRecording: (id: string) => void
  onRemoveUnpairedSlide: (id: string) => void
  onPairRemainingInOrder: () => void
  onDismissRejected: () => void
}

export function QueueComposer({
  recordingFile,
  slidesFiles,
  stagedPairs,
  unpairedRecordings,
  unpairedSlides,
  rejectedFiles,
  queueHasWork,
  hasActiveUploads,
  remainingUploadCount,
  uploadError,
  allUploadedNotice,
  onRecordingChange,
  onSlidesChange,
  onBulkFilesDrop,
  onStageCurrentPair,
  onSubmitImmediate,
  onStartStagedUploads,
  onMoveStagedPair,
  onRemoveStagedPair,
  onRetryStagedPair,
  onClearStaged,
  onManualPair,
  onRemoveUnpairedRecording,
  onRemoveUnpairedSlide,
  onPairRemainingInOrder,
  onDismissRejected,
}: QueueComposerProps) {
  const [isDraggingPanel, setIsDraggingPanel] = useState(false)
  const [selectedUnpairedRecId, setSelectedUnpairedRecId] = useState<string | null>(null)
  const [selectedUnpairedSlideId, setSelectedUnpairedSlideId] = useState<string | null>(null)

  const activeSelectedRecId = unpairedRecordings.some((item) => item.id === selectedUnpairedRecId)
    ? selectedUnpairedRecId
    : null
  const activeSelectedSlideId = unpairedSlides.some((item) => item.id === selectedUnpairedSlideId)
    ? selectedUnpairedSlideId
    : null

  const hasCompletePair = Boolean(recordingFile && slidesFiles.length > 0)
  const readyStagedCount = stagedPairs.filter(
    (pair) => pair.status === 'staged' || pair.status === 'upload_error',
  ).length
  const totalStagedBytes = stagedPairs.reduce((sum, pair) => sum + pair.totalBytes, 0)
  const pendingStartCount = readyStagedCount + (hasCompletePair ? 1 : 0)

  const handlePanelDragOver = (event: DragEvent<HTMLElement>) => {
    event.preventDefault()
    setIsDraggingPanel(true)
  }

  const handlePanelDragLeave = (event: DragEvent<HTMLElement>) => {
    event.preventDefault()
    if (!event.currentTarget.contains(event.relatedTarget as Node | null)) {
      setIsDraggingPanel(false)
    }
  }

  const handlePanelDrop = (event: DragEvent<HTMLElement>) => {
    event.preventDefault()
    setIsDraggingPanel(false)
    const files = Array.from(event.dataTransfer.files ?? [])
    if (files.length === 0) return
    onBulkFilesDrop(files)
  }

  const handlePairSelectedClick = () => {
    if (!activeSelectedRecId || !activeSelectedSlideId) return
    onManualPair(activeSelectedRecId, activeSelectedSlideId)
    setSelectedUnpairedRecId(null)
    setSelectedUnpairedSlideId(null)
  }

  const primarySingleLabel = queueHasWork || hasActiveUploads ? 'Add to Queue' : 'Generate Notes'

  return (
    <section
      className={`panel upload-panel ${isDraggingPanel ? 'upload-panel--dragging' : ''}`}
      aria-label="Upload files"
      onDragOver={handlePanelDragOver}
      onDragLeave={handlePanelDragLeave}
      onDrop={handlePanelDrop}
    >
      <div className="upload-grid">
        <UploadBox
          title="Video / Audio"
          subtitle="MP4, MOV, MKV, WEBM, MP3, WAV, M4A, and more"
          accept={RECORDING_ACCEPT}
          allowedExtensions={RECORDING_EXTENSIONS}
          invalidTypeMessage="Please select a supported video or audio file (.mp4, .mov, .mkv, .webm, .avi, .m4v, .mp3, .wav, .m4a, .aac, .ogg, .flac, .opus)."
          file={recordingFile}
          multiple
          onFileChange={onRecordingChange}
          onFilesSelect={onBulkFilesDrop}
        />

        <UploadBox
          title="Slides PDF"
          subtitle="PDF document (.pdf) — drop multiple to concatenate"
          accept={SLIDES_ACCEPT}
          allowedExtensions={SLIDES_EXTENSIONS}
          invalidTypeMessage="Please select valid .pdf slides files."
          files={slidesFiles}
          multiple
          onFilesChange={onSlidesChange}
        />
      </div>

      <p className="upload-panel__tip">
        Tip: Drop multiple recordings and slides PDFs together to auto-pair by filename.
      </p>

      {rejectedFiles.length > 0 ? (
        <div className="alert alert--warning upload-panel__alert" role="alert">
          <div className="alert__row">
            <div>
              <strong>Unsupported files skipped</strong>
              <p>{rejectedFiles.join(', ')}</p>
            </div>
            <button type="button" className="btn btn--ghost btn--sm" onClick={onDismissRejected}>
              Dismiss
            </button>
          </div>
        </div>
      ) : null}

      {uploadError ? (
        <div className="alert alert--error upload-panel__alert" role="alert">
          {uploadError}
        </div>
      ) : null}

      <div className="upload-panel__footer">
        {hasCompletePair ? (
          <button
            type="button"
            className="btn btn--ghost"
            onClick={onStageCurrentPair}
          >
            + Stage &amp; Add Another
          </button>
        ) : null}

        {readyStagedCount > 0 ? (
          <button
            type="button"
            className="btn btn--primary"
            onClick={onStartStagedUploads}
          >
            {`Upload & Start Queue (${pendingStartCount})`}
          </button>
        ) : (
          <button
            type="button"
            className="btn btn--primary"
            disabled={!hasCompletePair}
            onClick={onSubmitImmediate}
          >
            {primarySingleLabel}
          </button>
        )}
      </div>

      {unpairedRecordings.length > 0 || unpairedSlides.length > 0 ? (
        <div className="unpaired-drawer" aria-label="Unpaired files">
          <div className="unpaired-drawer__header">
            <div>
              <h3 className="unpaired-drawer__title">Unpaired files</h3>
              <p className="unpaired-drawer__subtitle">
                Pick one recording and one slides PDF to pair them manually.
              </p>
            </div>
            <div className="unpaired-drawer__actions">
              {unpairedRecordings.length > 0 && unpairedSlides.length > 0 ? (
                <>
                  <button
                    type="button"
                    className="btn btn--primary btn--sm"
                    disabled={!activeSelectedRecId || !activeSelectedSlideId}
                    onClick={handlePairSelectedClick}
                  >
                    Pair selected
                  </button>
                  <button
                    type="button"
                    className="btn btn--ghost btn--sm"
                    onClick={onPairRemainingInOrder}
                  >
                    Pair remaining in order
                  </button>
                </>
              ) : null}
            </div>
          </div>

          <div className="unpaired-drawer__columns">
            <div className="unpaired-column">
              <div className="unpaired-column__label">
                Recordings ({unpairedRecordings.length})
              </div>
              {unpairedRecordings.length === 0 ? (
                <p className="unpaired-column__empty">No unpaired recordings</p>
              ) : (
                <ul className="unpaired-list">
                  {unpairedRecordings.map((item) => {
                    const isSelected = activeSelectedRecId === item.id
                    return (
                      <li
                        key={item.id}
                        className={`unpaired-item ${isSelected ? 'unpaired-item--selected' : ''}`}
                      >
                        <button
                          type="button"
                          className="unpaired-item__select"
                          aria-pressed={isSelected}
                          onClick={() =>
                            setSelectedUnpairedRecId((prev) => (prev === item.id ? null : item.id))
                          }
                        >
                          <span className="unpaired-item__name" title={item.file.name}>
                            {item.file.name}
                          </span>
                          <span className="unpaired-item__size">
                            {formatFileSize(item.file.size)}
                          </span>
                        </button>
                        <button
                          type="button"
                          className="btn btn--ghost btn--sm unpaired-item__remove"
                          aria-label={`Remove unpaired recording ${item.file.name}`}
                          onClick={() => onRemoveUnpairedRecording(item.id)}
                        >
                          ×
                        </button>
                      </li>
                    )
                  })}
                </ul>
              )}
            </div>

            <div className="unpaired-column">
              <div className="unpaired-column__label">
                Slides PDFs ({unpairedSlides.length})
              </div>
              {unpairedSlides.length === 0 ? (
                <p className="unpaired-column__empty">No unpaired PDFs</p>
              ) : (
                <ul className="unpaired-list">
                  {unpairedSlides.map((item) => {
                    const isSelected = activeSelectedSlideId === item.id
                    return (
                      <li
                        key={item.id}
                        className={`unpaired-item ${isSelected ? 'unpaired-item--selected' : ''}`}
                      >
                        <button
                          type="button"
                          className="unpaired-item__select"
                          aria-pressed={isSelected}
                          onClick={() =>
                            setSelectedUnpairedSlideId((prev) =>
                              prev === item.id ? null : item.id,
                            )
                          }
                        >
                          <span className="unpaired-item__name" title={item.file.name}>
                            {item.file.name}
                          </span>
                          <span className="unpaired-item__size">
                            {formatFileSize(item.file.size)}
                          </span>
                        </button>
                        <button
                          type="button"
                          className="btn btn--ghost btn--sm unpaired-item__remove"
                          aria-label={`Remove unpaired slides ${item.file.name}`}
                          onClick={() => onRemoveUnpairedSlide(item.id)}
                        >
                          ×
                        </button>
                      </li>
                    )
                  })}
                </ul>
              )}
            </div>
          </div>
        </div>
      ) : null}

      {stagedPairs.length > 0 ? (
        <div className="staging-tray" aria-label="Staged uploads">
          <div className="staging-tray__header">
            <span className="staging-tray__title">
              Staged to Upload ({stagedPairs.length}{' '}
              {stagedPairs.length === 1 ? 'pair' : 'pairs'}
              {totalStagedBytes > 0 ? ` · ${formatFileSize(totalStagedBytes)}` : ''})
            </span>
            <button type="button" className="btn btn--ghost btn--sm" onClick={onClearStaged}>
              Clear
            </button>
          </div>

          <ul className="staging-list">
            {stagedPairs.map((pair, index) => {
              const isUploading = pair.status === 'uploading'
              const slidesDisplayName = formatSlidesDisplayName(pair.slides)
              const canMoveUp =
                !isUploading && index > 0 && stagedPairs[index - 1]?.status !== 'uploading'
              const canMoveDown =
                !isUploading &&
                index < stagedPairs.length - 1 &&
                stagedPairs[index + 1]?.status !== 'uploading'

              return (
                <li
                  key={pair.id}
                  className={`staging-item staging-item--${pair.status}`}
                >
                  <span className="staging-item__index">{index + 1}</span>
                  <div className="staging-item__body">
                    <div className="staging-item__files">
                      <span className="staging-item__filename" title={pair.recording.name}>
                        {pair.recording.name}
                      </span>
                      <span className="divider-dot" aria-hidden="true">
                        +
                      </span>
                      <span
                        className="staging-item__filename"
                        title={pair.slides.map((file) => file.name).join(' + ')}
                      >
                        {slidesDisplayName}
                      </span>
                    </div>

                    <div className="staging-item__status">
                      {pair.status === 'staged' ? (
                        <span className="staging-item__meta">
                          Ready to upload ({formatFileSize(pair.totalBytes)})
                        </span>
                      ) : pair.status === 'waiting' ? (
                        <span className="staging-item__meta">Waiting to upload</span>
                      ) : pair.status === 'uploading' ? (
                        <div className="staging-progress">
                          <div className="progress-track" aria-hidden="true">
                            <div
                              className="progress-track__fill"
                              style={{ width: `${pair.uploadPct}%` }}
                            />
                          </div>
                          <span className="staging-item__meta">
                            Uploading {pair.uploadPct}% — {formatFileSize(pair.loadedBytes)} /{' '}
                            {formatFileSize(pair.totalBytes)}
                          </span>
                        </div>
                      ) : pair.status === 'rate_limited' ? (
                        <span className="staging-item__warning" role="status">
                          Rate-limited, retrying in {pair.retryAfterSeconds ?? 1}s...
                        </span>
                      ) : (
                        <span className="staging-item__error" role="alert">
                          Upload failed: {pair.errorMessage ?? 'Please retry.'}
                        </span>
                      )}
                    </div>
                  </div>

                  <div className="staging-item__actions">
                    {pair.status === 'upload_error' ? (
                      <button
                        type="button"
                        className="btn btn--ghost btn--sm"
                        onClick={() => onRetryStagedPair(pair.id)}
                      >
                        Retry
                      </button>
                    ) : null}
                    <button
                      type="button"
                      className="btn btn--ghost btn--sm icon-btn"
                      aria-label={`Move staged pair ${slidesDisplayName} up`}
                      disabled={!canMoveUp}
                      onClick={() => onMoveStagedPair(pair.id, -1)}
                    >
                      ↑
                    </button>
                    <button
                      type="button"
                      className="btn btn--ghost btn--sm icon-btn"
                      aria-label={`Move staged pair ${slidesDisplayName} down`}
                      disabled={!canMoveDown}
                      onClick={() => onMoveStagedPair(pair.id, 1)}
                    >
                      ↓
                    </button>
                    <button
                      type="button"
                      className="btn btn--ghost btn--sm icon-btn"
                      aria-label={
                        isUploading
                          ? `Abort upload for ${slidesDisplayName}`
                          : `Remove staged pair ${slidesDisplayName}`
                      }
                      onClick={() => onRemoveStagedPair(pair.id)}
                    >
                      ×
                    </button>
                  </div>
                </li>
              )
            })}
          </ul>

          {hasActiveUploads ? (
            <p className="staging-tray__notice" role="status">
              Keep this tab open until uploads finish ({remainingUploadCount} remaining).
            </p>
          ) : null}
        </div>
      ) : null}

      {allUploadedNotice && stagedPairs.length === 0 ? (
        <p className="upload-panel__done-notice" role="status">
          All lectures uploaded — safe to close this tab.
        </p>
      ) : null}
    </section>
  )
}
