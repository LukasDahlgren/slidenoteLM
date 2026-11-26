import { useId, useRef, useState } from 'react'
import type { ChangeEvent, DragEvent, KeyboardEvent } from 'react'
import { formatFileSize, hasAllowedExtension } from '../constants'

interface UploadBoxProps {
  title: string
  subtitle: string
  accept: string
  allowedExtensions: readonly string[]
  invalidTypeMessage: string
  file?: File | null
  files?: File[]
  disabled?: boolean
  multiple?: boolean
  onFileChange?: (file: File | null) => void
  onFilesChange?: (files: File[]) => void
  onFilesSelect?: (files: File[]) => void
}

export function UploadBox({
  title,
  subtitle,
  accept,
  allowedExtensions,
  invalidTypeMessage,
  file = null,
  files,
  disabled = false,
  multiple = false,
  onFileChange,
  onFilesChange,
  onFilesSelect,
}: UploadBoxProps) {
  const inputId = useId()
  const inputRef = useRef<HTMLInputElement | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [validationError, setValidationError] = useState<string | null>(null)

  const isMultiFileList = files !== undefined && onFilesChange !== undefined
  const selectedFiles = isMultiFileList ? files : file ? [file] : []
  const hasSelection = selectedFiles.length > 0

  const handleSelectedFiles = (fileList: FileList | null) => {
    if (!fileList || fileList.length === 0) return
    const incoming = Array.from(fileList)

    if (isMultiFileList) {
      const allValid = incoming.every((candidate) =>
        hasAllowedExtension(candidate.name, allowedExtensions),
      )
      if (!allValid) {
        setValidationError(invalidTypeMessage)
        return
      }
      setValidationError(null)
      onFilesChange([...files, ...incoming])
      return
    }

    if (incoming.length > 1 && onFilesSelect) {
      setValidationError(null)
      onFilesSelect(incoming)
      return
    }

    const candidate = incoming[0]
    if (!candidate) return
    if (!hasAllowedExtension(candidate.name, allowedExtensions)) {
      setValidationError(invalidTypeMessage)
      return
    }
    setValidationError(null)
    onFileChange?.(candidate)
  }

  const handleMoveFile = (index: number, direction: -1 | 1) => {
    if (!isMultiFileList) return
    const targetIndex = index + direction
    if (targetIndex < 0 || targetIndex >= files.length) return
    const next = files.slice()
    const [moved] = next.splice(index, 1)
    next.splice(targetIndex, 0, moved)
    onFilesChange(next)
  }

  const handleRemoveFileAt = (index: number) => {
    if (!isMultiFileList) return
    onFilesChange(files.filter((_, idx) => idx !== index))
  }

  const handleInputChange = (event: ChangeEvent<HTMLInputElement>) => {
    handleSelectedFiles(event.target.files)
    event.target.value = ''
  }

  const handleDragOver = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    if (!disabled) {
      setIsDragging(true)
    }
  }

  const handleDragLeave = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    if (!event.currentTarget.contains(event.relatedTarget as Node | null)) {
      setIsDragging(false)
    }
  }

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    event.stopPropagation()
    setIsDragging(false)
    if (disabled) return
    handleSelectedFiles(event.dataTransfer.files)
  }

  const triggerPicker = () => {
    if (!disabled) {
      inputRef.current?.click()
    }
  }

  const handleKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.target !== event.currentTarget) return
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault()
      triggerPicker()
    }
  }

  const totalBytes = selectedFiles.reduce((sum, item) => sum + item.size, 0)

  const boxClasses = [
    'upload-box',
    isDragging ? 'upload-box--dragging' : '',
    hasSelection ? 'upload-box--selected' : '',
    validationError ? 'upload-box--error' : '',
    disabled ? 'upload-box--disabled' : '',
  ]
    .filter(Boolean)
    .join(' ')

  return (
    <div className="upload-box-wrapper">
      <div
        role="button"
        tabIndex={disabled ? -1 : 0}
        aria-label={title}
        aria-disabled={disabled}
        className={boxClasses}
        onClick={triggerPicker}
        onKeyDown={handleKeyDown}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <input
          ref={inputRef}
          id={inputId}
          type="file"
          accept={accept}
          multiple={multiple || isMultiFileList}
          aria-label={`${title} file input`}
          className="sr-only"
          disabled={disabled}
          onChange={handleInputChange}
        />

        <div className="upload-box__header">
          <div className="upload-box__title-group">
            <span className="upload-box__title">{title}</span>
            {isMultiFileList && selectedFiles.length > 1 ? (
              <span className="upload-box__badge">
                {selectedFiles.length} PDFs · combined
              </span>
            ) : null}
          </div>
          {hasSelection ? (
            <button
              type="button"
              className="upload-box__clear"
              disabled={disabled}
              onClick={(event) => {
                event.stopPropagation()
                setValidationError(null)
                if (isMultiFileList) {
                  onFilesChange([])
                } else {
                  onFileChange?.(null)
                }
              }}
            >
              {isMultiFileList && selectedFiles.length > 1 ? 'Clear' : 'Remove'}
            </button>
          ) : null}
        </div>

        {isMultiFileList && selectedFiles.length > 1 ? (
          <div className="upload-box__multi">
            <ul className="upload-box__file-list">
              {selectedFiles.map((item, index) => (
                <li key={`${item.name}-${item.size}-${index}`} className="upload-box__file-row">
                  <span className="upload-box__file-order">{index + 1}.</span>
                  <span className="upload-box__file-name" title={item.name}>
                    {item.name}
                  </span>
                  <span className="upload-box__file-size">{formatFileSize(item.size)}</span>
                  <div className="upload-box__file-actions">
                    <button
                      type="button"
                      className="upload-box__row-btn"
                      aria-label={`Move ${item.name} up`}
                      disabled={disabled || index === 0}
                      onClick={(event) => {
                        event.stopPropagation()
                        handleMoveFile(index, -1)
                      }}
                    >
                      ↑
                    </button>
                    <button
                      type="button"
                      className="upload-box__row-btn"
                      aria-label={`Move ${item.name} down`}
                      disabled={disabled || index === selectedFiles.length - 1}
                      onClick={(event) => {
                        event.stopPropagation()
                        handleMoveFile(index, 1)
                      }}
                    >
                      ↓
                    </button>
                    <button
                      type="button"
                      className="upload-box__row-btn upload-box__row-btn--remove"
                      aria-label={`Remove ${item.name}`}
                      disabled={disabled}
                      onClick={(event) => {
                        event.stopPropagation()
                        handleRemoveFileAt(index)
                      }}
                    >
                      ×
                    </button>
                  </div>
                </li>
              ))}
            </ul>
            <span className="upload-box__add-more">
              Total {formatFileSize(totalBytes)} · Drop or click to add another PDF
            </span>
          </div>
        ) : hasSelection ? (
          <div className="upload-box__file">
            <span className="upload-box__filename" title={selectedFiles[0].name}>
              {selectedFiles[0].name}
            </span>
            <span className="upload-box__meta">{formatFileSize(selectedFiles[0].size)}</span>
            {isMultiFileList ? (
              <span className="upload-box__add-more">
                Drop or click to add another PDF to concatenate
              </span>
            ) : null}
          </div>
        ) : (
          <div className="upload-box__placeholder">
            <span className="upload-box__prompt">
              {isMultiFileList
                ? 'Drop one or more PDFs here or click to browse'
                : multiple
                  ? 'Drop file(s) here or click to browse'
                  : 'Drop file here or click to browse'}
            </span>
            <span className="upload-box__hint">{subtitle}</span>
          </div>
        )}
      </div>

      {validationError ? (
        <p className="upload-box__error-text" role="alert">
          {validationError}
        </p>
      ) : null}
    </div>
  )
}
