import { useId, useRef, useState } from 'react'
import type { ChangeEvent, DragEvent, KeyboardEvent } from 'react'
import { formatFileSize, hasAllowedExtension } from '../constants'

interface UploadBoxProps {
  title: string
  subtitle: string
  accept: string
  allowedExtensions: readonly string[]
  invalidTypeMessage: string
  file: File | null
  disabled?: boolean
  multiple?: boolean
  onFileChange: (file: File | null) => void
  onFilesSelect?: (files: File[]) => void
}

export function UploadBox({
  title,
  subtitle,
  accept,
  allowedExtensions,
  invalidTypeMessage,
  file,
  disabled = false,
  multiple = false,
  onFileChange,
  onFilesSelect,
}: UploadBoxProps) {
  const inputId = useId()
  const inputRef = useRef<HTMLInputElement | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [validationError, setValidationError] = useState<string | null>(null)

  const handleSelectedFiles = (fileList: FileList | null) => {
    if (!fileList || fileList.length === 0) return
    const files = Array.from(fileList)

    if (files.length > 1 && onFilesSelect) {
      setValidationError(null)
      onFilesSelect(files)
      return
    }

    const candidate = files[0]
    if (!candidate) return
    if (!hasAllowedExtension(candidate.name, allowedExtensions)) {
      setValidationError(invalidTypeMessage)
      return
    }
    setValidationError(null)
    onFileChange(candidate)
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
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault()
      triggerPicker()
    }
  }

  const boxClasses = [
    'upload-box',
    isDragging ? 'upload-box--dragging' : '',
    file ? 'upload-box--selected' : '',
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
          multiple={multiple}
          aria-label={`${title} file input`}
          className="sr-only"
          disabled={disabled}
          onChange={handleInputChange}
        />

        <div className="upload-box__header">
          <span className="upload-box__title">{title}</span>
          {file ? (
            <button
              type="button"
              className="upload-box__clear"
              disabled={disabled}
              onClick={(event) => {
                event.stopPropagation()
                setValidationError(null)
                onFileChange(null)
              }}
            >
              Remove
            </button>
          ) : null}
        </div>

        {file ? (
          <div className="upload-box__file">
            <span className="upload-box__filename" title={file.name}>
              {file.name}
            </span>
            <span className="upload-box__meta">{formatFileSize(file.size)}</span>
          </div>
        ) : (
          <div className="upload-box__placeholder">
            <span className="upload-box__prompt">
              {multiple ? 'Drop file(s) here or click to browse' : 'Drop file here or click to browse'}
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
