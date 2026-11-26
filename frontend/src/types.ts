export type JobStatus = 'queued' | 'processing' | 'completed' | 'failed' | 'cancelled'
export type StageStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'
export type PipelineRole = 'primary' | 'preparing'
export type PipelineStage =
  | 'upload'
  | 'audio_prep'
  | 'transcription'
  | 'ocr'
  | 'alignment'
  | 'notes'

export interface JobCreateResponse {
  token: string
  result_url: string
  status: JobStatus
  current_stage: PipelineStage
  created_at: string
  queue_position?: number | null
  batch_id?: string | null
}

export interface StageStateResponse {
  stage: PipelineStage
  label: string
  status: StageStatus
  started_at: string | null
  completed_at: string | null
  message: string | null
}

export interface JobErrorResponse {
  stage: PipelineStage
  stage_label: string
  message: string
  hint: string
}

export interface JobStatsResponse {
  duration_seconds?: number | null
  slide_count?: number | null
  transcript_segments?: number | null
  word_count?: number | null
  [key: string]: unknown
}

export interface JobStatusResponse {
  token: string
  result_url: string
  status: JobStatus
  current_stage: PipelineStage
  current_stage_label: string
  progress_pct: number
  slides_filename: string
  recording_filename: string
  created_at: string
  updated_at: string
  started_at: string | null
  completed_at: string | null
  download_ready: boolean
  download_url: string | null
  download_filename: string | null
  stages: StageStateResponse[]
  error: JobErrorResponse | null
  artifacts?: Record<string, string>
  stats: JobStatsResponse
  warnings: string[]
  queue_position?: number | null
  queue_index?: number | null
  pipeline_role?: PipelineRole | null
  waiting_reason?: string | null
  batch_id?: string | null
  queue_label?: string | null
  can_cancel?: boolean
  can_retry?: boolean
  can_retry_from_scratch?: boolean
}

export interface QueueItemResponse {
  token: string
  result_url: string
  status: JobStatus
  current_stage: PipelineStage
  current_stage_label: string
  progress_pct: number
  slides_filename: string
  recording_filename: string
  queue_label?: string | null
  batch_id?: string | null
  queue_position?: number | null
  queue_index?: number | null
  pipeline_role?: PipelineRole | null
  waiting_reason?: string | null
  created_at: string
  updated_at: string
  started_at?: string | null
  completed_at?: string | null
  download_ready: boolean
  download_url: string | null
  download_filename: string | null
  error: JobErrorResponse | null
  stats: JobStatsResponse
  can_cancel?: boolean
  can_retry?: boolean
  can_retry_from_scratch?: boolean
}

export interface QueueCountsResponse {
  processing: number
  queued: number
  completed: number
  failed: number
  cancelled: number
}

export interface QueueSnapshotResponse {
  paused: boolean
  pause_reason: string | null
  max_in_flight_jobs: number
  max_queue_depth: number
  counts: QueueCountsResponse
  active: JobStatusResponse[]
  queued: QueueItemResponse[]
  recent: QueueItemResponse[]
}

export interface HistoryEntry {
  token: string
  slidesFilename: string
  recordingFilename: string
  status: JobStatus
  createdAt: string
  updatedAt: string
  downloadReady: boolean
  downloadUrl: string | null
  downloadFilename: string | null
  queueLabel?: string | null
  canRetry?: boolean
  errorSummary?: string | null
  source?: 'server' | 'local'
}

export interface AuthStatusResponse {
  auth_required: boolean
  authenticated: boolean
}

export interface ConsoleLogEntry {
  key: string
  timestamp: string
  stage: PipelineStage
  label: string
  status: StageStatus
  message: string
}

export type StagedPairStatus =
  | 'staged'
  | 'waiting'
  | 'uploading'
  | 'rate_limited'
  | 'upload_error'

export interface StagedLecturePair {
  id: string
  recording: File
  slides: File[]
  label: string | null
  batchId: string | null
  status: StagedPairStatus
  uploadPct: number
  loadedBytes: number
  totalBytes: number
  retryAfterSeconds: number | null
  errorMessage: string | null
}

export interface UnpairedFile {
  id: string
  file: File
}
