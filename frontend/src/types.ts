export type JobStatus = 'queued' | 'processing' | 'completed' | 'failed'
export type StageStatus = 'pending' | 'running' | 'completed' | 'failed'
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
