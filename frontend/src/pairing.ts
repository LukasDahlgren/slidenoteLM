import { RECORDING_EXTENSIONS, SLIDES_EXTENSIONS, hasAllowedExtension } from './constants'

const NOISE_TOKENS = new Set([
  'slides',
  'slide',
  'deck',
  'recording',
  'record',
  'rec',
  'video',
  'vid',
  'audio',
  'lecture',
  'lec',
  'final',
])

const VERSION_TOKEN_RE = /^v\d+$/
const LABELED_INDEX_RE =
  /(?:^|[\s._()[\]-])(?:lecture|lec|week|wk|chapter|ch|session|sess|module|mod|unit|part|class|lesson)[\s._-]*0*(\d+)(?=$|[\s._()[\]-])/i
const LEADING_INDEX_RE = /^0*(\d+)(?=[\s._-]|$)/

const naturalCollator = new Intl.Collator(undefined, {
  numeric: true,
  sensitivity: 'base',
})

export interface AutoMatchedPair {
  recording: File
  slides: File
  matchReason: 'stem' | 'index'
}

export interface AutoPairResult {
  pairs: AutoMatchedPair[]
  unpairedRecordings: File[]
  unpairedSlides: File[]
  rejected: File[]
}

export interface PartitionedFiles {
  recordings: File[]
  slides: File[]
  rejected: File[]
}

export function stripExtension(filename: string): string {
  const trimmed = filename.trim()
  const dotIndex = trimmed.lastIndexOf('.')
  if (dotIndex <= 0) return trimmed
  return trimmed.slice(0, dotIndex)
}

export function normalizeStem(filename: string): string {
  const stem = stripExtension(filename).toLowerCase()
  const rawTokens = stem
    .split(/[^a-z0-9]+/)
    .map((token) => token.trim())
    .filter(Boolean)

  if (rawTokens.length === 0) return ''

  const filtered = rawTokens.filter(
    (token) => !NOISE_TOKENS.has(token) && !VERSION_TOKEN_RE.test(token),
  )

  const effectiveTokens = filtered.length > 0 ? filtered : rawTokens
  return effectiveTokens.join('')
}

export function extractLectureIndex(filename: string): number | null {
  const stem = stripExtension(filename).trim().toLowerCase()
  if (!stem) return null

  const labeledMatch = stem.match(LABELED_INDEX_RE)
  if (labeledMatch) {
    const parsed = Number.parseInt(labeledMatch[1], 10)
    if (Number.isFinite(parsed) && parsed >= 0 && parsed <= 999) {
      return parsed
    }
  }

  const leadingMatch = stem.match(LEADING_INDEX_RE)
  if (leadingMatch) {
    const parsed = Number.parseInt(leadingMatch[1], 10)
    if (Number.isFinite(parsed) && parsed >= 0 && parsed <= 999) {
      return parsed
    }
  }

  return null
}

export function partitionFiles(files: readonly File[]): PartitionedFiles {
  const recordings: File[] = []
  const slides: File[] = []
  const rejected: File[] = []

  for (const file of files) {
    if (hasAllowedExtension(file.name, SLIDES_EXTENSIONS)) {
      slides.push(file)
    } else if (hasAllowedExtension(file.name, RECORDING_EXTENSIONS)) {
      recordings.push(file)
    } else {
      rejected.push(file)
    }
  }

  return { recordings, slides, rejected }
}

export function compareFilenamesNaturally(a: string, b: string): number {
  return naturalCollator.compare(a, b)
}

export function autoPairFiles(
  incomingFiles: readonly File[],
  existingUnpairedRecordings: readonly File[] = [],
  existingUnpairedSlides: readonly File[] = [],
): AutoPairResult {
  const partitioned = partitionFiles(incomingFiles)
  const recordings = [...existingUnpairedRecordings, ...partitioned.recordings]
  const slides = [...existingUnpairedSlides, ...partitioned.slides]

  const matchedRecordingIndices = new Set<number>()
  const matchedSlideIndices = new Set<number>()
  const pairs: AutoMatchedPair[] = []

  // Pass 1: exact normalized stem match (1-to-1 only to avoid mis-pairing duplicates)
  const recByStem = new Map<string, number[]>()
  const slideByStem = new Map<string, number[]>()

  recordings.forEach((file, index) => {
    const key = normalizeStem(file.name)
    if (!key) return
    const list = recByStem.get(key)
    if (list) list.push(index)
    else recByStem.set(key, [index])
  })

  slides.forEach((file, index) => {
    const key = normalizeStem(file.name)
    if (!key) return
    const list = slideByStem.get(key)
    if (list) list.push(index)
    else slideByStem.set(key, [index])
  })

  for (const [stemKey, recIndices] of recByStem.entries()) {
    const slideIndices = slideByStem.get(stemKey)
    if (recIndices.length === 1 && slideIndices && slideIndices.length === 1) {
      const rIdx = recIndices[0]
      const sIdx = slideIndices[0]
      matchedRecordingIndices.add(rIdx)
      matchedSlideIndices.add(sIdx)
      pairs.push({
        recording: recordings[rIdx],
        slides: slides[sIdx],
        matchReason: 'stem',
      })
    }
  }

  // Pass 2: unambiguous numeric lecture/week index match (1-to-1 only)
  const recByIndex = new Map<number, number[]>()
  const slideByIndex = new Map<number, number[]>()

  recordings.forEach((file, index) => {
    if (matchedRecordingIndices.has(index)) return
    const num = extractLectureIndex(file.name)
    if (num === null) return
    const list = recByIndex.get(num)
    if (list) list.push(index)
    else recByIndex.set(num, [index])
  })

  slides.forEach((file, index) => {
    if (matchedSlideIndices.has(index)) return
    const num = extractLectureIndex(file.name)
    if (num === null) return
    const list = slideByIndex.get(num)
    if (list) list.push(index)
    else slideByIndex.set(num, [index])
  })

  for (const [lectureNum, recIndices] of recByIndex.entries()) {
    const slideIndices = slideByIndex.get(lectureNum)
    if (recIndices.length === 1 && slideIndices && slideIndices.length === 1) {
      const rIdx = recIndices[0]
      const sIdx = slideIndices[0]
      matchedRecordingIndices.add(rIdx)
      matchedSlideIndices.add(sIdx)
      pairs.push({
        recording: recordings[rIdx],
        slides: slides[sIdx],
        matchReason: 'index',
      })
    }
  }

  pairs.sort((a, b) => compareFilenamesNaturally(a.slides.name, b.slides.name))

  const unpairedRecordings = recordings
    .filter((_, index) => !matchedRecordingIndices.has(index))
    .sort((a, b) => compareFilenamesNaturally(a.name, b.name))

  const unpairedSlides = slides
    .filter((_, index) => !matchedSlideIndices.has(index))
    .sort((a, b) => compareFilenamesNaturally(a.name, b.name))

  return {
    pairs,
    unpairedRecordings,
    unpairedSlides,
    rejected: partitioned.rejected,
  }
}

export function pairInNaturalOrder(
  recordings: readonly File[],
  slides: readonly File[],
): {
  pairs: Array<{ recording: File; slides: File }>
  remainingRecordings: File[]
  remainingSlides: File[]
} {
  const sortedRecordings = [...recordings].sort((a, b) =>
    compareFilenamesNaturally(a.name, b.name),
  )
  const sortedSlides = [...slides].sort((a, b) => compareFilenamesNaturally(a.name, b.name))

  const count = Math.min(sortedRecordings.length, sortedSlides.length)
  const pairs: Array<{ recording: File; slides: File }> = []

  for (let i = 0; i < count; i += 1) {
    pairs.push({
      recording: sortedRecordings[i],
      slides: sortedSlides[i],
    })
  }

  return {
    pairs,
    remainingRecordings: sortedRecordings.slice(count),
    remainingSlides: sortedSlides.slice(count),
  }
}
