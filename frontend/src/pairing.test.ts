import { describe, expect, it } from 'vitest'
import {
  autoPairFiles,
  extractLectureIndex,
  normalizeStem,
  pairInNaturalOrder,
  partitionFiles,
} from './pairing'

function makeFile(name: string, sizeBytes = 100): File {
  return new File(['x'.repeat(sizeBytes)], name)
}

describe('pairing utilities', () => {
  it('normalizes stems by removing extensions, noise words, version tokens, and punctuation', () => {
    expect(normalizeStem('03_Graphs_Slides.pdf')).toBe('03graphs')
    expect(normalizeStem('03_Graphs_Recording.mp4')).toBe('03graphs')
    expect(normalizeStem('03-Graphs-Final-v2.m4a')).toBe('03graphs')
    expect(normalizeStem('Lecture_05_Dynamic_Programming_Deck.PDF')).toBe('05dynamicprogramming')
    expect(normalizeStem('Lec 05 Dynamic Programming Video.mkv')).toBe('05dynamicprogramming')
    // Falls back to raw stem if all tokens are noise words
    expect(normalizeStem('lecture.mp4')).toBe('lecture')
    expect(normalizeStem('lecture.pdf')).toBe('lecture')
  })

  it('extracts lecture/week/chapter numeric indices cleanly and ignores 4-digit years', () => {
    expect(extractLectureIndex('lec03_video.mp4')).toBe(3)
    expect(extractLectureIndex('Lecture 3 - Sorting.pdf')).toBe(3)
    expect(extractLectureIndex('week-07-notes.pdf')).toBe(7)
    expect(extractLectureIndex('ch12_recording.mp3')).toBe(12)
    expect(extractLectureIndex('04_trees.mp4')).toBe(4)
    expect(extractLectureIndex('2026_03_01_recording.mp4')).toBeNull()
    expect(extractLectureIndex('overview_slides.pdf')).toBeNull()
  })

  it('partitions recordings, slides PDFs, and unsupported files', () => {
    const result = partitionFiles([
      makeFile('a.mp4'),
      makeFile('b.m4a'),
      makeFile('c.pdf'),
      makeFile('readme.txt'),
      makeFile('deck.pptx'),
    ])

    expect(result.recordings.map((f) => f.name)).toEqual(['a.mp4', 'b.m4a'])
    expect(result.slides.map((f) => f.name)).toEqual(['c.pdf'])
    expect(result.rejected.map((f) => f.name)).toEqual(['readme.txt', 'deck.pptx'])
  })

  it('pairs by exact normalized stem in Pass 1 and sorts results naturally', () => {
    const files = [
      makeFile('10_Heaps_Recording.mp4'),
      makeFile('2_Trees_Slides.pdf'),
      makeFile('2_Trees_Video_v2.mp4'),
      makeFile('10_Heaps_Slides_final.pdf'),
      makeFile('1_Intro_Deck.pdf'),
      makeFile('1_Intro_Audio.mp3'),
    ]

    const result = autoPairFiles(files)

    expect(result.pairs.map((p) => [p.slides.name, p.recording.name, p.matchReason])).toEqual([
      ['1_Intro_Deck.pdf', '1_Intro_Audio.mp3', 'stem'],
      ['2_Trees_Slides.pdf', '2_Trees_Video_v2.mp4', 'stem'],
      ['10_Heaps_Slides_final.pdf', '10_Heaps_Recording.mp4', 'stem'],
    ])
    expect(result.unpairedRecordings).toHaveLength(0)
    expect(result.unpairedSlides).toHaveLength(0)
    expect(result.rejected).toHaveLength(0)
  })

  it('pairs unambiguous numeric indices in Pass 2 when stems differ', () => {
    const files = [
      makeFile('lec03_zoom_capture.mp4'),
      makeFile('Week 3 - Graph Algorithms.pdf'),
      makeFile('04_classroom_audio.m4a'),
      makeFile('Lecture 4 Trees.pdf'),
    ]

    const result = autoPairFiles(files)

    expect(result.pairs.map((p) => [p.slides.name, p.recording.name, p.matchReason])).toEqual([
      ['Lecture 4 Trees.pdf', '04_classroom_audio.m4a', 'index'],
      ['Week 3 - Graph Algorithms.pdf', 'lec03_zoom_capture.mp4', 'index'],
    ])
    expect(result.unpairedRecordings).toHaveLength(0)
    expect(result.unpairedSlides).toHaveLength(0)
  })

  it('never silently mis-pairs ambiguous duplicates or unrelated names', () => {
    const files = [
      makeFile('week3_partA.mp4'),
      makeFile('week3_partB.mp4'),
      makeFile('week3_slides.pdf'),
      makeFile('guest_talk.mp4'),
      makeFile('bonus_handout.pdf'),
    ]

    const result = autoPairFiles(files)

    expect(result.pairs).toHaveLength(0)
    expect(result.unpairedRecordings.map((f) => f.name)).toEqual([
      'guest_talk.mp4',
      'week3_partA.mp4',
      'week3_partB.mp4',
    ])
    expect(result.unpairedSlides.map((f) => f.name)).toEqual([
      'bonus_handout.pdf',
      'week3_slides.pdf',
    ])
  })

  it('pairs remaining files in natural alphabetical order via pairInNaturalOrder', () => {
    const recordings = [makeFile('rec10.mp4'), makeFile('rec2.mp4'), makeFile('rec1.mp4')]
    const slides = [makeFile('deck2.pdf'), makeFile('deck1.pdf')]

    const result = pairInNaturalOrder(recordings, slides)

    expect(result.pairs.map((p) => [p.recording.name, p.slides.name])).toEqual([
      ['rec1.mp4', 'deck1.pdf'],
      ['rec2.mp4', 'deck2.pdf'],
    ])
    expect(result.remainingRecordings.map((f) => f.name)).toEqual(['rec10.mp4'])
    expect(result.remainingSlides).toHaveLength(0)
  })
})
