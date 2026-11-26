# Plan for the lecture source export

**Status** — Proposed prompt changes. The deliverable is one English Markdown source that preserves what a student could learn from the lecture recording and slide deck together. NotebookLM handles source partitioning, retrieval, and study aids after upload.

## Review of the proposed NotebookLM prompt

| Proposal | Decision and reasoning |
| --- | --- |
| Preserve every substantive spoken explanation | **Keep.** The existing generator already asks for mechanisms, examples, derivations, caveats, and substantive Q&A. Preserve the order of a worked explanation, including steps that cross slide boundaries. Let the depth of the teaching determine the length. |
| Separate slide text, diagrams, and speech for every topic | **Use attribution where it matters.** A slide can show two curves while the lecturer explains why they diverge. Describe those together. Identify the source explicitly when a claim appears only on the slide, a student proposes an interpretation, or speech conflicts with the diagram. |
| Flag assessment cues and scope boundaries | **Adopt the distinction.** Keep explicit requirements and explicit limits beside the relevant concept, with a timestamp. Preserve the lecturer’s level of certainty. “Important to understand” alone is a study cue. “Probably on the exam” carries a different commitment from “will be on the exam.” |
| Record ASR discrepancies and self-corrections | **Apply a meaning-based rule.** Use the slide’s spelling for an obvious transcription error when the intended term is clear. Retain a self-correction that changes the explanation. If two plausible readings would teach different things, identify what the transcript and slide each support and leave the uncertainty visible. |
| Reconstruct diagrams in ASCII or Mermaid | **Choose a representation by what the visual teaches.** Prose can capture a relationship, a table can preserve values and units, and a numbered sequence can explain a flow. A diagram is useful when the supplied evidence establishes its connections and a drawing conveys them more clearly. |
| Require full heading paths and six blocks per slide | **Use natural topic headings and source anchors.** Several slides can support one argument. Off-slide discussion can stand on its timestamp. Repeating the heading hierarchy or filling out empty categories adds text without recovering any teaching. |
| Add a formula glossary and exam checklist | **Leave generated study aids to NotebookLM.** Keep formulas, variable definitions supplied by the sources, and assessment guidance where they were taught. Preserve a concluding recap if the lecturer or deck actually provides one. |
| Replace section generation with one long-context request | **Retain the current pipeline for this change.** It already supplies structured slide extraction, timestamped speech, alignment, and continuity context, then assembles one file. The suggested single request is an alternative implementation. The prompt review supplies no evidence that switching would improve source fidelity. |

## Proposed implementation

1. **Revise the generation instructions.** In [prompts.py](backend/app/pipeline/prompts.py), update the section system prompt, Markdown rules, and section request together. State the job as preserving the lecture for use as a source. Keep descriptive headings, confirmed slide references, timestamps, and existing safeguards for uncertain alignment. Replace the instruction to make each chunk self-contained with a narrower instruction to resolve references such as “this arrow” when the slide establishes their meaning.
2. **Make attribution and uncertainty explicit.** Ask the model to preserve who asked, asserted, corrected, or speculated when that distinction affects meaning. Keep concrete assessment requirements and scope limits in the relevant passage. Use a short discrepancy note only when a reader needs it to interpret the teaching. Omit unknown metadata and use only source-supported course details.
3. **Reuse the existing section input and document assembly.** Feed the section writer its existing aligned transcript, slide text, visual descriptions, and extracted tables, code, and formulas. Continue assembling sections in lecture order through [notes.py](backend/app/pipeline/notes.py). The example course, lecturer, filenames, and subject vocabulary in the supplied prompt are sample input.

## How to decide whether the change helped

Compare the current and revised exports using the **same saved transcript, slide extraction, and alignment**. Choose the passages from the source material before reading the outputs. Use the existing real-lecture check for three speech-only questions, and inspect these features where the lecture contains them.

- Can a reader reconstruct a worked example, including what happened and why, using the export alone?
- When the speaker points to a figure, does the text identify the relevant element, relationship, or value accurately? Can the reader tell when its identity was uncertain?
- Does a classroom exchange retain who proposed an idea, how the lecturer responded, and any uncertainty in that response?
- Do explicit assignment requirements, exam hints, and limits on what to learn retain their scope and timestamp?
- Can the reader follow an explanation across a slide change or an off-slide passage without losing a step?

Accept the prompt revision when the sampled passages preserve those details and their anchors, and the simpler layout removes repeated labels without losing teaching. These checks evaluate the exported source directly.
