# Book Production Workflow

This workflow turns lecture PDFs and papers into chapters. It is resumable through `book/STATUS.md`.

## Per-Chapter Pipeline

### Stage 1 — Source Extraction

Input: PDFs listed for the chapter in `book/spine.yml`.

Output:

- extracted notes in `book/evidence/source-notes/<slug>.md`;
- source cards in `book/evidence/source-ledger/`;
- initial claim list with evidence grades.

Do not draft prose yet.

### Stage 2 — Chapter Brief

Input: source notes, source cards, chapter thesis from `book/spine.yml`.

Output: `book/chapters/<slug>-brief.md`.

The brief must define the reader problem, system bottleneck, explanation arc, figure list, and technical review risks.

### Stage 3 — Draft

Input: approved brief and source cards.

Output: `book/chapters/<slug>.md`.

The draft should read as a chapter, not as annotated slides. It must carry `[CITE: card-slug]` markers for load-bearing claims.

### Stage 4 — Technical Review

Check:

- equations and dimensional analysis;
- complexity claims;
- memory accounting;
- hardware and benchmark assumptions;
- pseudocode correctness;
- terminology consistency.

Output: review memo in `process/review-memos/<slug>-technical-review.md`.

### Stage 5 — Red Team

Attack the chapter as a skeptical technical reader:

- unsupported speedup claim;
- hidden condition on hardware or workload;
- overgeneralized system conclusion;
- missing baseline;
- confusing explanation order;
- diagram that hides the actual bottleneck.

Output: red-team memo in `process/review-memos/<slug>-red-team.md`.

### Stage 6 — Ready Promotion

Update `book/STATUS.md` only after source extraction, brief, draft, technical review, and red team are complete.

Allowed promotion: `red-team -> ready`.

Use `gated` if any chapter-level claim rests only on C-grade evidence.

## Wave Plan

1. Part I sequentially. It establishes vocabulary and the reader contract.
2. Part II sequentially. GPU chapters depend on earlier memory and tensor explanations.
3. Part III can split: DDP/model-parallel/ZeRO-MoE may be developed in parallel after framework notes are extracted.
4. Part V can split by serving subsystem after the inference cost model is written.
5. Part VI last. It synthesizes the whole book and should not be drafted early.

## Status Updates

After each stage:

1. Update the chapter row in `book/STATUS.md`.
2. Add or resolve open technical questions.
3. Record terminology decisions that affect later chapters.
4. Leave a handoff block in the artifact.

