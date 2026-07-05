# LLM Systems Book — Project Instructions

> Single source of truth. `CLAUDE.md` and `GEMINI.md` import this file. Codex reads this file directly. Edit this file when project rules change.

Project: a serious technical nonfiction book based on CMU's Large Language Model Systems course material and supporting papers in `downloads/llmsystem2026spring/source_pdfs/`.

Working title: **Large Language Model Systems: From Tokens to Serving Infrastructure**

## Core Promise

This book explains how modern large language model systems work as systems: the probability objective, Transformer computation, GPU kernels, training frameworks, distributed training, compression, inference scheduling, KV-cache management, and serving architecture.

It is not a prompt-engineering handbook, a product survey, or a loose set of lecture notes.

## Reader

Primary reader: an engineer, graduate student, or technical founder who knows basic programming and machine learning, and wants to understand the system design behind training and serving LLMs.

Assume comfort with Python and linear algebra. Explain CUDA, distributed systems, and compiler/runtime details from first principles when they become load-bearing.

## Five Over-Rules

1. **System problem first.** Every chapter begins from a concrete bottleneck or design problem, not from a pile of slide bullets.
2. **Evidence before fluency.** Do not make a technical claim more confident than its source supports.
3. **Conditions travel with numbers.** Any latency, throughput, memory, bandwidth, FLOP, or speedup claim must carry model, hardware, sequence length, batch size, precision, or source context.
4. **Mechanism before opinion.** Prefer equations, dataflow, memory layout, scheduling diagrams, and minimal pseudocode over vague judgments.
5. **Handoff cleanly.** Every substantial output states assumptions, evidence grade, open questions, and next owner.

## Default Chapter Questions

Every chapter should answer:

1. What problem does this chapter solve?
2. Why does the problem become hard at LLM scale?
3. What is the mathematical or algorithmic core?
4. What is the system bottleneck: compute, memory, bandwidth, communication, scheduling, or reliability?
5. Which lecture PDFs, papers, or systems support the explanation?
6. What tradeoff should the reader remember?
7. What common misconception should be corrected?

## Repository Layout

- `book/spine.yml` — canonical book structure.
- `book/STATUS.md` — status board and resumability ledger.
- `book/chapters/` — chapter briefs and drafts.
- `book/evidence/source-ledger/` — source cards behind factual and technical claims.
- `book/evidence/source-notes/` — extracted notes from lecture PDFs and papers.
- `book/figures/` — figure specs, diagrams, and generated assets.
- `book/registries/` — terminology, symbols, callbacks, and chapter dependency maps.
- `process/review-memos/` — technical review, red-team, and reader reports.
- `process/audits/` — cross-chapter consistency and source audits.
- `.claude/rules/` — writing and verification rules.
- `.claude/docs/` — templates and workflow.
- `.codex/` / `.gemini/` — tool bridge configuration.

## Source Material

Primary sources live in:

```text
downloads/llmsystem2026spring/source_pdfs/
```

Treat the PDFs as source material, not as the book's structure. A chapter may combine several lectures and papers if that better serves the reader's understanding.

## Evidence Standard

Use the technical evidence grades in `.claude/rules/02-technical-evidence-grades.md`.

Short form:

- A: lecture PDF, original paper, official documentation, source code, benchmark artifact with reproducible setup.
- B: high-quality engineering blog, author talk, reputable secondary explanation, benchmark report with enough context.
- C: community explanation, unresolved issue, unreproduced claim, ambiguous benchmark.
- D: do not use as a factual anchor.

A chapter-level technical claim cannot rest only on C-grade evidence.

## Citation Discipline

Use the three-layer citation model in `.claude/rules/04-citation-form.md`:

1. Name the source in prose when it matters to the reader.
2. Use inline `[CITE: card-slug]` markers only as source-ledger identifiers.
3. Generate endnotes or reference lists from source cards later.

## Style

Voice: precise, concrete, patient, engineering-minded.

Prefer:

- concrete bottlenecks;
- dimensional analysis;
- memory and dataflow diagrams;
- small examples before general equations;
- explicit tradeoffs;
- "under these conditions" language.

Avoid:

- marketing adjectives;
- unexplained acronyms;
- unsupported "best", "optimal", "always faster";
- dumping slide bullets into prose;
- turning every section into a survey.

## Chapter State Machine

Allowed chapter states:

```text
queued -> sources-extracted -> brief -> draft -> technical-review -> red-team -> ready
```

Use `gated` when a chapter cannot advance because of a missing source, unresolved formula, or disputed technical claim.

## Default Artifact Block

Every substantial artifact should end with:

```text
Owner:
Purpose:
Evidence grade:
Assumptions:
Open questions:
Handoff:
```

