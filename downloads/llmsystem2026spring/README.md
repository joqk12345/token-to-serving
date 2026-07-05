# LLM System 2026 Spring Learning Materials

This directory contains downloaded course materials, extracted text, freshman-oriented teaching pages, and the Qwen3.5-122B-A10B FP8 kernel analysis.

## Entry Points

- Freshman interactive course: [freshman_course/index.html](freshman_course/index.html)
- Qwen3.5-122B-A10B FP8 analysis visualization: [qwen_fp8_analysis/index.html](qwen_fp8_analysis/index.html)
- Qwen3.5-122B-A10B FP8 analysis notes: [qwen_fp8_analysis/analysis.md](qwen_fp8_analysis/analysis.md)
- Change history: [CHANGELOG.md](CHANGELOG.md)

## Directory Layout

```text
llmsystem2026spring/
  README.md
  CHANGELOG.md
  source_pdfs/
    Original downloaded course slides and reference papers.
  extracted_text/
    Text extracted from course PDFs for downstream summarization and rewriting.
  freshman_course/
    index.html
    modules/
      Per-lecture freshman-oriented teaching documents.
  qwen_fp8_analysis/
    index.html
    analysis.md
```

## What Is Included

### Freshman Course

`freshman_course/index.html` is the interactive HTML classroom for first-year university students. It groups the LLM Systems lectures into beginner-friendly modules, with search, filtering, glossary, quiz, and links to the rewritten per-lecture documents.

`freshman_course/modules/` contains one HTML teaching document per lecture or topic. These pages explain the original slides in clearer language and link back to the corresponding PDF in `source_pdfs/`.

### Source PDFs

`source_pdfs/` contains the original downloaded PDFs, including course lectures and several supporting papers. These files are kept as source material and are linked from the rewritten teaching pages.

### Extracted Text

`extracted_text/` contains text extracted from the course PDFs. These files are intermediate source artifacts used to generate and verify rewritten teaching material.

### Qwen FP8 Kernel Analysis

`qwen_fp8_analysis/index.html` is an interactive visual report for Qwen3.5-122B-A10B kernel cost analysis under BF16 vs FP8, including:

- R1-R8 profile bucket mapping.
- Transformer block, hybrid attention, and MoE feed-forward structure.
- TP8 communication and NCCL/all_reduce cost mapping.
- FP8 quant/dequant overhead placement.
- MoE expert GEMM vs dense GEMM distinction.
- TP8 layer-to-layer transfer and single-node 8-GPU expert placement animation.

`qwen_fp8_analysis/analysis.md` contains the written explanation behind the visualization.

## Path Conventions

- HTML files are self-contained and can be opened directly in a browser.
- Freshman module pages link to PDFs via `../../source_pdfs/...`.
- Freshman course index links to modules via `modules/...`.
- Qwen analysis files are isolated under `qwen_fp8_analysis/`.

## Maintenance Notes

- Keep original PDFs in `source_pdfs/`; do not mix generated reports into that directory.
- Put new beginner-facing lecture pages under `freshman_course/modules/`.
- Put Qwen/kernel/performance analysis artifacts under `qwen_fp8_analysis/`.
- If moving files, run an HTML link check or at least search for stale paths such as `freshman_docs/` and root-level `llmsys-*.pdf`.
