# LLM Systems Book Workspace

This workspace turns the CMU Large Language Model Systems Spring 2026 material in `downloads/llmsystem2026spring/source_pdfs/` into a technical nonfiction book.

Start here:

- `AGENTS.md` — project constitution and writing rules.
- `book/spine.yml` — proposed book structure.
- `book/STATUS.md` — current production state.
- `.claude/docs/book-production-workflow.md` — per-chapter pipeline.
- `.claude/docs/chapter-template.md` — chapter brief/draft template.
- `.claude/docs/source-ledger-template.md` — source card template.

The core rule is simple: do not rewrite slides into chapters. Start from a system problem, use the PDFs and papers as evidence, and make the mechanism clear enough that a reader can reason about tradeoffs.

## Verify

```bash
python3 scripts/validate_book_workspace.py
python3 scripts/scan_technical_overclaim.py
python3 -m unittest discover -s tests
```

`pytest` is optional; the baseline tests use only the Python standard library.
