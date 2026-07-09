# Index Generation Audit

Date: 2026-07-09

The book now has a generated `book/INDEX.md` derived from two sources:

- chapter map entries from the current manuscript files;
- core terms from `book/registries/terminology.yml`;
- topic headings from the final chapter markdown files.

The index is intentionally chapter-linked rather than page-numbered. That keeps it valid before final pagination/export decisions are made.

Validation:

- `python3 scripts/generate_index.py --check`
- `python3 scripts/validate_book_workspace.py`

No unresolved structural issues were found during generation.

Owner: Publishing Pipeline  
Purpose: Record the first generated index artifact and its validation status  
Evidence grade: A  
Assumptions: Chapter filenames remain stable until the final export pass  
Open questions: Whether the print edition needs a page-numbered index after pagination exists  
Handoff: Principal author / layout editor
