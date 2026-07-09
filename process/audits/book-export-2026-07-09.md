# Book Export Audit

Date: 2026-07-09

The repository now includes a generated HTML export of the assembled book:

- `book/export/book-source.md` is the combined source used for export.
- `book/export/book.html` is the standalone HTML artifact.
- `book/export/book.css` provides the export stylesheet.

The export is assembled with `pandoc`, includes the generated references and index, and rewrites relative links so the file works from the repository tree.

Validation:

- `python3 scripts/generate_book_export.py --check`
- `python3 scripts/validate_book_workspace.py`

Visual QA:

- Quick Look thumbnail rendered successfully for `book/export/book.html`.

Owner: Publishing Pipeline  
Purpose: Record the first book-level HTML export artifact and its validation status  
Evidence grade: A  
Assumptions: HTML is the current publication target until a print backend is introduced  
Open questions: Whether to add a PDF/print export once a TeX or browser-to-PDF backend is available  
Handoff: Principal author / layout editor
