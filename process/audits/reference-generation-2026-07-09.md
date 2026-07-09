# Reference Generation Audit

Date: 2026-07-09

## Result

The reference-generation pipeline resolves every chapter `[CITE: source-card-id]` marker to a source-ledger card and generates `book/REFERENCES.md`.

Current inventory:

- 256 inline citation markers;
- 219 unique cited source cards;
- 63 deduplicated cited works;
- 10 source cards not currently cited;
- zero missing cited cards.

Deduplication uses normalized title, author/issuer, date, and source location. Claim-level cards remain visible under each bibliography entry so provenance is not lost.

## Verification

- `python3 scripts/generate_references.py --check`: pass
- `python3 scripts/validate_book_workspace.py`: pass
- Regeneration is deterministic and produces no diff when source cards and chapter citations are unchanged.

Owner: Reference Pipeline  
Purpose: Reference-generation implementation and audit  
Evidence grade: Inherited from source cards  
Assumptions: Matching normalized work metadata identifies duplicate cards for one source  
Open questions: Final publisher citation style and incomplete lecture-PDF publication metadata  
Handoff: Principal author / copyeditor for final copyedit
