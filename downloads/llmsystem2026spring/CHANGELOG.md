# Changelog

## 2026-06-25

### Added

- Added project README with entry points, directory layout, artifact descriptions, and maintenance notes.
- Added this changelog.

### Changed

- Reorganized generated and source materials into a clearer directory structure:
  - Moved original PDFs into `source_pdfs/`.
  - Moved freshman interactive course entry to `freshman_course/index.html`.
  - Moved per-lecture freshman documents to `freshman_course/modules/`.
  - Moved Qwen3.5-122B-A10B FP8 analysis artifacts to `qwen_fp8_analysis/`.
- Updated freshman course links after the directory migration:
  - Module links now use `modules/...`.
  - Module pages now return to `../index.html`.
  - Original PDF links now point to `source_pdfs/`.

### Existing Content Preserved

- Original downloaded PDFs are preserved under `source_pdfs/`.
- Extracted text remains under `extracted_text/`.
- The Qwen3.5-122B-A10B FP8 visualization and written analysis are preserved under `qwen_fp8_analysis/`.

### Notes

- The old `freshman_docs/` directory was removed after its files were moved to `freshman_course/modules/`.
