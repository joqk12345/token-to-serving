#!/usr/bin/env python3
"""Validate the minimum structure of the LLM Systems book workspace."""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def fail(message: str) -> str:
    return f"FAIL: {message}"


def check_required_files() -> list[str]:
    required = [
        "AGENTS.md",
        "CLAUDE.md",
        "GEMINI.md",
        "book/STATUS.md",
        "book/spine.yml",
        ".claude/docs/book-production-workflow.md",
        ".claude/docs/chapter-template.md",
        ".claude/docs/source-ledger-template.md",
        ".claude/rules/01-style-guide.md",
        ".claude/rules/02-technical-evidence-grades.md",
        ".claude/rules/03-overclaim-language.md",
        ".claude/rules/04-citation-form.md",
    ]
    errors: list[str] = []
    for rel in required:
        if not (ROOT / rel).exists():
            errors.append(fail(f"missing {rel}"))
    return errors


def check_import_files() -> list[str]:
    errors: list[str] = []
    for rel in ["CLAUDE.md", "GEMINI.md"]:
        path = ROOT / rel
        if path.exists() and path.read_text(encoding="utf-8").strip() != "@AGENTS.md":
            errors.append(fail(f"{rel} must contain only @AGENTS.md"))
    return errors


def check_source_cards() -> list[str]:
    errors: list[str] = []
    card_dir = ROOT / "book/evidence/source-ledger"
    for path in sorted(card_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for field in ["Source ID:", "Title:", "Evidence grade:", "Claim supported:"]:
            if field not in text:
                errors.append(fail(f"{path.relative_to(ROOT)} missing {field}"))
        match = re.search(r"^Evidence grade:\s*([A-D])\s*$", text, re.MULTILINE)
        if not match:
            errors.append(fail(f"{path.relative_to(ROOT)} has invalid evidence grade"))
    return errors


def check_status_chapter_count() -> list[str]:
    status = ROOT / "book/STATUS.md"
    spine = ROOT / "book/spine.yml"
    if not status.exists() or not spine.exists():
        return []

    status_text = status.read_text(encoding="utf-8")
    spine_text = spine.read_text(encoding="utf-8")
    status_rows = len(re.findall(r"^\|\s*\d+\s*\|", status_text, re.MULTILINE))
    spine_chapters = len(re.findall(r"^\s*-\s+number:\s+\d+\s*$", spine_text, re.MULTILINE))
    if status_rows != spine_chapters:
        return [fail(f"STATUS chapter rows ({status_rows}) != spine chapters ({spine_chapters})")]
    return []


def check_generated_references() -> list[str]:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/generate_references.py"), "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        return [fail(f"generated references invalid: {detail}")]
    return []


def check_generated_index() -> list[str]:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/generate_index.py"), "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        return [fail(f"generated index invalid: {detail}")]
    return []


def check_generated_book_export() -> list[str]:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/generate_book_export.py"), "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        return [fail(f"book export invalid: {detail}")]
    return []


def main() -> int:
    errors: list[str] = []
    errors.extend(check_required_files())
    errors.extend(check_import_files())
    errors.extend(check_source_cards())
    errors.extend(check_status_chapter_count())
    errors.extend(check_generated_references())
    errors.extend(check_generated_index())
    errors.extend(check_generated_book_export())

    if errors:
        print("\n".join(errors))
        return 1
    print("PASS: book workspace structure is valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
