#!/usr/bin/env python3
"""Insert figure image references into chapter drafts from figure-placement.yml."""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
PLACEMENT = ROOT / "book/figures/figure-placement.yml"
CHAPTERS = ROOT / "book/chapters"
ARTWORK = ROOT / "book/figures/artwork"
CAPTION_GLOB = "part-*-figure-caption-review*.md"
CAPTION_DIR = ROOT / "process/review-memos"

FIGURE_BLOCK_RE = re.compile(
    r"\n*!\[([^\]]*)\]\(\.\./figures/artwork/ch\d+/([^)]+\.svg)\)\s*\n*",
    re.MULTILINE,
)
H2_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


def fail(msg: str) -> str:
    return f"FAIL: {msg}"


def load_captions() -> dict[str, str]:
    caps: dict[str, str] = {}
    for path in sorted(CAPTION_DIR.glob(CAPTION_GLOB)):
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(
            r"### `(fig-[^`]+)`\s*\n\s*\nCaption:\s*(.+?)(?:\n\n|\nSource)",
            text,
            re.S,
        ):
            caps[match.group(1)] = re.sub(r"\s+", " ", match.group(2).strip())
    return caps


def chapter_number(filename: str) -> int:
    return int(filename[:2])


def figure_path(fig_id: str, chapter_num: int) -> pathlib.Path:
    return ARTWORK / f"ch{chapter_num:02d}" / f"{fig_id}.svg"


def relative_from_chapter(fig_id: str, chapter_num: int) -> str:
    return f"../figures/artwork/ch{chapter_num:02d}/{fig_id}.svg"


def figure_block(fig_id: str, caption: str, chapter_num: int) -> str:
    path = relative_from_chapter(fig_id, chapter_num)
    # Pandoc treats a paragraph that is only an image as a figure; alt text becomes caption.
    return f"\n![{caption}]({path})\n"


def strip_existing_figures(text: str) -> str:
    """Remove previously inserted figure image blocks so re-runs are idempotent."""
    return FIGURE_BLOCK_RE.sub("\n\n", text)


def find_heading_span(text: str, heading: str) -> tuple[int, int] | None:
    """Return (start, end) of the first paragraph block after the given H2."""
    matches = list(H2_RE.finditer(text))
    for idx, match in enumerate(matches):
        if match.group(1).strip() != heading:
            continue
        section_start = match.end()
        section_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        section = text[section_start:section_end]
        # Skip blank lines after heading, then take first paragraph.
        pos = 0
        while pos < len(section) and section[pos] == "\n":
            pos += 1
        if pos >= len(section):
            insert_at = section_start + pos
            return insert_at, insert_at
        # First paragraph ends at double newline or next heading-like content.
        rest = section[pos:]
        para_end = rest.find("\n\n")
        if para_end == -1:
            insert_at = section_start + pos + len(rest.rstrip("\n"))
        else:
            insert_at = section_start + pos + para_end
        return insert_at, insert_at
    return None


def insert_figures_into_chapter(
    text: str,
    placements: list[dict],
    captions: dict[str, str],
    chapter_num: int,
    chapter_name: str,
) -> tuple[str, list[str]]:
    errors: list[str] = []
    text = strip_existing_figures(text)
    # Normalize triple+ newlines left by stripping.
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Insert from the bottom so earlier offsets stay valid.
    items: list[tuple[int, str, str]] = []
    for entry in placements:
        fig_id = entry["id"]
        heading = entry.get("after_heading")
        if fig_id not in captions:
            errors.append(fail(f"{chapter_name}: missing caption for {fig_id}"))
            continue
        art = figure_path(fig_id, chapter_num)
        if not art.exists():
            errors.append(fail(f"{chapter_name}: missing artwork {art.relative_to(ROOT)}"))
            continue
        if not heading:
            errors.append(fail(f"{chapter_name}: {fig_id} missing after_heading"))
            continue
        span = find_heading_span(text, heading)
        if span is None:
            errors.append(fail(f"{chapter_name}: heading not found for {fig_id}: {heading!r}"))
            continue
        block = figure_block(fig_id, captions[fig_id], chapter_num)
        items.append((span[0], fig_id, block))

    # Multiple figures for same heading: keep placement order by sorting
    # insert positions ascending then reverse apply with stable secondary key.
    # If same position, preserve original placement order via enumerate.
    ordered = sorted(enumerate(items), key=lambda t: (t[1][0], t[0]), reverse=True)
    for _, (pos, _fig_id, block) in ordered:
        # Ensure surrounding blank lines.
        before = text[:pos].rstrip("\n")
        after = text[pos:].lstrip("\n")
        text = before + "\n" + block + "\n" + after

    text = re.sub(r"\n{3,}", "\n\n", text)
    if not text.endswith("\n"):
        text += "\n"
    return text, errors


def load_placement() -> dict:
    """Parse the small placement YAML subset without requiring PyYAML."""
    text = PLACEMENT.read_text(encoding="utf-8")
    chapters: dict[str, list[dict]] = {}
    current_chapter: str | None = None
    current_entry: dict | None = None
    in_chapters = False

    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if line.strip() == "chapters:":
            in_chapters = True
            continue
        if not in_chapters:
            continue
        # chapter key: two-space indent + quoted filename
        m_ch = re.match(r'^  "([^"]+\.md)":\s*$', line)
        if m_ch:
            if current_entry and current_chapter:
                chapters[current_chapter].append(current_entry)
                current_entry = None
            current_chapter = m_ch.group(1)
            chapters[current_chapter] = []
            continue
        m_id = re.match(r"^\s+-\s+id:\s+(\S+)\s*$", line)
        if m_id:
            if current_entry and current_chapter:
                chapters[current_chapter].append(current_entry)
            current_entry = {"id": m_id.group(1)}
            continue
        m_head = re.match(r'^\s+after_heading:\s+"(.*)"\s*$', line)
        if m_head:
            if current_entry is None:
                raise RuntimeError(f"after_heading without id in {PLACEMENT}")
            current_entry["after_heading"] = m_head.group(1)
            continue
        if line.strip().startswith("after_heading:"):
            raise RuntimeError(f"unquoted after_heading not supported: {line!r}")

    if current_entry and current_chapter:
        chapters[current_chapter].append(current_entry)

    if not chapters:
        raise RuntimeError(f"invalid placement file: {PLACEMENT}")
    return chapters


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate without writing")
    parser.add_argument(
        "--chapter",
        action="append",
        default=[],
        help="Limit to chapter filename(s), e.g. 01-why-llm-systems.md",
    )
    args = parser.parse_args()

    try:
        chapters = load_placement()
    except Exception as exc:
        print(fail(str(exc)), file=sys.stderr)
        return 1

    captions = load_captions()
    if len(captions) < 1:
        print(fail("no captions loaded from review memos"), file=sys.stderr)
        return 1

    errors: list[str] = []
    changed = 0
    selected = set(args.chapter) if args.chapter else set(chapters)

    for chapter_name, placements in chapters.items():
        if chapter_name not in selected:
            continue
        path = CHAPTERS / chapter_name
        if not path.exists():
            errors.append(fail(f"missing chapter file {chapter_name}"))
            continue
        chapter_num = chapter_number(chapter_name)
        original = path.read_text(encoding="utf-8")
        updated, local_errors = insert_figures_into_chapter(
            original, placements, captions, chapter_num, chapter_name
        )
        errors.extend(local_errors)
        if local_errors:
            continue
        # Verify every figure id appears once.
        for entry in placements:
            fig_id = entry["id"]
            count = updated.count(f"{fig_id}.svg")
            if count != 1:
                errors.append(
                    fail(f"{chapter_name}: expected one embed of {fig_id}, found {count}")
                )
        if errors:
            continue
        if updated != original:
            changed += 1
            if not args.check:
                path.write_text(updated, encoding="utf-8")

    if errors:
        for err in errors:
            print(err, file=sys.stderr)
        return 1

    if args.check:
        print(f"PASS: figure placement valid ({len(selected)} chapters, {sum(len(chapters[c]) for c in selected if c in chapters)} figures)")
    else:
        print(f"WROTE: figure embeds into {changed} chapter file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
