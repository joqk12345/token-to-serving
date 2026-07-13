#!/usr/bin/env python3
"""Generate an EPUB 3 book from the chapters listed in book/spine.yml.

Examples:
    python3 scripts/generate_epub.py
    python3 scripts/generate_epub.py --output /tmp/llm-systems.epub
    python3 scripts/generate_epub.py --author "Author Name"
"""

from __future__ import annotations

import argparse
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
import zipfile


ROOT = pathlib.Path(__file__).resolve().parents[1]
BOOK = ROOT / "book"
SPINE = BOOK / "spine.yml"
CHAPTERS = BOOK / "chapters"
EXPORT = BOOK / "export"
DEFAULT_OUTPUT = EXPORT / "book.epub"
DEFAULT_COVER = BOOK / "figures/artwork/book-cover.svg"

H1_RE = re.compile(r"^#\s+(.+?)\s*$")
ID_WORD_RE = re.compile(r"[^a-z0-9]+")
CHAPTER_LINK_RE = re.compile(r"\((?:\.\./)?chapters/([a-z0-9-]+)\.md(?:#[^)]+)?\)")


def fail(message: str) -> str:
    return f"FAIL: {message}"


def unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def load_spine() -> tuple[str, list[tuple[str, list[tuple[int, str, str]]]]]:
    """Read the fixed subset of YAML used by the canonical book spine."""
    if not SPINE.exists():
        raise RuntimeError("book/spine.yml is missing")

    title = ""
    parts: list[tuple[str, list[tuple[int, str, str]]]] = []
    part_title: str | None = None
    chapters: list[tuple[int, str, str]] = []
    fields: dict[str, str] = {}

    def finish_chapter() -> None:
        nonlocal fields
        if not fields:
            return
        missing = {"number", "slug", "title"} - fields.keys()
        if missing:
            raise RuntimeError(f"incomplete spine chapter; missing {', '.join(sorted(missing))}")
        chapters.append((int(fields["number"]), fields["slug"], fields["title"]))
        fields = {}

    def finish_part() -> None:
        nonlocal chapters
        finish_chapter()
        if part_title is not None:
            parts.append((part_title, chapters))
        chapters = []

    for raw in SPINE.read_text(encoding="utf-8").splitlines():
        if not title and (match := re.match(r"^title:\s*(.+?)\s*$", raw)):
            title = unquote(match.group(1))
        elif match := re.match(r"^  - title:\s*(.+?)\s*$", raw):
            finish_part()
            part_title = unquote(match.group(1))
        elif match := re.match(r"^      - number:\s*(\d+)\s*$", raw):
            finish_chapter()
            fields["number"] = match.group(1)
        elif match := re.match(r"^        (slug|title):\s*(.+?)\s*$", raw):
            fields[match.group(1)] = unquote(match.group(2))

    finish_part()
    if not title or not parts or not any(chapters for _, chapters in parts):
        raise RuntimeError("book/spine.yml has no title or chapters")
    return title, parts


def strip_front_matter(text: str) -> str:
    if not text.startswith("---\n"):
        return text.lstrip()
    sections = text.split("---\n", 2)
    return sections[2].lstrip() if len(sections) == 3 else text.lstrip()


def pandoc_identifier(text: str) -> str:
    return ID_WORD_RE.sub("-", text.lower()).strip("-") or "section"


def prepare_chapter(text: str, number: int, title: str, slug: str) -> str:
    """Place chapters below parts and shift the chapter's section headings."""
    result: list[str] = []
    title_seen = False
    chapter_heading = f"Chapter {number}: {title}"
    for line in strip_front_matter(text).splitlines():
        if not title_seen and H1_RE.match(line):
            result.append(f"## {chapter_heading} {{#{pandoc_identifier(chapter_heading)} .chapter-title}}")
            title_seen = True
        elif line.startswith("## "):
            result.append("#" + line)
        elif line.startswith("### "):
            result.append("#" + line)
        else:
            result.append(line)
    if not title_seen:
        raise RuntimeError(f"book/chapters/{slug}.md has no level-one title")
    return "\n".join(result).rstrip()


def build_source(
    title: str,
    parts: list[tuple[str, list[tuple[int, str, str]]]],
    language: str,
) -> tuple[str, int]:
    identifier = f"urn:uuid:{uuid.uuid5(uuid.NAMESPACE_URL, title)}"
    lines = ["---", f'title: "{title}"', f'lang: "{language}"', f'identifier: "{identifier}"', "---", ""]
    chapter_ids: dict[str, str] = {}
    chapter_count = 0

    for part_title, chapters in parts:
        lines.extend([f"# {part_title} {{#{pandoc_identifier(part_title)} .part-title}}", ""])
        for number, slug, chapter_title in chapters:
            path = CHAPTERS / f"{slug}.md"
            if not path.exists():
                raise RuntimeError(f"spine chapter is missing: {path.relative_to(ROOT)}")
            heading = f"Chapter {number}: {chapter_title}"
            chapter_ids[slug] = pandoc_identifier(heading)
            lines.extend([prepare_chapter(path.read_text(encoding="utf-8"), number, chapter_title, slug), ""])
            chapter_count += 1

    references = BOOK / "REFERENCES.md"
    if references.exists():
        lines.extend([references.read_text(encoding="utf-8").rstrip(), ""])

    index = BOOK / "INDEX.md"
    if index.exists():
        index_text = index.read_text(encoding="utf-8")

        def replace_index_link(match: re.Match[str]) -> str:
            slug = match.group(1)
            if slug not in chapter_ids:
                raise RuntimeError(f"book/INDEX.md links to a chapter outside the spine: {slug}")
            return f"(#{chapter_ids[slug]})"

        lines.extend([CHAPTER_LINK_RE.sub(replace_index_link, index_text).rstrip(), ""])
    return "\n".join(lines).rstrip() + "\n", chapter_count


def epub_css() -> str:
    return """
body { color: #18212f; font-family: Georgia, "Iowan Old Style", serif; line-height: 1.65; margin: 5%; }
h1, h2, h3, h4 { line-height: 1.2; }
h1.part-title { color: #173e76; text-align: center; }
h2.chapter-title { color: #173e76; page-break-before: always; }
p, li { orphans: 2; widows: 2; }
a { color: #173e76; }
img { display: block; height: auto; margin: 1.25em auto; max-width: 100%; }
figure { break-inside: avoid; margin: 1.5em 0; }
figcaption { color: #566173; font-size: 0.88em; line-height: 1.4; }
pre { background: #f6f8fb; border: 1px solid #d9e0ea; font-size: 0.82em; overflow-wrap: break-word; padding: 0.75em; white-space: pre-wrap; }
code { font-family: ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace; }
table { border-collapse: collapse; font-size: 0.88em; width: 100%; }
th, td { border: 1px solid #d9e0ea; padding: 0.35em; vertical-align: top; }
th { background: #e8eef8; }
blockquote { border-left: 0.25em solid #173e76; margin-left: 0; padding-left: 1em; }
""".strip() + "\n"


def render_epub(source: pathlib.Path, css: pathlib.Path, output: pathlib.Path, author: str | None, cover: pathlib.Path | None) -> None:
    pandoc = shutil.which("pandoc")
    if not pandoc:
        raise RuntimeError("pandoc is unavailable; install Pandoc and run this script again")
    command = [
        pandoc,
        source.name,
        "--from=markdown+yaml_metadata_block+fenced_code_blocks+pipe_tables+smart",
        "--to=epub3",
        "--toc",
        "--toc-depth=2",
        "--split-level=2",
        "--resource-path",
        f"{source.parent}:{CHAPTERS}:{BOOK}",
        "--css",
        css.name,
        "--output",
        str(output),
    ]
    if author:
        command.extend(["--metadata", f"author={author}"])
    if cover:
        command.extend(["--epub-cover-image", str(cover)])

    output.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(command, cwd=source.parent, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout).strip() or "Pandoc EPUB generation failed")


def validate_epub(path: pathlib.Path, expected_chapters: int) -> None:
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError("Pandoc did not create a non-empty EPUB")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if "mimetype" not in names or "META-INF/container.xml" not in names:
            raise RuntimeError("generated file is missing required EPUB entries")
        if archive.read("mimetype") != b"application/epub+zip":
            raise RuntimeError("generated file has an invalid EPUB mimetype")
        content_documents = [name for name in names if re.search(r"/ch\d+\.xhtml$", name)]
        if len(content_documents) < expected_chapters:
            raise RuntimeError(f"EPUB has {len(content_documents)} content documents; expected at least {expected_chapters}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the book as EPUB 3.")
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT, help="output path (default: book/export/book.epub)")
    parser.add_argument("--author", help="optional author metadata")
    parser.add_argument("--language", default="en", help="language metadata (default: en)")
    parser.add_argument("--no-cover", action="store_true", help="omit the repository cover image")
    args = parser.parse_args()

    try:
        title, parts = load_spine()
        source_text, chapter_count = build_source(title, parts, args.language)
        cover = None if args.no_cover else DEFAULT_COVER
        if cover is not None and not cover.exists():
            raise RuntimeError("book cover is missing; restore it or use --no-cover")

        EXPORT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".epub-build-", dir=EXPORT) as temporary:
            workdir = pathlib.Path(temporary)
            source = workdir / "book-source.md"
            css = workdir / "epub.css"
            source.write_text(source_text, encoding="utf-8")
            css.write_text(epub_css(), encoding="utf-8")
            output = args.output.expanduser().resolve()
            render_epub(source, css, output, args.author, cover)
        validate_epub(output, chapter_count)
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        print(fail(str(exc)), file=sys.stderr)
        return 1

    print(f"WROTE: {output} ({chapter_count} chapters)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
