"""Read and normalize source PDF data with full page traceability and error logging."""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from .models import DocumentPage, PageLog

# Running headers/footers observed in World Bank Taking Stock PDFs
_HEADER_PATTERNS = [
    re.compile(r"^\d+\s*$"),
    re.compile(r"^TAKING\s+STOCK\s+(AUGUST|MARCH|APRIL|SEPTEMBER)\s+\d{4}\s*$", re.IGNORECASE),
    re.compile(r"^REACHING\s+NEW\s+HEIGHTS\s+IN\s+CAPITAL\s+MARKETS\s*$", re.IGNORECASE),
    re.compile(
        r"^ELECTRIFYING\s+JOURNEYS:\s*E-MOBILITY\s+TRANSITION\s+IN\s+VIET\s+NAM\s*I\s*([ivxlcdm]+|\d+)?\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^([ivxlcdm]+|\d+)\s*I\s*TAKING\s+STOCK\s+(MARCH|AUGUST|APRIL|SEPTEMBER)\s+\d{4}\s*$",
        re.IGNORECASE,
    ),
]

_HEADING_PREFIX = re.compile(r"^(Figure|Table|Box|Chapter|\d+(\.\d+)*\.?)\s+[A-Z0-9]")
_BULLET_PREFIX = re.compile(r"^([•\-\*]|\(\w+\))\s+")


def _is_running_header(line: str) -> bool:
    return any(pat.match(line) for pat in _HEADER_PATTERNS)


_CONTINUATION_TAIL_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "into", "is", "of", "on", "or", "that", "the", "to", "was", "were",
    "which", "with",
}


def _is_heading_line(line: str) -> bool:
    if len(line) > 85 or len(line) < 4:
        return False
    if line.endswith((",", ";", "(", "-")):
        return False
    if _HEADING_PREFIX.match(line):
        return True
    last_word = line.rstrip(".").split()[-1].lower()
    if last_word in _CONTINUATION_TAIL_WORDS:
        return False
    # Title-like line without trailing period and starting with uppercase
    return line[0].isupper() and not line.endswith(".")


def normalize_text(text: str) -> str:
    """Normalize Unicode and reconstruct paragraph breaks from PDF soft wraps."""
    if not text:
        return ""

    # Unicode NFC and clean invisible/non-standard whitespace & punctuation
    cleaned = unicodedata.normalize("NFC", text)
    cleaned = cleaned.replace("\xa0", " ").replace("\u200b", "").replace("\ufeff", "")
    cleaned = cleaned.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    cleaned = cleaned.replace("–", "-").replace("—", "-")
    # Remove ASCII control characters other than newline/tab
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", cleaned)

    raw_lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in cleaned.splitlines()]
    lines = [ln for ln in raw_lines if ln and not _is_running_header(ln)]
    if not lines:
        return ""

    paragraphs: list[str] = []
    current_parts: list[str] = []

    for idx, line in enumerate(lines):
        if not current_parts:
            current_parts.append(line)
            continue

        prev = lines[idx - 1]
        # Start a new paragraph when the previous line clearly ended a paragraph or standalone heading
        prev_ended_sentence = prev.endswith((".", "!", "?", ":")) and len(prev) < 88
        prev_is_standalone_heading = _is_heading_line(prev) and not line[0].islower()
        is_break = (
            prev_ended_sentence
            or prev_is_standalone_heading
            or _HEADING_PREFIX.match(line) is not None
            or _BULLET_PREFIX.match(line) is not None
        )

        if is_break:
            paragraphs.append(" ".join(current_parts))
            current_parts = [line]
        else:
            # Join hyphenated split only if both sides are lowercase letters
            if current_parts[-1].endswith("-") and len(current_parts[-1]) >= 2:
                prev_char = current_parts[-1][-2]
                if prev_char.islower() and line[0].islower():
                    current_parts[-1] = current_parts[-1][:-1] + line
                    continue
            current_parts.append(line)

    if current_parts:
        paragraphs.append(" ".join(current_parts))

    return "\n\n".join(p.strip() for p in paragraphs if p.strip())


def resolve_section(page_num: int, section_ranges: list[dict[str, Any]], default_section: str) -> str:
    """Resolve section name from documented physical page ranges."""
    for item in section_ranges:
        if item["start_page"] <= page_num <= item["end_page"]:
            return str(item["section"])
    return default_section


def load_pdf(
    path: Path,
    document_id: str,
    source: str,
    base_metadata: dict[str, Any] | None = None,
    excluded_pages: dict[int, str] | None = None,
    section_ranges: list[dict[str, Any]] | None = None,
    min_content_chars: int = 150,
) -> tuple[list[DocumentPage], list[PageLog]]:
    """Extract physical pages from PDF, keep source/page traceability, and log skipped/empty pages."""
    reader = PdfReader(str(path))
    pages: list[DocumentPage] = []
    logs: list[PageLog] = []
    excluded = excluded_pages or {}
    ranges = section_ranges or []
    meta_template = dict(base_metadata or {})

    for idx, raw_page in enumerate(reader.pages, start=1):
        try:
            raw_text = raw_page.extract_text() or ""
        except Exception as exc:  # pragma: no cover - defensive PDF error capture
            logs.append(
                PageLog(
                    document_id=document_id,
                    page=idx,
                    raw_chars=0,
                    normalized_chars=0,
                    status="error",
                    reason=f"extract_text failed: {exc}",
                )
            )
            continue

        norm_text = normalize_text(raw_text)
        raw_len = len(raw_text.strip())
        norm_len = len(norm_text)

        if idx in excluded:
            logs.append(
                PageLog(
                    document_id=document_id,
                    page=idx,
                    raw_chars=raw_len,
                    normalized_chars=norm_len,
                    status="excluded_front_back",
                    reason=excluded[idx],
                )
            )
            continue

        if norm_len < min_content_chars:
            logs.append(
                PageLog(
                    document_id=document_id,
                    page=idx,
                    raw_chars=raw_len,
                    normalized_chars=norm_len,
                    status="empty_or_divider",
                    reason=f"normalized_chars={norm_len} < {min_content_chars}",
                )
            )
            continue

        page_meta = dict(meta_template)
        page_meta["section"] = resolve_section(
            idx, ranges, default_section=str(meta_template.get("section", "general"))
        )
        page_meta["document_id"] = document_id
        page_meta["source"] = source
        page_meta["page"] = idx

        pages.append(
            DocumentPage(
                document_id=document_id,
                source=source,
                page=idx,
                text=norm_text,
                metadata=page_meta,
            )
        )
        logs.append(
            PageLog(
                document_id=document_id,
                page=idx,
                raw_chars=raw_len,
                normalized_chars=norm_len,
                status="included",
                reason="substantive content page",
            )
        )

    return pages, logs
