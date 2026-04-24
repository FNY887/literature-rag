from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from agentic_rag.core.models import ChunkRecord
from agentic_rag.core.utils import clean_title_text, normalize_for_search, normalize_title, normalize_whitespace


STOPWORDS = {
    "about", "after", "against", "among", "because", "between", "could", "first",
    "found", "from", "have", "into", "journal", "paper", "results", "study",
    "their", "these", "using", "with",
}

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
IMAGE_RE = re.compile(r"^!\[[^\]]*\]\([^)]+\)\s*$")
EMAIL_RE = re.compile(r"[\w.\-+]+@[\w.\-]+\.\w+")
NUMBERED_HEADING_RE = re.compile(r"^\s*(?:\d+(?:\.\d+)*[.)]?\s+|[ivxlcdm]+\.?\s+)", re.IGNORECASE)

FRONT_MATTER_DROP_HEADINGS = {
    "article info",
    "articleinfo",
    "article history",
    "articlehistory",
    "keywords",
    "keyword",
    "graphical abstract",
    "graphicalabstract",
    "abbreviations",
    "just accepted",
    "accepted article",
    "accepted manuscript",
    "author s accepted manuscript",
    "authors accepted manuscript",
    "manuscript version accepted manuscript",
    "affiliations",
    "affiliation",
    "articles you may be interested in",
    "reuse",
    "takedown",
}
SUMMARY_HEADINGS = {
    "statement of significance",
    "statementofsignificance",
    "highlights",
    "highlight",
}
TAIL_STOP_HEADINGS = {
    "references",
    "bibliography",
    "acknowledgements",
    "acknowledgments",
    "funding",
    "funding statement",
    "fundingstatement",
    "author contributions",
    "authorcontributions",
    "credit authorship contribution statement",
    "creditauthorshipcontributionstatement",
    "conflict of interest",
    "conflicts of interest",
    "conflictsofinterest",
    "declaration of interest",
    "declaration of interests",
    "declaration of competing interest",
    "declaration of competing interests",
    "declarationofinterest",
    "declarationofinterests",
    "declarationofcompetinginterest",
    "declarationofcompetinginterests",
    "supplementary information",
    "supplementary material",
    "supplementary materials",
    "supplementaryinformation",
    "supplementarymaterial",
    "supplementarymaterials",
    "appendix",
}
COMMON_SECTION_HEADINGS = {
    "abstract",
    "introduction",
    "background",
    "methods",
    "materials and methods",
    "methods and materials",
    "experimental",
    "materials",
    "results",
    "results and discussion",
    "discussion",
    "conclusion",
    "conclusions",
    "outlook",
    "limitations",
    "statistical analysis",
}
INLINE_SUMMARY_PREFIXES = (
    "statement of significance:",
    "highlights:",
)
NON_TITLE_LEVEL_ONE_HEADINGS = {
    "research article",
    "article open",
    "communication",
    "full length article",
    "review article",
    "review",
    "original article",
    "original research",
    "research",
    "article",
    "perspective",
    "just accepted",
    "accepted article",
    "accepted manuscript",
    "author s accepted manuscript",
    "authors accepted manuscript",
    "manuscript version accepted manuscript",
    "short communication",
    "rapid communication",
    "letter",
    "letters",
    "reuse",
    "takedown",
    "method article",
    "corresponding author",
    "corresponding authors",
    "regular article",
}
NON_TITLE_PAGE_HEADINGS = {
    "view article online",
    "view journal",
    "view issue",
    "view journal view issue",
    "check for updates",
    "international edition",
    "view online",
    "export citation",
}
PUBLICATION_MASTHEAD_HEADINGS = {
    "nanoscale",
    "chemcomm",
    "crystengcomm",
    "journal of materials chemistry b",
    "soft matter",
    "a journal of the gesellschaft deutscher chemiker angewandte gdch international edition chemie",
}
TITLE_RESET_HEADINGS = {
    "article",
    "just accepted",
    "accepted article",
    "accepted manuscript",
    "author s accepted manuscript",
    "authors accepted manuscript",
    "manuscript version accepted manuscript",
    "reuse",
    "takedown",
}
CONFERENCE_CODE_RE = re.compile(r"^[A-Z]{1,4}\d{2,5}[A-Z]?$")


@dataclass(slots=True)
class RawBlock:
    kind: str
    text: str
    level: int | None = None


@dataclass(slots=True)
class TextUnit:
    text: str
    section_hint: str | None
    kind: str
    overlap_eligible: bool = False


@dataclass(slots=True)
class ChunkPlan:
    section_hint: str | None
    units: list[TextUnit]
    include_title: bool = False


@dataclass(slots=True)
class TitleCandidate:
    title: str
    block_index: int
    block_end_index: int
    score: int


@dataclass(slots=True)
class BodySection:
    heading_path: tuple[str, ...]
    parts: list[tuple[str, str]]


def _build_doc_id(title: str) -> str:
    normalized = normalize_title(title)
    if not normalized:
        raise ValueError("Document title must be a non-empty Markdown heading.")
    slug = re.sub(r"[^0-9a-zA-Z]+", "-", normalized).strip("-").lower()
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:8]
    return f"{slug[:48]}-{digest}" if slug else digest


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _extract_keywords(title: str, section_hint: str | None, text: str, limit: int = 12) -> str:
    candidates: list[str] = []
    for source in (title, section_hint or "", text[:600]):
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]{2,}", source):
            lowered = token.lower()
            if lowered in STOPWORDS:
                continue
            candidates.append(token)
    deduped: list[str] = []
    seen: set[str] = set()
    for token in candidates:
        lowered = token.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        deduped.append(token)
        if len(deduped) >= limit:
            break
    return ", ".join(deduped)


def _canonical_heading_key(text: str) -> str:
    cleaned = normalize_whitespace(text).strip(":")
    cleaned = NUMBERED_HEADING_RE.sub("", cleaned)
    tokens = re.findall(r"[A-Za-z0-9]+", cleaned)
    if tokens and all(token.isalpha() and len(token) == 1 for token in tokens):
        return "".join(tokens).lower()
    return " ".join(tokens).lower()


def _clean_heading(text: str) -> str:
    cleaned = clean_title_text(text).strip(":")
    if _canonical_heading_key(cleaned) == "abstract":
        return "Abstract"
    return cleaned


def _compact_title_key(text: str) -> str:
    return normalize_title(text).replace(" ", "")


def _format_heading_path(headings: tuple[str, ...] | list[str]) -> str | None:
    cleaned = [heading for heading in headings if heading]
    if not cleaned:
        return None
    return " > ".join(cleaned)


def _render_section_hint(section_hint: str | None) -> str | None:
    if not section_hint:
        return None
    parts = [part.strip() for part in section_hint.split(" > ") if part.strip()]
    if not parts:
        return None
    return "\n\n".join(parts)


def _parse_blocks(content: str) -> list[RawBlock]:
    blocks: list[RawBlock] = []
    paragraph_lines: list[str] = []

    def flush_paragraph() -> None:
        nonlocal paragraph_lines
        if not paragraph_lines:
            return
        text = normalize_whitespace(" ".join(paragraph_lines))
        if text:
            blocks.append(RawBlock(kind="paragraph", text=text))
        paragraph_lines = []

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            flush_paragraph()
            continue
        heading_match = HEADING_RE.match(line)
        if heading_match:
            flush_paragraph()
            blocks.append(
                RawBlock(
                    kind="heading",
                    text=normalize_whitespace(heading_match.group(2)),
                    level=len(heading_match.group(1)),
                )
            )
            continue
        if IMAGE_RE.match(line):
            flush_paragraph()
            blocks.append(RawBlock(kind="image", text=line))
            continue
        paragraph_lines.append(line)

    flush_paragraph()
    return blocks


def _extract_title(content: str) -> str:
    for block in _parse_blocks(content):
        if block.kind == "heading":
            return _clean_heading(block.text)
    return ""


def _looks_like_spaced_banner_heading(text: str) -> bool:
    normalized = normalize_whitespace(text)
    letters_only = re.sub(r"[^A-Za-z]+", "", normalized)
    if not letters_only or letters_only != letters_only.upper():
        return False
    letter_tokens = re.findall(r"[A-Z]", normalized)
    word_tokens = re.findall(r"[A-Z]+", normalized)
    return (
        len(letter_tokens) >= 4
        and len(word_tokens) >= 4
        and all(len(token) == 1 for token in word_tokens)
    )


def _looks_like_author_heading(text: str) -> bool:
    normalized = normalize_whitespace(text)
    normalized = re.sub(r"^\s*Q\d+\s+", "", normalized)
    return _looks_like_author_list(normalized)


def _is_title_reset_heading(text: str) -> bool:
    return _canonical_heading_key(text) in TITLE_RESET_HEADINGS


def _is_non_title_level_one_heading(text: str) -> bool:
    canonical = _canonical_heading_key(text)
    if (
        _is_tail_stop_heading(text)
        or _is_front_matter_drop_heading(text)
        or _is_summary_heading(text)
        or _is_formal_section_heading(text)
        or _looks_like_publication_masthead(text)
        or _looks_like_author_heading(text)
    ):
        return True
    if canonical in NON_TITLE_LEVEL_ONE_HEADINGS or canonical in NON_TITLE_PAGE_HEADINGS:
        return True
    if canonical.startswith("theme "):
        return True
    if CONFERENCE_CODE_RE.fullmatch(normalize_whitespace(text)):
        return True
    return _looks_like_spaced_banner_heading(text)


def _looks_like_publication_masthead(text: str) -> bool:
    normalized = normalize_whitespace(text)
    lowered = normalized.lower()
    canonical = _canonical_heading_key(normalized)
    if lowered.startswith("journal of "):
        return True
    if lowered.startswith("a journal of ") or "gesellschaft deutscher chemiker" in lowered:
        return True
    if "angewandte" in lowered and "chemie" in lowered and len(normalized) <= 120:
        return True
    if canonical in PUBLICATION_MASTHEAD_HEADINGS:
        return True
    words = re.findall(r"[A-Za-z0-9]+", normalized)
    if 1 <= len(words) <= 4 and normalized.istitle():
        joined = " ".join(word.lower() for word in words)
        if joined in {"nature", "science", "cell", "small"}:
            return True
    return False


def _looks_like_contextual_publication_masthead(
    raw_blocks: list[RawBlock],
    block_index: int,
    text: str,
) -> bool:
    if _looks_like_publication_masthead(text):
        return True

    normalized = normalize_whitespace(text)
    words = re.findall(r"[A-Za-z0-9]+", normalized)
    if not words or len(words) > 3 or len(normalized) > 20:
        return False

    metadata_signals = 0
    for look_ahead in raw_blocks[block_index + 1 : block_index + 8]:
        if look_ahead.kind == "heading":
            break
        if look_ahead.kind != "paragraph":
            continue
        paragraph = _strip_nonprose_markup(normalize_whitespace(look_ahead.text))
        if not paragraph:
            continue
        lowered = paragraph.lower()
        if _looks_like_metadata_paragraph(paragraph):
            metadata_signals += 1
        if (
            "accepted manuscript" in lowered
            or "accepted version" in lowered
            or "royal society of chemistry" in lowered
            or "can be cited before page numbers" in lowered
            or "information for authors" in lowered
            or "author guidelines" in lowered
            or "white rose research online" in lowered
        ):
            metadata_signals += 1
    return metadata_signals > 0


def _score_title_candidate(
    *,
    text: str,
    block_index: int,
    first_formal_section_index: int,
) -> int:
    normalized = normalize_whitespace(text)
    words = re.findall(r"[A-Za-z0-9]+(?:[-–][A-Za-z0-9]+)?", normalized)
    score = 0
    if first_formal_section_index < 0 or block_index < first_formal_section_index:
        score += 4
    else:
        score -= 6
    score += min(len(words), 8)
    score += min(len(normalized) // 30, 4)
    if re.search(r"[a-z]", normalized):
        score += 2
    if _looks_like_publication_masthead(normalized):
        score -= 8
    return score


def _title_tokens(text: str) -> list[str]:
    return [token for token in normalize_title(text).split() if token]


def _is_token_subsequence(shorter: list[str], longer: list[str]) -> bool:
    if not shorter:
        return False
    position = 0
    for token in longer:
        if token == shorter[position]:
            position += 1
            if position == len(shorter):
                return True
    return False


def _titles_look_like_variants(left: str, right: str) -> bool:
    left_tokens = _title_tokens(left)
    right_tokens = _title_tokens(right)
    if min(len(left_tokens), len(right_tokens)) < 6:
        return False
    shorter, longer = (left_tokens, right_tokens) if len(left_tokens) <= len(right_tokens) else (right_tokens, left_tokens)
    if shorter[:4] != longer[:4]:
        return False
    if shorter[-1] != longer[-1]:
        return False
    if len(longer) - len(shorter) > 4:
        return False
    return _is_token_subsequence(shorter, longer)


def _merge_title_candidate(
    candidates_by_title: dict[str, TitleCandidate],
    candidate: TitleCandidate,
    compact_heading: str,
    *,
    seen_body_prose_after_title: bool,
) -> None:
    existing = candidates_by_title.get(compact_heading)
    if existing is not None:
        if existing is None or (
            not seen_body_prose_after_title and candidate.block_end_index > existing.block_end_index
        ):
            candidates_by_title[compact_heading] = candidate
        return

    for existing_key, existing_candidate in list(candidates_by_title.items()):
        if not _titles_look_like_variants(existing_candidate.title, candidate.title):
            continue
        if candidate.score > existing_candidate.score:
            candidates_by_title.pop(existing_key)
            candidates_by_title[compact_heading] = candidate
            return
        if candidate.score == existing_candidate.score and len(normalize_title(candidate.title)) > len(normalize_title(existing_candidate.title)):
            candidates_by_title.pop(existing_key)
            candidates_by_title[compact_heading] = candidate
        return

    candidates_by_title[compact_heading] = candidate


def _collect_title_heading_group(
    raw_blocks: list[RawBlock],
    start_index: int,
    *,
    allow_split_titles: bool,
) -> tuple[str, int]:
    headings = [_clean_heading(raw_blocks[start_index].text)]
    end_index = start_index
    if not allow_split_titles:
        return headings[0], end_index

    for index in range(start_index + 1, len(raw_blocks)):
        block = raw_blocks[index]
        if block.kind != "heading" or block.level != 1:
            break
        heading = _clean_heading(block.text)
        if (
            _is_non_title_level_one_heading(heading)
            or _looks_like_contextual_publication_masthead(raw_blocks, index, heading)
        ):
            break
        if not normalize_title(heading):
            break
        headings.append(heading)
        end_index = index

    return normalize_whitespace(" ".join(headings)), end_index


def _select_document_title_block(raw_blocks: list[RawBlock]) -> TitleCandidate | None:
    title_reset_indices = [
        index
        for index, block in enumerate(raw_blocks)
        if block.kind == "heading"
        and block.level == 1
        and _is_title_reset_heading(_clean_heading(block.text))
    ]
    title_reset_index = title_reset_indices[-1] if title_reset_indices else -1
    scan_start = title_reset_index + 1 if title_reset_index >= 0 else 0
    allow_split_titles = title_reset_index >= 0
    first_formal_section_index = next(
        (
            index
            for index, block in enumerate(raw_blocks)
            if index >= scan_start
            and block.kind == "heading"
            and _is_formal_section_heading(_clean_heading(block.text))
        ),
        -1,
    )
    candidates_by_title: dict[str, TitleCandidate] = {}
    seen_title_candidate = False
    seen_body_prose_after_title = False
    drop_details_block = False
    index = scan_start
    while index < len(raw_blocks):
        block = raw_blocks[index]
        if block.kind == "paragraph" and seen_title_candidate:
            text = normalize_whitespace(block.text)
            lowered = text.lower()
            if "<details" in lowered and "</details>" not in lowered:
                drop_details_block = True
                index += 1
                continue
            if drop_details_block:
                if "</details>" in lowered:
                    drop_details_block = False
                index += 1
                continue
            text = _strip_nonprose_markup(text)
            if (
                text
                and not _looks_like_table_body(text)
                and not _looks_like_diagram_body(text)
                and not _is_front_matter_noise_paragraph(text)
                and _is_substantial_paragraph(text)
                and _looks_like_sentence_paragraph(text)
            ):
                seen_body_prose_after_title = True
            index += 1
            continue
        if block.kind != "heading" or block.level != 1:
            index += 1
            continue
        heading = _clean_heading(block.text)
        if (
            _is_non_title_level_one_heading(heading)
            or _looks_like_contextual_publication_masthead(raw_blocks, index, heading)
        ):
            index += 1
            continue
        grouped_title, block_end_index = _collect_title_heading_group(
            raw_blocks,
            index,
            allow_split_titles=allow_split_titles,
        )
        normalized_heading = normalize_title(grouped_title)
        compact_heading = _compact_title_key(grouped_title)
        if not normalized_heading:
            index = block_end_index + 1
            continue
        if seen_body_prose_after_title and compact_heading not in candidates_by_title:
            index = block_end_index + 1
            continue
        candidate = TitleCandidate(
            title=grouped_title,
            block_index=index,
            block_end_index=block_end_index,
            score=_score_title_candidate(
                text=grouped_title,
                block_index=index,
                first_formal_section_index=first_formal_section_index,
            ),
        )
        _merge_title_candidate(
            candidates_by_title,
            candidate,
            compact_heading,
            seen_body_prose_after_title=seen_body_prose_after_title,
        )
        seen_title_candidate = True
        index = block_end_index + 1

    candidates = list(candidates_by_title.values())
    viable = [candidate for candidate in candidates if candidate.score >= 5]
    if not viable:
        return None
    viable.sort(key=lambda candidate: (-candidate.score, candidate.block_index))
    best = viable[0]
    if len(viable) > 1 and best.score - viable[1].score < 2:
        raise ValueError(
            "Markdown paper contains multiple plausible level-1 title headings; "
            "clean the Markdown so only the paper title remains as a valid H1."
        )
    return best


def _extract_document_title(content: str) -> str:
    selected = _select_document_title_block(_parse_blocks(content))
    if selected is None:
        return ""
    return selected.title


def _parse_sections(content: str) -> list[tuple[str | None, str]]:
    sections: list[tuple[str | None, str]] = []
    current_heading: str | None = None
    body_parts: list[str] = []

    def flush_section() -> None:
        nonlocal current_heading, body_parts
        if body_parts:
            sections.append((current_heading, "\n\n".join(body_parts).strip()))
        current_heading = None
        body_parts = []

    for block in _parse_blocks(content):
        if block.kind == "heading":
            flush_section()
            current_heading = _clean_heading(block.text)
        elif block.kind == "paragraph":
            body_parts.append(block.text)

    flush_section()
    return sections


def _is_tail_stop_heading(text: str) -> bool:
    canonical = _canonical_heading_key(text)
    return canonical in TAIL_STOP_HEADINGS or canonical.startswith("received ")


def _is_front_matter_drop_heading(text: str) -> bool:
    return _canonical_heading_key(text) in FRONT_MATTER_DROP_HEADINGS


def _is_abstract_heading(text: str) -> bool:
    return _canonical_heading_key(text) == "abstract"


def _is_summary_heading(text: str) -> bool:
    return _canonical_heading_key(text) in SUMMARY_HEADINGS


def _is_formal_section_heading(text: str) -> bool:
    canonical = _canonical_heading_key(text)
    if _is_tail_stop_heading(text) or _is_front_matter_drop_heading(text):
        return False
    if canonical in COMMON_SECTION_HEADINGS or canonical in SUMMARY_HEADINGS or canonical == "abstract":
        return True
    return bool(NUMBERED_HEADING_RE.match(normalize_whitespace(text)))


def _is_page_or_panel_label(text: str) -> bool:
    normalized = normalize_whitespace(text)
    lowered = normalized.lower()
    if re.fullmatch(r"(?:page\s+)?\d+", lowered):
        return True
    if re.fullmatch(r"\(?[a-z]\)?", lowered):
        return True
    if re.fullmatch(r"\(?[a-z]\)?\s*(?:[.,]|-\s*\d+\s*hrs?)?", lowered):
        return True
    return False


def _looks_like_caption(text: str) -> bool:
    return bool(re.match(r"^(fig(?:ure)?|scheme|table)\b", normalize_whitespace(text).lower()))


def _looks_like_caption_continuation(text: str) -> bool:
    lowered = normalize_whitespace(text).lower()
    return (
        bool(re.match(r"^(?:\(?[a-z]\)?[,.:]|\([a-z]\)|left\b|right\b|top\b|bottom\b)", lowered))
        or "scale bar" in lowered
        or "scale bars" in lowered
        or "inset" in lowered
        or "arrowhead" in lowered
        or "arrowheads" in lowered
        or lowered.startswith("movie available")
    )


def _extract_inline_summary(text: str) -> str | None:
    lowered = text.lower()
    for prefix in INLINE_SUMMARY_PREFIXES:
        if lowered.startswith(prefix):
            return normalize_whitespace(text)
    return None


def _extract_inline_abstract(text: str) -> str | None:
    if re.match(r"^\s*abstract\s*[:：]", text, re.IGNORECASE):
        return normalize_whitespace(text)
    return None


def _is_substantial_paragraph(text: str) -> bool:
    words = re.findall(r"\b\w+\b", text)
    return len(words) >= 35 or len(text) >= 240


def _looks_like_sentence_paragraph(text: str) -> bool:
    return bool(re.search(r"[.!?;:]", text))


def _ends_with_sentence_boundary(text: str) -> bool:
    normalized = normalize_whitespace(text)
    return bool(re.search(r"[.!?;:](?:[\"')\]]|\s)*$", normalized))


def _split_overlong_sentence(text: str, target_size: int) -> list[str]:
    words = text.split()
    if len(words) <= 1:
        return [
            text[index : index + target_size].strip()
            for index in range(0, len(text), target_size)
            if text[index : index + target_size].strip()
        ]

    segments: list[str] = []
    buffer: list[str] = []
    buffer_len = 0
    for word in words:
        projected = buffer_len + len(word) + (1 if buffer else 0)
        if buffer and projected > target_size:
            segments.append(" ".join(buffer))
            buffer = [word]
            buffer_len = len(word)
        else:
            buffer.append(word)
            buffer_len = projected
    if buffer:
        segments.append(" ".join(buffer))
    return segments or [text]


def _looks_like_continuation_paragraph(text: str) -> bool:
    normalized = normalize_whitespace(text)
    if not normalized:
        return False
    if re.match(r"^[a-z]", normalized):
        return True
    if re.match(r"^[0-9(\[]", normalized):
        return True
    if re.match(r"^(?:Fig\.|Figure|Table|Eq\.|\$)", normalized):
        return True
    return True


def _should_merge_body_paragraphs(previous_text: str, current_text: str) -> bool:
    if not previous_text.strip() or not current_text.strip():
        return False
    if _ends_with_sentence_boundary(previous_text):
        return False
    return _looks_like_continuation_paragraph(current_text)


def _repair_body_paragraph_breaks(parts: list[tuple[str, str]]) -> list[tuple[str, str]]:
    repaired: list[tuple[str, str]] = []
    for kind, text in parts:
        if (
            repaired
            and kind == "body_paragraph"
            and repaired[-1][0] == "body_paragraph"
            and _should_merge_body_paragraphs(repaired[-1][1], text)
        ):
            previous_kind, previous_text = repaired[-1]
            repaired[-1] = (previous_kind, f"{previous_text} {text}".strip())
            continue
        repaired.append((kind, text))
    return repaired


def _looks_like_author_list(text: str) -> bool:
    if len(text) > 500 or ":" in text:
        return False
    lowered = text.lower()
    if any(keyword in lowered for keyword in ("university", "department", "institute", "school", "hospital")):
        return False
    normalized = re.sub(r"\s+and\s+", ", ", text)
    pieces = [piece.strip() for piece in re.split(r"[,*†‡§]+", normalized) if piece.strip()]
    has_honorific = any(token in lowered for token in ("prof.", "dr.", "mr.", "mrs.", "ms."))
    if len(pieces) < 3 and not has_honorific:
        return False
    name_like = 0
    for piece in pieces:
        words = re.findall(r"[A-Za-z][A-Za-z'\-]+", piece)
        if 1 <= len(words) <= 5 and sum(word[0].isupper() for word in words) >= max(1, len(words) - 1):
            name_like += 1
    threshold = 2 if has_honorific else max(3, len(pieces) // 2)
    return name_like >= threshold


def _looks_like_affiliation(text: str) -> bool:
    lowered = text.lower()
    affiliation_keywords = (
        "university",
        "department",
        "school",
        "institute",
        "laboratory",
        "faculty",
        "academy",
        "campus",
        "hospital",
        "college",
        "centre",
        "center",
    )
    country_keywords = (
        "germany",
        "france",
        "china",
        "israel",
        "sweden",
        "denmark",
        "japan",
        "italy",
        "spain",
        "usa",
        "united kingdom",
        "uk",
    )
    if EMAIL_RE.search(text):
        return True
    affiliation_matches = sum(keyword in lowered for keyword in affiliation_keywords)
    if affiliation_matches:
        return len(text) <= 500 or affiliation_matches >= 2
    if any(keyword in lowered for keyword in country_keywords) and len(text) <= 120:
        return True
    if re.search(r"\b\d{4,6}\b", text) and len(text) <= 120:
        return True
    return False


def _looks_like_metadata_paragraph(text: str) -> bool:
    lowered = text.lower()
    canonical = _canonical_heading_key(text)
    metadata_fragments = (
        "just accepted manuscript",
        "accepted article",
        "this is an accepted manuscript",
        "accepted manuscripts are published online shortly after acceptance",
        "you can find more information about accepted manuscripts",
        "author guidelines",
        "information for authors",
        "can be cited before page numbers have been issued",
        "peer review process",
        "replace this accepted manuscript",
        "this document is the unedited author's version",
        "final edited and published work",
        "downloaded from http://pubs.acs.org",
        "downloaded from https://pubs.acs.org",
        "available online",
        "published online",
        "corresponding author",
        "deposited via the university of",
        "white rose research online",
        "accepted version",
        "withdrawal request",
        "view supplementary material",
        "submit your article to this journal",
        "view related articles",
        "view crossmark data",
        "supplementary information",
        "supplementary material",
        "conflict of interest",
        "declaration of competing interest",
        "data availability",
        "orcid",
        "doi:",
        "creativecommons",
        "open access article",
        "published by",
        "published under",
        "exclusive license",
        "all rights reserved",
    )
    if re.match(r"^\s*(?:title|authors?)\s*[:：]", text, re.IGNORECASE):
        return True
    if re.match(r"^\s*to\s+(?:cite|link)\s+to\s+this\s+article\s*[:：]", text, re.IGNORECASE):
        return True
    if re.match(r"^\s*version\s*:\s*accepted version\b", text, re.IGNORECASE):
        return True
    if re.match(r"^\s*deposited via\b", text, re.IGNORECASE):
        return True
    if re.match(r"^\s*article\s+views\s*[:：]", text, re.IGNORECASE):
        return True
    if re.match(r"^\s*(?:received|accepted)\b", text, re.IGNORECASE):
        return True
    if re.match(r"^\s*(?:cite\s+as|cite\s+this)\s*[:：]", text, re.IGNORECASE):
        return True
    if re.fullmatch(r"(?:https?://|www\.|rsc\.li/)\S+", text.strip(), re.IGNORECASE):
        return True
    if canonical in NON_TITLE_PAGE_HEADINGS:
        return True
    if "angewandte.org" in lowered or "gesellschaft deutscher chemiker" in lowered:
        return True
    return any(fragment in lowered for fragment in metadata_fragments)


def _is_front_matter_noise_paragraph(text: str) -> bool:
    return (
        _is_page_or_panel_label(text)
        or _looks_like_author_list(text)
        or _looks_like_affiliation(text)
        or _looks_like_metadata_paragraph(text)
    )


def _looks_like_table_body(text: str) -> bool:
    normalized = normalize_whitespace(text)
    lowered = normalized.lower()
    if (
        lowered.startswith("<table")
        or lowered.startswith("</table")
        or "<tr" in lowered
        or "</tr" in lowered
        or "<td" in lowered
        or "</td" in lowered
        or "<th" in lowered
        or "</th" in lowered
    ):
        return True
    if normalized.startswith("|") and normalized.count("|") >= 2:
        return True
    if "|" in normalized and re.fullmatch(r"[\s:\-|]+", normalized):
        return True
    return False


def _strip_nonprose_markup(text: str) -> str:
    if "<details" not in text.lower() and "```mermaid" not in text.lower():
        return text
    cleaned = re.sub(r"<details\b.*?</details>", " ", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"```mermaid.*?```", " ", cleaned, flags=re.IGNORECASE)
    cleaned = normalize_whitespace(cleaned)
    cleaned = re.sub(r"^(?:a total of|the following)\s+", "", cleaned, flags=re.IGNORECASE)
    if re.fullmatch(r"(?:fig(?:ure)?\.?\s*)?\d+[a-z]?[.]?", cleaned, flags=re.IGNORECASE):
        return ""
    if re.fullmatch(r"fig(?:ure)?\.?\s*\d+[a-z]?[.]?", cleaned, flags=re.IGNORECASE):
        return ""
    return cleaned


def _looks_like_diagram_body(text: str) -> bool:
    lowered = text.lower()
    if "<details" in lowered or "```mermaid" in lowered:
        return True
    return lowered.startswith("graph td") or text.count("-->") >= 3


def _split_long_text(text: str, target_size: int, max_size: int) -> list[str]:
    if len(text) <= max_size:
        return [text]
    sentences = re.split(r"(?<=[.!?])(?:[\"')\]]+)?\s+", text)
    sentences = [normalize_whitespace(sentence) for sentence in sentences if normalize_whitespace(sentence)]
    if len(sentences) <= 1:
        return [text]

    segments: list[str] = []
    buffer: list[str] = []
    buffer_len = 0
    for sentence in sentences:
        sentence_len = len(sentence)
        projected = buffer_len + sentence_len + (1 if buffer else 0)
        if buffer and projected > max_size:
            segments.append(" ".join(buffer))
            buffer = [sentence]
            buffer_len = sentence_len
        else:
            buffer.append(sentence)
            buffer_len = projected
        if buffer_len >= target_size:
            segments.append(" ".join(buffer))
            buffer = []
            buffer_len = 0
    if buffer:
        segments.append(" ".join(buffer))
    return segments or [text]


def _make_units(
    texts: list[tuple[str, str]],
    section_hint: str | None,
    *,
    target_chars: int,
    max_chars: int,
) -> list[TextUnit]:
    units: list[TextUnit] = []
    for kind, text in texts:
        segments = _split_long_text(text, target_size=target_chars, max_size=max_chars)
        for segment in segments:
            units.append(
                TextUnit(
                    text=segment,
                    section_hint=section_hint,
                    kind=kind,
                    overlap_eligible=len(segments) > 1,
                )
            )
    return units


def _make_body_units(
    body_sections: list[BodySection],
    *,
    target_chars: int,
    max_chars: int,
) -> list[TextUnit]:
    units: list[TextUnit] = []
    for section in body_sections:
        section_hint = _format_heading_path(section.heading_path)
        parts = section.parts
        if not parts:
            continue
        repaired_parts = _repair_body_paragraph_breaks(parts)
        heading_text = _render_section_hint(section_hint)
        if heading_text:
            units.append(
                TextUnit(
                    text=heading_text,
                    section_hint=section_hint,
                    kind="section_heading",
                    overlap_eligible=False,
                )
            )
        units.extend(
            _make_units(
                repaired_parts,
                section_hint,
                target_chars=target_chars,
                max_chars=max_chars,
            )
        )
    return units


def _build_chunk_plans(
    raw_blocks: list[RawBlock],
    title_candidate: TitleCandidate,
    *,
    chunk_size: int,
) -> list[ChunkPlan]:
    candidate_blocks = raw_blocks[title_candidate.block_end_index + 1 :] if title_candidate.block_end_index >= 0 else raw_blocks
    selected_title = title_candidate.title
    selected_title_key = _compact_title_key(selected_title)

    abstract_parts: list[tuple[str, str]] = []
    summary_parts: list[tuple[str, str]] = []
    body_sections: list[BodySection] = []
    active_heading_stack: list[tuple[int, str]] = []
    current_body_parts: list[tuple[str, str]] = []
    mode = "prelude"
    image_pending = False
    caption_mode = False
    abstract_inferred = False
    drop_front_matter = False
    drop_details_block = False

    def flush_body_section() -> None:
        nonlocal current_body_parts
        if current_body_parts:
            body_sections.append(
                BodySection(
                    heading_path=tuple(heading for _, heading in active_heading_stack),
                    parts=current_body_parts,
                )
            )
        current_body_parts = []

    def _set_active_heading(level: int, heading: str) -> None:
        while active_heading_stack and active_heading_stack[-1][0] >= level:
            active_heading_stack.pop()
        active_heading_stack.append((level, heading))

    def _ensure_default_heading() -> None:
        if not active_heading_stack:
            active_heading_stack.append((1, "Introduction"))

    for block in candidate_blocks:
        if block.kind == "heading":
            image_pending = False
            caption_mode = False
            if drop_details_block:
                continue
            heading = _clean_heading(block.text)
            if selected_title_key and _compact_title_key(heading) == selected_title_key:
                flush_body_section()
                current_body_parts = []
                active_heading_stack = []
                mode = "prelude"
                drop_front_matter = False
                abstract_inferred = False
                continue
            if _looks_like_author_heading(heading):
                drop_front_matter = True
                continue
            if _is_tail_stop_heading(heading):
                break
            if _is_front_matter_drop_heading(heading):
                drop_front_matter = True
                continue
            drop_front_matter = False
            if _is_abstract_heading(heading):
                mode = "abstract"
                continue
            if _is_summary_heading(heading) and mode != "body":
                mode = "summary"
                continue
            flush_body_section()
            _set_active_heading(block.level or 1, heading)
            mode = "body"
            continue

        if block.kind == "image":
            image_pending = True
            continue

        text = normalize_whitespace(block.text)
        lowered = text.lower()
        if "<details" in lowered and "</details>" not in lowered:
            drop_details_block = True
            continue
        if drop_details_block:
            if "</details>" in lowered:
                drop_details_block = False
            continue
        if not text or _is_page_or_panel_label(text):
            continue
        text = _strip_nonprose_markup(text)
        if not text or _looks_like_diagram_body(text):
            image_pending = False
            caption_mode = False
            continue
        if _looks_like_table_body(text):
            image_pending = False
            caption_mode = False
            continue

        inline_summary = _extract_inline_summary(text)
        if inline_summary is not None and mode != "body":
            summary_parts.append(("summary_paragraph", inline_summary))
            continue

        inline_abstract = _extract_inline_abstract(text)
        if inline_abstract is not None and mode != "body":
            abstract_parts.append(("abstract_paragraph", inline_abstract))
            abstract_inferred = True
            mode = "abstract"
            continue

        if drop_front_matter:
            continue

        if image_pending and (_looks_like_caption(text) or _looks_like_caption_continuation(text)):
            image_pending = False
            caption_mode = True
            if mode == "body" or active_heading_stack or abstract_inferred:
                _ensure_default_heading()
                current_body_parts.append(("figure_caption", text))
                mode = "body"
            elif mode in {"abstract", "summary"}:
                summary_parts.append(("figure_caption", text))
            continue

        image_pending = False
        if caption_mode and mode == "body" and _looks_like_caption_continuation(text):
            current_body_parts.append(("figure_caption", text))
            continue
        caption_mode = False

        if mode == "abstract":
            if not _is_front_matter_noise_paragraph(text):
                abstract_parts.append(("abstract_paragraph", text))
            continue

        if mode == "summary":
            if not _is_front_matter_noise_paragraph(text):
                summary_parts.append(("summary_paragraph", text))
            continue

        if mode == "body":
            if not _is_front_matter_noise_paragraph(text):
                _ensure_default_heading()
                current_body_parts.append(("body_paragraph", text))
            continue

        if drop_front_matter or _is_front_matter_noise_paragraph(text):
            continue

        if not abstract_inferred and _is_substantial_paragraph(text) and _looks_like_sentence_paragraph(text):
            abstract_parts.append(("abstract_paragraph", text))
            abstract_inferred = True
            continue

        if abstract_inferred:
            _ensure_default_heading()
            current_body_parts.append(("body_paragraph", text))
            mode = "body"

    flush_body_section()

    target_chars = max(400, int(chunk_size * 0.68))
    max_chars = max(800, chunk_size)
    plans: list[ChunkPlan] = []
    if abstract_parts or summary_parts:
        section_hint = "Abstract" if abstract_parts else "Summary"
        plans.append(
            ChunkPlan(
                section_hint=section_hint,
                units=_make_units(
                    abstract_parts + summary_parts,
                    section_hint,
                    target_chars=target_chars,
                    max_chars=max_chars,
                ),
                include_title=True,
            )
        )

    body_units = _make_body_units(body_sections, target_chars=target_chars, max_chars=max_chars)
    if body_units:
        plans.append(
            ChunkPlan(
                section_hint=None,
                units=body_units,
                include_title=False,
            )
        )
    return plans


def _tail_units(units: list[TextUnit], chunk_overlap: int) -> list[TextUnit]:
    if chunk_overlap <= 0:
        return []
    overlap_units: list[TextUnit] = []
    overlap_chars = 0
    for unit in reversed(units):
        if not unit.overlap_eligible:
            continue
        overlap_units.insert(0, unit)
        overlap_chars += len(unit.text)
        if overlap_chars >= chunk_overlap:
            break
    return overlap_units


def _compose_chunk_text(
    *,
    title: str,
    section_hint: str | None,
    units: list[TextUnit],
    include_title: bool,
) -> str:
    parts: list[str] = []
    if include_title:
        parts.append(title)
    rendered_section_hint = _render_section_hint(section_hint)
    if rendered_section_hint and (not units or units[0].kind != "section_heading"):
        parts.append(rendered_section_hint)
    parts.extend(unit.text for unit in units)
    return "\n\n".join(part for part in parts if part).strip()


def _first_section_hint(units: list[TextUnit], fallback: str | None) -> str | None:
    for unit in units:
        if unit.section_hint:
            return unit.section_hint
    return fallback


def _buffer_only_has_section_heading(units: list[TextUnit]) -> bool:
    return bool(units) and all(unit.kind == "section_heading" for unit in units)


def chunk_markdown(
    path: str | Path,
    chunk_size: int = 2200,
    chunk_overlap: int = 100,
) -> tuple[str, str, str, list[ChunkRecord]]:
    source_path = Path(path)
    content = source_path.read_text(encoding="utf-8")
    raw_blocks = _parse_blocks(content)
    try:
        title_candidate = _select_document_title_block(raw_blocks)
    except ValueError as exc:
        raise ValueError(f"Markdown paper {source_path}: {exc}") from exc
    if title_candidate is None:
        raise ValueError(
            f"Markdown paper {source_path} must define a level-1 title heading like '# Paper Title', "
            "and journal masthead headings do not count."
        )
    title = title_candidate.title
    doc_id = _build_doc_id(title)
    file_hash = _hash_file(source_path)

    plans = _build_chunk_plans(raw_blocks, title_candidate, chunk_size=chunk_size)
    if not plans:
        return doc_id, title, file_hash, []

    min_chars = max(300, int(chunk_size * 0.41))
    target_chars = max(400, int(chunk_size * 0.68))
    max_chars = max(800, chunk_size)
    chunks: list[ChunkRecord] = []
    chunk_index = 1

    for plan in plans:
        buffer: list[TextUnit] = []
        buffer_chars = 0
        include_title_for_next_chunk = plan.include_title

        def flush_buffer(*, keep_overlap: bool) -> None:
            nonlocal buffer, buffer_chars, chunk_index, include_title_for_next_chunk
            if not buffer:
                return
            section_hint = _first_section_hint(buffer, plan.section_hint)
            text = _compose_chunk_text(
                title=title,
                section_hint=section_hint,
                units=buffer,
                include_title=include_title_for_next_chunk,
            )
            chunks.append(
                ChunkRecord(
                    chunk_id=f"{doc_id}:{chunk_index:04d}",
                    doc_id=doc_id,
                    title=title,
                    source_path=str(source_path),
                    page_start=0,
                    page_end=0,
                    text=text,
                    section_hint=section_hint,
                    keywords_hint=_extract_keywords(title, section_hint, text),
                    normalized_text=normalize_for_search(text),
                    block_start=0,
                    block_end=0,
                )
            )
            chunk_index += 1
            include_title_for_next_chunk = False
            buffer = _tail_units(buffer, chunk_overlap=chunk_overlap) if keep_overlap else []
            buffer_chars = sum(len(unit.text) for unit in buffer)

        for index, unit in enumerate(plan.units):
            next_unit = plan.units[index + 1] if index + 1 < len(plan.units) else None
            unit_len = len(unit.text)
            if unit.kind == "section_heading":
                next_len = len(next_unit.text) if next_unit is not None else 0
                protected_len = unit_len + next_len
                if buffer and (buffer_chars >= target_chars or buffer_chars + protected_len > max_chars):
                    flush_buffer(keep_overlap=False)

            if (
                buffer
                and unit.kind != "section_heading"
                and buffer_chars + unit_len > max_chars
                and not _buffer_only_has_section_heading(buffer)
                and (unit.kind != "figure_caption" or buffer_chars >= min_chars)
            ):
                flush_buffer(keep_overlap=buffer_chars >= min_chars)

            buffer.append(unit)
            buffer_chars += unit_len

            next_len = len(next_unit.text) if next_unit is not None else 0
            if unit.kind == "section_heading":
                continue
            if next_unit is None:
                flush_buffer(keep_overlap=False)
            elif next_unit.kind == "section_heading" and buffer_chars >= target_chars:
                flush_buffer(keep_overlap=False)
            elif buffer_chars >= target_chars and buffer_chars + next_len > max_chars:
                flush_buffer(keep_overlap=True)

        if buffer:
            flush_buffer(keep_overlap=False)

    deduped: list[ChunkRecord] = []
    seen_texts: set[str] = set()
    for chunk in chunks:
        if chunk.text in seen_texts:
            continue
        seen_texts.add(chunk.text)
        deduped.append(chunk)
    return doc_id, title, file_hash, deduped
