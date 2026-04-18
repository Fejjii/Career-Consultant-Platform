"""Transform API citation payloads into grouped, deduplicated sources for the Streamlit UI."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from typing import Any, Iterable
from urllib.parse import urlparse

# Display limits (after deduplication): show at most this many unique sources.
DEFAULT_MAX_SOURCES = 5
SNIPPET_CHARS = 200

# Ordered for UI sections; unknown families render last under "Other sources".
FAMILY_ORDER: tuple[str, ...] = ("wef", "esco", "youtube", "web", "other")
_ALLOWED_VIDEO_HOSTS = ("youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be")
_ALLOWED_THUMBNAIL_HOST_SUFFIXES = ("ytimg.com", "googleusercontent.com")


@dataclass
class MergedSourceCard:
    """One row in the Sources panel after merging duplicate chunks."""

    family: str
    title: str
    type_label: str
    sublabel: str
    snippet: str
    reference_count: int
    citation_ids: list[int] = field(default_factory=list)
    detail_lines: list[tuple[str, str]] = field(default_factory=list)


def infer_source_family(cite: dict[str, Any]) -> str:
    """Map citation dict to a coarse source family (future: web, youtube)."""
    raw = (cite.get("source") or "").strip().lower()
    if raw in ("wef", "esco", "youtube", "web"):
        return raw
    sid = str(cite.get("source_id") or "")
    lowered = sid.lower()
    if lowered.startswith("wef::"):
        return "wef"
    if lowered.startswith("esco::"):
        return "esco"
    return "other"


def type_label_for_family(family: str) -> str:
    return {
        "wef": "WEF",
        "esco": "ESCO",
        "youtube": "YouTube",
        "web": "Web",
        "other": "Source",
    }.get(family, "Source")


def humanize_esco_doc_type(slug: str | None) -> str | None:
    if not slug:
        return None
    cleaned = str(slug).replace("_", " ").strip()
    if not cleaned:
        return None
    return cleaned[0].upper() + cleaned[1:] if len(cleaned) > 1 else cleaned.upper()


def _parse_page_number(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float):
        n = int(value)
        return n if n > 0 else None
    text = str(value).strip()
    if text.isdigit():
        n = int(text)
        return n if n > 0 else None
    m = re.search(r"\b(\d{1,4})\b", text)
    if m:
        n = int(m.group(1))
        return n if n > 0 else None
    return None


def build_sublabel(cite: dict[str, Any], family: str) -> str:
    """One-line context: page for PDFs, ESCO doc type, or section."""
    page_num = cite.get("page_number")
    pn = _parse_page_number(page_num)
    if pn is not None:
        return f"Page {pn}"
    pol = cite.get("page_or_loc")
    if isinstance(pol, str) and pol.strip():
        return pol.strip()
    esco_label = humanize_esco_doc_type(cite.get("esco_doc_type"))
    if esco_label and family == "esco":
        return esco_label
    if esco_label:
        return esco_label
    sec = cite.get("section")
    if isinstance(sec, str) and sec.strip():
        return sec.strip()[:120] + ("…" if len(sec.strip()) > 120 else "")
    return ""


def _truncate(text: str, max_len: int = SNIPPET_CHARS) -> str:
    t = " ".join(str(text).split())
    if len(t) <= max_len:
        return t
    return t[: max_len - 1].rstrip() + "…"


def _merge_dedupe_key(cite: dict[str, Any], family: str) -> tuple[Any, ...]:
    """Identity for merging multiple chunks from the same logical source."""
    return (
        family,
        str(cite.get("source_id") or ""),
        cite.get("page_number"),
        (str(cite.get("page_or_loc") or "")).strip().lower(),
        (str(cite.get("esco_doc_type") or "")).strip().lower(),
        (str(cite.get("section") or "")).strip().lower(),
        (str(cite.get("title") or "")).strip().lower(),
    )


def _detail_pairs(cite: dict[str, Any]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    if cite.get("file_name"):
        pairs.append(("File", str(cite["file_name"])))
    if cite.get("entity_type"):
        pairs.append(("Entity", str(cite["entity_type"])))
    if cite.get("publish_year") is not None:
        pairs.append(("Year", str(cite["publish_year"])))
    if cite.get("source_id"):
        pairs.append(("Source ID", str(cite["source_id"])))
    uri = _sanitize_link(str(cite.get("uri") or ""))
    if uri:
        pairs.append(("URI", uri))
    sec = cite.get("section")
    if isinstance(sec, str) and sec.strip():
        pairs.append(("Section", sec.strip()))
    return pairs


def merge_citations(citations: Iterable[dict[str, Any]]) -> list[MergedSourceCard]:
    """Merge citations that share the same dedupe key; preserve first-seen order."""
    rows = [dict(c) for c in citations if isinstance(c, dict)]
    rows.sort(key=lambda c: int(c.get("id", 0) or 0))

    merged: dict[tuple[Any, ...], MergedSourceCard] = {}
    order_keys: list[tuple[Any, ...]] = []

    for cite in rows:
        family = infer_source_family(cite)
        key = _merge_dedupe_key(cite, family)
        cid = int(cite.get("id", 0) or 0)
        excerpt = str(cite.get("excerpt") or "")
        title = str(cite.get("title") or "Untitled").strip() or "Untitled"
        sub = build_sublabel(cite, family)
        snippet = _truncate(excerpt)

        if key not in merged:
            order_keys.append(key)
            merged[key] = MergedSourceCard(
                family=family,
                title=title,
                type_label=type_label_for_family(family),
                sublabel=sub,
                snippet=snippet,
                reference_count=0,
                citation_ids=[],
                detail_lines=[],
            )
        card = merged[key]
        card.reference_count += 1
        if cid and cid not in card.citation_ids:
            card.citation_ids.append(cid)
        if len(excerpt) > len(card.snippet.replace("…", "")) and excerpt.strip():
            card.snippet = _truncate(excerpt)
        for label, value in _detail_pairs(cite):
            if (label, value) not in card.detail_lines:
                card.detail_lines.append((label, value))

    return [merged[k] for k in order_keys]


def group_by_family_ordered(cards: list[MergedSourceCard]) -> list[tuple[str, list[MergedSourceCard]]]:
    """Group cards by family, returning sections in a stable product order."""
    buckets: dict[str, list[MergedSourceCard]] = {}
    for card in cards:
        fam = card.family if card.family in ("wef", "esco", "youtube", "web") else "other"
        buckets.setdefault(fam, []).append(card)
    return [(fam, buckets[fam]) for fam in FAMILY_ORDER if fam in buckets]


def panel_families_present(cards: list[MergedSourceCard]) -> list[str]:
    seen = {c.family for c in cards}
    ordered = [f for f in FAMILY_ORDER if f in seen]
    extras = sorted(seen.difference(FAMILY_ORDER))
    return ordered + extras


def prepare_sources_panel_rows(
    citations: list[dict[str, Any]],
    *,
    max_sources: int = DEFAULT_MAX_SOURCES,
) -> tuple[list[MergedSourceCard], dict[str, Any]]:
    """Return capped cards plus debug stats for the panel."""
    merged = merge_citations(citations)
    cap = max(1, min(int(max_sources), DEFAULT_MAX_SOURCES))
    capped = merged[:cap]
    debug = {
        "raw_citation_count": len(citations),
        "unique_sources": len(merged),
        "families": panel_families_present(merged),
        "displayed": len(capped),
        "truncated": max(0, len(merged) - len(capped)),
    }
    return capped, debug


def family_section_title(family: str) -> str:
    return {
        "wef": "WEF sources",
        "esco": "ESCO sources",
        "youtube": "YouTube sources",
        "web": "Web sources",
        "other": "Other sources",
    }.get(family, "Sources")


def family_icon_html(family: str) -> str:
    """Return a small branded icon badge for source-group and card headers."""
    icon_map = {
        "wef": "W",
        "esco": "E",
        "youtube": "▶",
        "web": "◎",
        "other": "•",
    }
    safe_family = html.escape(family if family in icon_map else "other")
    safe_icon = html.escape(icon_map.get(family, icon_map["other"]))
    return f'<span class="ci-source-family-icon ci-source-family-{safe_family}" aria-hidden="true">{safe_icon}</span>'


def render_merged_card_html(card: MergedSourceCard) -> str:
    """HTML for one source card (used inside expander summary area or standalone)."""
    badge = ""
    if card.reference_count > 1:
        badge = (
            f'<span class="badge" title="Merged from {card.reference_count} retrieved chunks">'
            f"{card.reference_count} references</span>"
        )
    sub = html.escape(card.sublabel) if card.sublabel else ""
    title = html.escape(card.title)
    snippet = html.escape(card.snippet) if card.snippet else ""
    type_badge = html.escape(card.type_label)
    meta_line = " · ".join(x for x in [type_badge, sub] if x)
    refs = ", ".join(f"[{i}]" for i in card.citation_ids) if card.citation_ids else ""
    refs_html = f'<div class="meta">Inline citations: {html.escape(refs)}</div>' if refs else ""
    family_icon = family_icon_html(card.family)

    return (
        f'<div class="ci-source-card">'
        f'<div class="topline">'
        f'<div class="brand">{family_icon}<span class="label">{meta_line}</span></div>'
        f"{badge}"
        f"</div>"
        f'<div class="title">{title}</div>'
        + refs_html
        + (f'<div class="ex">“{snippet}”</div>' if snippet else "")
        + "</div>"
    )


def format_detail_lines_for_code(card: MergedSourceCard) -> str:
    """Plain multiline block for st.code (avoids markdown injection from metadata)."""
    if not card.detail_lines:
        return "No extra metadata for this source."
    return "\n".join(f"{label}: {value}" for label, value in card.detail_lines)


def format_youtube_sources_html(videos: Iterable[dict[str, Any]]) -> str:
    """HTML block for supplemental YouTube rows (separate from RAG citation cards)."""
    rows: list[str] = []
    for raw in videos:
        if not isinstance(raw, dict):
            continue
        title = html.escape(str(raw.get("title") or "Video").strip() or "Video")
        channel = html.escape(str(raw.get("channel_name") or "").strip() or "Channel")
        url = _sanitize_link(str(raw.get("video_url") or ""), allowed_hosts=_ALLOWED_VIDEO_HOSTS)
        thumb = _sanitize_link(
            str(raw.get("thumbnail_url") or ""),
            allowed_host_suffixes=_ALLOWED_THUMBNAIL_HOST_SUFFIXES,
        )
        if not url:
            continue
        safe_url = html.escape(url, quote=True)
        thumb_html = ""
        if thumb:
            safe_thumb = html.escape(thumb, quote=True)
            thumb_html = (
                f'<div class="ci-yt-thumb"><img src="{safe_thumb}" alt="" loading="lazy" /></div>'
            )
        rows.append(
            '<div class="ci-yt-row">'
            f"{thumb_html}"
            '<div class="ci-yt-body">'
            f'<a class="ci-yt-title" href="{safe_url}" target="_blank" rel="noopener noreferrer">{title}</a>'
            f'<div class="ci-yt-channel">{channel}</div>'
            "</div></div>"
        )
    if not rows:
        return ""
    head = html.escape(family_section_title("youtube"))
    return (
        '<div class="ci-source-group-head">'
        f'{family_icon_html("youtube")}'
        f'<p class="ci-sources-group-head">{head}</p>'
        "</div>"
        f'<div class="ci-yt-list">{"".join(rows)}</div>'
    )


def _sanitize_link(
    value: str,
    *,
    allowed_hosts: tuple[str, ...] = (),
    allowed_host_suffixes: tuple[str, ...] = (),
) -> str:
    raw = value.strip()
    if not raw:
        return ""
    parsed = urlparse(raw)
    host = (parsed.netloc or "").lower()
    if parsed.scheme not in {"http", "https"} or not host:
        return ""
    if allowed_hosts and host not in allowed_hosts:
        return ""
    if allowed_host_suffixes and not any(host == suffix or host.endswith(f".{suffix}") for suffix in allowed_host_suffixes):
        return ""
    return raw
