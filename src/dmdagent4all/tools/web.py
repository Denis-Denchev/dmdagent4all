from __future__ import annotations

import ipaddress
import json
import re
import socket
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from dmdagent4all.security import redact_text
from dmdagent4all.tools.base import ToolRuntimeContext
from dmdagent4all.tools.storage import internet_files_dir


DEFAULT_TIMEOUT_SECONDS = 15
DEFAULT_MAX_RESPONSE_BYTES = 1_000_000
DEFAULT_MAX_TEXT_CHARS = 12_000
USER_AGENT = "DMD-Agent-4-All/1.0 safety-first browser-read"


@dataclass(frozen=True)
class FetchedPage:
    url: str
    final_url: str
    status: int
    content_type: str
    body: str
    bytes_read: int
    truncated: bool


@dataclass(frozen=True)
class ArticleCandidate:
    title: str
    url: str
    summary: str = ""
    source: str = ""
    order: int = 0
    score: int = 0


@dataclass
class _HTMLNode:
    tag: str
    attrs: dict[str, str]
    children: list[_HTMLNode | str]
    parent: _HTMLNode | None = None


def browser_open(args: dict[str, Any], context: ToolRuntimeContext) -> dict[str, Any]:
    page = fetch_page(str(args.get("url") or ""), context)
    metadata = extract_metadata(page.body, page.content_type)
    return {
        "url": page.url,
        "final_url": page.final_url,
        "status": page.status,
        "content_type": page.content_type,
        "title": metadata.get("title", ""),
        "description": metadata.get("description", ""),
        "bytes_read": page.bytes_read,
        "truncated": page.truncated,
    }


def browser_extract_text(args: dict[str, Any], context: ToolRuntimeContext) -> dict[str, Any]:
    page = fetch_page(str(args.get("url") or ""), context)
    max_chars = _max_text_chars(args, context)
    text = extract_readable_text(page.body, page.content_type)
    truncated = len(text) > max_chars
    if truncated:
        text = text[:max_chars].rstrip() + "\n[content truncated]"
    return {
        "url": page.url,
        "final_url": page.final_url,
        "status": page.status,
        "content_type": page.content_type,
        "text": redact_text(text),
        "bytes_read": page.bytes_read,
        "response_truncated": page.truncated,
        "text_truncated": truncated,
    }


def browser_scrape_markdown(args: dict[str, Any], context: ToolRuntimeContext) -> dict[str, Any]:
    page = fetch_page(str(args.get("url") or ""), context)
    max_chars = _max_text_chars(args, context)
    metadata = extract_metadata(page.body, page.content_type)
    title = metadata.get("title", "").strip() or _title_from_url(page.final_url)
    instructions = _clean_optional_text(args.get("instructions"), max_chars=2000)
    scrape_options = _scrape_options(args, instructions=instructions)
    scraped_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    candidates: list[ArticleCandidate] = []
    targeted_failed = False
    raw_fallback = False

    if scrape_options["mode"] == "targeted" and scrape_options["content_type"] == "articles":
        candidates = _extract_article_candidates(page.body, page.content_type, base_url=page.final_url, metadata=metadata)
        limit = scrape_options["limit"]
        if candidates:
            candidates = candidates[:limit]
            markdown = _build_targeted_articles_markdown(
                page_title=title,
                source_url=page.url,
                final_url=page.final_url,
                status=page.status,
                content_type=page.content_type,
                scraped_at=scraped_at,
                instructions=instructions,
                articles=candidates,
                requested_limit=limit,
            )
        else:
            targeted_failed = True
            raw_fallback = True
            content = html_to_markdown(page.body, page.content_type, base_url=page.final_url)
            if not content.strip():
                content = _metadata_fallback_markdown(metadata)
            markdown = _build_scrape_markdown(
                title=title,
                source_url=page.url,
                final_url=page.final_url,
                status=page.status,
                content_type=page.content_type,
                scraped_at=scraped_at,
                instructions=instructions,
                content=content,
                scrape_mode="raw_page",
                requested_mode="targeted",
                targeted_failed=True,
            )
    else:
        content = html_to_markdown(page.body, page.content_type, base_url=page.final_url)
        if not content.strip():
            content = _metadata_fallback_markdown(metadata)
        markdown = _build_scrape_markdown(
            title=title,
            source_url=page.url,
            final_url=page.final_url,
            status=page.status,
            content_type=page.content_type,
            scraped_at=scraped_at,
            instructions=instructions,
            content=content,
            scrape_mode="raw_page",
        )
    markdown = redact_text(markdown)
    markdown_truncated = len(markdown) > max_chars
    if markdown_truncated:
        markdown = markdown[:max_chars].rstrip() + "\n\n[content truncated]\n"

    output_dir = internet_files_dir(context.config, default_root=context.workspace_root, folder="scrapefiles")
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = _safe_markdown_filename(
        raw_filename=args.get("filename"),
        title=title,
        url=page.final_url,
    )
    output_path = _unique_markdown_path(output_dir, filename)
    output_path.write_text(markdown, encoding="utf-8")

    workspace_root = output_dir.parent.resolve()
    try:
        relative_path = str(output_path.relative_to(workspace_root))
    except ValueError:
        relative_path = str(output_path)

    return {
        "url": page.url,
        "final_url": page.final_url,
        "status": page.status,
        "content_type": page.content_type,
        "title": title,
        "markdown": markdown,
        "path": str(output_path),
        "relative_path": relative_path,
        "scrape_mode": "raw_page" if raw_fallback else scrape_options["mode"],
        "requested_mode": scrape_options["mode"],
        "content_type_requested": scrape_options["content_type"],
        "format": scrape_options["format"],
        "extraction_status": "raw_fallback" if raw_fallback else "ok",
        "targeted_failed": targeted_failed,
        "raw_fallback": raw_fallback,
        "extracted_count": len(candidates),
        "requested_limit": scrape_options["limit"],
        "items": [
            {"title": item.title, "url": item.url, "summary": item.summary}
            for item in candidates
        ],
        "bytes_read": page.bytes_read,
        "response_truncated": page.truncated,
        "markdown_truncated": markdown_truncated,
    }


def fetch_page(raw_url: str, context: ToolRuntimeContext) -> FetchedPage:
    url = normalize_url(raw_url)
    assert_public_http_url(url)
    browser_config = context.config.get("browser", {})
    timeout = int(browser_config.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS))
    max_bytes = int(browser_config.get("max_response_bytes", DEFAULT_MAX_RESPONSE_BYTES))
    request = Request(url, headers={"User-Agent": USER_AGENT})
    opener = build_opener(_SafeRedirectHandler)
    try:
        with opener.open(request, timeout=timeout) as response:
            final_url = response.geturl()
            assert_public_http_url(final_url)
            content_type = response.headers.get("content-type", "")
            body_bytes = response.read(max_bytes + 1)
            truncated = len(body_bytes) > max_bytes
            if truncated:
                body_bytes = body_bytes[:max_bytes]
            body = body_bytes.decode(_charset_from_content_type(content_type), errors="replace")
            return FetchedPage(
                url=url,
                final_url=final_url,
                status=int(getattr(response, "status", 200)),
                content_type=content_type,
                body=body,
                bytes_read=len(body_bytes),
                truncated=truncated,
            )
    except HTTPError as exc:
        raise ValueError(f"HTTP request failed with status {exc.code}.") from exc
    except URLError as exc:
        raise ValueError(f"HTTP request failed: {exc.reason}") from exc


def normalize_url(raw_url: str) -> str:
    value = raw_url.strip()
    if not value:
        raise ValueError("browser tools require url.")
    if "://" not in value:
        value = f"https://{value}"
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("browser tools require a valid http or https URL.")
    if parsed.username or parsed.password:
        raise ValueError("browser tools do not accept URLs with embedded credentials.")
    return parsed.geturl()


def assert_public_http_url(raw_url: str) -> None:
    parsed = urlparse(raw_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("browser tools require a valid http or https URL.")
    hostname = parsed.hostname.strip().lower()
    if hostname in {"localhost", "localhost.localdomain"} or hostname.endswith(".local"):
        raise ValueError("browser tools block local hostnames.")
    try:
        _assert_public_ip(ipaddress.ip_address(hostname))
        return
    except ValueError:
        pass
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        infos = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValueError(f"Could not resolve host: {hostname}") from exc
    if not infos:
        raise ValueError(f"Could not resolve host: {hostname}")
    for info in infos:
        address = info[4][0]
        _assert_public_ip(ipaddress.ip_address(address))


def extract_metadata(body: str, content_type: str) -> dict[str, str]:
    if "html" not in content_type.lower() and not _looks_like_html(body):
        return {"title": "", "description": ""}
    parser = _MetadataParser()
    parser.feed(body)
    return {
        "title": _collapse_space(parser.title),
        "description": _collapse_space(parser.description),
    }


def extract_readable_text(body: str, content_type: str) -> str:
    if "html" not in content_type.lower() and not _looks_like_html(body):
        return _collapse_space(body)
    parser = _VisibleTextParser()
    parser.feed(body)
    return _collapse_space(" ".join(parser.parts))


def html_to_markdown(body: str, content_type: str, *, base_url: str) -> str:
    if "html" not in content_type.lower() and not _looks_like_html(body):
        return _plain_text_to_markdown(body)
    parser = _MarkdownHTMLParser(base_url=base_url)
    parser.feed(body)
    markdown = parser.markdown()
    if markdown:
        return markdown
    return _plain_text_to_markdown(extract_readable_text(body, content_type))


def _assert_public_ip(address: ipaddress._BaseAddress) -> None:
    if (
        address.is_loopback
        or address.is_private
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    ):
        raise ValueError("browser tools block local, private, reserved, and link-local addresses.")


def _charset_from_content_type(content_type: str) -> str:
    for part in content_type.split(";"):
        key, separator, value = part.strip().partition("=")
        if separator and key.lower() == "charset" and value.strip():
            return value.strip()
    return "utf-8"


def _max_text_chars(args: dict[str, Any], context: ToolRuntimeContext) -> int:
    browser_config = context.config.get("browser", {})
    configured_cap = int(browser_config.get("max_text_chars", DEFAULT_MAX_TEXT_CHARS))
    raw = args.get("max_chars", configured_cap)
    try:
        requested = int(raw)
    except (TypeError, ValueError):
        requested = configured_cap
    return max(500, min(requested, configured_cap))


def _scrape_options(args: dict[str, Any], *, instructions: str) -> dict[str, Any]:
    normalized_instructions = _collapse_space(instructions).casefold()
    raw_mode = str(args.get("mode") or "").strip().casefold()
    raw_content_type = str(args.get("content_type") or "").strip().casefold()
    raw_format = str(args.get("format") or "").strip().casefold()

    limit = _coerce_limit(args.get("limit"))
    if limit is None:
        limit = _limit_from_instructions(normalized_instructions)

    content_type = raw_content_type or ""
    if content_type not in {"articles"}:
        content_type = "articles" if _instructions_ask_for_articles(normalized_instructions) else "page"

    mode = raw_mode or ""
    if mode not in {"raw_page", "targeted"}:
        mode = "targeted" if content_type == "articles" or limit is not None else "raw_page"

    if mode == "targeted" and content_type == "page":
        content_type = "articles" if _instructions_ask_for_articles(normalized_instructions) else "page"

    output_format = raw_format if raw_format in {"markdown", "clean_markdown"} else "markdown"
    if mode == "targeted" and output_format == "markdown":
        output_format = "clean_markdown"

    return {
        "mode": mode,
        "content_type": content_type,
        "limit": max(1, min(int(limit or 10), 50)),
        "format": output_format,
    }


def _coerce_limit(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        limit = int(value)
    except (TypeError, ValueError):
        return None
    if limit < 1:
        return None
    return min(limit, 50)


def _limit_from_instructions(normalized: str) -> int | None:
    patterns = [
        r"\bfirst\s+(?P<limit>\d{1,2})\b",
        r"\btop\s+(?P<limit>\d{1,2})\b",
        r"\blatest\s+(?P<limit>\d{1,2})\b",
        r"\b(?P<limit>\d{1,2})\s+(?:articles|posts|stories|news)\b",
        r"\bпърв(?:ите|и)\s+(?P<limit>\d{1,2})\b",
        r"\bпоследн(?:ите|и)\s+(?P<limit>\d{1,2})\b",
        r"\b(?P<limit>\d{1,2})\s+(?:статии|новини|публикации)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            return _coerce_limit(match.group("limit"))
    return None


def _instructions_ask_for_articles(normalized: str) -> bool:
    return any(
        marker in normalized
        for marker in {
            "article",
            "articles",
            "post",
            "posts",
            "story",
            "stories",
            "newsarticle",
            "news article",
            "статия",
            "статии",
            "новина",
            "новини",
            "публикация",
            "публикации",
        }
    )


def _looks_like_html(value: str) -> bool:
    head = value.lstrip()[:200].lower()
    return head.startswith("<!doctype html") or head.startswith("<html") or "<body" in head


def _collapse_space(value: str) -> str:
    return " ".join(value.split())


def _plain_text_to_markdown(value: str) -> str:
    lines = [line.rstrip() for line in value.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    compact: list[str] = []
    blank = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if compact and not blank:
                compact.append("")
            blank = True
            continue
        compact.append(stripped)
        blank = False
    return "\n".join(compact).strip()


def _extract_article_candidates(
    body: str,
    content_type: str,
    *,
    base_url: str,
    metadata: dict[str, str],
) -> list[ArticleCandidate]:
    if "html" not in content_type.lower() and not _looks_like_html(body):
        return []

    candidates: list[ArticleCandidate] = []
    order = 0
    for candidate in _json_ld_article_candidates(body, base_url=base_url):
        candidates.append(
            ArticleCandidate(
                title=candidate.title,
                url=candidate.url,
                summary=candidate.summary,
                source=candidate.source,
                order=order,
                score=candidate.score,
            )
        )
        order += 1

    parser = _HTMLTreeParser()
    parser.feed(body)
    root = parser.root
    main_nodes = list(_iter_nodes(root, {"main"}))
    search_roots = main_nodes or [_body_node(root) or root]

    for article in _iter_nodes(root, {"article"}):
        if _is_ignored_node(article):
            continue
        candidate = _candidate_from_article_node(article, base_url=base_url, order=order)
        if candidate is not None:
            candidates.append(candidate)
            order += 1

    for search_root in search_roots:
        for heading in _iter_nodes(search_root, {"h1", "h2", "h3"}):
            if _is_ignored_node(heading):
                continue
            candidate = _candidate_from_heading_node(heading, base_url=base_url, order=order)
            if candidate is not None:
                candidates.append(candidate)
                order += 1

        for link in _iter_nodes(search_root, {"a"}):
            if _is_ignored_node(link):
                continue
            candidate = _candidate_from_link_node(link, base_url=base_url, order=order)
            if candidate is not None:
                candidates.append(candidate)
                order += 1

    if len(candidates) <= 1:
        og_candidate = _candidate_from_open_graph(metadata, base_url=base_url, order=order)
        if og_candidate is not None:
            candidates.append(og_candidate)

    return _dedupe_article_candidates(candidates, base_url=base_url)


def _json_ld_article_candidates(body: str, *, base_url: str) -> list[ArticleCandidate]:
    parser = _JsonLdParser()
    parser.feed(body)
    candidates: list[ArticleCandidate] = []
    for script in parser.scripts:
        try:
            payload = json.loads(script)
        except json.JSONDecodeError:
            continue
        for item in _iter_jsonld_items(payload):
            article_type = item.get("@type") or item.get("type")
            article_types = article_type if isinstance(article_type, list) else [article_type]
            if not any(str(value).casefold() in {"newsarticle", "article", "blogposting"} for value in article_types):
                continue
            title = _collapse_space(str(item.get("headline") or item.get("name") or ""))
            url = _jsonld_url(item)
            absolute_url = _safe_markdown_link(str(url or ""), base_url=base_url)
            if not title or not absolute_url:
                continue
            summary = _collapse_space(str(item.get("description") or ""))
            candidates.append(
                ArticleCandidate(
                    title=title,
                    url=absolute_url,
                    summary=summary,
                    source="json_ld",
                    score=100,
                )
            )
    return candidates


def _iter_jsonld_items(value: Any) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if isinstance(value, dict):
        items.append(value)
        graph = value.get("@graph")
        if isinstance(graph, list):
            for child in graph:
                items.extend(_iter_jsonld_items(child))
    elif isinstance(value, list):
        for child in value:
            items.extend(_iter_jsonld_items(child))
    return items


def _jsonld_url(item: dict[str, Any]) -> str:
    raw_url = item.get("url")
    if isinstance(raw_url, str):
        return raw_url
    main_entity = item.get("mainEntityOfPage")
    if isinstance(main_entity, str):
        return main_entity
    if isinstance(main_entity, dict):
        entity_id = main_entity.get("@id") or main_entity.get("url")
        if isinstance(entity_id, str):
            return entity_id
    return ""


def _candidate_from_article_node(node: _HTMLNode, *, base_url: str, order: int) -> ArticleCandidate | None:
    heading = next(_iter_nodes(node, {"h1", "h2", "h3"}), None)
    title = _node_text(heading) if heading is not None else ""
    link = _first_link(heading or node, base_url=base_url, prefer_article_like=True)
    if not title and link is not None:
        title = link[0]
    if link is None:
        link = _first_link(node, base_url=base_url, prefer_article_like=False)
    if link is None:
        return None
    url = link[1]
    if not _is_article_like_url(url, base_url=base_url, title=title):
        return None
    summary = _summary_from_node(node, title=title)
    return _article_candidate(title=title, url=url, summary=summary, source="article", order=order, score=85)


def _candidate_from_heading_node(node: _HTMLNode, *, base_url: str, order: int) -> ArticleCandidate | None:
    title = _node_text(node)
    link = _first_link(node, base_url=base_url, prefer_article_like=True)
    container = _nearest_content_container(node)
    if link is None and container is not None:
        link = _first_link(container, base_url=base_url, prefer_article_like=True)
    if link is None:
        return None
    url = link[1]
    if not _is_article_like_url(url, base_url=base_url, title=title):
        return None
    summary = _summary_from_node(container, title=title) if container is not None and container.tag == "article" else ""
    return _article_candidate(title=title, url=url, summary=summary, source="heading", order=order, score=75)


def _candidate_from_link_node(node: _HTMLNode, *, base_url: str, order: int) -> ArticleCandidate | None:
    title = _node_text(node)
    href = _node_attr(node, "href")
    url = _safe_markdown_link(href, base_url=base_url)
    if not url or not _is_article_like_url(url, base_url=base_url, title=title):
        return None
    return _article_candidate(title=title, url=url, summary="", source="link", order=order, score=55)


def _candidate_from_open_graph(
    metadata: dict[str, str],
    *,
    base_url: str,
    order: int,
) -> ArticleCandidate | None:
    title = _collapse_space(metadata.get("title", ""))
    summary = _collapse_space(metadata.get("description", ""))
    if not title or not _is_article_like_url(base_url, base_url=base_url, title=title):
        return None
    return _article_candidate(title=title, url=base_url, summary=summary, source="metadata", order=order, score=35)


def _article_candidate(
    *,
    title: str,
    url: str,
    summary: str,
    source: str,
    order: int,
    score: int,
) -> ArticleCandidate | None:
    clean_title = _clean_article_title(title)
    if not clean_title:
        return None
    clean_summary = _clean_article_summary(summary, title=clean_title)
    return ArticleCandidate(
        title=clean_title,
        url=url,
        summary=clean_summary,
        source=source,
        order=order,
        score=score,
    )


def _clean_article_title(title: str) -> str:
    clean = _collapse_space(title)
    clean = re.sub(r"^\s*[>»›]+\s*", "", clean)
    clean = clean.strip(" -|/\t\r\n")
    if len(clean) < 8 or len(clean) > 220:
        return ""
    if _is_navigation_text(clean):
        return ""
    return clean


def _clean_article_summary(summary: str, *, title: str) -> str:
    clean = _collapse_space(summary)
    if not clean or clean == title:
        return ""
    if clean.startswith(title):
        clean = clean[len(title) :].strip(" -:|")
    if len(clean) > 360:
        clean = clean[:357].rstrip() + "..."
    return clean


def _summary_from_node(node: _HTMLNode | None, *, title: str) -> str:
    if node is None:
        return ""
    for paragraph in _iter_nodes(node, {"p"}):
        if _is_ignored_node(paragraph):
            continue
        text = _node_text(paragraph)
        if text and text != title and len(text) >= 20:
            return text
    return ""


def _dedupe_article_candidates(candidates: list[ArticleCandidate], *, base_url: str) -> list[ArticleCandidate]:
    by_key: dict[str, ArticleCandidate] = {}
    for candidate in candidates:
        if not candidate.title or not candidate.url:
            continue
        if not _is_article_like_url(candidate.url, base_url=base_url, title=candidate.title):
            continue
        key = _article_candidate_key(candidate)
        existing = by_key.get(key)
        if existing is None:
            by_key[key] = candidate
            continue
        if candidate.score > existing.score:
            by_key[key] = ArticleCandidate(
                title=candidate.title,
                url=candidate.url,
                summary=candidate.summary or existing.summary,
                source=candidate.source,
                order=min(candidate.order, existing.order),
                score=candidate.score,
            )
        elif not existing.summary and candidate.summary:
            by_key[key] = ArticleCandidate(
                title=existing.title,
                url=existing.url,
                summary=candidate.summary,
                source=existing.source,
                order=existing.order,
                score=existing.score,
            )
    return sorted(by_key.values(), key=lambda item: (item.order, -item.score))


def _article_candidate_key(candidate: ArticleCandidate) -> str:
    parsed = urlparse(candidate.url)
    path = parsed.path.rstrip("/") or "/"
    if parsed.netloc and path != "/":
        return f"url:{parsed.scheme}://{parsed.netloc}{path}".casefold()
    return f"title:{_normalize_title_key(candidate.title)}"


def _normalize_title_key(title: str) -> str:
    return re.sub(r"\W+", "", title.casefold(), flags=re.UNICODE)


def _is_article_like_url(raw_url: str, *, base_url: str, title: str) -> bool:
    parsed = urlparse(raw_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return False
    base = urlparse(base_url)
    if base.netloc and parsed.netloc != base.netloc:
        return False
    path = parsed.path.strip("/")
    if not path:
        return False
    if parsed.fragment:
        return False
    segments = [segment for segment in path.split("/") if segment]
    if not segments:
        return False
    lowered_segments = {segment.casefold() for segment in segments}
    blocked_segments = {
        "account",
        "accounts",
        "author",
        "authors",
        "category",
        "categories",
        "competition",
        "competitions",
        "league",
        "leagues",
        "login",
        "logout",
        "menu",
        "profile",
        "register",
        "registration",
        "schedule",
        "search",
        "standings",
        "tag",
        "tags",
        "team",
        "teams",
        "tv-schedule",
        "user",
        "users",
    }
    if lowered_segments & blocked_segments:
        return False
    blocked_prefixes = ("category-", "competition-", "league-", "tag-", "team-")
    if any(segment.casefold().startswith(blocked_prefixes) for segment in segments):
        return False
    clean_title = _clean_article_title(title)
    if not clean_title:
        return False
    last = segments[-1]
    title_is_substantial = len(clean_title) >= 18 and len(clean_title.split()) >= 3
    path_is_sluggy = "-" in last or "_" in last or bool(re.search(r"\d{3,}", last))
    return title_is_substantial or path_is_sluggy


def _is_navigation_text(text: str) -> bool:
    normalized = _collapse_space(text).casefold()
    if len(normalized) <= 2:
        return True
    blocked = {
        "back",
        "home",
        "login",
        "menu",
        "register",
        "search",
        "sign in",
        "sign up",
        "view all",
        "виж всички",
        "вход",
        "изход",
        "назад",
        "регистрирай се",
        "търсене",
    }
    return normalized in blocked


def _body_node(root: _HTMLNode) -> _HTMLNode | None:
    return next(_iter_nodes(root, {"body"}), None)


def _iter_nodes(node: _HTMLNode, tags: set[str]) -> Any:
    for child in node.children:
        if not isinstance(child, _HTMLNode):
            continue
        if child.tag in tags:
            yield child
        yield from _iter_nodes(child, tags)


def _node_text(node: _HTMLNode | None) -> str:
    if node is None:
        return ""
    parts: list[str] = []
    for child in node.children:
        if isinstance(child, str):
            parts.append(child)
        elif not _is_structural_noise_node(child):
            parts.append(_node_text(child))
    return _collapse_space(" ".join(part for part in parts if part))


def _node_attr(node: _HTMLNode | None, name: str) -> str:
    if node is None:
        return ""
    return node.attrs.get(name, "")


def _first_link(
    node: _HTMLNode,
    *,
    base_url: str,
    prefer_article_like: bool,
) -> tuple[str, str] | None:
    fallback: tuple[str, str] | None = None
    for link in _iter_nodes(node, {"a"}):
        title = _node_text(link)
        href = _node_attr(link, "href")
        url = _safe_markdown_link(href, base_url=base_url)
        if not url or not title:
            continue
        pair = (title, url)
        if prefer_article_like:
            if _is_article_like_url(url, base_url=base_url, title=title):
                return pair
            if fallback is None:
                fallback = pair
        else:
            return pair
    return fallback if not prefer_article_like else None


def _nearest_content_container(node: _HTMLNode) -> _HTMLNode | None:
    current = node.parent
    while current is not None:
        if current.tag in {"article", "li", "div", "section"}:
            return current
        current = current.parent
    return None


def _is_ignored_node(node: _HTMLNode) -> bool:
    current: _HTMLNode | None = node
    while current is not None:
        if _is_structural_noise_node(current):
            return True
        current = current.parent
    return False


def _is_structural_noise_node(node: _HTMLNode) -> bool:
    if node.tag in {"aside", "footer", "form", "header", "nav"}:
        return True
    values = " ".join(
        value
        for key, value in node.attrs.items()
        if key in {"aria-label", "class", "id", "role"}
    ).casefold()
    if not values:
        return False
    return any(
        marker in values
        for marker in {
            "account",
            "auth",
            "breadcrumb",
            "cookie",
            "footer",
            "header",
            "login",
            "menu",
            "modal",
            "nav",
            "profile",
            "register",
            "search",
            "share",
            "sidebar",
            "social",
        }
    )


def _build_scrape_markdown(
    *,
    title: str,
    source_url: str,
    final_url: str,
    status: int,
    content_type: str,
    scraped_at: str,
    instructions: str,
    content: str,
    scrape_mode: str = "raw_page",
    requested_mode: str = "",
    targeted_failed: bool = False,
) -> str:
    lines = [
        "---",
        f'source_url: "{_escape_yaml_scalar(source_url)}"',
        f'final_url: "{_escape_yaml_scalar(final_url)}"',
        f'scraped_at: "{_escape_yaml_scalar(scraped_at)}"',
        f"status: {status}",
        f'content_type: "{_escape_yaml_scalar(content_type)}"',
        f'scrape_mode: "{_escape_yaml_scalar(scrape_mode)}"',
    ]
    if requested_mode:
        lines.append(f'requested_mode: "{_escape_yaml_scalar(requested_mode)}"')
    if targeted_failed:
        lines.append("targeted_failed: true")
        lines.append("raw_fallback: true")
    lines.extend(["---", "", f"# {_escape_markdown_heading(title or 'Scraped Page')}", ""])
    if targeted_failed:
        lines.extend(
            [
                "## Targeted Extraction Failed",
                "",
                "No article candidates matched the requested targeted extraction. Saved a raw page fallback instead.",
                "",
            ]
        )
    if instructions:
        lines.extend(
            [
                "## Scrape Instructions",
                "",
                instructions,
                "",
            ]
        )
    lines.extend(["## Content", "", content.strip() or "_No readable page content extracted._", ""])
    return "\n".join(lines)


def _build_targeted_articles_markdown(
    *,
    page_title: str,
    source_url: str,
    final_url: str,
    status: int,
    content_type: str,
    scraped_at: str,
    instructions: str,
    articles: list[ArticleCandidate],
    requested_limit: int,
) -> str:
    source_host = urlparse(final_url).netloc or urlparse(source_url).netloc or "source"
    count = min(len(articles), requested_limit)
    lines = [
        "---",
        f'source_url: "{_escape_yaml_scalar(source_url)}"',
        f'final_url: "{_escape_yaml_scalar(final_url)}"',
        f'scraped_at: "{_escape_yaml_scalar(scraped_at)}"',
        f"status: {status}",
        f'content_type: "{_escape_yaml_scalar(content_type)}"',
        'scrape_mode: "targeted"',
        'content_type_requested: "articles"',
        f"requested_limit: {requested_limit}",
        f"extracted_count: {count}",
        "---",
        "",
        f"# {_escape_markdown_heading(page_title or source_host)} - first {count} articles",
        "",
        f"Source: {final_url}",
        f"Scraped at: {scraped_at}",
        "",
    ]
    if instructions:
        lines.extend(["## Scrape Instructions", "", instructions, ""])
    for index, article in enumerate(articles[:requested_limit], start=1):
        summary = article.summary or "No summary found"
        lines.extend(
            [
                f"## {index}. {_escape_markdown_heading(article.title)}",
                f"- Link: {article.url}",
                f"- Summary: {summary}",
                "",
            ]
        )
    return "\n".join(lines)


def _metadata_fallback_markdown(metadata: dict[str, str]) -> str:
    description = _collapse_space(metadata.get("description", ""))
    if not description:
        return ""
    return description


def _clean_optional_text(value: Any, *, max_chars: int) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = _plain_text_to_markdown(text)
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "\n[truncated]"
    return redact_text(text)


def _title_from_url(raw_url: str) -> str:
    parsed = urlparse(raw_url)
    path = parsed.path.strip("/")
    if path:
        seed = path.rsplit("/", 1)[-1]
    else:
        seed = parsed.netloc or "Scraped Page"
    seed = re.sub(r"[-_]+", " ", seed)
    return seed.strip().title() or "Scraped Page"


def _safe_markdown_filename(*, raw_filename: Any, title: str, url: str) -> str:
    if raw_filename:
        candidate = Path(str(raw_filename)).name
    else:
        candidate = title or _url_filename_seed(url)
    if candidate.casefold().endswith(".md"):
        candidate = candidate[:-3]
    slug = _slugify_filename(candidate)
    if not slug:
        slug = _slugify_filename(_url_filename_seed(url))
    if not slug:
        slug = "scraped-page"
    return f"{slug[:80].strip('-')}.md"


def _url_filename_seed(raw_url: str) -> str:
    parsed = urlparse(raw_url)
    path = parsed.path.strip("/")
    if path:
        return f"{parsed.netloc}-{path.rsplit('/', 1)[-1]}"
    return parsed.netloc or "scraped-page"


def _slugify_filename(value: str) -> str:
    slug = re.sub(r"[^\w.-]+", "-", value.lower(), flags=re.UNICODE)
    slug = re.sub(r"-{2,}", "-", slug).strip("._- ")
    if slug in {"", ".", ".."}:
        return ""
    return slug


def _unique_markdown_path(folder: Path, filename: str) -> Path:
    path = folder / filename
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix or ".md"
    for index in range(2, 10_000):
        candidate = folder / f"{stem}-{index}{suffix}"
        if not candidate.exists():
            return candidate
    raise ValueError("Could not allocate a unique scrape output filename.")


def _escape_yaml_scalar(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _escape_markdown_heading(value: str) -> str:
    return value.replace("\n", " ").strip().lstrip("#").strip() or "Scraped Page"


def _escape_markdown_link_text(value: str) -> str:
    return value.replace("[", "\\[").replace("]", "\\]").strip()


def _safe_markdown_link(raw_href: str, *, base_url: str) -> str:
    href = raw_href.strip()
    if not href:
        return ""
    if href.lower().startswith(("javascript:", "data:", "file:", "vbscript:")):
        return ""
    absolute = urljoin(base_url, href)
    parsed = urlparse(absolute)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    return absolute.replace(" ", "%20").replace(")", "%29")


class _JsonLdParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.scripts: list[str] = []
        self._collecting = False
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "script":
            return
        attr_map = {name.lower(): value or "" for name, value in attrs}
        script_type = attr_map.get("type", "").casefold()
        if "ld+json" in script_type:
            self._collecting = True
            self._parts = []

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "script" and self._collecting:
            script = "".join(self._parts).strip()
            if script:
                self.scripts.append(script)
            self._collecting = False
            self._parts = []

    def handle_data(self, data: str) -> None:
        if self._collecting:
            self._parts.append(data)


class _HTMLTreeParser(HTMLParser):
    SKIP_TAGS = {"canvas", "iframe", "script", "style", "noscript", "template", "svg"}
    VOID_TAGS = {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = _HTMLNode(tag="document", attrs={}, children=[])
        self._stack: list[_HTMLNode] = [self.root]
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        attr_map = {name.lower(): value or "" for name, value in attrs}
        parent = self._stack[-1]
        node = _HTMLNode(tag=tag, attrs=attr_map, children=[], parent=parent)
        parent.children.append(node)
        if tag not in self.VOID_TAGS:
            self._stack.append(node)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self.SKIP_TAGS:
            if self._skip_depth:
                self._skip_depth -= 1
            return
        if self._skip_depth:
            return
        for index in range(len(self._stack) - 1, 0, -1):
            if self._stack[index].tag == tag:
                del self._stack[index:]
                return

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = _collapse_space(data)
        if text:
            self._stack[-1].children.append(text)


class _SafeRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        assert_public_http_url(normalize_url(newurl))
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class _MetadataParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._in_title = False
        self.title = ""
        self.description = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "title":
            self._in_title = True
            return
        if tag.lower() != "meta":
            return
        attr_map = {name.lower(): value or "" for name, value in attrs}
        if attr_map.get("name", "").lower() == "description":
            self.description = attr_map.get("content", "")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data


class _VisibleTextParser(HTMLParser):
    SKIP_TAGS = {"canvas", "head", "iframe", "script", "style", "noscript", "template", "svg"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag.lower() in self.SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self.SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        stripped = data.strip()
        if stripped:
            self.parts.append(stripped)


class _MarkdownHTMLParser(HTMLParser):
    SKIP_TAGS = {"canvas", "head", "iframe", "script", "style", "noscript", "template", "svg"}
    BLOCK_TAGS = {
        "address",
        "article",
        "aside",
        "body",
        "dd",
        "details",
        "dialog",
        "div",
        "dl",
        "dt",
        "figcaption",
        "figure",
        "footer",
        "form",
        "header",
        "main",
        "nav",
        "p",
        "section",
    }

    def __init__(self, *, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self._skip_depth = 0
        self._heading_level: int | None = None
        self._quote_depth = 0
        self._pre_depth = 0
        self._list_stack: list[dict[str, int | str]] = []
        self._in_list_item = False
        self._code_depth = 0
        self._current_parts: list[str] = []
        self._link_stack: list[tuple[str, list[str]]] = []
        self._lines: list[str] = []
        self._last_kind = ""

    def markdown(self) -> str:
        self._flush_current()
        while self._lines and self._lines[-1] == "":
            self._lines.pop()
        return "\n".join(self._lines).strip()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        attr_map = {name.lower(): value or "" for name, value in attrs}
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._flush_current()
            self._heading_level = int(tag[1])
            return
        if tag in self.BLOCK_TAGS:
            self._flush_current()
            return
        if tag == "blockquote":
            self._flush_current()
            self._quote_depth += 1
            return
        if tag in {"ul", "ol"}:
            self._flush_current()
            self._list_stack.append({"kind": tag, "next": 1})
            return
        if tag == "li":
            self._flush_current()
            self._in_list_item = True
            return
        if tag == "br":
            self._flush_current()
            return
        if tag == "hr":
            self._flush_current()
            self._emit_line("---", kind="rule")
            return
        if tag == "pre":
            self._flush_current()
            self._pre_depth += 1
            return
        if tag == "code" and not self._pre_depth:
            self._code_depth += 1
            self._append_text("`")
            return
        if tag == "a":
            href = _safe_markdown_link(attr_map.get("href", ""), base_url=self.base_url)
            if href:
                self._link_stack.append((href, self._current_parts))
                self._current_parts = []

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self.SKIP_TAGS:
            if self._skip_depth > 0:
                self._skip_depth -= 1
            return
        if self._skip_depth:
            return
        if tag == "a" and self._link_stack:
            href, previous_parts = self._link_stack.pop()
            link_text = _collapse_inline("".join(self._current_parts))
            self._current_parts = previous_parts
            if link_text:
                self._append_text(f"[{_escape_markdown_link_text(link_text)}]({href})")
            return
        if tag == "code" and not self._pre_depth and self._code_depth:
            self._append_text("`")
            self._code_depth -= 1
            return
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._flush_current()
            self._heading_level = None
            return
        if tag == "li":
            self._flush_current()
            self._in_list_item = False
            return
        if tag in {"ul", "ol"}:
            self._flush_current()
            if self._list_stack:
                self._list_stack.pop()
            return
        if tag == "blockquote":
            self._flush_current()
            if self._quote_depth > 0:
                self._quote_depth -= 1
            return
        if tag == "pre":
            self._flush_current()
            if self._pre_depth > 0:
                self._pre_depth -= 1
            return
        if tag in self.BLOCK_TAGS:
            self._flush_current()

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        self._append_text(data)

    def _append_text(self, value: str) -> None:
        if not value:
            return
        self._current_parts.append(value)

    def _flush_current(self) -> None:
        if not self._current_parts:
            return
        if self._pre_depth:
            text = "".join(self._current_parts).strip("\n")
            self._current_parts = []
            if text.strip():
                self._emit_line(f"```\n{text.rstrip()}\n```", kind="pre")
            return

        text = _collapse_inline("".join(self._current_parts))
        self._current_parts = []
        if not text:
            return
        if self._heading_level is not None:
            self._emit_line(f"{'#' * self._heading_level} {text}", kind="heading")
            return
        if self._in_list_item:
            self._emit_line(self._format_list_item(text), kind="list")
            return
        if self._quote_depth:
            prefix = "> " * self._quote_depth
            self._emit_line(f"{prefix}{text}", kind="quote")
            return
        self._emit_line(text, kind="paragraph")

    def _format_list_item(self, text: str) -> str:
        depth = max(0, len(self._list_stack) - 1)
        indent = "  " * depth
        if not self._list_stack:
            return f"{indent}- {text}"
        state = self._list_stack[-1]
        if state.get("kind") == "ol":
            number = int(state.get("next", 1))
            state["next"] = number + 1
            return f"{indent}{number}. {text}"
        return f"{indent}- {text}"

    def _emit_line(self, line: str, *, kind: str) -> None:
        if not line.strip():
            return
        compact_with_previous = kind == "list" and self._last_kind == "list"
        if self._lines and self._lines[-1] != "" and not compact_with_previous:
            self._lines.append("")
        self._lines.append(line.rstrip())
        self._last_kind = kind


def _collapse_inline(value: str) -> str:
    return " ".join(value.split())
