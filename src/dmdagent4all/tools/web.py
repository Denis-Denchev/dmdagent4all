from __future__ import annotations

import ipaddress
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
    content = html_to_markdown(page.body, page.content_type, base_url=page.final_url)
    if not content.strip():
        content = _metadata_fallback_markdown(metadata)
    markdown = _build_scrape_markdown(
        title=title,
        source_url=page.url,
        final_url=page.final_url,
        status=page.status,
        content_type=page.content_type,
        scraped_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        instructions=instructions,
        content=content,
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
) -> str:
    lines = [
        "---",
        f'source_url: "{_escape_yaml_scalar(source_url)}"',
        f'final_url: "{_escape_yaml_scalar(final_url)}"',
        f'scraped_at: "{_escape_yaml_scalar(scraped_at)}"',
        f"status: {status}",
        f'content_type: "{_escape_yaml_scalar(content_type)}"',
        "---",
        "",
        f"# {_escape_markdown_heading(title or 'Scraped Page')}",
        "",
    ]
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
