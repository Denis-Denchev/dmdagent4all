from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from dmdagent4all.security import redact_text
from dmdagent4all.tools.base import ToolRuntimeContext


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
    SKIP_TAGS = {"script", "style", "noscript", "template", "svg"}

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
