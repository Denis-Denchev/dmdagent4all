from __future__ import annotations

from pathlib import Path
from typing import Any

from dmdcore.security import redact_text
from dmdcore.tools.base import ToolRuntimeContext
from dmdcore.tools.storage import internet_files_dir
from dmdcore.tools.web import assert_public_http_url, normalize_url


def browser_click(args: dict[str, Any], context: ToolRuntimeContext) -> dict[str, Any]:
    url = _required_url(args)
    selector = str(args.get("selector") or "").strip()
    text = str(args.get("text") or "").strip()
    if not selector and not text:
        raise ValueError("browser.click requires selector or text.")

    def action(page: Any, timeout_ms: int) -> None:
        if selector:
            page.locator(selector).first.click(timeout=timeout_ms)
        else:
            page.get_by_text(text).first.click(timeout=timeout_ms)
        _settle_page(page, timeout_ms)

    return _run_browser_action("click", url, context, action)


def browser_fill_form(args: dict[str, Any], context: ToolRuntimeContext) -> dict[str, Any]:
    url = _required_url(args)
    fields = args.get("fields")
    if not isinstance(fields, dict) or not fields:
        raise ValueError("browser.fill_form requires fields object.")

    def action(page: Any, timeout_ms: int) -> None:
        _fill_fields(page, fields, timeout_ms)

    return _run_browser_action("fill_form", url, context, action)


def browser_submit(args: dict[str, Any], context: ToolRuntimeContext) -> dict[str, Any]:
    url = _required_url(args)
    fields = args.get("fields")
    selector = str(args.get("selector") or args.get("submit_selector") or "").strip()

    def action(page: Any, timeout_ms: int) -> None:
        if isinstance(fields, dict) and fields:
            _fill_fields(page, fields, timeout_ms)
        if selector:
            page.locator(selector).first.click(timeout=timeout_ms)
        else:
            page.locator("form").first.evaluate(
                "form => form.requestSubmit ? form.requestSubmit() : form.submit()"
            )
        _settle_page(page, timeout_ms)

    return _run_browser_action("submit", url, context, action)


def _run_browser_action(
    action_name: str,
    raw_url: str,
    context: ToolRuntimeContext,
    action: Any,
) -> dict[str, Any]:
    url = normalize_url(raw_url)
    assert_public_http_url(url)
    runtime = _load_playwright()
    if runtime is None:
        return _runtime_unavailable(url)

    sync_playwright, playwright_timeout_error = runtime
    browser_config = context.config.get("browser", {})
    timeout_ms = int(browser_config.get("timeout_seconds", 15)) * 1000
    profile_dir = _browser_profile_dir(context)
    downloads_dir = internet_files_dir(context.config, default_root=context.workspace_root, folder="browser-downloads")
    profile_dir.mkdir(parents=True, exist_ok=True)
    downloads_dir.mkdir(parents=True, exist_ok=True)

    try:
        with sync_playwright() as playwright:
            browser_context = playwright.chromium.launch_persistent_context(
                user_data_dir=str(profile_dir),
                headless=True,
                accept_downloads=bool(browser_config.get("downloads_to_workspace", True)),
                downloads_path=str(downloads_dir),
            )
            try:
                page = browser_context.new_page()
                response = page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
                assert_public_http_url(page.url)
                action(page, timeout_ms)
                assert_public_http_url(page.url)
                return {
                    "status": "ok",
                    "action": action_name,
                    "url": url,
                    "final_url": page.url,
                    "http_status": None if response is None else response.status,
                    "title": redact_text(page.title()),
                    "profile": str(profile_dir),
                    "downloads": str(downloads_dir),
                }
            finally:
                browser_context.close()
    except playwright_timeout_error as exc:
        raise ValueError(f"Browser action timed out: {exc}") from exc
    except RuntimeError as exc:
        return _runtime_unavailable(url, detail=str(exc))


def _load_playwright():
    try:
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
        from playwright.sync_api import sync_playwright
    except ModuleNotFoundError:
        return None
    return sync_playwright, PlaywrightTimeoutError


def _runtime_unavailable(url: str, *, detail: str = "") -> dict[str, Any]:
    message = (
        "Browser automation runtime is unavailable. Install optional browser support "
        "with `pip install -e .[browser]` and `python -m playwright install chromium`."
    )
    if detail:
        message = f"{message} Runtime detail: {redact_text(detail)}"
    return {
        "status": "not_implemented",
        "message": message,
        "url": url,
    }


def _required_url(args: dict[str, Any]) -> str:
    raw = str(args.get("url") or "").strip()
    if not raw:
        raise ValueError("browser automation tools require url.")
    return raw


def _browser_profile_dir(context: ToolRuntimeContext) -> Path:
    return context.workspace_root / "browser-profile"


def _fill_fields(page: Any, fields: dict[str, Any], timeout_ms: int) -> None:
    for selector, value in fields.items():
        page.locator(str(selector)).first.fill(str(value), timeout=timeout_ms)


def _settle_page(page: Any, timeout_ms: int) -> None:
    try:
        page.wait_for_load_state("domcontentloaded", timeout=timeout_ms)
    except Exception:
        pass
