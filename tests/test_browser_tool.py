import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dmdagent4all.tools.base import ToolRuntimeContext
from dmdagent4all.tools.browser_automation import browser_click
from dmdagent4all.tools.web import (
    FetchedPage,
    assert_public_http_url,
    browser_extract_text,
    browser_scrape_markdown,
    extract_metadata,
    extract_readable_text,
)


class BrowserToolTest(unittest.TestCase):
    def test_blocks_local_and_private_addresses(self) -> None:
        for url in ["http://localhost", "http://127.0.0.1", "http://10.0.0.1"]:
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    assert_public_http_url(url)

    def test_extracts_metadata_and_visible_text(self) -> None:
        html = """
        <html>
          <head>
            <title>Example Page</title>
            <meta name="description" content="Short description">
            <script>secret()</script>
          </head>
          <body><h1>Hello</h1><p>Visible text</p></body>
        </html>
        """

        metadata = extract_metadata(html, "text/html")
        text = extract_readable_text(html, "text/html")

        self.assertEqual(metadata["title"], "Example Page")
        self.assertEqual(metadata["description"], "Short description")
        self.assertIn("Hello Visible text", text)
        self.assertNotIn("secret", text)

    def test_browser_extract_text_uses_fetch_guard_and_redacts_secret_like_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            context = ToolRuntimeContext(
                memory_root=Path(tmp) / "memory",
                workspace_root=Path(tmp) / "workspace",
                config={"browser": {"max_text_chars": 12000}},
            )
            page = FetchedPage(
                url="https://example.com",
                final_url="https://example.com",
                status=200,
                content_type="text/html",
                body="<html><body>api_key=abcdefghijklmnop secret body</body></html>",
                bytes_read=64,
                truncated=False,
            )
            with mock.patch("dmdagent4all.tools.web.fetch_page", return_value=page):
                result = browser_extract_text({"url": "https://example.com"}, context)

        self.assertIn("api_key=[REDACTED_SECRET]", result["text"])
        self.assertNotIn("abcdefghijklmnop", result["text"])

    def test_browser_scrape_markdown_saves_markdown_and_redacts_secret_like_output(self) -> None:
        html = """
        <html>
          <head><title>Example Page</title></head>
          <body>
            <main>
              <h1>Launch Notes</h1>
              <p>Read the <a href="/docs">docs</a>.</p>
              <ul>
                <li>Alpha item</li>
                <li>Beta item api_key=abcdefghijklmnop</li>
              </ul>
              <script>hiddenSecret()</script>
            </main>
          </body>
        </html>
        """
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            context = ToolRuntimeContext(
                memory_root=Path(tmp) / "memory",
                workspace_root=workspace,
                config={"browser": {"max_text_chars": 12000}},
            )
            page = FetchedPage(
                url="https://example.com/posts",
                final_url="https://example.com/posts",
                status=200,
                content_type="text/html",
                body=html,
                bytes_read=512,
                truncated=False,
            )
            with mock.patch("dmdagent4all.tools.web.fetch_page", return_value=page):
                result = browser_scrape_markdown(
                    {
                        "url": "https://example.com/posts",
                        "instructions": "Only capture launch notes.",
                        "filename": "launch.md",
                    },
                    context,
                )

            output_path = Path(result["path"])
            saved_markdown = output_path.read_text(encoding="utf-8")

        self.assertEqual(result["relative_path"], "scrapefiles/launch.md")
        self.assertEqual(result["markdown"], saved_markdown)
        self.assertIn("# Example Page", saved_markdown)
        self.assertIn("## Scrape Instructions", saved_markdown)
        self.assertIn("Only capture launch notes.", saved_markdown)
        self.assertIn("## Content", saved_markdown)
        self.assertIn("# Launch Notes", saved_markdown)
        self.assertIn("[docs](https://example.com/docs)", saved_markdown)
        self.assertIn("- Alpha item", saved_markdown)
        self.assertIn("api_key=[REDACTED_SECRET]", saved_markdown)
        self.assertNotIn("abcdefghijklmnop", saved_markdown)
        self.assertNotIn("hiddenSecret", saved_markdown)

    def test_browser_click_reports_missing_optional_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            context = ToolRuntimeContext(
                memory_root=Path(tmp) / "memory",
                workspace_root=Path(tmp) / "workspace",
                config={"browser": {"timeout_seconds": 1}},
            )
            with mock.patch("dmdagent4all.tools.browser_automation._load_playwright", return_value=None):
                result = browser_click(
                    {"url": "https://example.com", "selector": "a"},
                    context,
                )

        self.assertEqual(result["status"], "not_implemented")
        self.assertIn("pip install -e .[browser]", result["message"])


if __name__ == "__main__":
    unittest.main()
