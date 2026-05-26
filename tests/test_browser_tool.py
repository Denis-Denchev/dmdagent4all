import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dmdcore.tools.base import ToolRuntimeContext
from dmdcore.tools.browser_automation import browser_click
from dmdcore.tools.web import (
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
            with mock.patch("dmdcore.tools.web.fetch_page", return_value=page):
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
            with mock.patch("dmdcore.tools.web.fetch_page", return_value=page):
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
        self.assertEqual(result["scrape_mode"], "raw_page")
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

    def test_browser_scrape_markdown_uses_meta_description_when_body_has_no_text(self) -> None:
        html = """
        <html>
          <head>
            <title>JS App</title>
            <meta name="description" content="Useful fallback summary">
          </head>
          <body><div id="root"></div><script src="/app.js"></script></body>
        </html>
        """
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
                body=html,
                bytes_read=256,
                truncated=False,
            )
            with mock.patch("dmdcore.tools.web.fetch_page", return_value=page):
                result = browser_scrape_markdown({"url": "https://example.com"}, context)

        self.assertIn("# JS App", result["markdown"])
        self.assertIn("Useful fallback summary", result["markdown"])
        self.assertNotIn("_No readable page content extracted._", result["markdown"])

    def test_browser_scrape_markdown_honors_configured_downloads_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            downloads_root = Path(tmp) / "internet-files"
            context = ToolRuntimeContext(
                memory_root=Path(tmp) / "memory",
                workspace_root=Path(tmp) / "workspace",
                config={
                    "browser": {"max_text_chars": 12000},
                    "storage": {"downloads_root": str(downloads_root)},
                },
            )
            page = FetchedPage(
                url="https://example.com",
                final_url="https://example.com",
                status=200,
                content_type="text/html",
                body="<html><head><title>Saved Page</title></head><body><p>Body</p></body></html>",
                bytes_read=128,
                truncated=False,
            )
            with mock.patch("dmdcore.tools.web.fetch_page", return_value=page):
                result = browser_scrape_markdown({"url": "https://example.com"}, context)

        self.assertTrue(str(result["path"]).startswith(str(downloads_root.resolve())))
        self.assertEqual(result["relative_path"], "scrapefiles/saved-page.md")

    def test_targeted_article_scrape_outputs_only_requested_articles(self) -> None:
        html = """
        <html>
          <head><title>Daily Example</title></head>
          <body>
            <nav class="main-menu">
              <a href="/league-1">Premier League category menu</a>
              <a href="/teams">Teams index</a>
            </nav>
            <main>
              <article>
                <h2><a href="/news/first-real-story-1001">First real article title today</a></h2>
                <p>First article summary from the homepage card.</p>
              </article>
              <article>
                <h2><a href="/news/second-real-story-1002">Second real article headline here</a></h2>
                <p>Second article summary from the homepage card.</p>
              </article>
              <article>
                <h2><a href="/news/third-real-story-1003">Third real article headline here</a></h2>
                <p>Third article summary from the homepage card.</p>
              </article>
              <article>
                <h2><a href="/news/fourth-real-story-1004">Fourth real article headline here</a></h2>
                <p>Fourth article summary should not be included.</p>
              </article>
            </main>
          </body>
        </html>
        """
        with tempfile.TemporaryDirectory() as tmp:
            context = ToolRuntimeContext(
                memory_root=Path(tmp) / "memory",
                workspace_root=Path(tmp) / "workspace",
                config={"browser": {"max_text_chars": 12000}},
            )
            page = FetchedPage(
                url="https://news.example.test",
                final_url="https://news.example.test/",
                status=200,
                content_type="text/html",
                body=html,
                bytes_read=2048,
                truncated=False,
            )
            with mock.patch("dmdcore.tools.web.fetch_page", return_value=page):
                result = browser_scrape_markdown(
                    {
                        "url": "https://news.example.test",
                        "mode": "targeted",
                        "content_type": "articles",
                        "limit": 3,
                        "format": "clean_markdown",
                        "instructions": "Extract only the first 3 articles from the homepage.",
                    },
                    context,
                )

        markdown = result["markdown"]
        self.assertEqual(result["scrape_mode"], "targeted")
        self.assertEqual(result["extracted_count"], 3)
        self.assertEqual(markdown.count("- Link:"), 3)
        self.assertIn("## 1. First real article title today", markdown)
        self.assertIn("https://news.example.test/news/first-real-story-1001", markdown)
        self.assertIn("## 3. Third real article headline here", markdown)
        self.assertNotIn("Fourth real article headline here", markdown)
        self.assertNotIn("Premier League category menu", markdown)
        self.assertNotIn("Teams index", markdown)

    def test_targeted_article_scrape_can_extract_json_ld_articles(self) -> None:
        html = """
        <html>
          <head>
            <title>JSON News</title>
            <script type="application/ld+json">
            [
              {
                "@type": "NewsArticle",
                "headline": "JSON-LD article one headline",
                "url": "/news/json-one-1001",
                "description": "JSON one summary."
              },
              {
                "@type": "NewsArticle",
                "headline": "JSON-LD article two headline",
                "mainEntityOfPage": {"@id": "/news/json-two-1002"}
              }
            ]
            </script>
          </head>
          <body><main><a href="/category">Category link</a></main></body>
        </html>
        """
        with tempfile.TemporaryDirectory() as tmp:
            context = ToolRuntimeContext(
                memory_root=Path(tmp) / "memory",
                workspace_root=Path(tmp) / "workspace",
                config={"browser": {"max_text_chars": 12000}},
            )
            page = FetchedPage(
                url="https://another.example.test",
                final_url="https://another.example.test/",
                status=200,
                content_type="text/html",
                body=html,
                bytes_read=1024,
                truncated=False,
            )
            with mock.patch("dmdcore.tools.web.fetch_page", return_value=page):
                result = browser_scrape_markdown(
                    {
                        "url": "https://another.example.test",
                        "mode": "targeted",
                        "content_type": "articles",
                        "limit": 2,
                    },
                    context,
                )

        self.assertEqual(result["scrape_mode"], "targeted")
        self.assertEqual(result["extracted_count"], 2)
        self.assertIn("JSON-LD article one headline", result["markdown"])
        self.assertIn("https://another.example.test/news/json-one-1001", result["markdown"])

    def test_targeted_article_failure_is_marked_as_raw_fallback(self) -> None:
        html = """
        <html>
          <head><title>Directory Page</title></head>
          <body>
            <nav><a href="/league-1">League directory</a></nav>
            <main><h1>Directory only</h1><p>No article cards here.</p></main>
          </body>
        </html>
        """
        with tempfile.TemporaryDirectory() as tmp:
            context = ToolRuntimeContext(
                memory_root=Path(tmp) / "memory",
                workspace_root=Path(tmp) / "workspace",
                config={"browser": {"max_text_chars": 12000}},
            )
            page = FetchedPage(
                url="https://directory.example.test",
                final_url="https://directory.example.test/",
                status=200,
                content_type="text/html",
                body=html,
                bytes_read=512,
                truncated=False,
            )
            with mock.patch("dmdcore.tools.web.fetch_page", return_value=page):
                result = browser_scrape_markdown(
                    {
                        "url": "https://directory.example.test",
                        "mode": "targeted",
                        "content_type": "articles",
                        "limit": 3,
                    },
                    context,
                )

        self.assertEqual(result["scrape_mode"], "raw_page")
        self.assertTrue(result["raw_fallback"])
        self.assertTrue(result["targeted_failed"])
        self.assertEqual(result["extracted_count"], 0)
        self.assertIn("Targeted Extraction Failed", result["markdown"])
        self.assertIn("raw page fallback", result["markdown"])
        self.assertIn("Directory only", result["markdown"])

    def test_browser_click_reports_missing_optional_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            context = ToolRuntimeContext(
                memory_root=Path(tmp) / "memory",
                workspace_root=Path(tmp) / "workspace",
                config={"browser": {"timeout_seconds": 1}},
            )
            with mock.patch("dmdcore.tools.browser_automation._load_playwright", return_value=None):
                result = browser_click(
                    {"url": "https://example.com", "selector": "a"},
                    context,
                )

        self.assertEqual(result["status"], "not_implemented")
        self.assertIn("pip install -e .[browser]", result["message"])


if __name__ == "__main__":
    unittest.main()
