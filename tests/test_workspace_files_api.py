import tempfile
import unittest
from pathlib import Path

from fastapi import HTTPException

from dmdagent4all.app_paths import AppPaths
from dmdagent4all.server import (
    _configuration_response,
    _read_workspace_text_preview,
    _resolve_workspace_file,
    _update_agent_configuration,
    _workspace_files_response,
    AgentConfigurationRequest,
)


class WorkspaceFilesApiTest(unittest.TestCase):
    def test_lists_and_reads_only_allowed_download_folders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dmdagent4all"
            paths_config = AppPaths(
                root=root,
                config=root / "config.yaml",
                memory=root / "memory",
                workspace=root / "workspace",
                logs=root / "logs",
                audit_db=root / "audit.db",
                vector_index=root / "vector_index",
                tokens_marker=root / "tokens.enc",
            )
            paths_config.ensure()
            scrape_dir = root / "workspace" / "scrapefiles"
            downloads_dir = root / "workspace" / "browser-downloads"
            scrape_dir.mkdir(parents=True, exist_ok=True)
            downloads_dir.mkdir(parents=True, exist_ok=True)
            (scrape_dir / "page.md").write_text("# Page\n\nScraped body", encoding="utf-8")
            (downloads_dir / "report.txt").write_text("downloaded text", encoding="utf-8")
            (root / "workspace" / "private.md").write_text("outside allowed roots", encoding="utf-8")

            listed = _workspace_files_response(paths_config, {})
            preview_path = _resolve_workspace_file(paths_config, {}, "scrapefiles/page.md")
            preview_content, preview_truncated = _read_workspace_text_preview(preview_path)

        self.assertEqual(listed["workspace"], str((root / "workspace").resolve()))
        paths = {item["path"] for item in listed["files"]}
        self.assertEqual(paths, {"scrapefiles/page.md", "browser-downloads/report.txt"})
        self.assertEqual(preview_content, "# Page\n\nScraped body")
        self.assertFalse(preview_truncated)
        with self.assertRaises(HTTPException):
            _resolve_workspace_file(paths_config, {}, "private.md")
        with self.assertRaises(HTTPException):
            _resolve_workspace_file(paths_config, {}, "../config.yaml")

    def test_downloads_root_can_be_configured(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dmdagent4all"
            custom_downloads = Path(tmp) / "internet-files"
            paths_config = AppPaths(
                root=root,
                config=root / "config.yaml",
                memory=root / "memory",
                workspace=root / "workspace",
                logs=root / "logs",
                audit_db=root / "audit.db",
                vector_index=root / "vector_index",
                tokens_marker=root / "tokens.enc",
            )
            paths_config.ensure()
            config: dict = {}

            _update_agent_configuration(
                config,
                AgentConfigurationRequest(downloads_root=str(custom_downloads)),
            )
            (custom_downloads / "scrapefiles").mkdir(parents=True, exist_ok=True)
            (custom_downloads / "scrapefiles" / "custom.md").write_text("custom", encoding="utf-8")

            listed = _workspace_files_response(paths_config, config)

        self.assertEqual(listed["workspace"], str(custom_downloads.resolve()))
        self.assertEqual([item["path"] for item in listed["files"]], ["scrapefiles/custom.md"])

    def test_system_prompt_overrides_are_exposed_with_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dmdagent4all"
            paths_config = AppPaths(
                root=root,
                config=root / "config.yaml",
                memory=root / "memory",
                workspace=root / "workspace",
                logs=root / "logs",
                audit_db=root / "audit.db",
                vector_index=root / "vector_index",
                tokens_marker=root / "tokens.enc",
            )
            paths_config.ensure()
            config: dict = {}

            _update_agent_configuration(
                config,
                AgentConfigurationRequest(planner_system_prompt="Custom planner prompt"),
            )
            response = _configuration_response(paths_config, config)

        self.assertEqual(response["system_prompts"]["planner"]["custom"], "Custom planner prompt")
        self.assertEqual(response["system_prompts"]["planner"]["effective"], "Custom planner prompt")
        self.assertTrue(response["system_prompts"]["planner"]["customized"])
        self.assertIn("personal assistant", response["system_prompts"]["answer"]["default"])


if __name__ == "__main__":
    unittest.main()
