import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dmdagent4all.memory.manager import MemoryManager, MemoryPathError


class MemoryManagerTest(unittest.TestCase):
    def test_write_read_and_list_memory_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = MemoryManager(Path(tmp))
            manager.write("projects/test.md", "Project note")
            self.assertIn("projects/test.md", manager.list_files())
            self.assertIn("Project note", manager.read("projects/test.md"))

    def test_rejects_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = MemoryManager(Path(tmp))
            with self.assertRaises(MemoryPathError):
                manager.write("../outside.md", "no")

    def test_requires_markdown_extension(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = MemoryManager(Path(tmp))
            with self.assertRaises(MemoryPathError):
                manager.write("bad.txt", "no")

    def test_bootstrap_creates_long_term_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = MemoryManager(Path(tmp))
            manager.bootstrap()

            self.assertIn("long-term/profile.md", manager.list_files())
            self.assertIn("long-term/facts/personal.md", manager.list_files())

    def test_short_term_memory_expires_and_is_purged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = MemoryManager(Path(tmp))
            manager.write(
                "short-term/session.md",
                "Temporary context",
                metadata={"memory_scope": "short-term", "ttl_hours": 1},
            )

            files_before = manager.list_files()
            removed = manager.purge_expired(
                now=datetime.now(timezone.utc) + timedelta(hours=2),
            )

            self.assertIn("short-term/session.md", files_before)
            self.assertEqual(removed, ["short-term/session.md"])
            self.assertNotIn("short-term/session.md", manager.list_files())


if __name__ == "__main__":
    unittest.main()
