import tempfile
import unittest
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


if __name__ == "__main__":
    unittest.main()
