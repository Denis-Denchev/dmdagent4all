import contextlib
import io
import unittest

from dmdagent4all.cli import main


class CliTest(unittest.TestCase):
    def test_help_imports_and_exits(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as context:
                main(["--help"])
        self.assertEqual(context.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
