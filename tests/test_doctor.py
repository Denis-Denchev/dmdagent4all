import tempfile
import unittest
from pathlib import Path

from dmdcore.app_paths import AppPaths
from dmdcore.config import DEFAULT_CONFIG
from dmdcore.doctor import doctor_summary, run_doctor


class DoctorTest(unittest.TestCase):
    def test_doctor_flags_direct_llm_secret_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = AppPaths(
                root=root,
                config=root / "config.yaml",
                memory=root / "memory",
                workspace=root / "workspace",
                logs=root / "logs",
                audit_db=root / "audit.db",
                vector_index=root / "vector_index",
                tokens_marker=root / "tokens.enc",
            )
            paths.ensure()
            config = {
                **DEFAULT_CONFIG,
                "llm": {
                    **DEFAULT_CONFIG["llm"],
                    "provider": "openai",
                    "api_key": "do-not-store-this",
                },
            }

            checks = run_doctor(paths=paths, config=config, check_network=False)

            self.assertGreaterEqual(doctor_summary(checks)["fail"], 1)
            self.assertTrue(
                any("direct secret field" in check.message for check in checks)
            )


if __name__ == "__main__":
    unittest.main()
