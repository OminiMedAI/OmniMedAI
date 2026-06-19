"""Distribution metadata checks."""

import re
import unittest
from pathlib import Path


class TestPackageMetadata(unittest.TestCase):
    def test_distribution_version_matches_import_version(self):
        import omnimedai

        text = Path("pyproject.toml").read_text(encoding="utf-8")
        match = re.search(r'^version = "([^"]+)"$', text, flags=re.MULTILINE)
        self.assertIsNotNone(match)
        self.assertEqual(omnimedai.__version__, match.group(1))


if __name__ == "__main__":
    unittest.main()
