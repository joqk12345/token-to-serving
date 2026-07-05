import subprocess
import sys
import unittest


class ValidateBookWorkspaceTest(unittest.TestCase):
    def test_validate_book_workspace(self):
        result = subprocess.run(
            [sys.executable, "scripts/validate_book_workspace.py"],
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
