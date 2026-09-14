import unittest
from pathlib import Path
import subprocess
from unittest.mock import patch

from format_code import changed_lines, owned, replacements, run


class FormattingTests(unittest.TestCase):
    def test_tool_failures_are_not_success(self):
        with patch("format_code.subprocess.check_output", side_effect=FileNotFoundError("formatter missing")):
            with self.assertRaises(FileNotFoundError):
                run(Path.cwd(), "missing")
        with patch("format_code.subprocess.check_output", return_value="clang-format version 18.1"):
            with self.assertRaises(ValueError):
                run(Path.cwd(), "wrong-version")
        with patch("format_code.subprocess.check_output", side_effect=subprocess.CalledProcessError(1, "formatter")):
            with self.assertRaises(subprocess.CalledProcessError):
                run(Path.cwd(), "failing")

    def test_boundaries(self):
        for name in ("include/stable-diffusion.h", "tests/test.cpp", "examples/fashn-preprocess/a.h",
                     "examples/server/a.h", "src/core/a.hpp", "src\\core\\a.cpp"):
            self.assertTrue(owned(name), name)
        for name in ("src/tokenizers/vocab/a.h", "src\\tokenizers\\vocab\\a.cpp",
                     "examples/server/frontend/a.cpp", "ggml/a.h", "thirdparty/a.cpp",
                     "models/a.cpp", "build/a.cpp", "../src/a.cpp", "/src/a.cpp"):
            self.assertFalse(owned(name), name)

    def test_only_changed_lines(self):
        self.assertEqual(changed_lines(b"old\nsame\n", b"new\nsame\n"), {1})
        source = b"bad\nnew\n"
        xml = '<replacements><replacement offset="0" length="3">good</replacement>' \
              '<replacement offset="4" length="3">NEW</replacement></replacements>'
        self.assertEqual(replacements(source, xml, {2}), [(4, 3, b"NEW")])
        self.assertEqual(len(replacements(source, xml)), 2)


if __name__ == "__main__":
    unittest.main()
