import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class CommandTests(unittest.TestCase):
    def test_help_without_gpu_dependencies(self):
        result = subprocess.run([sys.executable, str(ROOT / "infer.py"), "--help"],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--exr_dtype", result.stdout)

    def test_invalid_scaling_rejected_before_importing_torch(self):
        result = subprocess.run([sys.executable, str(ROOT / "infer.py"),
                                 "--input", "absent.mp4", "--output_dir", "unused",
                                 "--model_scale", "1.5"],
                                capture_output=True, text=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("model_scale must be finite and in (0, 1]", result.stderr)

    def test_output_aliases_and_comma_lists(self):
        # Import is now lightweight; no Torch, NumPy, or OpenCV required.
        import infer
        self.assertEqual(infer._parse_outputs(["raw_fg,alpha", "foreground"]), {"fg", "alpha"})
        self.assertEqual(infer._parse_outputs([]), {"fg", "alpha"})
        with self.assertRaises(ValueError):
            infer._parse_outputs(["unknown"])


if __name__ == "__main__":
    unittest.main()
