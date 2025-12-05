from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.ui_raja.path_pairs import load_path_pairs


class LoadPathPairsTests(unittest.TestCase):
    def test_relative_paths_resolved_against_config_dir(self) -> None:
        with TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            config_dir = base_dir / "config"
            data_dir = base_dir / "data_root"
            cvat_dir = base_dir / "cvat_root"
            other_dir = base_dir / "elsewhere"

            config_dir.mkdir()
            data_dir.mkdir()
            cvat_dir.mkdir()
            other_dir.mkdir()

            config_path = config_dir / "pairs.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "pairs:",
                        "  - name: Relative paths",
                        "    data_root: data_root",
                        "    cvat_root: cvat_root",
                    ]
                ),
                encoding="utf8",
            )

            original_cwd = Path.cwd()
            os.chdir(other_dir)
            try:
                pairs = load_path_pairs(config_path)
            finally:
                os.chdir(original_cwd)

            self.assertEqual(len(pairs), 1)
            pair = pairs[0]
            self.assertEqual(pair.data_root, data_dir.resolve())
            self.assertEqual(pair.cvat_root, cvat_dir.resolve())


if __name__ == "__main__":
    unittest.main()
