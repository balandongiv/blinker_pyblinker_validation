import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.ui_raja.status_store import StatusStore


class StatusStoreTests(unittest.TestCase):
    def test_status_store_persists_and_loads(self) -> None:
        with TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            data_root = base_dir / "data"
            session_path = data_root / "S1" / "session_a" / "ear_eog.fif"
            session_path.parent.mkdir(parents=True, exist_ok=True)
            session_path.touch()

            store_path = base_dir / "status.yaml"
            store = StatusStore(store_path)

            self.assertEqual(store.load_for_root(data_root), {})

            store.update_status(data_root, session_path, "Ongoing")

            reloaded = StatusStore(store_path)
            loaded_status = reloaded.load_for_root(data_root)

            self.assertEqual(loaded_status[session_path.resolve()], "Ongoing")


if __name__ == "__main__":
    unittest.main()
