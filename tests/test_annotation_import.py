from src.ui_raja.annotation_import import import_annotations
from src.ui_raja.constants import DEFAULT_CVAT_ROOT, DEFAULT_DATA_ROOT
from src.ui_raja.discovery import RajaDataset


def test_import_annotations_uses_mock_fallback_when_root_missing():
    dataset = RajaDataset(DEFAULT_DATA_ROOT)
    session = dataset.sessions_for("S13")[0]

    missing_root = DEFAULT_CVAT_ROOT / "not_here"
    frame, source = import_annotations(session, missing_root)

    assert len(frame) > 0
    assert DEFAULT_CVAT_ROOT in source.parents
