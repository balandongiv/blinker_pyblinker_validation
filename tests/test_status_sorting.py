import unittest

from src.ui_raja.app import status_sort_key, subject_sort_key


class StatusSortingTests(unittest.TestCase):
    def test_subject_sort_key_numeric_order(self) -> None:
        subjects = ["S1", "S10", "S2", "S3", "S11", "S9"]
        sorted_subjects = sorted(subjects, key=subject_sort_key)
        self.assertEqual(sorted_subjects, ["S1", "S2", "S3", "S9", "S10", "S11"])

    def test_status_sort_key_respects_defined_order(self) -> None:
        statuses = ["Complete", "Pending", "Ongoing"]
        sorted_statuses = sorted(statuses, key=status_sort_key)
        self.assertEqual(sorted_statuses, ["Pending", "Ongoing", "Complete"])


if __name__ == "__main__":
    unittest.main()
