import unittest

import pandas as pd


from ylivertainen.cleaning import YlivertainenDataCleaningSurg


class DuplicateHelpersTest(unittest.TestCase):
    def make_project(self, df: pd.DataFrame) -> YlivertainenDataCleaningSurg:
        project = YlivertainenDataCleaningSurg.__new__(YlivertainenDataCleaningSurg)
        project.csvs = []
        project.df = df.copy()
        return project

    def test_find_dupes_empty_id_cols_returns_empty_frame(self):
        project = self.make_project(pd.DataFrame({"patient_card_no": ["123", "123"]}))

        result = project.resolve_dupes([])

        self.assertIs(result, project)
        self.assertEqual(len(project.df), 2)

    def test_find_dupes_raises_clear_error_for_missing_id_column(self):
        project = self.make_project(pd.DataFrame({"patient_card_no": ["123"]}))

        with self.assertRaisesRegex(ValueError, r"ID columns not found: \['missing_id'\]"):
            project.resolve_dupes(["missing_id"])

    def test_find_dupes_normalizes_whitespace_and_case(self):
        project = self.make_project(
            pd.DataFrame({"patient_card_no": ["123", " 123 ", "ABC", "abc", "xyz"]})
        )

        returned = project.resolve_dupes(["patient_card_no"], include_first=True)
        _, _, group_dupe_mask = project._resolve_duplicate_masks(["patient_card_no"])
        result = project.df.loc[group_dupe_mask]

        self.assertIs(returned, project)
        self.assertEqual(result["patient_card_no"].tolist(), ["123", " 123 ", "ABC", "abc"])

    def test_find_dupes_ignores_missing_blank_and_whitespace_only_ids(self):
        project = self.make_project(
            pd.DataFrame({"patient_card_no": [pd.NA, "", "   ", pd.NA, "", "x", "X"]})
        )

        returned = project.resolve_dupes(["patient_card_no"], include_first=True)
        _, _, group_dupe_mask = project._resolve_duplicate_masks(["patient_card_no"])
        result = project.df.loc[group_dupe_mask]

        self.assertIs(returned, project)
        self.assertEqual(result["patient_card_no"].tolist(), ["x", "X"])

    def test_find_dupes_include_first_false_returns_only_later_repeats(self):
        project = self.make_project(
            pd.DataFrame({"patient_card_no": ["abc", " ABC ", "def", "DEF", "ghi"]})
        )

        returned = project.resolve_dupes(["patient_card_no"], include_first=False)
        _, later_dupe_mask, _ = project._resolve_duplicate_masks(["patient_card_no"])
        result = project.df.loc[later_dupe_mask]

        self.assertIs(returned, project)
        self.assertEqual(result["patient_card_no"].tolist(), [" ABC ", "DEF"])

    def test_find_dupes_multi_column_ignores_incomplete_ids(self):
        project = self.make_project(
            pd.DataFrame(
                {
                    "patient_card_no": ["001", "001", "002", "002", "003"],
                    "patient_surname": ["Smith", " smith ", pd.NA, "Jones", "Jones"],
                }
            )
        )

        returned = project.resolve_dupes(["patient_card_no", "patient_surname"], include_first=True)
        _, _, group_dupe_mask = project._resolve_duplicate_masks(["patient_card_no", "patient_surname"])
        result = project.df.loc[group_dupe_mask]

        self.assertIs(returned, project)
        self.assertEqual(result["patient_card_no"].tolist(), ["001", "001"])
        self.assertEqual(result["patient_surname"].tolist(), ["Smith", " smith "])

    def test_drop_dupes_keeps_first_complete_key_and_preserves_incomplete_rows(self):
        project = self.make_project(
            pd.DataFrame(
                {
                    "patient_card_no": ["123", " 123 ", pd.NA, "", "abc", "ABC"],
                    "value": [1, 2, 3, 4, 5, 6],
                }
            )
        )

        returned = project.resolve_dupes(["patient_card_no"], drop=True)

        self.assertIs(returned, project)
        self.assertEqual(project.df["patient_card_no"].iloc[0], "123")
        self.assertTrue(pd.isna(project.df["patient_card_no"].iloc[1]))
        self.assertEqual(project.df["patient_card_no"].iloc[2], "")
        self.assertEqual(project.df["patient_card_no"].iloc[3], "abc")
        self.assertEqual(project.df["value"].tolist(), [1, 3, 4, 5])


if __name__ == "__main__":
    unittest.main()
