import tempfile
import unittest
from pathlib import Path

import pandas as pd

from experiments.train_cfd_surrogate import (
    infer_crucible_rpm,
    infer_crystal_rpm,
    infer_temperature,
    select_cases,
)


def write_case(path: Path):
    pd.DataFrame(
        {
            "r": [0.0, 0.1],
            "z": [0.0, 0.2],
            "u_r": [0.0, 0.01],
            "u_z": [0.0, 0.02],
            "u_swirl": [0.0, 0.03],
            "p": [0.0, 1.0],
            "T": [1700.0, 1701.0],
        }
    ).to_csv(path, index=False)


class CfdSurrogateDataLoadingTest(unittest.TestCase):
    def test_parameter_inference_supports_corrected_zip_names(self):
        self.assertEqual(infer_temperature(Path("cz_1780.csv")), 1780.0)
        self.assertEqual(infer_temperature(Path("cz_(temp 1780).csv")), 1780.0)
        self.assertEqual(infer_crucible_rpm(Path("CZ_crucible_m3p5rpm_steady.csv")), -3.5)
        self.assertEqual(infer_crystal_rpm(Path("CZ_crystal_07rpm_steady.csv")), 7.0)

    def test_select_cases_supports_extracted_data_zip_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data" / "temperature").mkdir(parents=True)
            (root / "data" / "crucible").mkdir(parents=True)
            (root / "data" / "crystal").mkdir(parents=True)
            write_case(root / "data" / "temperature" / "cz_1775.csv")
            write_case(root / "data" / "temperature" / "cz_1780.csv")
            write_case(root / "data" / "crucible" / "CZ_crucible_m4rpm_steady.csv")
            write_case(root / "data" / "crucible" / "CZ_crucible_m5rpm_steady.csv")
            write_case(root / "data" / "crystal" / "CZ_crystal_06rpm_steady.csv")
            write_case(root / "data" / "crystal" / "CZ_crystal_07rpm_steady.csv")

            self.assertEqual(len(select_cases("temperature", root)), 2)
            self.assertEqual(len(select_cases("crucible", root)), 2)
            self.assertEqual(len(select_cases("crystal", root)), 2)


if __name__ == "__main__":
    unittest.main()
