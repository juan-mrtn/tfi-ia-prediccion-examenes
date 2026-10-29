"""Tests de DataPreprocessor."""

from pathlib import Path

import pandas as pd
import pytest

from src.preprocessor import DataPreprocessor

ROOT = Path(__file__).resolve().parents[1]
METADATA = ROOT / "data" / "metadata" / "student_metadata.csv"


@pytest.fixture
def prep():
    return DataPreprocessor(str(METADATA))


def _row(**kw):
    base = dict(study_hours=7, attendance=56, sleep_hours=8, internet_usage=7,
                assignments_completed=13, previous_score=62, exam_score=88.5)
    base.update(kw)
    return base


def test_feature_cols(prep):
    assert prep.feature_cols == ["study_hours", "attendance", "internet_usage",
                                 "assignments_completed", "previous_score"]
    assert prep.target_col == "exam_score"


def test_hourly_check_drops_invalid_rows(prep):
    df = pd.DataFrame([_row(), _row(sleep_hours=10, study_hours=15)])
    out = prep.validate_hourly_consistency(df)
    assert len(out) == 1
    assert prep.invalid_rows_ == 1


def test_only_tasks_are_scaled(prep):
    out = prep.scale_to_base_100_int(pd.DataFrame([_row()]))
    assert out.loc[0, "assignments_completed"] == 65
    assert out.loc[0, "study_hours"] == 7
    assert out.loc[0, "attendance"] == 56


def test_scaled_values_are_int_in_range(prep):
    df = pd.DataFrame([_row(assignments_completed=v) for v in range(0, 21)])
    out = prep.scale_to_base_100_int(df)
    for col in prep.feature_cols:
        assert pd.api.types.is_integer_dtype(out[col])
    assert out["assignments_completed"].between(0, 100).all()


def test_train_test_split_80_20(prep):
    df = pd.DataFrame([_row(previous_score=i) for i in range(100)])
    train, test = prep.train_test_split(df)
    assert len(train) == 80 and len(test) == 20
    assert set(train.previous_score).isdisjoint(test.previous_score)


def test_save_csv_three_digits(prep, tmp_path):
    out = prep.transform(pd.DataFrame([_row()]))
    path = tmp_path / "out.csv"
    prep.save_csv(out, str(path))
    text = path.read_text().splitlines()[1]
    assert text.startswith("007,056,007,065,062,88.5")


def test_missing_column_raises(prep, tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame([{"study_hours": 1}]).to_csv(path, index=False)
    with pytest.raises(ValueError):
        prep.load(str(path))


def test_dataset_2_metadata():
    prep2 = DataPreprocessor(str(ROOT / "data" / "metadata" / "factors_metadata.csv"))
    assert prep2.feature_cols == ["Hours_Studied", "Attendance", "Sleep_Hours",
                                  "Tutoring_Sessions", "Previous_Scores"]
    assert prep2.target_col == "Exam_Score"
    fila = pd.DataFrame([dict(Hours_Studied=22, Attendance=80, Sleep_Hours=7,
                              Tutoring_Sessions=2, Previous_Scores=75, Exam_Score=67)])
    out = prep2.scale_to_base_100_int(fila)
    # Horas por semana (máx. 44) y tutorías (máx. 8) se escalan a 100; el resto no.
    assert out.loc[0, "Hours_Studied"] == 50
    assert out.loc[0, "Tutoring_Sessions"] == 25
    assert out.loc[0, "Sleep_Hours"] == 7


def test_dataset_2_hourly_check_uses_sleep_only():
    prep2 = DataPreprocessor(str(ROOT / "data" / "metadata" / "factors_metadata.csv"))
    df = pd.DataFrame({"Sleep_Hours": [8, 25], "Hours_Studied": [40, 40]})
    assert len(prep2.validate_hourly_consistency(df)) == 1
