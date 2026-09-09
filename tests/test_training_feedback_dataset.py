from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPORT_PATH = ROOT / "scripts" / "export_training_feedback_dataset.py"
FIXTURE_PATH = ROOT / "tests" / "test_experiment_database.py"

EXPORT_SPEC = importlib.util.spec_from_file_location("export_training_feedback_dataset_test", EXPORT_PATH)
assert EXPORT_SPEC is not None and EXPORT_SPEC.loader is not None
dataset_exporter = importlib.util.module_from_spec(EXPORT_SPEC)
sys.modules[EXPORT_SPEC.name] = dataset_exporter
EXPORT_SPEC.loader.exec_module(dataset_exporter)

FIXTURE_SPEC = importlib.util.spec_from_file_location("experiment_database_fixture_test", FIXTURE_PATH)
assert FIXTURE_SPEC is not None and FIXTURE_SPEC.loader is not None
experiment_database_fixture = importlib.util.module_from_spec(FIXTURE_SPEC)
sys.modules[FIXTURE_SPEC.name] = experiment_database_fixture
FIXTURE_SPEC.loader.exec_module(experiment_database_fixture)


def test_export_training_feedback_dataset_writes_manifest_and_jsonl(tmp_path: Path) -> None:
    database = experiment_database_fixture.build_fixture_database(tmp_path, include_actions="all")
    output_dir = tmp_path / "dataset"

    report = dataset_exporter.export_training_feedback_dataset(database=database, output_dir=output_dir)

    assert report["counts"]["model_registry"] == 2
    assert report["counts"]["experiment_registry"] == 2
    assert report["counts"]["failure_summary_rows"] == 1
    assert report["counts"]["sample_candidates"] == 1
    assert report["counts"]["extreme_decision_candidates"] == 1
    assert report["counts"]["decision_events"] == 3
    assert report["counts"]["training_asset_records"] == 0
    assert report["counts"]["records"] == 5
    assert report["top_failure_types"][0]["primary_failure_type"] == "execution_click"

    jsonl_path = Path(report["jsonl"])
    lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 5
    records = [json.loads(line) for line in lines]
    assert records[0]["sample_type"] == "failure_attribution"
    assert records[0]["label"] == "policy_loss_review"
    assert records[0]["terminal_edge"] == 1
    assert records[0]["terminal_corner"] == 0
    assert any(record["sample_type"] == "decision_event" for record in records)
    assert any(record["sample_type"] == "extreme_decision_candidate" for record in records)
    assert "training_asset_record" in report["schema"]["sample_types"]
    assert (output_dir / "training_feedback_dataset.md").exists()
