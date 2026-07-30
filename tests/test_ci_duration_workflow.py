from __future__ import annotations

from pathlib import Path


def test_duration_artifacts_are_isolated_before_json_merge() -> None:
    workflow = (Path(__file__).parents[1] / ".github" / "workflows" / "tests.yml").read_text(
        encoding="utf-8"
    )
    duration_job = workflow.split("  save-durations:", 1)[1]

    assert "merge-multiple: true" not in duration_job
    assert "glob.glob('durations/**/test_durations.json', recursive=True)" in duration_job
