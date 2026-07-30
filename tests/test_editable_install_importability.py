"""Regression coverage for editable installs outside the source directory."""

import os
import subprocess
import sys


def test_session_state_modules_import_outside_source_tree(tmp_path):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            (
                "from hermes_cli import kanban_db; "
                "import hermes_state_common; "
                "import hermes_state_portability; "
                "import hermes_state_schema; "
                "import hermes_state_search"
            ),
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
