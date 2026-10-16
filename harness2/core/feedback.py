# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Optional human feedback on a task: one Markdown file per task in a feedback directory.

    <feedback-dir>/<task-slug>.md      free text from the person running the sweep

Off unless a directory is passed. The text reaches the selector and the experience session,
never the proposers. It is a hint toward the answer, so benchmark runs leave it off; the
grader-verdict tripwire still applies to it, and feedback carrying a score is refused.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from . import context
from .library import slug


def path_for(feedback_dir: Path | str, task_id: str) -> Path:
    return Path(feedback_dir) / f"{slug(task_id)}.md"


def read(feedback_dir: Path | str | None, task_id: str) -> str:
    """The task's feedback text, or "" when there is no directory, no file, or an empty one."""
    if feedback_dir is None:
        return ""
    path = path_for(feedback_dir, task_id)
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8").strip()
    for marker in context.GRADER_VERDICT_MARKERS:
        if marker in text:
            raise ValueError(
                f"{path} contains grader-verdict marker {marker!r}; feedback is text for the "
                f"selector and the reflector, not a score"
            )
    return text


def fingerprint(text: str) -> str:
    """Recorded on the summary so a resumed task re-runs when its feedback changes."""
    return hashlib.sha1(text.encode()).hexdigest()[:16] if text else ""
