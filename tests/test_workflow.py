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

"""Focused regressions for the paper's execution, evidence, and adoption contracts."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from harness2 import config
from harness2.core import context, feedback, judge, recursion, schedule, tree
from harness2.core.library import Library
from harness2.core.proposer import Proposer, ProposerError, Reply
from harness2.core.spec import resolve_channel_model
from harness2.demo import BASE_SCRIPT, DemoAdapter, DemoProposer


class MissionLog(DemoProposer):
    """DemoProposer that keeps (label, context dir, mission) for every session."""

    def call(self, mission, context_dir, label, **kwargs):
        self.__dict__.setdefault("missions", []).append((label, str(context_dir), mission))
        return super().call(mission, context_dir, label, **kwargs)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.adapter = DemoAdapter(self.root)
        self.proposer = DemoProposer(self.adapter.spec.channels["proposer"])

    def run_stream(self, k=2, proposer=None, feedback_dir=None):
        return schedule.run_domain(
            "accounting",
            self.adapter.tasks(),
            k,
            adapter=self.adapter,
            prop=proposer or self.proposer,
            runs_root=self.adapter.spec.runs_root,
            library_root=self.root / "library" / str(k),
            feedback_dir=feedback_dir,
        )

    def test_recursion_budget_and_resume(self):
        with patch.dict(config.PROPOSER_MODEL_BY_CHANNEL, {"opencode": "", "claude_cli": ""}):
            unresolved = resolve_channel_model(None, None)
            self.assertEqual(unresolved.model, "")
            with self.assertRaises(ValueError):
                Proposer(unresolved)
            with self.assertRaises(ValueError):
                resolve_channel_model("opencode", "")
        for k in (1, 2):
            with self.subTest(k=k):
                result = self.run_stream(k)
                self.assertEqual((result["done"], result["failed"]), (2, 0))
                for task_id in self.adapter.tasks():
                    summary = json.loads(
                        (
                            self.adapter.spec.runs_root
                            / f"demo-oc-agg{k}"
                            / task_id
                            / "summary.json"
                        ).read_text()
                    )
                    for mode in ("agg", "seq"):
                        ledger = summary["modes"][mode]["ledger"]
                        self.assertEqual((ledger["R"], ledger["P"]), (4 * k, 3 * k))
                        selection = summary["selections"][mode]
                        expected = "Total: 85.15\n" if task_id == "ledger-1" else "Total: 20.25\n"
                        self.assertEqual(
                            self.adapter.deliverable_text(Path(selection["chosen_run_dir"])),
                            expected,
                        )
                executions, calls = self.adapter.execution_count, len(self.proposer._calls)
                resumed = self.run_stream(k)
                self.assertEqual(
                    (self.adapter.execution_count, len(self.proposer._calls)), (executions, calls)
                )
                self.assertEqual(
                    [row["status"] for row in resumed["tasks"]], ["skipped", "skipped"]
                )
        self.assertFalse(
            any(
                reply.label.endswith(".agg1") or reply.label.endswith(".seq1")
                for reply in self.proposer._calls
                if reply.label.startswith("select.")
            )
        )
        work = self.adapter.spec.runs_root / "demo-oc-seq2" / "ledger-1"
        summary_path = self.adapter.spec.runs_root / "demo-oc-agg2/ledger-1/summary.json"
        chosen = json.loads(summary_path.read_text())["selections"]["seq"]
        (work / "harnesses" / chosen["chosen_label"] / "scripts/summarize.py").write_text(
            "changed after selection\n"
        )
        repaired = self.run_stream()
        self.assertEqual([row["status"] for row in repaired["tasks"]], ["ok", "skipped"])
        chosen = json.loads(summary_path.read_text())["selections"]["agg"]
        (Path(chosen["chosen_run_dir"]) / "answer.txt").unlink()
        repaired = self.run_stream()
        self.assertEqual([row["status"] for row in repaired["tasks"]], ["ok", "skipped"])

    def test_complete_evidence_and_library_prefix(self):
        (self.adapter.task_files("ledger-1") / "reference_answer.txt").write_text('"pass_rate": 1')
        self.run_stream()
        work = self.adapter.spec.runs_root / "demo-oc-seq2" / "ledger-1"
        reflection = work / "reflection/seq2"
        required = [
            "source/ledger.csv",
            "rollouts/base1.md",
            "rollouts/c2.md",
            "pool/inv1/harness.diff",
            "decisions/c1.md",
            "adoption.json",
            "base_harness/scripts/summarize.py",
            "harness/scripts/summarize.py",
        ]
        self.assertEqual([name for name in required if not (reflection / name).is_file()], [])
        self.assertTrue((work / "context/inv2/pool/c1/rollout.md").is_file())
        self.assertEqual(list(work.rglob("reference_answer.txt")), [])
        next_context = work.parent / "ledger-2/context/inv1/library/harnesses"
        self.assertEqual([path.name for path in next_context.iterdir()], ["ledger-1"])
        with self.assertRaises(ValueError):
            context.assert_tree_has_no_grader_verdicts({"EXPERIENCE.md": '"pass_rate": 1'})
        previous_sessions = []

        def run_opencode(command, **kwargs):
            database = Path(kwargs["env"]["XDG_DATA_HOME"]) / "opencode.db"
            previous_sessions.append(database.exists())
            database.parent.mkdir(parents=True, exist_ok=True)
            database.write_text("private evidence from this session")
            return Mock(stdout='{"type":"text","part":{"text":"done"}}', stderr="", returncode=0)

        proposer = Proposer(self.proposer.spec)
        with (
            patch("harness2.core.proposer._opencode_dir", return_value=self.root),
            patch("harness2.core.proposer.subprocess.run", side_effect=run_opencode),
        ):
            for index in range(2):
                proposer.call("Read this task only", None, f"isolation{index}")
        self.assertEqual(previous_sessions, [False, False])

    def test_invalid_proposal_retries_with_feedback_and_accumulates_cost(self):
        class RepairProposer(DemoProposer):
            feedback = ""

            def call(self, mission, context_dir, label, **kwargs):
                script = Path(context_dir) / "harness/scripts/summarize.py"
                if label.endswith(".a1"):
                    script.write_text("def broken(:\n")
                    return Reply(
                        label,
                        "invalid edit",
                        True,
                        cost_usd=1,
                        attempts=1,
                        workspace=script.parent.parent,
                    )
                self.feedback = mission
                script.write_text(BASE_SCRIPT)
                reply = super().call(mission, context_dir, label, **kwargs)
                reply.cost_usd = 2
                return reply

        proposer = RepairProposer(self.proposer.spec)
        stage = recursion._run_stage(
            label="inv1",
            dim="INV",
            job="Repair the script",
            adapter=self.adapter,
            prop=proposer,
            ctx_dir=self.root / "context",
            task_id="ledger-1",
            rollouts=[],
            fallback=tree.read_tree(self.adapter.substrate.surface(), self.adapter.base_harness()),
            harness_dir=self.adapter.base_harness(),
            library_dir=None,
        )
        self.assertEqual((stage.fell_back, stage.attempts, stage.usd), (False, 2, 3))
        self.assertIn("PREVIOUS ATTEMPT WAS REJECTED", proposer.feedback)

    def test_no_entry_stays_out_of_library_on_resume(self):
        class NoExperience(DemoProposer):
            def call(self, mission, context_dir, label, **kwargs):
                if label.startswith("experience."):
                    if label.endswith("seq1"):
                        raise ProposerError("reflection failed", attempts=2, cost_usd=2)
                    return Reply(label, "NO_ENTRY", True, cost_usd=1)
                return super().call(mission, context_dir, label, **kwargs)

        proposer = NoExperience(self.proposer.spec)
        self.run_stream(k=1, proposer=proposer)
        self.run_stream(k=1, proposer=proposer)
        for mode in ("agg", "seq"):
            library = Library(
                self.root / "library/1" / mode, "accounting", self.adapter.substrate.surface()
            )
            self.assertEqual(library.entries(), [])
        summary_path = self.adapter.spec.runs_root / "demo-oc-agg1/ledger-1/summary.json"
        selections = json.loads(summary_path.read_text())["selections"]
        self.assertEqual([selections[mode]["usd"] for mode in ("agg", "seq")], [1, 2])

    def test_failed_selection_is_never_judged(self):
        class FailedSelector(DemoProposer):
            def call(self, mission, context_dir, label, **kwargs):
                if label.startswith("select."):
                    raise ProposerError("selection failed", attempts=2, cost_usd=3)
                return super().call(mission, context_dir, label, **kwargs)

        self.run_stream(proposer=FailedSelector(self.proposer.spec))
        summary_path = self.adapter.spec.runs_root / "demo-oc-agg2/ledger-1/summary.json"
        summary = json.loads(summary_path.read_text())
        self.assertEqual([summary["selections"][mode]["usd"] for mode in ("agg", "seq")], [3, 3])
        self.adapter.score = Mock(return_value={"n_passed": 1, "n_criteria": 1})
        rows = judge.judge_selections(
            self.adapter,
            list(summary["selections"].values()),
        )
        self.adapter.score.assert_not_called()
        self.assertEqual((rows[0].graded, rows[0].n_passed), (False, None))

    def test_feedback_reaches_selection_and_reflection_only(self):
        feedback_dir = self.root / "feedback"
        feedback_dir.mkdir()
        note = "Credits are part of the total: keep the negative amounts."
        (feedback_dir / "ledger-1.md").write_text(note + "\n")
        proposer = MissionLog(self.proposer.spec)
        self.run_stream(proposer=proposer, feedback_dir=feedback_dir)
        runs = self.adapter.spec.runs_root
        for mode in ("agg", "seq"):
            work = runs / f"demo-oc-{mode}2" / "ledger-1"
            for where in ("selection", "reflection"):
                self.assertEqual((work / where / f"{mode}2/feedback.md").read_text(), note + "\n")
            self.assertTrue((work / f"reflection/{mode}2.experience/feedback.md").is_file())
        self.assertEqual(list(runs.glob("*/ledger-2/**/feedback.md")), [])
        for label, where, mission in proposer.missions:
            reviewed = "/ledger-1/" in where and label.startswith(("select.", "experience."))
            with self.subTest(label=label, where=where):
                self.assertEqual("A PERSON HAS REVIEWED THIS TASK" in mission, reviewed)
                self.assertEqual("context/feedback.md" in mission, reviewed)
        selections = json.loads((runs / "demo-oc-agg2/ledger-1/summary.json").read_text())
        self.assertEqual(
            {sel["feedback_fingerprint"] for sel in selections["selections"].values()},
            {feedback.fingerprint(note)},
        )
        other = json.loads((runs / "demo-oc-agg2/ledger-2/summary.json").read_text())
        self.assertFalse(any("feedback_fingerprint" in sel for sel in other["selections"].values()))

    def test_feedback_off_changes_nothing_and_new_feedback_reruns_its_task(self):
        plain = MissionLog(self.proposer.spec)
        self.run_stream(proposer=plain)

        second = Path(self.enterContext(tempfile.TemporaryDirectory()))
        adapter = DemoAdapter(second)
        (second / "feedback").mkdir()
        empty = MissionLog(adapter.spec.channels["proposer"])
        schedule.run_domain(
            "accounting",
            adapter.tasks(),
            2,
            adapter=adapter,
            prop=empty,
            runs_root=adapter.spec.runs_root,
            library_root=second / "library" / "2",
            feedback_dir=second / "feedback",
        )

        def written(root):
            files = sorted(p for p in (root / "runs").rglob("*") if p.is_file())
            return {
                p.relative_to(root).as_posix(): p.read_text().replace(str(root), "<root>")
                for p in files
            }

        self.assertEqual(written(self.root), written(second))
        self.assertEqual(
            sorted((label, mission) for label, _, mission in plain.missions),
            sorted((label, mission) for label, _, mission in empty.missions),
        )

        feedback_dir = self.root / "feedback"
        feedback_dir.mkdir()
        (feedback_dir / "ledger-1.md").write_text("Keep the credits in the total.\n")

        def statuses():
            return [row["status"] for row in self.run_stream(feedback_dir=feedback_dir)["tasks"]]

        self.assertEqual(statuses(), ["ok", "skipped"])
        self.assertEqual(statuses(), ["skipped", "skipped"])
        (feedback_dir / "ledger-1.md").unlink()
        self.assertEqual(statuses(), ["ok", "skipped"])

    def test_feedback_never_carries_a_score(self):
        feedback_dir = self.root / "feedback"
        feedback_dir.mkdir()
        (feedback_dir / "team__ledger-1.md").write_text('{"pass_rate": 1}\n')
        with self.assertRaisesRegex(ValueError, "grader-verdict marker"):
            feedback.read(feedback_dir, "team/ledger-1")
        self.assertEqual(feedback.read(feedback_dir, "ledger-2"), "")
        self.assertEqual(feedback.read(None, "team/ledger-1"), "")


if __name__ == "__main__":
    unittest.main()
