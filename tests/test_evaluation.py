import tempfile
import unittest
from pathlib import Path

from adaptive_agent_lab.core import Action, FinalAnswer
from adaptive_agent_lab.evaluation import (
    BASELINE_CASES,
    EvalCase,
    load_eval_cases,
    run_baseline,
    run_evaluation,
    _score_answer,
)
from adaptive_agent_lab.trajectory import TrajectoryStore


class ScriptedModel:
    def __init__(self, *decisions):
        self.decisions = list(decisions)

    def decide(self, messages, tools):
        return self.decisions.pop(0)


class MemorySensitiveModel:
    def decide(self, messages, tools):
        latest = messages[-1]
        if latest.role == "user":
            if any(message.role == "memory" for message in messages):
                return Action("word_count", {"text": "hello world"})
            return Action("word_count", {"text": 123})
        if latest.role == "tool" and latest.content["error"]:
            return Action("word_count", {"text": "hello world"})
        return FinalAnswer(str(latest.content["output"]))


def seed_recovered_trajectory(store: TrajectoryStore) -> None:
    run_id = store.start_run("Count words after correcting invalid arguments")
    store.append_step(
        run_id,
        0,
        "model_action",
        {"tool_name": "word_count", "arguments": {"text": 123}},
    )
    store.append_step(
        run_id,
        0,
        "tool_result",
        {
            "tool_name": "word_count",
            "output": None,
            "error": "ValueError: Argument text must be a string",
        },
    )
    store.append_step(
        run_id,
        1,
        "model_action",
        {"tool_name": "word_count", "arguments": {"text": "hello world"}},
    )
    store.append_step(
        run_id,
        1,
        "tool_result",
        {"tool_name": "word_count", "output": 2, "error": None},
    )
    store.finish_run(run_id, "completed", "2")


class EvaluationTests(unittest.TestCase):
    def test_scores_one_numeric_value_inside_a_short_explanation(self) -> None:
        self.assertEqual(
            _score_answer(
                "4",
                "4 words. The words are: I, love, building, agents.",
            ),
            (True, "single_number"),
        )
        self.assertEqual(
            _score_answer("4", "The answer might be 4 or 5."),
            (False, "single_number"),
        )

    def test_normalizes_text_without_accepting_extra_explanation(self) -> None:
        self.assertEqual(
            _score_answer("Paris", "  paris.  "),
            (True, "normalized_text"),
        )
        self.assertEqual(
            _score_answer("Paris", "The answer is Paris."),
            (False, "normalized_text"),
        )

    def test_baseline_is_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = run_baseline(Path(directory) / "eval.db")

        self.assertEqual(report["summary"]["cases"], len(BASELINE_CASES))
        self.assertEqual(report["summary"]["success_rate"], 1.0)
        self.assertEqual(report["summary"]["tool_selection_accuracy"], 1.0)
        self.assertTrue(all(result["passed"] for result in report["results"]))

    def test_answer_and_tool_selection_are_scored_separately(self) -> None:
        model = ScriptedModel(
            Action("lookup", {"key": "capital:france"}),
            FinalAnswer("84"),
        )
        cases = [
            EvalCase(
                "wrong-tool-right-answer",
                "What is 12 * 7?",
                "84",
                "calculator",
            )
        ]

        with tempfile.TemporaryDirectory() as directory:
            report = run_evaluation(model, cases, Path(directory) / "eval.db")

        result = report["results"][0]
        self.assertTrue(result["passed"])
        self.assertFalse(result["tool_selection_passed"])
        self.assertEqual(result["selected_tool"], "lookup")
        self.assertEqual(report["summary"]["success_rate"], 1.0)
        self.assertEqual(report["summary"]["tool_selection_accuracy"], 0.0)

    def test_reports_first_attempt_failure_and_recovery(self) -> None:
        model = ScriptedModel(
            Action("lookup", {"key": "country:Japan"}),
            Action("lookup", {"key": "capital:Japan"}),
            FinalAnswer("Tokyo"),
        )
        cases = [
            EvalCase(
                "lookup-recovery",
                "What is the capital of Japan?",
                "Tokyo",
                "lookup",
            )
        ]

        with tempfile.TemporaryDirectory() as directory:
            report = run_evaluation(model, cases, Path(directory) / "eval.db")

        result = report["results"][0]
        self.assertTrue(result["passed"])
        self.assertFalse(result["first_tool_call_succeeded"])
        self.assertEqual(result["tool_errors"], 1)
        self.assertTrue(result["recovered_after_tool_error"])
        self.assertEqual(report["summary"]["first_tool_call_success_rate"], 0.0)
        self.assertEqual(report["summary"]["total_tool_errors"], 1)
        self.assertEqual(report["summary"]["recovery_rate"], 1.0)

    def test_uses_an_isolated_memory_database_and_reports_recall(self) -> None:
        cases = [
            EvalCase(
                "memory-word-count",
                "Count words after correcting bad arguments",
                "2",
                "word_count",
            )
        ]

        with tempfile.TemporaryDirectory() as directory:
            memory_db = Path(directory) / "memory.db"
            result_db = Path(directory) / "results.db"
            seed_recovered_trajectory(TrajectoryStore(memory_db))

            report = run_evaluation(
                MemorySensitiveModel(),
                cases,
                result_db,
                memory_db_path=memory_db,
            )

        result = report["results"][0]
        self.assertTrue(result["memory_recalled"])
        self.assertEqual(result["recalled_memories"], 1)
        self.assertEqual(result["tool_calls"], 1)
        self.assertTrue(report["summary"]["memory_enabled"])
        self.assertEqual(report["summary"]["memory_recall_rate"], 1.0)

    def test_rejects_using_the_result_database_as_memory(self) -> None:
        cases = [EvalCase("one", "Count words", "2", "word_count")]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shared.db"
            with self.assertRaisesRegex(
                ValueError,
                "Memory and evaluation databases must be different",
            ):
                run_evaluation(
                    MemorySensitiveModel(),
                    cases,
                    path,
                    memory_db_path=path,
                )

    def test_rejects_a_missing_memory_database(self) -> None:
        cases = [EvalCase("one", "Count words", "2", "word_count")]

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Memory database does not exist"):
                run_evaluation(
                    MemorySensitiveModel(),
                    cases,
                    Path(directory) / "results.db",
                    memory_db_path=Path(directory) / "missing.db",
                )

    def test_loads_jsonl_cases_and_rejects_duplicate_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.jsonl"
            path.write_text(
                '{"id":"one","task":"First","expected":"1"}\n'
                '{"id":"one","task":"Second","expected":"2"}\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Duplicate eval case ID: one"):
                load_eval_cases(path)


if __name__ == "__main__":
    unittest.main()
