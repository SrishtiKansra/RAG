"""Tests for src.evaluator's pure aggregation logic.

Evaluator.summarize() turns per-question rows into the accuracy/hallucination
percentages that end up in the report's tables and figures — so a bug here
would silently corrupt every headline number. It takes plain dicts and does
no network/API calls, so it's cheap to test directly.
"""

from __future__ import annotations

from src.evaluator import Evaluator, _normalise_answer


def _row(dataset, experiment, model, top_k, is_correct, is_hallucinated, is_poisoned_target):
    return {
        "dataset": dataset,
        "experiment": experiment,
        "experiment_label": experiment,
        "model": model,
        "top_k": top_k,
        "is_correct": is_correct,
        "is_hallucinated": is_hallucinated,
        "is_poisoned_target": is_poisoned_target,
        "self_consistency": "",
    }


class TestSummarize:
    def test_accuracy_over_all_items(self):
        rows = [
            _row("synthetic", "exp1", "m", 4, True, False, False),
            _row("synthetic", "exp1", "m", 4, True, False, False),
            _row("synthetic", "exp1", "m", 4, False, False, False),
            _row("synthetic", "exp1", "m", 4, True, False, False),
        ]
        summary = Evaluator.summarize(rows)
        assert len(summary) == 1
        assert summary[0]["accuracy"] == 0.75

    def test_hallucination_rate_only_counts_poisoned_targets(self):
        rows = [
            _row("synthetic", "exp2", "m", 4, False, True, True),   # poisoned, hallucinated
            _row("synthetic", "exp2", "m", 4, True, False, True),   # poisoned, not hallucinated
            _row("synthetic", "exp2", "m", 4, True, False, False),  # not a poisoned target at all
        ]
        summary = Evaluator.summarize(rows)[0]
        # denominator should be 2 (poisoned targets only), not 3
        assert summary["num_poisoned_targets"] == 2
        assert summary["hallucination_rate"] == 0.5

    def test_hallucination_rate_is_zero_with_no_poisoned_targets(self):
        rows = [_row("fever", "exp1", "m", 4, True, False, False)]
        summary = Evaluator.summarize(rows)[0]
        assert summary["num_poisoned_targets"] == 0
        assert summary["hallucination_rate"] == 0.0

    def test_groups_are_kept_separate_by_dataset_experiment_model_topk(self):
        rows = [
            _row("synthetic", "exp1", "m", 4, True, False, False),
            _row("hotpotqa", "exp1", "m", 4, False, False, False),
        ]
        summary = Evaluator.summarize(rows)
        assert len(summary) == 2
        by_dataset = {s["dataset"]: s["accuracy"] for s in summary}
        assert by_dataset["synthetic"] == 1.0
        assert by_dataset["hotpotqa"] == 0.0

    def test_empty_input_returns_empty_summary(self):
        assert Evaluator.summarize([]) == []


class TestNormaliseAnswer:
    def test_collapses_whitespace(self):
        assert _normalise_answer("hello   \n\n world") == "hello world"

    def test_lowercases(self):
        assert _normalise_answer("YES") == "yes"

    def test_truncates_to_200_chars(self):
        long_answer = "a" * 500
        assert len(_normalise_answer(long_answer)) == 200

    def test_handles_none_gracefully(self):
        assert _normalise_answer(None) == ""
