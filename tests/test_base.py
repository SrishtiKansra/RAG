"""Tests for src.datasets_p5.base — Task, label parsing, and scoring.

score_answer() is the function behind every number in the report (accuracy,
hallucination rate), so its edge cases matter more than almost anything else
in the codebase — including the "correct AND hallucinated at once" case
called out in the report's limitations section.
"""

from __future__ import annotations

from src.datasets_p5.base import (
    LABEL_NEI,
    LABEL_REFUTED,
    LABEL_SUPPORTED,
    Task,
    parse_label,
    score_answer,
)


class TestTask:
    def test_is_poisoned_true_when_poison_doc_present(self):
        task = Task(
            id="t1", dataset="synthetic", question="q", gold="2011",
            real_docs=["doc"], poison_doc="fake doc",
        )
        assert task.is_poisoned is True

    def test_is_poisoned_false_when_no_poison_doc(self):
        task = Task(
            id="t2", dataset="synthetic", question="q", gold="2011",
            real_docs=["doc"], poison_doc=None,
        )
        assert task.is_poisoned is False


class TestParseLabel:
    def test_supported(self):
        assert parse_label("SUPPORTED\nThe claim is confirmed by the source.") == LABEL_SUPPORTED

    def test_refuted(self):
        assert parse_label("REFUTED - the source contradicts this.") == LABEL_REFUTED

    def test_not_enough_info_explicit(self):
        assert parse_label("NOT ENOUGH INFO, the sources are silent on this.") == LABEL_NEI

    def test_nei_checked_before_support_refute(self):
        # "do not support" contains "support" but should read as NEI-ish /
        # not a false SUPPORTED — regression test for the ordering noted
        # in the base.py docstring.
        result = parse_label("There is insufficient evidence to support this claim.")
        assert result == LABEL_NEI

    def test_ambiguous_text_defaults_to_nei(self):
        assert parse_label("I have no idea what this means.") == LABEL_NEI

    def test_case_insensitive(self):
        assert parse_label("supported by all sources") == LABEL_SUPPORTED

    def test_empty_string_defaults_to_nei(self):
        assert parse_label("") == LABEL_NEI


class TestScoreAnswerSubstring:
    def _task(self, gold="2011", poison_answer="1998"):
        return Task(
            id="t", dataset="synthetic", question="q", gold=gold,
            real_docs=["doc"], poison_answer=poison_answer, scoring="substring",
        )

    def test_correct_answer_is_flagged_correct(self):
        result = score_answer("The formation year was 2011.", self._task())
        assert result["is_correct"] is True
        assert result["is_hallucinated"] is False

    def test_poisoned_answer_is_flagged_hallucinated(self):
        result = score_answer("The formation year was 1998.", self._task())
        assert result["is_correct"] is False
        assert result["is_hallucinated"] is True

    def test_both_flags_can_be_true_at_once(self):
        # Regression test for the exact scoring artifact documented in the
        # report (§4.2): a hedged answer that mentions both values scores
        # as correct AND hallucinated simultaneously.
        answer = (
            "Sources disagree: either 2011 (Source 1) or 1998 (Source 2). "
            "I can't be certain which is correct."
        )
        result = score_answer(answer, self._task())
        assert result["is_correct"] is True
        assert result["is_hallucinated"] is True

    def test_neither_flag_when_answer_is_unrelated(self):
        result = score_answer("I don't know.", self._task())
        assert result["is_correct"] is False
        assert result["is_hallucinated"] is False

    def test_no_poison_answer_means_hallucination_always_false(self):
        task = self._task(poison_answer="")
        result = score_answer("Anything at all.", task)
        assert result["is_hallucinated"] is False

    def test_matching_is_case_insensitive(self):
        result = score_answer("the answer is LISBON", self._task(gold="Lisbon", poison_answer="Oslo"))
        assert result["is_correct"] is True


class TestScoreAnswerLabel:
    def _task(self, gold=LABEL_SUPPORTED, poison_answer=LABEL_REFUTED):
        return Task(
            id="t", dataset="fever", question="claim", gold=gold,
            real_docs=["doc"], poison_answer=poison_answer, scoring="label",
        )

    def test_correct_label(self):
        result = score_answer("SUPPORTED. The evidence confirms this.", self._task())
        assert result["is_correct"] is True
        assert result["is_hallucinated"] is False

    def test_hallucinated_label(self):
        result = score_answer("REFUTED. The evidence contradicts this.", self._task())
        assert result["is_correct"] is False
        assert result["is_hallucinated"] is True

    def test_nei_is_neither_correct_nor_hallucinated_when_gold_is_supported(self):
        result = score_answer("NOT ENOUGH INFO to decide.", self._task())
        assert result["is_correct"] is False
        assert result["is_hallucinated"] is False
