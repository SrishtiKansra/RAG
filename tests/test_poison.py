"""Tests for src.datasets_p5.poison — the fabricated-answer generator.

These are pure functions (no network, no API key needed), so they run fast
and are worth pinning down: if fake_answer() ever produced a false answer
equal to the truth, the whole "poisoning" experiment would be measuring
nothing.
"""

from __future__ import annotations

from src.datasets_p5.poison import fake_answer, make_poison_passage


class TestFakeAnswerBooleans:
    def test_yes_flips_to_no(self):
        assert fake_answer("yes") == "no"

    def test_no_flips_to_yes(self):
        assert fake_answer("no") == "yes"

    def test_boolean_variants_are_case_insensitive(self):
        assert fake_answer("Yes") == "no"
        assert fake_answer("TRUE") == "no"
        assert fake_answer("Incorrect") == "yes"


class TestFakeAnswerNumbers:
    def test_integer_is_nudged_but_stays_numeric(self):
        result = fake_answer("2011")
        assert result != "2011"
        assert result.lstrip("-").isdigit()

    def test_integer_nudge_is_within_expected_range(self):
        # poison.py nudges ints by one of {-7,-5,-3,3,5,7}
        result = int(fake_answer("2011"))
        assert result != 2011
        assert abs(result - 2011) in {3, 5, 7}

    def test_float_is_nudged_but_stays_numeric(self):
        result = fake_answer("3.5")
        assert result != "3.5"
        float(result)  # should not raise

    def test_number_with_thousands_separator(self):
        result = fake_answer("1,200")
        assert result != "1,200"
        assert result.lstrip("-").isdigit()


class TestFakeAnswerFreeText:
    def test_result_differs_from_truth(self):
        assert fake_answer("Lisbon") != "Lisbon"

    def test_uses_pool_when_available(self):
        pool = ["Lisbon", "Oslo", "Zagreb", "Berlin"]
        result = fake_answer("Lisbon", pool=pool)
        assert result != "Lisbon"
        assert result in pool

    def test_pool_never_returns_a_numeric_candidate(self):
        # Numbers in the pool shouldn't leak into a free-text answer's lie,
        # or the poison stops being plausible.
        pool = ["Lisbon", "2011", "Oslo"]
        for _ in range(20):
            result = fake_answer("Lisbon", pool=pool)
            assert result != "2011"

    def test_falls_back_gracefully_with_no_pool(self):
        result = fake_answer("Lisbon")
        assert result != "Lisbon"
        assert result  # non-empty

    def test_empty_truth_does_not_crash(self):
        result = fake_answer("")
        assert isinstance(result, str)


class TestMakePoisonPassage:
    def test_fact_style_contains_false_answer(self):
        passage = make_poison_passage(
            "the formation year of Quantum Echo", "1998", style="fact"
        )
        assert "1998" in passage
        assert "Quantum Echo" in passage

    def test_claim_style_does_not_require_false_answer_in_text(self):
        # FEVER-style poison asserts the claim itself, not a bare value.
        passage = make_poison_passage(
            "the band was formed in 1998", "", style="claim"
        )
        assert "band was formed in 1998" in passage

    def test_strips_trailing_punctuation_from_subject(self):
        passage = make_poison_passage("What year was it founded?", "1998", style="fact")
        assert "founded??" not in passage
