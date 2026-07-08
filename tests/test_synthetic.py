"""Tests for src.datasets_p5.synthetic — the offline, deterministic dataset.

This adapter never touches the network, so it's a good place to check that
the Task objects handed to the evaluator are internally consistent (poison
answer never equals the gold answer, real doc actually contains the gold
value, etc.) without needing HotpotQA/FEVER downloads or a Groq API key.
"""

from __future__ import annotations

from src.datasets_p5.synthetic import SyntheticAdapter


class TestSyntheticAdapter:
    def test_load_returns_requested_count(self):
        tasks = SyntheticAdapter().load(5)
        assert len(tasks) == 5

    def test_load_caps_at_available_facts(self):
        tasks = SyntheticAdapter().load(1000)
        assert 0 < len(tasks) < 1000

    def test_load_zero_returns_empty(self):
        assert SyntheticAdapter().load(0) == []

    def test_every_task_is_poisoned(self):
        tasks = SyntheticAdapter().load(10)
        assert all(t.is_poisoned for t in tasks)

    def test_poison_answer_never_equals_gold(self):
        tasks = SyntheticAdapter().load(20)
        for t in tasks:
            assert t.poison_answer.strip().lower() != t.gold.strip().lower()

    def test_real_doc_contains_the_gold_value(self):
        tasks = SyntheticAdapter().load(20)
        for t in tasks:
            assert t.gold in t.real_docs[0]

    def test_poison_doc_contains_the_fake_value(self):
        tasks = SyntheticAdapter().load(20)
        for t in tasks:
            assert t.poison_answer in t.poison_doc

    def test_task_ids_are_unique(self):
        tasks = SyntheticAdapter().load(20)
        ids = [t.id for t in tasks]
        assert len(ids) == len(set(ids))

    def test_scoring_mode_is_substring(self):
        tasks = SyntheticAdapter().load(3)
        assert all(t.scoring == "substring" for t in tasks)

    def test_load_is_deterministic_across_fresh_runs(self):
        # poison.py uses one shared, module-level RNG, so it advances with
        # each call *within* a process — calling load() twice in the same
        # session does NOT reproduce the same fake answers. The reproducibility
        # claim in the report is about separate process runs (e.g. rerunning
        # `python -m src.evaluator`), so that's what we actually check here.
        import subprocess
        import sys

        script = (
            "from src.datasets_p5.synthetic import SyntheticAdapter; "
            "print([t.poison_answer for t in SyntheticAdapter().load(10)])"
        )
        run = lambda: subprocess.run(
            [sys.executable, "-c", script],
            cwd=".", capture_output=True, text=True, check=True,
        ).stdout
        assert run() == run()
