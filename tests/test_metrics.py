"""Tests for the roll-up in eval/metrics.py.

These exist because of a specific mistake. A run of this harness reported
`avg_faithfulness: 1.0`, and the 1.0 was a mean over eight of fourteen
applicable cases: the judge had failed on the other six and `avg()` skipped
them. The failures were the long, contradictory, hedged answers, so dropping
them did not lower the mean, it raised it. A mean over an unknown denominator
is not a measurement.

Three things fixed that, and this file pins all three: None never becomes a
zero, every metric reports scored-against-applicable, and every judge failure
is attributable to its question and metric.
"""
from __future__ import annotations

import pytest

from eval import metrics


def q(cid: str, category: str = "multi_hop", **blocks) -> dict:
    """A per-question result. Only the blocks a caller passes are present,
    which is what makes 'applicable' meaningful: a refusal question carries a
    refusal block and a factual one does not."""
    return {"id": cid, "category": category, "latency_ms": 1000, "n_unique_domains": 2, **blocks}


# ------------------------------------------------------------ dropping vs zero

def test_a_failed_judge_is_excluded_from_the_mean_not_counted_as_zero():
    # Counting a judge failure as 0.0 would understate the system; counting it
    # as a score would overstate it. It is neither, so it leaves the numerator
    # and the denominator together.
    results = [
        q("a", faithfulness={"score": 1.0}),
        q("b", faithfulness={"score": None, "error": "judge_returned_unparseable"}),
    ]
    assert metrics.aggregate(results)["overall"]["avg_faithfulness"] == 1.0


def test_coverage_reports_the_denominator_the_mean_hid():
    # The whole point. The mean is 1.0 and the coverage says 1 of 2, so nobody
    # reads the 1.0 as "everything passed".
    results = [
        q("a", faithfulness={"score": 1.0}),
        q("b", faithfulness={"score": None, "error": "judge_call_failed"}),
    ]
    cov = metrics.aggregate(results)["coverage"]["faithfulness"]
    assert cov == {"scored": 1, "applicable": 2, "complete": False}


def test_coverage_is_complete_when_every_applicable_case_scored():
    results = [q("a", faithfulness={"score": 0.8}), q("b", faithfulness={"score": 0.6})]
    assert metrics.aggregate(results)["coverage"]["faithfulness"]["complete"] is True


def test_a_metric_nobody_was_applicable_for_is_absent_rather_than_zero():
    # No question asked for a conflict judgement, so there is nothing to report.
    # A 0 here would read as "it handled conflicts badly".
    results = [q("a", faithfulness={"score": 1.0})]
    assert "conflict_score" not in metrics.aggregate(results)["coverage"]


def test_every_judge_failing_leaves_the_mean_as_none():
    # None is the honest answer. A 0.0 would be a claim about quality that no
    # judgement supports.
    results = [q("a", faithfulness={"score": None, "error": "judge_call_failed"})]
    out = metrics.aggregate(results)
    assert out["overall"]["avg_faithfulness"] is None
    assert out["coverage"]["faithfulness"] == {"scored": 0, "applicable": 1, "complete": False}


# ------------------------------------------------------------- attribution

def test_each_failure_is_attributable_to_its_question_and_metric():
    # A count of failures is not actionable; knowing which question and which
    # judge is.
    results = [
        q("a", category="insufficient_evidence",
          refusal={"score": None, "reason": "judge_failed"}),
        q("b", faithfulness={"score": 0.9}),
    ]
    failures = metrics.aggregate(results)["judge_failures"]
    assert len(failures) == 1
    assert failures[0]["id"] == "a"
    assert failures[0]["metric"] == "refusal_score"
    assert failures[0]["category"] == "insufficient_evidence"


def test_the_failure_reason_is_carried_through_not_flattened():
    # "unparseable" and "all providers failed" call for different fixes: one is
    # a prompt or budget problem, the other is quota.
    results = [q("a", faithfulness={"score": None, "error": "judge_returned_unparseable"})]
    assert metrics.aggregate(results)["judge_failures"][0]["reason"] == "judge_returned_unparseable"


def test_a_block_that_was_never_applicable_is_not_a_failure():
    # Not asking a judge is not the same as asking and getting nothing back.
    results = [q("a", faithfulness={"score": 0.5})]
    assert metrics.aggregate(results)["judge_failures"] == []


def test_citation_precision_is_read_from_its_own_field():
    # It reports `precision`, not `score`, and reading the wrong key would make
    # every case look like a judge failure.
    results = [q("a", citation_precision={"precision": 0.9, "total_citation_events": 4})]
    out = metrics.aggregate(results)
    assert out["overall"]["avg_citation_precision"] == 0.9
    assert out["coverage"]["citation_precision"]["scored"] == 1
    assert out["judge_failures"] == []


# --------------------------------------------------------------- grouping

def test_categories_are_reported_separately_with_their_own_counts():
    results = [
        q("a", category="multi_hop", faithfulness={"score": 1.0}),
        q("b", category="multi_hop", faithfulness={"score": 0.5}),
        q("c", category="comparison", faithfulness={"score": 0.2}),
    ]
    by_cat = metrics.aggregate(results)["by_category"]
    assert by_cat["multi_hop"]["n"] == 2
    assert by_cat["multi_hop"]["avg_faithfulness"] == 0.75
    assert by_cat["comparison"]["avg_faithfulness"] == 0.2


def test_a_category_where_every_judge_failed_reports_none_not_zero():
    # This is what a rate-limited run looks like: a whole category with no
    # scores at all. It must not read as a category the system did badly on.
    results = [
        q("a", category="conflicting_sources", conflict={"score": None, "reason": "judge_failed"}),
        q("b", category="conflicting_sources", conflict={"score": None, "reason": "judge_failed"}),
    ]
    assert metrics.aggregate(results)["by_category"]["conflicting_sources"]["avg_conflict_score"] is None


def test_question_count_is_the_number_of_questions_not_of_scores():
    results = [q("a", faithfulness={"score": 1.0}), q("b"), q("c")]
    assert metrics.aggregate(results)["n_questions"] == 3


# ------------------------------------------------------------------ rounding

def test_means_are_rounded_to_three_places():
    results = [q("a", faithfulness={"score": 1 / 3}), q("b", faithfulness={"score": 1 / 3})]
    assert metrics.aggregate(results)["overall"]["avg_faithfulness"] == 0.333


@pytest.mark.parametrize("n_scored,n_total", [(9, 11), (8, 14), (2, 14)])
def test_coverage_survives_the_shapes_real_runs_produced(n_scored, n_total):
    # The three denominators actual runs of this harness reported behind a
    # headline number.
    results = [q(f"s{i}", faithfulness={"score": 1.0}) for i in range(n_scored)]
    results += [q(f"f{i}", faithfulness={"score": None, "error": "judge_call_failed"})
                for i in range(n_total - n_scored)]
    out = metrics.aggregate(results)
    assert out["overall"]["avg_faithfulness"] == 1.0
    assert out["coverage"]["faithfulness"]["scored"] == n_scored
    assert out["coverage"]["faithfulness"]["applicable"] == n_total
    assert out["coverage"]["faithfulness"]["complete"] is False
