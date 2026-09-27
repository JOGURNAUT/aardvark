"""Tests for the judge-calibration arithmetic.

The agreement numbers are the whole output of `eval/calibrate.py`, and they are
the kind of arithmetic that is wrong quietly: a kappa that returns 0 instead of
None on a degenerate sample reads as "no better than chance" when the truth is
"this sample cannot tell you". These cover the cases where that happens.

No network, no API keys, no labels of anyone's: every case here is constructed.
"""
from __future__ import annotations

import json

import pytest

from eval import calibrate


# ------------------------------------------------------------------ kappa

def test_perfect_agreement_is_one():
    pairs = [(1.0, 1.0), (0.0, 0.0), (1.0, 1.0), (0.0, 0.0)]
    assert calibrate.cohens_kappa(pairs) == 1.0


def test_chance_level_agreement_is_near_zero():
    # Two raters each saying 1 half the time, agreeing exactly as often as
    # chance predicts.
    pairs = [(1.0, 1.0), (1.0, 0.0), (0.0, 1.0), (0.0, 0.0)]
    assert calibrate.cohens_kappa(pairs) == 0.0


def test_worse_than_chance_is_negative():
    pairs = [(1.0, 0.0), (0.0, 1.0), (1.0, 0.0), (0.0, 1.0)]
    k = calibrate.cohens_kappa(pairs)
    assert k is not None and k < 0


def test_single_class_returns_none_not_zero():
    # Everything labelled 1 by both raters. Raw agreement is 1.0 and chance
    # agreement is also 1.0, so kappa is 0/0. Returning 0 here would read as
    # "no better than chance", which is the opposite of what the data says,
    # and returning 1.0 would claim a measurement the sample cannot support.
    pairs = [(1.0, 1.0)] * 9
    assert calibrate.cohens_kappa(pairs) is None


def test_empty_returns_none():
    assert calibrate.cohens_kappa([]) is None


def test_threshold_is_at_half():
    # 0.5 counts as positive; anything below does not. The refusal judge emits
    # 0.75 and 0.25 for its two partial-confidence cases, so this boundary
    # decides which side each of them lands on.
    assert calibrate.cohens_kappa([(0.5, 1.0), (0.49, 0.0)]) == 1.0


# -------------------------------------------------------------- agreement

def test_continuous_agreement_uses_the_tolerance():
    pairs = [(0.9, 1.0), (0.5, 1.0), (0.8, 0.7)]     # diffs 0.1, 0.5, 0.1
    s = calibrate.agreement(pairs, "continuous", tolerance=0.25)
    assert s["n"] == 3
    assert s["agreement"] == round(2 / 3, 3)
    assert s["mean_abs_diff"] == round((0.1 + 0.5 + 0.1) / 3, 3)


def test_binary_agreement_ignores_the_tolerance():
    # 0.9 and 0.6 are 0.3 apart, which fails the continuous tolerance, but both
    # mean "yes" and the binary metrics are about which side of the line.
    pairs = [(0.9, 0.6)]
    assert calibrate.agreement(pairs, "binary")["agreement"] == 1.0
    assert calibrate.agreement(pairs, "continuous")["agreement"] == 0.0


def test_bias_is_signed():
    # A judge that is consistently generous and one that is merely noisy have
    # the same mean_abs_diff. Only the sign separates them, and only one of the
    # two is fixable by moving a threshold.
    generous = calibrate.agreement([(0.9, 0.6), (0.8, 0.5)], "continuous")
    noisy = calibrate.agreement([(0.9, 0.6), (0.2, 0.5)], "continuous")
    assert generous["judge_bias"] == pytest.approx(0.3)
    assert noisy["judge_bias"] == pytest.approx(0.0)
    assert generous["mean_abs_diff"] == noisy["mean_abs_diff"]


def test_empty_sample_reports_none_not_zero():
    s = calibrate.agreement([], "continuous")
    assert s["n"] == 0
    assert s["agreement"] is None and s["mean_abs_diff"] is None


# ---------------------------------------------------------------- extract

@pytest.fixture()
def labels_file(tmp_path, monkeypatch):
    path = tmp_path / "human_labels.json"
    monkeypatch.setattr(calibrate, "LABELS", path)
    return path


def _results(tmp_path, rows):
    p = tmp_path / "results.json"
    p.write_text(json.dumps(rows), encoding="utf-8")
    return p


def test_extract_skips_metrics_the_judge_could_not_score(tmp_path, labels_file):
    # A None score is a judge failure, not a judgement. Labelling it would be
    # comparing a human against nothing. metrics.judge_failures already counts
    # these separately.
    src = _results(tmp_path, [{
        "id": "q1", "category": "multi_hop", "question": "why", "answer": "because",
        "faithfulness": {"score": None, "error": "judge_returned_unparseable"},
        "relevance": {"score": 0.8},
    }])
    out = calibrate.extract(src)
    assert [i["metric"] for i in out["items"]] == ["relevance"]


def test_extract_preserves_labels_already_given(tmp_path, labels_file):
    rows = [{"id": "q1", "category": "c", "question": "q", "answer": "a",
             "relevance": {"score": 0.8}}]
    calibrate.extract(_results(tmp_path, rows))

    data = json.loads(labels_file.read_text(encoding="utf-8"))
    data["items"][0]["human_score"] = 0.4
    data["items"][0]["note"] = "answers a different question"
    labels_file.write_text(json.dumps(data), encoding="utf-8")

    # A later run re-scores the same case. The hand label must survive, or
    # every re-run means relabelling from scratch and nobody does it twice.
    rows[0]["relevance"]["score"] = 0.95
    out = calibrate.extract(_results(tmp_path, rows))

    item = out["items"][0]
    assert item["human_score"] == 0.4
    assert item["note"] == "answers a different question"
    assert item["judge_score"] == 0.95


def test_extract_is_keyed_by_id_and_metric_not_position(tmp_path, labels_file):
    calibrate.extract(_results(tmp_path, [
        {"id": "q1", "category": "c", "question": "q", "answer": "a",
         "relevance": {"score": 0.8}, "faithfulness": {"score": 0.7}},
    ]))
    # Same question, reordered and with a new metric alongside.
    out = calibrate.extract(_results(tmp_path, [
        {"id": "q0", "category": "c", "question": "q0", "answer": "a",
         "relevance": {"score": 0.5}},
        {"id": "q1", "category": "c", "question": "q", "answer": "a",
         "relevance": {"score": 0.8}, "faithfulness": {"score": 0.7}},
    ]))
    keys = {i["key"] for i in out["items"]}
    assert keys == {"q1::relevance", "q1::faithfulness", "q0::relevance"}


def test_report_with_no_labels_says_so(tmp_path, labels_file, capsys):
    calibrate.extract(_results(tmp_path, [
        {"id": "q1", "category": "c", "question": "q", "answer": "a",
         "relevance": {"score": 0.8}},
    ]))
    assert calibrate.report() == {}
    assert "no labels yet" in capsys.readouterr().out
