"""Calibrate the LLM judges against hand labels.

The eval harness reports 0.876 citation precision and 0.859 faithfulness. Both
numbers come from a model grading another model. Using a second model family
removes the self-grading bias; it does not make the judge correct. Nothing in
the harness has ever checked whether the judge agrees with a person, so the
README's "LLM-as-judge is itself an approximation" was a caveat with no number
attached to it.

This attaches one. It does not try to validate the judge or tune it. It reports
how far the judge is from a human on the same cases, so the headline metrics can
be read with that distance next to them.

Three commands, in order:

    python eval/calibrate.py extract eval/results/<run>/results.json
    python eval/calibrate.py label
    python eval/calibrate.py report

`extract` pulls every scored (question, metric) pair out of a run into a label
file. `label` walks them one at a time and saves after every answer, so it can
be abandoned and resumed. `report` compares the two.

Standard library only, on purpose: labelling should not need the environment
that produced the run, and the agreement arithmetic is the part most worth
being able to read.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
LABELS = ROOT / "eval" / "dataset" / "human_labels.json"

# Which judge block and field each metric lives in, and how a person is asked
# about it. Mirrors _METRIC_SOURCES in metrics.py; kept separate because that
# one is about attribution of failures and this one is about elicitation.
METRICS: dict[str, dict[str, Any]] = {
    "faithfulness": {
        "block": "faithfulness", "field": "score", "kind": "continuous",
        "prompt": "What fraction of the claims in this answer are supported by the snippets?",
    },
    "citation_precision": {
        "block": "citation_precision", "field": "precision", "kind": "continuous",
        "prompt": "What fraction of the citation markers point at a snippet that supports the claim?",
    },
    "relevance": {
        "block": "relevance", "field": "score", "kind": "continuous",
        "prompt": "How directly does this answer address the question? 0 = not at all, 1 = fully.",
    },
    "refusal": {
        "block": "refusal", "field": "score", "kind": "binary",
        "prompt": "Does the answer explicitly say it cannot answer for lack of evidence? 1 = yes, 0 = no.",
    },
    "conflict": {
        "block": "conflict", "field": "score", "kind": "binary",
        "prompt": "Does the answer flag that sources disagree, and cite more than one? 1 = yes, 0 = no.",
    },
}

# A continuous judgement within this distance of the human counts as agreement.
# 0.25 is a quarter of the scale and is a choice, not a standard. It is stated
# in the report rather than buried, because moving it moves the headline.
TOLERANCE = 0.25


# ---------------------------------------------------------------------------
# extract
# ---------------------------------------------------------------------------

def extract(results_path: Path) -> dict:
    """Build label items from a results.json, preserving any labels already given.

    Re-running against a later run adds its new cases and keeps every label
    already entered, so relabelling is never necessary. A case is identified by
    question id and metric, not by position.
    """
    results = json.loads(results_path.read_text(encoding="utf-8"))
    existing = _load()
    by_key = {item["key"]: item for item in existing.get("items", [])}

    added = 0
    for r in results:
        for metric, spec in METRICS.items():
            block = r.get(spec["block"])
            if not block:
                continue
            judge_score = block.get(spec["field"])
            if judge_score is None:
                # A judge failure, not a judgement. judge_failures() in
                # metrics.py already surfaces these; labelling them would be
                # comparing a human against nothing.
                continue
            key = f"{r['id']}::{metric}"
            if key in by_key:
                by_key[key]["judge_score"] = judge_score
                continue
            by_key[key] = {
                "key": key,
                "id": r["id"],
                "category": r.get("category", ""),
                "metric": metric,
                "question": r.get("question", ""),
                "answer": r.get("answer", ""),
                "judge_score": judge_score,
                "human_score": None,
                "note": "",
            }
            added += 1

    out = {
        "source": str(results_path),
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tolerance": TOLERANCE,
        "items": list(by_key.values()),
    }
    _save(out)
    print(f"{added} new cases, {len(out['items'])} total, "
          f"{sum(1 for i in out['items'] if i['human_score'] is None)} unlabelled")
    return out


# ---------------------------------------------------------------------------
# label
# ---------------------------------------------------------------------------

def label() -> None:
    """Walk the unlabelled cases. Saves after every answer.

    The judge's score is deliberately not shown. Seeing it first turns the task
    into agreeing or disagreeing with a number, which is a different and much
    easier question than the one being asked.
    """
    data = _load()
    items = data.get("items", [])
    todo = [i for i in items if i["human_score"] is None]
    if not todo:
        print("nothing left to label")
        return

    print(f"{len(todo)} unlabelled. Enter a number 0-1, 's' to skip, 'q' to stop.\n")
    for n, item in enumerate(todo, 1):
        spec = METRICS[item["metric"]]
        print("=" * 72)
        print(f"[{n}/{len(todo)}]  {item['id']}  {item['metric']}  ({item['category']})")
        print(f"\nQ: {item['question']}\n")
        print(f"A: {item['answer'][:1200]}")
        if len(item["answer"]) > 1200:
            print(f"   ... [{len(item['answer']) - 1200} more characters]")
        print(f"\n{spec['prompt']}")

        raw = input("> ").strip().lower()
        if raw == "q":
            break
        if raw in ("s", ""):
            continue
        try:
            score = float(raw)
        except ValueError:
            print("not a number, skipping")
            continue
        if not 0.0 <= score <= 1.0:
            print("out of range, skipping")
            continue

        item["human_score"] = score
        item["note"] = input("note (optional) > ").strip()
        _save(data)          # after every answer, so quitting loses nothing

    remaining = sum(1 for i in items if i["human_score"] is None)
    print(f"\nsaved. {len(items) - remaining} labelled, {remaining} to go")


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def agreement(pairs: list[tuple[float, float]], kind: str,
              tolerance: float = TOLERANCE) -> dict[str, Any]:
    """Judge-versus-human agreement for one metric.

    `pairs` is (judge, human). Returns None for any statistic the sample cannot
    support rather than a number that looks like a measurement, which is the
    same rule aggregate() follows.
    """
    n = len(pairs)
    if n == 0:
        return {"n": 0, "agreement": None, "mean_abs_diff": None,
                "kappa": None, "judge_mean": None, "human_mean": None}

    judge = [p[0] for p in pairs]
    human = [p[1] for p in pairs]
    diffs = [abs(j - h) for j, h in pairs]

    if kind == "binary":
        agree = sum(1 for j, h in pairs if _bin(j) == _bin(h))
    else:
        agree = sum(1 for d in diffs if d <= tolerance)

    return {
        "n": n,
        "agreement": round(agree / n, 3),
        "mean_abs_diff": round(sum(diffs) / n, 3),
        "kappa": cohens_kappa(pairs),
        "judge_mean": round(sum(judge) / n, 3),
        "human_mean": round(sum(human) / n, 3),
        # Signed, so a judge that is consistently generous reads differently
        # from one that is merely noisy. Both show up as the same mean_abs_diff.
        "judge_bias": round((sum(judge) - sum(human)) / n, 3),
    }


def _bin(x: float) -> int:
    return 1 if x >= 0.5 else 0


def cohens_kappa(pairs: list[tuple[float, float]]) -> float | None:
    """Agreement above what two raters would reach by chance, on 0/1 labels.

    Raw agreement flatters any metric with a skewed distribution: if 9 of 11
    answers are faithful, a judge that says "faithful" every time scores 0.82
    while carrying no information. Kappa removes that floor.

    Returns None when every label falls in one class, because chance agreement
    is then 1.0 and the statistic is 0/0. That is a real limitation of a small
    skewed sample, not something to paper over with a 0.
    """
    n = len(pairs)
    if n == 0:
        return None
    a = [_bin(p[0]) for p in pairs]
    b = [_bin(p[1]) for p in pairs]

    po = sum(1 for x, y in zip(a, b) if x == y) / n
    p1 = (sum(a) / n) * (sum(b) / n)
    p0 = ((n - sum(a)) / n) * ((n - sum(b)) / n)
    pe = p1 + p0
    if abs(1.0 - pe) < 1e-9:
        return None
    return round((po - pe) / (1 - pe), 3)


def report() -> dict:
    data = _load()
    labelled = [i for i in data.get("items", []) if i["human_score"] is not None]
    total = len(data.get("items", []))

    if not labelled:
        print("no labels yet - run `python eval/calibrate.py label` first")
        return {}

    out: dict[str, Any] = {
        "labelled": len(labelled),
        "of_total": total,
        "tolerance": data.get("tolerance", TOLERANCE),
        "by_metric": {},
    }

    for metric, spec in METRICS.items():
        pairs = [(i["judge_score"], i["human_score"])
                 for i in labelled if i["metric"] == metric]
        if pairs:
            out["by_metric"][metric] = agreement(pairs, spec["kind"],
                                                 out["tolerance"])

    # The disagreements are the output. An agreement rate says how often the
    # judge is close; these say what it is wrong about, which is the part that
    # can be acted on.
    out["worst"] = sorted(
        ({"key": i["key"], "metric": i["metric"],
          "judge": i["judge_score"], "human": i["human_score"],
          "diff": round(abs(i["judge_score"] - i["human_score"]), 3),
          "note": i["note"]}
         for i in labelled),
        key=lambda d: -d["diff"],
    )[:10]

    _print_report(out)
    return out


def _print_report(out: dict) -> None:
    print(f"\nJudge calibration: {out['labelled']} of {out['of_total']} cases labelled")
    print(f"Continuous metrics count as agreeing within {out['tolerance']}\n")
    print(f"{'metric':<20}{'n':>4}{'agree':>8}{'MAD':>8}{'kappa':>8}{'bias':>8}")
    print("-" * 56)
    for metric, s in out["by_metric"].items():
        kappa = "n/a" if s["kappa"] is None else f"{s['kappa']:.3f}"
        print(f"{metric:<20}{s['n']:>4}{s['agreement']:>8.3f}"
              f"{s['mean_abs_diff']:>8.3f}{kappa:>8}{s['judge_bias']:>+8.3f}")

    print("\nLargest disagreements")
    for d in out["worst"][:5]:
        print(f"  {d['key']:<28} judge {d['judge']:.2f}  human {d['human']:.2f}"
              f"  ({d['diff']:.2f})")
        if d["note"]:
            print(f"      {d['note']}")

    if any(s["n"] < 10 for s in out["by_metric"].values()):
        print("\nSome metrics have fewer than 10 labels. Report the n with the "
              "number or do not report the number.")


# ---------------------------------------------------------------------------

def _load() -> dict:
    if LABELS.exists():
        return json.loads(LABELS.read_text(encoding="utf-8"))
    return {"items": []}


def _save(data: dict) -> None:
    LABELS.parent.mkdir(parents=True, exist_ok=True)
    LABELS.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    cmd = sys.argv[1]
    if cmd == "extract":
        if len(sys.argv) < 3:
            print("usage: calibrate.py extract eval/results/<run>/results.json")
            return 1
        path = Path(sys.argv[2])
        if not path.exists():
            print(f"no such file: {path}")
            return 1
        extract(path)
    elif cmd == "label":
        label()
    elif cmd == "report":
        report()
    else:
        print(f"unknown command: {cmd}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
