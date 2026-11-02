# SPDX-License-Identifier: Apache-2.0
"""Summarize paired quality results without discarding individual regressions."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--baseline", default="full")
    parser.add_argument("--candidate", default="kvzap")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(s) for s in args.input.read_text().splitlines()]
    cases = {}
    for row in rows:
        if row["event"] != "quality":
            continue
        case = cases.setdefault(row["id"], {})
        assert row["mode"] not in case, (row["id"], row["mode"])
        case[row["mode"]] = row
    if args.manifest:
        manifest = json.loads(args.manifest.read_text())
        assert set(cases) == {
            s["id"] for s in manifest["samples"]
        }, "Incomplete evaluation"
    pairs = []
    for name, modes in cases.items():
        a, b = modes[args.baseline], modes[args.candidate]
        assert a["prompt_sha256"] == b["prompt_sha256"]
        pairs.append(
            {
                "id": name,
                "dataset": a["dataset"],
                "baseline": a["score"],
                "candidate": b["score"],
                "delta_pp": 100 * (b["score"] - a["score"]),
                "same_output": a["text"] == b["text"],
                "same_first_logprobs": a["first_logprobs_sha256"]
                == b["first_logprobs_sha256"],
                "baseline_text": a["text"],
                "candidate_text": b["text"],
                "answers": a["answers"],
            }
        )
    rng = np.random.default_rng(20261102)

    def aggregate(items):
        delta = np.array([p["delta_pp"] for p in items])
        bootstrap = rng.choice(delta, (10000, len(delta)), replace=True).mean(axis=1)
        return {
            "n": len(items),
            "baseline": float(np.mean([p["baseline"] for p in items]) * 100),
            "candidate": float(np.mean([p["candidate"] for p in items]) * 100),
            "delta_pp": float(delta.mean()),
            "paired_bootstrap_95_ci_pp": np.quantile(
                bootstrap, [0.025, 0.975]
            ).tolist(),
            "identical_outputs": sum(p["same_output"] for p in items),
            "identical_first_logprobs": sum(p["same_first_logprobs"] for p in items),
        }

    summary = {
        "input": str(args.input),
        "baseline": args.baseline,
        "candidate": args.candidate,
        "overall": aggregate(pairs),
        "datasets": {
            d: aggregate([p for p in pairs if p["dataset"] == d])
            for d in sorted({p["dataset"] for p in pairs})
        },
        "regressions": sorted(
            [p for p in pairs if p["delta_pp"] < -1e-9], key=lambda p: p["delta_pp"]
        ),
        "improvements": sorted(
            [p for p in pairs if p["delta_pp"] > 1e-9], key=lambda p: -p["delta_pp"]
        ),
        "limitations": "Paired regression screen, not the full official LongBench evaluation. Bootstrap intervals reflect only this sample and model; they do not establish universal non-inferiority.",
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {"overall": summary["overall"], "datasets": summary["datasets"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
