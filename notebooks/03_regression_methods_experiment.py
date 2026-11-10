"""Experiment 03 — fixed threshold vs paired-CI regression decisions.

Question: when a stochastic agent is evaluated repeatedly, which
regression *decision method* is more trustworthy?

  (a) fixed_threshold — FAIL iff the run-level mean drop on the primary
      metric exceeds the quality gate's tolerance (0.05), nothing else.
  (b) paired_ci       — the noise discipline compare_runs applies,
      isolated as a method: FAIL iff the mean paired per-case delta
      exceeds the same tolerance AND the 95% CI on that delta excludes
      no-change in the bad direction.

Both methods judge the same pair of runs on the same primary metric
(required_facts), so the comparison is apples-to-apples. Three
conditions, 12 repetitions each (fresh baseline + fresh candidate run
per repetition):

  healthy  baseline p_drop=0.05 vs candidate p_drop=0.05 — same
           distribution, so any FAIL is a false alarm.
  subtle   baseline p_drop=0.05 vs candidate p_drop=0.20 — expected
           required_facts drop of about 0.15: small but real.
  clear    baseline p_drop=0.05 vs candidate degraded=True (plain
           MockAdapter behaviour: facts truncated to one, citations
           dropped) — an obvious regression both methods should catch.

Why a stochastic adapter and a generated dataset?

  MockAdapter is deterministic, so repeated runs have zero variance
  and there is nothing for a CI to discipline. StochasticMockAdapter
  (defined below) drops each fact independently with probability
  p_drop on every call, which is what makes repeated runs differ.

  datasets/mock.jsonl has only 3 cases — too few paired cases, and
  scores too coarse, for a paired-CI comparison. This script therefore
  generates a dedicated 12-case dataset (4 mock_facts per case) and
  writes it to a temp file loaded through the normal dataset engine.
  It is deliberately NOT added to datasets/: versioned golden datasets
  stay hand-reviewed, and a synthetic experiment fixture should not
  quietly become one.

Everything runs against a tempfile Store (never the repo's
eval_platform.db), with a fixed master seed, so re-running this script
reproduces identical numbers.

Run: python notebooks/03_regression_methods_experiment.py
Outputs: results/regression_methods.csv (+ .png if matplotlib exists)
"""
from __future__ import annotations

import csv
import json
import random
import sys
import tempfile
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval_platform.adapter import AdapterRegistry          # noqa: E402
from eval_platform.adapters.mock import MockAdapter        # noqa: E402
from eval_platform.orchestrator import Orchestrator        # noqa: E402
from eval_platform.regression import _ci95, _metric_scores  # noqa: E402
from eval_platform.store import Store                      # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

MASTER_SEED = 20261110
REPETITIONS = 12
PRIMARY_METRIC = "required_facts"
TOLERANCE = 0.05          # quality_gates.json maximum_regression
BASELINE_P_DROP = 0.05

# (condition name, candidate p_drop or None, candidate degraded?)
CONDITIONS = (
    ("healthy", 0.05, False),
    ("subtle", 0.20, False),
    ("clear", None, True),
)


class StochasticMockAdapter(MockAdapter):
    """MockAdapter with independent per-fact drop noise.

    On each run() call, every fact in input_data["mock_facts"] is kept
    independently with probability (1 - p_drop) — order preserved, and
    keeping zero facts is allowed — then the rewritten input is handed
    to the deterministic MockAdapter logic. The adapter owns a seeded
    random.Random that advances on every call, so repeated evaluations
    of the same case differ, while the whole experiment stays
    reproducible from the master seed. Abstention cases (mock_abstain)
    pass through untouched: abstention is a behaviour, not noise.
    """

    def __init__(self, version: str = "1.0.0", p_drop: float = 0.05,
                 rng: random.Random | None = None, **kwargs):
        super().__init__(version=version, **kwargs)
        self.p_drop = p_drop
        self.rng = rng if rng is not None else random.Random(0)

    def run(self, input_data: dict) -> dict:
        if input_data.get("mock_abstain"):
            return super().run(input_data)
        facts = input_data.get("mock_facts")
        if facts is None:
            return super().run(input_data)
        kept = [fact for fact in facts if self.rng.random() >= self.p_drop]
        rewritten = dict(input_data)
        rewritten["mock_facts"] = kept
        return super().run(rewritten)


# ----------------------------------------------------------------------
# experiment dataset: 12 cases x 4 facts, distinctive required substrings
# ----------------------------------------------------------------------

_CASE_SPECS = [
    ("What are the central library rules?", "easy",
     ["The central library opens at seven thirty in the morning on weekdays",
      "Students may borrow up to eight books at a time from the central library",
      "Overdue books incur a fine of fifty rupees per day at the central library",
      "The central library reading room seats one hundred twenty students"],
     ["seven thirty in the morning", "eight books at a time",
      "fifty rupees per day", "one hundred twenty students"]),
    ("What services does the dental clinic offer?", "medium",
     ["The dental clinic offers free checkups every first Monday of the month",
      "Braces consultations at the dental clinic require a referral letter",
      "The dental clinic sterilises all instruments after every single patient",
      "Evening appointments at the dental clinic run until nine o clock"],
     ["free checkups every first Monday", "referral letter",
      "sterilises all instruments", "until nine o clock"]),
    ("How is the north warehouse operated?", "hard",
     ["Forklift operators in the north warehouse must hold a category B licence",
      "The north warehouse stores perishable goods below four degrees Celsius",
      "Inventory counts in the north warehouse happen every Friday afternoon",
      "Hazardous materials are kept in the sealed yellow zone of the warehouse"],
     ["category B licence", "below four degrees Celsius",
      "every Friday afternoon", "sealed yellow zone"]),
    ("Tell me about the green line tram service.", "easy",
     ["The green line tram runs every twelve minutes during peak hours",
      "Senior citizens ride the green line tram free of charge on Sundays",
      "The green line tram depot is located beside the old market square",
      "Night service on the green line ends at one fifteen in the morning"],
     ["every twelve minutes", "free of charge on Sundays",
      "old market square", "one fifteen in the morning"]),
    ("What should customers know about Sunrise Bakery?", "medium",
     ["The Sunrise Bakery bakes sourdough loaves twice each morning",
      "Gluten free bread at Sunrise Bakery is prepared in a separate oven",
      "Sunrise Bakery donates unsold pastries to the food bank each evening",
      "The bakery master class at Sunrise Bakery costs ninety euros per person"],
     ["sourdough loaves twice", "separate oven",
      "food bank each evening", "ninety euros per person"]),
    ("What happens at the hilltop observatory?", "hard",
     ["The hilltop observatory houses a forty centimetre reflecting telescope",
      "Public viewing nights at the observatory are held on clear Fridays",
      "The observatory planetarium seats eighty visitors per show",
      "Astronomers at the observatory track near earth asteroids nightly"],
     ["forty centimetre reflecting telescope", "clear Fridays",
      "eighty visitors per show", "near earth asteroids"]),
    ("What are the main post office services?", "easy",
     ["The main post office counters close at six o clock on Saturdays",
      "International parcels at the main post office need a customs declaration form",
      "The main post office offers same day delivery within the city centre",
      "Stamps at the main post office vending machine cost one euro each"],
     ["six o clock on Saturdays", "customs declaration form",
      "same day delivery within the city centre", "one euro each"]),
    ("How does the Willow Farm cooperative work?", "medium",
     ["The Willow Farm cooperative grows organic tomatoes on twelve hectares",
      "Willow Farm delivers vegetable boxes every Wednesday to subscribers",
      "The farm shop at Willow Farm sells honey from its own beehives",
      "Willow Farm irrigates its fields with recycled rainwater only"],
     ["organic tomatoes on twelve hectares", "vegetable boxes every Wednesday",
      "honey from its own beehives", "recycled rainwater only"]),
    ("What is on at the Grand Theatre?", "hard",
     ["The Grand Theatre stage was renovated with reclaimed oak flooring",
      "Grand Theatre matinee tickets cost half price for students",
      "The Grand Theatre orchestra pit holds thirty five musicians",
      "Subtitles at the Grand Theatre are shown in three languages"],
     ["reclaimed oak flooring", "half price for students",
      "thirty five musicians", "three languages"]),
    ("What facilities does Riverside Sports Centre have?", "easy",
     ["The Riverside Sports Centre pool opens for lane swimming at six am",
      "Members of Riverside Sports Centre get two free guest passes monthly",
      "The climbing wall at Riverside Sports Centre is fourteen metres high",
      "Yoga classes at Riverside Sports Centre run in Studio B on Tuesdays"],
     ["lane swimming at six am", "two free guest passes",
      "fourteen metres high", "Studio B on Tuesdays"]),
    ("What can visitors see at the City Museum?", "medium",
     ["The City Museum displays a bronze astrolabe from the twelfth century",
      "City Museum guided tours leave the lobby every hour on the hour",
      "Photography inside the City Museum textile gallery is strictly forbidden",
      "The City Museum cafe serves lunch until three thirty in the afternoon"],
     ["bronze astrolabe", "every hour on the hour",
      "textile gallery", "three thirty in the afternoon"]),
    ("What does the Harbour Hotel offer guests?", "hard",
     ["The Harbour Hotel provides a free shuttle to the ferry terminal",
      "Breakfast at the Harbour Hotel is served from five thirty onwards",
      "The Harbour Hotel rooftop pool closes during thunderstorms",
      "Late checkout at the Harbour Hotel is available until two pm"],
     ["free shuttle to the ferry terminal", "five thirty onwards",
      "rooftop pool closes", "until two pm"]),
]


def build_dataset_cases() -> list[dict]:
    cases = []
    for idx, (question, difficulty, facts, required) in enumerate(_CASE_SPECS, 1):
        if idx % 2 == 0:
            sources = [
                {"title": f"Guide p.{idx}a",
                 "text": facts[0] + ". " + facts[1] + "."},
                {"title": f"Guide p.{idx}b",
                 "text": facts[2] + ". " + facts[3] + "."},
            ]
        else:
            sources = [{"title": f"Guide p.{idx}",
                        "text": ". ".join(facts) + "."}]
        cases.append({
            "id": f"exp3-{idx:03d}",
            "project": "mock",
            "input": {"question": question, "mock_facts": list(facts),
                      "mock_sources": sources},
            "expected": {"required_facts": list(required),
                         "must_include_citations": True,
                         "allowed_tools": ["search", "answer"]},
            "category": "factual_qa",
            "difficulty": difficulty,
            "tags": ["regression-methods-experiment"],
        })
    return cases


def write_dataset(path: Path) -> Path:
    with path.open("w", encoding="utf-8") as fh:
        for case in build_dataset_cases():
            fh.write(json.dumps(case) + "\n")
    return path


# ----------------------------------------------------------------------
# run + decision helpers
# ----------------------------------------------------------------------

def run(store, adapter, dataset_path):
    registry = AdapterRegistry()
    registry.register(adapter)
    return Orchestrator(store, registry).run_evaluation(
        "mock-agent", dataset_path, profile="standard")


def _seed(condition_index: int, repetition: int, role: int) -> int:
    """role: 0 = baseline, 1 = candidate."""
    return MASTER_SEED + condition_index * 10000 + repetition * 100 + role


def per_case_means(store, run_id: int) -> dict[str, float]:
    scores = _metric_scores(store, run_id).get(PRIMARY_METRIC, {})
    return {case: mean(vals) for case, vals in scores.items()}


def decide(store, baseline: dict, candidate: dict) -> dict:
    base_cases = per_case_means(store, baseline["run_id"])
    cand_cases = per_case_means(store, candidate["run_id"])
    baseline_mean = mean(base_cases.values())
    candidate_mean = mean(cand_cases.values())
    paired_keys = sorted(set(base_cases) & set(cand_cases))
    deltas = [cand_cases[k] - base_cases[k] for k in paired_keys]
    mean_delta, ci_half = _ci95(deltas)

    run_level_delta = candidate_mean - baseline_mean
    fixed = "FAIL" if run_level_delta < -TOLERANCE else "PASS"
    paired = "FAIL" if (mean_delta < -TOLERANCE
                        and mean_delta + ci_half < 0) else "PASS"
    return {"baseline_mean": baseline_mean, "candidate_mean": candidate_mean,
            "mean_delta": mean_delta, "paired_ci_half_width": ci_half,
            "fixed_threshold_decision": fixed, "paired_ci_decision": paired}


def plot(rows: list[dict], path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping regression_methods.png")
        return False
    conditions = [name for name, _, _ in CONDITIONS]
    fig, axes = plt.subplots(1, len(conditions), sharey=True,
                             figsize=(11, 3.6))
    for ax, condition in zip(axes, conditions):
        subset = [r for r in rows if r["condition"] == condition]
        xs = [r["repetition"] for r in subset]
        ys = [r["mean_delta"] for r in subset]
        errs = [r["paired_ci_half_width"] for r in subset]
        ax.errorbar(xs, ys, yerr=errs, fmt="o", ms=4, capsize=3,
                    color="#1f5fa6", ecolor="#7aa7d9")
        ax.axhline(0.0, color="black", lw=1)
        ax.axhline(-TOLERANCE, color="#c0392b", lw=1, ls="--",
                   label="tolerance -0.05")
        ax.set_title(condition)
        ax.set_xlabel("repetition")
        ax.set_xticks(xs)
    axes[0].set_ylabel(f"mean delta ({PRIMARY_METRIC}, candidate - baseline)")
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("Experiment 03: paired mean delta per repetition, 95% CI whiskers")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def main() -> None:
    tmpdir = Path(tempfile.mkdtemp())
    store = Store(tmpdir / "experiment.db")
    dataset_path = write_dataset(tmpdir / "regression_methods.jsonl")

    rows: list[dict] = []
    for cond_idx, (condition, cand_p_drop, cand_degraded) in enumerate(CONDITIONS):
        for rep in range(REPETITIONS):
            baseline_adapter = StochasticMockAdapter(
                version="1.0.0", p_drop=BASELINE_P_DROP,
                rng=random.Random(_seed(cond_idx, rep, 0)))
            baseline = run(store, baseline_adapter, dataset_path)
            if cand_degraded:
                candidate_adapter = MockAdapter(version="1.1.0", degraded=True)
            else:
                candidate_adapter = StochasticMockAdapter(
                    version="1.0.1", p_drop=cand_p_drop,
                    rng=random.Random(_seed(cond_idx, rep, 1)))
            candidate = run(store, candidate_adapter, dataset_path)
            outcome = decide(store, baseline, candidate)
            rows.append({
                "condition": condition,
                "repetition": rep,
                "baseline_mean_required_facts": round(outcome["baseline_mean"], 4),
                "candidate_mean_required_facts": round(outcome["candidate_mean"], 4),
                "mean_delta": round(outcome["mean_delta"], 4),
                "paired_ci_half_width": round(outcome["paired_ci_half_width"], 4),
                "fixed_threshold_decision": outcome["fixed_threshold_decision"],
                "paired_ci_decision": outcome["paired_ci_decision"],
                "baseline_pass_rate": baseline["pass_rate"],
                "candidate_pass_rate": candidate["pass_rate"],
            })
            print(f"{condition} rep {rep}: baseline="
                  f"{rows[-1]['baseline_mean_required_facts']} candidate="
                  f"{rows[-1]['candidate_mean_required_facts']} delta="
                  f"{rows[-1]['mean_delta']:+.4f} ci_half="
                  f"{rows[-1]['paired_ci_half_width']:.4f} fixed="
                  f"{outcome['fixed_threshold_decision']} paired_ci="
                  f"{outcome['paired_ci_decision']}")

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    csv_path = results_dir / "regression_methods.csv"
    fieldnames = ["condition", "repetition", "baseline_mean_required_facts",
                  "candidate_mean_required_facts", "mean_delta",
                  "paired_ci_half_width", "fixed_threshold_decision",
                  "paired_ci_decision", "baseline_pass_rate",
                  "candidate_pass_rate"]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {csv_path}")

    png_path = results_dir / "regression_methods.png"
    if plot(rows, png_path):
        print(f"wrote {png_path}")

    print("\nPer-condition summary (12 repetitions each):")
    print(f"{'condition':<10} {'method':<16} {'FAILs':>5}  interpretation")
    for condition, _, _ in CONDITIONS:
        subset = [r for r in rows if r["condition"] == condition]
        for method, key in (("fixed_threshold", "fixed_threshold_decision"),
                            ("paired_ci", "paired_ci_decision")):
            fails = sum(1 for r in subset if r[key] == "FAIL")
            rate = fails / len(subset)
            if condition == "healthy":
                label = f"false-alarm rate {rate:.1%}"
            else:
                label = f"detection rate {rate:.1%}"
            print(f"{condition:<10} {method:<16} {fails:>5}  {label}")


if __name__ == "__main__":
    main()
