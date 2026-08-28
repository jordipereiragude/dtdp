"""Generate the tables and reported numerical text used in the manuscript.

Run this file directly with ``--tables``, ``--text``, or both. All inputs and
outputs are resolved relative to this file, so the command is independent of
the caller's working directory.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import heapq
import math
import statistics
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
PAPER_DIR = SCRIPT_DIR

import numpy as np
import pandas as pd
import re


def _build_ablation_helpers() -> SimpleNamespace:
    HERE = Path(__file__).resolve().parent

    ROOT = HERE.parent

    TIME_LIMITS = (60, 300, 600)

    EXPECTED_REPLICAS = 30

    EXPECTED_INSTANCE_COUNT = 120

    MAIN_FILES = (ROOT / "exp05" / "out.txt", HERE / "out.txt")

    INSTANCE_LIST_FILES = (
        ROOT / "exp05" / "instance-list.txt",
        HERE / "instance-list.txt",
    )

    FIXED_FILES = (
        ROOT / "exp05" / "out-fixed-60s.txt",
        ROOT / "exp05" / "out-fixed-300s.txt",
        ROOT / "exp05" / "out-fixed-600s.txt",
        HERE / "out-fixed.txt",
    )

    MAIN_SCHEDULES = {
        60: (30, 18, 68),
        300: (38, 24, 82),
        600: (31, 6, 83),
    }

    FIXED_SCHEDULES = {
        60: (45, 75, 45),
        300: (60, 75, 60),
        600: (55, 75, 55),
    }

    ABLATION_FLAGS = (
        "randomDestroy",
        "disableVariableFixing",
        "useMultistart",
        "usekDestroy",
        "useAlternativeDestroy",
        "noWarmstart",
        "heuristicPercentage",
    )

    VARIANTS = (
        ("Random destroy", "randomDestroy"),
        ("No fixing", "disableVariableFixing"),
        ("No warm start", "noWarmstart"),
        ("Only random start", "useMultistart"),
        ("Alternative destroy", "useAlternativeDestroy"),
        ("District-based destroy", "usekDestroy"),
        ("Heuristic repair", "heuristicPercentage"),
    )

    COMPARISON_LABELS = ("Fixed destroy size", *(label for label, _ in VARIANTS))

    @dataclass(frozen=True)
    class Run:
        instance: str
        seed: int
        minimum: int
        step: int
        maximum: int
        time_limit: int
        objective: float
        penalty: int
        time_best: float
        free_variables: int
        fixed_variables: int
        solving_repairs_time: float
        optimal_repairs: int
        repairs: int
        balance_improvements: int
        compactness_improvements: int
        flags: tuple[str, ...]

        @property
        def key(self) -> tuple[str, int, int]:
            return self.instance, self.seed, self.time_limit

        @property
        def score(self) -> tuple[int, float]:
            return self.penalty, self.objective

    @dataclass(frozen=True)
    class VanElterenResult:
        time_limit: int
        label: str
        strata: int
        mean_superiority: float
        full_favored: int
        neutral: int
        variant_favored: int
        z_statistic: float
        p_value: float
        instance_superiority: tuple[float, ...]

    def parse_run(line: str, source: Path, line_number: int) -> Run:
        fields = line.split()
        if len(fields) < 4 or fields[0] != "RESULTS:":
            raise ValueError(f"{source}:{line_number}: malformed result line")

        tokens = fields[2:]
        if len(tokens) % 2:
            raise ValueError(f"{source}:{line_number}: unmatched key/value field")
        values = {
            tokens[index].removesuffix(":"): tokens[index + 1]
            for index in range(0, len(tokens), 2)
        }
        try:
            return Run(
                instance=Path(fields[1]).stem,
                seed=int(values["seed"]),
                minimum=int(values["min"]),
                step=int(values["step"]),
                maximum=int(values["max"]),
                time_limit=int(values["tLim"]),
                objective=float(values["obj"]),
                penalty=int(values["penalty"]),
                time_best=float(values["tBest"]),
                free_variables=int(values["free"]),
                fixed_variables=int(values["fixed"]),
                solving_repairs_time=float(values["tSolvingRepairs"]),
                optimal_repairs=int(values["numRepairsOptimality"]),
                repairs=int(values["numRepairs"]),
                balance_improvements=int(values["numImprovesBalance"]),
                compactness_improvements=int(values["numImprovesCompactness"]),
                flags=tuple(values[flag] for flag in ABLATION_FLAGS),
            )
        except (KeyError, ValueError) as error:
            raise ValueError(f"{source}:{line_number}: invalid result field: {error}") from error

    def read_runs(source: Path) -> list[Run]:
        return [
            parse_run(line, source, line_number)
            for line_number, line in enumerate(source.read_text().splitlines(), start=1)
            if line.strip()
        ]

    def read_all_runs(sources: Iterable[Path]) -> list[Run]:
        return [run for source in sources for run in read_runs(source)]

    def expected_instances() -> set[str]:
        instance_lists = []
        for source in INSTANCE_LIST_FILES:
            instances = {
                Path(line).stem
                for raw_line in source.read_text().splitlines()
                if (line := raw_line.strip()) and not line.startswith("#")
            }
            instance_lists.append(instances)

        overlap = instance_lists[0] & instance_lists[1]
        if overlap:
            raise ValueError(
                f"exp05 and exp08 instance lists overlap on {len(overlap)} instances"
            )
        combined = set().union(*instance_lists)
        if len(combined) != EXPECTED_INSTANCE_COUNT:
            raise ValueError(
                f"expected {EXPECTED_INSTANCE_COUNT} combined instances, "
                f"found {len(combined)}"
            )
        return combined

    def active_ablation_flag(run: Run) -> str | None:
        if any(value not in {"true", "false"} for value in run.flags):
            raise ValueError(f"{run.key}: invalid ablation flag value")
        active = [
            flag
            for flag, value in zip(ABLATION_FLAGS, run.flags)
            if value == "true"
        ]
        if len(active) > 1:
            raise ValueError(f"{run.key}: multiple ablation flags are active")
        return active[0] if active else None

    def load_main_runs() -> list[Run]:
        runs = read_all_runs(MAIN_FILES)
        instances = expected_instances()
        for run in runs:
            if run.instance not in instances:
                raise ValueError(f"unexpected main-run instance: {run.instance}")
            if run.time_limit not in TIME_LIMITS:
                raise ValueError(f"{run.key}: unexpected time limit")
            if (run.minimum, run.step, run.maximum) != MAIN_SCHEDULES[run.time_limit]:
                raise ValueError(f"{run.key}: unexpected increasing schedule")
            active_ablation_flag(run)
        return runs

    def load_fixed_runs() -> dict[int, list[Run]]:
        grouped = {time_limit: [] for time_limit in TIME_LIMITS}
        instances = expected_instances()
        for run in read_all_runs(FIXED_FILES):
            if run.instance not in instances:
                raise ValueError(f"unexpected fixed-run instance: {run.instance}")
            if run.time_limit not in TIME_LIMITS:
                raise ValueError(f"{run.key}: unexpected fixed-run time limit")
            if active_ablation_flag(run) is not None:
                raise ValueError(f"{run.key}: fixed run contains another ablation")
            if (run.minimum, run.step, run.maximum) != FIXED_SCHEDULES[run.time_limit]:
                raise ValueError(f"{run.key}: unexpected fixed-size schedule")
            grouped[run.time_limit].append(run)
        return grouped

    def load_method_runs() -> dict[tuple[int, str], list[Run]]:
        all_runs = load_main_runs()
        fixed_runs = load_fixed_runs()
        methods: dict[tuple[int, str], list[Run]] = {}
        for time_limit in TIME_LIMITS:
            methods[time_limit, "Full LNS"] = [
                run
                for run in all_runs
                if run.time_limit == time_limit and active_ablation_flag(run) is None
            ]
            methods[time_limit, "Fixed destroy size"] = fixed_runs[time_limit]
            for label, active_flag in VARIANTS:
                methods[time_limit, label] = [
                    run
                    for run in all_runs
                    if run.time_limit == time_limit
                    and active_ablation_flag(run) == active_flag
                ]

        validate_method_runs(methods)
        return methods

    def validate_method_runs(methods: dict[tuple[int, str], list[Run]]) -> None:
        instances = expected_instances()
        expected_seeds = set(range(1, EXPECTED_REPLICAS + 1))
        labels = ("Full LNS", *COMPARISON_LABELS)
        for time_limit in TIME_LIMITS:
            for label in labels:
                grouped: dict[str, list[int]] = defaultdict(list)
                for run in methods[time_limit, label]:
                    grouped[run.instance].append(run.seed)
                if set(grouped) != instances:
                    raise ValueError(
                        f"{label} at {time_limit} seconds does not cover all "
                        f"{EXPECTED_INSTANCE_COUNT} instances"
                    )
                for instance, seeds in grouped.items():
                    if len(seeds) != EXPECTED_REPLICAS or set(seeds) != expected_seeds:
                        raise ValueError(
                            f"{label}/{time_limit}/{instance}: expected seeds "
                            f"1--{EXPECTED_REPLICAS} exactly once"
                        )

    def mann_whitney_components(
        reference: list[Run], variant: list[Run]
    ) -> tuple[float, float, float]:
        """Return Full-LNS-favorable U, its null variance, and superiority."""
        n_reference = len(reference)
        n_variant = len(variant)
        total = n_reference + n_variant
        better = sum(
            reference_run.score < variant_run.score
            for reference_run in reference
            for variant_run in variant
        )
        ties = sum(
            reference_run.score == variant_run.score
            for reference_run in reference
            for variant_run in variant
        )
        favorable_u = better + 0.5 * ties
        pair_count = n_reference * n_variant

        tie_counts: dict[tuple[int, float], int] = defaultdict(int)
        for run in (*reference, *variant):
            tie_counts[run.score] += 1
        tie_term = sum(count**3 - count for count in tie_counts.values())
        variance = pair_count / 12.0 * (
            total + 1.0 - tie_term / (total * (total - 1.0))
        )
        return favorable_u, variance, favorable_u / pair_count

    def mann_whitney_p_value(
        favorable_u: float, variance: float, n_reference: int, n_variant: int
    ) -> float:
        if variance == 0.0:
            return 1.0
        expected = n_reference * n_variant / 2.0
        corrected_deviation = max(0.0, abs(favorable_u - expected) - 0.5)
        z_statistic = corrected_deviation / math.sqrt(variance)
        return math.erfc(z_statistic / math.sqrt(2.0))

    def significance_stars(p_value: float) -> str:
        if p_value < 0.001:
            return "***"
        if p_value < 0.01:
            return "**"
        if p_value < 0.05:
            return "*"
        return ""

    def van_elteren_results(
        methods: dict[tuple[int, str], list[Run]],
    ) -> list[VanElterenResult]:
        results = []
        for time_limit in TIME_LIMITS:
            for label in COMPARISON_LABELS:
                reference_by_instance: dict[str, list[Run]] = defaultdict(list)
                variant_by_instance: dict[str, list[Run]] = defaultdict(list)
                for run in methods[time_limit, "Full LNS"]:
                    reference_by_instance[run.instance].append(run)
                for run in methods[time_limit, label]:
                    variant_by_instance[run.instance].append(run)

                superiority = []
                weighted_centered = 0.0
                weighted_variance = 0.0
                for instance in sorted(reference_by_instance):
                    reference = reference_by_instance[instance]
                    variant = variant_by_instance[instance]
                    favorable_u, variance, effect = mann_whitney_components(
                        reference, variant
                    )
                    pair_count = len(reference) * len(variant)
                    total = len(reference) + len(variant)
                    superiority.append(effect)
                    weight = 1.0 / (total + 1.0)
                    weighted_centered += weight * (favorable_u - pair_count / 2.0)
                    weighted_variance += weight**2 * variance

                z_statistic = weighted_centered / math.sqrt(weighted_variance)
                p_value = math.erfc(abs(z_statistic) / math.sqrt(2.0))
                results.append(
                    VanElterenResult(
                        time_limit=time_limit,
                        label=label,
                        strata=len(superiority),
                        mean_superiority=statistics.mean(superiority),
                        full_favored=sum(value > 0.5 for value in superiority),
                        neutral=sum(value == 0.5 for value in superiority),
                        variant_favored=sum(value < 0.5 for value in superiority),
                        z_statistic=z_statistic,
                        p_value=p_value,
                        instance_superiority=tuple(superiority),
                    )
                )
        return results

    def holm_adjust(p_values: dict[tuple[int, str], float]) -> dict[tuple[int, str], float]:
        ordered = sorted(p_values, key=p_values.get)
        adjusted: dict[tuple[int, str], float] = {}
        running_max = 0.0
        total = len(ordered)
        for rank, key in enumerate(ordered):
            value = min(1.0, (total - rank) * p_values[key])
            running_max = max(running_max, value)
            adjusted[key] = running_max
        return adjusted

    def format_p_value(value: float) -> str:
        return "$<0.001$" if value < 0.001 else f"{value:.3f}"

    def feasible_objectives_by_instance(runs: Iterable[Run]) -> dict[str, list[float]]:
        grouped: dict[str, list[float]] = defaultdict(list)
        for run in runs:
            if run.penalty == 0:
                grouped[run.instance].append(run.objective)
        return grouped

    def instance_objective_gaps(
        methods: dict[tuple[int, str], list[Run]], time_limit: int, label: str
    ) -> list[float]:
        reference = feasible_objectives_by_instance(methods[time_limit, "Full LNS"])
        variant = feasible_objectives_by_instance(methods[time_limit, label])
        gaps = []
        for instance in sorted(reference):
            if not reference[instance] or not variant[instance]:
                continue
            reference_median = statistics.median(reference[instance])
            gaps.append(
                100.0
                * (statistics.median(variant[instance]) - reference_median)
                / reference_median
            )
        return gaps

    MAIN_FILES = (
        ROOT / "resultsLNS" / "lns.results.runs.txt",
        ROOT / "resultsLNS" / "lns.ablation.results.runs.txt",
    )
    FIXED_FILES = (ROOT / "resultsLNS" / "lns.fixed.results.runs.txt",)

    def expected_instances() -> set[str]:
        instances = {
            path.stem
            for group in ("group1", "group2")
            for path in (ROOT / "instances" / group).glob("*.graphml")
        }
        if len(instances) != EXPECTED_INSTANCE_COUNT:
            raise ValueError(
                f"expected {EXPECTED_INSTANCE_COUNT} instances, found {len(instances)}"
            )
        return instances

    return SimpleNamespace(
        EXPECTED_INSTANCE_COUNT=EXPECTED_INSTANCE_COUNT,
        EXPECTED_REPLICAS=EXPECTED_REPLICAS,
        FIXED_FILES=FIXED_FILES,
        FIXED_SCHEDULES=FIXED_SCHEDULES,
        MAIN_FILES=MAIN_FILES,
        MAIN_SCHEDULES=MAIN_SCHEDULES,
        Run=Run,
        TIME_LIMITS=TIME_LIMITS,
        expected_instances=expected_instances,
        format_p_value=format_p_value,
        holm_adjust=holm_adjust,
        instance_objective_gaps=instance_objective_gaps,
        load_method_runs=load_method_runs,
        mann_whitney_components=mann_whitney_components,
        mann_whitney_p_value=mann_whitney_p_value,
        significance_stars=significance_stars,
        van_elteren_results=van_elteren_results
    )


ablation = _build_ablation_helpers()


def _build_lns_vns_helpers() -> SimpleNamespace:
    EXPECTED_INSTANCE_COUNT = ablation.EXPECTED_INSTANCE_COUNT
    EXPECTED_REPLICAS = ablation.EXPECTED_REPLICAS
    Run = ablation.Run
    expected_instances = ablation.expected_instances
    holm_adjust = ablation.holm_adjust
    load_method_runs = ablation.load_method_runs
    mann_whitney_components = ablation.mann_whitney_components
    mann_whitney_p_value = ablation.mann_whitney_p_value
    significance_stars = ablation.significance_stars

    HERE = Path(__file__).resolve().parent

    ROOT = HERE.parent

    VNS_OUTPUTS = (
        ROOT / "exp05" / "out-vns.txt",
        HERE / "out-vns.txt",
    )

    LOWER_BOUND_OUTPUT = ROOT / "exp03" / "out.txt"

    COMPARISONS = (
        (60, "VNS(l)", "test6WithSeed.py"),
        (300, "VNS(m)", "test7WithSeed.py"),
        (600, "VNS(h)", "test8WithSeed.py"),
    )

    VNS_PATTERN = re.compile(
        r"Version: (?P<version>\S+) "
        r"Instance: (?P<instance>\S+) "
        r"Seed: (?P<seed>\d+) "
        r"bestObjective: (?P<objective>\S+) "
        r"infeasibility: (?P<penalty>\d+) "
        r"Check: (?P<check>\S+) "
        r"totalTime\(s\): (?P<runtime>\S+)"
    )

    LOWER_BOUND_PATTERN = re.compile(
        r"iterativeRelaxation: (?P<instance>\S+) value: (?P<value>\S+)"
    )

    @dataclass(frozen=True)
    class VnsRun:
        instance: str
        seed: int
        version: str
        objective: float
        penalty: int
        runtime: float

        @property
        def score(self) -> tuple[int, float]:
            return self.penalty, self.objective

    @dataclass(frozen=True)
    class ComparisonResult:
        time_limit: int
        vns_label: str
        lns_favored: int
        vns_favored: int
        z_statistic: float
        p_value: float
        median_compactness_difference: float
        q1_compactness_difference: float
        q3_compactness_difference: float
        median_all_instance_rcd: float
        q1_all_instance_rcd: float
        q3_all_instance_rcd: float
        lns_feasible_runs: float
        vns_feasible_runs: float

    @dataclass(frozen=True)
    class RuntimeSummary:
        method: str
        average: float
        q1: float
        median: float
        q3: float
        standard_deviation: float
        runs: int

    @dataclass(frozen=True)
    class BoundGapSummary:
        method: str
        instances: int
        runs: int
        feasible_runs: int
        minimum_feasible_runs_per_instance: int
        maximum_feasible_runs_per_instance: int
        mean_all_run_instance_gap: float
        median_all_run_instance_gap: float
        mean_feasible_run_instance_gap: float
        median_feasible_run_instance_gap: float

    def read_vns_runs() -> dict[str, list[VnsRun]]:
        instances = expected_instances()
        runs: dict[str, list[VnsRun]] = defaultdict(list)
        for source in VNS_OUTPUTS:
            for line_number, line in enumerate(
                source.read_text().splitlines(), start=1
            ):
                match = VNS_PATTERN.fullmatch(line)
                if not match:
                    raise ValueError(f"{source}:{line_number}: malformed VNS result")
                values = match.groupdict()
                if values["check"] != "OK":
                    raise ValueError(f"{source}:{line_number}: failed VNS result check")
                version = values["version"]
                instance = Path(values["instance"]).stem
                if instance not in instances:
                    raise ValueError(f"{source}:{line_number}: unexpected VNS instance")
                runs[version].append(
                    VnsRun(
                        instance=instance,
                        seed=int(values["seed"]),
                        version=version,
                        objective=float(values["objective"]),
                        penalty=int(values["penalty"]),
                        runtime=float(values["runtime"]),
                    )
                )

        expected_versions = {version for _, _, version in COMPARISONS}
        if set(runs) != expected_versions:
            raise ValueError("VNS outputs do not contain the expected variants")
        expected_keys = {
            (instance, seed)
            for instance in instances
            for seed in range(1, EXPECTED_REPLICAS + 1)
        }
        for version, version_runs in runs.items():
            keys = [(run.instance, run.seed) for run in version_runs]
            if len(keys) != len(set(keys)):
                raise ValueError(f"{version}: duplicate instance/seed run")
            actual_keys = set(keys)
            if actual_keys != expected_keys:
                raise ValueError(
                    f"{version}: {len(expected_keys - actual_keys)} missing and "
                    f"{len(actual_keys - expected_keys)} unexpected runs"
                )
        return dict(runs)

    def group_by_instance(
        runs: list[Run] | list[VnsRun],
    ) -> dict[str, list[Run] | list[VnsRun]]:
        grouped = defaultdict(list)
        for run in runs:
            grouped[run.instance].append(run)
        return dict(grouped)

    def read_instance_adjacency(
        source: Path,
    ) -> list[list[tuple[int, float]]]:
        tokens = iter(source.read_text().split())
        try:
            node_count = int(next(tokens))
            edge_count = int(next(tokens))
            resource_count = int(next(tokens))
            next(tokens)  # number of districts
            next(tokens)  # balance tolerance

            node_indices: dict[int, int] = {}
            for index in range(node_count):
                node_indices[int(next(tokens))] = index
                for _ in range(resource_count):
                    next(tokens)

            adjacency: list[list[tuple[int, float]]] = [
                [] for _ in range(node_count)
            ]
            for _ in range(edge_count):
                left = node_indices[int(next(tokens))]
                right = node_indices[int(next(tokens))]
                distance = float(next(tokens))
                adjacency[left].append((right, distance))
                adjacency[right].append((left, distance))
        except (KeyError, StopIteration, ValueError) as error:
            raise ValueError(f"{source}: malformed instance file") from error
        return adjacency

    def preceding_shortest_path_distance(
        source: Path,
        reported_threshold: float,
    ) -> float:
        """Recover the p-dispersion bound preceding the reported threshold."""
        adjacency = read_instance_adjacency(source)
        # exp03 prints the first infeasible distance threshold to three decimal
        # places. The p-dispersion bound is the preceding distinct all-pairs
        # shortest-path distance.
        cutoff = reported_threshold + 0.000501
        distances: set[float] = set()
        for start in range(len(adjacency)):
            shortest = [float("inf")] * len(adjacency)
            shortest[start] = 0.0
            queue = [(0.0, start)]
            while queue:
                distance, node = heapq.heappop(queue)
                if distance != shortest[node]:
                    continue
                if distance > cutoff:
                    continue
                distances.add(distance)
                for neighbor, edge_distance in adjacency[node]:
                    candidate = distance + edge_distance
                    if candidate < shortest[neighbor] and candidate <= cutoff:
                        shortest[neighbor] = candidate
                        heapq.heappush(queue, (candidate, neighbor))

        ordered_distances = sorted(distances)
        threshold_index = min(
            range(len(ordered_distances)),
            key=lambda index: abs(
                ordered_distances[index] - reported_threshold
            ),
        )
        recovered_threshold = ordered_distances[threshold_index]
        if abs(recovered_threshold - reported_threshold) > 0.000501:
            raise ValueError(
                f"{source}: cannot recover threshold {reported_threshold:.3f}"
            )
        if threshold_index == 0:
            raise ValueError(f"{source}: threshold has no preceding distance")
        return ordered_distances[threshold_index - 1]

    def read_discrete_dispersion_lower_bounds() -> dict[str, float]:
        instances = expected_instances()
        lower_bounds: dict[str, float] = {}
        for line_number, line in enumerate(
            LOWER_BOUND_OUTPUT.read_text().splitlines(), start=1
        ):
            match = LOWER_BOUND_PATTERN.fullmatch(line)
            if not match:
                raise ValueError(
                    f"{LOWER_BOUND_OUTPUT}:{line_number}: malformed lower-bound result"
                )
            instance = Path(match.group("instance")).stem
            if instance not in instances:
                continue
            if instance in lower_bounds:
                raise ValueError(f"duplicate lower bound for {instance}")
            instance_source = (
                LOWER_BOUND_OUTPUT.parent / match.group("instance")
            ).resolve()
            lower_bounds[instance] = preceding_shortest_path_distance(
                instance_source,
                float(match.group("value")),
            )

        if set(lower_bounds) != instances:
            raise ValueError(
                f"expected lower bounds for {len(instances)} instances, "
                f"found {len(lower_bounds)}"
            )
        return lower_bounds

    def summarize_bound_gaps(
        method: str,
        runs: list[Run] | list[VnsRun],
        lower_bounds: dict[str, float],
    ) -> BoundGapSummary:
        runs_by_instance = group_by_instance(runs)
        if set(runs_by_instance) != set(lower_bounds):
            raise ValueError(f"{method}: run and lower-bound instance sets differ")
        if any(run.objective <= 0.0 for run in runs):
            raise ValueError(f"{method}: found a nonpositive compactness value")

        feasible_runs = [run for run in runs if run.penalty == 0]
        feasible_counts = [
            sum(run.penalty == 0 for run in instance_runs)
            for instance_runs in runs_by_instance.values()
        ]
        all_run_instance_gaps = []
        feasible_run_instance_gaps = []
        for instance, instance_runs in runs_by_instance.items():
            median_all_runs = statistics.median(
                run.objective for run in instance_runs
            )
            all_run_instance_gaps.append(
                100.0
                * (median_all_runs - lower_bounds[instance])
                / median_all_runs
            )
            feasible_instance_runs = [
                run for run in instance_runs if run.penalty == 0
            ]
            if not feasible_instance_runs:
                raise ValueError(f"{method}: {instance} has no feasible runs")
            median_feasible_runs = statistics.median(
                run.objective for run in feasible_instance_runs
            )
            feasible_run_instance_gaps.append(
                100.0
                * (median_feasible_runs - lower_bounds[instance])
                / median_feasible_runs
            )
        return BoundGapSummary(
            method=method,
            instances=len(runs_by_instance),
            runs=len(runs),
            feasible_runs=len(feasible_runs),
            minimum_feasible_runs_per_instance=min(feasible_counts),
            maximum_feasible_runs_per_instance=max(feasible_counts),
            mean_all_run_instance_gap=statistics.mean(all_run_instance_gaps),
            median_all_run_instance_gap=statistics.median(
                all_run_instance_gaps
            ),
            mean_feasible_run_instance_gap=statistics.mean(
                feasible_run_instance_gaps
            ),
            median_feasible_run_instance_gap=statistics.median(
                feasible_run_instance_gaps
            ),
        )

    def bound_gap_summaries(
        methods: dict[tuple[int, str], list[Run]],
        vns_runs: dict[str, list[VnsRun]],
    ) -> list[BoundGapSummary]:
        lower_bounds = read_discrete_dispersion_lower_bounds()
        return [
            summarize_bound_gaps(
                "LNS (600 s)", methods[600, "Full LNS"], lower_bounds
            ),
            summarize_bound_gaps(
                "VNS(h)", vns_runs["test8WithSeed.py"], lower_bounds
            ),
        ]

    def compare_samples(
        time_limit: int,
        vns_label: str,
        lns_runs: list[Run],
        vns_runs: list[VnsRun],
    ) -> ComparisonResult:
        lns_by_instance = group_by_instance(lns_runs)
        vns_by_instance = group_by_instance(vns_runs)
        if lns_by_instance.keys() != vns_by_instance.keys():
            raise ValueError(
                f"LNS {time_limit} s vs {vns_label}: instance sets differ"
            )

        p_values: dict[str, float] = {}
        superiority: dict[str, float] = {}
        weighted_centered = 0.0
        weighted_variance = 0.0
        for instance in sorted(lns_by_instance):
            lns_sample = lns_by_instance[instance]
            vns_sample = vns_by_instance[instance]
            favorable_u, variance, effect = mann_whitney_components(
                lns_sample, vns_sample
            )
            p_values[instance] = mann_whitney_p_value(
                favorable_u, variance, len(lns_sample), len(vns_sample)
            )
            superiority[instance] = effect

            pair_count = len(lns_sample) * len(vns_sample)
            total = len(lns_sample) + len(vns_sample)
            weight = 1.0 / (total + 1.0)
            weighted_centered += weight * (favorable_u - pair_count / 2.0)
            weighted_variance += weight**2 * variance

        adjusted = holm_adjust(p_values)
        z_statistic = weighted_centered / math.sqrt(weighted_variance)
        p_value = math.erfc(abs(z_statistic) / math.sqrt(2.0))
        compactness_differences = []
        all_instance_rcds = []
        for instance in sorted(lns_by_instance):
            lns_median_all_runs = statistics.median(
                run.objective for run in lns_by_instance[instance]
            )
            vns_median_all_runs = statistics.median(
                run.objective for run in vns_by_instance[instance]
            )
            all_instance_rcds.append(
                100.0
                * (vns_median_all_runs - lns_median_all_runs)
                / lns_median_all_runs
            )

            lns_feasible = [
                run.objective
                for run in lns_by_instance[instance]
                if run.penalty == 0
            ]
            vns_feasible = [
                run.objective
                for run in vns_by_instance[instance]
                if run.penalty == 0
            ]
            if not lns_feasible or not vns_feasible:
                continue
            lns_median = statistics.median(lns_feasible)
            compactness_differences.append(
                100.0
                * (statistics.median(vns_feasible) - lns_median)
                / lns_median
            )
        q1, _, q3 = statistics.quantiles(
            compactness_differences, n=4, method="inclusive"
        )
        q1_all, _, q3_all = statistics.quantiles(
            all_instance_rcds, n=4, method="inclusive"
        )
        return ComparisonResult(
            time_limit=time_limit,
            vns_label=vns_label,
            lns_favored=sum(
                adjusted[instance] < 0.05 and superiority[instance] > 0.5
                for instance in adjusted
            ),
            vns_favored=sum(
                adjusted[instance] < 0.05 and superiority[instance] < 0.5
                for instance in adjusted
            ),
            z_statistic=z_statistic,
            p_value=p_value,
            median_compactness_difference=statistics.median(
                compactness_differences
            ),
            q1_compactness_difference=q1,
            q3_compactness_difference=q3,
            median_all_instance_rcd=statistics.median(all_instance_rcds),
            q1_all_instance_rcd=q1_all,
            q3_all_instance_rcd=q3_all,
            lns_feasible_runs=100.0
            * sum(run.penalty == 0 for run in lns_runs)
            / len(lns_runs),
            vns_feasible_runs=100.0
            * sum(run.penalty == 0 for run in vns_runs)
            / len(vns_runs),
        )

    def format_p_value(value: float) -> str:
        return "<0.001" if value < 0.001 else f"{value:.3f}"

    def format_z_value(value: float, p_value: float) -> str:
        stars = significance_stars(p_value)
        suffix = f"^{{{stars}}}" if stars else ""
        return "$" + f"{value:.3f}" + suffix + "$"

    def pairwise_van_elteren(
        reference_runs: list[Run] | list[VnsRun],
        comparison_runs: list[Run] | list[VnsRun],
    ) -> tuple[float, float]:
        """Return the van Elteren z statistic and two-sided p-value."""
        reference_by_instance = group_by_instance(reference_runs)
        comparison_by_instance = group_by_instance(comparison_runs)
        if reference_by_instance.keys() != comparison_by_instance.keys():
            raise ValueError("pairwise van Elteren comparison uses different instances")

        weighted_centered = 0.0
        weighted_variance = 0.0
        for instance in sorted(reference_by_instance):
            reference_sample = reference_by_instance[instance]
            comparison_sample = comparison_by_instance[instance]
            favorable_u, variance, _ = mann_whitney_components(
                reference_sample, comparison_sample
            )
            pair_count = len(reference_sample) * len(comparison_sample)
            total = len(reference_sample) + len(comparison_sample)
            weight = 1.0 / (total + 1.0)
            weighted_centered += weight * (favorable_u - pair_count / 2.0)
            weighted_variance += weight**2 * variance

        z_statistic = weighted_centered / math.sqrt(weighted_variance)
        p_value = math.erfc(abs(z_statistic) / math.sqrt(2.0))
        return z_statistic, p_value

    def runtime_summaries(
        methods: dict[tuple[int, str], list[Run]],
        vns_runs: dict[str, list[VnsRun]],
    ) -> list[RuntimeSummary]:
        """Summarize VNS total runtime and LNS time to the best solution."""
        configurations = (
            (
                "VNS(l)",
                [run.runtime for run in vns_runs["test6WithSeed.py"]],
            ),
            (
                "VNS(m)",
                [run.runtime for run in vns_runs["test7WithSeed.py"]],
            ),
            (
                "VNS(h)",
                [run.runtime for run in vns_runs["test8WithSeed.py"]],
            ),
            (
                "LNS 60 s",
                [run.time_best for run in methods[60, "Full LNS"]],
            ),
            (
                "LNS 300 s",
                [run.time_best for run in methods[300, "Full LNS"]],
            ),
            (
                "LNS 600 s",
                [run.time_best for run in methods[600, "Full LNS"]],
            ),
        )
        summaries = []
        for label, runtimes in configurations:
            if not runtimes:
                raise ValueError(f"{label}: no runtime values")
            if len(runtimes) == 1:
                q1 = q3 = runtimes[0]
            else:
                q1, _, q3 = statistics.quantiles(
                    runtimes, n=4, method="inclusive"
                )
            summaries.append(
                RuntimeSummary(
                    method=label,
                    average=statistics.mean(runtimes),
                    q1=q1,
                    median=statistics.median(runtimes),
                    q3=q3,
                    standard_deviation=(
                        statistics.stdev(runtimes) if len(runtimes) > 1 else 0.0
                    ),
                    runs=len(runtimes),
                )
            )
        return summaries

    VNS_OUTPUTS = (
        ROOT / "AlyEtAl" / "vns.low.results.runs.txt",
        ROOT / "AlyEtAl" / "vns.medium.results.runs.txt",
        ROOT / "AlyEtAl" / "vns.high.results.runs.txt",
    )
    LOWER_BOUND_OUTPUT = ROOT / "resultsLNS" / "lower.bounds.results.txt"

    return SimpleNamespace(
        BoundGapSummary=BoundGapSummary,
        COMPARISONS=COMPARISONS,
        ComparisonResult=ComparisonResult,
        LOWER_BOUND_OUTPUT=LOWER_BOUND_OUTPUT,
        LOWER_BOUND_PATTERN=LOWER_BOUND_PATTERN,
        RuntimeSummary=RuntimeSummary,
        VNS_OUTPUTS=VNS_OUTPUTS,
        VnsRun=VnsRun,
        bound_gap_summaries=bound_gap_summaries,
        compare_samples=compare_samples,
        expected_instances=expected_instances,
        format_p_value=format_p_value,
        format_z_value=format_z_value,
        group_by_instance=group_by_instance,
        pairwise_van_elteren=pairwise_van_elteren,
        read_vns_runs=read_vns_runs,
        runtime_summaries=runtime_summaries
    )


lns_vns = _build_lns_vns_helpers()


FEASIBILITY_TOL = 1e-6
FEASIBILITY_OUTPUT = PAPER_DIR / "table7.tex"
OBJGAP_FULL_WINS_OUTPUT = PAPER_DIR / "table8.tex"
RUN_LEVEL_PAIRED_COUNTS_OUTPUT = PAPER_DIR / "table3.tex"
EFFECT_SIZES_OUTPUT = PAPER_DIR / "table4.tex"
LNS_VNS_STATISTICAL_COMPARISON_OUTPUT = (
    PAPER_DIR / "table5.tex"
)
LNS_VNS_VAN_ELTEREN_MATRIX_OUTPUT = (
    PAPER_DIR / "table6.tex"
)
COMPUTATIONAL_NUMBERS_OUTPUT = (
    PAPER_DIR / "computational_experiment_numbers.txt"
)
VNSL_INPUT = ROOT / "AlyEtAl" / "vns.low.results.txt"
VNSM_INPUT = ROOT / "AlyEtAl" / "vns.medium.results.txt"
VNSH_INPUT = ROOT / "AlyEtAl" / "vns.high.results.txt"
LNS_INPUT = ROOT / "resultsLNS" / "lns.results.txt"
LNS_RUNS_INPUT = ROOT / "resultsLNS" / "lns.results.runs.txt"
LNS_ABLATION_RUNS_INPUT = (
    ROOT / "resultsLNS" / "lns.ablation.results.runs.txt"
)
LNS_FIXED_RUNS_INPUT = ROOT / "resultsLNS" / "lns.fixed.results.runs.txt"
VNSL_RUNS_INPUT = ROOT / "AlyEtAl" / "vns.low.results.runs.txt"
VNSM_RUNS_INPUT = ROOT / "AlyEtAl" / "vns.medium.results.runs.txt"
VNSH_RUNS_INPUT = ROOT / "AlyEtAl" / "vns.high.results.runs.txt"
LOWER_BOUNDS_INPUT = ROOT / "resultsLNS" / "lower.bounds.results.txt"
INSTANCES_DIR = ROOT / "instances"
GROUP3_EXPECTED_INSTANCE_COUNT = 2430
ALGORITHM_COLUMNS = [
    ("LNS60", "LNS (60~s)"),
    ("LNS300", "LNS (300~s)"),
    ("LNS600", "LNS (600~s)"),
    ("VNSl", "VNS(l)"),
    ("VNSm", "VNS(m)"),
    ("VNSh", "VNS(h)"),
]
GROUP3_TABLE_ALGORITHM_COLUMNS = [
    ("VNSl", "VNS(l)"),
    ("VNSm", "VNS(m)"),
    ("VNSh", "VNS(h)"),
    ("LNS60", "LNS (60~s)"),
    ("LNS300", "LNS (300~s)"),
    ("LNS600", "LNS (600~s)"),
]
CMP_ALGORITHMS = [alg for alg, _ in ALGORITHM_COLUMNS]
LEVEL_MAP = {"l": "25%", "m": "50%", "h": "90%"}
COLUMNS = [
    "name",
    "size",
    "type",
    "dValue",
    "wValue",
    "cValue",
    "alg",
    "obj",
    "penalty",
    "totalTime",
    "timeBest",
]
GROUP12_ROW_ORDER = [
    ("group1", "planar", 500, "Planar"),
    ("group1", "planar", 600, "Planar"),
    ("group1", "planar", 700, "Planar"),
    ("group2", "Center", 486, "Center"),
    ("group2", "Center", 600, "Center"),
    ("group2", "Center", 726, "Center"),
    ("group2", "Corners", 486, "Corners"),
    ("group2", "Corners", 600, "Corners"),
    ("group2", "Corners", 726, "Corners"),
    ("group2", "Diagonal", 486, "Diagonal"),
    ("group2", "Diagonal", 600, "Diagonal"),
    ("group2", "Diagonal", 726, "Diagonal"),
    ("group3", "Center", 486, "Center'"),
    ("group3", "Center", 600, "Center'"),
    ("group3", "Center", 726, "Center'"),
    ("group3", "Corners", 486, "Corners'"),
    ("group3", "Corners", 600, "Corners'"),
    ("group3", "Corners", 726, "Corners'"),
    ("group3", "Diagonal", 486, "Diagonal'"),
    ("group3", "Diagonal", 600, "Diagonal'"),
    ("group3", "Diagonal", 726, "Diagonal'"),
]

ALY_PATTERN = re.compile(
    r"^(?:Version:\s+(?P<version>\S+)\s+)?"
    r"Instance:\s+(?P<instance>\S+)\s+"
    r"(?:Seed:\s+\S+\s+)?"
    r"Best objective:\s+(?P<obj>\S+)\s+"
    r"Infeasibility:\s+(?P<penalty>\S+)\s+"
    r"Total time \(s\):\s+(?P<total_time>\S+)\s*$"
)
LNS_PATTERN = re.compile(
    r"^RESULTS:\s+(?P<instance>\S+)\s+"
    r"min\s+\S+\s+step\s+\S+\s+max\s+\S+\s+t\s+(?P<time_limit>\S+)\s+"
    r"obj:\s+(?P<obj>\S+)\s+"
    r"penalty:\s+(?P<penalty>\S+)\s+"
    r"t:\s+(?P<total_time>\S+)\s+"
    r"tBest:\s+(?P<time_best>\S+)\s*$"
)
GROUP3_PATTERN = re.compile(
    r"^d-(?P<d>[hlm])_w-(?P<w>[hlm])_c-(?P<c>[hlm])-(?P<kind>[A-Za-z]+)(?P<size>\d+)_G\d+$"
)
BASE_PATTERN = re.compile(r"^(?P<kind>[A-Za-z]+)(?P<size>\d+)_G\d+$")
PLANAR_PATTERN = re.compile(r"^(?P<kind>planar)(?P<size>\d+)_G\d+$")


def parse_name_fields(raw_name: str) -> dict[str, object]:
    match = GROUP3_PATTERN.match(raw_name)
    if match:
        return {
            "name": raw_name,
            "size": int(match.group("size")),
            "type": match.group("kind"),
            "dValue": match.group("d"),
            "wValue": match.group("w"),
            "cValue": match.group("c"),
        }

    match = PLANAR_PATTERN.match(raw_name)
    if match:
        return {
            "name": raw_name,
            "size": int(match.group("size")),
            "type": match.group("kind"),
            "dValue": None,
            "wValue": None,
            "cValue": None,
        }

    match = BASE_PATTERN.match(raw_name)
    if match:
        return {
            "name": raw_name,
            "size": int(match.group("size")),
            "type": match.group("kind"),
            "dValue": None,
            "wValue": None,
            "cValue": None,
        }

    raise ValueError(f"Could not parse instance name fields from '{raw_name}'")


def normalize_lns_time_limit(time_limit: float) -> str:
    rounded_limit = int(round(time_limit))
    if rounded_limit == 60:
        return "LNS60"
    if rounded_limit == 300:
        return "LNS300"
    if rounded_limit == 600:
        return "LNS600"
    raise ValueError(f"Unsupported LNS time limit: {time_limit}")


def parse_vns_line(line: str, algorithm: str) -> dict[str, object] | None:
    match = ALY_PATTERN.match(line.strip())
    if not match:
        return None

    raw_name = Path(match.group("instance")).stem
    row = parse_name_fields(raw_name)
    row.update(
        {
            "alg": algorithm,
            "obj": float(match.group("obj")),
            "penalty": float(match.group("penalty")),
            "totalTime": float(match.group("total_time")),
            "timeBest": None,
        }
    )
    return row


def parse_lns_line(line: str) -> dict[str, object] | None:
    match = LNS_PATTERN.match(line.strip())
    if not match:
        return None

    raw_name = Path(match.group("instance")).stem
    row = parse_name_fields(raw_name)
    time_limit = float(match.group("time_limit"))
    row.update(
        {
            "alg": normalize_lns_time_limit(time_limit),
            "obj": float(match.group("obj")),
            "penalty": float(match.group("penalty")),
            "totalTime": float(match.group("total_time")),
            "timeBest": float(match.group("time_best")),
        }
    )
    return row


def parse_vns_file(path: Path, algorithm: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = parse_vns_line(stripped, algorithm)
            if row is None:
                raise ValueError(f"Unrecognized VNS line at {path}:{lineno}")
            rows.append(row)
    return rows


def parse_lns_file(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = parse_lns_line(stripped)
            if row is None:
                raise ValueError(f"Unrecognized LNS line at {path}:{lineno}")
            rows.append(row)
    return rows


def build_dataframe(vnsl_file: Path, vnsm_file: Path, vnsh_file: Path, lns_file: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rows.extend(parse_vns_file(vnsl_file, "VNSl"))
    rows.extend(parse_vns_file(vnsm_file, "VNSm"))
    rows.extend(parse_vns_file(vnsh_file, "VNSh"))
    rows.extend(parse_lns_file(lns_file))

    dataframe = pd.DataFrame(rows, columns=COLUMNS)
    dataframe.sort_values(by=["alg", "type", "size", "name"], inplace=True, na_position="last")
    dataframe.reset_index(drop=True, inplace=True)
    return dataframe


def infer_group(dataframe: pd.DataFrame) -> pd.Series:
    return pd.Series(
        [
            "group3"
            if pd.notna(d_value)
            else "group1"
            if instance_type == "planar"
            else "group2"
            for d_value, instance_type in zip(dataframe["dValue"], dataframe["type"])
        ],
        index=dataframe.index,
    )


def validated_group3_results(dataframe: pd.DataFrame) -> pd.DataFrame:
    filtered = dataframe[
        dataframe["alg"].isin(CMP_ALGORITHMS)
        & dataframe["dValue"].notna()
        & dataframe["wValue"].notna()
        & dataframe["cValue"].notna()
    ].copy()

    duplicate_count = int(filtered.duplicated(["name", "alg"]).sum())
    if duplicate_count:
        raise ValueError(
            f"Found {duplicate_count} duplicate group-3 instance/method results"
        )

    method_instance_counts = filtered.groupby("alg")["name"].nunique().to_dict()
    expected_counts = {
        algorithm: GROUP3_EXPECTED_INSTANCE_COUNT
        for algorithm in CMP_ALGORITHMS
    }
    if method_instance_counts != expected_counts:
        raise ValueError(
            "Expected exactly "
            f"{GROUP3_EXPECTED_INSTANCE_COUNT} group-3 instances per method; "
            f"found {method_instance_counts}"
        )
    return filtered


def build_group3_feasibility_by_type_size_table(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    filtered = validated_group3_results(dataframe)
    feasible = filtered[
        filtered["penalty"].fillna(float("inf")) < FEASIBILITY_TOL
    ]
    feasible_counts = feasible.groupby(["type", "size", "alg"]).size().to_dict()
    instance_counts = filtered.groupby(["type", "size"])["name"].nunique().to_dict()

    rows: list[dict[str, object]] = []
    for instance_type in ("Center", "Corners", "Diagonal"):
        for size in (486, 600, 726):
            row: dict[str, object] = {
                "type": instance_type,
                "size": size,
                "instances": int(instance_counts.get((instance_type, size), 0)),
            }
            for algorithm, column_name in ALGORITHM_COLUMNS:
                row[column_name] = int(
                    feasible_counts.get((instance_type, size, algorithm), 0)
                )
            rows.append(row)
    return pd.DataFrame(rows)


def build_group3_feasibility_by_variability_table(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    filtered = validated_group3_results(dataframe)
    feasible = filtered[
        filtered["penalty"].fillna(float("inf")) < FEASIBILITY_TOL
    ]
    grouping = ["dValue", "wValue", "cValue"]
    feasible_counts = feasible.groupby([*grouping, "alg"]).size().to_dict()
    instance_counts = filtered.groupby(grouping)["name"].nunique().to_dict()

    rows: list[dict[str, object]] = []
    for demand in ("l", "m", "h"):
        for workload in ("l", "m", "h"):
            for customers in ("l", "m", "h"):
                key = (demand, workload, customers)
                row: dict[str, object] = {
                    "Demand": LEVEL_MAP[demand],
                    "Workload": LEVEL_MAP[workload],
                    "Customers": LEVEL_MAP[customers],
                    "instances": int(instance_counts.get(key, 0)),
                }
                for algorithm, column_name in ALGORITHM_COLUMNS:
                    row[column_name] = int(
                        feasible_counts.get((*key, algorithm), 0)
                    )
                rows.append(row)
    return pd.DataFrame(rows)


def build_all_instances_subset(dataframe: pd.DataFrame) -> pd.DataFrame:
    enriched = dataframe.copy()
    enriched["group"] = infer_group(enriched)
    filtered = enriched[enriched["alg"].isin(CMP_ALGORITHMS)].copy()
    available_counts = filtered.groupby("name")["alg"].nunique()
    eligible_names = available_counts[available_counts == len(CMP_ALGORITHMS)].index
    return filtered[filtered["name"].isin(eligible_names)].copy()


def build_group12_obj_gap_full_with_wins_table(dataframe: pd.DataFrame) -> pd.DataFrame:
    filtered = build_all_instances_subset(dataframe)
    best_by_instance = filtered.groupby("name")["obj"].min().to_dict()
    filtered["gap"] = filtered.apply(
        lambda row: ((row["obj"] - best_by_instance[row["name"]]) / best_by_instance[row["name"]]) * 100
        if abs(best_by_instance[row["name"]]) >= FEASIBILITY_TOL
        else 0.0,
        axis=1,
    )
    filtered["is_best"] = filtered.apply(
        lambda row: abs(row["obj"] - best_by_instance[row["name"]]) < FEASIBILITY_TOL,
        axis=1,
    )

    gap_means = filtered.groupby(["group", "type", "size", "alg"], dropna=False)["gap"].mean().to_dict()
    win_counts = (
        filtered[filtered["is_best"]]
        .groupby(["group", "type", "size", "alg"], dropna=False)
        .size()
        .to_dict()
    )
    instance_counts = filtered.groupby(["group", "type", "size"], dropna=False)["name"].nunique().to_dict()

    rows: list[dict[str, object]] = []
    for group, instance_type, size, display_type in GROUP12_ROW_ORDER:
        row = {
            "type": display_type,
            "size": size,
            "instances": int(instance_counts.get((group, instance_type, size), 0)),
        }
        for alg_key, column_name in ALGORITHM_COLUMNS:
            avg_gap = float(gap_means.get((group, instance_type, size, alg_key), 0.0))
            wins = int(win_counts.get((group, instance_type, size, alg_key), 0))
            row[column_name] = f"{avg_gap:.2f} ({wins:,})"
        rows.append(row)
    return pd.DataFrame(rows)


def format_feasibility_percentage(feasible: int, instances: int) -> str:
    if instances and feasible == instances:
        return r"100\%"
    tenths = (1000 * feasible // instances) if instances else 0
    return f"{tenths / 10:.1f}\\%"


def feasibility_percentage_entries(row: pd.Series) -> list[str]:
    instances = int(row["instances"])
    return [
        format_feasibility_percentage(int(row[column_name]), instances)
        for _, column_name in GROUP3_TABLE_ALGORITHM_COLUMNS
    ]


def total_feasibility_percentage_entries(table: pd.DataFrame) -> list[str]:
    total_instances = int(table["instances"].sum())
    return [
        format_feasibility_percentage(int(table[column_name].sum()), total_instances)
        for _, column_name in GROUP3_TABLE_ALGORITHM_COLUMNS
    ]


def group3_type_size_rows(
    table: pd.DataFrame,
    entry_formatter: Callable[[pd.Series], list[str]],
) -> list[str]:
    rows = []
    for row_number, (_, row) in enumerate(table.iterrows()):
        graph_cell = ""
        if row_number % 3 == 0:
            graph_cell = (
                rf"\multicolumn{{1}}{{l}}{{\multirow{{3}}{{*}}{{{row['type']}}}}}"
            )
        bu_cell = rf"\multicolumn{{1}}{{l}}{{{int(row['size'])}}}"
        rows.append(
            f"{graph_cell} & {bu_cell} & & "
            + " & ".join(entry_formatter(row))
            + r" \\"
        )
        if row_number % 3 == 2 and row_number != len(table) - 1:
            rows.append(r"\hline")
    return rows


def group3_variability_rows(
    table: pd.DataFrame,
    entry_formatter: Callable[[pd.Series], list[str]],
) -> list[str]:
    rows = []
    for row_number, (_, row) in enumerate(table.iterrows()):
        demand_label = str(row["Demand"]).replace("%", r"\%")
        workload_label = str(row["Workload"]).replace("%", r"\%")
        customers_label = str(row["Customers"]).replace("%", r"\%")
        demand_cell = (
            rf"\multirow{{9}}{{*}}{{{demand_label}}}"
            if row_number % 9 == 0
            else ""
        )
        workload_cell = (
            rf"\multirow{{3}}{{*}}{{{workload_label}}}"
            if row_number % 3 == 0
            else ""
        )
        rows.append(
            f"{demand_cell} & {workload_cell} & {customers_label} & "
            + " & ".join(entry_formatter(row))
            + r" \\"
        )
        if row_number % 9 == 8 and row_number != len(table) - 1:
            rows.append(r"\hline")
        elif row_number % 3 == 2 and row_number != len(table) - 1:
            rows.append(r"\cline{2-9}")
    return rows


def render_group3_feasibility_table(dataframe: pd.DataFrame) -> str:
    type_size_table = build_group3_feasibility_by_type_size_table(dataframe)
    variability_table = build_group3_feasibility_by_variability_table(dataframe)
    method_headers = " & ".join(
        column_name for _, column_name in GROUP3_TABLE_ALGORITHM_COLUMNS
    )
    rows = [
        r"\multicolumn{9}{l}{\textit{Panel A: Spatial distribution and number of BUs}} \\",
        rf"\multicolumn{{1}}{{l}}{{Graph}} & \multicolumn{{1}}{{l}}{{\# BUs}} & & {method_headers} \\",
        r"\midrule",
    ]
    rows.extend(
        group3_type_size_rows(
            type_size_table,
            feasibility_percentage_entries,
        )
    )
    rows.extend(
        [
            r"\midrule",
            r"\multicolumn{9}{l}{\textit{Panel B: Attribute-requirement variability}} \\",
            rf"Demand & Workload & Customers & {method_headers} \\",
            r"\midrule",
        ]
    )

    rows.extend(
        group3_variability_rows(
            variability_table,
            feasibility_percentage_entries,
        )
    )
    rows.extend(
        [
            r"\midrule",
            "Total & & & "
            + " & ".join(
                total_feasibility_percentage_entries(variability_table)
            )
            + r" \\",
        ]
    )
    body = "\n".join(rows)
    return f"""\\begingroup
\\begin{{table}}[p]
\\centering
\\caption{{Percentage of the 2,430 third-set instances for which each method finds a balance-feasible solution. Panel A groups instances by spatial distribution and number of BUs, with 270 instances per group. Panel B groups them by the shares of low-valued demand, workload, and customer requirements, with 90 instances per combination. The final row in Panel B reports the percentage over all 2,430 instances. Non-exact percentages are rounded down to one decimal place; exact 100\\% values are shown without decimals.}}
\\label{{tab:feasibility-group3}}
\\scriptsize
\\setlength{{\\tabcolsep}}{{2pt}}
\\renewcommand{{\\arraystretch}}{{1.12}}
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{cccrrrrrr}}
\\toprule
{body}
\\bottomrule
\\end{{tabular}}
}}
\\renewcommand{{\\arraystretch}}{{1}}
\\end{{table}}
\\endgroup
"""


def group12_obj_gap_full_with_wins_table_to_latex(dataframe: pd.DataFrame) -> str:
    group3 = validated_group3_results(dataframe)
    table = build_group12_obj_gap_full_with_wins_table(group3)
    table = table[table["instances"] > 0].copy()
    metric_columns = [column_name for _, column_name in ALGORITHM_COLUMNS]
    column_spec = "l" + "r" * (1 + len(metric_columns))
    filtered = build_all_instances_subset(group3)
    best_by_instance = filtered.groupby("name")["obj"].min().to_dict()
    filtered["gap"] = filtered.apply(
        lambda row: ((row["obj"] - best_by_instance[row["name"]]) / best_by_instance[row["name"]]) * 100
        if abs(best_by_instance[row["name"]]) >= FEASIBILITY_TOL
        else 0.0,
        axis=1,
    )
    filtered["is_best"] = filtered.apply(
        lambda row: abs(row["obj"] - best_by_instance[row["name"]]) < FEASIBILITY_TOL,
        axis=1,
    )

    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Average relative compactness gap for each LNS and VNS configuration across the 2,430 third-set instances, calculated with respect to the best compactness value obtained by any of the six configurations, irrespective of balance feasibility. Values in parentheses report the number of instances on which each configuration attains the best compactness value; ties are counted for every attaining configuration. Each layout and size combination contains 270 instances, and the final row reports the overall average gaps and total attainment counts.}",
        r"\label{tab:objgap-full-wins}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        rf"Layout & \# BUs & {' & '.join(metric_columns)} \\",
        r"\midrule",
    ]

    for _, row in table.iterrows():
        layout = str(row["type"]).removesuffix("'")
        lines.append(
            f"{layout} & {int(row['size'])} & "
            + " & ".join(str(row[column_name]) for column_name in metric_columns)
            + r" \\"
        )

    total_entries = []
    for alg_key, column_name in ALGORITHM_COLUMNS:
        subset = filtered[filtered["alg"] == alg_key]
        total_entries.append(
            f"{subset['gap'].mean():.2f} ({int(subset['is_best'].sum()):,})"
        )

    lines.extend(
        [
            r"\midrule",
            r"\multicolumn{2}{l}{Total} & " + " & ".join(total_entries) + r" \\",
            r"\bottomrule",
            r"\end{tabular}}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def render_run_level_paired_counts(
    methods: dict[tuple[int, str], list[ablation.Run]],
) -> str:
    results = ablation.van_elteren_results(methods)
    result_by_key = {
        (result.time_limit, result.label): result for result in results
    }
    van_elteren_holm = ablation.holm_adjust(
        {key: result.p_value for key, result in result_by_key.items()}
    )
    cells: dict[tuple[int, str], tuple[int, int, str, str]] = {}
    for result in results:
        reference_by_instance: dict[str, list[ablation.Run]] = defaultdict(list)
        variant_by_instance: dict[str, list[ablation.Run]] = defaultdict(list)
        for run in methods[result.time_limit, "Full LNS"]:
            reference_by_instance[run.instance].append(run)
        for run in methods[result.time_limit, result.label]:
            variant_by_instance[run.instance].append(run)
        p_values = {}
        superiority = {}
        for instance in sorted(reference_by_instance):
            reference = reference_by_instance[instance]
            variant = variant_by_instance[instance]
            favorable_u, variance, effect = ablation.mann_whitney_components(
                reference, variant
            )
            p_values[instance] = ablation.mann_whitney_p_value(
                favorable_u, variance, len(reference), len(variant)
            )
            superiority[instance] = effect
        adjusted = ablation.holm_adjust(p_values)
        full_significant = sum(
            adjusted[instance] < 0.05 and superiority[instance] > 0.5
            for instance in adjusted
        )
        variant_significant = sum(
            adjusted[instance] < 0.05 and superiority[instance] < 0.5
            for instance in adjusted
        )
        stars = ablation.significance_stars(result.p_value)
        z_value = (
            f"${result.z_statistic:.3f}^{{{stars}}}$"
            if stars
            else f"${result.z_statistic:.3f}$"
        )
        key = result.time_limit, result.label
        cells[key] = (
            full_significant,
            variant_significant,
            z_value,
            ablation.format_p_value(van_elteren_holm[key]),
        )

    rows = []
    table_labels = (
        "Random destroy",
        "Alternative destroy",
        "District-based destroy",
        "Fixed destroy size",
        "No fixing",
        "No warm start",
        "Only random start",
        "Heuristic repair",
    )
    for label in table_labels:
        values = []
        for time_limit in ablation.TIME_LIMITS:
            full, variant, z_value, holm = cells[time_limit, label]
            values.extend((str(full), str(variant), z_value, holm))
        display_label = (
            "Distance-based destroy" if label == "Alternative destroy" else label
        )
        rows.append(f"{display_label} & " + " & ".join(values) + r" \\")
        if label == "District-based destroy":
            rows.append(r"\midrule")

    body = "\n".join(rows)
    return f"""\\begingroup
\\begin{{table}}[htbp]
\\centering
\\caption{{Comparison of the Full LNS variant with each ablated variant. The full and variant columns count instances in which a Mann--Whitney test significantly favors that method at $\\alpha=0.05$, after Holm adjustment across the {ablation.EXPECTED_INSTANCE_COUNT} instance tests for that comparison. Each $z$ column reports the van Elteren statistic, with a positive value favoring the Full LNS and the stars showing the significance thresholds ($^{{*}}p<0.05$, $^{{**}}p<0.01$, $^{{***}}p<0.001$). The \\textit{{Holm}} columns report the van Elteren $p$-values after Holm adjustment across all 24 time-limit and variant comparisons.}}
\\label{{tab:exp08-van-elteren-summary}}
\\scriptsize
\\setlength{{\\tabcolsep}}{{2pt}}
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{lcccccccccccc}}
\\toprule
& \\multicolumn{{4}}{{c}}{{60 s}} & \\multicolumn{{4}}{{c}}{{300 s}} & \\multicolumn{{4}}{{c}}{{600 s}} \\\\
\\cmidrule(lr){{2-5}} \\cmidrule(lr){{6-9}} \\cmidrule(lr){{10-13}}
Variant & full & variant & $z$ & Holm & full & variant & $z$ & Holm & full & variant & $z$ & Holm \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
}}
\\end{{table}}
\\endgroup
"""


def render_effect_sizes(
    methods: dict[tuple[int, str], list[ablation.Run]],
) -> str:
    superiority = {
        (result.time_limit, result.label): result.mean_superiority
        for result in ablation.van_elteren_results(methods)
    }
    table_labels = (
        ("Random destroy", "Random destroy"),
        ("Alternative destroy", "Distance-based destroy"),
        ("District-based destroy", "District-based destroy"),
        ("Fixed destroy size", "Fixed destroy size"),
        ("No fixing", "No fixing"),
        ("No warm start", "No warm start"),
        ("Only random start", "Only random start"),
        ("Heuristic repair", "Heuristic repair"),
    )
    rows = []
    for time_limit in ablation.TIME_LIMITS:
        if rows:
            rows.append(r"\midrule")
        for label, display_label in table_labels:
            full_runs = methods[time_limit, "Full LNS"]
            variant_runs = methods[time_limit, label]
            full_feasibility = (
                100.0 * sum(run.penalty == 0 for run in full_runs) / len(full_runs)
            )
            variant_feasibility = (
                100.0
                * sum(run.penalty == 0 for run in variant_runs)
                / len(variant_runs)
            )
            gaps = ablation.instance_objective_gaps(methods, time_limit, label)
            q1, _, q3 = statistics.quantiles(gaps, n=4, method="inclusive")
            rows.append(
                f"{time_limit} & {display_label} & "
                f"{100.0 * superiority[time_limit, label]:.1f} & "
                f"{full_feasibility:.2f} & {variant_feasibility:.2f} & "
                f"{statistics.median(gaps):.1f} "
                f"[{q1:.1f}, {q3:.1f}] & "
                f"{sum(gap > 0 for gap in gaps)}/"
                f"{sum(gap == 0 for gap in gaps)}/"
                f"{sum(gap < 0 for gap in gaps)} \\\\"
            )

    body = "\n".join(rows)
    return f"""\\begingroup
\\begin{{table}}[htbp]
\\centering
\\caption{{Effect-size comparison of Full LNS with each ablation. Values of $\\pi_F$ above 50\\% favor Full LNS. Full and Abl. give the percentages of zero-penalty runs. RCD denotes the instance-level relative compactness difference among balance-feasible runs; Med. RCD $[Q_1,Q_3]$ reports its median and quartiles, with positive values favoring Full LNS. Full W/T/L gives the eligible-instance counts with lower, equal, or higher median compactness for Full LNS, respectively.}}
\\label{{tab:exp08-effect-size-summary}}
\\scriptsize
\\setlength{{\\tabcolsep}}{{3pt}}
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{rlrrrrr}}
\\toprule
& & \\multicolumn{{1}}{{c}}{{Pairwise comparison}} & \\multicolumn{{2}}{{c}}{{Balance-feasible runs}} & \\multicolumn{{2}}{{c}}{{Compactness of feasible runs}} \\\\
\\cmidrule(lr){{3-3}} \\cmidrule(lr){{4-5}} \\cmidrule(lr){{6-7}}
Time (s) & Ablation & $\\pi_F$ (\\%) & Full (\\%) & Abl. (\\%) & Med. RCD $[Q_1,Q_3]$ (\\%) & Full W/T/L \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
}}
\\end{{table}}
\\endgroup
"""


def render_lns_vns_statistical_comparison(
    results: list[lns_vns.ComparisonResult],
) -> str:
    rows = [
        " & ".join(
            (
                f"LNS ({result.time_limit}~s) vs {result.vns_label}",
                str(result.lns_favored),
                str(result.vns_favored),
                lns_vns.format_z_value(result.z_statistic, result.p_value),
                f"{result.lns_feasible_runs:.1f}\\%",
                f"{result.vns_feasible_runs:.1f}\\%",
                f"{result.median_compactness_difference:.1f} "
                f"[{result.q1_compactness_difference:.1f}, "
                f"{result.q3_compactness_difference:.1f}]",
            )
        )
        for result in results
    ]
    body = " \\\\\n".join(rows) + r" \\"
    return f"""\\begingroup
\\begin{{table}}[htbp]
\\centering
\\caption{{Comparison of the Full LNS configurations with their corresponding VNS variants. The LNS and VNS columns count instances in which a Mann--Whitney test significantly favors that method at $\\alpha=0.05$, after Holm adjustment across the {ablation.EXPECTED_INSTANCE_COUNT} instance tests for that comparison. The $z$ column reports the van Elteren statistic, with a positive value favoring LNS and the stars showing the significance threshold ($^{{***}}p<0.001$). The run-feasibility columns report the percentage of zero-penalty runs. RCD denotes the instance-level relative compactness difference between the VNS and LNS median compactness values among balance-feasible runs; Med. RCD $[Q_1,Q_3]$ reports its median and quartiles, with positive values favoring LNS.}}
\\label{{tab:exp08-lns-vns-statistical-comparison}}
\\scriptsize
\\setlength{{\\tabcolsep}}{{3pt}}
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{lrrrrrr}}
\\toprule
& & & & \\multicolumn{{2}}{{c}}{{Run feasibility}} & \\multicolumn{{1}}{{c}}{{Compactness of feasible runs}} \\\\
\\cmidrule(lr){{5-6}} \\cmidrule(lr){{7-7}}
Comparison & LNS & VNS & $z$ & LNS & VNS & Med. RCD $[Q_1,Q_3]$ (\\%) \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
}}
\\end{{table}}
\\endgroup
"""


def render_lns_vns_van_elteren_matrix(
    methods: dict[tuple[int, str], list[ablation.Run]],
    vns_runs: dict[str, list[lns_vns.VnsRun]],
) -> str:
    configurations: tuple[
        tuple[str, list[ablation.Run] | list[lns_vns.VnsRun]], ...
    ] = (
        ("LNS (60~s)", methods[60, "Full LNS"]),
        ("LNS (300~s)", methods[300, "Full LNS"]),
        ("LNS (600~s)", methods[600, "Full LNS"]),
        ("VNS(l)", vns_runs["test6WithSeed.py"]),
        ("VNS(m)", vns_runs["test7WithSeed.py"]),
        ("VNS(h)", vns_runs["test8WithSeed.py"]),
    )
    z_statistics: dict[tuple[int, int], float] = {}
    p_values: dict[tuple[int, int], float] = {}
    for row_index, (_, row_runs) in enumerate(configurations):
        for column_index in range(row_index + 1, len(configurations)):
            _, column_runs = configurations[column_index]
            z_statistic, p_value = lns_vns.pairwise_van_elteren(
                row_runs, column_runs
            )
            z_statistics[row_index, column_index] = z_statistic
            p_values[row_index, column_index] = p_value

    adjusted = ablation.holm_adjust(p_values)
    if any(p_value >= 0.001 for p_value in adjusted.values()):
        raise ValueError(
            "not all pairwise van Elteren comparisons have "
            "Holm-adjusted p < 0.001"
        )

    rows = []
    for row_index, (row_label, _) in enumerate(configurations):
        cells = []
        for column_index in range(len(configurations)):
            if row_index == column_index:
                cells.append("--")
            elif row_index < column_index:
                cells.append(f"{z_statistics[row_index, column_index]:.3f}")
            else:
                cells.append(f"{-z_statistics[column_index, row_index]:.3f}")
        rows.append(row_label + " & " + " & ".join(cells) + r" \\")

    header = " & ".join(label for label, _ in configurations)
    body = "\n".join(rows)
    return f"""\\begingroup
\\begin{{table}}[htbp]
\\centering
\\caption{{Pairwise van Elteren $z$ statistics for the three Full-LNS configurations and the three VNS variants over {ablation.EXPECTED_INSTANCE_COUNT} instances and {ablation.EXPECTED_REPLICAS} independent runs per method and instance. Each test compares run outcomes lexicographically, prioritizing lower balance penalty and then lower compactness, and treats instances as strata. Positive values favor the method in the row, whereas negative values favor the method in the column. All 15 unique pairwise comparisons have Holm-adjusted $p<0.001$.}}
\\label{{tab:exp08-lns-vns-van-elteren-matrix}}
\\scriptsize
\\setlength{{\\tabcolsep}}{{3pt}}
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{lrrrrrr}}
\\toprule
Method & {header} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
}}
\\end{{table}}
\\endgroup
"""


def instance_rcds(
    reference_runs: list[ablation.Run],
    comparison_runs: list[lns_vns.VnsRun],
    feasible_only: bool,
) -> list[float]:
    """Return one VNS-versus-LNS relative compactness difference per instance."""
    reference_by_instance = lns_vns.group_by_instance(reference_runs)
    comparison_by_instance = lns_vns.group_by_instance(comparison_runs)
    if reference_by_instance.keys() != comparison_by_instance.keys():
        raise ValueError("RCD calculation uses different instance sets")

    values = []
    for instance in sorted(reference_by_instance):
        reference = [
            run.objective
            for run in reference_by_instance[instance]
            if not feasible_only or run.penalty == 0
        ]
        comparison = [
            run.objective
            for run in comparison_by_instance[instance]
            if not feasible_only or run.penalty == 0
        ]
        if not reference or not comparison:
            continue
        reference_median = statistics.median(reference)
        if reference_median <= 0.0:
            raise ValueError(f"{instance}: nonpositive LNS median")
        values.append(
            100.0
            * (statistics.median(comparison) - reference_median)
            / reference_median
        )
    return values


def load_group3_lower_bounds(source: Path) -> dict[str, float]:
    """Load discrete group-3 bounds from the archived threshold results."""
    lower_bounds: dict[str, float] = {}
    for line_number, line in enumerate(
        source.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line:
            continue
        match = lns_vns.LOWER_BOUND_PATTERN.fullmatch(line)
        if not match:
            raise ValueError(f"{source}:{line_number}: malformed lower bound")
        instance = Path(match.group("instance")).stem
        if not GROUP3_PATTERN.fullmatch(instance):
            continue
        if instance in lower_bounds:
            raise ValueError(f"duplicate group-3 lower bound: {instance}")
        # The threshold is the first infeasible integer distance.
        lower_bounds[instance] = float(match.group("value")) - 1.0
    if len(lower_bounds) != GROUP3_EXPECTED_INSTANCE_COUNT:
        raise ValueError(
            f"Expected {GROUP3_EXPECTED_INSTANCE_COUNT} group-3 lower bounds; "
            f"found {len(lower_bounds)}"
        )
    return lower_bounds


def group3_numerical_summary(
    dataframe: pd.DataFrame,
    lower_bound_source: Path,
) -> dict[str, object]:
    group3 = validated_group3_results(dataframe)
    objectives = group3.pivot(index="name", columns="alg", values="obj")
    penalties = group3.pivot(index="name", columns="alg", values="penalty")

    infeasible_counts = {
        method: int(
            (penalties[method].fillna(float("inf")) >= FEASIBILITY_TOL).sum()
        )
        for method in CMP_ALGORITHMS
    }
    feasibility_rates = {
        method: 100.0
        * (GROUP3_EXPECTED_INSTANCE_COUNT - count)
        / GROUP3_EXPECTED_INSTANCE_COUNT
        for method, count in infeasible_counts.items()
    }

    lns_methods = ("LNS60", "LNS300", "LNS600")
    vns_methods = ("VNSl", "VNSm", "VNSh")
    lns_best = objectives[list(lns_methods)].min(axis=1)
    vns_best = objectives[list(vns_methods)].min(axis=1)
    lns_wins = lns_best + FEASIBILITY_TOL < vns_best
    vns_wins = vns_best + FEASIBILITY_TOL < lns_best
    ties = (lns_best - vns_best).abs() < FEASIBILITY_TOL

    vns_best_infeasible_wins = 0
    for instance in objectives.index[vns_wins]:
        attaining = [
            method
            for method in vns_methods
            if abs(objectives.at[instance, method] - vns_best.at[instance])
            < FEASIBILITY_TOL
        ]
        if attaining and all(
            penalties.at[instance, method] >= FEASIBILITY_TOL
            for method in attaining
        ):
            vns_best_infeasible_wins += 1

    best_all = objectives[CMP_ALGORITHMS].min(axis=1)
    compactness_gap_means = {
        method: float(
            (100.0 * (objectives[method] - best_all) / best_all).mean()
        )
        for method in CMP_ALGORITHMS
    }

    lower_bounds = load_group3_lower_bounds(lower_bound_source)
    if set(lower_bounds) != set(objectives.index):
        raise ValueError("group-3 runs and lower-bound instance sets differ")
    bounds = pd.Series(lower_bounds).reindex(objectives.index)
    gap_values = {
        method: 100.0 * (objectives[method] - bounds) / objectives[method]
        for method in ("LNS600", "VNSh")
    }
    vnsh_feasible = penalties["VNSh"] < FEASIBILITY_TOL
    lower_bound_summary = {
        "all": {
            method: {
                "instances": GROUP3_EXPECTED_INSTANCE_COUNT,
                "mean": float(values.mean()),
                "median": float(values.median()),
            }
            for method, values in gap_values.items()
        },
        "vnsh_feasible": {
            method: {
                "instances": int(vnsh_feasible.sum()),
                "mean": float(values[vnsh_feasible].mean()),
                "median": float(values[vnsh_feasible].median()),
            }
            for method, values in gap_values.items()
        },
    }

    return {
        "infeasible_counts": infeasible_counts,
        "feasibility_rates": feasibility_rates,
        "family": {
            "lns_wins": int(lns_wins.sum()),
            "vns_wins": int(vns_wins.sum()),
            "ties": int(ties.sum()),
            "vns_best_infeasible_wins": vns_best_infeasible_wins,
        },
        "compactness_gap_means": compactness_gap_means,
        "lower_bounds": lower_bound_summary,
    }


def render_computational_experiment_numbers(
    dataframe: pd.DataFrame,
    methods: dict[tuple[int, str], list[ablation.Run]],
    vns_runs: dict[str, list[lns_vns.VnsRun]],
    comparison_results: list[lns_vns.ComparisonResult],
    runtime_summaries: list[lns_vns.RuntimeSummary],
    bound_summaries: list[lns_vns.BoundGapSummary],
) -> str:
    """Render all numerical results cited outside the manuscript tables."""
    group3 = group3_numerical_summary(dataframe, lns_vns.LOWER_BOUND_OUTPUT)
    runtime_by_method = {summary.method: summary for summary in runtime_summaries}
    bounds_by_method = {summary.method: summary for summary in bound_summaries}

    feasible_rcds = instance_rcds(
        methods[600, "Full LNS"], vns_runs["test8WithSeed.py"], True
    )
    all_rcds = instance_rcds(
        methods[600, "Full LNS"], vns_runs["test8WithSeed.py"], False
    )
    if len(feasible_rcds) != 120 or len(all_rcds) != 120:
        raise ValueError("Expected 120 RCD values for the 600-second comparison")

    effect_results = {
        (result.time_limit, result.label): result.mean_superiority
        for result in ablation.van_elteren_results(methods)
    }
    ablation_gaps = {
        label: ablation.instance_objective_gaps(methods, 600, label)
        for label in (
            "Random destroy",
            "Alternative destroy",
            "District-based destroy",
            "Fixed destroy size",
            "No fixing",
            "No warm start",
            "Only random start",
            "Heuristic repair",
        )
    }

    repeated_lns = bounds_by_method["LNS (600 s)"]
    repeated_vns = bounds_by_method["VNS(h)"]
    repeated_bound_differences = (
        round(repeated_vns.mean_all_run_instance_gap, 2)
        - round(repeated_lns.mean_all_run_instance_gap, 2),
        round(repeated_vns.median_all_run_instance_gap, 2)
        - round(repeated_lns.median_all_run_instance_gap, 2),
        round(repeated_vns.mean_feasible_run_instance_gap, 2)
        - round(repeated_lns.mean_feasible_run_instance_gap, 2),
        round(repeated_vns.median_feasible_run_instance_gap, 2)
        - round(repeated_lns.median_feasible_run_instance_gap, 2),
    )

    group3_bounds = group3["lower_bounds"]
    group3_bound_differences = (
        group3_bounds["all"]["VNSh"]["mean"]
        - group3_bounds["all"]["LNS600"]["mean"],
        group3_bounds["all"]["VNSh"]["median"]
        - group3_bounds["all"]["LNS600"]["median"],
        group3_bounds["vnsh_feasible"]["VNSh"]["mean"]
        - group3_bounds["vnsh_feasible"]["LNS600"]["mean"],
        group3_bounds["vnsh_feasible"]["VNSh"]["median"]
        - group3_bounds["vnsh_feasible"]["LNS600"]["median"],
    )
    all_bound_differences = (
        *repeated_bound_differences,
        *group3_bound_differences,
    )

    lines = [
        "[experimental design and tuned schedules]",
        "repeated instances = 120",
        "replicas per method and instance = 30",
        "runs per repeated configuration = 3,600",
        "single-run instances = 2,430",
        "LNS time limits (s) = 60, 300, 600",
    ]
    for time_limit in ablation.TIME_LIMITS:
        minimum, step, maximum = ablation.MAIN_SCHEDULES[time_limit]
        schedule = list(range(minimum, maximum + 1, step))
        lines.append(
            f"LNS {time_limit} s destroy percentages = "
            + ", ".join(str(value) for value in schedule)
        )
    lines.extend(
        [
            "fixed destroy percentages (60/300/600 s) = "
            + ", ".join(
                str(ablation.FIXED_SCHEDULES[time_limit][0])
                for time_limit in ablation.TIME_LIMITS
            ),
            "",
            "[600 s ablation results cited in prose]",
        ]
    )
    for label in (
        "Alternative destroy",
        "No warm start",
        "Only random start",
        "Fixed destroy size",
        "No fixing",
        "Random destroy",
        "District-based destroy",
        "Heuristic repair",
    ):
        display = (
            "Distance-based destroy" if label == "Alternative destroy" else label
        )
        gaps = ablation_gaps[label]
        lines.append(
            f"{display}: median RCD = {statistics.median(gaps):.1f}%; "
            f"pairwise index = {100.0 * effect_results[600, label]:.1f}%; "
            f"Full W/T/L = {sum(gap > 0 for gap in gaps)}/"
            f"{sum(gap == 0 for gap in gaps)}/"
            f"{sum(gap < 0 for gap in gaps)}"
        )

    lines.extend(["", "[LNS--VNS repeated-run results]"])
    for result in comparison_results:
        lines.append(
            f"LNS {result.time_limit} s vs {result.vns_label}: "
            f"significant instances LNS/VNS = "
            f"{result.lns_favored}/{result.vns_favored}; "
            f"run feasibility LNS/VNS = "
            f"{result.lns_feasible_runs:.1f}%/{result.vns_feasible_runs:.1f}%; "
            f"feasible-run median RCD = "
            f"{result.median_compactness_difference:.1f}%; "
            f"all-run median RCD [Q1,Q3] = "
            f"{result.median_all_instance_rcd:.1f}% "
            f"[{result.q1_all_instance_rcd:.1f}%, "
            f"{result.q3_all_instance_rcd:.1f}%]"
        )
    lines.extend(["", "[running times]"])
    for method in (
        "VNS(l)",
        "VNS(m)",
        "VNS(h)",
        "LNS 60 s",
        "LNS 300 s",
        "LNS 600 s",
    ):
        summary = runtime_by_method[method]
        lines.append(
            f"{method}: mean = {summary.average:,.1f} s; "
            f"median = {summary.median:,.1f} s; runs = {summary.runs:,}"
        )

    lines.extend(["", "[120-instance lower-bound results]"])
    for method in ("LNS (600 s)", "VNS(h)"):
        summary = bounds_by_method[method]
        lines.append(
            f"{method}: all-run mean/median gap = "
            f"{summary.mean_all_run_instance_gap:.2f}%/"
            f"{summary.median_all_run_instance_gap:.2f}%; "
            f"feasible-run mean/median gap = "
            f"{summary.mean_feasible_run_instance_gap:.2f}%/"
            f"{summary.median_feasible_run_instance_gap:.2f}%; "
            f"feasible runs per instance = "
            f"{summary.minimum_feasible_runs_per_instance}--"
            f"{summary.maximum_feasible_runs_per_instance}"
        )
    lines.append(
        "VNS(h) minus LNS gap reductions "
        "(all mean/all median/feasible mean/feasible median) = "
        + "/".join(f"{value:.2f}" for value in repeated_bound_differences)
        + " percentage points"
    )

    infeasible = group3["infeasible_counts"]
    feasibility = group3["feasibility_rates"]
    family = group3["family"]
    compactness = group3["compactness_gap_means"]
    lines.extend(
        [
            "",
            "[2,430-instance single-run results]",
            f"LNS infeasible counts (60/300/600 s) = "
            f"{infeasible['LNS60']}/{infeasible['LNS300']}/"
            f"{infeasible['LNS600']}",
            f"LNS (600 s)/VNS(h) feasibility = "
            f"{feasibility['LNS600']:.1f}%/{feasibility['VNSh']:.1f}%; "
            f"difference = "
            f"{feasibility['LNS600'] - feasibility['VNSh']:.1f} "
            "percentage points",
            f"best family LNS/VNS/tie = {family['lns_wins']:,}/"
            f"{family['vns_wins']:,}/{family['ties']:,}",
            f"VNS-family wins for which every attaining VNS result is "
            f"balance-infeasible = {family['vns_best_infeasible_wins']:,}",
            f"mean compactness gap LNS (600 s)/VNS(h) = "
            f"{compactness['LNS600']:.1f}%/{compactness['VNSh']:.1f}%",
        ]
    )
    for subset, label in (
        ("all", "all"),
        ("vnsh_feasible", "VNS(h)-feasible"),
    ):
        for method, display in (
            ("LNS600", "LNS (600 s)"),
            ("VNSh", "VNS(h)"),
        ):
            summary = group3_bounds[subset][method]
            lines.append(
                f"{label} {display}: instances = {summary['instances']:,}; "
                f"lower-bound mean/median gap = "
                f"{summary['mean']:.2f}%/{summary['median']:.2f}%"
            )

    lines.extend(
        [
            "",
            "[concluding cross-experiment results]",
            f"600 s feasible-run RCD mean/median = "
            f"{statistics.mean(feasible_rcds):.1f}%/"
            f"{statistics.median(feasible_rcds):.1f}%",
            f"600 s all-run RCD mean/median = "
            f"{statistics.mean(all_rcds):.1f}%/"
            f"{statistics.median(all_rcds):.1f}%",
            f"LNS lower-bound advantage range = "
            f"{min(all_bound_differences):.1f}--"
            f"{max(all_bound_differences):.1f} percentage points",
            f"LNS mean lower-bound gap range = "
            f"{min(repeated_lns.mean_all_run_instance_gap, group3_bounds['all']['LNS600']['mean']):.2f}--"
            f"{max(repeated_lns.mean_all_run_instance_gap, group3_bounds['all']['LNS600']['mean']):.2f}%",
            f"LNS median lower-bound gap range = "
            f"{min(repeated_lns.median_all_run_instance_gap, group3_bounds['all']['LNS600']['median']):.2f}--"
            f"{max(repeated_lns.median_all_run_instance_gap, group3_bounds['all']['LNS600']['median']):.2f}%",
        ]
    )
    return "\n".join(lines) + "\n"


def write_generated_output(output: Path, rendered: str) -> Path:
    output = validate_repo_path(output, "Output")
    output.write_text(rendered, encoding="utf-8")
    return output


def validate_repo_path(path: Path, description: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError as error:
        raise ValueError(
            f"{description} path must remain inside {ROOT}: {resolved}"
        ) from error
    return resolved


def validate_input_file(path: Path) -> Path:
    if not path.is_absolute():
        raise ValueError(f"Input path must be absolute: {path}")
    path = validate_repo_path(path, "Input")
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")
    return path


def configure_repeated_run_inputs() -> None:
    """Validate the archived inputs used by the embedded repeated-run analysis."""
    for input_path in (
        LNS_RUNS_INPUT,
        LNS_ABLATION_RUNS_INPUT,
        LNS_FIXED_RUNS_INPUT,
        VNSL_RUNS_INPUT,
        VNSM_RUNS_INPUT,
        VNSH_RUNS_INPUT,
        LOWER_BOUNDS_INPUT,
    ):
        validate_input_file(input_path)
    instances_dir = validate_repo_path(INSTANCES_DIR, "Instance directory")
    if not instances_dir.is_dir():
        raise FileNotFoundError(f"Instance directory not found: {instances_dir}")

    instances_by_group = {
        group: {path.stem for path in (instances_dir / group).glob("*.graphml")}
        for group in ("group1", "group2")
    }
    expected_counts = {"group1": 30, "group2": 90}
    actual_counts = {
        group: len(instances) for group, instances in instances_by_group.items()
    }
    if actual_counts != expected_counts:
        raise ValueError(
            f"Expected group1/group2 instance counts {expected_counts}; "
            f"found {actual_counts}"
        )
    overlap = instances_by_group["group1"] & instances_by_group["group2"]
    if overlap:
        raise ValueError(
            f"group1 and group2 overlap on {len(overlap)} instance names"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate exactly the tables and numerical text used in the manuscript."
    )
    parser.add_argument(
        "--tables",
        action="store_true",
        help="Generate all six LaTeX tables included in the manuscript.",
    )
    parser.add_argument(
        "--text",
        action="store_true",
        help="Generate the numerical results cited in the manuscript text.",
    )
    args = parser.parse_args()
    if not (args.tables or args.text):
        parser.error("select --tables, --text, or both")
    return args


def main() -> int:
    args = parse_args()
    vnsl_file = validate_input_file(VNSL_INPUT)
    vnsm_file = validate_input_file(VNSM_INPUT)
    vnsh_file = validate_input_file(VNSH_INPUT)
    lns_file = validate_input_file(LNS_INPUT)
    configure_repeated_run_inputs()

    dataframe = build_dataframe(vnsl_file, vnsm_file, vnsh_file, lns_file)
    ablation_methods = ablation.load_method_runs()
    vns_runs = lns_vns.read_vns_runs()
    comparison_results = [
        lns_vns.compare_samples(
            time_limit,
            vns_label,
            ablation_methods[time_limit, "Full LNS"],
            vns_runs[version],
        )
        for time_limit, vns_label, version in lns_vns.COMPARISONS
    ]

    if args.tables:
        table_outputs = {
            RUN_LEVEL_PAIRED_COUNTS_OUTPUT: render_run_level_paired_counts(
                ablation_methods
            ),
            EFFECT_SIZES_OUTPUT: render_effect_sizes(ablation_methods),
            LNS_VNS_STATISTICAL_COMPARISON_OUTPUT:
                render_lns_vns_statistical_comparison(comparison_results),
            LNS_VNS_VAN_ELTEREN_MATRIX_OUTPUT:
                render_lns_vns_van_elteren_matrix(
                    ablation_methods, vns_runs
                ),
            FEASIBILITY_OUTPUT: render_group3_feasibility_table(dataframe),
            OBJGAP_FULL_WINS_OUTPUT:
                group12_obj_gap_full_with_wins_table_to_latex(dataframe) + "\n",
        }
        for output_path, rendered in table_outputs.items():
            print(write_generated_output(output_path, rendered))

    if args.text:
        runtime_summaries = lns_vns.runtime_summaries(
            ablation_methods, vns_runs
        )
        bound_summaries = lns_vns.bound_gap_summaries(
            ablation_methods, vns_runs
        )
        print(
            write_generated_output(
                COMPUTATIONAL_NUMBERS_OUTPUT,
                render_computational_experiment_numbers(
                    dataframe,
                    ablation_methods,
                    vns_runs,
                    comparison_results,
                    runtime_summaries,
                    bound_summaries,
                ),
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
