#!/usr/bin/env python3
"""Generate the tables and reported numerical text used in the supplement.

Run this file directly with ``--tables``, ``--text``, or both. All inputs and
outputs are resolved relative to this file, so the command is independent of
the caller's working directory.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import heapq
from itertools import combinations
import math
from pathlib import Path
import re
import statistics
from types import SimpleNamespace
from typing import Iterable
import warnings
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR

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
    class FixedComparison:
        time_limit: int
        increasing_values: tuple[int, ...]
        fixed_value: int
        increasing_feasibility: float
        fixed_feasibility: float
        increasing_median: float
        fixed_median: float
        increasing_wins: int
        ties: int
        fixed_wins: int
        feasible_pairs: int
        median_gap: float

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

    def schedule_values(run: Run) -> tuple[int, ...]:
        return tuple(range(run.minimum, run.maximum + 1, run.step))

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
        FixedComparison=FixedComparison,
        MAIN_FILES=MAIN_FILES,
        Run=Run,
        TIME_LIMITS=TIME_LIMITS,
        expected_instances=expected_instances,
        instance_objective_gaps=instance_objective_gaps,
        load_method_runs=load_method_runs,
        schedule_values=schedule_values
    )


ablation = _build_ablation_helpers()


def _build_lns_vns_helpers() -> SimpleNamespace:
    EXPECTED_INSTANCE_COUNT = ablation.EXPECTED_INSTANCE_COUNT
    EXPECTED_REPLICAS = ablation.EXPECTED_REPLICAS
    Run = ablation.Run
    expected_instances = ablation.expected_instances
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
    class RuntimeSummary:
        method: str
        average: float
        q1: float
        median: float
        q3: float
        standard_deviation: float
        runs: int

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

    def read_graphml_adjacency(
        source: Path,
    ) -> list[list[tuple[int, float]]]:
        try:
            root = ET.parse(source).getroot()
            namespace = ""
            if root.tag.startswith("{"):
                namespace = root.tag[1 : root.tag.index("}")]

            def tag(local_name: str) -> str:
                return f"{{{namespace}}}{local_name}" if namespace else local_name

            if root.tag != tag("graphml"):
                raise ValueError("root element is not graphml")

            distance_keys = {
                key.attrib["id"]
                for key in root.findall(tag("key"))
                if key.attrib.get("for") in {"edge", "all"}
                and key.attrib.get("attr.name") == "distance"
            }
            if len(distance_keys) != 1:
                raise ValueError("expected one edge distance key")
            distance_key = next(iter(distance_keys))

            graph = root.find(tag("graph"))
            if graph is None:
                raise ValueError("missing graph element")
            if graph.attrib.get("edgedefault") != "undirected":
                raise ValueError("expected an undirected graph")

            nodes = graph.findall(tag("node"))
            node_indices = {
                node.attrib["id"]: index for index, node in enumerate(nodes)
            }
            if len(node_indices) != len(nodes):
                raise ValueError("duplicate node identifier")

            adjacency: list[list[tuple[int, float]]] = [
                [] for _ in nodes
            ]
            for edge in graph.findall(tag("edge")):
                left = node_indices[edge.attrib["source"]]
                right = node_indices[edge.attrib["target"]]
                distance_values = [
                    data.text
                    for data in edge.findall(tag("data"))
                    if data.attrib.get("key") == distance_key
                ]
                if len(distance_values) != 1 or distance_values[0] is None:
                    raise ValueError("edge is missing its distance")
                distance = float(distance_values[0])
                adjacency[left].append((right, distance))
                adjacency[right].append((left, distance))
        except (ET.ParseError, KeyError, OSError, ValueError) as error:
            raise ValueError(f"{source}: malformed GraphML instance") from error
        return adjacency

    def preceding_shortest_path_distance(
        source: Path,
        reported_threshold: float,
    ) -> float:
        """Recover the p-dispersion bound preceding the reported threshold."""
        adjacency = read_graphml_adjacency(source)
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
            # The archived result records retain the old .txt instance names,
            # but the bound is recovered directly from the original GraphML.
            graphml_source = (
                LOWER_BOUND_OUTPUT.parent / match.group("instance")
            ).resolve().with_suffix(".graphml")
            lower_bounds[instance] = preceding_shortest_path_distance(
                graphml_source,
                float(match.group("value")),
            )

        if set(lower_bounds) != instances:
            raise ValueError(
                f"expected lower bounds for {len(instances)} instances, "
                f"found {len(lower_bounds)}"
            )
        return lower_bounds

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
        COMPARISONS=COMPARISONS,
        LOWER_BOUND_OUTPUT=LOWER_BOUND_OUTPUT,
        VNS_OUTPUTS=VNS_OUTPUTS,
        VnsRun=VnsRun,
        expected_instances=expected_instances,
        group_by_instance=group_by_instance,
        read_discrete_dispersion_lower_bounds=read_discrete_dispersion_lower_bounds,
        read_vns_runs=read_vns_runs,
        runtime_summaries=runtime_summaries
    )


lns_vns = _build_lns_vns_helpers()


VARIANT_MAPPING_OUTPUT = OUTPUT_DIR / "tableS4.tex"
MEAN_STANDARD_ABLATION_OUTPUT = OUTPUT_DIR / "tableS5.tex"
BEST_KNOWN_FEASIBLE_OUTPUT = OUTPUT_DIR / "tableS6.tex"
REPAIR_DIAGNOSTICS_OUTPUT = OUTPUT_DIR / "tableS7.tex"
DESTROY_SIZE_SCHEDULE_OUTPUT = OUTPUT_DIR / "tableS8.tex"
FULL_LNS_TIME_COMPARISON_OUTPUT = OUTPUT_DIR / "tableS9.tex"
LNS_VNS_RUNTIME_OUTPUT = OUTPUT_DIR / "tableS10.tex"
PER_INSTANCE_ABLATION_OUTPUT = OUTPUT_DIR / "tableS11.tex"
ATTRIBUTE_COMPACTNESS_OUTPUT = OUTPUT_DIR / "tableS12.tex"
MIXED_EFFECTS_OUTPUT = OUTPUT_DIR / "tableS13.tex"
LOWER_BOUND_GAPS_OUTPUT = OUTPUT_DIR / "tableS14.tex"
NUMERICAL_RESULTS_OUTPUT = OUTPUT_DIR / "online_supplement_numbers.txt"
RAW_MODEL_REPORT = (
    SCRIPT_DIR / "anova_analysis_output" / "anova_analysis_report.txt"
)
LOG_MODEL_REPORT = (
    SCRIPT_DIR / "anova_analysis_output" / "anova_log_response_report.txt"
)
LNS_RUNS_INPUT = ROOT / "resultsLNS" / "lns.results.runs.txt"
LNS_ABLATION_RUNS_INPUT = (
    ROOT / "resultsLNS" / "lns.ablation.results.runs.txt"
)
LNS_FIXED_RUNS_INPUT = ROOT / "resultsLNS" / "lns.fixed.results.runs.txt"
VNS_RUN_INPUTS = (
    ROOT / "AlyEtAl" / "vns.low.results.runs.txt",
    ROOT / "AlyEtAl" / "vns.medium.results.runs.txt",
    ROOT / "AlyEtAl" / "vns.high.results.runs.txt",
)
GROUP3_VNS_INPUTS = (
    ("VNSl", ROOT / "AlyEtAl" / "vns.low.results.txt"),
    ("VNSm", ROOT / "AlyEtAl" / "vns.medium.results.txt"),
    ("VNSh", ROOT / "AlyEtAl" / "vns.high.results.txt"),
)
GROUP3_LNS_INPUT = ROOT / "resultsLNS" / "lns.results.txt"
LOWER_BOUNDS_INPUT = ROOT / "resultsLNS" / "lower.bounds.results.txt"
INSTANCES_DIR = ROOT / "instances"


def display_ablation_label(label: str) -> str:
    return "Distance-based destroy" if label == "Alternative destroy" else label


GROUP3_METHODS = (
    ("LNS60", "LNS (60~s)"),
    ("LNS300", "LNS (300~s)"),
    ("LNS600", "LNS (600~s)"),
    ("VNSl", "VNS(l)"),
    ("VNSm", "VNS(m)"),
    ("VNSh", "VNS(h)"),
)
GROUP3_TABLE_METHODS = (
    ("VNSl", "VNS(l)"),
    ("VNSm", "VNS(m)"),
    ("VNSh", "VNS(h)"),
    ("LNS60", "LNS (60~s)"),
    ("LNS300", "LNS (300~s)"),
    ("LNS600", "LNS (600~s)"),
)
MODEL_METHODS = (
    ("LNS (60s)", "LNS (60~s)"),
    ("LNS (300s)", "LNS (300~s)"),
    ("LNS (600s)", "LNS (600~s)"),
    ("VNS(l)", "VNS(l)"),
    ("VNS(m)", "VNS(m)"),
    ("VNS(h)", "VNS(h)"),
)
ANOVA_METHOD_ORDER = ("LNS60", "LNS300", "LNS600", "VNSl", "VNSm", "VNSh")
ANOVA_METHOD_LABELS = {
    "LNS60": "LNS (60s)",
    "LNS300": "LNS (300s)",
    "LNS600": "LNS (600s)",
    "VNSl": "VNS(l)",
    "VNSm": "VNS(m)",
    "VNSh": "VNS(h)",
}
ANOVA_FACTOR_COLUMNS = (
    "method",
    "size",
    "demand",
    "workload",
    "customers",
    "type",
)
LEVEL_LABELS = {"l": "25\\%", "m": "50\\%", "h": "90\\%"}
GROUP3_NAME_PATTERN = re.compile(
    r"^d-(?P<d>[hlm])_w-(?P<w>[hlm])_c-(?P<c>[hlm])-"
    r"(?P<kind>[A-Za-z]+)(?P<size>\d+)_G\d+$"
)
GROUP3_VNS_PATTERN = re.compile(
    r"^Instance: (?P<instance>\S+) Best objective: (?P<objective>\S+) "
    r"Infeasibility: (?P<penalty>\S+) Total time \(s\): (?P<runtime>\S+)$"
)
GROUP3_LNS_PATTERN = re.compile(
    r"^RESULTS: (?P<instance>\S+) min \S+ step \S+ max \S+ "
    r"t (?P<time_limit>\S+) obj: (?P<objective>\S+) "
    r"penalty: (?P<penalty>\S+) t: \S+ tBest: \S+$"
)
GROUP3_BOUND_PATTERN = re.compile(
    r"^iterativeRelaxation: (?P<instance>\S+) value: (?P<value>\S+)$"
)
PER_INSTANCE_NAME_PATTERN = re.compile(
    r"^(?P<layout>[A-Za-z]+)(?P<business_units>\d+)_G(?P<identifier>\d+)$"
)


@dataclass(frozen=True)
class SingleRun:
    instance: str
    objective: float
    penalty: float


@dataclass(frozen=True)
class PairwiseResult:
    difference: float
    ci_low: float
    ci_high: float
    adjusted_p_value: float


@dataclass(frozen=True)
class ModelReport:
    fixed_effects: dict[str, tuple[int, float, float]]
    marginal_means: dict[str, float]
    pairwise: dict[tuple[str, str], PairwiseResult]


@dataclass(frozen=True)
class Full600Comparison:
    method: str
    feasibility_change: float
    median_compactness_change: float
    full_score: float
    feasible_pairs: int
    median_gap: float
    mean_gap: float


def validate_repo_path(path: Path, description: str) -> Path:
    """Resolve a path and reject anything outside otherRepo/dtdp."""
    resolved = path.resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError as error:
        raise ValueError(
            f"{description} path must remain inside {ROOT}: {resolved}"
        ) from error
    return resolved


def validate_input_file(path: Path) -> Path:
    path = validate_repo_path(path, "Input")
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")
    return path


def configure_local_inputs() -> None:
    """Validate the archived inputs used by the embedded analyses."""
    for input_path in (
        LNS_RUNS_INPUT,
        LNS_ABLATION_RUNS_INPUT,
        LNS_FIXED_RUNS_INPUT,
        *VNS_RUN_INPUTS,
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
    if instances_by_group["group1"] & instances_by_group["group2"]:
        raise ValueError("group1 and group2 instance names overlap")


def render_variant_mapping_table() -> str:
    rows = (
        (
            "Full LNS",
            "All components",
            "Complete method with increasing destroy size",
        ),
        (
            "Random destroy",
            "BU selection for reassignment",
            "Selects freed BUs uniformly at random",
        ),
        (
            "Distance-based destroy",
            "BU selection for reassignment",
            "Frees BUs farthest from a district reference",
        ),
        (
            "District-based destroy",
            "BU selection for reassignment",
            "Frees all BUs in three selected districts",
        ),
        (
            "Fixed destroy size",
            "Neighborhood-size adaptation",
            "Uses one fixed percentage per time limit",
        ),
        (
            "No fixing",
            "Repair phase",
            "Disables dominance-based variable fixing",
        ),
        (
            "No warm start",
            "Repair phase",
            "Omits the incumbent MIP start",
        ),
        (
            "Heuristic repair",
            "Repair phase",
            "Uses greedy reconstruction and local search",
        ),
        (
            "Only random start",
            "Initial-solution generation",
            "Uses random generation for the first start",
        ),
    )
    body = "\n".join(" & ".join(row) + r" \\" for row in rows)
    return f"""\\begingroup
\\begin{{table}}[htbp]
\\centering
\\caption{{Ablation variants considered in the repeated-run analysis. Complete descriptions of the variants and their implementation are provided in Section~5.3 of the main manuscript.}}
\\label{{tab:exp08-variant-mapping}}
\\small
\\setlength{{\\tabcolsep}}{{3pt}}
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{lll}}
\\toprule
Label & Component & Short description \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
}}
\\end{{table}}
\\endgroup
"""


def render_mean_standard_ablation_table(
    methods: dict[tuple[int, str], list[ablation.Run]],
) -> str:
    labels = (
        "Full LNS",
        "Random destroy",
        "Alternative destroy",
        "District-based destroy",
        "Fixed destroy size",
        "No fixing",
        "No warm start",
        "Heuristic repair",
        "Only random start",
    )
    rows = []
    for label in labels:
        values = []
        for time_limit in ablation.TIME_LIMITS:
            grouped: dict[str, list[ablation.Run]] = defaultdict(list)
            for run in methods[time_limit, label]:
                grouped[run.instance].append(run)

            feasible_run_counts = [
                sum(run.penalty == 0 for run in instance_runs)
                for instance_runs in grouped.values()
            ]
            values.extend(
                (
                    str(min(feasible_run_counts)),
                    f"{statistics.mean(feasible_run_counts):.1f}",
                    f"{statistics.stdev(feasible_run_counts):.1f}",
                )
            )
            if label == "Full LNS":
                values.extend(("--", "--"))
            else:
                differences = ablation.instance_objective_gaps(
                    methods, time_limit, label
                )
                mean_difference = statistics.mean(differences)
                if abs(mean_difference) < 0.05:
                    mean_difference = 0.0
                values.extend(
                    (
                        f"{mean_difference:.1f}\\%",
                        f"{statistics.stdev(differences):.1f}",
                    )
                )
        rows.append(
            f"{display_ablation_label(label)} & "
            + " & ".join(values)
            + r" \\"
        )
        if label == "Full LNS":
            rows.append(r"\midrule")

    body = "\n".join(rows)
    return f"""\\begingroup
\\begin{{table}}[htbp]
\\centering
\\caption{{Distribution of balance-feasible runs and relative compactness difference (RCD) by method and time limit. Under Feasible runs, Min., Mean, and $s$ summarize, across the 120 instances, the number of balance-feasible outcomes among the 30 independent runs. Under RCD, Mean and $s$ are the mean and sample standard deviation of the instance-level RCD values defined in Section~5.3 of the main manuscript. The RCD cells are omitted for Full LNS because its difference from itself is zero by definition.}}
\\label{{tab:exp08-method-mean-summary}}
\\scriptsize
\\setlength{{\\tabcolsep}}{{2pt}}
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{lrrrrrrrrrrrrrrr}}
\\toprule
& \\multicolumn{{5}}{{c}}{{60 s}} & \\multicolumn{{5}}{{c}}{{300 s}} & \\multicolumn{{5}}{{c}}{{600 s}} \\\\
\\cmidrule(lr){{2-6}} \\cmidrule(lr){{7-11}} \\cmidrule(lr){{12-16}}
& \\multicolumn{{3}}{{c}}{{Feasible runs}} & \\multicolumn{{2}}{{c}}{{RCD}}
& \\multicolumn{{3}}{{c}}{{Feasible runs}} & \\multicolumn{{2}}{{c}}{{RCD}}
& \\multicolumn{{3}}{{c}}{{Feasible runs}} & \\multicolumn{{2}}{{c}}{{RCD}} \\\\
\\cmidrule(lr){{2-4}} \\cmidrule(lr){{5-6}}
\\cmidrule(lr){{7-9}} \\cmidrule(lr){{10-11}}
\\cmidrule(lr){{12-14}} \\cmidrule(lr){{15-16}}
Method & Min. & Mean & $s$ & Mean & $s$ & Min. & Mean & $s$ & Mean & $s$ & Min. & Mean & $s$ & Mean & $s$ \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
}}
\\end{{table}}
\\endgroup
"""


def render_best_known_feasible_table(
    methods: dict[tuple[int, str], list[ablation.Run]],
    vns_runs: dict[str, list[lns_vns.VnsRun]],
) -> str:
    lns_labels = (
        "Full LNS",
        "Random destroy",
        "Alternative destroy",
        "District-based destroy",
        "Fixed destroy size",
        "No fixing",
        "No warm start",
        "Heuristic repair",
        "Only random start",
    )
    configuration_runs: dict[
        str, list[ablation.Run] | list[lns_vns.VnsRun]
    ] = {
        f"{label}|{time_limit}": methods[time_limit, label]
        for label in lns_labels
        for time_limit in ablation.TIME_LIMITS
    }
    vns_keys = []
    for _, label, version in lns_vns.COMPARISONS:
        key = f"VNS|{label}"
        configuration_runs[key] = vns_runs[version]
        vns_keys.append(key)
    if len(configuration_runs) != 30:
        raise ValueError(
            f"expected 30 configurations, found {len(configuration_runs)}"
        )

    feasible: dict[str, dict[str, list[float]]] = {}
    for key, runs in configuration_runs.items():
        by_instance: dict[str, list[float]] = defaultdict(list)
        for run in runs:
            if run.penalty == 0:
                by_instance[run.instance].append(run.objective)
        feasible[key] = dict(by_instance)

    instances = sorted(ablation.expected_instances())
    if len(instances) != 120:
        raise ValueError(f"expected 120 instances, found {len(instances)}")
    best_known = {
        instance: min(
            min(by_instance[instance])
            for by_instance in feasible.values()
            if instance in by_instance
        )
        for instance in instances
    }

    def format_cell(key: str) -> str:
        by_instance = feasible[key]
        best_count = sum(
            instance in by_instance
            and math.isclose(
                min(by_instance[instance]),
                best_known[instance],
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            for instance in instances
        )
        if any(instance not in by_instance for instance in instances):
            mean_compactness = "--"
        else:
            per_instance_means = (
                statistics.mean(by_instance[instance]) for instance in instances
            )
            mean_compactness = f"{statistics.mean(per_instance_means):.1f}"
        return f"{best_count} ({mean_compactness})"

    lns_rows = "\n".join(
        f"{display_ablation_label(label)} & "
        + " & ".join(
            format_cell(f"{label}|{time_limit}")
            for time_limit in ablation.TIME_LIMITS
        )
        + r" \\"
        for label in lns_labels
    )
    vns_cells = " & ".join(format_cell(key) for key in vns_keys)
    return f"""\\begingroup
\\begin{{table}}[htbp]
\\centering
\\caption{{Best-known balance-feasible compactness across 30 algorithm configurations. Each cell reports the number of instances on which the configuration attains the overall best balance-feasible compactness across all runs and configurations. The parenthetical value reports mean compactness over balance-feasible runs, averaged first within each instance and then across the 120 instances.}}
\\label{{tab:best-known-feasible}}
\\small
\\setlength{{\\tabcolsep}}{{5pt}}
\\begin{{tabular}}{{lrrr}}
\\toprule
LNS variant & 60 s & 300 s & 600 s \\\\
\\midrule
{lns_rows}
\\midrule
& VNS(l) & VNS(m) & VNS(h) \\\\
VNS & {vns_cells} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
\\endgroup
"""


def render_repair_diagnostics_table(
    methods: dict[tuple[int, str], list[ablation.Run]],
) -> str:
    labels = (
        "Full LNS",
        "Random destroy",
        "Alternative destroy",
        "District-based destroy",
        "Fixed destroy size",
        "No warm start",
        "Only random start",
        "No fixing",
    )
    rows = []
    for label in labels:
        values = []
        for time_limit in ablation.TIME_LIMITS:
            runs = methods[time_limit, label]
            total_repairs = sum(run.repairs for run in runs)
            repairs_per_run = statistics.mean(run.repairs for run in runs)
            if total_repairs:
                candidate_variables = sum(
                    run.free_variables + run.fixed_variables for run in runs
                )
                fixed_percentage = (
                    ""
                    if label == "No fixing"
                    else f"{100.0 * sum(run.fixed_variables for run in runs) / candidate_variables:.1f}\\%"
                )
                values.extend(
                    (
                        f"{sum(run.free_variables for run in runs) / total_repairs:,.1f}",
                        fixed_percentage,
                        f"{repairs_per_run:.1f}",
                        f"{sum(run.solving_repairs_time for run in runs) / total_repairs:.2f}",
                        f"{100.0 * sum(run.optimal_repairs for run in runs) / total_repairs:.1f}\\%",
                        f"{100.0 * sum(run.balance_improvements for run in runs) / total_repairs:.1f}\\%",
                        f"{100.0 * sum(run.compactness_improvements for run in runs) / total_repairs:.1f}\\%",
                    )
                )
            else:
                values.extend(
                    (
                        "--",
                        "--",
                        f"{repairs_per_run:.1f}",
                        "--",
                        "--",
                        "--",
                        "--",
                    )
                )
        rows.append(
            f"{display_ablation_label(label)} & "
            + " & ".join(values)
            + r" \\"
        )
        if label == "Full LNS":
            rows.append(r"\midrule")

    body = "\n".join(rows)
    return f"""\\begingroup
\\begin{{table}}[!htbp]
\\centering
\\caption{{Repair-search diagnostics by method and time limit. Free vars is the mean number of BU--district assignment variables left free in each MIP repair. Fixed is the percentage of candidate assignment variables eliminated by variable fixing; R/run is the mean number of repairs per run; t/repair is the mean MIP repair time in seconds; Opt. is the percentage of repairs solved to optimality; and Bal. and Comp. are the percentages of repairs improving balance and compactness, respectively. Except for R/run, the statistics use aggregate totals over 120 instances and 30 runs each. The Fixed cells are blank for No fixing because that mechanism is disabled. Heuristic repair is omitted because it does not invoke MIP repair.}}
\\label{{tab:exp08-repair-diagnostics}}
\\scriptsize
\\setlength{{\\tabcolsep}}{{2pt}}
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{l*{{21}}{{r}}}}
\\toprule
& \\multicolumn{{7}}{{c}}{{60 s}} & \\multicolumn{{7}}{{c}}{{300 s}} & \\multicolumn{{7}}{{c}}{{600 s}} \\\\
\\cmidrule(lr){{2-8}} \\cmidrule(lr){{9-15}} \\cmidrule(lr){{16-22}}
Method & Free vars & Fixed & R/run & t/repair (s) & Opt. & Bal. & Comp. & Free vars & Fixed & R/run & t/repair (s) & Opt. & Bal. & Comp. & Free vars & Fixed & R/run & t/repair (s) & Opt. & Bal. & Comp. \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
}}
\\end{{table}}
\\endgroup
"""


def compare_destroy_size_schedules(
    methods: dict[tuple[int, str], list[ablation.Run]],
) -> list[ablation.FixedComparison]:
    comparisons = []
    for time_limit in ablation.TIME_LIMITS:
        increasing = methods[time_limit, "Full LNS"]
        fixed = methods[time_limit, "Fixed destroy size"]
        increasing_by_instance: dict[str, list[ablation.Run]] = defaultdict(list)
        fixed_by_instance: dict[str, list[ablation.Run]] = defaultdict(list)
        for run in increasing:
            increasing_by_instance[run.instance].append(run)
        for run in fixed:
            fixed_by_instance[run.instance].append(run)
        if increasing_by_instance.keys() != fixed_by_instance.keys():
            raise ValueError(
                f"variable- and fixed-size methods cover different instances at "
                f"{time_limit} seconds"
            )

        increasing_wins = ties = fixed_wins = 0
        feasible_gaps = []
        for instance in increasing_by_instance:
            increasing_feasible = [
                run
                for run in increasing_by_instance[instance]
                if run.penalty == 0
            ]
            fixed_feasible = [
                run for run in fixed_by_instance[instance] if run.penalty == 0
            ]
            for increasing_run in increasing_feasible:
                for fixed_run in fixed_feasible:
                    if increasing_run.objective < fixed_run.objective:
                        increasing_wins += 1
                    elif fixed_run.objective < increasing_run.objective:
                        fixed_wins += 1
                    else:
                        ties += 1
                    feasible_gaps.append(
                        100.0
                        * (fixed_run.objective - increasing_run.objective)
                        / increasing_run.objective
                    )

        increasing_objectives = [
            run.objective for run in increasing if run.penalty == 0
        ]
        fixed_objectives = [run.objective for run in fixed if run.penalty == 0]
        increasing_schedules = {
            ablation.schedule_values(run) for run in increasing
        }
        fixed_values = {run.minimum for run in fixed}
        if len(increasing_schedules) != 1 or len(fixed_values) != 1:
            raise ValueError(
                f"inconsistent destroy-size parameters at {time_limit} seconds"
            )

        comparisons.append(
            ablation.FixedComparison(
                time_limit=time_limit,
                increasing_values=increasing_schedules.pop(),
                fixed_value=fixed_values.pop(),
                increasing_feasibility=(
                    100.0 * len(increasing_objectives) / len(increasing)
                ),
                fixed_feasibility=100.0 * len(fixed_objectives) / len(fixed),
                increasing_median=statistics.median(increasing_objectives),
                fixed_median=statistics.median(fixed_objectives),
                increasing_wins=increasing_wins,
                ties=ties,
                fixed_wins=fixed_wins,
                feasible_pairs=len(feasible_gaps),
                median_gap=statistics.median(feasible_gaps),
            )
        )
    return comparisons


def format_destroy_sizes(values: tuple[int, ...]) -> str:
    if len(values) <= 5:
        content = ", ".join(str(value) for value in values)
    else:
        content = (
            f"{values[0]}, {values[1]}, \\ldots, "
            f"{values[-2]}, {values[-1]}"
        )
    return f"\\{{{content}\\}}"


def render_destroy_size_schedule_table(
    comparisons: list[ablation.FixedComparison],
) -> str:
    rows = []
    for result in comparisons:
        increasing_values = format_destroy_sizes(result.increasing_values)
        fixed_value = format_destroy_sizes((result.fixed_value,))
        rows.append(
            f"{result.time_limit} & {increasing_values} & {fixed_value} & "
            f"{result.increasing_feasibility:.1f}\\% & "
            f"{result.fixed_feasibility:.1f}\\% & "
            f"{result.increasing_median:.1f} & {result.fixed_median:.1f} & "
            f"{result.increasing_wins:,} & {result.ties:,} & "
            f"{result.fixed_wins:,} & {result.feasible_pairs:,} & "
            f"{result.median_gap:.1f}\\% \\\\"
        )

    body = "\n".join(rows)
    return f"""\\begingroup
\\begin{{table}}[htbp]
\\centering
\\caption{{Comparison of the Full LNS variable destroy-size schedule and the fixed destroy-size method over 120 instances and 30 runs per method and instance. Destroy size lists the percentages of BUs selected for destruction. Feasibility and median compactness summarize the runs. The pairwise outcomes compare every balance-feasible run of one method with every balance-feasible run of the other method on the same instance, and Feas. pairs gives the number of such comparisons.}}
\\label{{tab:exp08-fixed-p-status}}
\\scriptsize
\\setlength{{\\tabcolsep}}{{2pt}}
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{rllrrrrrrrrr}}
\\toprule
& \\multicolumn{{2}}{{c}}{{Destroy size (\\%)}} & \\multicolumn{{2}}{{c}}{{Feasibility}} & \\multicolumn{{2}}{{c}}{{Median compactness}} & \\multicolumn{{4}}{{c}}{{Pairwise outcome}} & \\\\
\\cmidrule(lr){{2-3}} \\cmidrule(lr){{4-5}} \\cmidrule(lr){{6-7}} \\cmidrule(lr){{8-11}}
Time (s) & Full LNS & Fixed size & Full LNS & Fixed size & Full LNS & Fixed size & Full & Tie & Fixed & Feas. pairs & Med. gap \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
}}
\\end{{table}}
\\endgroup
"""


def compare_with_full_lns_600(
    methods: dict[tuple[int, str], list[ablation.Run]],
    vns_runs: dict[str, list[lns_vns.VnsRun]],
) -> list[Full600Comparison]:
    reference = methods[600, "Full LNS"]
    comparators: tuple[
        tuple[str, list[ablation.Run] | list[lns_vns.VnsRun]], ...
    ] = (
        ("Full LNS (60~s)", methods[60, "Full LNS"]),
        ("Full LNS (300~s)", methods[300, "Full LNS"]),
        ("VNS(l)", vns_runs["test6WithSeed.py"]),
        ("VNS(m)", vns_runs["test7WithSeed.py"]),
        ("VNS(h)", vns_runs["test8WithSeed.py"]),
    )
    reference_by_instance: dict[str, list[ablation.Run]] = defaultdict(list)
    for run in reference:
        reference_by_instance[run.instance].append(run)
    reference_objectives = [run.objective for run in reference if run.penalty == 0]
    reference_feasibility = 100.0 * len(reference_objectives) / len(reference)
    comparisons = []
    for label, comparator in comparators:
        comparator_by_instance = defaultdict(list)
        for run in comparator:
            comparator_by_instance[run.instance].append(run)
        if reference_by_instance.keys() != comparator_by_instance.keys():
            raise ValueError(f"{label} and Full LNS (600~s) cover different instances")

        score_total = 0.0
        feasible_gaps = []
        instance_median_changes = []
        for instance in sorted(reference_by_instance):
            reference_feasible = [
                run
                for run in reference_by_instance[instance]
                if run.penalty == 0
            ]
            comparator_feasible = [
                run
                for run in comparator_by_instance[instance]
                if run.penalty == 0
            ]
            if reference_feasible and comparator_feasible:
                instance_median_changes.append(
                    statistics.median(
                        run.objective for run in comparator_feasible
                    )
                    - statistics.median(
                        run.objective for run in reference_feasible
                    )
                )
            for reference_run in reference_feasible:
                for comparator_run in comparator_feasible:
                    if math.isclose(
                        reference_run.objective,
                        comparator_run.objective,
                        rel_tol=1e-9,
                        abs_tol=1e-9,
                    ):
                        score_total += 0.5
                    elif reference_run.objective < comparator_run.objective:
                        score_total += 1.0
                    feasible_gaps.append(
                        100.0
                        * (comparator_run.objective - reference_run.objective)
                        / reference_run.objective
                    )

        comparator_objectives = [
            run.objective for run in comparator if run.penalty == 0
        ]
        if not feasible_gaps or not comparator_objectives:
            raise ValueError(f"{label} has no feasible comparison with Full LNS")
        if len(instance_median_changes) != len(reference_by_instance):
            raise ValueError(
                f"{label} and Full LNS are not jointly feasible on every instance"
            )
        comparator_feasibility = (
            100.0 * len(comparator_objectives) / len(comparator)
        )
        comparisons.append(
            Full600Comparison(
                method=label,
                feasibility_change=reference_feasibility - comparator_feasibility,
                median_compactness_change=statistics.median(
                    instance_median_changes
                ),
                full_score=score_total / len(feasible_gaps),
                feasible_pairs=len(feasible_gaps),
                median_gap=statistics.median(feasible_gaps),
                mean_gap=statistics.mean(feasible_gaps),
            )
        )
    return comparisons


def render_full_lns_time_comparison_table(
    comparisons: list[Full600Comparison],
) -> str:
    rows = "\n".join(
        f"{result.method} & {result.feasibility_change:.1f} & "
        f"{result.median_compactness_change:.1f} & "
        f"{result.full_score:.3f} & {result.feasible_pairs:,} & "
        f"{result.median_gap:.1f}\\% & {result.mean_gap:.1f}\\% \\\\"
        for result in comparisons
    )
    return f"""\\begingroup
\\begin{{table}}[htbp]
\\centering
\\caption{{Comparison of Full LNS (600~s) with the shorter Full-LNS configurations and the three VNS configurations over 120 instances and 30 runs per method and instance.}}
\\label{{tab:exp08-methods-vs-full-600}}
\\small
\\setlength{{\\tabcolsep}}{{5pt}}
\\begin{{tabular}}{{lrrrrrr}}
\\toprule
Method & Feas. change & Comp. change & Score & Feas. pairs & Med. gap & Mean gap \\\\
\\midrule
{rows}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
\\endgroup
"""


def render_lns_vns_runtime_table(
    methods: dict[tuple[int, str], list[ablation.Run]],
    vns_runs: dict[str, list[lns_vns.VnsRun]],
) -> str:
    summaries = {
        summary.method: summary
        for summary in lns_vns.runtime_summaries(methods, vns_runs)
    }
    configurations = (
        ("LNS (60~s)", "LNS 60 s", "Time to best", methods[60, "Full LNS"]),
        ("VNS(l)", "VNS(l)", "Total runtime", vns_runs["test6WithSeed.py"]),
        (
            "LNS (300~s)",
            "LNS 300 s",
            "Time to best",
            methods[300, "Full LNS"],
        ),
        ("VNS(m)", "VNS(m)", "Total runtime", vns_runs["test7WithSeed.py"]),
        (
            "LNS (600~s)",
            "LNS 600 s",
            "Time to best",
            methods[600, "Full LNS"],
        ),
        ("VNS(h)", "VNS(h)", "Total runtime", vns_runs["test8WithSeed.py"]),
    )
    rows = []
    for index, (display, summary_key, measure, runs) in enumerate(configurations):
        summary = summaries[summary_key]
        if summary.runs != len(runs):
            raise ValueError(f"{summary_key}: inconsistent run count")
        rows.append(
            f"{display} & {measure} & {summary.average:,.1f} & "
            f"{summary.q1:,.1f} & {summary.median:,.1f} & "
            f"{summary.q3:,.1f} & {summary.standard_deviation:,.1f} \\\\"
        )
        if index in (1, 3):
            rows.append(r"\midrule")

    body = "\n".join(rows)
    return f"""\\begingroup
\\begin{{table}}[htbp]
\\centering
\\caption{{Computing-time summary for the LNS and VNS configurations. For LNS, the statistics report the time at which the solution ultimately returned as best was first reached; execution continues until the stated time limit. For VNS, they report total runtime. Thus, VNS(l), VNS(m), and VNS(h) are parameter configurations paired with LNS (60~s), LNS (300~s), and LNS (600~s), respectively, rather than methods stopped at those time limits. $Q_1$ and $Q_3$ are the first and third quartiles, and Std. dev. is the sample standard deviation.}}
\\label{{tab:exp08-lns-vns-runtime}}
\\small
\\setlength{{\\tabcolsep}}{{4pt}}
\\begin{{tabular}}{{llrrrrr}}
\\toprule
Method & Time measure & Mean (s) & $Q_1$ (s) & Median (s) & $Q_3$ (s) & Std. dev. (s) \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
\\endgroup
"""


def render_per_instance_ablation_table(
    methods: dict[tuple[int, str], list[ablation.Run]],
) -> str:
    labels = (
        "Full LNS",
        "Random destroy",
        "Alternative destroy",
        "District-based destroy",
        "Fixed destroy size",
        "No fixing",
        "No warm start",
        "Heuristic repair",
        "Only random start",
    )
    grouped: dict[tuple[int, str], dict[str, list[ablation.Run]]] = {}
    for time_limit in ablation.TIME_LIMITS:
        for label in labels:
            by_instance: dict[str, list[ablation.Run]] = defaultdict(list)
            for run in methods[time_limit, label]:
                by_instance[run.instance].append(run)
            grouped[time_limit, label] = dict(by_instance)

    instances = []
    for instance in ablation.expected_instances():
        match = PER_INSTANCE_NAME_PATTERN.fullmatch(instance)
        if not match:
            raise ValueError(f"cannot parse instance name: {instance}")
        instances.append(
            (
                match.group("layout"),
                int(match.group("business_units")),
                int(match.group("identifier")),
                instance,
            )
        )

    instances.sort(
        key=lambda values: (values[0].casefold(), values[1], values[2])
    )
    rows = []
    for instance_index, (
        layout,
        business_units,
        identifier,
        instance,
    ) in enumerate(instances):
        summaries = {}
        for label in labels:
            for time_limit in ablation.TIME_LIMITS:
                runs = grouped[time_limit, label][instance]
                feasible = [run for run in runs if run.penalty == 0]
                if feasible:
                    objectives = [run.objective for run in feasible]
                    deviation = (
                        statistics.stdev(objectives)
                        if len(objectives) > 1
                        else 0.0
                    )
                    summaries[time_limit, label] = (
                        len(feasible),
                        min(objectives),
                        statistics.mean(objectives),
                        statistics.median(objectives),
                        deviation,
                    )
                else:
                    summaries[time_limit, label] = (0, None, None, None, None)

        best_feasibility = max(summary[0] for summary in summaries.values())
        best_compactness = {
            statistic_index: min(
                float(f"{summary[statistic_index]:.1f}")
                for summary in summaries.values()
                if summary[statistic_index] is not None
            )
            for statistic_index in (1, 2, 3)
        }

        for label_index, label in enumerate(labels):
            cells = [
                layout if label_index == 0 else "",
                str(business_units) if label_index == 0 else "",
                str(identifier) if label_index == 0 else "",
                display_ablation_label(label),
            ]
            for time_limit in ablation.TIME_LIMITS:
                summary = summaries[time_limit, label]
                feasibility = str(summary[0])
                if summary[0] == best_feasibility:
                    feasibility = f"\\textbf{{{feasibility}}}"
                cells.append(feasibility)
                for statistic_index, value in enumerate(summary[1:], start=1):
                    if value is None:
                        cells.append("--")
                        continue
                    displayed = f"{value:.1f}"
                    if (
                        statistic_index in best_compactness
                        and float(displayed) == best_compactness[statistic_index]
                    ):
                        displayed = f"\\textbf{{{displayed}}}"
                    cells.append(displayed)
            row_ending = r" \\*" if label_index < len(labels) - 1 else r" \\"
            rows.append(" & ".join(cells) + row_ending)
        if instance_index < len(instances) - 1:
            rows.append(r"\midrule")

    body = "\n".join(rows)
    caption = (
        "Per-instance repeated-run summary for the LNS ablation study. Layout, "
        "BUs, and ID identify each instance; blank cells in these columns continue "
        "the instance identified above. Each instance has one row for Full LNS and "
        "each of the eight ablation variants. Within each time limit, Feas. is the "
        "number of balance-feasible runs among the 30 independent runs. Best, Mean, "
        "Median, and $s$ are calculated from compactness values of balance-feasible "
        "runs only; $s$ is the sample standard deviation and is reported as zero "
        "when only one run is feasible. A dash means that no compactness summary "
        "can be calculated because none of the runs is balance-feasible. Within "
        "each instance, bold identifies the best value across all 27 method and "
        "time-limit combinations: the highest Feas. value and the lowest Best, "
        "Mean, and Median values. All ties at the displayed precision are bold."
    )
    time_header = " & ".join(
        f"\\multicolumn{{5}}{{c}}{{{time_limit} s}}"
        for time_limit in ablation.TIME_LIMITS
    )
    time_rules = " ".join(
        f"\\cmidrule(lr){{{start}-{start + 4}}}"
        for start in range(5, 20, 5)
    )
    metric_header = " & ".join(
        ("Feas.", "Best", "Mean", "Median", "$s$")
        * len(ablation.TIME_LIMITS)
    )
    return f"""\\begingroup
\\tiny
\\setlength{{\\LTcapwidth}}{{\\textwidth}}
\\setlength{{\\tabcolsep}}{{0.75pt}}
\\renewcommand{{\\arraystretch}}{{1.05}}
\\begin{{longtable}}{{lrrl{("rrrrr" * len(ablation.TIME_LIMITS))}}}
\\caption{{{caption}}}\\label{{tab:exp08-per-instance-summary}} \\\\
\\toprule
& & & & {time_header} \\\\
{time_rules}
Layout & BUs & ID & Method & {metric_header} \\\\
\\midrule
\\endfirsthead
\\multicolumn{{19}}{{c}}{{\\tablename~\\thetable\\ (continued)}}\\\\
\\toprule
& & & & {time_header} \\\\
{time_rules}
Layout & BUs & ID & Method & {metric_header} \\\\
\\midrule
\\endhead
\\midrule
\\multicolumn{{19}}{{r}}{{Continued on next page}}\\\\
\\endfoot
\\bottomrule
\\endlastfoot
{body}
\\end{{longtable}}
\\endgroup
"""


def load_group3_single_run_results(
) -> tuple[dict[str, list[SingleRun]], dict[str, float]]:
    methods: dict[str, list[SingleRun]] = defaultdict(list)
    for method, source_path in GROUP3_VNS_INPUTS:
        source = validate_input_file(source_path)
        for line_number, line in enumerate(
            source.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line:
                continue
            match = GROUP3_VNS_PATTERN.fullmatch(line)
            if not match:
                raise ValueError(f"{source}:{line_number}: malformed VNS result")
            instance = Path(match.group("instance")).stem
            if GROUP3_NAME_PATTERN.fullmatch(instance):
                methods[method].append(
                    SingleRun(
                        instance=instance,
                        objective=float(match.group("objective")),
                        penalty=float(match.group("penalty")),
                    )
                )

    lns_source = validate_input_file(GROUP3_LNS_INPUT)
    expected_method_keys = {key for key, _ in GROUP3_METHODS}
    for line_number, line in enumerate(
        lns_source.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line:
            continue
        match = GROUP3_LNS_PATTERN.fullmatch(line)
        if not match:
            raise ValueError(f"{lns_source}:{line_number}: malformed LNS result")
        instance = Path(match.group("instance")).stem
        if not GROUP3_NAME_PATTERN.fullmatch(instance):
            continue
        time_limit = int(round(float(match.group("time_limit"))))
        method = f"LNS{time_limit}"
        if method not in expected_method_keys:
            continue
        methods[method].append(
            SingleRun(
                instance=instance,
                objective=float(match.group("objective")),
                penalty=float(match.group("penalty")),
            )
        )

    if set(methods) != expected_method_keys:
        raise ValueError(f"unexpected group-3 methods: {sorted(methods)}")
    expected_instances: set[str] | None = None
    for method, runs in methods.items():
        instances = [run.instance for run in runs]
        if len(instances) != 2430 or len(set(instances)) != 2430:
            raise ValueError(f"{method}: expected 2,430 unique instances")
        if expected_instances is None:
            expected_instances = set(instances)
        elif set(instances) != expected_instances:
            raise ValueError(f"{method}: inconsistent instance set")

    bound_source = validate_input_file(LOWER_BOUNDS_INPUT)
    lower_bounds: dict[str, float] = {}
    for line_number, line in enumerate(
        bound_source.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line:
            continue
        match = GROUP3_BOUND_PATTERN.fullmatch(line)
        if not match:
            raise ValueError(
                f"{bound_source}:{line_number}: malformed lower-bound result"
            )
        instance = Path(match.group("instance")).stem
        if not GROUP3_NAME_PATTERN.fullmatch(instance):
            continue
        if instance in lower_bounds:
            raise ValueError(f"duplicate lower bound for {instance}")
        lower_bounds[instance] = float(match.group("value")) - 1.0
    if expected_instances is None or set(lower_bounds) != expected_instances:
        raise ValueError("group-3 runs and lower-bound instance sets differ")
    return dict(methods), lower_bounds


def render_attribute_compactness_table(
    methods: dict[str, list[SingleRun]],
) -> str:
    run_by_method_instance = {
        method: {run.instance: run for run in runs}
        for method, runs in methods.items()
    }
    instances = sorted(next(iter(run_by_method_instance.values())))
    best_by_instance = {
        instance: min(
            run_by_method_instance[method][instance].objective
            for method, _ in GROUP3_METHODS
        )
        for instance in instances
    }
    grouped_instances: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for instance in instances:
        match = GROUP3_NAME_PATTERN.fullmatch(instance)
        if match is None:
            raise ValueError(f"malformed group-3 instance name: {instance}")
        grouped_instances[
            match.group("d"), match.group("w"), match.group("c")
        ].append(instance)

    def summarize_method(method: str, selected_instances: list[str]) -> str:
        gaps = []
        attainments = 0
        for instance in selected_instances:
            objective = run_by_method_instance[method][instance].objective
            best = best_by_instance[instance]
            gaps.append(100.0 * (objective - best) / best)
            attainments += math.isclose(
                objective, best, rel_tol=1e-9, abs_tol=1e-9
            )
        return f"{statistics.mean(gaps):.2f}\\% ({attainments:,})"

    rows = []
    levels = ("l", "m", "h")
    for demand in levels:
        for workload in levels:
            for customers in levels:
                group = grouped_instances[demand, workload, customers]
                if len(group) != 90:
                    raise ValueError("expected 90 instances per attribute group")
                cells = [
                    summarize_method(method, group)
                    for method, _ in GROUP3_TABLE_METHODS
                ]
                demand_cell = (
                    rf"\multirow{{9}}{{*}}{{{LEVEL_LABELS[demand]}}}"
                    if workload == levels[0] and customers == levels[0]
                    else ""
                )
                workload_cell = (
                    rf"\multirow{{3}}{{*}}{{{LEVEL_LABELS[workload]}}}"
                    if customers == levels[0]
                    else ""
                )
                rows.append(
                    " & ".join(
                        (
                            demand_cell,
                            workload_cell,
                            LEVEL_LABELS[customers],
                            *cells,
                        )
                    )
                    + r" \\"
                )
            if workload != levels[-1]:
                rows.append(r"\cline{2-9}")
        if demand != levels[-1]:
            rows.append(r"\hline")

    total_cells = [
        summarize_method(method, instances)
        for method, _ in GROUP3_TABLE_METHODS
    ]
    rows.extend(
        [
            r"\midrule",
            "Total & & & " + " & ".join(total_cells) + r" \\"
        ]
    )

    method_headers = " & ".join(label for _, label in GROUP3_TABLE_METHODS)
    body = "\n".join(rows)
    return f"""\\begingroup
\\begin{{table}}[p]
\\centering
\\caption{{Average relative compactness gap for each LNS and VNS configuration across the 2,430 third-set instances, calculated with respect to the best compactness value obtained by any of the six configurations, irrespective of balance feasibility. Values in parentheses report the number of instances on which each method achieves the best compactness value, with ties counted for every method. Each combination of low-valued demand, workload, and customer group contains 90 instances, and the final row reports the overall average gaps and total counts.}}
\\label{{tab:group3-attribute-compactness}}
\\scriptsize
\\setlength{{\\tabcolsep}}{{2pt}}
\\renewcommand{{\\arraystretch}}{{1.12}}
\\resizebox{{0.8\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{cccrrrrrr}}
\\toprule
Demand & Workload & Customers & {method_headers} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
}}
\\renewcommand{{\\arraystretch}}{{1}}
\\end{{table}}
\\endgroup
"""


def build_anova_dataframe(
    methods: dict[str, list[SingleRun]],
) -> pd.DataFrame:
    """Build the repeated-measures frame used by the mixed-effects tests."""
    if set(methods) != set(ANOVA_METHOD_ORDER):
        raise ValueError("ANOVA input does not contain the six expected methods")

    rows = []
    for method in ANOVA_METHOD_ORDER:
        for run in methods[method]:
            match = GROUP3_NAME_PATTERN.fullmatch(run.instance)
            if match is None:
                raise ValueError(
                    f"Could not parse ANOVA instance name: {run.instance}"
                )
            rows.append(
                {
                    "instance": run.instance,
                    "name": run.instance,
                    "obj": run.objective,
                    "penalty": run.penalty,
                    "method": method,
                    "size": match.group("size"),
                    "demand": LEVEL_LABELS[match.group("d")].replace(
                        "\\%", "%"
                    ),
                    "workload": LEVEL_LABELS[match.group("w")].replace(
                        "\\%", "%"
                    ),
                    "customers": LEVEL_LABELS[match.group("c")].replace(
                        "\\%", "%"
                    ),
                    "type": match.group("kind"),
                }
            )

    dataframe = pd.DataFrame(rows)
    dataframe.sort_values(
        ["method", "type", "size", "instance"], inplace=True
    )
    dataframe.reset_index(drop=True, inplace=True)
    dataframe["method"] = pd.Categorical(
        dataframe["method"], categories=ANOVA_METHOD_ORDER, ordered=True
    )
    dataframe["size"] = pd.Categorical(
        dataframe["size"], categories=("486", "600", "726"), ordered=True
    )
    for factor in ("demand", "workload", "customers"):
        dataframe[factor] = pd.Categorical(
            dataframe[factor],
            categories=("25%", "50%", "90%"),
            ordered=True,
        )
    dataframe["type"] = pd.Categorical(
        dataframe["type"],
        categories=("Center", "Corners", "Diagonal"),
        ordered=True,
    )
    return dataframe


def validate_anova_repeated_measures(dataframe: pd.DataFrame) -> None:
    duplicate_count = int(
        dataframe.duplicated(("instance", "method")).sum()
    )
    if duplicate_count:
        raise ValueError(
            f"Found {duplicate_count} duplicate ANOVA instance-method rows"
        )
    method_counts = dataframe.groupby(
        "instance", observed=False
    )["method"].nunique()
    incomplete = method_counts[
        method_counts != len(ANOVA_METHOD_ORDER)
    ]
    if not incomplete.empty:
        raise ValueError(
            f"ANOVA repeated-measures data has {len(incomplete)} incomplete "
            "instances"
        )


def add_anova_response(
    dataframe: pd.DataFrame, log_response: bool
) -> tuple[pd.DataFrame, str]:
    frame = dataframe.copy()
    if log_response:
        if (frame["obj"] <= 0).any():
            raise ValueError("Cannot log-transform nonpositive objectives")
        frame["response"] = np.log(frame["obj"])
        return frame, "log(obj)"
    frame["response"] = frame["obj"].astype(float)
    return frame, "obj"


def fit_anova_mixed_model(dataframe: pd.DataFrame):
    try:
        import statsmodels.formula.api as smf
        from statsmodels.tools.sm_exceptions import ConvergenceWarning
    except ImportError as error:
        raise RuntimeError(
            "doSupplementTables.py requires statsmodels to generate the "
            "mixed-effects reports"
        ) from error

    formula = (
        "response ~ C(method) + C(size) + C(demand) + C(workload) + "
        "C(customers) + C(type)"
    )
    model = smf.mixedlm(
        formula,
        data=dataframe,
        groups=dataframe["instance"],
        re_formula="1",
    )
    last_error: Exception | None = None
    for fit_kwargs in (
        {"method": "lbfgs", "reml": False, "disp": False},
        {"method": "powell", "reml": False, "disp": False},
        {"method": "nm", "reml": False, "disp": False},
    ):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                warnings.simplefilter("ignore", UserWarning)
                warnings.simplefilter("ignore", ConvergenceWarning)
                return model.fit(**fit_kwargs)
        except Exception as error:  # pragma: no cover - optimizer fallback
            last_error = error
    raise RuntimeError(
        f"Mixed model failed to converge: {last_error}"
    ) from last_error


def anova_factor_constraints(
    result, factor: str
) -> tuple[np.ndarray, list[str]]:
    parameter_names = list(result.fe_params.index)
    matched = [
        (index, name)
        for index, name in enumerate(parameter_names)
        if name.startswith(f"C({factor})[T.")
    ]
    constraint = np.zeros((len(matched), len(parameter_names)))
    labels = []
    for row_index, (column_index, label) in enumerate(matched):
        constraint[row_index, column_index] = 1.0
        labels.append(label)
    return constraint, labels


def anova_wald_tests(result, alpha: float) -> pd.DataFrame:
    from scipy import stats

    fixed_names = list(result.fe_params.index)
    covariance = result.cov_params().loc[fixed_names, fixed_names]
    rows = []
    for factor in ANOVA_FACTOR_COLUMNS:
        _, labels = anova_factor_constraints(result, factor)
        beta = result.fe_params.loc[labels].to_numpy(dtype=float)
        cov = covariance.loc[labels, labels].to_numpy(dtype=float)
        statistic = float(beta.T @ np.linalg.pinv(cov) @ beta)
        p_value = float(stats.chi2.sf(statistic, df=len(labels)))
        rows.append(
            {
                "factor": factor,
                "df": len(labels),
                "chi2": statistic,
                "pvalue": p_value,
                "significant": p_value < alpha,
            }
        )
    return pd.DataFrame(rows)


def anova_marginal_means(
    result, dataframe: pd.DataFrame, factor: str
) -> pd.DataFrame:
    rows = []
    for level in dataframe[factor].cat.categories:
        scenario = dataframe.copy()
        scenario[factor] = level
        predictions = result.predict(exog=scenario)
        rows.append(
            {
                "factor": factor,
                "level": ANOVA_METHOD_LABELS.get(str(level), str(level)),
                "mean_prediction": float(np.mean(predictions)),
            }
        )
    return pd.DataFrame(rows)


def anova_pairwise_method_comparisons(
    result, alpha: float
) -> pd.DataFrame:
    from scipy import stats
    from statsmodels.stats.multitest import multipletests

    fixed_names = list(result.fe_params.index)
    covariance = result.cov_params().loc[
        fixed_names, fixed_names
    ].to_numpy(dtype=float)
    parameters = result.fe_params.to_numpy(dtype=float)
    index_by_name = {
        name: index for index, name in enumerate(fixed_names)
    }

    def method_vector(method: str) -> np.ndarray:
        vector = np.zeros(len(fixed_names), dtype=float)
        if method != ANOVA_METHOD_ORDER[0]:
            vector[index_by_name[f"C(method)[T.{method}]"]] = 1.0
        return vector

    rows = []
    for left_index, left in enumerate(ANOVA_METHOD_ORDER):
        for right in ANOVA_METHOD_ORDER[left_index + 1 :]:
            contrast = method_vector(left) - method_vector(right)
            difference = float(contrast @ parameters)
            variance = float(contrast @ covariance @ contrast)
            standard_error = float(np.sqrt(max(variance, 0.0)))
            z_statistic = difference / standard_error
            p_value = float(2.0 * stats.norm.sf(abs(z_statistic)))
            margin = float(stats.norm.ppf(0.975) * standard_error)
            rows.append(
                {
                    "left": ANOVA_METHOD_LABELS[left],
                    "right": ANOVA_METHOD_LABELS[right],
                    "mean_diff": difference,
                    "z_stat": z_statistic,
                    "raw_pvalue": p_value,
                    "ci_low": difference - margin,
                    "ci_high": difference + margin,
                }
            )

    pairwise = pd.DataFrame(rows)
    reject, adjusted, _, _ = multipletests(
        pairwise["raw_pvalue"], alpha=alpha, method="holm"
    )
    pairwise["adjusted_pvalue"] = adjusted
    pairwise["significant"] = reject
    return pairwise.sort_values(
        ["adjusted_pvalue", "left", "right"]
    ).reset_index(drop=True)


def anova_dataset_summary(dataframe: pd.DataFrame) -> str:
    def format_series(series: pd.Series) -> str:
        return "\n".join(
            f"  {index}: {value}" for index, value in series.items()
        )

    return "\n".join(
        (
            "Dataset summary",
            f"  Rows: {len(dataframe)}",
            f"  Instances: {dataframe['instance'].nunique()}",
            "  Methods: "
            + ", ".join(
                ANOVA_METHOD_LABELS[method]
                for method in ANOVA_METHOD_ORDER
            ),
            "  Rows per method:",
            format_series(
                dataframe["method"]
                .value_counts()
                .sort_index()
                .rename(index=ANOVA_METHOD_LABELS)
            ),
            "  Levels per factor:",
            format_series(
                pd.Series(
                    {
                        factor: dataframe[factor].nunique()
                        for factor in ANOVA_FACTOR_COLUMNS[1:]
                    }
                )
            ),
        )
    )


def write_anova_report(
    output_path: Path,
    response_label: str,
    dataframe: pd.DataFrame,
    result,
    fixed_effect_tests: pd.DataFrame,
    marginal_tables: dict[str, pd.DataFrame],
    pairwise: pd.DataFrame,
) -> Path:
    fitted = np.asarray(result.predict(exog=dataframe), dtype=float)
    residuals = np.asarray(dataframe["response"], dtype=float) - fitted
    residual_summary = pd.Series(residuals).describe(
        percentiles=(0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
    )
    lines = [
        f"Response: {response_label}",
        "",
        anova_dataset_summary(dataframe),
        "",
        "Omnibus fixed-effect tests",
        fixed_effect_tests.to_string(index=False),
        "",
        "Residual summary",
        residual_summary.to_string(),
    ]
    for factor, table in marginal_tables.items():
        lines.extend(
            (
                "",
                f"Estimated marginal means: {factor}",
                table.to_string(index=False),
            )
        )
    lines.extend(
        (
            "",
            "Pairwise method comparisons",
            pairwise.to_string(index=False),
        )
    )
    output_path = validate_repo_path(output_path, "ANOVA report")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def generate_anova_reports(
    methods: dict[str, list[SingleRun]], alpha: float = 0.05
) -> tuple[Path, Path]:
    """Fit raw/log mixed models and write the reports consumed by Table S13."""
    dataframe = build_anova_dataframe(methods)
    validate_anova_repeated_measures(dataframe)
    outputs = []
    for log_response, output_path in (
        (False, RAW_MODEL_REPORT),
        (True, LOG_MODEL_REPORT),
    ):
        analysis, response_label = add_anova_response(
            dataframe, log_response
        )
        result = fit_anova_mixed_model(analysis)
        fixed_effect_tests = anova_wald_tests(result, alpha)
        marginal_tables = {
            factor: anova_marginal_means(result, analysis, factor)
            for factor in ANOVA_FACTOR_COLUMNS
        }
        method_p_value = float(
            fixed_effect_tests.loc[
                fixed_effect_tests["factor"] == "method", "pvalue"
            ].iloc[0]
        )
        if method_p_value >= alpha:
            raise ValueError(
                "The omnibus method effect is not significant; method "
                "contrasts cannot be generated"
            )
        pairwise = anova_pairwise_method_comparisons(result, alpha)
        outputs.append(
            write_anova_report(
                output_path,
                response_label,
                analysis,
                result,
                fixed_effect_tests,
                marginal_tables,
                pairwise,
            )
        )
    return outputs[0], outputs[1]


def load_model_report(source: Path) -> ModelReport:
    fixed_effects: dict[str, tuple[int, float, float]] = {}
    marginal_means: dict[str, float] = {}
    pairwise: dict[tuple[str, str], PairwiseResult] = {}
    section = ""
    method_pattern = r"(LNS \(\d+s\)|VNS\([lmh]\))"
    pair_pattern = re.compile(
        rf"^\s*{method_pattern}\s+{method_pattern}\s+(?P<values>.+)$"
    )
    for raw_line in source.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line == "Omnibus fixed-effect tests":
            section = "fixed"
            continue
        if line == "Residual summary":
            section = ""
            continue
        if line == "Estimated marginal means: method":
            section = "marginal"
            continue
        if line.startswith("Estimated marginal means:"):
            section = ""
            continue
        if line == "Pairwise method comparisons":
            section = "pairwise"
            continue
        if line == "Diagnostic files":
            section = ""
            continue

        if section == "fixed":
            tokens = line.split()
            if tokens and tokens[0] in {
                "method",
                "size",
                "demand",
                "workload",
                "customers",
                "type",
            }:
                fixed_effects[tokens[0]] = (
                    int(tokens[1]),
                    float(tokens[2]),
                    float(tokens[3]),
                )
        elif section == "marginal":
            tokens = line.split()
            if tokens and tokens[0] == "method":
                marginal_means[" ".join(tokens[1:-1])] = float(tokens[-1])
        elif section == "pairwise":
            match = pair_pattern.fullmatch(raw_line)
            if match:
                values = match.group("values").split()
                if len(values) != 7:
                    raise ValueError(f"{source}: malformed pairwise result")
                pairwise[match.group(1), match.group(2)] = PairwiseResult(
                    difference=float(values[0]),
                    ci_low=float(values[3]),
                    ci_high=float(values[4]),
                    adjusted_p_value=float(values[5]),
                )

    if len(fixed_effects) != 6:
        raise ValueError(f"{source}: expected six fixed-effect tests")
    if len(marginal_means) != 6:
        raise ValueError(f"{source}: expected six method marginal means")
    if len(pairwise) != 15:
        raise ValueError(f"{source}: expected 15 pairwise method comparisons")
    return ModelReport(fixed_effects, marginal_means, pairwise)


def format_table_p_value(value: float) -> str:
    return "$<0.001$" if value < 0.001 else f"{value:.3f}"


def render_mixed_effects_table(
    raw: ModelReport,
    log: ModelReport,
) -> str:
    factors = (
        ("method", "Method"),
        ("size", "Number of BUs"),
        ("type", "Spatial layout"),
        ("demand", "Demand"),
        ("workload", "Workload"),
        ("customers", "Customers"),
    )
    fixed_rows = []
    for key, label in factors:
        raw_df, raw_chi2, raw_p = raw.fixed_effects[key]
        log_df, log_chi2, log_p = log.fixed_effects[key]
        if raw_df != log_df:
            raise ValueError(f"{key}: inconsistent model degrees of freedom")
        fixed_rows.append(
            f"{label} & {raw_df} & {raw_chi2:.3f} & "
            f"{format_table_p_value(raw_p)} & {log_chi2:.3f} & "
            f"{format_table_p_value(log_p)} \\\\"
        )

    raw_ranks = {
        method: rank
        for rank, (method, _) in enumerate(
            sorted(raw.marginal_means.items(), key=lambda item: item[1]), start=1
        )
    }
    log_ranks = {
        method: rank
        for rank, (method, _) in enumerate(
            sorted(log.marginal_means.items(), key=lambda item: item[1]), start=1
        )
    }
    marginal_rows = []
    for method, display in MODEL_METHODS:
        marginal_rows.append(
            f"{display} & {raw.marginal_means[method]:.3f} & "
            f"{raw_ranks[method]} & {log.marginal_means[method]:.4f} & "
            f"{log_ranks[method]} \\\\"
        )

    display_labels = dict(MODEL_METHODS)
    contrast_rows = []
    method_keys = tuple(key for key, _ in MODEL_METHODS)
    for left, right in combinations(method_keys, 2):
        raw_result = raw.pairwise[left, right]
        log_result = log.pairwise[left, right]
        contrast_rows.append(
            f"{display_labels[left]} $-$ {display_labels[right]} & "
            f"{raw_result.difference:.3f} & "
            f"[{raw_result.ci_low:.3f}, {raw_result.ci_high:.3f}] & "
            f"{format_table_p_value(raw_result.adjusted_p_value)} & "
            f"{log_result.difference:.4f} & "
            f"[{log_result.ci_low:.4f}, {log_result.ci_high:.4f}] & "
            f"{format_table_p_value(log_result.adjusted_p_value)} \\\\"
        )

    fixed_body = "\n".join(fixed_rows)
    marginal_body = "\n".join(marginal_rows)
    contrast_body = "\n".join(contrast_rows)
    return f"""\\begingroup
\\begin{{table}}[p]
\\centering
\\caption{{Mixed-effects analysis of the compactness metric from the 2,430 instances and six methods. Models using raw and log-transformed compactness are evaluated. These models include the instance identifier as a random effect as well as fixed effects for method, number of BUs, spatial layout, and the shares of low-valued demand, workload, and customer requirements. Panel A provides omnibus Wald tests for both models. Panel B1 gives estimated marginal means (EMMs) and ranks, where lower values and ranks are to be preferred. Panel B2 reports method contrasts with unadjusted 95\\% confidence intervals and Holm-adjusted $p$-values across the 15 method comparisons, separately on each response scale. Balance-infeasible outcomes are retained.}}
\\label{{tab:group3-mixed-effects}}
\\scriptsize
\\setlength{{\\tabcolsep}}{{3pt}}
\\begin{{minipage}}[t]{{0.57\\textwidth}}
\\centering
\\textit{{Panel A: Omnibus fixed-effect tests}}\\par\\smallskip
\\resizebox{{\\linewidth}}{{!}}{{%
\\begin{{tabular}}{{lrrrrr}}
\\toprule
& & \\multicolumn{{2}}{{c}}{{Raw compactness}} & \\multicolumn{{2}}{{c}}{{Log compactness}} \\\\
\\cmidrule(lr){{3-4}} \\cmidrule(lr){{5-6}}
Factor & df & Wald $\\chi^2$ & $p$ & Wald $\\chi^2$ & $p$ \\\\
\\midrule
{fixed_body}
\\bottomrule
\\end{{tabular}}
}}
\\end{{minipage}}\\hfill
\\begin{{minipage}}[t]{{0.40\\textwidth}}
\\centering
\\textit{{Panel B1: Method estimated marginal means}}\\par\\smallskip
\\resizebox{{\\linewidth}}{{!}}{{%
\\begin{{tabular}}{{lrrrr}}
\\toprule
Method & Raw EMM & Rank & Log EMM & Rank \\\\
\\midrule
{marginal_body}
\\bottomrule
\\end{{tabular}}
}}
\\end{{minipage}}

\\par\\medskip
\\textit{{Panel B2: Pairwise method contrasts}}\\par\\smallskip
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{lrrrrrr}}
\\toprule
& \\multicolumn{{3}}{{c}}{{Raw compactness}} & \\multicolumn{{3}}{{c}}{{Log compactness}} \\\\
\\cmidrule(lr){{2-4}} \\cmidrule(lr){{5-7}}
Comparison & Difference & 95\\% CI & Holm $p$ & Difference & 95\\% CI & Holm $p$ \\\\
\\midrule
{contrast_body}
\\bottomrule
\\end{{tabular}}
}}
\\end{{table}}
\\endgroup
"""


def render_lower_bound_gap_table(
    repeated_methods: dict[tuple[int, str], list[ablation.Run]],
    vns_runs: dict[str, list[lns_vns.VnsRun]],
    single_methods: dict[str, list[SingleRun]],
    single_lower_bounds: dict[str, float],
) -> str:
    def gap(upper_bound: float, lower_bound: float) -> float:
        if upper_bound <= 0.0:
            raise ValueError("compactness upper bounds must be positive")
        return 100.0 * (upper_bound - lower_bound) / upper_bound

    def describe_gaps(gaps: list[float]) -> tuple[float, ...]:
        if not gaps:
            raise ValueError("cannot summarize an empty gap distribution")
        if len(gaps) == 1:
            q1 = q3 = gaps[0]
            standard_deviation = 0.0
        else:
            q1, _, q3 = statistics.quantiles(
                gaps, n=4, method="inclusive"
            )
            standard_deviation = statistics.stdev(gaps)
        return (
            statistics.mean(gaps),
            standard_deviation,
            q1,
            statistics.median(gaps),
            q3,
        )

    def render_gap_cells(gaps: list[float]) -> str:
        return " & ".join(f"{value:.2f}" for value in describe_gaps(gaps))

    def repeated_instance_gaps(
        label: str,
        runs: list[ablation.Run] | list[lns_vns.VnsRun],
        lower_bounds: dict[str, float],
    ) -> tuple[list[float], list[float]]:
        runs_by_instance = lns_vns.group_by_instance(runs)
        if set(runs_by_instance) != set(lower_bounds):
            raise ValueError(f"{label}: run and lower-bound instance sets differ")

        all_gaps = []
        feasible_gaps = []
        for instance, instance_runs in runs_by_instance.items():
            all_upper_bound = statistics.median(
                run.objective for run in instance_runs
            )
            all_gaps.append(gap(all_upper_bound, lower_bounds[instance]))

            feasible_runs = [
                run for run in instance_runs if run.penalty < 1e-6
            ]
            if feasible_runs:
                feasible_upper_bound = statistics.median(
                    run.objective for run in feasible_runs
                )
                feasible_gaps.append(
                    gap(feasible_upper_bound, lower_bounds[instance])
                )
        return all_gaps, feasible_gaps

    repeated_lower_bounds = lns_vns.read_discrete_dispersion_lower_bounds()
    repeated_configurations = (
        ("LNS (60~s)", repeated_methods[60, "Full LNS"]),
        ("LNS (300~s)", repeated_methods[300, "Full LNS"]),
        ("LNS (600~s)", repeated_methods[600, "Full LNS"]),
        ("VNS(l)", vns_runs["test6WithSeed.py"]),
        ("VNS(m)", vns_runs["test7WithSeed.py"]),
        ("VNS(h)", vns_runs["test8WithSeed.py"]),
    )
    repeated_rows = []
    for label, runs in repeated_configurations:
        all_gaps, feasible_gaps = repeated_instance_gaps(
            label, runs, repeated_lower_bounds
        )
        repeated_rows.append(
            f"{label} & {render_gap_cells(all_gaps)} & "
            f"{len(feasible_gaps):,} & {render_gap_cells(feasible_gaps)} \\\\"
        )

    single_rows = []
    for method, label in GROUP3_METHODS:
        runs = single_methods[method]
        all_gaps = [
            gap(run.objective, single_lower_bounds[run.instance])
            for run in runs
        ]
        feasible_runs = [run for run in runs if run.penalty < 1e-6]
        feasible_gaps = [
            gap(run.objective, single_lower_bounds[run.instance])
            for run in feasible_runs
        ]
        single_rows.append(
            f"{label} & {render_gap_cells(all_gaps)} & "
            f"{len(feasible_runs):,} & {render_gap_cells(feasible_gaps)} \\\\"
        )

    repeated_body = "\n".join(repeated_rows)
    single_body = "\n".join(single_rows)
    return f"""\\begingroup
\\begin{{table}}[p]
\\centering
\\caption{{Comparison of the six method configurations with the discrete $p$-dispersion lower bound in both experiments. Panel A reports results calculated from the median compactness among the replicas for each of the 120 repeated-run instances, whereas Panel B reports results calculated from the actual compactness value obtained for each of the 2,430 single-run instances. Results are shown for all outcomes and for balance-feasible outcomes. For the balance-feasible summaries, \\textit{{Inst.}} is the number of eligible instances: repeated-run instances with at least one balance-feasible replica in Panel A, and single-run instances with a balance-feasible outcome in Panel B.}}
\\label{{tab:lower-bound-gaps-all-methods}}
\\small
\\setlength{{\\tabcolsep}}{{3pt}}
\\begin{{tabular}}{{lrrrrr@{{\\hspace{{0.25in}}}}rrrrrr}}
\\toprule
& \\multicolumn{{5}}{{c}}{{All outcomes}} & \\multicolumn{{6}}{{c}}{{Balance-feasible outcomes}} \\\\
\\cmidrule(lr){{2-6}} \\cmidrule(lr){{7-12}}
Method & Mean & $s$ & $Q_1$ & Median & $Q_3$ & Inst. & Mean & $s$ & $Q_1$ & Median & $Q_3$ \\\\
\\midrule
\\multicolumn{{12}}{{l}}{{\\textit{{Panel A: 120-instance repeated-run experiment}}}} \\\\
{repeated_body}
\\midrule
\\multicolumn{{12}}{{l}}{{\\textit{{Panel B: 2,430-instance single-run experiment}}}} \\\\
{single_body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
\\endgroup
"""


def best_known_feasible_summaries(
    methods: dict[tuple[int, str], list[ablation.Run]],
    vns_runs: dict[str, list[lns_vns.VnsRun]],
) -> dict[str, tuple[int, float | None]]:
    """Return the count and eligible-instance average shown in the table."""
    lns_labels = (
        "Full LNS",
        "Random destroy",
        "Alternative destroy",
        "District-based destroy",
        "Fixed destroy size",
        "No fixing",
        "No warm start",
        "Heuristic repair",
        "Only random start",
    )
    configurations: dict[
        str, list[ablation.Run] | list[lns_vns.VnsRun]
    ] = {
        f"{label}|{time_limit}": methods[time_limit, label]
        for label in lns_labels
        for time_limit in ablation.TIME_LIMITS
    }
    for _, label, version in lns_vns.COMPARISONS:
        configurations[f"VNS|{label}"] = vns_runs[version]

    feasible: dict[str, dict[str, list[float]]] = {}
    for key, runs in configurations.items():
        grouped: dict[str, list[float]] = defaultdict(list)
        for run in runs:
            if run.penalty == 0:
                grouped[run.instance].append(run.objective)
        feasible[key] = dict(grouped)

    instances = sorted(ablation.expected_instances())
    best_known = {
        instance: min(
            min(grouped[instance])
            for grouped in feasible.values()
            if instance in grouped
        )
        for instance in instances
    }
    summaries = {}
    for key, grouped in feasible.items():
        count = sum(
            instance in grouped
            and math.isclose(
                min(grouped[instance]),
                best_known[instance],
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            for instance in instances
        )
        average = None
        if all(instance in grouped for instance in instances):
            average = statistics.mean(
                statistics.mean(grouped[instance]) for instance in instances
            )
        summaries[key] = count, average
    return summaries


def repair_summary(
    methods: dict[tuple[int, str], list[ablation.Run]],
    time_limit: int,
    label: str,
) -> dict[str, float]:
    runs = methods[time_limit, label]
    repairs = sum(run.repairs for run in runs)
    candidates = sum(
        run.free_variables + run.fixed_variables for run in runs
    )
    return {
        "free": sum(run.free_variables for run in runs) / repairs,
        "fixed": 100.0
        * sum(run.fixed_variables for run in runs)
        / candidates,
        "repairs_per_run": statistics.mean(run.repairs for run in runs),
        "balance": 100.0
        * sum(run.balance_improvements for run in runs)
        / repairs,
        "compactness": 100.0
        * sum(run.compactness_improvements for run in runs)
        / repairs,
    }


def gap_distribution(values: list[float]) -> tuple[float, ...]:
    q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
    return (
        statistics.mean(values),
        statistics.stdev(values),
        q1,
        statistics.median(values),
        q3,
    )


def lower_bound_summaries(
    methods: dict[tuple[int, str], list[ablation.Run]],
    vns_runs: dict[str, list[lns_vns.VnsRun]],
    single_methods: dict[str, list[SingleRun]],
    single_lower_bounds: dict[str, float],
) -> tuple[
    dict[str, tuple[tuple[float, ...], tuple[float, ...]]],
    dict[str, tuple[tuple[float, ...], tuple[float, ...]]],
]:
    def gap(upper_bound: float, lower_bound: float) -> float:
        return 100.0 * (upper_bound - lower_bound) / upper_bound

    repeated_bounds = lns_vns.read_discrete_dispersion_lower_bounds()
    repeated_configurations = (
        ("LNS (60 s)", methods[60, "Full LNS"]),
        ("LNS (300 s)", methods[300, "Full LNS"]),
        ("LNS (600 s)", methods[600, "Full LNS"]),
        ("VNS(l)", vns_runs["test6WithSeed.py"]),
        ("VNS(m)", vns_runs["test7WithSeed.py"]),
        ("VNS(h)", vns_runs["test8WithSeed.py"]),
    )
    repeated = {}
    for label, runs in repeated_configurations:
        grouped = lns_vns.group_by_instance(runs)
        all_values = []
        feasible_values = []
        for instance, instance_runs in grouped.items():
            all_values.append(
                gap(
                    statistics.median(run.objective for run in instance_runs),
                    repeated_bounds[instance],
                )
            )
            feasible_runs = [
                run for run in instance_runs if run.penalty < 1e-6
            ]
            if feasible_runs:
                feasible_values.append(
                    gap(
                        statistics.median(
                            run.objective for run in feasible_runs
                        ),
                        repeated_bounds[instance],
                    )
                )
        repeated[label] = (
            (*gap_distribution(all_values), len(all_values)),
            (*gap_distribution(feasible_values), len(feasible_values)),
        )

    single = {}
    for method, label in GROUP3_METHODS:
        runs = single_methods[method]
        all_values = [
            gap(run.objective, single_lower_bounds[run.instance])
            for run in runs
        ]
        feasible_values = [
            gap(run.objective, single_lower_bounds[run.instance])
            for run in runs
            if run.penalty < 1e-6
        ]
        single[label.replace("~", "")] = (
            (*gap_distribution(all_values), len(all_values)),
            (*gap_distribution(feasible_values), len(feasible_values)),
        )
    return repeated, single


def attribute_requirement_summary(
    methods: dict[str, list[SingleRun]],
) -> tuple[dict[str, int], int, int]:
    by_method = {
        method: {run.instance: run for run in runs}
        for method, runs in methods.items()
    }
    instances = sorted(next(iter(by_method.values())))
    best = {
        instance: min(
            by_method[method][instance].objective
            for method, _ in GROUP3_METHODS
        )
        for instance in instances
    }
    grouped: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for instance in instances:
        match = GROUP3_NAME_PATTERN.fullmatch(instance)
        if match is None:
            raise ValueError(f"malformed group-3 instance name: {instance}")
        grouped[
            match.group("d"), match.group("w"), match.group("c")
        ].append(instance)

    leader_counts = {method: 0 for method, _ in GROUP3_METHODS}
    lns600_below_every_vns = 0
    lns600_attainment_leads_when_lns300_gap_leads = 0
    for group in grouped.values():
        means = {
            method: statistics.mean(
                100.0
                * (by_method[method][instance].objective - best[instance])
                / best[instance]
                for instance in group
            )
            for method, _ in GROUP3_METHODS
        }
        minimum = min(means.values())
        for method, value in means.items():
            if math.isclose(
                value, minimum, rel_tol=1e-12, abs_tol=1e-12
            ):
                leader_counts[method] += 1
        if all(
            means["LNS600"] < means[method]
            for method in ("VNSl", "VNSm", "VNSh")
        ):
            lns600_below_every_vns += 1
        attainments = {
            method: sum(
                math.isclose(
                    by_method[method][instance].objective,
                    best[instance],
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                )
                for instance in group
            )
            for method, _ in GROUP3_METHODS
        }
        if (
            means["LNS300"] < means["LNS600"]
            and attainments["LNS600"] == max(attainments.values())
        ):
            lns600_attainment_leads_when_lns300_gap_leads += 1
    return (
        leader_counts,
        lns600_below_every_vns,
        lns600_attainment_leads_when_lns300_gap_leads,
    )


def render_online_supplement_numbers(
    methods: dict[tuple[int, str], list[ablation.Run]],
    vns_runs: dict[str, list[lns_vns.VnsRun]],
    single_methods: dict[str, list[SingleRun]],
    single_lower_bounds: dict[str, float],
    raw_model: ModelReport,
    log_model: ModelReport,
    destroy_comparisons: list[ablation.FixedComparison],
    full_600_comparisons: list[Full600Comparison],
) -> str:
    """Render every numerical result cited in the supplement prose."""
    def feasible_counts(time_limit: int, label: str) -> tuple[int, float, float]:
        grouped: dict[str, list[ablation.Run]] = defaultdict(list)
        for run in methods[time_limit, label]:
            grouped[run.instance].append(run)
        counts = [
            sum(run.penalty == 0 for run in instance_runs)
            for instance_runs in grouped.values()
        ]
        return min(counts), statistics.mean(counts), statistics.stdev(counts)

    def mean_rcd(time_limit: int, label: str) -> float:
        return statistics.mean(
            ablation.instance_objective_gaps(methods, time_limit, label)
        )

    def outcome_counts(time_limit: int, label: str) -> tuple[int, int, int]:
        gaps = ablation.instance_objective_gaps(methods, time_limit, label)
        return (
            sum(value > 0 for value in gaps),
            sum(value == 0 for value in gaps),
            sum(value < 0 for value in gaps),
        )

    best_known = best_known_feasible_summaries(methods, vns_runs)
    repair = {
        label: repair_summary(methods, 600, label)
        for label in (
            "Full LNS",
            "No fixing",
            "Random destroy",
            "District-based destroy",
        )
    }
    runtimes = {
        summary.method: summary
        for summary in lns_vns.runtime_summaries(methods, vns_runs)
    }
    repeated_bounds, single_bounds = lower_bound_summaries(
        methods, vns_runs, single_methods, single_lower_bounds
    )
    (
        leaders,
        lns600_below_vns,
        lns600_attainment_leads,
    ) = attribute_requirement_summary(single_methods)

    random_min_60, _, _ = feasible_counts(60, "Random destroy")
    random_min_300, _, _ = feasible_counts(300, "Random destroy")
    full_min_60, full_mean_60, _ = feasible_counts(60, "Full LNS")
    full_min_300, full_mean_300, _ = feasible_counts(300, "Full LNS")
    _, fixed_mean_60, _ = feasible_counts(60, "Fixed destroy size")
    _, fixed_mean_300, _ = feasible_counts(300, "Fixed destroy size")
    destructive_repair_rcds = [
        mean_rcd(600, label)
        for label in (
            "Random destroy",
            "District-based destroy",
            "Heuristic repair",
        )
    ]

    lines = [
        "[repeated-run experiment design]",
        "instances = 120",
        "ablated LNS variants = 8",
        "LNS configurations = 27",
        "VNS configurations = 3",
        "total configurations = 30",
        "replicas per method and instance = 30",
        "maximum within-instance run pairs = 900",
        "Full-LNS-600 comparison metric definitions = 5",
        "score win/tie/loss weights = 1/0.5/0",
        "score neutral value = 0.5",
        "pairwise percentage-gap multiplier = 100",
        "",
        "[ablation summaries cited in prose]",
        f"600 s mean RCD: distance-based destroy = "
        f"{mean_rcd(600, 'Alternative destroy'):.1f}%; "
        f"fixed destroy size = {mean_rcd(600, 'Fixed destroy size'):.1f}%; "
        f"no fixing = {mean_rcd(600, 'No fixing'):.1f}%; "
        f"only random start = {mean_rcd(600, 'Only random start'):.1f}%",
        f"no warm start mean RCD at 60 s = "
        f"{mean_rcd(60, 'No warm start'):.1f}%",
        f"600 s random/district/heuristic mean-RCD range = "
        f"{min(destructive_repair_rcds):.1f}--"
        f"{max(destructive_repair_rcds):.1f}%",
        f"random destroy minimum feasible runs at 60/300 s = "
        f"{random_min_60}/{random_min_300}",
        f"Full LNS minimum feasible runs at 60/300 s = "
        f"{full_min_60}/{full_min_300}",
        f"random destroy mean RCD at 60/300/600 s = "
        f"{mean_rcd(60, 'Random destroy'):.1f}%/"
        f"{mean_rcd(300, 'Random destroy'):.1f}%/"
        f"{mean_rcd(600, 'Random destroy'):.1f}%",
        f"fixed destroy mean feasible runs at 60/300 s = "
        f"{fixed_mean_60:.1f}/{fixed_mean_300:.1f}",
        f"Full LNS mean feasible runs at 60/300 s = "
        f"{full_mean_60:.1f}/{full_mean_300:.1f}",
        "",
        "[best-known balance-feasible compactness]",
    ]
    for key, display in (
        ("Full LNS|60", "Full LNS 60 s"),
        ("Full LNS|300", "Full LNS 300 s"),
        ("Full LNS|600", "Full LNS 600 s"),
        ("Only random start|600", "Only random start 600 s"),
        ("No warm start|600", "No warm start 600 s"),
        ("VNS|VNS(h)", "VNS(h)"),
    ):
        count, average = best_known[key]
        average_text = "--" if average is None else f"{average:.1f}"
        lines.append(
            f"{display}: best-known instances = {count}; "
            f"eligible-instance average compactness = {average_text}"
        )

    lines.extend(
        [
            "",
            "[repair diagnostics at 600 s]",
            f"Full LNS: fixed variables = "
            f"{repair['Full LNS']['fixed']:.1f}%; "
            f"free variables per repair = "
            f"{repair['Full LNS']['free']:,.1f}; repairs per run = "
            f"{repair['Full LNS']['repairs_per_run']:.1f}",
            f"No fixing: free variables per repair = "
            f"{repair['No fixing']['free']:,.1f}; repairs per run = "
            f"{repair['No fixing']['repairs_per_run']:.1f}",
            f"compactness-improvement rate random destroy/Full LNS = "
            f"{repair['Random destroy']['compactness']:.1f}%/"
            f"{repair['Full LNS']['compactness']:.1f}%",
            f"balance-improvement rate district destroy/Full LNS = "
            f"{repair['District-based destroy']['balance']:.1f}%/"
            f"{repair['Full LNS']['balance']:.1f}%",
            f"compactness-improvement rate district destroy/Full LNS = "
            f"{repair['District-based destroy']['compactness']:.1f}%/"
            f"{repair['Full LNS']['compactness']:.1f}%",
            "",
            "[variable versus fixed destroy size]",
        ]
    )
    for result in destroy_comparisons:
        lines.append(
            f"{result.time_limit} s: Full/fixed feasibility = "
            f"{result.increasing_feasibility:.1f}%/"
            f"{result.fixed_feasibility:.1f}%; advantage = "
            f"{result.increasing_feasibility - result.fixed_feasibility:.1f} pp; "
            f"Full/tie/fixed = {result.increasing_wins:,}/"
            f"{result.ties:,}/{result.fixed_wins:,}; feasible pairs = "
            f"{result.feasible_pairs:,}; median gap = "
            f"{result.median_gap:.1f}%"
        )

    lines.extend(["", "[comparisons with Full LNS 600 s]"])
    for result in full_600_comparisons:
        lines.append(
            f"{result.method}: feasibility change = "
            f"{result.feasibility_change:.1f} pp; "
            f"median compactness change = "
            f"{result.median_compactness_change:.1f}; score = "
            f"{result.full_score:.3f}; feasible pairs = "
            f"{result.feasible_pairs:,}; median/mean gap = "
            f"{result.median_gap:.1f}%/{result.mean_gap:.1f}%"
        )
    vns_comparisons = full_600_comparisons[2:]
    lines.extend(
        [
            "VNS comparison ranges: feasibility change = "
            f"{min(value.feasibility_change for value in vns_comparisons):.1f}--"
            f"{max(value.feasibility_change for value in vns_comparisons):.1f} pp; "
            "compactness change = "
            f"{min(value.median_compactness_change for value in vns_comparisons):.1f}--"
            f"{max(value.median_compactness_change for value in vns_comparisons):.1f}; "
            "score = "
            f"{min(value.full_score for value in vns_comparisons):.3f}--"
            f"{max(value.full_score for value in vns_comparisons):.3f}; "
            "median gap = "
            f"{min(value.median_gap for value in vns_comparisons):.1f}--"
            f"{max(value.median_gap for value in vns_comparisons):.1f}%; "
            "mean gap = "
            f"{min(value.mean_gap for value in vns_comparisons):.1f}--"
            f"{max(value.mean_gap for value in vns_comparisons):.1f}%",
            "",
            "[running times over 3,600 runs]",
        ]
    )
    for method in (
        "VNS(l)",
        "LNS 60 s",
        "VNS(m)",
        "LNS 300 s",
        "VNS(h)",
        "LNS 600 s",
    ):
        summary = runtimes[method]
        lines.append(
            f"{method}: mean/Q1/median/Q3/s = "
            f"{summary.average:,.1f}/{summary.q1:,.1f}/"
            f"{summary.median:,.1f}/{summary.q3:,.1f}/"
            f"{summary.standard_deviation:,.1f} s; runs = "
            f"{summary.runs:,}"
        )

    lines.extend(
        [
            "",
            "[per-instance median compactness outcomes]",
            "reported LNS variants = 9",
            "reported time limits = 3",
        ]
    )
    for time_limit, label in (
        (600, "Random destroy"),
        (600, "District-based destroy"),
        (600, "Heuristic repair"),
        (300, "Random destroy"),
        (300, "District-based destroy"),
        (300, "Heuristic repair"),
        (600, "Only random start"),
        (600, "No warm start"),
        (600, "Alternative destroy"),
        (60, "No warm start"),
        (600, "Fixed destroy size"),
        (600, "No fixing"),
    ):
        worse, ties, better = outcome_counts(time_limit, label)
        lines.append(
            f"{time_limit} s {display_ablation_label(label)}: "
            f"worse/tie/better than Full LNS = {worse}/{ties}/{better}"
        )

    lines.extend(
        [
            "",
            "[2,430-instance attribute experiment]",
            "methods = 6",
            "runs per method and instance = 1",
            "compactness outcomes = 14,580",
            "attribute combinations = 27",
            "instances per combination = 90",
            f"group-average leaders LNS 600 s/LNS 300 s = "
            f"{leaders['LNS600']}/{leaders['LNS300']}",
            f"groups where LNS 600 s is below all three VNS averages = "
            f"{lns600_below_vns}",
            f"LNS 600 s best-count leads in groups led by LNS 300 s gap = "
            f"{lns600_attainment_leads}",
            "",
            "[mixed-effects results]",
            "method pairwise comparisons = 15",
            "confidence level = 95%",
            "method ranks best-to-worst = 1--6; VNS ranks = 3--5",
            "raw p-values method/size/layout/demand/workload/customers = "
            + "/".join(
                format_table_p_value(raw_model.fixed_effects[key][2])
                .replace("$", "")
                for key in (
                    "method",
                    "size",
                    "type",
                    "demand",
                    "workload",
                    "customers",
                )
            ),
            "log p-values method/size/layout/demand/workload/customers = "
            + "/".join(
                format_table_p_value(log_model.fixed_effects[key][2])
                .replace("$", "")
                for key in (
                    "method",
                    "size",
                    "type",
                    "demand",
                    "workload",
                    "customers",
                )
            ),
            "raw method ranking = "
            + " > ".join(
                method
                for method, _ in sorted(
                    raw_model.marginal_means.items(),
                    key=lambda item: item[1],
                )
            ),
            "log method ranking = "
            + " > ".join(
                method
                for method, _ in sorted(
                    log_model.marginal_means.items(),
                    key=lambda item: item[1],
                )
            ),
            "",
            "[discrete p-dispersion lower-bound gaps]",
            "percentage-gap multiplier = 100",
            "repeated outcomes summarized per instance over 30 runs",
        ]
    )

    def append_bound_rows(
        experiment: str,
        summaries: dict[
            str, tuple[tuple[float, ...], tuple[float, ...]]
        ],
    ) -> None:
        lines.append(experiment)
        for method, (all_values, feasible_values) in summaries.items():
            lines.append(
                f"{method}: all mean/s/Q1/median/Q3 = "
                + "/".join(f"{value:.2f}%" for value in all_values[:5])
                + f"; instances = {int(all_values[5]):,}; feasible "
                "mean/s/Q1/median/Q3 = "
                + "/".join(
                    f"{value:.2f}%" for value in feasible_values[:5]
                )
                + f"; eligible instances = {int(feasible_values[5]):,}"
            )

    append_bound_rows(
        "120-instance repeated-run experiment", repeated_bounds
    )
    append_bound_rows(
        "2,430-instance single-run experiment", single_bounds
    )
    all_reported_gaps = [
        value
        for summaries in (repeated_bounds, single_bounds)
        for all_values, feasible_values in summaries.values()
        for values in (all_values, feasible_values)
        for value in (values[0], values[3])
    ]
    lines.append(
        f"reported mean/median gap span = "
        f"{min(all_reported_gaps):.0f}--{max(all_reported_gaps):.0f}%"
    )
    return "\n".join(lines) + "\n"


def write_output(output_path: Path, rendered: str) -> Path:
    output_path = validate_repo_path(output_path, "Output")
    output_path.write_text(rendered, encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate exactly the tables and numerical text used in the "
            "online supplement."
        )
    )
    parser.add_argument(
        "--tables",
        action="store_true",
        help="Generate all eleven data-derived LaTeX tables in the supplement.",
    )
    parser.add_argument(
        "--text",
        action="store_true",
        help="Generate the numerical results cited in the supplement text.",
    )
    args = parser.parse_args()
    if not (args.tables or args.text):
        parser.error("select --tables, --text, or both")
    return args


def main() -> None:
    args = parse_args()
    configure_local_inputs()
    methods = ablation.load_method_runs()
    vns_runs = lns_vns.read_vns_runs()
    single_methods, single_lower_bounds = load_group3_single_run_results()
    for report_path in generate_anova_reports(single_methods):
        print(report_path)
    raw_model = load_model_report(RAW_MODEL_REPORT)
    log_model = load_model_report(LOG_MODEL_REPORT)
    destroy_comparisons = compare_destroy_size_schedules(methods)
    full_600_comparisons = compare_with_full_lns_600(methods, vns_runs)
    if args.tables:
        table_outputs = {
            VARIANT_MAPPING_OUTPUT: render_variant_mapping_table(),
            MEAN_STANDARD_ABLATION_OUTPUT:
                render_mean_standard_ablation_table(methods),
            BEST_KNOWN_FEASIBLE_OUTPUT: render_best_known_feasible_table(
                methods, vns_runs
            ),
            REPAIR_DIAGNOSTICS_OUTPUT: render_repair_diagnostics_table(methods),
            DESTROY_SIZE_SCHEDULE_OUTPUT: render_destroy_size_schedule_table(
                destroy_comparisons
            ),
            FULL_LNS_TIME_COMPARISON_OUTPUT:
                render_full_lns_time_comparison_table(full_600_comparisons),
            LNS_VNS_RUNTIME_OUTPUT: render_lns_vns_runtime_table(
                methods, vns_runs
            ),
            PER_INSTANCE_ABLATION_OUTPUT:
                render_per_instance_ablation_table(methods),
            ATTRIBUTE_COMPACTNESS_OUTPUT: render_attribute_compactness_table(
                single_methods
            ),
            MIXED_EFFECTS_OUTPUT: render_mixed_effects_table(
                raw_model, log_model
            ),
            LOWER_BOUND_GAPS_OUTPUT: render_lower_bound_gap_table(
                methods,
                vns_runs,
                single_methods,
                single_lower_bounds,
            ),
        }
        for output_path, rendered in table_outputs.items():
            generated = "% Auto-generated by scripts/doSupplementTables.py\n" + rendered
            print(write_output(output_path, generated))

    if args.text:
        rendered = render_online_supplement_numbers(
            methods,
            vns_runs,
            single_methods,
            single_lower_bounds,
            raw_model,
            log_model,
            destroy_comparisons,
            full_600_comparisons,
        )
        print(write_output(NUMERICAL_RESULTS_OUTPUT, rendered))


if __name__ == "__main__":
    main()
