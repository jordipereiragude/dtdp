"""Print the supplementary tables in Markdown.

Input CSV paths are resolved relative to this file, so the command is
independent of the caller's working directory.
"""

from itertools import combinations

import numpy as np
import pandas as pd
from patsy import build_design_matrices, dmatrix
from statsmodels.regression.mixed_linear_model import MixedLM

if __package__:
    from .doTables import (
        ATTRIBUTE_LEVELS, SINGLE_RESULT_METHODS, TIME_LIMITS, main,
        _holm_adjust, _identify_ablation_variants, _prepare_ablation_runs, _prepare_lns_vns_samples,
        _prepare_single_compactness, _prepare_single_results, _prepare_vns_runs, _table_output, _validate_run_grid,
    )
else:
    from doTables import (
        ATTRIBUTE_LEVELS, SINGLE_RESULT_METHODS, TIME_LIMITS, main,
        _holm_adjust, _identify_ablation_variants, _prepare_ablation_runs, _prepare_lns_vns_samples,
        _prepare_single_compactness, _prepare_single_results, _prepare_vns_runs, _table_output, _validate_run_grid,
    )


S5_METHODS = (
    "Full",
    "Random destroy",
    "Distance-based destroy",
    "District-based destroy",
    "Fixed destroy size",
    "No fixing",
    "No warm start",
    "Heuristic repair",
    "Only random start",
)
S6_METHODS = (*S5_METHODS, "VNS")
S7_METHODS = tuple(method for method in S5_METHODS if method != "Heuristic repair")
S9_METHODS = (
    ("LNS60", "Full LNS (60 s)"),
    ("LNS300", "Full LNS (300 s)"),
    ("VNS(l)", "VNS (l)"),
    ("VNS(m)", "VNS (m)"),
    ("VNS(h)", "VNS (h)"),
)
S13_METHODS = (
    "LNS (60 s)", "LNS (300 s)", "LNS (600 s)", "VNS(l)", "VNS(m)", "VNS(h)",
)
S13_FACTORS = (
    ("Method", "Method"), ("BUs", "Number of BUs"), ("Layout", "Spatial layout"),
    ("Demand", "Demand"), ("Workload", "Workload"), ("Customer", "Customers"),
)
S13_FORMULA = " + ".join(f"C({factor})" for factor, _ in S13_FACTORS)
S14_METHODS = (
    ("LNS60", "LNS (60 s)"), ("LNS300", "LNS (300 s)"), ("LNS600", "LNS (600 s)"),
    ("VNS(l)", "VNS(l)"), ("VNS(m)", "VNS(m)"), ("VNS(h)", "VNS(h)"),
)
S14_BOUND_COLUMNS = (
    "dispersion", "lbKnapsack", "lpBase", "lpPairwiseLB", "ipBaseLB", "ipPairwiseLB",
)


def _table_s5_statistics(lns_runs, lns_ablation_runs, lns_fixed_runs):
    """Summarize feasible-run counts and feasible-median RCDs across instances."""
    runs, _ = _prepare_ablation_runs(lns_runs, lns_ablation_runs, lns_fixed_runs)
    keys = ["variant", "tLim", "Instance"]
    feasible = runs["penalty"].eq(0)
    counts = runs.assign(feasible=feasible).groupby(keys)["feasible"].sum()
    medians = runs.loc[feasible].groupby(keys)["obj"].median().reindex(counts.index)

    summaries = []
    for method in S5_METHODS:
        for limit in TIME_LIMITS:
            feasible_counts = counts.loc[(method, limit)]
            paired = pd.concat(
                {"full": medians.loc[("Full", limit)], "method": medians.loc[(method, limit)]},
                axis=1,
            ).dropna()
            if paired["full"].eq(0).any():
                raise ValueError(f"{method}/{limit}: RCD is undefined because Full's feasible median is zero")
            rcd = 100 * (paired["method"] - paired["full"]) / paired["full"]
            summaries.append({
                "method": method,
                "tLim": limit,
                "feasible_min": int(feasible_counts.min()),
                "feasible_mean": float(feasible_counts.mean()),
                "feasible_s": float(feasible_counts.std(ddof=1)),
                "rcd_mean": float(rcd.mean()),
                "rcd_s": float(rcd.std(ddof=1)),
                "eligible_instances": len(rcd),
            })
    return pd.DataFrame(summaries).set_index(["method", "tLim"])


def doTableS5(lns_runs, lns_ablation_runs, lns_fixed_runs):
    """Return Table S5 in Markdown with five statistics per time limit.

    Feasible runs: minimum, mean and sample standard deviation of the 120
    instance-level counts (0--30), including zero counts. RCD: mean and sample
    standard deviation of 100 * (UB_method - UB_Full) / UB_Full, where each UB
    is the median compactness among that method's feasible runs at the same
    time limit. Include only instances with at least one feasible run from both
    methods. Full is its own RCD reference. Undefined summaries appear as dashes.
    """
    results = _table_s5_statistics(lns_runs, lns_ablation_runs, lns_fixed_runs)
    statistics = (
        ("feasible_min", "Feasible runs Min."),
        ("feasible_mean", "Feasible runs Mean"),
        ("feasible_s", "Feasible runs s"),
        ("rcd_mean", "RCD Mean (%)"),
        ("rcd_s", "RCD s (%)"),
    )
    # Explicit time prefixes preserve the column groups in standard Markdown.
    headers = ["Method", *(f"{limit} s {label}" for limit in TIME_LIMITS for _, label in statistics)]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| :--- | " + " | ".join(["---:"] * 15) + " |",
    ]
    for method in S5_METHODS:
        cells = ["Full LNS" if method == "Full" else method]
        for limit in TIME_LIMITS:
            result = results.loc[(method, limit)]
            for key, _ in statistics:
                value = result[key]
                cells.append(
                    "—" if pd.isna(value) else
                    str(int(value)) if key == "feasible_min" else f"{value:.1f}"
                )
        lines.append("| " + " | ".join(cells) + " |")
    return _table_output("S5", lines)


def _table_s6_statistics(
    lns_runs, lns_ablation_runs, lns_fixed_runs, vns_low_runs, vns_medium_runs, vns_high_runs
):
    """Count globally best feasible attainments and equally weight instance means."""
    lns, instances = _prepare_ablation_runs(lns_runs, lns_ablation_runs, lns_fixed_runs)
    columns = ["variant", "tLim", "Instance", "seed", "penalty", "obj"]
    sources = [lns[columns]]
    expected_vns = pd.MultiIndex.from_product(
        [instances, range(1, 31)], names=["Instance", "seed"]
    )
    for limit, version, dataframe in (
        (60, "l", vns_low_runs),
        (300, "m", vns_medium_runs),
        (600, "h", vns_high_runs),
    ):
        vns = _prepare_vns_runs(dataframe, expected_vns, f"VNS({version})")
        # tLim denotes the display column for VNS, not a VNS time limit.
        sources.append(vns.assign(variant="VNS", tLim=limit)[columns])
    runs = pd.concat(sources, ignore_index=True)
    feasible = runs.loc[runs["penalty"].eq(0)]
    best = feasible.groupby("Instance")["obj"].min().reindex(instances)
    expected = pd.MultiIndex.from_product(
        [S6_METHODS, TIME_LIMITS, instances], names=["variant", "tLim", "Instance"]
    )
    instance_results = (
        feasible.groupby(["variant", "tLim", "Instance"])["obj"]
        .agg(["min", "mean"]).reindex(expected).sort_index()
    )
    summaries = []
    for method in S6_METHODS:
        for limit in TIME_LIMITS:
            outcomes = instance_results.loc[(method, limit)]
            eligible = int(outcomes["mean"].notna().sum())
            summaries.append({
                "method": method,
                "tLim": limit,
                "best_instances": int(outcomes["min"].eq(best).sum()),
                "mean_compactness": (
                    float(outcomes["mean"].mean()) if eligible == len(instances) else float("nan")
                ),
                "feasible_instances": eligible,
            })
    return pd.DataFrame(summaries).set_index(["method", "tLim"])


def doTableS6(
    lns_runs, lns_ablation_runs, lns_fixed_runs, vns_low_runs, vns_medium_runs, vns_high_runs
):
    """Return best feasible attainment counts and mean compactness in Markdown.

    The per-instance benchmark is the best zero-penalty compactness across
    every run of all 27 LNS configurations and all three VNS variants. Every
    configuration attaining this value receives one count for that instance,
    regardless of how many runs attain it. Parentheses contain compactness
    averaged first over feasible runs within an instance, then equally over
    all 120 instances. The mean is undefined if any instance has no feasible run.
    """
    results = _table_s6_statistics(
        lns_runs, lns_ablation_runs, lns_fixed_runs, vns_low_runs, vns_medium_runs, vns_high_runs
    )
    lines = [
        "| Method | 60 s / VNS(l) | 300 s / VNS(m) | 600 s / VNS(h) |",
        "| :--- | ---: | ---: | ---: |",
    ]
    for method in S6_METHODS:
        cells = ["Full LNS" if method == "Full" else method]
        for limit in TIME_LIMITS:
            result = results.loc[(method, limit)]
            mean = result["mean_compactness"]
            rendered_mean = "—" if pd.isna(mean) else f"{mean:.1f}"
            cells.append(f"{int(result['best_instances'])} ({rendered_mean})")
        lines.append("| " + " | ".join(cells) + " |")
    return _table_output("S6", lines)


def _table_s7_statistics(lns_runs, lns_ablation_runs, lns_fixed_runs):
    """Pool MIP repair counters across all 3,600 runs per method and time limit."""
    runs, _ = _prepare_ablation_runs(lns_runs, lns_ablation_runs, lns_fixed_runs)
    runs = runs.loc[runs["variant"].isin(S7_METHODS)]
    columns = [
        "free", "fixed", "numRepairs", "tSolvingRepairs", "numRepairsOptimality",
        "numImprovesBalance", "numImprovesCompactness",
    ]
    missing = set(columns).difference(runs.columns)
    if missing:
        raise ValueError(f"MIP repair diagnostics: missing columns {sorted(missing)}")
    diagnostics = runs[columns].astype(float)
    if not np.isfinite(diagnostics.to_numpy()).all() or diagnostics.lt(0).any().any():
        raise ValueError("MIP repair diagnostics must be finite and nonnegative")

    groups = diagnostics.groupby([runs["variant"], runs["tLim"]])
    totals = groups.sum()
    run_counts = groups.size()
    summaries = []
    for method in S7_METHODS:
        for limit in TIME_LIMITS:
            total = totals.loc[(method, limit)]
            repairs = total["numRepairs"]
            repair_denominator = repairs if repairs > 0 else float("nan")
            candidates = total["free"] + total["fixed"]
            summaries.append({
                "method": method,
                "tLim": limit,
                "free_vars": total["free"] / repair_denominator,
                "fixed_pct": (
                    100 * total["fixed"] / candidates
                    if method != "No fixing" and candidates > 0 else float("nan")
                ),
                "repairs_per_run": repairs / run_counts.loc[(method, limit)],
                "time_per_repair": total["tSolvingRepairs"] / repair_denominator,
                "optimal_pct": 100 * total["numRepairsOptimality"] / repair_denominator,
                "balance_pct": 100 * total["numImprovesBalance"] / repair_denominator,
                "compactness_pct": 100 * total["numImprovesCompactness"] / repair_denominator,
            })
    return pd.DataFrame(summaries).set_index(["method", "tLim"])


def doTableS7(lns_runs, lns_ablation_runs, lns_fixed_runs):
    """Return eight methods' MIP repair diagnostics as a Markdown table.

    Free vars and t/repair divide aggregate free-variable counts and MIP time
    by aggregate repair counts. Fixed divides fixed-variable counts by the
    sum of free and fixed counts. Opt., Bal. and Comp. divide their respective
    counters by repair counts. R/run divides repair counts by 3,600 runs.
    All runs contribute regardless of feasibility. Heuristic repair is omitted;
    No fixing has blank Fixed cells. Statistics are displayed to one decimal,
    except t/repair, which uses two decimals.
    """
    results = _table_s7_statistics(lns_runs, lns_ablation_runs, lns_fixed_runs)
    statistics = (
        ("free_vars", "Free vars"),
        ("fixed_pct", "Fixed (%)"),
        ("repairs_per_run", "R/run"),
        ("time_per_repair", "t/repair (s)"),
        ("optimal_pct", "Opt. (%)"),
        ("balance_pct", "Bal. (%)"),
        ("compactness_pct", "Comp. (%)"),
    )
    # Standard Markdown has one header row; time prefixes preserve the groups.
    headers = ["Method", *(f"{limit} s {label}" for limit in TIME_LIMITS for _, label in statistics)]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| :--- | " + " | ".join(["---:"] * (len(TIME_LIMITS) * len(statistics))) + " |",
    ]
    for method in S7_METHODS:
        cells = ["Full LNS" if method == "Full" else method]
        for limit in TIME_LIMITS:
            result = results.loc[(method, limit)]
            for key, _ in statistics:
                value = result[key]
                cells.append(
                    "" if method == "No fixing" and key == "fixed_pct" else
                    "—" if pd.isna(value) else
                    f"{value:.2f}" if key == "time_per_repair" else f"{value:.1f}"
                )
        lines.append("| " + " | ".join(cells) + " |")
    return _table_output("S7", lines)


def _table_s8_statistics(lns_runs, lns_fixed_runs):
    """Compare Full and Fixed size using all feasible within-instance run pairs."""
    methods = {
        "full": _identify_ablation_variants(lns_runs, "lns.results.runs.txt"),
        "fixed": _identify_ablation_variants(lns_fixed_runs, "lns.fixed.results.runs.txt"),
    }
    instances = sorted(methods["full"]["Instance"].unique())
    if len(instances) != 120:
        raise ValueError(f"Expected 120 instance names, found {len(instances)}")
    expected = pd.MultiIndex.from_product(
        [TIME_LIMITS, instances, range(1, 31)], names=["tLim", "Instance", "seed"]
    )
    for method, runs in methods.items():
        _validate_run_grid(runs, expected, f"Table S8 {method}")

    summaries = []
    for limit in TIME_LIMITS:
        row = {"tLim": limit}
        samples = {}
        for method, runs in methods.items():
            subset = runs.loc[runs["tLim"].eq(limit)]
            feasible = subset.loc[subset["penalty"].eq(0)]
            row[f"{method}_feasible_pct"] = 100 * len(feasible) / len(subset)
            row[f"{method}_compactness"] = float(feasible["obj"].median())
            samples[method] = {
                instance: group["obj"].to_numpy(dtype=float)
                for instance, group in feasible.groupby("Instance")
            }

        full_wins = ties = fixed_wins = feasible_pairs = 0
        gaps = []
        for instance in instances:
            if instance not in samples["full"] or instance not in samples["fixed"]:
                continue
            full = samples["full"][instance][:, np.newaxis]
            fixed = samples["fixed"][instance][np.newaxis, :]
            if np.any(full == 0):
                raise ValueError(
                    f"Table S8/{limit}/{instance}: pairwise gap is undefined "
                    "because a feasible Full compactness is zero"
                )
            # Both penalties are zero, so lexicographic comparisons reduce to compactness.
            full_wins += int(np.count_nonzero(full < fixed))
            ties += int(np.count_nonzero(full == fixed))
            fixed_wins += int(np.count_nonzero(full > fixed))
            feasible_pairs += full.size * fixed.size
            gaps.append((100 * (fixed - full) / full).ravel())
        row.update({
            "full_wins": full_wins,
            "ties": ties,
            "fixed_wins": fixed_wins,
            "feasible_pairs": feasible_pairs,
            "median_gap": float(np.median(np.concatenate(gaps))) if gaps else float("nan"),
        })
        summaries.append(row)
    return pd.DataFrame(summaries).set_index("tLim")


def doTableS8(lns_runs, lns_fixed_runs):
    """Return Full versus Fixed size feasibility and feasible-pair comparisons.

    Median compactness pools all available individual feasible runs, giving
    each feasible run equal weight. Instances with no feasible runs contribute
    no observations; only an empty pool is undefined. Pair counts and median gap pool all
    Cartesian products of feasible runs within the same instance and time
    limit. Gap = 100 * (UB_fixed - UB_Full) / UB_Full; positive favors Full.
    """
    results = _table_s8_statistics(lns_runs, lns_fixed_runs)
    columns = (
        ("full_feasible_pct", "Feasibility Full LNS (%)"),
        ("fixed_feasible_pct", "Feasibility Fixed size (%)"),
        ("full_compactness", "Median compactness Full LNS"),
        ("fixed_compactness", "Median compactness Fixed size"),
        ("full_wins", "Pairwise outcome Full"),
        ("ties", "Pairwise outcome Tie"),
        ("fixed_wins", "Pairwise outcome Fixed"),
        ("feasible_pairs", "Pairwise outcome Feas. pairs"),
        ("median_gap", "Med. gap (%)"),
    )
    counts = {"full_wins", "ties", "fixed_wins", "feasible_pairs"}
    # Group prefixes keep the four supercolumns readable in standard Markdown.
    headers = ["Time limit", *(label for _, label in columns)]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| :--- | " + " | ".join(["---:"] * len(columns)) + " |",
    ]
    for limit in TIME_LIMITS:
        cells = [f"{limit} s"]
        for key, _ in columns:
            value = results.loc[limit, key]
            cells.append(
                "--" if pd.isna(value) else str(int(value)) if key in counts else f"{value:.1f}"
            )
        lines.append("| " + " | ".join(cells) + " |")
    return _table_output("S8", lines)


def _table_s9_statistics(lns_runs, vns_low_runs, vns_medium_runs, vns_high_runs):
    """Compare five configurations with Full LNS600 using validated run samples."""
    samples = _prepare_lns_vns_samples(lns_runs, vns_low_runs, vns_medium_runs, vns_high_runs)
    feasible = {
        method: {
            instance: outcomes[outcomes[:, 0] == 0, 1]
            for instance, outcomes in instances.items()
        }
        for method, instances in samples.items()
    }
    reference = feasible["LNS600"]
    reference_rate = (
        sum(values.size for values in reference.values())
        / sum(len(values) for values in samples["LNS600"].values())
    )
    summaries = []
    for method, label in S9_METHODS:
        other = feasible[method]
        method_rate = (
            sum(values.size for values in other.values())
            / sum(len(values) for values in samples[method].values())
        )
        differences, gap_arrays = [], []
        wins = ties = pairs = 0
        for instance, full in reference.items():
            candidate = other[instance]
            if not full.size or not candidate.size:
                continue
            if np.any(full == 0):
                raise ValueError(
                    f"Table S9/{label}/{instance}: pairwise gap is undefined "
                    "because a feasible Full LNS600 compactness is zero"
                )
            differences.append(float(np.median(candidate) - np.median(full)))
            full_column = full[:, np.newaxis]
            method_row = candidate[np.newaxis, :]
            wins += int(np.count_nonzero(full_column < method_row))
            ties += int(np.count_nonzero(full_column == method_row))
            pairs += full.size * candidate.size
            gap_arrays.append((100 * (method_row - full_column) / full_column).ravel())
        gaps = np.concatenate(gap_arrays) if gap_arrays else np.array([], dtype=float)
        summaries.append({
            "method": label,
            "feasibility_change": 100 * (reference_rate - method_rate),
            "compactness_change": (
                float(np.median(differences)) if len(differences) == len(reference) else float("nan")
            ),
            "score": (wins + 0.5 * ties) / pairs if pairs else float("nan"),
            "feasible_pairs": pairs,
            "median_gap": float(np.median(gaps)) if pairs else float("nan"),
            "mean_gap": float(np.mean(gaps)) if pairs else float("nan"),
        })
    return pd.DataFrame(summaries).set_index("method")


def doTableS9(lns_runs, vns_low_runs, vns_medium_runs, vns_high_runs):
    """Return five configurations' comparisons against Full LNS at 600 s.

    Feas. change is reference minus row-method feasibility in percentage points.
    Comp. change is the median across 120 differences of instance-level feasible
    medians (row method minus reference). Score and gaps pool all feasible
    within-instance Cartesian run pairs, giving each pair equal weight. Score
    credits a reference win with one, a tie with one half and a loss with zero.
    Positive changes and gaps, or Score above 0.5, favor Full LNS600.
    """
    results = _table_s9_statistics(lns_runs, vns_low_runs, vns_medium_runs, vns_high_runs)
    columns = (
        ("feasibility_change", "Feas. change (pp)"),
        ("compactness_change", "Comp. change"),
        ("score", "Score"),
        ("feasible_pairs", "Feas. pairs"),
        ("median_gap", "Med. gap (%)"),
        ("mean_gap", "Mean gap (%)"),
    )
    headers = ["Method", *(label for _, label in columns)]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| :--- | " + " | ".join(["---:"] * len(columns)) + " |",
    ]
    for _, label in S9_METHODS:
        cells = [label]
        for key, _ in columns:
            value = results.loc[label, key]
            cells.append(
                "--" if pd.isna(value) else
                str(int(value)) if key == "feasible_pairs" else
                f"{value:.3f}" if key == "score" else f"{value:.1f}"
            )
        lines.append("| " + " | ".join(cells) + " |")
    return _table_output("S9", lines)


def _table_s10_statistics(lns_runs, vns_low_runs, vns_medium_runs, vns_high_runs):
    """Summarize individual-run LNS time to best and VNS total runtime."""
    lns = _identify_ablation_variants(lns_runs, "lns.results.runs.txt")
    instances = sorted(lns["Instance"].unique())
    if len(instances) != 120:
        raise ValueError(f"Expected 120 instance names, found {len(instances)}")
    expected_lns = pd.MultiIndex.from_product(
        [TIME_LIMITS, instances, range(1, 31)], names=["tLim", "Instance", "seed"]
    )
    _validate_run_grid(lns, expected_lns, "Full LNS")
    expected_vns = pd.MultiIndex.from_product(
        [instances, range(1, 31)], names=["Instance", "seed"]
    )
    summaries = []
    for limit, variant, vns in zip(
        TIME_LIMITS, ("l", "m", "h"), (vns_low_runs, vns_medium_runs, vns_high_runs)
    ):
        _prepare_vns_runs(vns, expected_vns, f"VNS ({variant})")
        for label, dataframe, column in (
            (f"Full LNS ({limit} s)", lns.loc[lns["tLim"].eq(limit)], "tBest"),
            (f"VNS ({variant})", vns, "time"),
        ):
            if column not in dataframe.columns:
                raise ValueError(f"{label}: missing time column {column}")
            times = dataframe[column].astype(float)
            if not np.isfinite(times.to_numpy()).all() or times.lt(0).any():
                raise ValueError(f"{label}: {column} must contain finite, nonnegative seconds")
            quartiles = times.quantile([0.25, 0.5, 0.75], interpolation="linear")
            summaries.append({
                "method": label,
                "mean": float(times.mean()),
                "q1": float(quartiles.loc[0.25]),
                "median": float(quartiles.loc[0.5]),
                "q3": float(quartiles.loc[0.75]),
                "std": float(times.std(ddof=1)),
            })
    return pd.DataFrame(summaries).set_index("method")


def doTableS10(lns_runs, vns_low_runs, vns_medium_runs, vns_high_runs):
    """Return runtime summaries in Markdown, with five columns in seconds.

    Full LNS uses tBest (time to best result), and VNS uses time (total runtime).
    Each summary pools all 3,600 individual runs, regardless of feasibility.
    Quartiles use linear interpolation; standard deviation uses n - 1.
    Recorded times are used directly, including zero and any time-limit overrun.
    """
    results = _table_s10_statistics(lns_runs, vns_low_runs, vns_medium_runs, vns_high_runs)
    lines = [
        "| Method | Mean (s) | Q1 (s) | Median (s) | Q3 (s) | Std. dev. (s) |",
        "| :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method, row in results.iterrows():
        cells = [method, *(f"{row[key]:.1f}" for key in ("mean", "q1", "median", "q3", "std"))]
        lines.append("| " + " | ".join(cells) + " |")
    return _table_output("S10", lines)


def _parse_run_instance_names(instances):
    """Extract layout, BU count and graph ID from the 120-instance runs design."""
    names = pd.Index(instances, name="Instance")
    metadata = pd.Series(names, index=names).str.extract(
        r"\A(?P<Layout>Center|Corners|Diagonal|planar)(?P<BUs>[0-9]+)_(?P<ID>G[0-9]+)\Z"
    )
    invalid = metadata.isna().any(axis=1)
    if invalid.any():
        raise ValueError(f"Invalid repeated-run instance names: {names[invalid][:3].tolist()}")
    metadata["Layout"] = metadata["Layout"].replace({"planar": "Planar"})
    metadata["BUs"] = metadata["BUs"].astype(int)
    return (
        metadata.assign(id_number=metadata["ID"].str[1:].astype(int))
        .sort_values(["Layout", "BUs", "id_number"])
        .drop(columns="id_number")
    )


def _table11_statistics(lns_runs, lns_ablation_runs, lns_fixed_runs):
    """Return instance metadata and all 3,240 feasible-run summary groups."""
    runs, instances = _prepare_ablation_runs(lns_runs, lns_ablation_runs, lns_fixed_runs)
    metadata = _parse_run_instance_names(instances)
    expected = pd.MultiIndex.from_product(
        [metadata.index, S5_METHODS, TIME_LIMITS], names=["Instance", "method", "tLim"]
    )
    feasible = runs.loc[runs["penalty"].eq(0)].rename(columns={"variant": "method"})
    results = (
        feasible.groupby(["Instance", "method", "tLim"])["obj"]
        .agg(feasible="count", best="min", mean="mean", median="median", s="std")
        .reindex(expected)
    )
    results["feasible"] = results["feasible"].fillna(0).astype(int)
    # Sample standard deviation is defined as zero for one feasible run here.
    results.loc[results["feasible"].eq(1), "s"] = 0.0
    return metadata, results


def doTable11(lns_runs, lns_ablation_runs, lns_fixed_runs):
    """Return 120 instance blocks, each with nine methods and three time limits.

    Feas. counts zero-penalty runs out of 30. Best, Mean, Median and sample
    standard deviation use feasible compactness only. No feasible runs gives
    dashes for compactness summaries; one feasible run gives s = 0.
    Within each instance, bold the highest Feas. and lowest Best, Mean and
    Median across all 27 method/time combinations, including all ties after
    formatting to one decimal. Standard deviations are never highlighted.
    """
    metadata, results = _table11_statistics(lns_runs, lns_ablation_runs, lns_fixed_runs)
    statistics = (
        ("feasible", "Feas."), ("best", "Best"), ("mean", "Mean"),
        ("median", "Median"), ("s", "s"),
    )
    # Explicit time prefixes preserve the three groups in standard Markdown.
    headers = [
        "Layout", "BUs", "ID", "Method",
        *(f"{limit} s {label}" for limit in TIME_LIMITS for _, label in statistics),
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| :--- | ---: | :--- | :--- | " + " | ".join(["---:"] * 15) + " |",
    ]
    for instance, identity in metadata.iterrows():
        block = results.xs(instance, level="Instance")
        best_display = {"feasible": str(int(block["feasible"].max()))}
        for key in ("best", "mean", "median"):
            value = block[key].min()
            best_display[key] = None if pd.isna(value) else f"{value:.1f}"
        for index, method in enumerate(S5_METHODS):
            cells = (
                [identity["Layout"], str(identity["BUs"]), identity["ID"]]
                if index == 0 else ["", "", ""]
            )
            cells.append("Full LNS" if method == "Full" else method)
            for limit in TIME_LIMITS:
                row = block.loc[(method, limit)]
                for key, _ in statistics:
                    value = row[key]
                    displayed = (
                        "—" if pd.isna(value) else
                        str(int(value)) if key == "feasible" else f"{value:.1f}"
                    )
                    if displayed == best_display.get(key):
                        displayed = f"**{displayed}**"
                    cells.append(displayed)
            lines.append("| " + " | ".join(cells) + " |")
    return _table_output("S11", lines)


def _table_s12_statistics(lns, vns_low, vns_medium, vns_high):
    """Group non-run compactness gaps and best counts by three characteristics."""
    metadata, compactness = _prepare_single_compactness(lns, vns_low, vns_medium, vns_high)
    compactness = compactness.loc[:, list(SINGLE_RESULT_METHODS)]
    best = compactness.min(axis=1)
    gaps = 100 * compactness.sub(best, axis=0).div(best, axis=0)
    attainments = compactness.eq(best, axis=0)
    attributes = ["Demand", "Workload", "Customer"]
    groups = [metadata[attribute] for attribute in attributes]
    order = pd.MultiIndex.from_product([ATTRIBUTE_LEVELS] * 3, names=attributes)
    mean_gaps = gaps.groupby(groups).mean().reindex(order)
    best_counts = attainments.groupby(groups).sum().reindex(order)
    total = pd.MultiIndex.from_tuples([("Total", "", "")], names=attributes)
    mean_gaps = pd.concat([mean_gaps, pd.DataFrame([gaps.mean()], index=total)])
    best_counts = pd.concat([best_counts, pd.DataFrame([attainments.sum()], index=total)])
    return mean_gaps, best_counts


def doTableS12(lns, vns_low, vns_medium, vns_high):
    """Return 27 characteristic groups and a total row of gaps and attainments.

    Each instance's best compactness is the minimum over all six configurations,
    irrespective of balance feasibility. Each cell averages 100 * (UB - best)
    / best over its 90 instances and counts all instances attaining the exact
    best, crediting every tied method. Total uses all 2,430 instance-level gaps
    and sums best attainments. Display gaps to two decimals and counts as integers.
    """
    mean_gaps, best_counts = _table_s12_statistics(lns, vns_low, vns_medium, vns_high)
    headers = ["Demand", "Workload", "Customer", *SINGLE_RESULT_METHODS]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| :--- | :--- | :--- | " + " | ".join(["---:"] * len(SINGLE_RESULT_METHODS)) + " |",
    ]
    for group, row in mean_gaps.iterrows():
        cells = [
            *group,
            *(f"{row[method]:.2f} ({int(best_counts.loc[group, method])})" for method in SINGLE_RESULT_METHODS),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return _table_output("S12", lines)


def _table_s13_data(lns, vns_low, vns_medium, vns_high):
    """Prepare 14,580 unfiltered outcomes and categorical instance characteristics."""
    metadata, compactness = _prepare_single_compactness(lns, vns_low, vns_medium, vns_high)
    source_order = [*SINGLE_RESULT_METHODS[3:], *SINGLE_RESULT_METHODS[:3]]
    compactness = compactness.loc[:, source_order].rename(columns=dict(zip(source_order, S13_METHODS)))
    data = (
        compactness.reset_index()
        .melt(id_vars="Instance", var_name="Method", value_name="compactness")
        .merge(metadata.reset_index(), on="Instance", validate="many_to_one")
        .rename(columns={"#BU": "BUs", "Graph": "Layout"})
    )
    categories = {
        "Method": S13_METHODS,
        "BUs": (486, 600, 726),
        "Layout": ("Center", "Corners", "Diagonal"),
        **{factor: ATTRIBUTE_LEVELS for factor in ("Demand", "Workload", "Customer")},
    }
    for factor, levels in categories.items():
        data[factor] = pd.Categorical(data[factor], categories=levels)
    data["log_compactness"] = np.log(data["compactness"].to_numpy(dtype=float))
    return data


def _table_s13_statistics(lns, vns_low, vns_medium, vns_high):
    """Fit raw/log random-intercept models by ML and calculate Wald, EMM and contrast panels.

    All six fixed effects are categorical and additive. EMMs equally average
    fixed-effect predictions over the 243 combinations of the other factors;
    random intercepts are set to their population mean of zero. Wald tests use
    chi-square reference distributions; contrasts use normal-reference Wald
    intervals and two-sided p-values. The 15 p-values are Holm-adjusted per model.
    """
    data = _table_s13_data(lns, vns_low, vns_medium, vns_high)
    design = dmatrix(S13_FORMULA, data, return_type="dataframe", NA_action="raise")
    other_factors = [factor for factor, _ in S13_FACTORS if factor != "Method"]
    grid = data[other_factors].drop_duplicates().reset_index(drop=True)
    emm_rows = []
    for method in S13_METHODS:
        reference = grid.assign(Method=pd.Categorical([method] * len(grid), categories=S13_METHODS))
        matrix = build_design_matrices([design.design_info], reference, NA_action="raise")[0]
        emm_rows.append(np.asarray(matrix).mean(axis=0))
    emm_rows = np.asarray(emm_rows)

    omnibus, diagnostics = [], []
    emms = pd.DataFrame(index=pd.Index(S13_METHODS, name="method"))
    contrast_index = pd.MultiIndex.from_tuples(list(combinations(S13_METHODS, 2)), names=["left", "right"])
    contrasts = pd.DataFrame(index=contrast_index)
    for scale, response in (("Raw", "compactness"), ("Log", "log_compactness")):
        y = data[response].to_numpy(dtype=float)
        center, response_scale = float(y.mean()), float(y.std(ddof=0))
        if response_scale <= 0:
            raise ValueError(f"Table S13 {scale}: a constant response cannot support mixed-model inference")
        # Numerical standardization avoids unit-dependent boundary warnings.
        # Estimates, intervals and variances are converted back to raw/log units.
        fit = MixedLM((y - center) / response_scale, design, groups=data["Instance"]).fit(
            reml=False, method="bfgs", maxiter=1000, gtol=1e-8, disp=False,
        )
        covariance = fit.cov_params().to_numpy()
        fixed_covariance = covariance[:design.shape[1], :design.shape[1]]
        if (
            not fit.converged or not np.isfinite(fit.params).all()
            or not np.isfinite(covariance).all() or fit.scale <= 0
            or np.linalg.eigvalsh(fixed_covariance).min() <= 0
        ):
            raise RuntimeError(f"Table S13 {scale}: mixed model did not yield a converged, valid fit")

        for factor, label in S13_FACTORS:
            term = design.design_info.term_name_slices[f"C({factor})"]
            positions = np.arange(term.start, term.stop)
            restriction = np.eye(len(fit.params))[positions]
            test = fit.wald_test(restriction, use_f=False, scalar=True)
            omnibus.append({
                "scale": scale, "factor": label, "df": len(positions),
                "chi2": float(test.statistic), "p": float(test.pvalue),
            })

        prefix = scale.lower()
        estimates = center + response_scale * (emm_rows @ fit.fe_params.to_numpy())
        emms[f"{prefix}_emm"] = estimates
        emms[f"{prefix}_rank"] = pd.Series(estimates).rank(method="min").to_numpy(dtype=int)
        comparisons = []
        for left, right in combinations(range(len(S13_METHODS)), 2):
            restriction = (emm_rows[left] - emm_rows[right])[np.newaxis, :]
            test = fit.t_test(restriction, use_t=False)
            lower, upper = test.conf_int(alpha=0.05)[0] * response_scale
            comparisons.append({
                f"{prefix}_difference": float(np.asarray(test.effect).item()) * response_scale,
                f"{prefix}_se": float(np.asarray(test.sd).item()) * response_scale,
                f"{prefix}_ci_low": float(lower),
                f"{prefix}_ci_high": float(upper),
                f"{prefix}_p": float(np.asarray(test.pvalue).item()),
            })
        panel = pd.DataFrame(comparisons, index=contrast_index)
        panel[f"{prefix}_holm"] = _holm_adjust(panel[f"{prefix}_p"].to_numpy())
        contrasts = contrasts.join(panel)
        diagnostics.append({
            "scale": scale, "converged": bool(fit.converged), "observations": int(fit.nobs),
            "instances": data["Instance"].nunique(), "fixed_coefficients": design.shape[1],
            "random_intercept_variance": float(fit.cov_re.iloc[0, 0]) * response_scale ** 2,
            "residual_variance": float(fit.scale) * response_scale ** 2,
        })
    return {
        "omnibus": pd.DataFrame(omnibus).set_index(["scale", "factor"]),
        "emms": emms,
        "contrasts": contrasts,
        "diagnostics": pd.DataFrame(diagnostics).set_index("scale"),
    }


def _table_s13_markdown(results):
    """Render mixed-model panels; log EMMs, differences and CI limits use four decimals."""
    def pvalue(value):
        return "<0.001" if value < 0.001 else f"{value:.3f}"

    lines = [
        "**Panel A**", "",
        "| Factor | Raw df | Raw Wald χ² | Raw p | Log df | Log Wald χ² | Log p |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, factor in S13_FACTORS:
        cells = [factor]
        for scale in ("Raw", "Log"):
            row = results["omnibus"].loc[(scale, factor)]
            cells.extend([str(int(row["df"])), f"{row['chi2']:.3f}", pvalue(row["p"])])
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend([
        "", "**Panel B1**", "",
        "| Method | Raw EMM | Raw rank | Log EMM | Log rank |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ])
    for method, row in results["emms"].iterrows():
        lines.append(
            f"| {method} | {row['raw_emm']:.3f} | {int(row['raw_rank'])} | "
            f"{row['log_emm']:.4f} | {int(row['log_rank'])} |"
        )
    lines.extend([
        "", "**Panel B2**", "",
        "| Comparison | Raw difference | Raw 95% CI | Raw Holm p | Log difference | Log 95% CI | Log Holm p |",
        "| :--- | ---: | :--- | ---: | ---: | :--- | ---: |",
    ])
    for (left, right), row in results["contrasts"].iterrows():
        cells = [f"{left} − {right}"]
        for scale in ("raw", "log"):
            precision = 4 if scale == "log" else 3
            cells.extend([
                f"{row[f'{scale}_difference']:.{precision}f}",
                f"[{row[f'{scale}_ci_low']:.{precision}f}, {row[f'{scale}_ci_high']:.{precision}f}]",
                pvalue(row[f"{scale}_holm"]),
            ])
        lines.append("| " + " | ".join(cells) + " |")
    return _table_output("S13", lines)


def doTableS13(lns, vns_low, vns_medium, vns_high):
    """Return raw/log mixed-model Wald tests, EMMs and method contrasts in Markdown."""
    return _table_s13_markdown(_table_s13_statistics(lns, vns_low, vns_medium, vns_high))


def _table_s14_bounds(lower_bounds, instances):
    """Match full instance names and retain dispersion and the best available bound."""
    required = {"Instance", *S14_BOUND_COLUMNS}
    missing = required.difference(lower_bounds.columns)
    if missing:
        raise ValueError(f"Table S14 bounds: missing columns {sorted(missing)}")
    if lower_bounds["Instance"].isna().any() or lower_bounds["Instance"].duplicated().any():
        raise ValueError("Table S14 bounds: missing or duplicate instance names")
    bounds = lower_bounds.set_index("Instance")
    missing = instances.difference(bounds.index)
    if len(missing):
        raise ValueError(f"Table S14 bounds: missing {len(missing)} instances, e.g. {missing[:3].tolist()}")
    bounds = bounds.loc[instances, list(S14_BOUND_COLUMNS)].astype(float)
    values = bounds.to_numpy()
    if (
        not np.isfinite(bounds["dispersion"]).all()
        or np.isinf(values).any() or np.any(values < 0)
    ):
        raise ValueError("Table S14 bounds: require finite, nonnegative bounds and nonmissing dispersion")
    # Missing optional bounds are unavailable; dispersion is always required.
    return pd.DataFrame({"dispersion": bounds["dispersion"], "best": bounds.max(axis=1)})


def _table_s14_gap_summary(compactness, feasible_compactness, bounds):
    """Summarize instance-level gaps with equal weights and sample standard deviations."""
    all_gaps = 100 * compactness.sub(bounds["dispersion"], axis=0).div(compactness)
    feasible_gaps = 100 * feasible_compactness.sub(bounds["best"], axis=0).div(feasible_compactness)
    rows = []
    for method in compactness.columns:
        row = {"method": method, "feasible_instances": int(feasible_compactness[method].notna().sum())}
        for prefix, gaps in (("all", all_gaps[method]), ("feasible", feasible_gaps[method])):
            gaps = gaps.dropna()
            quartiles = gaps.quantile([0.25, 0.5, 0.75])
            row.update({
                f"{prefix}_mean": float(gaps.mean()),
                f"{prefix}_s": float(gaps.std(ddof=1)),
                f"{prefix}_q1": float(quartiles.loc[0.25]),
                f"{prefix}_median": float(quartiles.loc[0.5]),
                f"{prefix}_q3": float(quartiles.loc[0.75]),
            })
        rows.append(row)
    return pd.DataFrame(rows).set_index("method")


def _table_s14_statistics(
    lns_runs, vns_low_runs, vns_medium_runs, vns_high_runs,
    lns, vns_low, vns_medium, vns_high, lower_bounds,
):
    """Calculate repeated-run and single-run bound-gap panels without pooling runs."""
    samples = _prepare_lns_vns_samples(lns_runs, vns_low_runs, vns_medium_runs, vns_high_runs)
    run_medians, feasible_run_medians = {}, {}
    for method, label in S14_METHODS:
        run_medians[label], feasible_run_medians[label] = {}, {}
        for instance, outcomes in samples[method].items():
            if np.any(outcomes[:, 1] <= 0) or np.any(outcomes[:, 0] < 0):
                raise ValueError("Table S14 runs: compactness must be positive and penalties nonnegative")
            run_medians[label][instance] = float(np.median(outcomes[:, 1]))
            feasible = outcomes[outcomes[:, 0] == 0, 1]
            feasible_run_medians[label][instance] = float(np.median(feasible)) if len(feasible) else np.nan
    run_medians = pd.DataFrame(run_medians).rename_axis("Instance")
    feasible_run_medians = pd.DataFrame(feasible_run_medians).rename_axis("Instance")

    _, single_compactness = _prepare_single_compactness(lns, vns_low, vns_medium, vns_high)
    single_feasibility = _prepare_single_results(lns, vns_low, vns_medium, vns_high)
    source_order = [*SINGLE_RESULT_METHODS[3:], *SINGLE_RESULT_METHODS[:3]]
    labels = dict(zip(source_order, [label for _, label in S14_METHODS]))
    single_compactness = single_compactness.loc[:, source_order].rename(columns=labels)
    single_feasibility = single_feasibility.loc[:, source_order].rename(columns=labels)
    bounds = _table_s14_bounds(lower_bounds, run_medians.index.union(single_compactness.index))
    return {
        "A": _table_s14_gap_summary(run_medians, feasible_run_medians, bounds.loc[run_medians.index]),
        "B": _table_s14_gap_summary(
            single_compactness, single_compactness.where(single_feasibility),
            bounds.loc[single_compactness.index],
        ),
    }


def doTableS14(
    lns_runs, vns_low_runs, vns_medium_runs, vns_high_runs,
    lns, vns_low, vns_medium, vns_high, lower_bounds,
):
    """Return two Markdown panels of percentage bound gaps, 100 * (UB - LB) / UB.

    Panel A uses one median per method/instance: all 30 runs for the all-instance
    columns, and only zero-penalty runs for the feasible columns. Panel B uses
    the single-run compactness. All-instance gaps use dispersion; feasible gaps
    use the maximum available bound. Each instance has equal weight. Counts
    are integers; other statistics are rounded to two decimals only for display.
    """
    panels = _table_s14_statistics(
        lns_runs, vns_low_runs, vns_medium_runs, vns_high_runs,
        lns, vns_low, vns_medium, vns_high, lower_bounds,
    )
    statistics = (("mean", "Mean"), ("s", "s"), ("q1", "Q1"), ("median", "Median"), ("q3", "Q3"))
    # Standard Markdown has no column spans, so repeat the group names in headers.
    headers = [
        "Method", *(f"All {label}" for _, label in statistics), "Feasible Inst.",
        *(f"Feasible {label}" for _, label in statistics),
    ]
    lines = []
    for panel, title in (
        ("A", "Panel A"),
        ("B", "Panel B"),
    ):
        lines.extend([
            f"**{title}**", "",
            "| " + " | ".join(headers) + " |",
            "| :--- | " + " | ".join(["---:"] * 11) + " |",
        ])
        for method, row in panels[panel].iterrows():
            cells = [method]
            for prefix in ("all", "feasible"):
                if prefix == "feasible":
                    cells.append(str(int(row["feasible_instances"])))
                for key, _ in statistics:
                    value = row[f"{prefix}_{key}"]
                    cells.append("—" if pd.isna(value) else f"{value:.2f}")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    return _table_output("S14", lines)


if __name__ == "__main__":
    dataframes = main()
    ablation_inputs = (
        dataframes["lns_runs"],
        dataframes["lns_ablation_runs"],
        dataframes["lns_fixed_runs"],
    )
    run_inputs = (
        dataframes["lns_runs"],
        dataframes["vns_low_runs"],
        dataframes["vns_medium_runs"],
        dataframes["vns_high_runs"],
    )
    single_inputs = (
        dataframes["lns"],
        dataframes["vns_low"],
        dataframes["vns_medium"],
        dataframes["vns_high"],
    )
    print(doTableS5(*ablation_inputs), end="")
    print(doTableS6(*ablation_inputs, *run_inputs[1:]), end="")
    print(doTableS7(*ablation_inputs), end="")
    print(doTableS8(dataframes["lns_runs"], dataframes["lns_fixed_runs"]), end="")
    print(doTableS9(*run_inputs), end="")
    print(doTableS10(*run_inputs), end="")
    print(doTable11(*ablation_inputs), end="")
    print(doTableS12(*single_inputs), end="")
    print(doTableS13(*single_inputs), end="")
    print(doTableS14(*run_inputs, *single_inputs, dataframes["lower_bounds"]), end="")
