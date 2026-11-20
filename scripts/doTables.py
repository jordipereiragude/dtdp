"""Print the manuscript tables in Markdown.

Input CSV paths are resolved relative to this file, so the command is
independent of the caller's working directory.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, norm, rankdata


ABLATION_FLAGS = {
    "randomDestroy": "Random destroy",
    "useAlternativeDestroy": "Distance-based destroy",
    "usekDestroy": "District-based destroy",
    "disableVariableFixing": "No fixing",
    "noWarmstart": "No warm start",
    "useMultistart": "Only random start",
    "heuristicPercentage": "Heuristic repair",
}

ABLATION_VARIANTS = (
    "Random destroy",
    "Distance-based destroy",
    "District-based destroy",
    "Fixed destroy size",
    "No fixing",
    "No warm start",
    "Only random start",
    "Heuristic repair",
)

TIME_LIMITS = (60, 300, 600)
LNS_VNS_METHODS = ("LNS60", "LNS300", "LNS600", "VNS(l)", "VNS(m)", "VNS(h)")
SINGLE_RESULT_METHODS = ("VNS(l)", "VNS(m)", "VNS(h)", "LNS (60s)", "LNS (300s)", "LNS (600s)")
INSTANCE_GRAPHS = ("Center", "Corners", "Diagonal")
INSTANCE_BU_SIZES = (486, 600, 726)
ATTRIBUTE_LEVELS = ("Low", "Medium", "High")


def _table_output(number, lines):
    """Label a Markdown table and leave three blank lines after it."""
    body = "\n".join(lines).strip()
    return f"Table {number}\n\n{body}\n\n\n\n"


def _identify_ablation_variants(dataframe, source):
    """Label a copy of a result DataFrame using its source and ablation flags."""
    flags = list(ABLATION_FLAGS)
    required = {"Instance", "seed", "tLim", "min", "max", "penalty", "obj", *flags}
    missing = required.difference(dataframe.columns)
    if missing:
        raise ValueError(f"{source}: missing columns {sorted(missing)}")
    if dataframe.empty or dataframe[list(required)].isna().any().any():
        raise ValueError(f"{source}: empty data or missing identification/outcome values")
    for flag in flags:
        if not pd.api.types.is_bool_dtype(dataframe[flag]):
            raise ValueError(f"{source}: {flag} must contain boolean values")
    if not np.isfinite(dataframe[["penalty", "obj"]].to_numpy(dtype=float)).all():
        raise ValueError(f"{source}: penalty and objective must be finite")

    runs = dataframe.copy()
    active_flags = runs[flags].sum(axis=1)
    if source == "lns.ablation.results.runs.txt":
        if not active_flags.eq(1).all():
            raise ValueError(f"{source}: each row must have exactly one active flag")
        runs["variant"] = runs[flags].idxmax(axis=1).map(ABLATION_FLAGS)
    elif source in ("lns.results.runs.txt", "lns.fixed.results.runs.txt"):
        if not active_flags.eq(0).all():
            raise ValueError(f"{source}: all ablation flags must be false")
        runs["variant"] = (
            "Full" if source == "lns.results.runs.txt" else "Fixed destroy size"
        )
    else:
        raise ValueError(f"Unknown result file: {source}")

    if source == "lns.fixed.results.runs.txt":
        if not runs["min"].eq(runs["max"]).all():
            raise ValueError(f"{source}: fixed destroy size requires min == max")
    elif not runs["min"].lt(runs["max"]).all():
        raise ValueError(f"{source}: increasing destroy size requires min < max")

    runs["source"] = source
    return runs


def _validate_run_grid(runs, expected, source):
    """Require each expected run identifier exactly once, with no extra runs."""
    keys = list(expected.names)
    if runs[keys].isna().any().any():
        raise ValueError(f"{source}: missing run identifiers")
    duplicates = runs.duplicated(keys)
    if duplicates.any():
        example = runs.loc[duplicates, keys].iloc[0].to_dict()
        raise ValueError(f"{source}: Duplicate run identifiers: {example}")

    actual = pd.MultiIndex.from_frame(runs[keys])
    missing = expected.difference(actual, sort=False)
    unexpected = actual.difference(expected, sort=False)
    if len(missing) or len(unexpected):
        raise ValueError(
            f"{source}: incomplete comparison: {len(missing)} missing runs, "
            f"{len(unexpected)} unexpected runs. "
            f"Examples: missing={missing[:3].tolist()}, "
            f"unexpected={unexpected[:3].tolist()}"
        )


def _validate_ablation_runs(runs, expected_instances):
    """Require nine variants, three limits, 120 common instances and seeds 1--30."""
    expected_instances = sorted(set(expected_instances))
    if len(expected_instances) != 120:
        raise ValueError(f"Expected 120 instance names, found {len(expected_instances)}")
    expected = pd.MultiIndex.from_product(
        [("Full", *ABLATION_VARIANTS), TIME_LIMITS, expected_instances, range(1, 31)],
        names=["variant", "tLim", "Instance", "seed"],
    )
    _validate_run_grid(runs, expected, "Ablation results")


def _prepare_ablation_runs(lns_runs, lns_ablation_runs, lns_fixed_runs):
    """Identify and validate the three input DataFrames without modifying them."""
    runs = pd.concat(
        [
            _identify_ablation_variants(lns_runs, "lns.results.runs.txt"),
            _identify_ablation_variants(lns_ablation_runs, "lns.ablation.results.runs.txt"),
            _identify_ablation_variants(lns_fixed_runs, "lns.fixed.results.runs.txt"),
        ],
        ignore_index=True,
    )
    instances = sorted(runs.loc[runs["variant"].eq("Full"), "Instance"].unique())
    _validate_ablation_runs(runs, instances)
    return runs, instances


def _holm_adjust(pvalues):
    """Return Holm step-down adjusted p-values in the original order."""
    pvalues = np.asarray(pvalues, dtype=float)
    if pvalues.ndim != 1 or not np.isfinite(pvalues).all():
        raise ValueError("p-values must be a finite one-dimensional array")
    if np.any((pvalues < 0) | (pvalues > 1)):
        raise ValueError("p-values must lie between zero and one")
    order = np.argsort(pvalues, kind="stable")
    adjusted = np.empty_like(pvalues)
    adjusted[order] = np.minimum(
        1.0,
        np.maximum.accumulate(pvalues[order] * np.arange(len(pvalues), 0, -1)),
    )
    return adjusted


def _compare_instance(full, variant):
    """Compare arrays of (penalty, objective), with lower values ranked better.

    All runs are included. Outcomes are ordered lexicographically: lower
    penalty first, then lower objective. Identical pairs receive average ranks.
    Mann--Whitney uses a two-sided, tie-corrected asymptotic p-value with the
    usual 0.5 continuity correction. The centered U favors Full when positive.
    """
    full = np.asarray(full, dtype=float)
    variant = np.asarray(variant, dtype=float)
    for sample in (full, variant):
        if sample.ndim != 2 or sample.shape[1] != 2 or len(sample) == 0:
            raise ValueError("Each sample must contain (penalty, objective) pairs")
        if not np.isfinite(sample).all():
            raise ValueError("Outcome values must be finite")

    n_full, n_variant = len(full), len(variant)
    n = n_full + n_variant
    outcomes = np.concatenate((full, variant))
    # np.unique sorts rows lexicographically; its inverse preserves outcome ties.
    _, scores = np.unique(outcomes, axis=0, return_inverse=True)
    ranks = rankdata(scores, method="average")
    # This permutation variance includes the correction for tied outcomes.
    variance = (
        n_full * n_variant / (n * (n - 1))
        * np.sum((ranks - (n + 1) / 2.0) ** 2)
    )
    if variance == 0:
        # All outcomes tie: no evidence of a difference, including in Holm's family.
        return {"centered_u": 0.0, "variance": 0.0, "weight": 1.0 / (n + 1), "p": 1.0}

    test = mannwhitneyu(
        ranks[:n_full],
        ranks[n_full:],
        alternative="two-sided",
        method="asymptotic",
        use_continuity=True,
    )
    return {
        "centered_u": n_full * n_variant / 2.0 - float(test.statistic),
        "variance": float(variance),
        "weight": 1.0 / (n + 1),
        "p": float(test.pvalue),
    }


def _van_elteren(strata):
    """Return signed z and a two-sided asymptotic p-value, without continuity correction.

    Each instance is a stratum, weighted by 1 / (n_full + n_variant + 1).
    Use tie-corrected variances from _compare_instance. Positive z favors Full.
    Formula: https://pmc.ncbi.nlm.nih.gov/articles/PMC8717684/ (Eqs. 3, 4, 6).
    """
    numerator = sum(item["weight"] * item["centered_u"] for item in strata)
    variance = sum(item["weight"] ** 2 * item["variance"] for item in strata)
    if variance == 0:
        return 0.0, 1.0
    z = numerator / np.sqrt(variance)
    return float(z), float(2.0 * norm.sf(abs(z)))


def _comparison_statistics(full_samples, other_samples):
    """Count Holm-significant instance results and calculate the stratified test."""
    strata = [
        _compare_instance(full, other_samples[instance])
        for instance, full in full_samples.items()
    ]
    adjusted = _holm_adjust([item["p"] for item in strata])
    directions = np.array([item["centered_u"] for item in strata])
    significant = adjusted < 0.05
    z, pvalue = _van_elteren(strata)
    return {
        "full": int(np.sum(significant & (directions > 0))),
        "variant": int(np.sum(significant & (directions < 0))),
        "z": z,
        "p": pvalue,
    }


def _table3_markdown(results):
    """Render the three time-limit groups using a two-row Markdown header."""
    groups = ["Ablated variant"]
    for limit in TIME_LIMITS:
        groups.extend([f"**{limit} s**", "", "", ""])
    lines = [
        "| " + " | ".join(groups) + " |",
        "| :--- | " + " | ".join(["---:"] * 12) + " |",
        "| | " + " | ".join(["full", "variant", "z", "Holm"] * 3) + " |",
    ]
    for variant in ABLATION_VARIANTS:
        cells = [variant]
        for limit in TIME_LIMITS:
            result = results.loc[(variant, limit)]
            stars = sum(result["p"] < threshold for threshold in (0.05, 0.01, 0.001))
            z = f"{result['z']:.3f}" + r"\*" * stars
            adjusted = result["Holm"]
            holm = "<0.001" if adjusted < 0.001 else f"{adjusted:.3f}"
            cells.extend([str(int(result["full"])), str(int(result["variant"])), z, holm])
        lines.append("| " + " | ".join(cells) + " |")
    return _table_output(3, lines)


def doTable3(lns_runs, lns_ablation_runs, lns_fixed_runs):
    """Print Table 3 in Markdown and return its 24 unrounded comparison results.

    Inputs are the three LNS repeated-run DataFrames. Detection and validation
    use only these inputs, which are not modified. The result is indexed by
    (ablation, tLim), with full/variant counts, signed z, raw p and adjusted Holm.
    """
    runs, instances = _prepare_ablation_runs(lns_runs, lns_ablation_runs, lns_fixed_runs)
    samples = {
        key: group[["penalty", "obj"]].to_numpy(dtype=float)
        for key, group in runs.groupby(["variant", "tLim", "Instance"])
    }

    comparisons = []
    for variant in ABLATION_VARIANTS:
        for limit in TIME_LIMITS:
            comparisons.append({
                "ablation": variant,
                "tLim": limit,
                **_comparison_statistics(
                    {instance: samples["Full", limit, instance] for instance in instances},
                    {instance: samples[variant, limit, instance] for instance in instances},
                ),
            })

    results = pd.DataFrame(comparisons).set_index(["ablation", "tLim"])
    results["Holm"] = _holm_adjust(results["p"].to_numpy())
    print(_table3_markdown(results), end="")
    return results


def _pairwise_full_index(full, ablation):
    """Fraction of all cross-method pairs won by Full, with half credit for ties."""
    full = np.asarray(full, dtype=float)
    ablation = np.asarray(ablation, dtype=float)
    full_penalty, full_obj = full[:, 0, None], full[:, 1, None]
    abl_penalty, abl_obj = ablation[None, :, 0], ablation[None, :, 1]
    equal_penalty = full_penalty == abl_penalty
    wins = (full_penalty < abl_penalty) | (equal_penalty & (full_obj < abl_obj))
    ties = equal_penalty & (full_obj == abl_obj)
    return float((wins.sum() + 0.5 * ties.sum()) / wins.size)


def _effect_size_summary(full_samples, ablation_samples):
    """Summarize instance-keyed arrays of (penalty, compactness) outcomes.

    Pairwise indices and RCDs give each instance equal weight. Feasibility
    percentages count runs. Only penalty == 0 runs enter the compactness
    medians, and both methods need at least one such run for RCD and W/T/L.
    Quartiles use linear interpolation (NumPy's default quantile convention).
    """
    indices, rcds = [], []
    full_feasible = ablation_feasible = full_total = ablation_total = 0
    wins = ties = losses = 0
    for instance, full in full_samples.items():
        full = np.asarray(full, dtype=float)
        ablation = np.asarray(ablation_samples[instance], dtype=float)
        indices.append(_pairwise_full_index(full, ablation))
        feasible_full = full[full[:, 0] == 0, 1]
        feasible_ablation = ablation[ablation[:, 0] == 0, 1]
        full_feasible += len(feasible_full)
        ablation_feasible += len(feasible_ablation)
        full_total += len(full)
        ablation_total += len(ablation)
        if len(feasible_full) == 0 or len(feasible_ablation) == 0:
            continue

        ub_full = float(np.median(feasible_full))
        ub_ablation = float(np.median(feasible_ablation))
        if ub_full == 0:
            raise ValueError(f"{instance}: RCD is undefined because Full's feasible median is zero")
        rcds.append(100.0 * (ub_ablation - ub_full) / ub_full)
        wins += ub_full < ub_ablation
        ties += ub_full == ub_ablation
        losses += ub_full > ub_ablation

    q1, median, q3 = (
        np.quantile(rcds, [0.25, 0.5, 0.75], method="linear")
        if rcds else (np.nan, np.nan, np.nan)
    )
    return {
        "pi_full": 100.0 * float(np.mean(indices)),
        "full_feasible": 100.0 * full_feasible / full_total,
        "ablation_feasible": 100.0 * ablation_feasible / ablation_total,
        "rcd_median": float(median),
        "rcd_q1": float(q1),
        "rcd_q3": float(q3),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "eligible_instances": len(rcds),
    }


def _table4_markdown(results):
    """Render effect sizes with two decimals for feasibility and one otherwise."""
    lines = [
        "| Time limit (s) | Ablation | π_F (%) | % Full | % Abl | Med. RCD [Q1, Q3] (%) | W/T/L |",
        "| ---: | :--- | ---: | ---: | ---: | ---: | :--- |",
    ]
    for (limit, ablation), result in results.iterrows():
        rcd = (
            f"{result['rcd_median']:.1f} [{result['rcd_q1']:.1f}, {result['rcd_q3']:.1f}]"
            if result["eligible_instances"] else "—"
        )
        counts = "/".join(str(int(result[key])) for key in ("wins", "ties", "losses"))
        cells = [
            str(limit), ablation, f"{result['pi_full']:.1f}",
            f"{result['full_feasible']:.2f}", f"{result['ablation_feasible']:.2f}",
            rcd, counts,
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return _table_output(4, lines)


def doTable4(lns_runs, lns_ablation_runs, lns_fixed_runs):
    """Print Table 4 in Markdown and return the 24 unrounded effect-size summaries.

    Rows are ordered by time limit (60, 300, 600), then the eight ablations.
    Inputs are not modified. Results are indexed by (tLim, ablation); RCD
    statistics are NaN if no instances have feasible runs from both methods.
    """
    runs, instances = _prepare_ablation_runs(lns_runs, lns_ablation_runs, lns_fixed_runs)
    samples = {
        key: group[["penalty", "obj"]].to_numpy(dtype=float)
        for key, group in runs.groupby(["variant", "tLim", "Instance"])
    }
    comparisons = []
    for limit in TIME_LIMITS:
        full_samples = {instance: samples["Full", limit, instance] for instance in instances}
        for ablation in ABLATION_VARIANTS:
            ablation_samples = {
                instance: samples[ablation, limit, instance] for instance in instances
            }
            comparisons.append({
                "tLim": limit,
                "ablation": ablation,
                **_effect_size_summary(full_samples, ablation_samples),
            })
    results = pd.DataFrame(comparisons).set_index(["tLim", "ablation"])
    print(_table4_markdown(results), end="")
    return results


def _prepare_vns_runs(dataframe, expected, source):
    """Normalize VNS columns and validate repeated runs against the LNS instances."""
    columns = ["Instance", "seed", "Infeasibility", "bestObjective"]
    missing = set(columns).difference(dataframe.columns)
    if missing:
        raise ValueError(f"{source}: missing columns {sorted(missing)}")
    runs = dataframe[columns].rename(
        columns={"Infeasibility": "penalty", "bestObjective": "obj"}
    )
    if not np.isfinite(runs[["penalty", "obj"]].to_numpy(dtype=float)).all():
        raise ValueError(f"{source}: penalty and objective must be finite")
    _validate_run_grid(runs, expected, source)
    return runs


def _prepare_lns_vns_samples(lns_runs, vns_low_runs, vns_medium_runs, vns_high_runs):
    """Validate six methods and group their outcome arrays by method and instance."""
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
    lns_samples = {
        key: group[["penalty", "obj"]].to_numpy(dtype=float)
        for key, group in lns.groupby(["tLim", "Instance"])
    }
    samples = {
        f"LNS{limit}": {instance: lns_samples[limit, instance] for instance in instances}
        for limit in TIME_LIMITS
    }
    for method, dataframe in zip(LNS_VNS_METHODS[3:], (vns_low_runs, vns_medium_runs, vns_high_runs)):
        vns = _prepare_vns_runs(dataframe, expected_vns, method)
        samples[method] = {
            instance: group[["penalty", "obj"]].to_numpy(dtype=float)
            for instance, group in vns.groupby("Instance")
        }
    return samples


def _table5_markdown(results):
    """Render significant-instance counts, feasibility and feasible-only RCD."""
    lines = [
        "| Comparison | LNS | VNS | z | Feasible LNS (%) | Feasible VNS (%) | Median RCD (%) | Q1 (%) | Q3 (%) |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for comparison, result in results.iterrows():
        stars = sum(result["p"] < threshold for threshold in (0.05, 0.01, 0.001))
        z = f"{result['z']:.3f}" + r"\*" * stars
        rcd = [
            f"{result[key]:.1f}" if result["eligible_instances"] else "—"
            for key in ("rcd_median", "rcd_q1", "rcd_q3")
        ]
        cells = [
            comparison, str(int(result["lns"])), str(int(result["vns"])), z,
            f"{result['lns_feasible']:.1f}", f"{result['vns_feasible']:.1f}", *rcd,
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return _table_output(5, lines)


def doTable5(lns_runs, vns_low_runs, vns_medium_runs, vns_high_runs):
    """Print the three LNS--VNS comparisons and return their unrounded results.

    Compare LNS 60/300/600 s with VNS(l)/(m)/(h), respectively. Reuse Table 3's
    rank tests and Table 4's feasible-run medians and RCD summaries. Validate
    120 common instances and seeds 1--30 for every method; inputs are unchanged.
    """
    samples = _prepare_lns_vns_samples(lns_runs, vns_low_runs, vns_medium_runs, vns_high_runs)
    comparisons = []
    for limit, version in ((60, "l"), (300, "m"), (600, "h")):
        reference = samples[f"LNS{limit}"]
        other = samples[f"VNS({version})"]
        statistics = _comparison_statistics(reference, other)
        effects = _effect_size_summary(reference, other)
        comparisons.append({
            "comparison": f"LNS ({limit} s) vs VNS({version})",
            "lns": statistics["full"],
            "vns": statistics["variant"],
            "z": statistics["z"],
            "p": statistics["p"],
            "lns_feasible": effects["full_feasible"],
            "vns_feasible": effects["ablation_feasible"],
            "rcd_median": effects["rcd_median"],
            "rcd_q1": effects["rcd_q1"],
            "rcd_q3": effects["rcd_q3"],
            "eligible_instances": effects["eligible_instances"],
        })
    results = pd.DataFrame(comparisons).set_index("comparison")
    print(_table5_markdown(results), end="")
    return results


def _van_elteren_matrix(samples):
    """Compute each of the 15 pairs once; positive z favors the row method."""
    matrix = pd.DataFrame(np.nan, index=LNS_VNS_METHODS, columns=LNS_VNS_METHODS)
    for position, row_method in enumerate(LNS_VNS_METHODS):
        for column_method in LNS_VNS_METHODS[position + 1:]:
            strata = [
                _compare_instance(row_outcomes, samples[column_method][instance])
                for instance, row_outcomes in samples[row_method].items()
            ]
            z, _ = _van_elteren(strata)
            matrix.loc[row_method, column_method] = z
            matrix.loc[column_method, row_method] = -z if z != 0 else 0.0
    return matrix


def doTable6(lns_runs, vns_low_runs, vns_medium_runs, vns_high_runs):
    """Return a Markdown matrix of pairwise Van Elteren z statistics.

    Validate 120 common instances and 30 independent runs per method/instance.
    Rank all outcomes by lower penalty, then lower compactness. Instances are
    strata; use tie-corrected variances and weights 1/(n_row + n_column + 1),
    without continuity correction, as in Tables 3 and 5. Positive z favors the
    row method. Self-comparisons are dashes; distinct methods that tie have z=0.
    """
    samples = _prepare_lns_vns_samples(lns_runs, vns_low_runs, vns_medium_runs, vns_high_runs)
    matrix = _van_elteren_matrix(samples)
    lines = [
        "| Method | " + " | ".join(LNS_VNS_METHODS) + " |",
        "| :--- | " + " | ".join(["---:"] * len(LNS_VNS_METHODS)) + " |",
    ]
    for row_method in LNS_VNS_METHODS:
        cells = [
            "—" if row_method == column_method else f"{matrix.loc[row_method, column_method]:.3f}"
            for column_method in LNS_VNS_METHODS
        ]
        lines.append("| " + " | ".join([row_method, *cells]) + " |")
    return _table_output(6, lines)


def _parse_single_instance_names(instances):
    """Extract graph, BU count, attribute levels and replicate from instance names."""
    names = pd.Series(list(instances), dtype="string")
    characteristics = names.str.extract(
        r"^d-(?P<Demand>[lmh])_w-(?P<Workload>[lmh])_c-(?P<Customer>[lmh])-"
        r"(?P<Graph>Center|Corners|Diagonal)(?P<BU>486|600|726)_G(?P<replicate>[0-9])\Z"
    )
    invalid = characteristics.isna().any(axis=1)
    if invalid.any():
        raise ValueError(f"Invalid single-run instance names: {names[invalid].head(3).tolist()}")
    for attribute in ("Demand", "Workload", "Customer"):
        characteristics[attribute] = characteristics[attribute].map(
            {"l": "Low", "m": "Medium", "h": "High"}
        )
    characteristics = characteristics.rename(columns={"BU": "#BU"})
    characteristics["#BU"] = characteristics["#BU"].astype(int)
    characteristics["replicate"] = characteristics["replicate"].astype(int)
    characteristics.index = pd.Index(names, name="Instance")
    return characteristics


def _single_result_feasibility(dataframe, penalty_column, expected, source):
    """Validate non-run result coverage and compute a zero-penalty indicator."""
    columns = [*expected.names, penalty_column]
    missing = set(columns).difference(dataframe.columns)
    if missing:
        raise ValueError(f"{source}: missing columns {sorted(missing)}")
    results = dataframe[columns].copy()
    _validate_run_grid(results, expected, source)
    penalties = results[penalty_column].to_numpy(dtype=float)
    if not np.isfinite(penalties).all() or np.any(penalties < 0):
        raise ValueError(f"{source}: balance penalties must be finite and nonnegative")
    results["feasible"] = penalties == 0
    return results


def _single_instance_design(lns):
    """Identify the 2,430-instance design and its expected LNS/VNS result keys."""
    if "Instance" not in lns.columns or lns["Instance"].isna().any():
        raise ValueError("LNS non-run results require nonmissing Instance names")
    instances = sorted(lns["Instance"].unique())
    if len(instances) != 2430:
        raise ValueError(f"Expected 2430 non-run instances, found {len(instances)}")
    # The bounded name pattern admits exactly 3^3 * 3 * 3 * 10 = 2,430 names.
    # Thus 2,430 distinct valid names cover the entire characteristic design.
    characteristics = _parse_single_instance_names(instances)
    expected_lns = pd.MultiIndex.from_product(
        [instances, TIME_LIMITS], names=["Instance", "tLim"]
    )
    expected_vns = pd.MultiIndex.from_product([instances], names=["Instance"])
    return characteristics, expected_lns, expected_vns


def _prepare_single_results(lns, vns_low, vns_medium, vns_high):
    """Align six methods on the same 2,430 fully specified non-run instances."""
    characteristics, expected_lns, expected_vns = _single_instance_design(lns)
    lns_results = _single_result_feasibility(lns, "penalty", expected_lns, "LNS non-run")
    for method, dataframe in zip(SINGLE_RESULT_METHODS[:3], (vns_low, vns_medium, vns_high)):
        results = _single_result_feasibility(dataframe, "Infeasibility", expected_vns, method)
        characteristics[method] = results.set_index("Instance")["feasible"]
    for limit, method in zip(TIME_LIMITS, SINGLE_RESULT_METHODS[3:]):
        results = lns_results.loc[lns_results["tLim"].eq(limit)]
        characteristics[method] = results.set_index("Instance")["feasible"]
    return characteristics


def _feasibility_panel_markdown(panel, title):
    """Render floored feasibility percentages, omitting decimals for exact 100%."""
    headers = [*panel.index.names, *panel.columns]
    alignments = ["---:" if name == "#BU" else ":---" for name in panel.index.names]
    lines = [
        f"**{title}**", "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join([*alignments, *(["---:"] * len(panel.columns))]) + " |",
    ]
    for group, row in panel.iterrows():
        cells = [
            *(str(value) for value in group),
            *("100" if value == 100 else f"{value:.1f}" for value in row),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def doTable7(lns, vns_low, vns_medium, vns_high):
    """Return two Markdown panels of feasibility for the 2,430 non-run instances.

    Panel A groups by graph and BU count; Panel B by the joint demand/workload/
    customer levels, ordered Low, Medium, High. Each percentage uses that
    method's zero-penalty instances divided by all instances in the group.
    Panel B ends with a Total row calculated directly over all 2,430 instances.
    Percentages are rounded down to one decimal using integer counts, with
    exact 100% values shown without decimals, as specified in the paper.
    """
    results = _prepare_single_results(lns, vns_low, vns_medium, vns_high)
    methods = list(SINGLE_RESULT_METHODS)
    graph_groups = pd.MultiIndex.from_product(
        [INSTANCE_GRAPHS, INSTANCE_BU_SIZES], names=["Graph", "#BU"]
    )
    attribute_groups = pd.MultiIndex.from_product(
        [ATTRIBUTE_LEVELS] * 3, names=["Demand", "Workload", "Customer"]
    )
    def grouped_percentages(keys, order):
        grouped = results.groupby(keys)[methods]
        counts = grouped.sum().reindex(order)
        sizes = grouped.size().reindex(order)
        # Integer division floors tenths of a percentage point exactly, avoiding
        # binary floating-point errors at exact decimal boundaries.
        return (1000 * counts).floordiv(sizes, axis=0) / 10

    panel_a = grouped_percentages(["Graph", "#BU"], graph_groups)
    panel_b = grouped_percentages(["Demand", "Workload", "Customer"], attribute_groups)
    total = pd.DataFrame(
        [(1000 * results[methods].sum() // len(results)) / 10],
        index=pd.MultiIndex.from_tuples([("Total", "", "")], names=attribute_groups.names),
    )
    panel_b = pd.concat([panel_b, total])
    return _table_output(7, [
        _feasibility_panel_markdown(panel_a, "Panel A"),
        "",
        _feasibility_panel_markdown(panel_b, "Panel B"),
    ])


def _single_result_compactness(dataframe, objective_column, expected, source):
    """Validate non-run result keys and positive compactness values, ignoring penalties."""
    columns = [*expected.names, objective_column]
    missing = set(columns).difference(dataframe.columns)
    if missing:
        raise ValueError(f"{source}: missing columns {sorted(missing)}")
    results = dataframe[columns].rename(columns={objective_column: "compactness"}).copy()
    _validate_run_grid(results, expected, source)
    values = results["compactness"].to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError(f"{source}: compactness values must be finite and positive")
    results["compactness"] = values
    return results


def _prepare_single_compactness(lns, vns_low, vns_medium, vns_high):
    """Align all six compactness results per instance, with LNS columns first."""
    characteristics, expected_lns, expected_vns = _single_instance_design(lns)
    compactness = pd.DataFrame(index=characteristics.index)
    lns_results = _single_result_compactness(lns, "obj", expected_lns, "LNS non-run")
    for limit, method in zip(TIME_LIMITS, SINGLE_RESULT_METHODS[3:]):
        results = lns_results.loc[lns_results["tLim"].eq(limit)]
        compactness[method] = results.set_index("Instance")["compactness"]
    for method, dataframe in zip(SINGLE_RESULT_METHODS[:3], (vns_low, vns_medium, vns_high)):
        results = _single_result_compactness(dataframe, "bestObjective", expected_vns, method)
        compactness[method] = results.set_index("Instance")["compactness"]
    return characteristics, compactness


def _compactness_gap_summary(characteristics, compactness):
    """Average per-instance gaps and count all best attainments, including ties."""
    best = compactness.min(axis=1)
    # Normalize each method's excess compactness by the best value on that instance.
    gaps = 100 * compactness.sub(best, axis=0).div(best, axis=0)
    attainments = compactness.eq(best, axis=0)
    groups = [characteristics["Graph"].rename("Layout"), characteristics["#BU"].rename("BUs")]
    group_order = pd.MultiIndex.from_product(
        [INSTANCE_GRAPHS, INSTANCE_BU_SIZES], names=["Layout", "BUs"]
    )
    mean_gaps = gaps.groupby(groups).mean().reindex(group_order)
    best_counts = attainments.groupby(groups).sum().reindex(group_order)
    total_index = pd.MultiIndex.from_tuples([("Total", "")], names=group_order.names)
    mean_gaps = pd.concat([mean_gaps, pd.DataFrame([gaps.mean()], index=total_index)])
    best_counts = pd.concat([best_counts, pd.DataFrame([attainments.sum()], index=total_index)])
    return mean_gaps, best_counts


def doTable8(lns, vns_low, vns_medium, vns_high):
    """Return mean compactness gaps and best-attainment counts in Markdown.

    For each of the 2,430 non-run instances, best is the minimum compactness
    among all six configurations, regardless of balance feasibility. A method's
    gap is 100 * (UB - best) / best. Average these gaps within each layout/BU group
    and over all instances. Every configuration tied at best receives a count.
    """
    characteristics, compactness = _prepare_single_compactness(lns, vns_low, vns_medium, vns_high)
    mean_gaps, best_counts = _compactness_gap_summary(characteristics, compactness)
    methods = list(compactness.columns)
    lines = [
        "| Layout | BUs | " + " | ".join(methods) + " |",
        "| :--- | ---: | " + " | ".join(["---:"] * len(methods)) + " |",
    ]
    for (layout, size), row in mean_gaps.iterrows():
        cells = [
            f"{row[method]:.2f} ({int(best_counts.loc[(layout, size), method])})"
            for method in methods
        ]
        lines.append("| " + " | ".join([layout, str(size), *cells]) + " |")
    return _table_output(8, lines)


def main():
    """Read the experiment results and return a named DataFrame for each file."""
    root = Path(__file__).resolve().parent.parent
    vns_dir = root / "AlyEtAl"
    lns_dir = root / "resultsLNS"

    dataframes = {
        "vns_high_runs": pd.read_csv(vns_dir / "vns.high.results.runs.txt"),
        "vns_medium_runs": pd.read_csv(vns_dir / "vns.medium.results.runs.txt"),
        "vns_low_runs": pd.read_csv(vns_dir / "vns.low.results.runs.txt"),
        "vns_high": pd.read_csv(vns_dir / "vns.high.results.txt"),
        "vns_medium": pd.read_csv(vns_dir / "vns.medium.results.txt"),
        "vns_low": pd.read_csv(vns_dir / "vns.low.results.txt"),
        "lns_ablation_runs": pd.read_csv(lns_dir / "lns.ablation.results.runs.txt"),
        "lns_runs": pd.read_csv(lns_dir / "lns.results.runs.txt"),
        "lower_bounds": pd.read_csv(lns_dir / "lower.bounds.results.txt"),
        "lns_fixed_runs": pd.read_csv(lns_dir / "lns.fixed.results.runs.txt"),
        "lns": pd.read_csv(lns_dir / "lns.results.txt"),
    }
    return dataframes


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
    doTable3(*ablation_inputs)
    doTable4(*ablation_inputs)
    doTable5(*run_inputs)
    print(doTable6(*run_inputs), end="")
    print(doTable7(*single_inputs), end="")
    print(doTable8(*single_inputs), end="")
