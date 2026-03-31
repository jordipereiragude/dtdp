from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
from matplotlib import transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.stats.multitest import multipletests


FEASIBILITY_TOL = 1e-6
FIG1_OUTPUT = Path(__file__).resolve().with_name("fig-lns600-comparison.pdf")
ALGORITHM_COLUMNS = [
    ("LNS60", "LNS (60s)"),
    ("LNS300", "LNS (300s)"),
    ("LNS600", "LNS (600s)"),
    ("VNSl", "VNS(l)"),
    ("VNSm", "VNS(m)"),
    ("VNSh", "VNS(h)"),
]
CMP_ALGORITHMS = [alg for alg, _ in ALGORITHM_COLUMNS]
VNS_ALGORITHMS = {"VNSl", "VNSm", "VNSh"}
METHOD_ORDER = ["LNS60", "LNS300", "LNS600", "VNSl", "VNSm", "VNSh"]
METHOD_LABELS = dict(ALGORITHM_COLUMNS)
LEVEL_MAP = {"l": "25%", "m": "50%", "h": "90%"}
FACTOR_COLUMNS = ["method", "size", "demand", "workload", "customers", "type"]
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
    r"^Instance:\s+(?P<instance>\S+)\s+"
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


def build_group_rows(
    counts: dict[tuple[str, str, int, str], object],
    instance_counts: dict[tuple[str, str, int], object],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group, instance_type, size, display_type in GROUP12_ROW_ORDER:
        row = {
            "type": display_type,
            "size": size,
            "instances": int(instance_counts.get((group, instance_type, size), 0)),
        }
        for alg_key, column_name in ALGORITHM_COLUMNS:
            row[column_name] = counts.get((group, instance_type, size, alg_key), 0)
        rows.append(row)
    return pd.DataFrame(rows)


def build_group12_success_table(dataframe: pd.DataFrame) -> pd.DataFrame:
    enriched = dataframe.copy()
    enriched["group"] = infer_group(enriched)
    filtered = enriched[
        enriched["alg"].isin(CMP_ALGORITHMS)
        & (enriched["penalty"].fillna(float("inf")) < FEASIBILITY_TOL)
    ].copy()
    counts = filtered.groupby(["group", "type", "size", "alg"], dropna=False).size().to_dict()
    instance_counts = enriched.groupby(["group", "type", "size"], dropna=False)["name"].nunique().to_dict()
    return build_group_rows(counts, instance_counts)


def build_all_feasible_subset(dataframe: pd.DataFrame) -> pd.DataFrame:
    enriched = dataframe.copy()
    enriched["group"] = infer_group(enriched)
    filtered = enriched[enriched["alg"].isin(CMP_ALGORITHMS)].copy()
    filtered = filtered[filtered["penalty"].fillna(float("inf")) < FEASIBILITY_TOL]
    feasible_counts = filtered.groupby("name")["alg"].nunique()
    eligible_names = feasible_counts[feasible_counts == len(CMP_ALGORITHMS)].index
    return filtered[filtered["name"].isin(eligible_names)].copy()


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
        lambda row: ((row["obj"] - best_by_instance[row["name"]]) / row["obj"]) * 100
        if abs(row["obj"]) >= FEASIBILITY_TOL
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
            row[column_name] = f"{avg_gap:.2f} ({wins})"
        rows.append(row)
    return pd.DataFrame(rows)


def build_group12_time_table(dataframe: pd.DataFrame) -> pd.DataFrame:
    enriched = dataframe.copy()
    enriched["group"] = infer_group(enriched)
    filtered = enriched[enriched["alg"].isin(CMP_ALGORITHMS)].copy()
    filtered["time_metric"] = filtered.apply(
        lambda row: row["totalTime"] if row["alg"] in VNS_ALGORITHMS else row["timeBest"],
        axis=1,
    )
    time_means = filtered.groupby(["group", "type", "size", "alg"], dropna=False)["time_metric"].mean().to_dict()
    instance_counts = filtered.groupby(["group", "type", "size"], dropna=False)["name"].nunique().to_dict()
    return build_group_rows(time_means, instance_counts)


def build_group12_lns_vs_vns_best_table(dataframe: pd.DataFrame) -> pd.DataFrame:
    filtered = build_all_instances_subset(dataframe)
    metadata = filtered.groupby("name", dropna=False)[["group", "type", "size"]].first().reset_index()

    lns_best = (
        filtered[filtered["alg"].isin({"LNS60", "LNS300", "LNS600"})]
        .groupby("name", dropna=False)["obj"]
        .min()
        .rename("lns_best_obj")
    )
    vns_rows = filtered[filtered["alg"].isin(VNS_ALGORITHMS)].copy()
    vns_best = vns_rows.groupby("name", dropna=False)["obj"].min().rename("vns_best_obj")
    vns_rows = vns_rows.merge(vns_best, on="name", how="left")
    vns_rows["matches_vns_best"] = (vns_rows["obj"] - vns_rows["vns_best_obj"]).abs() < FEASIBILITY_TOL
    vns_best_infeasible = (
        vns_rows[vns_rows["matches_vns_best"]]
        .groupby("name", dropna=False)["penalty"]
        .apply(lambda penalties: bool((penalties.fillna(0.0) > 0.0).any()))
        .rename("vns_best_infeasible")
    )

    comparison = metadata.merge(lns_best, on="name", how="left").merge(vns_best, on="name", how="left")
    comparison = comparison.merge(vns_best_infeasible, on="name", how="left")
    comparison["vns_best_infeasible"] = comparison["vns_best_infeasible"].fillna(False)
    comparison["lns_best"] = comparison["lns_best_obj"] + FEASIBILITY_TOL < comparison["vns_best_obj"]
    comparison["vns_best"] = comparison["vns_best_obj"] + FEASIBILITY_TOL < comparison["lns_best_obj"]
    comparison["tie"] = (comparison["lns_best_obj"] - comparison["vns_best_obj"]).abs() < FEASIBILITY_TOL
    comparison["vns_best_infeasible_win"] = comparison["vns_best"] & comparison["vns_best_infeasible"]

    counts = (
        comparison.groupby(["group", "type", "size"], dropna=False)[
            ["lns_best", "vns_best", "tie", "vns_best_infeasible_win"]
        ]
        .sum()
        .to_dict("index")
    )
    instance_counts = comparison.groupby(["group", "type", "size"], dropna=False)["name"].nunique().to_dict()

    rows: list[dict[str, object]] = []
    for group, instance_type, size, display_type in GROUP12_ROW_ORDER:
        group_counts = counts.get((group, instance_type, size), {})
        rows.append(
            {
                "type": display_type,
                "size": size,
                "instances": int(instance_counts.get((group, instance_type, size), 0)),
                "LNS best": int(group_counts.get("lns_best", 0)),
                "VNS best": int(group_counts.get("vns_best", 0)),
                "Tie": int(group_counts.get("tie", 0)),
                "VNS best infeasible": int(group_counts.get("vns_best_infeasible_win", 0)),
            }
        )
    return pd.DataFrame(rows)


def build_group3_plot_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    filtered = dataframe[
        dataframe["alg"].isin(CMP_ALGORITHMS)
        & dataframe["dValue"].notna()
        & dataframe["wValue"].notna()
        & dataframe["cValue"].notna()
        & dataframe["obj"].notna()
    ].copy()
    filtered["Demand"] = filtered["dValue"].map(LEVEL_MAP)
    filtered["Workload"] = filtered["wValue"].map(LEVEL_MAP)
    filtered["Customers"] = filtered["cValue"].map(LEVEL_MAP)
    filtered["SizeLabel"] = filtered["size"].astype(int).astype(str)
    filtered["Algorithm"] = filtered["alg"].map(
        {
            "LNS60": "LNS 60s",
            "LNS300": "LNS 300s",
            "LNS600": "LNS 600s",
            "VNSl": "VNS(l)",
            "VNSm": "VNS(m)",
            "VNSh": "VNS(h)",
        }
    )
    return filtered


def build_group3_avg_obj_table(dataframe: pd.DataFrame) -> pd.DataFrame:
    plot_df = build_group3_plot_dataframe(dataframe)
    means = (
        plot_df.groupby(["Demand", "Workload", "Customers", "Algorithm"], dropna=False)["obj"]
        .mean()
        .to_dict()
    )
    algorithm_labels = ["LNS 60s", "LNS 300s", "LNS 600s", "VNS(l)", "VNS(m)", "VNS(h)"]

    rows: list[dict[str, object]] = []
    for demand in ["25%", "50%", "90%"]:
        for workload in ["25%", "50%", "90%"]:
            for customers in ["25%", "50%", "90%"]:
                row = {"Demand": demand, "Workload": workload, "Customers": customers}
                for algorithm in algorithm_labels:
                    row[algorithm] = means.get((demand, workload, customers, algorithm), 0.0)
                rows.append(row)
    return pd.DataFrame(rows)


def plot_group3_lns600_comparison(dataframe: pd.DataFrame, output_path: Path) -> Path:
    plot_df = build_group3_plot_dataframe(dataframe)
    plot_df = plot_df[plot_df["Algorithm"] == "LNS 600s"].copy()
    demand_levels = ["25%", "50%", "90%"]
    workload_levels = ["25%", "50%", "90%"]
    customer_levels = ["25%", "50%", "90%"]
    size_levels = ["486", "600", "726"]
    combo_labels = [
        f"W({workload}) C({customers})"
        for workload in workload_levels
        for customers in customer_levels
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    (ROOT / ".matplotlib").mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        nrows=3,
        ncols=3,
        figsize=(16, 10.5),
        sharey="row",
        constrained_layout=True,
    )

    line_color = "#303030"
    workload_facecolors = {"25%": "#f0f0f0", "50%": "#bdbdbd", "90%": "#7a7a7a"}
    customer_hatches = {"25%": "", "50%": "///", "90%": "xx"}

    def outlier_cap(values_by_group: list[list[float]]) -> float:
        flat = [value for values in values_by_group for value in values]
        if not flat:
            return 1.0
        flat_arr = np.array(flat, dtype=float)
        q1, q3 = np.percentile(flat_arr, [25, 75])
        iqr = max(q3 - q1, 1e-9)
        whisker_cap = q3 + 1.5 * iqr
        p95 = np.percentile(flat_arr, 95)
        upper = min(float(flat_arr.max()), max(float(p95), float(whisker_cap)))
        if upper <= float(flat_arr.min()):
            upper = float(flat_arr.max())
        return upper

    def upper_whisker(values: list[float]) -> float:
        if not values:
            return 0.0
        arr = np.array(values, dtype=float)
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = max(q3 - q1, 1e-9)
        high_bound = float(q3 + 1.5 * iqr)
        within = arr[arr <= high_bound]
        if within.size == 0:
            return float(arr.min())
        return float(within.max())

    for row_idx, size in enumerate(size_levels):
        row_ylim_bottom: float | None = None
        row_ylim_upper: float | None = None
        for col_idx, demand in enumerate(demand_levels):
            ax = axes[row_idx, col_idx]
            subset = plot_df[(plot_df["SizeLabel"] == size) & (plot_df["Demand"] == demand)]
            data = [
                subset.loc[
                    (subset["Workload"] == workload) & (subset["Customers"] == customers),
                    "obj",
                ].dropna().tolist()
                for workload in workload_levels
                for customers in customer_levels
            ]
            combo_specs = [
                (workload, customers)
                for workload in workload_levels
                for customers in customer_levels
            ]
            positions = list(range(1, len(data) + 1))
            nonempty = [value for values in data for value in values]
            ylim_top = outlier_cap(data)
            ymin = min(nonempty) if nonempty else 0.0
            pad = max((ylim_top - ymin) * 0.08, 1.0)

            bp = ax.boxplot(
                data,
                positions=positions,
                widths=0.6,
                patch_artist=True,
                manage_ticks=False,
                showfliers=True,
                boxprops={"edgecolor": line_color, "linewidth": 0.9},
                medianprops={"color": "#111111", "linewidth": 1.1},
                whiskerprops={"color": line_color, "linewidth": 0.9},
                capprops={"color": line_color, "linewidth": 0.9},
                flierprops={
                    "marker": "o",
                    "markersize": 2.0,
                    "markerfacecolor": "#111111",
                    "markeredgecolor": "#111111",
                    "alpha": 0.65,
                },
            )
            for box, (workload, customers) in zip(bp["boxes"], combo_specs):
                box.set_facecolor(workload_facecolors[workload])
                box.set_hatch(customer_hatches[customers])

            if nonempty:
                whisker_top = max(upper_whisker(values) for values in data if values)
                visible_top = max(ylim_top, whisker_top)
                ylim_bottom = ymin - pad * 0.55
                top_pad_factor = 0.55 if size == "486" else 0.28
                ylim_upper = visible_top + pad * top_pad_factor
                if size == "486" and demand == "50%":
                    ylim_upper = max(ylim_upper, whisker_top + max(pad * 1.4, 10.0), 276.0)
                row_ylim_bottom = ylim_bottom if row_ylim_bottom is None else min(row_ylim_bottom, ylim_bottom)
                row_ylim_upper = ylim_upper if row_ylim_upper is None else max(row_ylim_upper, ylim_upper)

            ax.grid(axis="y", color="#e0e0e0", linewidth=0.6)
            ax.set_axisbelow(True)
            ax.set_xticks(positions)
            ax.set_xticklabels(combo_labels, rotation=45, ha="right", fontsize=7)
            ax.tick_params(axis="y", labelsize=8)
            if col_idx == 0:
                ax.set_ylabel(f"{size} BUs\nObjective", fontsize=9)
            if row_idx == 0:
                ax.set_title(f"Demand {demand}", fontsize=10, pad=6)
            if row_idx == len(size_levels) - 1:
                ax.set_xlabel("Workload (W) Customers (C) combinations", fontsize=9)

            if nonempty:
                top_band_transform = transforms.blended_transform_factory(ax.transData, ax.transAxes)
                for pos, values in zip(positions, data):
                    hidden_outliers = [value for value in values if value > visible_top]
                    if hidden_outliers:
                        ax.text(
                            pos,
                            1.0,
                            str(len(hidden_outliers)),
                            ha="center",
                            va="center",
                            fontsize=7,
                            color="#111111",
                            transform=top_band_transform,
                            clip_on=False,
                            bbox={
                                "boxstyle": "round,pad=0.12",
                                "facecolor": "white",
                                "edgecolor": "none",
                                "alpha": 0.92,
                            },
                            zorder=5,
                        )

        if row_ylim_bottom is not None and row_ylim_upper is not None:
            for ax in axes[row_idx, :]:
                ax.set_ylim(row_ylim_bottom, row_ylim_upper)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def group12_success_pct_table_to_latex(dataframe: pd.DataFrame) -> str:
    table = build_group12_success_table(dataframe)
    metric_columns = [column_name for _, column_name in ALGORITHM_COLUMNS]
    lines = [
        r"\begin{table}",
        r"\caption{For each instance group, the table reports the percentage of instances for which each method finds a feasible solution with respect to the balance constraints. Results are shown for LNS with time limits of 60s, 300s, and 600s, and for the VNS low, medium, and high configurations, denoted by VNS(l), VNS(m), and VNS(h).}",
        r"\label{tab:feasibility}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Instance group & LNS (60s) & LNS (300s) & LNS (600s) & VNS(l) & VNS(m) & VNS(h) \\",
        r"\midrule",
    ]

    def format_percentage(value: float, decimals: int) -> str:
        if abs(value - 100.0) < 5e-12:
            return r"100\%"
        return f"{value:.{decimals}f}\\%"

    for _, row in table.iterrows():
        instance_label = f"{row['type']}{int(row['size'])}"
        decimals = 2 if "'" in str(row["type"]) else 0
        entries = []
        for column in metric_columns:
            percentage = 0.0
            if row["instances"]:
                percentage = row[column] / row["instances"] * 100
            entries.append(format_percentage(percentage, decimals))
        lines.append(f"{instance_label} & " + " & ".join(entries) + r" \\")

    total_instances = table["instances"].sum()
    total_entries = []
    for column in metric_columns:
        percentage = 0.0
        if total_instances:
            percentage = table[column].sum() / total_instances * 100
        total_entries.append(format_percentage(percentage, 2))
    lines.extend(
        [
            r"\midrule",
            "Total & " + " & ".join(total_entries) + r" \\",
            r"\bottomrule",
            r"\end{tabular}}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def group12_obj_gap_full_with_wins_table_to_latex(dataframe: pd.DataFrame) -> str:
    table = build_group12_obj_gap_full_with_wins_table(dataframe)
    metric_columns = [column_name for _, column_name in ALGORITHM_COLUMNS]
    column_spec = "l" + "r" * (2 + len(metric_columns))
    filtered = build_all_instances_subset(dataframe)
    best_by_instance = filtered.groupby("name")["obj"].min().to_dict()
    filtered["gap"] = filtered.apply(
        lambda row: ((row["obj"] - best_by_instance[row["name"]]) / row["obj"]) * 100
        if abs(row["obj"]) >= FEASIBILITY_TOL
        else 0.0,
        axis=1,
    )
    filtered["is_best"] = filtered.apply(
        lambda row: abs(row["obj"] - best_by_instance[row["name"]]) < FEASIBILITY_TOL,
        axis=1,
    )

    lines = [
        r"\begin{table}",
        r"\caption{Average percentage gap with respect to the best objective found on all instances, with the number of best found solutions in parenthesis. The final row reports the overall averages and total best found solutions across all instances.}",
        r"\label{tab:objgap-full-wins}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        rf"Instance group & \# & {' & '.join(metric_columns)} \\",
        r"\midrule",
    ]

    for _, row in table.iterrows():
        instance_label = f"{row['type']}{int(row['size'])}"
        lines.append(
            f"{instance_label} & {int(row['instances'])} & "
            + " & ".join(str(row[column_name]) for column_name in metric_columns)
            + r" \\"
        )

    total_instances = filtered["name"].nunique()
    total_entries = []
    for alg_key, column_name in ALGORITHM_COLUMNS:
        subset = filtered[filtered["alg"] == alg_key]
        total_entries.append(f"{subset['gap'].mean():.2f} ({int(subset['is_best'].sum())})")

    lines.extend(
        [
            r"\midrule",
            f"Total & {total_instances} & " + " & ".join(total_entries) + r" \\",
            r"\bottomrule",
            r"\end{tabular}}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def group12_time_table_to_latex(dataframe: pd.DataFrame) -> str:
    table = build_group12_time_table(dataframe)
    metric_columns = [column_name for _, column_name in ALGORITHM_COLUMNS]
    column_spec = "l" + "r" * (1 + len(metric_columns))
    lines = [
        r"\begin{table}",
        r"\caption{Average running time in secods. For the LNS columns, the table reports the average time required to reach the best found solution, while for the tables for VNS report the total time required by the method. The final row reports the overall average across all instances.}",
        r"\label{tab:time}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        rf"Instance group & {' & '.join(metric_columns)} \\",
        r"\midrule",
    ]

    for _, row in table.iterrows():
        instance_label = f"{row['type']}{int(row['size'])}"
        entries = [f"{row[column_name]:.2f}" for column_name in metric_columns]
        lines.append(f"{instance_label} & " + " & ".join(entries) + r" \\")

    total_instances = int(table["instances"].sum())
    total_entries = []
    for column_name in metric_columns:
        total_value = (table[column_name] * table["instances"]).sum() / total_instances
        total_entries.append(f"{total_value:.2f}")
    lines.extend(
        [
            r"\midrule",
            "Total & " + " & ".join(total_entries) + r" \\",
            r"\bottomrule",
            r"\end{tabular}}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def group3_avg_obj_table_to_latex(dataframe: pd.DataFrame) -> str:
    table = build_group3_avg_obj_table(dataframe)
    method_columns = ["LNS 60s", "LNS 300s", "LNS 600s", "VNS(l)", "VNS(m)", "VNS(h)"]
    lines = [
        r"\begin{table}",
        r"\caption{Average objective value for the third family of instances for each combination of low-value shares in demand, workload, and customers.}",
        r"\label{tab:group3-avg-obj}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lllrrrrrr}",
        r"\toprule",
        r"Demand & Workload & Customers & LNS (60s) & LNS (300s) & LNS (600s) & VNS(l) & VNS(m) & VNS(h) \\",
        r"\midrule",
    ]

    previous_demand = None
    previous_workload = None
    demand_row_index = 0
    workload_row_index = 0
    for _, row in table.iterrows():
        if previous_demand is not None:
            if row["Demand"] != previous_demand:
                lines.append(r"\hline")
                demand_row_index = 0
                workload_row_index = 0
            elif row["Workload"] != previous_workload:
                lines.append(r"\cline{2-9}")
                workload_row_index = 0
        entries = [f"{row[column]:.2f}" for column in method_columns]
        demand = rf"\multirow{{9}}{{*}}{{{row['Demand']}}}" if demand_row_index == 0 else ""
        workload = rf"\multirow{{3}}{{*}}{{{row['Workload']}}}" if workload_row_index == 0 else ""
        lines.append(
            f"{demand} & {workload} & {row['Customers']} & " + " & ".join(entries) + r" \\"
        )
        previous_demand = row["Demand"]
        previous_workload = row["Workload"]
        demand_row_index += 1
        workload_row_index += 1

    aggregate_entries = [f"{table[column].mean():.2f}" for column in method_columns]
    lines.extend(
        [
            r"\midrule",
            r"Aggregate &  &  & " + " & ".join(aggregate_entries) + r" \\",
            r"\bottomrule",
            r"\end{tabular}}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def build_comment_table(dataframe: pd.DataFrame) -> pd.DataFrame:
    table = build_group12_lns_vs_vns_best_table(dataframe)
    total_instances = int(table["instances"].sum())
    total_lns_best = int(table["LNS best"].sum())
    total_vns_best = int(table["VNS best"].sum())
    total_tie = int(table["Tie"].sum())
    total_vns_best_infeasible = int(table["VNS best infeasible"].sum())
    return pd.DataFrame(
        [
            {
                "Instances": total_instances,
                "Tie": total_tie,
                "LNS best": total_lns_best,
                "VNS best": total_vns_best,
                "VNS best infeasible": total_vns_best_infeasible,
            }
        ]
    )


def build_anova_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    filtered = dataframe[
        dataframe["dValue"].notna()
        & dataframe["wValue"].notna()
        & dataframe["cValue"].notna()
        & dataframe["obj"].notna()
        & dataframe["alg"].isin(METHOD_ORDER)
    ].copy()
    filtered["instance"] = filtered["name"]
    filtered["method"] = pd.Categorical(filtered["alg"], categories=METHOD_ORDER, ordered=True)
    filtered["size"] = pd.Categorical(
        filtered["size"].astype(int).astype(str),
        categories=["486", "600", "726"],
        ordered=True,
    )
    filtered["demand"] = pd.Categorical(
        filtered["dValue"].map(LEVEL_MAP),
        categories=["25%", "50%", "90%"],
        ordered=True,
    )
    filtered["workload"] = pd.Categorical(
        filtered["wValue"].map(LEVEL_MAP),
        categories=["25%", "50%", "90%"],
        ordered=True,
    )
    filtered["customers"] = pd.Categorical(
        filtered["cValue"].map(LEVEL_MAP),
        categories=["25%", "50%", "90%"],
        ordered=True,
    )
    filtered["type"] = pd.Categorical(
        filtered["type"],
        categories=["Center", "Corners", "Diagonal"],
        ordered=True,
    )
    return filtered[
        ["instance", "obj", "penalty", "method", "size", "demand", "workload", "customers", "type"]
    ].copy()


def validate_repeated_measures(dataframe: pd.DataFrame) -> None:
    duplicate_count = int(dataframe.duplicated(["instance", "method"]).sum())
    if duplicate_count:
        raise ValueError(
            f"Found {duplicate_count} duplicate rows for the same instance-method combination."
        )
    method_counts = dataframe.groupby("instance")["method"].nunique()
    incomplete = method_counts[method_counts != len(METHOD_ORDER)]
    if not incomplete.empty:
        preview = ", ".join(
            f"{instance}({count})" for instance, count in incomplete.head(10).items()
        )
        raise ValueError(
            "Repeated-measures structure is incomplete. "
            f"Expected 6 methods per instance, found mismatches such as: {preview}"
        )


def add_response_column(dataframe: pd.DataFrame, log_response: bool) -> tuple[pd.DataFrame, str]:
    frame = dataframe.copy()
    if log_response:
        if (frame["obj"] <= 0).any():
            raise ValueError("Cannot log-transform the response because some objective values are <= 0.")
        frame["response"] = np.log(frame["obj"])
        return frame, "log(obj)"
    frame["response"] = frame["obj"].astype(float)
    return frame, "obj"


def fit_mixed_model(dataframe: pd.DataFrame):
    formula = (
        "response ~ C(method) + C(size) + C(demand) + C(workload) + "
        "C(customers) + C(type)"
    )
    model = smf.mixedlm(formula, data=dataframe, groups=dataframe["instance"], re_formula="1")
    last_error: Exception | None = None
    for fit_kwargs in (
        {"method": "lbfgs", "reml": False, "disp": False},
        {"method": "powell", "reml": False, "disp": False},
        {"method": "nm", "reml": False, "disp": False},
    ):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                warnings.simplefilter("ignore", ConvergenceWarning)
                result = model.fit(**fit_kwargs)
            return result
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Mixed model failed to converge: {last_error}") from last_error


def get_factor_constraints(result, factor: str) -> tuple[np.ndarray, list[str]]:
    parameter_names = list(result.fe_params.index)
    matched = []
    for index, name in enumerate(parameter_names):
        if name.startswith(f"C({factor})[T."):
            matched.append((index, name))
    if not matched:
        return np.zeros((0, len(parameter_names))), []
    constraint = np.zeros((len(matched), len(parameter_names)))
    labels = []
    for row_idx, (col_idx, label) in enumerate(matched):
        constraint[row_idx, col_idx] = 1.0
        labels.append(label)
    return constraint, labels


def wald_test_by_factor(result, alpha: float) -> pd.DataFrame:
    fe_names = list(result.fe_params.index)
    covariance = result.cov_params().loc[fe_names, fe_names]
    rows = []
    for factor in FACTOR_COLUMNS:
        _, labels = get_factor_constraints(result, factor)
        if not labels:
            rows.append({"factor": factor, "df": 0, "chi2": np.nan, "pvalue": np.nan, "significant": False})
            continue
        beta = result.fe_params.loc[labels].to_numpy(dtype=float)
        cov = covariance.loc[labels, labels].to_numpy(dtype=float)
        statistic = float(beta.T @ np.linalg.pinv(cov) @ beta)
        pvalue = float(stats.chi2.sf(statistic, df=len(labels)))
        rows.append(
            {
                "factor": factor,
                "df": int(len(labels)),
                "chi2": statistic,
                "pvalue": pvalue,
                "significant": pvalue < alpha,
            }
        )
    return pd.DataFrame(rows)


def marginal_means(result, dataframe: pd.DataFrame, factor: str) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    categories = list(dataframe[factor].cat.categories)
    for level in categories:
        scenario = dataframe.copy()
        scenario[factor] = level
        predictions = result.predict(exog=scenario)
        rows.append(
            {
                "factor": factor,
                "level": METHOD_LABELS.get(str(level), str(level)),
                "mean_prediction": float(np.mean(predictions)),
            }
        )
    return pd.DataFrame(rows)


def pairwise_method_comparisons(result, alpha: float) -> pd.DataFrame:
    fe_names = list(result.fe_params.index)
    covariance = result.cov_params().loc[fe_names, fe_names].to_numpy(dtype=float)
    fe_params = result.fe_params.to_numpy(dtype=float)
    index_by_name = {name: idx for idx, name in enumerate(fe_names)}

    def method_vector(method: str) -> np.ndarray:
        vector = np.zeros(len(fe_names), dtype=float)
        if method != METHOD_ORDER[0]:
            name = f"C(method)[T.{method}]"
            vector[index_by_name[name]] = 1.0
        return vector

    rows = []
    for left_idx, left in enumerate(METHOD_ORDER):
        for right in METHOD_ORDER[left_idx + 1 :]:
            contrast = method_vector(left) - method_vector(right)
            mean_diff = float(contrast @ fe_params)
            variance = float(contrast @ covariance @ contrast)
            se = float(np.sqrt(max(variance, 0.0)))
            if se <= 0.0:
                z_stat = np.nan
                pvalue = np.nan
                ci_low = np.nan
                ci_high = np.nan
            else:
                z_stat = mean_diff / se
                pvalue = float(2.0 * stats.norm.sf(abs(z_stat)))
                margin = float(stats.norm.ppf(0.975) * se)
                ci_low = mean_diff - margin
                ci_high = mean_diff + margin
            rows.append(
                {
                    "left": METHOD_LABELS[left],
                    "right": METHOD_LABELS[right],
                    "mean_diff": mean_diff,
                    "z_stat": float(z_stat),
                    "raw_pvalue": float(pvalue),
                    "ci_low": float(ci_low),
                    "ci_high": float(ci_high),
                }
            )

    pairwise = pd.DataFrame(rows)
    reject, corrected, _, _ = multipletests(pairwise["raw_pvalue"], alpha=alpha, method="holm")
    pairwise["adjusted_pvalue"] = corrected
    pairwise["significant"] = reject
    return pairwise.sort_values(["adjusted_pvalue", "left", "right"]).reset_index(drop=True)


def summarize_method_ranking(marginal_table: pd.DataFrame) -> pd.DataFrame:
    ranking = marginal_table.copy().sort_values("mean_prediction", ascending=True).reset_index(drop=True)
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    return ranking[["rank", "level", "mean_prediction"]]


def anova_report(dataframe: pd.DataFrame, alpha: float = 0.05) -> str:
    analysis_df = build_anova_dataframe(dataframe)
    validate_repeated_measures(analysis_df)

    raw_df, _ = add_response_column(analysis_df, log_response=False)
    raw_result = fit_mixed_model(raw_df)
    raw_tests = wald_test_by_factor(raw_result, alpha)
    raw_marginal_method = marginal_means(raw_result, raw_df, "method")
    raw_pairwise = pairwise_method_comparisons(raw_result, alpha)

    log_df, _ = add_response_column(analysis_df, log_response=True)
    log_result = fit_mixed_model(log_df)
    log_tests = wald_test_by_factor(log_result, alpha)
    log_marginal_method = marginal_means(log_result, log_df, "method")

    def fmt_pvalue(value: float) -> str:
        if value < 1e-16:
            return "< 1e-16"
        return f"{value:.3g}"

    raw_test_map = raw_tests.set_index("factor")
    log_test_map = log_tests.set_index("factor")
    raw_method_p = float(raw_test_map.loc["method", "pvalue"])
    raw_size_p = float(raw_test_map.loc["size", "pvalue"])
    raw_type_p = float(raw_test_map.loc["type", "pvalue"])
    raw_workload_p = float(raw_test_map.loc["workload", "pvalue"])
    raw_customers_p = float(raw_test_map.loc["customers", "pvalue"])
    raw_demand_p = float(raw_test_map.loc["demand", "pvalue"])
    log_demand_p = float(log_test_map.loc["demand", "pvalue"])

    ranking = summarize_method_ranking(raw_marginal_method)
    log_ranking = summarize_method_ranking(log_marginal_method)
    nonsig_pairwise = raw_pairwise[~raw_pairwise["significant"]]

    lines = [
        "ANOVA summary",
        f"Method effect p-value: {fmt_pvalue(raw_method_p)}",
        f"Size effect p-value: {fmt_pvalue(raw_size_p)}",
        f"Type effect p-value: {fmt_pvalue(raw_type_p)}",
        f"Demand effect p-value: {fmt_pvalue(raw_demand_p)}",
        f"Workload effect p-value: {fmt_pvalue(raw_workload_p)}",
        f"Customers effect p-value: {fmt_pvalue(raw_customers_p)}",
        "",
        "Method ranking on raw objective scale:",
        ranking.to_string(index=False),
        "",
        "Pairwise comparisons after Holm adjustment:",
        raw_pairwise[["left", "right", "adjusted_pvalue", "significant"]].to_string(index=False),
        "",
        "Non-significant pairwise comparisons:",
        nonsig_pairwise[["left", "right", "adjusted_pvalue"]].to_string(index=False)
        if not nonsig_pairwise.empty
        else "None",
        "",
        "Log-response robustness check",
        f"Demand effect p-value on log scale: {fmt_pvalue(log_demand_p)}",
        "Method ranking on log objective scale:",
        log_ranking.to_string(index=False),
        "",
        "Fixed-effect tests on raw scale:",
        raw_tests.to_string(index=False),
        "",
        "Fixed-effect tests on log scale:",
        log_tests.to_string(index=False),
    ]
    return "\n".join(lines)


def validate_input_file(path: Path) -> Path:
    if not path.is_absolute():
        raise ValueError(f"Input path must be absolute: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the paper tables, figure, comment summary, and ANOVA report from VNS/LNS result files."
    )
    parser.add_argument("--vnsl-file", type=Path, required=True)
    parser.add_argument("--vnsm-file", type=Path, required=True)
    parser.add_argument("--vnsh-file", type=Path, required=True)
    parser.add_argument("--lns-file", type=Path, required=True)
    parser.add_argument("--fig1", action="store_true")
    parser.add_argument("--table3", action="store_true")
    parser.add_argument("--table4", action="store_true")
    parser.add_argument("--table5", action="store_true")
    parser.add_argument("--table6", action="store_true")
    parser.add_argument("--comment", action="store_true")
    parser.add_argument("--anova", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    vnsl_file = validate_input_file(args.vnsl_file.resolve())
    vnsm_file = validate_input_file(args.vnsm_file.resolve())
    vnsh_file = validate_input_file(args.vnsh_file.resolve())
    lns_file = validate_input_file(args.lns_file.resolve())

    if not any((args.fig1, args.table3, args.table4, args.table5, args.table6, args.comment, args.anova)):
        raise ValueError("Select at least one output flag.")

    dataframe = build_dataframe(vnsl_file, vnsm_file, vnsh_file, lns_file)

    if args.fig1:
        output_path = plot_group3_lns600_comparison(dataframe, FIG1_OUTPUT)
        print(output_path)
    if args.table3:
        print(group12_success_pct_table_to_latex(dataframe))
    if args.table4:
        print(group12_obj_gap_full_with_wins_table_to_latex(dataframe))
    if args.table5:
        print(group12_time_table_to_latex(dataframe))
    if args.table6:
        print(group3_avg_obj_table_to_latex(dataframe))
    if args.comment:
        print(build_comment_table(dataframe).to_markdown(index=False))
    if args.anova:
        print(anova_report(dataframe))

    return 0


if __name__ == "__main__":
    sys.exit(main())
