"""Reusable two-sample and two-group/two-round statistical analysis utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import energy_distance, ks_2samp, t, ttest_1samp, ttest_ind


def _finite_array(values, name="values"):
    """Convert values to a one-dimensional finite float array."""
    array = np.asarray(values, dtype=float).ravel()
    array = array[np.isfinite(array)]
    if array.size < 2:
        raise ValueError(f"{name} must contain at least two finite observations.")
    return array


def _summary(values):
    values = _finite_array(values)
    q1, q3 = np.quantile(values, [0.25, 0.75])
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)),
        "median": float(np.median(values)),
        "q1": float(q1),
        "q3": float(q3),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def holm_adjust(p_values):
    """Holm-adjust a family of p-values."""
    p_values = np.asarray(p_values, dtype=float)
    if p_values.ndim != 1 or np.any(~np.isfinite(p_values)):
        raise ValueError("p_values must be a one-dimensional finite array.")
    if np.any((p_values < 0) | (p_values > 1)):
        raise ValueError("All p-values must be between 0 and 1.")

    m = len(p_values)
    order = np.argsort(p_values)
    sorted_p = p_values[order]
    adjusted_sorted = np.maximum.accumulate((m - np.arange(m)) * sorted_p)
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)

    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted


def _welch_test(x, y, alpha=0.05):
    """Welch test and effect sizes for x minus y."""
    x = _finite_array(x, "x")
    y = _finite_array(y, "y")

    nx, ny = len(x), len(y)
    mean_x, mean_y = np.mean(x), np.mean(y)
    var_x, var_y = np.var(x, ddof=1), np.var(y, ddof=1)
    estimate = mean_x - mean_y

    variance_estimate = var_x / nx + var_y / ny
    standard_error = np.sqrt(variance_estimate)

    result = ttest_ind(x, y, equal_var=False, alternative="two-sided")

    if variance_estimate > 0:
        df = variance_estimate**2 / (
            (var_x / nx) ** 2 / (nx - 1) + (var_y / ny) ** 2 / (ny - 1)
        )
        critical = t.ppf(1 - alpha / 2, df)
        ci = (
            estimate - critical * standard_error,
            estimate + critical * standard_error,
        )
    else:
        df = np.nan
        ci = (estimate, estimate)

    pooled_sd = np.sqrt(((nx - 1) * var_x + (ny - 1) * var_y) / (nx + ny - 2))
    if pooled_sd > 0:
        cohens_d = estimate / pooled_sd
        correction = 1 - 3 / (4 * (nx + ny) - 9)
        hedges_g = correction * cohens_d
    else:
        cohens_d = np.nan
        hedges_g = np.nan

    return {
        "estimate": float(estimate),
        "standard_error": float(standard_error),
        "confidence_level": float(1 - alpha),
        "confidence_interval": (float(ci[0]), float(ci[1])),
        "t_statistic": float(result.statistic),
        "degrees_of_freedom": float(df),
        "p_value": float(result.pvalue),
        "cohens_d": float(cohens_d),
        "hedges_g": float(hedges_g),
        "significant": bool(result.pvalue < alpha),
    }


def compare_independent_samples(
    x,
    y,
    *,
    label_x="X",
    label_y="Y",
    variable="Variable",
    alpha=0.05,
    n_permutations=9999,
    seed=None,
    run_energy_test=True,
):
    """
    Compare two independent numeric samples.

    Welch's test addresses equality of means. The energy permutation test
    addresses equality of the complete distributions. The KS distance is
    reported descriptively as the maximum ECDF separation.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")

    x = _finite_array(x, label_x)
    y = _finite_array(y, label_y)

    mean_test = _welch_test(x, y, alpha=alpha)
    ks_distance = float(
        ks_2samp(x, y, alternative="two-sided", method="auto").statistic
    )

    distribution_test = {
        "energy_distance": np.nan,
        "energy_statistic": np.nan,
        "p_value": np.nan,
        "n_permutations": 0,
        "significant": False,
    }

    if run_energy_test:
        if int(n_permutations) != n_permutations or n_permutations < 1:
            raise ValueError("n_permutations must be a positive integer.")
        n_permutations = int(n_permutations)

        nx, ny = len(x), len(y)
        pooled = np.concatenate([x, y])
        observed_distance = energy_distance(x, y)
        observed_statistic = (nx * ny / (nx + ny)) * observed_distance**2

        rng = np.random.default_rng(seed)
        exceedances = 0

        for _ in range(n_permutations):
            shuffled = rng.permutation(pooled)
            x_perm = shuffled[:nx]
            y_perm = shuffled[nx:]
            perm_distance = energy_distance(x_perm, y_perm)
            perm_statistic = (nx * ny / (nx + ny)) * perm_distance**2
            exceedances += perm_statistic >= observed_statistic

        p_value = (exceedances + 1) / (n_permutations + 1)
        distribution_test = {
            "energy_distance": float(observed_distance),
            "energy_statistic": float(observed_statistic),
            "p_value": float(p_value),
            "n_permutations": n_permutations,
            "significant": bool(p_value < alpha),
        }

    return {
        "analysis": "independent_samples",
        "variable": variable,
        "label_x": label_x,
        "label_y": label_y,
        "contrast": f"{label_x} - {label_y}",
        "alpha": float(alpha),
        "summary": {
            label_x: _summary(x),
            label_y: _summary(y),
        },
        "mean_test": mean_test,
        "distribution_test": distribution_test,
        "ks_distance": ks_distance,
    }


def extract_station_round(
    dataframe,
    variable,
    *,
    group_prefix,
    round_value,
    station_level="station_id",
    round_level="round",
):
    """
    Extract a finite Series indexed only by station ID from a MultiIndex frame.

    Raises an error when more than one value remains for a station/round.
    """
    station_values = dataframe.index.get_level_values(station_level).astype(str)
    round_values = dataframe.index.get_level_values(round_level)

    mask = station_values.str.startswith(group_prefix) & (round_values == round_value)

    series = dataframe.loc[mask, variable].copy()
    series = series.replace([np.inf, -np.inf], np.nan).dropna()
    series.index = series.index.get_level_values(station_level)

    if not series.index.is_unique:
        duplicates = series.index[series.index.duplicated()].unique().tolist()
        raise ValueError(
            "More than one observation remains for these station IDs: "
            f"{duplicates}. Include an additional subject identifier or "
            "define an aggregation rule before analysis."
        )

    return series.sort_index()


def align_repeated_rounds(round_1, round_2):
    """Align two Series by subject ID and retain finite complete pairs."""
    if not isinstance(round_1, pd.Series) or not isinstance(round_2, pd.Series):
        raise TypeError("round_1 and round_2 must be pandas Series.")

    if not round_1.index.is_unique or not round_2.index.is_unique:
        raise ValueError("Each Series index must uniquely identify subjects.")

    paired = pd.concat(
        [
            round_1.rename("round_1"),
            round_2.rename("round_2"),
        ],
        axis=1,
        join="inner",
    )
    paired = paired.replace([np.inf, -np.inf], np.nan).dropna()
    paired["change"] = paired["round_2"] - paired["round_1"]

    if len(paired) < 2:
        raise ValueError("At least two complete subject pairs are required.")

    return paired


def _paired_change_test(change, alpha=0.05):
    """One-sample t-test of paired changes against zero."""
    change = _finite_array(change, "change")
    n = len(change)
    mean_change = np.mean(change)
    sd_change = np.std(change, ddof=1)
    standard_error = sd_change / np.sqrt(n)

    result = ttest_1samp(change, popmean=0, alternative="two-sided")
    critical = t.ppf(1 - alpha / 2, n - 1)
    ci = (
        mean_change - critical * standard_error,
        mean_change + critical * standard_error,
    )

    cohens_dz = mean_change / sd_change if sd_change > 0 else np.nan

    return {
        "n": int(n),
        "mean_change": float(mean_change),
        "sd_change": float(sd_change),
        "standard_error": float(standard_error),
        "confidence_level": float(1 - alpha),
        "confidence_interval": (float(ci[0]), float(ci[1])),
        "t_statistic": float(result.statistic),
        "degrees_of_freedom": int(n - 1),
        "p_value": float(result.pvalue),
        "cohens_dz": float(cohens_dz),
        "significant": bool(result.pvalue < alpha),
    }


def analyze_two_group_two_round(
    group_a_round_1,
    group_a_round_2,
    group_b_round_1,
    group_b_round_2,
    *,
    group_a="A",
    group_b="B",
    round_1_label="Round 1",
    round_2_label="Round 2",
    variable="Variable",
    alpha=0.05,
    n_permutations=9999,
    seed=None,
    run_energy_tests=True,
):
    """
    Complete analysis for two independent groups observed at two rounds.

    Subjects are paired within each group by Series index. The groups are
    treated as independent from each other.

    Includes:
    - Between-group comparisons at each round.
    - Paired change tests within each group.
    - Difference-in-differences test comparing subject-level changes.
    - Holm adjustment within related two-test families.
    """
    paired_a = align_repeated_rounds(group_a_round_1, group_a_round_2)
    paired_b = align_repeated_rounds(group_b_round_1, group_b_round_2)

    between_r1 = compare_independent_samples(
        group_a_round_1,
        group_b_round_1,
        label_x=group_a,
        label_y=group_b,
        variable=variable,
        alpha=alpha,
        n_permutations=n_permutations,
        seed=seed,
        run_energy_test=run_energy_tests,
    )
    between_r2 = compare_independent_samples(
        group_a_round_2,
        group_b_round_2,
        label_x=group_a,
        label_y=group_b,
        variable=variable,
        alpha=alpha,
        n_permutations=n_permutations,
        seed=None if seed is None else seed + 1,
        run_energy_test=run_energy_tests,
    )

    change_a = _paired_change_test(paired_a["change"], alpha=alpha)
    change_b = _paired_change_test(paired_b["change"], alpha=alpha)

    # Difference-in-differences: change A minus change B.
    did = _welch_test(
        paired_a["change"],
        paired_b["change"],
        alpha=alpha,
    )

    # Holm adjustments within logically related families.
    between_adjusted = holm_adjust(
        [
            between_r1["mean_test"]["p_value"],
            between_r2["mean_test"]["p_value"],
        ]
    )
    between_r1["mean_test"]["p_value_holm"] = float(between_adjusted[0])
    between_r2["mean_test"]["p_value_holm"] = float(between_adjusted[1])
    between_r1["mean_test"]["significant_holm"] = bool(between_adjusted[0] < alpha)
    between_r2["mean_test"]["significant_holm"] = bool(between_adjusted[1] < alpha)

    within_adjusted = holm_adjust(
        [
            change_a["p_value"],
            change_b["p_value"],
        ]
    )
    change_a["p_value_holm"] = float(within_adjusted[0])
    change_b["p_value_holm"] = float(within_adjusted[1])
    change_a["significant_holm"] = bool(within_adjusted[0] < alpha)
    change_b["significant_holm"] = bool(within_adjusted[1] < alpha)

    if run_energy_tests:
        energy_adjusted = holm_adjust(
            [
                between_r1["distribution_test"]["p_value"],
                between_r2["distribution_test"]["p_value"],
            ]
        )
        between_r1["distribution_test"]["p_value_holm"] = float(energy_adjusted[0])
        between_r2["distribution_test"]["p_value_holm"] = float(energy_adjusted[1])

    return {
        "analysis": "two_group_two_round",
        "variable": variable,
        "group_a": group_a,
        "group_b": group_b,
        "round_1_label": round_1_label,
        "round_2_label": round_2_label,
        "alpha": float(alpha),
        "paired_data": {
            group_a: paired_a,
            group_b: paired_b,
        },
        "between_groups": {
            round_1_label: between_r1,
            round_2_label: between_r2,
        },
        "within_group_changes": {
            group_a: change_a,
            group_b: change_b,
        },
        "difference_in_differences": {
            **did,
            "contrast": (
                f"({group_a} {round_2_label} - {round_1_label}) - "
                f"({group_b} {round_2_label} - {round_1_label})"
            ),
        },
    }


def analysis_tables(results):
    """Create shareable descriptive, mean-test, and distribution-test tables."""
    group_a = results["group_a"]
    group_b = results["group_b"]
    r1 = results["round_1_label"]
    r2 = results["round_2_label"]

    summary_rows = []
    for round_label in [r1, r2]:
        comparison = results["between_groups"][round_label]
        for group in [group_a, group_b]:
            row = {
                "variable": results["variable"],
                "round": round_label,
                "group": group,
            }
            row.update(comparison["summary"][group])
            summary_rows.append(row)

    mean_rows = []
    for round_label in [r1, r2]:
        test = results["between_groups"][round_label]["mean_test"]
        mean_rows.append(
            {
                "analysis": "Between groups",
                "contrast": f"{group_a} - {group_b} at {round_label}",
                "estimate": test["estimate"],
                "ci_lower": test["confidence_interval"][0],
                "ci_upper": test["confidence_interval"][1],
                "t": test["t_statistic"],
                "df": test["degrees_of_freedom"],
                "p": test["p_value"],
                "p_holm": test.get("p_value_holm", np.nan),
                "effect_size": test["hedges_g"],
                "effect_size_name": "Hedges g",
            }
        )

    for group in [group_a, group_b]:
        test = results["within_group_changes"][group]
        mean_rows.append(
            {
                "analysis": "Within-group change",
                "contrast": f"{group}: {r2} - {r1}",
                "estimate": test["mean_change"],
                "ci_lower": test["confidence_interval"][0],
                "ci_upper": test["confidence_interval"][1],
                "t": test["t_statistic"],
                "df": test["degrees_of_freedom"],
                "p": test["p_value"],
                "p_holm": test["p_value_holm"],
                "effect_size": test["cohens_dz"],
                "effect_size_name": "Cohen dz",
            }
        )

    did = results["difference_in_differences"]
    mean_rows.append(
        {
            "analysis": "Difference-in-differences",
            "contrast": f"Change in {group_a} - change in {group_b}",
            "estimate": did["estimate"],
            "ci_lower": did["confidence_interval"][0],
            "ci_upper": did["confidence_interval"][1],
            "t": did["t_statistic"],
            "df": did["degrees_of_freedom"],
            "p": did["p_value"],
            "p_holm": np.nan,
            "effect_size": did["hedges_g"],
            "effect_size_name": "Hedges g of changes",
        }
    )

    distribution_rows = []
    for round_label in [r1, r2]:
        comparison = results["between_groups"][round_label]
        energy = comparison["distribution_test"]
        distribution_rows.append(
            {
                "round": round_label,
                "contrast": f"{group_a} vs {group_b}",
                "energy_distance": energy["energy_distance"],
                "energy_statistic": energy["energy_statistic"],
                "energy_p": energy["p_value"],
                "energy_p_holm": energy.get("p_value_holm", np.nan),
                "ks_distance": comparison["ks_distance"],
                "n_permutations": energy["n_permutations"],
            }
        )

    return {
        "descriptive": pd.DataFrame(summary_rows),
        "mean_tests": pd.DataFrame(mean_rows),
        "distribution_tests": pd.DataFrame(distribution_rows),
    }


def _format_p(p):
    if not np.isfinite(p):
        return "NA"
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}"


def markdown_report(results):
    """Create a concise Markdown report suitable for copying and sharing."""
    group_a = results["group_a"]
    group_b = results["group_b"]
    r1 = results["round_1_label"]
    r2 = results["round_2_label"]
    variable = results["variable"]

    lines = [f"# Statistical summary: {variable}", ""]

    lines.append("## Between-group comparisons")
    for round_label in [r1, r2]:
        comparison = results["between_groups"][round_label]
        test = comparison["mean_test"]
        sx = comparison["summary"][group_a]
        sy = comparison["summary"][group_b]
        lines.extend(
            [
                f"- **{round_label}:** {group_a} mean = {sx['mean']:.3f} "
                f"(SD {sx['sd']:.3f}, n={sx['n']}); "
                f"{group_b} mean = {sy['mean']:.3f} "
                f"(SD {sy['sd']:.3f}, n={sy['n']}).",
                f"  Mean difference ({group_a} - {group_b}) = "
                f"{test['estimate']:.3f}, 95% CI "
                f"[{test['confidence_interval'][0]:.3f}, "
                f"{test['confidence_interval'][1]:.3f}], "
                f"t({test['degrees_of_freedom']:.2f}) = "
                f"{test['t_statistic']:.3f}, p = {_format_p(test['p_value'])}, "
                f"Holm-adjusted p = {_format_p(test.get('p_value_holm', np.nan))}, "
                f"Hedges g = {test['hedges_g']:.3f}.",
            ]
        )
        energy = comparison["distribution_test"]
        if np.isfinite(energy["p_value"]):
            lines.append(
                f"  Energy permutation test: p = "
                f"{_format_p(energy['p_value'])}; "
                f"KS distance = {comparison['ks_distance']:.3f}."
            )

    lines.extend(["", "## Within-group changes"])
    for group in [group_a, group_b]:
        test = results["within_group_changes"][group]
        lines.append(
            f"- **{group}:** mean change ({r2} - {r1}) = "
            f"{test['mean_change']:.3f}, 95% CI "
            f"[{test['confidence_interval'][0]:.3f}, "
            f"{test['confidence_interval'][1]:.3f}], "
            f"t({test['degrees_of_freedom']}) = "
            f"{test['t_statistic']:.3f}, p = {_format_p(test['p_value'])}, "
            f"Holm-adjusted p = {_format_p(test['p_value_holm'])}, "
            f"Cohen dz = {test['cohens_dz']:.3f}, n = {test['n']}."
        )

    did = results["difference_in_differences"]
    lines.extend(
        [
            "",
            "## Difference-in-differences",
            f"- Estimated difference in change ({group_a} - {group_b}) = "
            f"{did['estimate']:.3f}, 95% CI "
            f"[{did['confidence_interval'][0]:.3f}, "
            f"{did['confidence_interval'][1]:.3f}], "
            f"t({did['degrees_of_freedom']:.2f}) = "
            f"{did['t_statistic']:.3f}, p = {_format_p(did['p_value'])}.",
            "- This interaction test is the direct test of whether the groups "
            "changed differently over time.",
            "",
            "## Interpretation guardrail",
            "- A significant change in one group and a non-significant change in "
            "the other does not, by itself, demonstrate different changes. Use "
            "the difference-in-differences result for that conclusion.",
        ]
    )

    return "\n".join(lines)
