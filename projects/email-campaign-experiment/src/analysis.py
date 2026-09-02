"""Standard-library analysis for the Hillstrom randomized email experiment.

The estimand is the intention-to-treat difference in outcome means. The code
uses unpooled (Neyman/Welch) standard errors, large-sample normal confidence
intervals, and Holm family-wise error-rate adjustment for the pre-specified
families of hypotheses.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw" / "hillstrom.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
EXPECTED_SHA256 = "0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece"
EXPECTED_ROWS = 64_000
SOURCE_PAGE = "https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html"
SOURCE_CSV = (
    "http://www.minethatdata.com/"
    "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
)

CONTROL = "No E-Mail"
TREATMENTS = ("Mens E-Mail", "Womens E-Mail")
ARM_ORDER = (CONTROL, *TREATMENTS)
OUTCOMES = ("visit", "conversion", "spend")
BINARY_OUTCOMES = {"visit", "conversion"}
NUMERIC_BALANCE_FIELDS = ("recency", "history", "mens", "womens", "newbie")
CATEGORICAL_BALANCE_FIELDS = ("history_segment", "zip_code", "channel")
EXPECTED_COLUMNS = (
    "recency",
    "history_segment",
    "history",
    "mens",
    "womens",
    "zip_code",
    "newbie",
    "channel",
    "segment",
    "visit",
    "conversion",
    "spend",
)
Z_975 = 1.959963984540054


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_int(value: str, field: str, row_number: int) -> int:
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"Row {row_number}: {field}={value!r} is not an integer") from error


def _parse_float(value: str, field: str, row_number: int) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"Row {row_number}: {field}={value!r} is not numeric") from error
    if not math.isfinite(parsed):
        raise ValueError(f"Row {row_number}: {field} is not finite")
    return parsed


def load_rows(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load the released CSV, retaining source-quality facts for the audit."""

    rows: list[dict[str, Any]] = []
    raw_rows: Counter[tuple[str, ...]] = Counter()
    missing_cells = 0
    source_zip_typo_rows = 0

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != EXPECTED_COLUMNS:
            raise ValueError(
                "Unexpected columns. Expected "
                f"{list(EXPECTED_COLUMNS)}, observed {reader.fieldnames}"
            )
        for row_number, source_row in enumerate(reader, start=2):
            normalized_source = {key: (value or "").strip() for key, value in source_row.items()}
            missing_cells += sum(value == "" for value in normalized_source.values())
            raw_rows[tuple(normalized_source[column] for column in EXPECTED_COLUMNS)] += 1
            if normalized_source["zip_code"] == "Surburban":
                source_zip_typo_rows += 1

            rows.append(
                {
                    "recency": _parse_int(normalized_source["recency"], "recency", row_number),
                    "history_segment": normalized_source["history_segment"],
                    "history": _parse_float(normalized_source["history"], "history", row_number),
                    "mens": _parse_int(normalized_source["mens"], "mens", row_number),
                    "womens": _parse_int(normalized_source["womens"], "womens", row_number),
                    "zip_code": (
                        "Suburban"
                        if normalized_source["zip_code"] == "Surburban"
                        else normalized_source["zip_code"]
                    ),
                    "newbie": _parse_int(normalized_source["newbie"], "newbie", row_number),
                    "channel": normalized_source["channel"],
                    "segment": normalized_source["segment"],
                    "visit": _parse_int(normalized_source["visit"], "visit", row_number),
                    "conversion": _parse_int(
                        normalized_source["conversion"], "conversion", row_number
                    ),
                    "spend": _parse_float(normalized_source["spend"], "spend", row_number),
                }
            )

    metadata = {
        "missing_cells": missing_cells,
        "exact_duplicate_rows": sum(count - 1 for count in raw_rows.values() if count > 1),
        "source_zip_typo_rows": source_zip_typo_rows,
    }
    return rows, metadata


def validate_rows(rows: Sequence[dict[str, Any]], require_canonical: bool = True) -> None:
    errors: list[str] = []
    if require_canonical and len(rows) != EXPECTED_ROWS:
        errors.append(f"expected {EXPECTED_ROWS:,} rows, observed {len(rows):,}")

    observed_arms = {row["segment"] for row in rows}
    if observed_arms != set(ARM_ORDER):
        errors.append(f"unexpected experiment arms: {sorted(observed_arms)}")

    for field in ("mens", "womens", "newbie", "visit", "conversion"):
        bad = sum(row[field] not in (0, 1) for row in rows)
        if bad:
            errors.append(f"{field} has {bad} non-binary rows")

    if any(not 1 <= row["recency"] <= 12 for row in rows):
        errors.append("recency contains values outside 1..12")
    if any(row["history"] < 0 for row in rows):
        errors.append("history contains negative values")
    if any(row["spend"] < 0 for row in rows):
        errors.append("spend contains negative values")
    if any(row["conversion"] > row["visit"] for row in rows):
        errors.append("conversion=1 occurs when visit=0")
    if any((row["spend"] > 0) != (row["conversion"] == 1) for row in rows):
        errors.append("spend positivity and conversion indicator are inconsistent")

    allowed_zip = {"Urban", "Suburban", "Rural"}
    allowed_channel = {"Web", "Phone", "Multichannel"}
    if {row["zip_code"] for row in rows} - allowed_zip:
        errors.append("zip_code contains an unexpected category")
    if {row["channel"] for row in rows} - allowed_channel:
        errors.append("channel contains an unexpected category")

    if errors:
        raise ValueError("Data validation failed: " + "; ".join(errors))


def mean_and_variance(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("At least one observation is required")
    mean = fmean(values)
    if len(values) == 1:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, variance


def difference_in_means(
    treatment_values: Sequence[float], control_values: Sequence[float]
) -> dict[str, float | int]:
    """Unpooled difference-in-means inference using a large-sample normal pivot."""

    treatment_mean, treatment_variance = mean_and_variance(treatment_values)
    control_mean, control_variance = mean_and_variance(control_values)
    estimate = treatment_mean - control_mean
    standard_error = math.sqrt(
        treatment_variance / len(treatment_values) + control_variance / len(control_values)
    )
    if standard_error == 0:
        p_value = 1.0 if estimate == 0 else 0.0
    else:
        p_value = math.erfc(abs(estimate / standard_error) / math.sqrt(2.0))
    return {
        "n_treatment": len(treatment_values),
        "n_comparator": len(control_values),
        "mean_treatment": treatment_mean,
        "mean_comparator": control_mean,
        "estimate": estimate,
        "standard_error": standard_error,
        "ci_95_low": estimate - Z_975 * standard_error,
        "ci_95_high": estimate + Z_975 * standard_error,
        "p_value": p_value,
    }


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Return Holm-adjusted p-values in the original order."""

    count = len(p_values)
    ordered = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [0.0] * count
    running_max = 0.0
    for rank, (original_index, p_value) in enumerate(ordered):
        running_max = max(running_max, (count - rank) * p_value)
        adjusted[original_index] = min(1.0, running_max)
    return adjusted


def standardized_mean_difference(
    treatment_values: Sequence[float], control_values: Sequence[float]
) -> float:
    treatment_mean, treatment_variance = mean_and_variance(treatment_values)
    control_mean, control_variance = mean_and_variance(control_values)
    pooled_sd = math.sqrt((treatment_variance + control_variance) / 2.0)
    if pooled_sd == 0:
        return 0.0 if treatment_mean == control_mean else math.copysign(math.inf, treatment_mean - control_mean)
    return (treatment_mean - control_mean) / pooled_sd


def values_for(
    rows: Iterable[dict[str, Any]], arm: str, field: str, newbie: int | None = None
) -> list[float]:
    return [
        float(row[field])
        for row in rows
        if row["segment"] == arm and (newbie is None or row["newbie"] == newbie)
    ]


def build_arm_summary(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for arm in ARM_ORDER:
        arm_rows = [row for row in rows if row["segment"] == arm]
        result.append(
            {
                "arm": arm,
                "n": len(arm_rows),
                "visit_rate": fmean(row["visit"] for row in arm_rows),
                "conversion_rate": fmean(row["conversion"] for row in arm_rows),
                "mean_spend": fmean(row["spend"] for row in arm_rows),
                "total_spend": sum(row["spend"] for row in arm_rows),
            }
        )
    return result


def _effect_record(
    rows: Sequence[dict[str, Any]], family: str, treatment: str, comparator: str, outcome: str
) -> dict[str, Any]:
    statistics = difference_in_means(
        values_for(rows, treatment, outcome), values_for(rows, comparator, outcome)
    )
    comparator_mean = float(statistics["mean_comparator"])
    return {
        "family": family,
        "treatment": treatment,
        "comparator": comparator,
        "outcome": outcome,
        "unit": "proportion" if outcome in BINARY_OUTCOMES else "currency_units_per_customer",
        **statistics,
        "relative_lift": (
            float(statistics["estimate"]) / comparator_mean if comparator_mean != 0 else None
        ),
    }


def _add_holm_adjustment(records: list[dict[str, Any]]) -> None:
    adjusted = holm_adjust([float(record["p_value"]) for record in records])
    for record, p_value_holm in zip(records, adjusted):
        record["p_value_holm"] = p_value_holm
        record["significant_holm_0_05"] = p_value_holm < 0.05


def build_treatment_effects(
    rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    primary = [
        _effect_record(rows, "primary_email_vs_control", treatment, CONTROL, outcome)
        for treatment in TREATMENTS
        for outcome in OUTCOMES
    ]
    _add_holm_adjustment(primary)

    secondary = [
        _effect_record(
            rows,
            "secondary_mens_vs_womens",
            "Mens E-Mail",
            "Womens E-Mail",
            outcome,
        )
        for outcome in OUTCOMES
    ]
    _add_holm_adjustment(secondary)
    return primary, secondary


def build_balance(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    levels = {
        field: sorted({str(row[field]) for row in rows}) for field in CATEGORICAL_BALANCE_FIELDS
    }
    for treatment in TREATMENTS:
        for field in NUMERIC_BALANCE_FIELDS:
            treatment_values = values_for(rows, treatment, field)
            control_values = values_for(rows, CONTROL, field)
            treatment_mean = fmean(treatment_values)
            control_mean = fmean(control_values)
            result.append(
                {
                    "treatment": treatment,
                    "comparator": CONTROL,
                    "covariate": field,
                    "level": "",
                    "feature_type": "binary" if field in {"mens", "womens", "newbie"} else "numeric",
                    "mean_treatment": treatment_mean,
                    "mean_comparator": control_mean,
                    "difference": treatment_mean - control_mean,
                    "standardized_mean_difference": standardized_mean_difference(
                        treatment_values, control_values
                    ),
                }
            )

        for field in CATEGORICAL_BALANCE_FIELDS:
            for level in levels[field]:
                treatment_values = [
                    float(row[field] == level) for row in rows if row["segment"] == treatment
                ]
                control_values = [
                    float(row[field] == level) for row in rows if row["segment"] == CONTROL
                ]
                treatment_mean = fmean(treatment_values)
                control_mean = fmean(control_values)
                result.append(
                    {
                        "treatment": treatment,
                        "comparator": CONTROL,
                        "covariate": field,
                        "level": level,
                        "feature_type": "categorical_indicator",
                        "mean_treatment": treatment_mean,
                        "mean_comparator": control_mean,
                        "difference": treatment_mean - control_mean,
                        "standardized_mean_difference": standardized_mean_difference(
                            treatment_values, control_values
                        ),
                    }
                )
    return result


def build_newbie_segment(
    rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Pre-specified visit analysis by the baseline newbie indicator.

    The inferential question is the treatment-by-newbie interaction, not whether
    one stratum happens to cross a significance threshold while another does not.
    """

    stratum_effects: list[dict[str, Any]] = []
    interactions: list[dict[str, Any]] = []
    for treatment in TREATMENTS:
        by_stratum: dict[int, dict[str, float | int]] = {}
        for newbie in (0, 1):
            statistics = difference_in_means(
                values_for(rows, treatment, "visit", newbie),
                values_for(rows, CONTROL, "visit", newbie),
            )
            by_stratum[newbie] = statistics
            stratum_effects.append(
                {
                    "treatment": treatment,
                    "comparator": CONTROL,
                    "outcome": "visit",
                    "segment_variable": "newbie",
                    "segment_value": newbie,
                    **{key: value for key, value in statistics.items() if key != "p_value"},
                }
            )

        interaction_estimate = float(by_stratum[1]["estimate"]) - float(
            by_stratum[0]["estimate"]
        )
        interaction_se = math.sqrt(
            float(by_stratum[1]["standard_error"]) ** 2
            + float(by_stratum[0]["standard_error"]) ** 2
        )
        interaction_p = (
            math.erfc(abs(interaction_estimate / interaction_se) / math.sqrt(2.0))
            if interaction_se
            else (1.0 if interaction_estimate == 0 else 0.0)
        )
        interactions.append(
            {
                "family": "prespecified_newbie_visit_interactions",
                "treatment": treatment,
                "comparator": CONTROL,
                "outcome": "visit",
                "contrast": "ATE(newbie=1) - ATE(newbie=0)",
                "estimate": interaction_estimate,
                "standard_error": interaction_se,
                "ci_95_low": interaction_estimate - Z_975 * interaction_se,
                "ci_95_high": interaction_estimate + Z_975 * interaction_se,
                "p_value": interaction_p,
            }
        )

    _add_holm_adjustment(interactions)
    return stratum_effects, interactions


def build_data_quality(
    rows: Sequence[dict[str, Any]], load_metadata: dict[str, Any]
) -> dict[str, Any]:
    arm_sizes = Counter(str(row["segment"]) for row in rows)
    return {
        "row_count": len(rows),
        "arm_sizes": {arm: arm_sizes[arm] for arm in ARM_ORDER},
        "missing_cells": load_metadata["missing_cells"],
        "exact_duplicate_rows_not_dropped": load_metadata["exact_duplicate_rows"],
        "source_zip_typo_rows_normalized_in_memory": load_metadata["source_zip_typo_rows"],
        "conversion_without_visit_rows": sum(
            row["conversion"] > row["visit"] for row in rows
        ),
        "spend_conversion_inconsistency_rows": sum(
            (row["spend"] > 0) != (row["conversion"] == 1) for row in rows
        ),
        "note": (
            "The release has no customer identifier; exact repeated rows cannot be "
            "classified as duplicate customers and remain in the ITT analysis."
        ),
    }


def analyze(
    rows: Sequence[dict[str, Any]], load_metadata: dict[str, Any], source_hash: str
) -> dict[str, Any]:
    validate_rows(rows)
    arm_summary = build_arm_summary(rows)
    primary, secondary = build_treatment_effects(rows)
    balance = build_balance(rows)
    stratum_effects, interactions = build_newbie_segment(rows)
    max_balance = max(balance, key=lambda record: abs(record["standardized_mean_difference"]))
    balance_over_point_one = sum(
        abs(record["standardized_mean_difference"]) >= 0.1 for record in balance
    )
    any_interaction = any(record["significant_holm_0_05"] for record in interactions)

    return {
        "summary": {
            "schema_version": "1.0",
            "dataset": {
                "name": "MineThatData E-Mail Analytics and Data Mining Challenge",
                "source_page": SOURCE_PAGE,
                "source_csv": SOURCE_CSV,
                "source_sha256": source_hash,
                "raw_data_redistributed": False,
            },
            "design": {
                "design": "three-arm randomized controlled experiment",
                "estimand": "intention-to-treat difference in outcome means",
                "follow_up": "two weeks",
                "primary_family": (
                    "Six two-sided email-versus-control tests: two email arms by visit, "
                    "conversion, and spend; Holm-adjusted together."
                ),
                "secondary_family": (
                    "Three two-sided Mens-versus-Womens email tests; Holm-adjusted separately."
                ),
                "uncertainty": (
                    "Unpooled difference-in-means standard errors and large-sample 95% "
                    "normal confidence intervals."
                ),
            },
            "data_quality": build_data_quality(rows, load_metadata),
            "balance": {
                "maximum_absolute_standardized_mean_difference": abs(
                    max_balance["standardized_mean_difference"]
                ),
                "maximum_smd_covariate": max_balance["covariate"],
                "maximum_smd_level": max_balance["level"],
                "maximum_smd_treatment": max_balance["treatment"],
                "indicators_at_or_above_abs_0_1": balance_over_point_one,
                "interpretation": (
                    "No material imbalance by the conventional |SMD| < 0.10 diagnostic."
                    if balance_over_point_one == 0
                    else "At least one baseline diagnostic has |SMD| >= 0.10; inspect the CSV."
                ),
            },
            "arm_outcomes": arm_summary,
            "primary_effects": primary,
            "secondary_effects": secondary,
            "prespecified_segment": {
                "variable": "newbie",
                "outcome": "visit",
                "reason": (
                    "A single binary, pre-treatment lifecycle indicator declared in "
                    "the case-study plan; this is not an external preregistration."
                ),
                "interaction_family": interactions,
                "interpretation": (
                    "At least one Holm-adjusted treatment-by-newbie interaction is detectable."
                    if any_interaction
                    else "No Holm-adjusted evidence that the visit effect differs by newbie status."
                ),
            },
            "limitations": [
                "Revenue is observed, but campaign cost and gross margin are not; profit is not identified.",
                "Spend is zero-inflated and right-skewed; the mean remains the business estimand, and the large sample supports normal inference.",
                "The historical retail experiment may not transport to a different customer base, channel, or creative.",
                "The dataset has no delivery, open, unsubscribe, or customer-ID fields.",
            ],
        },
        "tables": {
            "arm_outcomes": arm_summary,
            "treatment_effects": primary + secondary,
            "balance": balance,
            "newbie_segment_effects": stratum_effects,
            "newbie_segment_interactions": interactions,
        },
    }


def _csv_cell(value: Any) -> Any:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        if math.isfinite(value):
            return format(value, ".12g")
        return str(value)
    if value is None:
        return ""
    return value


def write_csv(path: Path, records: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow({field: _csv_cell(record.get(field)) for field in fields})


def write_outputs(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "experiment_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result["summary"], handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    tables = result["tables"]
    write_csv(
        output_dir / "arm_outcomes.csv",
        tables["arm_outcomes"],
        ("arm", "n", "visit_rate", "conversion_rate", "mean_spend", "total_spend"),
    )
    write_csv(
        output_dir / "treatment_effects.csv",
        tables["treatment_effects"],
        (
            "family",
            "treatment",
            "comparator",
            "outcome",
            "unit",
            "n_treatment",
            "n_comparator",
            "mean_treatment",
            "mean_comparator",
            "estimate",
            "standard_error",
            "ci_95_low",
            "ci_95_high",
            "relative_lift",
            "p_value",
            "p_value_holm",
            "significant_holm_0_05",
        ),
    )
    write_csv(
        output_dir / "balance_summary.csv",
        tables["balance"],
        (
            "treatment",
            "comparator",
            "covariate",
            "level",
            "feature_type",
            "mean_treatment",
            "mean_comparator",
            "difference",
            "standardized_mean_difference",
        ),
    )
    write_csv(
        output_dir / "newbie_segment_effects.csv",
        tables["newbie_segment_effects"],
        (
            "treatment",
            "comparator",
            "outcome",
            "segment_variable",
            "segment_value",
            "n_treatment",
            "n_comparator",
            "mean_treatment",
            "mean_comparator",
            "estimate",
            "standard_error",
            "ci_95_low",
            "ci_95_high",
        ),
    )
    write_csv(
        output_dir / "newbie_segment_interactions.csv",
        tables["newbie_segment_interactions"],
        (
            "family",
            "treatment",
            "comparator",
            "outcome",
            "contrast",
            "estimate",
            "standard_error",
            "ci_95_low",
            "ci_95_high",
            "p_value",
            "p_value_holm",
            "significant_holm_0_05",
        ),
    )


def print_key_results(result: dict[str, Any]) -> None:
    print("\nEmail-versus-control ITT effects (95% CI; Holm-adjusted p):")
    for record in result["summary"]["primary_effects"]:
        scale = 100.0 if record["outcome"] in BINARY_OUTCOMES else 1.0
        suffix = " pp" if record["outcome"] in BINARY_OUTCOMES else ""
        print(
            f"  {record['treatment']:15s} {record['outcome']:10s} "
            f"{record['estimate'] * scale:+.4f}{suffix} "
            f"[{record['ci_95_low'] * scale:+.4f}, {record['ci_95_high'] * scale:+.4f}] "
            f"p_holm={record['p_value_holm']:.4g}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--allow-unverified-input",
        action="store_true",
        help="Analyze a file whose SHA-256 differs from the canonical public release.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    if not arguments.input.exists():
        raise SystemExit(
            f"Input not found: {arguments.input}\nRun: python3 -m src.download_data"
        )
    source_hash = sha256_file(arguments.input)
    if source_hash != EXPECTED_SHA256 and not arguments.allow_unverified_input:
        raise SystemExit(
            "Input checksum does not match the canonical release. "
            "Inspect the file or pass --allow-unverified-input explicitly."
        )
    rows, load_metadata = load_rows(arguments.input)
    result = analyze(rows, load_metadata, source_hash)
    write_outputs(result, arguments.output_dir)
    print(f"Validated {len(rows):,} rows; wrote aggregate outputs to {arguments.output_dir}")
    print_key_results(result)


if __name__ == "__main__":
    main()
