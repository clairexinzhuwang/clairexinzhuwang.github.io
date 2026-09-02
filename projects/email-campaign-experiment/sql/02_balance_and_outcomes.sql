-- Hillstrom experiment: aggregate balance and ITT outcome queries.
-- Run 01_cohort_and_quality_audit.sql first to create hillstrom_analysis.
-- Python is the inferential source of truth for confidence intervals and Holm
-- correction; these SQL queries make cohort logic and raw aggregates auditable.

-- Arm-level outcomes.
SELECT
    segment AS arm,
    COUNT(*) AS n,
    AVG(visit) AS visit_rate,
    AVG(conversion) AS conversion_rate,
    AVG(spend) AS mean_spend,
    SUM(spend) AS total_spend
FROM hillstrom_analysis
GROUP BY segment
ORDER BY segment;

-- Numeric/binary pre-treatment balance, expressed as standardized mean
-- differences versus control. Balance is diagnosed by magnitude, not p-values.
WITH balance_long AS (
    SELECT segment, 'recency' AS covariate, CAST(recency AS DOUBLE PRECISION) AS value
    FROM hillstrom_analysis
    UNION ALL
    SELECT segment, 'history', CAST(history AS DOUBLE PRECISION)
    FROM hillstrom_analysis
    UNION ALL
    SELECT segment, 'mens', CAST(mens AS DOUBLE PRECISION)
    FROM hillstrom_analysis
    UNION ALL
    SELECT segment, 'womens', CAST(womens AS DOUBLE PRECISION)
    FROM hillstrom_analysis
    UNION ALL
    SELECT segment, 'newbie', CAST(newbie AS DOUBLE PRECISION)
    FROM hillstrom_analysis
), arm_stats AS (
    SELECT
        segment,
        covariate,
        COUNT(*) AS n,
        AVG(value) AS mean_value,
        VAR_SAMP(value) AS variance_value
    FROM balance_long
    GROUP BY segment, covariate
), comparisons(treatment, comparator) AS (
    VALUES
        ('Mens E-Mail', 'No E-Mail'),
        ('Womens E-Mail', 'No E-Mail')
)
SELECT
    comparisons.treatment,
    comparisons.comparator,
    treatment_stats.covariate,
    treatment_stats.mean_value AS mean_treatment,
    control_stats.mean_value AS mean_comparator,
    treatment_stats.mean_value - control_stats.mean_value AS difference,
    (treatment_stats.mean_value - control_stats.mean_value)
        / NULLIF(SQRT((treatment_stats.variance_value + control_stats.variance_value) / 2.0), 0)
        AS standardized_mean_difference
FROM comparisons
JOIN arm_stats AS treatment_stats
    ON treatment_stats.segment = comparisons.treatment
JOIN arm_stats AS control_stats
    ON control_stats.segment = comparisons.comparator
   AND control_stats.covariate = treatment_stats.covariate
ORDER BY comparisons.treatment, treatment_stats.covariate;

-- Categorical shares by arm. The Python output additionally computes one-hot
-- SMDs for every level.
WITH categorical_long AS (
    SELECT segment, 'history_segment' AS covariate, history_segment AS level
    FROM hillstrom_analysis
    UNION ALL
    SELECT segment, 'zip_code', zip_code
    FROM hillstrom_analysis
    UNION ALL
    SELECT segment, 'channel', channel
    FROM hillstrom_analysis
), counts AS (
    SELECT segment, covariate, level, COUNT(*) AS level_n
    FROM categorical_long
    GROUP BY segment, covariate, level
), arm_sizes AS (
    SELECT segment, COUNT(*) AS arm_n
    FROM hillstrom_analysis
    GROUP BY segment
)
SELECT
    counts.segment,
    counts.covariate,
    counts.level,
    counts.level_n,
    counts.level_n * 1.0 / arm_sizes.arm_n AS arm_share
FROM counts
JOIN arm_sizes ON arm_sizes.segment = counts.segment
ORDER BY counts.covariate, counts.level, counts.segment;

-- Unadjusted ITT differences and unpooled standard errors for each outcome.
-- See outputs/treatment_effects.csv for CIs, raw p-values, and family-specific
-- Holm-adjusted p-values.
WITH outcome_long AS (
    SELECT segment, 'visit' AS outcome, CAST(visit AS DOUBLE PRECISION) AS value
    FROM hillstrom_analysis
    UNION ALL
    SELECT segment, 'conversion', CAST(conversion AS DOUBLE PRECISION)
    FROM hillstrom_analysis
    UNION ALL
    SELECT segment, 'spend', CAST(spend AS DOUBLE PRECISION)
    FROM hillstrom_analysis
), arm_stats AS (
    SELECT
        segment,
        outcome,
        COUNT(*) AS n,
        AVG(value) AS mean_value,
        VAR_SAMP(value) AS variance_value
    FROM outcome_long
    GROUP BY segment, outcome
), comparisons(treatment, comparator, family) AS (
    VALUES
        ('Mens E-Mail', 'No E-Mail', 'primary_email_vs_control'),
        ('Womens E-Mail', 'No E-Mail', 'primary_email_vs_control'),
        ('Mens E-Mail', 'Womens E-Mail', 'secondary_mens_vs_womens')
)
SELECT
    comparisons.family,
    comparisons.treatment,
    comparisons.comparator,
    treatment_stats.outcome,
    treatment_stats.n AS n_treatment,
    control_stats.n AS n_comparator,
    treatment_stats.mean_value AS mean_treatment,
    control_stats.mean_value AS mean_comparator,
    treatment_stats.mean_value - control_stats.mean_value AS estimate,
    SQRT(
        treatment_stats.variance_value / treatment_stats.n
        + control_stats.variance_value / control_stats.n
    ) AS standard_error
FROM comparisons
JOIN arm_stats AS treatment_stats
    ON treatment_stats.segment = comparisons.treatment
JOIN arm_stats AS control_stats
    ON control_stats.segment = comparisons.comparator
   AND control_stats.outcome = treatment_stats.outcome
ORDER BY comparisons.family, comparisons.treatment, treatment_stats.outcome;

-- The sole pre-specified segment view: visit by baseline newbie status. The
-- interaction (ATE for newbie=1 minus ATE for newbie=0) is tested in Python.
SELECT
    segment AS arm,
    newbie,
    COUNT(*) AS n,
    AVG(visit) AS visit_rate
FROM hillstrom_analysis
GROUP BY segment, newbie
ORDER BY newbie, segment;

