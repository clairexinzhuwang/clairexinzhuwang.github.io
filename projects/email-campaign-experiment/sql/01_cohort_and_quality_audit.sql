-- Hillstrom experiment: cohort construction and source-data audit.
-- Dialect: DuckDB/PostgreSQL-style SQL.
-- Assumption: the original CSV has been loaded without row filtering as
-- hillstrom_raw, with the twelve source column names preserved.

-- The source has no customer ID. This row number is an analysis-only locator,
-- not a reconstructed identity. Correct the known source label only in-view.
CREATE OR REPLACE VIEW hillstrom_analysis AS
SELECT
    ROW_NUMBER() OVER () AS analysis_row_number,
    CAST(recency AS INTEGER) AS recency,
    history_segment,
    CAST(history AS DOUBLE PRECISION) AS history,
    CAST(mens AS INTEGER) AS mens,
    CAST(womens AS INTEGER) AS womens,
    CASE WHEN zip_code = 'Surburban' THEN 'Suburban' ELSE zip_code END AS zip_code,
    CAST(newbie AS INTEGER) AS newbie,
    channel,
    segment,
    CAST(visit AS INTEGER) AS visit,
    CAST(conversion AS INTEGER) AS conversion,
    CAST(spend AS DOUBLE PRECISION) AS spend
FROM hillstrom_raw;

-- Cohort flow: the ITT cohort includes every randomized record. There are no
-- post-randomization exclusions.
SELECT
    COUNT(*) AS released_rows,
    COUNT(*) AS randomized_rows_in_itt,
    0 AS post_randomization_exclusions
FROM hillstrom_analysis;

-- Assignment counts.
SELECT
    segment,
    COUNT(*) AS customers,
    COUNT(*) * 1.0 / SUM(COUNT(*)) OVER () AS allocation_share
FROM hillstrom_analysis
GROUP BY segment
ORDER BY segment;

-- Missingness by released field.
SELECT
    SUM(CASE WHEN recency IS NULL THEN 1 ELSE 0 END) AS recency_missing,
    SUM(CASE WHEN history_segment IS NULL OR TRIM(history_segment) = '' THEN 1 ELSE 0 END)
        AS history_segment_missing,
    SUM(CASE WHEN history IS NULL THEN 1 ELSE 0 END) AS history_missing,
    SUM(CASE WHEN mens IS NULL THEN 1 ELSE 0 END) AS mens_missing,
    SUM(CASE WHEN womens IS NULL THEN 1 ELSE 0 END) AS womens_missing,
    SUM(CASE WHEN zip_code IS NULL OR TRIM(zip_code) = '' THEN 1 ELSE 0 END)
        AS zip_code_missing,
    SUM(CASE WHEN newbie IS NULL THEN 1 ELSE 0 END) AS newbie_missing,
    SUM(CASE WHEN channel IS NULL OR TRIM(channel) = '' THEN 1 ELSE 0 END)
        AS channel_missing,
    SUM(CASE WHEN segment IS NULL OR TRIM(segment) = '' THEN 1 ELSE 0 END)
        AS segment_missing,
    SUM(CASE WHEN visit IS NULL THEN 1 ELSE 0 END) AS visit_missing,
    SUM(CASE WHEN conversion IS NULL THEN 1 ELSE 0 END) AS conversion_missing,
    SUM(CASE WHEN spend IS NULL THEN 1 ELSE 0 END) AS spend_missing
FROM hillstrom_analysis;

-- Domain and cross-field checks. Every count should be zero.
SELECT
    SUM(CASE WHEN segment NOT IN ('No E-Mail', 'Mens E-Mail', 'Womens E-Mail')
        THEN 1 ELSE 0 END) AS invalid_assignment,
    SUM(CASE WHEN recency NOT BETWEEN 1 AND 12 THEN 1 ELSE 0 END) AS invalid_recency,
    SUM(CASE WHEN history < 0 THEN 1 ELSE 0 END) AS negative_history,
    SUM(CASE WHEN mens NOT IN (0, 1) THEN 1 ELSE 0 END) AS invalid_mens,
    SUM(CASE WHEN womens NOT IN (0, 1) THEN 1 ELSE 0 END) AS invalid_womens,
    SUM(CASE WHEN newbie NOT IN (0, 1) THEN 1 ELSE 0 END) AS invalid_newbie,
    SUM(CASE WHEN visit NOT IN (0, 1) THEN 1 ELSE 0 END) AS invalid_visit,
    SUM(CASE WHEN conversion NOT IN (0, 1) THEN 1 ELSE 0 END) AS invalid_conversion,
    SUM(CASE WHEN spend < 0 THEN 1 ELSE 0 END) AS negative_spend,
    SUM(CASE WHEN conversion > visit THEN 1 ELSE 0 END) AS conversion_without_visit,
    SUM(CASE WHEN (spend > 0 AND conversion = 0)
               OR (spend = 0 AND conversion = 1) THEN 1 ELSE 0 END)
        AS spend_conversion_inconsistency
FROM hillstrom_analysis;

-- Confirm and quantify the typo in the untouched source table.
SELECT zip_code, COUNT(*) AS rows
FROM hillstrom_raw
GROUP BY zip_code
ORDER BY zip_code;

-- Indistinguishable released rows. Without a customer ID these cannot be
-- classified as duplicate customers, so the analysis reports but retains them.
WITH repeated_rows AS (
    SELECT
        recency, history_segment, history, mens, womens, zip_code, newbie,
        channel, segment, visit, conversion, spend,
        COUNT(*) AS copies
    FROM hillstrom_raw
    GROUP BY
        recency, history_segment, history, mens, womens, zip_code, newbie,
        channel, segment, visit, conversion, spend
    HAVING COUNT(*) > 1
)
SELECT
    COUNT(*) AS repeated_row_patterns,
    COALESCE(SUM(copies - 1), 0) AS additional_indistinguishable_rows
FROM repeated_rows;

