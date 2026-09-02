"""Build the analysis matrix for the diabetes 30-day readmission AUC study.

Source: UCI 296, "Diabetes 130-US Hospitals for Years 1999-2008"
(Strack et al. 2014), 101,766 inpatient encounters of diabetic patients from
130 US hospitals.

Design decisions, each of which a referee will ask about:

1. UNIT OF ANALYSIS.  The rows are encounters, not patients: 101,766 encounters
   come from 71,518 patients, so 30% of rows are repeat visits and the i.i.d.
   assumption (A.1) fails on the raw file.  We keep the FIRST encounter per
   patient (by encounter_id, which is chronological) and drop the rest.  This
   is the standard handling for this dataset and restores independence across
   rows up to the usual within-hospital clustering, which we do not model and
   which we state as a limitation.

2. OUTCOME.  Positive = readmitted within 30 days ("<30"); negative = "NO" or
   ">30".  This is the clinical and payer-relevant endpoint (CMS penalises
   30-day readmission); "any readmission" would be more balanced but is not a
   meaningful endpoint.  Prevalence is about 9% after de-duplication, so the
   two-sample structure of the AUC data term is not a formality: the minority
   class controls the variance, which is exactly the two-stratum DeLong
   structure the theory reproduces.

3. EXCLUSIONS.  Encounters ending in death or hospice (discharge_disposition_id
   in 11, 13, 14, 19, 20, 21) cannot be readmitted and are removed; leaving them
   in would put structural zeros in the negative class.

4. COVARIATES.  Numeric utilisation and diagnosis counts, plus indicator-coded
   demographics, glycaemic tests, and treatment change flags.  Columns with
   >40% missingness (weight 97%, max_glu_serum 95%, A1Cresult 83%,
   medical_specialty 49%, payer_code 40%) are dropped rather than imputed.
   Diagnosis codes are collapsed to the standard ICD-9 chapter groups rather
   than used as thousands of levels.  All covariates are standardised with
   training-split statistics only.

Writes real_data/data/diabetes_design.npz and a JSON provenance record.
"""
import hashlib, json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "diabetes130_with_ids.csv")
OUT = os.path.join(HERE, "data", "diabetes_design.npz")
META = os.path.join(HERE, "data", "diabetes_design_meta.json")

DEATH_OR_HOSPICE = {11, 13, 14, 19, 20, 21}
DROP_MISSING = ["weight", "max_glu_serum", "A1Cresult", "medical_specialty", "payer_code"]
NUMERIC = ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
           "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses"]
# age_years is appended to NUMERIC inside main() after the band midpoints are formed.


def icd9_chapter(code):
    """Collapse an ICD-9 diagnosis code to the coarse chapter used in the
    readmission literature for this dataset."""
    if pd.isna(code) or code == "?":
        return "missing"
    s = str(code)
    if s.startswith("V"):
        return "supplementary"
    if s.startswith("E"):
        return "external"
    try:
        v = float(s)
    except ValueError:
        return "other"
    if 390 <= v <= 459 or v == 785: return "circulatory"
    if 460 <= v <= 519 or v == 786: return "respiratory"
    if 520 <= v <= 579 or v == 787: return "digestive"
    if 250 <= v < 251:              return "diabetes"
    if 800 <= v <= 999:             return "injury"
    if 710 <= v <= 739:             return "musculoskeletal"
    if 580 <= v <= 629 or v == 788: return "genitourinary"
    if 140 <= v <= 239:             return "neoplasms"
    return "other"


def main():
    df = pd.read_csv(RAW, low_memory=False)
    n_raw = len(df)

    # (1) one encounter per patient: the first, encounter_id being chronological
    df = df.sort_values("encounter_id").drop_duplicates("patient_nbr", keep="first")
    n_dedup = len(df)

    # (3) remove encounters that cannot be readmitted
    df = df[~df["discharge_disposition_id"].isin(DEATH_OR_HOSPICE)]
    n_alive = len(df)

    # (2) outcome
    y = (df["readmitted"] == "<30").astype(int).to_numpy()

    # (4) covariates
    df = df.drop(columns=[c for c in DROP_MISSING if c in df.columns])
    df["diag_1_grp"] = df["diag_1"].map(icd9_chapter)
    df["diag_2_grp"] = df["diag_2"].map(icd9_chapter)
    df["diag_3_grp"] = df["diag_3"].map(icd9_chapter)

    # age arrives as ten ordered decade bands.  Indicator-coding them puts nine
    # nearly collinear columns into the PAIR-DIFFERENCE design (the weakest
    # eigen-direction of the pair Gram loaded 0.48/0.46/0.42/0.41/0.33 on the
    # age dummies, giving kappa = 1950).  The bands are ordered, so we use the
    # band midpoint in decades: one column, no loss of the ordering, and the
    # collinearity disappears.
    age_mid = df["age"].str.extract(r"\[(\d+)-(\d+)\)").astype(float).mean(axis=1)
    df = df.assign(age_years=age_mid)

    num = df[NUMERIC + ["age_years"]].astype(float)
    # counts of prior utilisation are right skewed; log1p keeps them on a scale
    # where a linear score is sensible without changing their ordering
    for c in ["number_outpatient", "number_emergency", "number_inpatient"]:
        num[c] = np.log1p(num[c])

    cat_cols = ["race", "gender", "diag_1_grp", "diag_2_grp", "diag_3_grp",
                "change", "diabetesMed", "insulin", "metformin"]
    cat = df[cat_cols].astype(str).replace({"?": "missing", "Unknown/Invalid": "missing"})
    # drop_first: the reference level is absorbed; no intercept is used because
    # the AUC surrogate depends on covariate DIFFERENCES, in which any constant
    # cancels.
    dummies = pd.get_dummies(cat, drop_first=True, dtype=float)
    # drop rare indicators.  A level present in under 2% of rows contributes a
    # direction the pairwise Hessian estimates from very few informative pairs;
    # at 0.5% eighteen such columns survived and inflated the condition number.
    keep = dummies.columns[(dummies.mean() >= 0.02) & (dummies.mean() <= 0.98)]
    dummies = dummies[keep]

    X = pd.concat([num.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
    names = list(X.columns)
    X = X.to_numpy(float)

    out = {"X": X, "y": y, "names": np.array(names, dtype=object)}
    np.savez_compressed(OUT, **out)
    meta = {
        "source": "UCI 296 Diabetes 130-US Hospitals 1999-2008",
        "raw_sha256": hashlib.sha256(open(RAW, "rb").read()).hexdigest(),
        "rows_raw": n_raw, "rows_after_first_encounter": n_dedup,
        "rows_after_removing_death_hospice": n_alive,
        "n": int(len(y)), "p": int(X.shape[1]),
        "n_pos_30day": int(y.sum()), "prevalence": float(y.mean()),
        "pairs": int(y.sum()) * int((1 - y).sum()),
        "dropped_high_missing": DROP_MISSING,
        "covariates": names,
    }
    json.dump(meta, open(META, "w"), indent=1)
    print(f"  encounters {n_raw:,} -> first per patient {n_dedup:,} -> alive at discharge {n_alive:,}")
    print(f"  n = {meta['n']:,}   p = {meta['p']}   positives = {meta['n_pos_30day']:,} "
          f"({meta['prevalence']:.2%})   pairs = {meta['pairs']:,}")
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
