-- ============================================================
-- MIMIC-IV hosp: AMR Prediction — ML-Ready Feature Matrix
-- Format  : Wide (one row per admission)
-- Target  : had_resistance (binary 0/1)
-- Features: Demographics + Lab aggregates (_first, _max)
-- Engine  : BigQuery (physionet-data.mimiciv_3_1_hosp)
--
-- Column naming convention:
--   <lab>_first  = first recorded value on admission (baseline)
--   <lab>_max    = peak value during admission (severity)
--
-- Before training:
--   DROP → hadm_id, subject_id, organisms_cultured, specimen_types
--   ENCODE → gender, race, insurance, admission_type (one-hot / ordinal)
-- ============================================================

WITH

-- ── 1. DEMOGRAPHICS ─────────────────────────────────────────
demographics AS (
    SELECT
        a.hadm_id,
        p.subject_id,
        p.anchor_age                                    AS age,
        p.gender,
        a.race,
        a.insurance,
        a.admission_type,
        TIMESTAMP_DIFF(a.dischtime, a.admittime, HOUR)  AS los_hours,
        a.hospital_expire_flag                          -- useful covariate / confounder
    FROM `physionet-data.mimiciv_3_1_hosp.admissions` a
    INNER JOIN `physionet-data.mimiciv_3_1_hosp.patients` p
        ON p.subject_id = a.subject_id
),

-- ── 2. TAG EACH LAB EVENT WITH A CLEAN FEATURE NAME ─────────
-- Maps MIMIC label strings → clean snake_case feature names.
-- NULL means the lab is not in our AMR panel — filtered out below.
tagged_labs AS (
    SELECT
        le.hadm_id,
        le.charttime,
        le.valuenum,
        CASE LOWER(dl.label)

            -- Infection markers
            WHEN 'wbc'                                  THEN 'wbc'
            WHEN 'bands'                                THEN 'bands'
            WHEN 'neutrophils'                          THEN 'neutrophils'
            WHEN 'lymphocytes'                          THEN 'lymphocytes'
            WHEN 'monocytes'                            THEN 'monocytes'

            -- Inflammatory markers
            WHEN 'c-reactive protein'                   THEN 'crp'
            WHEN 'procalcitonin'                        THEN 'procalcitonin'
            WHEN 'ferritin'                             THEN 'ferritin'

            -- Organ dysfunction / sepsis
            WHEN 'lactate'                              THEN 'lactate'
            WHEN 'creatinine'                           THEN 'creatinine'
            WHEN 'blood urea nitrogen'                  THEN 'bun'
            WHEN 'bilirubin, total'                     THEN 'bilirubin_total'
            WHEN 'alanine aminotransferase (alt)'       THEN 'alt'
            WHEN 'aspartate aminotransferase (ast)'     THEN 'ast'
            WHEN 'albumin'                              THEN 'albumin'
            WHEN 'glucose'                              THEN 'glucose'

            -- Coagulation (DIC screen in sepsis)
            WHEN 'inr(pt)'                              THEN 'inr'
            WHEN 'ptt'                                  THEN 'ptt'
            WHEN 'platelet count'                       THEN 'platelets'
            WHEN 'fibrinogen'                           THEN 'fibrinogen'
            WHEN 'd-dimer'                              THEN 'd_dimer'

            -- Blood gas / acid-base
            WHEN 'ph'                                   THEN 'ph'
            WHEN 'po2'                                  THEN 'po2'
            WHEN 'pco2'                                 THEN 'pco2'
            WHEN 'bicarbonate'                          THEN 'bicarbonate'
            WHEN 'hemoglobin'                           THEN 'hemoglobin'
            WHEN 'hematocrit'                           THEN 'hematocrit'

            -- Electrolytes
            WHEN 'sodium'                               THEN 'sodium'
            WHEN 'potassium'                            THEN 'potassium'
            WHEN 'chloride'                             THEN 'chloride'
            WHEN 'calcium, total'                       THEN 'calcium'
            WHEN 'magnesium'                            THEN 'magnesium'
            WHEN 'phosphate'                            THEN 'phosphate'

            ELSE NULL  -- not in our panel; filtered below
        END AS feature_name

    FROM `physionet-data.mimiciv_3_1_hosp.labevents` le
    INNER JOIN `physionet-data.mimiciv_3_1_hosp.d_labitems` dl
        ON dl.itemid = le.itemid
    WHERE le.valuenum IS NOT NULL   -- numeric values only
      AND le.hadm_id  IS NOT NULL
),

-- ── 3. PIVOT: FIRST + MAX PER LAB PER ADMISSION ─────────────
-- ARRAY_AGG ... LIMIT 1 gives the chronologically first value.
-- MAX gives the peak value during the admission.
-- NULL = lab was never measured for this admission (handle in Python).
lab_features AS (
    SELECT
        hadm_id,

        -- ── Infection markers ────────────────────────────────
        ARRAY_AGG(IF(feature_name='wbc',          valuenum, NULL) IGNORE NULLS ORDER BY charttime LIMIT 1)[SAFE_OFFSET(0)] AS wbc_first,
        MAX(IF(feature_name='wbc',                valuenum, NULL))                                                          AS wbc_max,

        ARRAY_AGG(IF(feature_name='bands',        valuenum, NULL) IGNORE NULLS ORDER BY charttime LIMIT 1)[SAFE_OFFSET(0)] AS bands_first,
        MAX(IF(feature_name='bands',              valuenum, NULL))                                                          AS bands_max,

        ARRAY_AGG(IF(feature_name='neutrophils',  valuenum, NULL) IGNORE NULLS ORDER BY charttime LIMIT 1)[SAFE_OFFSET(0)] AS neutrophils_first,
        MAX(IF(feature_name='neutrophils',        valuenum, NULL))                                                          AS neutrophils_max,

        ARRAY_AGG(IF(feature_name='lymphocytes',  valuenum, NULL) IGNORE NULLS ORDER BY charttime LIMIT 1)[SAFE_OFFSET(0)] AS lymphocytes_first,
        MAX(IF(feature_name='lymphocytes',        valuenum, NULL))                                                          AS lymphocytes_max,

        ARRAY_AGG(IF(feature_name='monocytes',    valuenum, NULL) IGNORE NULLS ORDER BY charttime LIMIT 1)[SAFE_OFFSET(0)] AS monocytes_first,
        MAX(IF(feature_name='monocytes',          valuenum, NULL))                                                          AS monocytes_max,

        -- ── Inflammatory markers ─────────────────────────────
        ARRAY_AGG(IF(feature_name='crp',          valuenum, NULL) IGNORE NULLS ORDER BY charttime LIMIT 1)[SAFE_OFFSET(0)] AS crp_first,
        MAX(IF(feature_name='crp',                valuenum, NULL))                                                          AS crp_max,

        ARRAY_AGG(IF(feature_name='procalcitonin',valuenum, NULL) IGNORE NULLS ORDER BY charttime LIMIT 1)[SAFE_OFFSET(0)] AS procalcitonin_first,
        MAX(IF(feature_name='procalcitonin',      valuenum, NULL))                                                          AS procalcitonin_max,

        ARRAY_AGG(IF(feature_name='ferritin',     valuenum, NULL) IGNORE NULLS ORDER BY charttime LIMIT 1)[SAFE_OFFSET(0)] AS ferritin_first,
        MAX(IF(feature_name='ferritin',           valuenum, NULL))                                                          AS ferritin_max,

        -- ── Organ dysfunction ────────────────────────────────
        ARRAY_AGG(IF(feature_name='lactate',      valuenum, NULL) IGNORE NULLS ORDER BY charttime LIMIT 1)[SAFE_OFFSET(0)] AS lactate_first,
        MAX(IF(feature_name='lactate',            valuenum, NULL))                                                          AS lactate_max,

        ARRAY_AGG(IF(feature_name='creatinine',   valuenum, NULL) IGNORE NULLS ORDER BY charttime LIMIT 1)[SAFE_OFFSET(0)] AS creatinine_first,
        MAX(IF(feature_name='creatinine',         valuenum, NULL))                                                          AS creatinine_max,

        ARRAY_AGG(IF(feature_name='bun',          valuenum, NULL) IGNORE NULLS ORDER BY charttime LIMIT 1)[SAFE_OFFSET(0)] AS bun_first,
        MAX(IF(feature_name='bun',                valuenum, NULL))                                                          AS bun_max,

        ARRAY_AGG(IF(feature_name='bilirubin_total', valuenum, NULL) IGNORE NULLS ORDER BY charttime LIMIT 1)[SAFE_OFFSET(0)] AS bilirubin_total_first,
        MAX(IF(feature_name='bilirubin_total',    valuenum, NULL))                                                          AS bilirubin_total_max,

        ARRAY_AGG(IF(feature_name='alt',          valuenum, NULL) IGNORE NULLS ORDER BY charttime LIMIT 1)[SAFE_OFFSET(0)] AS alt_first,
        MAX(IF(feature_name='alt',                valuenum, NULL))                                                          AS alt_max,

        ARRAY_AGG(IF(feature_name='ast',          valuenum, NULL) IGNORE NULLS ORDER BY charttime LIMIT 1)[SAFE_OFFSET(0)] AS ast_first,
        MAX(IF(feature_name='ast',                valuenum, NULL))                                                          AS ast_max,

        ARRAY_AGG(IF(feature_name='albumin',      valuenum, NULL) IGNORE NULLS ORDER BY charttime LIMIT 1)[SAFE_OFFSET(0)] AS albumin_first,
        MAX(IF(feature_name='albumin',            valuenum, NULL))                                                          AS albumin_max,

        ARRAY_AGG(IF(feature_name='glucose',      valuenum, NULL) IGNORE NULLS ORDER BY charttime LIMIT 1)[SAFE_OFFSET(0)] AS glucose_first,
        MAX(IF(feature_name='glucose',            valuenum, NULL))                                                          AS glucose_max,

        -- ── Coagulation ──────────────────────────────────────
        ARRAY_AGG(IF(feature_name='inr',          valuenum, NULL) IGNORE NULLS ORDER BY charttime LIMIT 1)[SAFE_OFFSET(0)] AS inr_first,
        MAX(IF(feature_name='inr',                valuenum, NULL))                                                          AS inr_max,

        ARRAY_AGG(IF(feature_name='ptt',          valuenum, NULL) IGNORE NULLS ORDER BY charttime LIMIT 1)[SAFE_OFFSET(0)] AS ptt_first,
        MAX(IF(feature_name='ptt',                valuenum, NULL))                                                          AS ptt_max,

        ARRAY_AGG(IF(feature_name='platelets',    valuenum, NULL) IGNORE NULLS ORDER BY charttime LIMIT 1)[SAFE_OFFSET(0)] AS platelets_first,
        MAX(IF(feature_name='platelets',          valuenum, NULL))                                                          AS platelets_max,

        ARRAY_AGG(IF(feature_name='fibrinogen',   valuenum, NULL) IGNORE NULLS ORDER BY charttime LIMIT 1)[SAFE_OFFSET(0)] AS fibrinogen_first,
        MAX(IF(feature_name='fibrinogen',         valuenum, NULL))                                                          AS fibrinogen_max,

        ARRAY_AGG(IF(feature_name='d_dimer',      valuenum, NULL) IGNORE NULLS ORDER BY charttime LIMIT 1)[SAFE_OFFSET(0)] AS d_dimer_first,
        MAX(IF(feature_name='d_dimer',            valuenum, NULL))                                                          AS d_dimer_max,

        -- ── Blood gas / acid-base ────────────────────────────
        ARRAY_AGG(IF(feature_name='ph',           valuenum, NULL) IGNORE NULLS ORDER BY charttime LIMIT 1)[SAFE_OFFSET(0)] AS ph_first,
        MAX(IF(feature_name='ph',                 valuenum, NULL))                                                          AS ph_max,

        ARRAY_AGG(IF(feature_name='po2',          valuenum, NULL) IGNORE NULLS ORDER BY charttime LIMIT 1)[SAFE_OFFSET(0)] AS po2_first,
        MAX(IF(feature_name='po2',                valuenum, NULL))                                                          AS po2_max,

        ARRAY_AGG(IF(feature_name='pco2',         valuenum, NULL) IGNORE NULLS ORDER BY charttime LIMIT 1)[SAFE_OFFSET(0)] AS pco2_first,
        MAX(IF(feature_name='pco2',               valuenum, NULL))                                                          AS pco2_max,

        ARRAY_AGG(IF(feature_name='bicarbonate',  valuenum, NULL) IGNORE NULLS ORDER BY charttime LIMIT 1)[SAFE_OFFSET(0)] AS bicarbonate_first,
        MAX(IF(feature_name='bicarbonate',        valuenum, NULL))                                                          AS bicarbonate_max,

        ARRAY_AGG(IF(feature_name='hemoglobin',   valuenum, NULL) IGNORE NULLS ORDER BY charttime LIMIT 1)[SAFE_OFFSET(0)] AS hemoglobin_first,
        MAX(IF(feature_name='hemoglobin',         valuenum, NULL))                                                          AS hemoglobin_max,

        ARRAY_AGG(IF(feature_name='hematocrit',   valuenum, NULL) IGNORE NULLS ORDER BY charttime LIMIT 1)[SAFE_OFFSET(0)] AS hematocrit_first,
        MAX(IF(feature_name='hematocrit',         valuenum, NULL))                                                          AS hematocrit_max,

        -- ── Electrolytes ─────────────────────────────────────
        ARRAY_AGG(IF(feature_name='sodium',       valuenum, NULL) IGNORE NULLS ORDER BY charttime LIMIT 1)[SAFE_OFFSET(0)] AS sodium_first,
        MAX(IF(feature_name='sodium',             valuenum, NULL))                                                          AS sodium_max,

        ARRAY_AGG(IF(feature_name='potassium',    valuenum, NULL) IGNORE NULLS ORDER BY charttime LIMIT 1)[SAFE_OFFSET(0)] AS potassium_first,
        MAX(IF(feature_name='potassium',          valuenum, NULL))                                                          AS potassium_max,

        ARRAY_AGG(IF(feature_name='chloride',     valuenum, NULL) IGNORE NULLS ORDER BY charttime LIMIT 1)[SAFE_OFFSET(0)] AS chloride_first,
        MAX(IF(feature_name='chloride',           valuenum, NULL))                                                          AS chloride_max,

        ARRAY_AGG(IF(feature_name='calcium',      valuenum, NULL) IGNORE NULLS ORDER BY charttime LIMIT 1)[SAFE_OFFSET(0)] AS calcium_first,
        MAX(IF(feature_name='calcium',            valuenum, NULL))                                                          AS calcium_max,

        ARRAY_AGG(IF(feature_name='magnesium',    valuenum, NULL) IGNORE NULLS ORDER BY charttime LIMIT 1)[SAFE_OFFSET(0)] AS magnesium_first,
        MAX(IF(feature_name='magnesium',          valuenum, NULL))                                                          AS magnesium_max,

        ARRAY_AGG(IF(feature_name='phosphate',    valuenum, NULL) IGNORE NULLS ORDER BY charttime LIMIT 1)[SAFE_OFFSET(0)] AS phosphate_first,
        MAX(IF(feature_name='phosphate',          valuenum, NULL))                                                          AS phosphate_max

    FROM tagged_labs
    WHERE feature_name IS NOT NULL
    GROUP BY hadm_id
),

-- ── 4. TARGET VARIABLE ───────────────────────────────────────
-- had_resistance = 1 if ANY culture during this admission showed R
-- organisms_cultured and specimen_types are metadata — drop before training
micro_target AS (
    SELECT
        hadm_id,
        MAX(IF(interpretation = 'R', 1, 0))             AS had_resistance,
        STRING_AGG(DISTINCT org_name, ', ') AS organisms_cultured,
        STRING_AGG(DISTINCT spec_type_desc, ', ') AS specimen_types
    FROM `physionet-data.mimiciv_3_1_hosp.microbiologyevents`
    WHERE hadm_id IS NOT NULL
    GROUP BY hadm_id
)

-- ── 5. FINAL ML-READY MATRIX ─────────────────────────────────
-- Row   = one hospital admission
-- Cols  = identifiers | demographics | lab features | target
--
-- Notes for preprocessing in Python:
--   • NULLs in lab columns = lab not measured → impute (median / MICE)
--   • gender, race, insurance, admission_type → encode categorically
--   • Drop: hadm_id, subject_id, organisms_cultured, specimen_types
--   • had_resistance = 0 where no cultures were taken (COALESCE below)
SELECT

    -- ── Identifiers (drop before training) ───────────────────
    d.subject_id,
    d.hadm_id,

    -- ── Demographics ─────────────────────────────────────────
    d.age,
    d.gender,                   -- M / F  → encode
    d.race,                     -- ~30 categories → group or encode
    d.insurance,                -- Medicare / Medicaid / Other → encode
    d.admission_type,           -- EMERGENCY / ELECTIVE / URGENT → encode
    d.los_hours,                -- length of stay (potential data leakage — see note)
    d.hospital_expire_flag,     -- confounder; include with caution

    -- ── Infection markers ────────────────────────────────────
    lf.wbc_first,           lf.wbc_max,
    lf.bands_first,         lf.bands_max,
    lf.neutrophils_first,   lf.neutrophils_max,
    lf.lymphocytes_first,   lf.lymphocytes_max,
    lf.monocytes_first,     lf.monocytes_max,

    -- ── Inflammatory markers ─────────────────────────────────
    lf.crp_first,           lf.crp_max,
    lf.procalcitonin_first, lf.procalcitonin_max,
    lf.ferritin_first,      lf.ferritin_max,

    -- ── Organ dysfunction ────────────────────────────────────
    lf.lactate_first,           lf.lactate_max,
    lf.creatinine_first,        lf.creatinine_max,
    lf.bun_first,               lf.bun_max,
    lf.bilirubin_total_first,   lf.bilirubin_total_max,
    lf.alt_first,               lf.alt_max,
    lf.ast_first,               lf.ast_max,
    lf.albumin_first,           lf.albumin_max,
    lf.glucose_first,           lf.glucose_max,

    -- ── Coagulation ──────────────────────────────────────────
    lf.inr_first,           lf.inr_max,
    lf.ptt_first,           lf.ptt_max,
    lf.platelets_first,     lf.platelets_max,
    lf.fibrinogen_first,    lf.fibrinogen_max,
    lf.d_dimer_first,       lf.d_dimer_max,

    -- ── Blood gas / acid-base ────────────────────────────────
    lf.ph_first,            lf.ph_max,
    lf.po2_first,           lf.po2_max,
    lf.pco2_first,          lf.pco2_max,
    lf.bicarbonate_first,   lf.bicarbonate_max,
    lf.hemoglobin_first,    lf.hemoglobin_max,
    lf.hematocrit_first,    lf.hematocrit_max,

    -- ── Electrolytes ─────────────────────────────────────────
    lf.sodium_first,        lf.sodium_max,
    lf.potassium_first,     lf.potassium_max,
    lf.chloride_first,      lf.chloride_max,
    lf.calcium_first,       lf.calcium_max,
    lf.magnesium_first,     lf.magnesium_max,
    lf.phosphate_first,     lf.phosphate_max,

    -- ── Metadata (drop before training) ──────────────────────
    mt.organisms_cultured,
    mt.specimen_types,

    -- ── TARGET ───────────────────────────────────────────────
    -- INNER JOIN above guarantees every row had a culture taken
    -- 1 = at least one organism showed R; 0 = all cultures were S or I
    mt.had_resistance

FROM demographics d
INNER JOIN lab_features lf  ON lf.hadm_id = d.hadm_id
INNER JOIN micro_target mt  ON mt.hadm_id = d.hadm_id
-- INNER JOIN ensures only admissions where at least one culture was taken
-- are included. Admissions with no microbiologyevents row are excluded.

ORDER BY d.subject_id, d.hadm_id;
