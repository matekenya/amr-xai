# MIMIC-IV AMR Prediction Dataset

## Overview

This dataset is derived from the [MIMIC-IV](https://mimic.mit.edu/docs/iv/) clinical database (version 3.1), hosted on PhysioNet. It is designed for **machine learning research on antimicrobial resistance (AMR) prediction** using routinely collected hospital data.

Each row represents **one antibiotic susceptibility test** performed on a microbiology culture from a hospital admission. Patient demographics and laboratory results from the same admission are denormalised across rows, enabling direct use in tabular ML pipelines.

> **Access:** MIMIC-IV requires credentialed access via PhysioNet. Researchers must complete the required training and data use agreement before accessing the underlying data. See [https://physionet.org/content/mimiciv/](https://physionet.org/content/mimiciv/) for details.

---

## Source Tables

The dataset is assembled from the following tables in the `mimiciv_3_1_hosp` schema on BigQuery:

| Source Table | Role |
|---|---|
| `patients` | Patient demographics (age, gender, date of death) |
| `admissions` | Hospital admission metadata (race, insurance, admission type, timestamps) |
| `labevents` + `d_labitems` | Laboratory test results |
| `microbiologyevents` | Microbiology cultures, organisms grown, and antibiotic susceptibility testing |
| `prescriptions` | Medications prescribed during the admission |

---

## Row Structure

```
One row = one antibiotic × one organism × one culture × one admission
```

If a patient had a blood culture that grew *Klebsiella pneumoniae* and 10 antibiotics were tested against it, that produces 10 rows — one per antibiotic — each sharing the same patient demographics and lab values.

**Implication for modelling:** rows from the same admission are correlated. Use `GroupKFold` with `hadm_id` as the group key during cross-validation to prevent data leakage.

---

## Column Reference

### Identifiers *(drop before training)*

| Column | Type | Description |
|---|---|---|
| `subject_id` | INTEGER | Unique patient identifier |
| `hadm_id` | INTEGER | Unique hospital admission identifier |
| `micro_specimen_id` | INTEGER | Unique specimen (culture) identifier |
| `isolate_num` | INTEGER | Isolate number; distinguishes multiple organisms from the same specimen |

---

### Culture Context

| Column | Type | Description |
|---|---|---|
| `culture_time` | TIMESTAMP | Date and time the specimen was collected |
| `specimen_type` | STRING | Type of specimen cultured (e.g. `BLOOD CULTURE`, `URINE`, `SPUTUM`) |
| `organism` | STRING | Organism grown (e.g. `Klebsiella pneumoniae`, `Escherichia coli`) |
| `antibiotic_tested` | STRING | Antibiotic tested against the organism (e.g. `Meropenem`, `Vancomycin`) |
| `mic_value` | STRING | Minimum inhibitory concentration (MIC) value if recorded (e.g. `<=0.5`) |

---

### Demographics

| Column | Type | Description |
|---|---|---|
| `age` | INTEGER | Patient age at time of admission (anchor age) |
| `gender` | STRING | Patient gender (`M` / `F`) |
| `race` | STRING | Patient race/ethnicity (~30 categories in MIMIC-IV) |
| `insurance` | STRING | Insurance type at admission (`Medicare`, `Medicaid`, `Other`) |
| `admission_type` | STRING | Type of admission (`EMERGENCY`, `ELECTIVE`, `URGENT`, etc.) |
| `los_hours` | INTEGER | Length of stay in hours (admittime → dischtime) |
| `hospital_expire_flag` | INTEGER | `1` if patient died during this admission, `0` otherwise |

> ⚠️ **`los_hours` data leakage warning:** length of stay is only known at discharge. If your model simulates a real-time clinical decision at admission, exclude this column.

---

### Laboratory Features

All lab features are aggregated at the admission level and denormalised across rows.

- **`_first`** suffix = first recorded value during the admission (baseline on arrival)
- **`_max`** suffix = peak value during the admission (severity marker)
- **`NULL`** = lab was not measured during this admission

#### Infection Markers

| Column Pair | Units | Clinical Relevance |
|---|---|---|
| `wbc_first` / `wbc_max` | 10⁹/L | Elevated in bacterial infection; leukopenia in severe sepsis |
| `bands_first` / `bands_max` | % | Immature neutrophils; elevated in acute bacterial infection |
| `neutrophils_first` / `neutrophils_max` | % | Primary cellular response to bacterial infection |
| `lymphocytes_first` / `lymphocytes_max` | % | Lymphopenia common in severe sepsis |
| `monocytes_first` / `monocytes_max` | % | Elevated in chronic or mycobacterial infection |

#### Inflammatory Markers

| Column Pair | Units | Clinical Relevance |
|---|---|---|
| `crp_first` / `crp_max` | mg/L | Acute phase reactant; strongly elevated in bacterial infection |
| `procalcitonin_first` / `procalcitonin_max` | ng/mL | Highly specific for bacterial sepsis; used to guide antibiotic therapy |
| `ferritin_first` / `ferritin_max` | ng/mL | Elevated in systemic inflammation and sepsis |

> ℹ️ These columns have **very high missingness** (60–97%) because they are targeted tests ordered only when infection is strongly suspected. The **presence or absence of these tests is itself clinically informative** — consider adding missingness indicator columns before imputing.

#### Organ Dysfunction / Sepsis

| Column Pair | Units | Clinical Relevance |
|---|---|---|
| `lactate_first` / `lactate_max` | mmol/L | Tissue hypoperfusion; key sepsis severity marker |
| `creatinine_first` / `creatinine_max` | mg/dL | Renal function; AKI common in sepsis |
| `bun_first` / `bun_max` | mg/dL | Renal function and hydration status |
| `bilirubin_total_first` / `bilirubin_total_max` | mg/dL | Liver dysfunction in sepsis |
| `alt_first` / `alt_max` | U/L | Hepatocellular injury |
| `ast_first` / `ast_max` | U/L | Hepatocellular injury; also elevated in muscle damage |
| `albumin_first` / `albumin_max` | g/dL | Low in chronic illness and sepsis; negative acute phase reactant |
| `glucose_first` / `glucose_max` | mg/dL | Hyperglycaemia common in sepsis and critical illness |

#### Coagulation

| Column Pair | Units | Clinical Relevance |
|---|---|---|
| `inr_first` / `inr_max` | ratio | Coagulopathy; elevated in DIC and liver failure |
| `ptt_first` / `ptt_max` | seconds | Clotting time; elevated in DIC |
| `platelets_first` / `platelets_max` | 10⁹/L | Thrombocytopenia common in sepsis and DIC |
| `fibrinogen_first` / `fibrinogen_max` | mg/dL | Acute phase reactant; consumed in DIC |
| `d_dimer_first` / `d_dimer_max` | mg/L FEU | Elevated in thrombosis, DIC, and sepsis |

#### Blood Gas / Acid-Base

| Column Pair | Units | Clinical Relevance |
|---|---|---|
| `ph_first` / `ph_max` | — | Acidosis indicates severe illness; alkalosis in early sepsis |
| `po2_first` / `po2_max` | mmHg | Oxygenation; low in respiratory failure |
| `pco2_first` / `pco2_max` | mmHg | Ventilation; low in compensatory hyperventilation |
| `bicarbonate_first` / `bicarbonate_max` | mmol/L | Metabolic acid-base balance |
| `hemoglobin_first` / `hemoglobin_max` | g/dL | Anaemia common in chronic infection and critical illness |
| `hematocrit_first` / `hematocrit_max` | % | Red cell volume fraction |

#### Electrolytes

| Column Pair | Units | Clinical Relevance |
|---|---|---|
| `sodium_first` / `sodium_max` | mmol/L | Dysnatraemia in sepsis and critical illness |
| `potassium_first` / `potassium_max` | mmol/L | Hyperkalaemia in renal failure; hypokalaemia in GI loss |
| `chloride_first` / `chloride_max` | mmol/L | Acid-base and hydration |
| `calcium_first` / `calcium_max` | mg/dL | Hypocalcaemia common in sepsis |
| `magnesium_first` / `magnesium_max` | mg/dL | Deficiency common in critical illness |
| `phosphate_first` / `phosphate_max` | mg/dL | Hypophosphataemia in sepsis and refeeding |

---

### Antibiotic Exposure Features

These columns capture the **total hours** each antibiotic class was prescribed during the admission. A value of `0` means the drug was never prescribed — this is a meaningful signal, not missing data. No imputation is needed for these columns.

| Column | Drug / Class | Notes |
|---|---|---|
| `abx_pip_tazo_hours` | Piperacillin-tazobactam | Broad-spectrum; common empirical choice |
| `abx_amp_sulbactam_hours` | Ampicillin-sulbactam | Moderate spectrum |
| `abx_ampicillin_hours` | Ampicillin | Narrow spectrum |
| `abx_oxacillin_hours` | Oxacillin | Anti-staphylococcal |
| `abx_nafcillin_hours` | Nafcillin | Anti-staphylococcal |
| `abx_cefazolin_hours` | Cefazolin | 1st generation cephalosporin |
| `abx_ceftriaxone_hours` | Ceftriaxone | 3rd generation cephalosporin |
| `abx_cefepime_hours` | Cefepime | 4th generation cephalosporin |
| `abx_ceftazidime_hours` | Ceftazidime | 3rd generation; anti-pseudomonal |
| `abx_cefdinir_hours` | Cefdinir | 3rd generation oral cephalosporin |
| `abx_meropenem_hours` | Meropenem | **Last-resort** carbapenem |
| `abx_imipenem_hours` | Imipenem | **Last-resort** carbapenem |
| `abx_ertapenem_hours` | Ertapenem | Carbapenem; no pseudomonal cover |
| `abx_vancomycin_hours` | Vancomycin | Glycopeptide; MRSA cover |
| `abx_gentamicin_hours` | Gentamicin | Aminoglycoside |
| `abx_amikacin_hours` | Amikacin | Aminoglycoside; broader resistance cover |
| `abx_tobramycin_hours` | Tobramycin | Aminoglycoside; anti-pseudomonal |
| `abx_ciprofloxacin_hours` | Ciprofloxacin | Fluoroquinolone |
| `abx_levofloxacin_hours` | Levofloxacin | Fluoroquinolone |
| `abx_moxifloxacin_hours` | Moxifloxacin | Fluoroquinolone |
| `abx_metronidazole_hours` | Metronidazole | Anaerobic cover |
| `abx_azithromycin_hours` | Azithromycin | Macrolide; atypical cover |
| `abx_clarithromycin_hours` | Clarithromycin | Macrolide |
| `abx_linezolid_hours` | Linezolid | **Last-resort** oxazolidinone; VRE / MRSA |
| `abx_daptomycin_hours` | Daptomycin | **Last-resort** lipopeptide |
| `abx_colistin_hours` | Colistin / Polymyxin | **Last-resort** polymyxin; CRE cover |
| `abx_doxycycline_hours` | Doxycycline | Tetracycline |
| `abx_tigecycline_hours` | Tigecycline | **Last-resort** tetracycline |
| `abx_tmp_smx_hours` | Trimethoprim-sulfamethoxazole | Sulfonamide |
| `abx_fluconazole_hours` | Fluconazole | Antifungal |
| `abx_micafungin_hours` | Micafungin | Antifungal echinocandin |
| `abx_caspofungin_hours` | Caspofungin | Antifungal echinocandin |
| `abx_unique_drugs_count` | — | Total distinct drugs prescribed (severity proxy) |
| `abx_total_exposure_hours` | — | Total antibiotic hours across all drugs |

---

### Target Variables

| Column | Type | Description |
|---|---|---|
| `interpretation` | STRING | Raw susceptibility result: `R` (Resistant), `S` (Susceptible), `I` (Intermediate). Drop before training. |
| `is_resistant` | INTEGER | **Primary ML target.** `1` = organism is resistant to `antibiotic_tested`; `0` = susceptible or intermediate |

---

## Cohort Definition

Admissions are included if they meet **all** of the following criteria:

1. At least one microbiology culture was taken during the admission
2. At least one antibiotic was tested against a grown organism (i.e. `ab_name IS NOT NULL`)
3. A susceptibility result (`interpretation`) was recorded
4. At least one laboratory result was recorded during the admission

Admissions with no cultures, no antibiotic susceptibility testing, or no lab data are excluded.

---

## Missingness

Lab columns contain `NULL` where a test was not performed. Missingness is **not random** — clinicians order targeted tests based on clinical suspicion. High missingness in inflammatory markers (CRP, procalcitonin, ferritin) indicates those tests are reserved for suspected infection cases.

**Recommended preprocessing strategy:**

```python
# Step 1: Add binary missingness indicators for all lab columns
# (the fact that a test was not ordered is itself a clinical signal)
for col in lab_cols:
    df[f'{col}_missing'] = df[col].isna().astype(int)

# Step 2: Impute remaining NULLs
# Baseline: median imputation (fast, tree-model friendly)
# Final:    MICE / IterativeImputer (captures inter-feature correlations)
```

Antibiotic exposure columns (`abx_*`) are **pre-filled with 0** and require no imputation.

---

## Recommended Preprocessing

```python
# Columns to drop before training
DROP_COLS = [
    'subject_id', 'hadm_id', 'micro_specimen_id',
    'isolate_num', 'culture_time', 'mic_value', 'interpretation'
]

# Columns to encode categorically
ENCODE_COLS = [
    'gender', 'race', 'insurance', 'admission_type',
    'specimen_type', 'organism', 'antibiotic_tested'
]

# Target
TARGET = 'is_resistant'

# Cross-validation: group by hadm_id to prevent leakage
# (rows from the same admission share lab values)
from sklearn.model_selection import GroupKFold
gkf = GroupKFold(n_splits=5)
for train_idx, test_idx in gkf.split(X, y, groups=df['hadm_id']):
    ...
```

---

## Class Balance

Expected approximate distribution across the full MIMIC-IV cohort:

| `is_resistant` | Meaning | Expected proportion |
|---|---|---|
| `0` | Susceptible or intermediate | ~65–75% |
| `1` | Resistant | ~25–35% |

Actual balance will vary depending on the antibiotic and organism. Some antibiotic–organism combinations (e.g. MRSA against methicillin) will be heavily skewed. Check per-antibiotic balance before training.

---

## Citation

If you use this dataset in your research, please cite the MIMIC-IV paper:

> Johnson, A.E.W., Bulgarelli, L., Shen, L., et al. (2023). MIMIC-IV, a freely accessible electronic health record dataset. *Scientific Data*, 10, 1. https://doi.org/10.1038/s41597-022-01899-x

---

## Contact

This dataset was prepared as part of PhD research on **Explainable AI for Clinical Decision Support in Antimicrobial Resistance** at Aalto University, Department of Industrial Engineering and Management.

For questions about the extraction query or feature engineering, refer to `mimic_amr_ml_matrix.sql` in this repository.
