#!/usr/bin/env python3
"""
Build a harmonized master parquet from Softlab & SILP Excel extractions.
Lab results: creatinine, CBC, troponin, BNP, electrolytes (2015-2025).
Uses calamine engine for fast xlsx reading.

Key fixes vs original:
- Split lab-code mapping by source (SOFTLAB vs SILP) to avoid code collisions.
- Explicitly handles the following collisions :
    HB1:
      Softlab = Hémoglobine
      SILP    = HLA-B1 (SSP)
    HYPO2:
      Softlab = Hypochromie
      SILP    = Test de tolérance au glucose 2h suivi FKP
    CL2:
      Softlab = Chlorure
      SILP    = Clairance de la créatinine; 2 h
- Keeps unmapped codes as UNMAPPED (QC-friendly).
- Robust numeric parsing + robust datetime parsing (Excel serial fallback).
"""

import warnings, gc, time
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd
import numpy as np
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    print(msg, flush=True)

SRC = Path("/media//data1/datasets/softlab")
OUT = Path("/volume/softlab")
OUT.mkdir(exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Column schemas
# ──────────────────────────────────────────────────────────────────────────────

SOFTLAB_COLS = [
    "last_name", "first_name", "order_id", "patient_id",
    "test_dt", "verified_dt",
    "group_test_id", "test_id", "result", "units", "status", "state"
]

SILP_COLS = [
    "last_name", "first_name", "patient_id", "order_id",
    "test_dt", "verified_dt", "collect_dt", "receive_dt",
    "group_test_id", "test_id", "result", "units", "status", "state"
]

FINAL_COLS = [
    "patient_id", "order_id", "test_dt", "verified_dt",
    "group_test_id", "test_id", "result_raw", "result_numeric",
    "units", "status", "state",
    "lab_category", "lab_name",
    "unmapped_flag",
    "source_file"
]


# ──────────────────────────────────────────────────────────────────────────────
# Lab mappings (source-specific!)
# ──────────────────────────────────────────────────────────────────────────────

LAB_MAP_SOFTLAB = {
    # --- Creatinine ---
    ("CREA", "CREAS"):  ("Creatinine", "Creatinine (serum)"),
    ("CREA", "CREI2"):  ("Creatinine", "Creatinine"),
    ("CREA", "DFGE4"):  ("Creatinine", "eGFR (MDRD)"),
    ("CREA", "DFGE5"):  ("Creatinine", "eGFR (MDRD)"),
    ("CREA", "ASP1"):   ("Creatinine", "Specimen appearance"),

    # --- Troponin ---
    ("TTROP", "TROT"):  ("Troponin", "Troponin T"),

    # --- BNP ---
    ("BNP",  "BNP2"):   ("BNP", "NT-proBNP"),
    ("NBNP", "NBNP"):   ("BNP", "NT-proBNP"),

    # --- Electrolytes ---
    ("NAKCL", "NA2"):   ("Electrolytes", "Sodium"),
    ("NAKCL", "K2"):    ("Electrolytes", "Potassium"),
    ("NAKCL", "CL2"):   ("Electrolytes", "Chloride"),     # Softlab meaning
    ("NAKCL", "ASP1"):  ("Electrolytes", "Specimen appearance"),
    ("ELEC",  "NA1"):   ("Electrolytes", "Sodium"),
    ("ELEC",  "K1"):    ("Electrolytes", "Potassium"),
    ("ELEC",  "CHLO"):  ("Electrolytes", "Chloride"),

    # --- CBC (FSC = Formule Sanguine Complete) ---
    ("FSC", "GBA"):     ("CBC", "WBC (absolute)"),
    ("FSC", "GBC2"):    ("CBC", "WBC (uncorrected)"),
    ("FSC", "LEU1"):    ("CBC", "WBC"),
    ("FSC", "ERY1"):    ("CBC", "RBC"),
    ("FSC", "HB"):      ("CBC", "Hemoglobin"),
    ("FSC", "HB1"):     ("CBC", "Hemoglobin"),            # Softlab meaning 
    ("FSC", "HT"):      ("CBC", "Hematocrit"),
    ("FSC", "HT1"):     ("CBC", "Hematocrit"),
    ("FSC", "PLA1"):    ("CBC", "Platelets"),
    ("FSC", "PLAE1"):   ("CBC", "Platelet estimate"),
    ("FSC", "PLASA"):   ("CBC", "Platelet (auto)"),
    ("FSC", "PLTA"):    ("CBC", "Platelet (auto)"),
    ("FSC", "VGM1"):    ("CBC", "MCV"),
    ("FSC", "VGMA"):    ("CBC", "MCV (auto)"),
    ("FSC", "TGMH1"):   ("CBC", "MCH"),
    ("FSC", "TGMHA"):   ("CBC", "MCH (auto)"),
    ("FSC", "CGMH1"):   ("CBC", "MCHC"),
    ("FSC", "CGMHA"):   ("CBC", "MCHC (auto)"),
    ("FSC", "NEA1"):    ("CBC", "Neutrophils (abs)"),
    ("FSC", "NER1"):    ("CBC", "Neutrophils (%)"),
    ("FSC", "NEUA"):    ("CBC", "Neutrophils (auto abs)"),
    ("FSC", "NEUAP"):   ("CBC", "Neutrophils (auto %)"),
    ("FSC", "NEUS"):    ("CBC", "Neutrophils (stab)"),
    ("FSC", "NEUT"):    ("CBC", "Neutrophils"),
    ("FSC", "NESTA"):   ("CBC", "Neutrophils (stab auto)"),
    ("FSC", "LYA1"):    ("CBC", "Lymphocytes (abs)"),
    ("FSC", "LYR1"):    ("CBC", "Lymphocytes (%)"),
    ("FSC", "LYMA"):    ("CBC", "Lymphocytes (auto abs)"),
    ("FSC", "LYMAN"):   ("CBC", "Lymphocytes (auto abs)"),
    ("FSC", "LYMAP"):   ("CBC", "Lymphocytes (auto %)"),
    ("FSC", "LYMP"):    ("CBC", "Lymphocytes"),
    ("FSC", "LYMVA"):   ("CBC", "Lymphocytes variant (auto)"),
    ("FSC", "VARLY"):   ("CBC", "Lymphocytes variant"),
    ("FSC", "MOA1"):    ("CBC", "Monocytes (abs)"),
    ("FSC", "MOR1"):    ("CBC", "Monocytes (%)"),
    ("FSC", "MONA"):    ("CBC", "Monocytes (auto abs)"),
    ("FSC", "MONAP"):   ("CBC", "Monocytes (auto %)"),
    ("FSC", "MONCT"):   ("CBC", "Monocytes"),
    ("FSC", "MORA2"):   ("CBC", "Morphology"),
    ("FSC", "MORPW"):   ("CBC", "Morphology (PW)"),
    ("FSC", "EOA1"):    ("CBC", "Eosinophils (abs)"),
    ("FSC", "EOR1"):    ("CBC", "Eosinophils (%)"),
    ("FSC", "EOSA"):    ("CBC", "Eosinophils (auto abs)"),
    ("FSC", "EOSAP"):   ("CBC", "Eosinophils (auto %)"),
    ("FSC", "EOSI"):    ("CBC", "Eosinophils"),
    ("FSC", "BAA1"):    ("CBC", "Basophils (abs)"),
    ("FSC", "BAR1"):    ("CBC", "Basophils (%)"),
    ("FSC", "BASA"):    ("CBC", "Basophils (auto abs)"),
    ("FSC", "BASAP"):   ("CBC", "Basophils (auto %)"),
    ("FSC", "BASO"):    ("CBC", "Basophils"),
    ("FSC", "DVE1"):    ("CBC", "RDW"),
    ("FSC", "DVEAP"):   ("CBC", "RDW (auto)"),
    ("FSC", "VPM1"):    ("CBC", "MPV"),
    ("FSC", "VPMA"):    ("CBC", "MPV (auto)"),
    ("FSC", "PDW1"):    ("CBC", "PDW"),
    ("FSC", "PCT1"):    ("CBC", "Plateletcrit"),
    ("FSC", "NRBC1"):   ("CBC", "NRBC"),
    ("FSC", "NRBCS"):   ("CBC", "Nucleated RBC suspect"),
    ("FSC", "GRA"):     ("CBC", "Granulocytes (auto abs)"),
    ("FSC", "GRIAP"):   ("CBC", "Immature granulocytes (auto %)"),
    ("FSC", "GRIM"):    ("CBC", "Immature granulocytes"),
    ("FSC", "GRIMN"):   ("CBC", "Immature granulocytes (manual)"),
    ("FSC", "GRN3"):    ("CBC", "Nucleated RBC (%)"),
    ("FSC", "GRN4"):    ("CBC", "Nucleated RBC (abs)"),
    ("FSC", "GRNA"):    ("CBC", "Granulocytes (auto)"),
    ("FSC", "GRNLA"):   ("CBC", "Large granulocytes (auto)"),
    ("FSC", "GRNUC"):   ("CBC", "Granulocytes (nucleated)"),
    ("FSC", "BLAA"):    ("CBC", "Blasts (auto)"),
    ("FSC", "BLASL"):   ("CBC", "Blasts"),
    ("FSC", "META"):    ("CBC", "Metamyelocytes"),
    ("FSC", "MYC"):     ("CBC", "Myelocytes"),
    ("FSC", "PRYA"):    ("CBC", "Promyelocytes (auto)"),
    ("FSC", "LSHIF"):   ("CBC", "Left shift"),
    ("FSC", "IMMG"):    ("CBC", "Immature granulocytes"),
    ("FSC", "SMA"):     ("CBC", "Smudge cells (auto)"),
    ("FSC", "ANIS2"):   ("CBC", "Anisocytosis"),
    ("FSC", "ANOH1"):   ("CBC", "Abnormal Hemoglobin"),
    ("FSC", "ANOR1"):   ("CBC", "Abnormal Reticulocyte Profile"),
    ("FSC", "DIMR"):    ("CBC", "Dimorphic RBC"),
    # HYPO2: Softlab = Hypochromie
    ("FSC", "HYPO2"):   ("CBC", "Hypochromia"),
    ("FSC", "MACR2"):   ("CBC", "Macrocytosis"),
    ("FSC", "MICR2"):   ("CBC", "Microcytosis"),
    ("FSC", "PANC2"):   ("CBC", "Pancytopenia"),
    ("FSC", "SAER2"):   ("CBC", "Erythrocyte Aggregation"),
    ("FSC", "SAPL2"):   ("CBC", "Platelet Aggregation"),
    ("FSC", "SBLA2"):   ("CBC", "Blasts"),
    ("FSC", "SCEL2"):   ("CBC", "Sickle Cells (Falciform)"),
    ("FSC", "SHIS3"):   ("CBC", "RBC Fragments/Microcytes"),
    ("FSC", "SPOI2"):   ("CBC", "Poikilocytosis"),
    ("FSC", "COM1"):    ("CBC", "Comment 1"),
    ("FSC", "COM4"):    ("CBC", "Comment 4"),
    ("FSC", "DIFM1"):   ("CBC", "Manual diff"),
    ("FSC", "DIFM5"):   ("CBC", "Manual diff 5"),
    ("FSC", "DIFMA"):   ("CBC", "Manual diff (auto)"),
    ("FSC", "RCODE"):   ("CBC", "Result code"),
    ("FSC", "VA1"):     ("CBC", "Flag VA1"),
    ("FSC", "VA3"):     ("CBC", "Flag VA3"),
    ("FSC", "VA4"):     ("CBC", "Flag VA4"),
    ("FSC", "VR1"):     ("CBC", "Flag VR1"),
}

LAB_MAP_SILP = {
    # Electrolytes (prefer CHLO for chloride; CL2 is creatinine clearance here)
    ("ELEC", "NA1"):   ("Electrolytes", "Sodium"),
    ("ELEC", "K1"):    ("Electrolytes", "Potassium"),
    ("ELEC", "CHLO"):  ("Electrolytes", "Chloride"),

    ("NAKCL", "NA2"):  ("Electrolytes", "Sodium"),
    ("NAKCL", "K2"):   ("Electrolytes", "Potassium (2nd sample)"),
    ("NAKCL", "CL2"):  ("Creatinine", "Creatinine clearance (2h)"),  # SILP meaning 

    # Troponin / BNP
    ("TTROP", "TROT"): ("Troponin", "Troponin T"),
    ("BNP",  "BNP2"):  ("BNP", "NT-proBNP"),
    ("NBNP", "NBNP"):  ("BNP", "NT-proBNP"),

    # Creatinine group
    ("CREA", "CREAS"): ("Creatinine", "Creatinine (serum)"),
    ("CREA", "CREI2"): ("Creatinine", "Creatinine"),
    ("CREA", "DFGE4"): ("Creatinine", "eGFR (MDRD)"),
    ("CREA", "DFGE5"): ("Creatinine", "eGFR (MDRD)"),
    ("CREA", "ASP1"):  ("Creatinine", "Specimen appearance"),

    # Collisions you confirmed
    ("FSC", "HB"):     ("CBC", "Hemoglobin"),
    ("FSC", "HB1"):    ("Immunology", "HLA-B1 (SSP)"),         # SILP meaning 
    ("FSC", "HYPO2"):  ("Glucose", "GTT 2h (suivi FKP)"),       # SILP meaning 



def build_map_df(lab_map: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [{"group_test_id": k[0], "test_id": k[1], "lab_category": v[0], "lab_name": v[1]}
         for k, v in lab_map.items()]
    )


# ──────────────────────────────────────────────────────────────────────────────
# Parsing helpers
# ──────────────────────────────────────────────────────────────────────────────

def parse_numeric_vec(series: pd.Series) -> pd.Series:
    """
    Vectorized parsing of numeric lab values in French formats:
    - handles commas as decimals
    - strips '<' '>' prefixes
    - extracts first numeric token from strings like "12,3 mg/L"
    """
    s = series.astype(str).str.strip()
    s_lower = s.str.lower()

    bad = s_lower.isin([".", "-", "", "----", "n/a", "na", "normal", "see comment", "nan", "none"])

    s = s.str.replace(r"^[<>]\s*", "", regex=True)
    s = s.str.extract(r"([-+]?\d+(?:[.,]\d+)?)", expand=False)
    s = s.str.replace(",", ".", regex=False)

    out = pd.to_numeric(s, errors="coerce")
    out[bad] = np.nan
    return out


def parse_dt(series: pd.Series) -> pd.Series:
    """
    Robust datetime parsing:
    1) standard to_datetime
    2) if many NaT and series looks numeric, try Excel serial dates.
    """
    dt = pd.to_datetime(series, errors="coerce")

    if dt.isna().mean() > 0.5:
        num = pd.to_numeric(series, errors="coerce")
        dt2 = pd.to_datetime(num, unit="D", origin="1899-12-30", errors="coerce")
        dt = dt.fillna(dt2)

    return dt


# ──────────────────────────────────────────────────────────────────────────────
# IO
# ──────────────────────────────────────────────────────────────────────────────

def read_file_calamine(filepath: Path, source_tag: str, is_silp: bool = False) -> pd.DataFrame:
    """Read all data sheets from an xlsx using calamine engine."""
    log(f"  Reading {filepath.name} ...")
    t0 = time.time()

    sheets_dict = pd.read_excel(filepath, sheet_name=None, header=None, engine="calamine")
    log(f"    Excel parse: {time.time()-t0:.0f}s")

    chunks = []
    for sn, df in sheets_dict.items():
        if str(sn).strip().upper() == "SQL":
            continue
        if df.shape[1] == 0:
            continue

        # Filter header-ish rows by first column text
        df0 = df[0].astype(str).str.strip().str.upper()
        df = df[~df0.isin(["LAST_NAME", "NAN", "NONE", ""])]

        # Drop fully empty rows
        df = df.dropna(subset=[0], how="all")
        if len(df) == 0:
            continue

        if is_silp:
            if df.shape[1] < 14:
                continue
            df = df.iloc[:, :14]
            df.columns = SILP_COLS
            df = df.drop(columns=["collect_dt", "receive_dt"])
        else:
            if df.shape[1] < 12:
                continue
            df = df.iloc[:, :12]
            df.columns = SOFTLAB_COLS

        df["source_file"] = source_tag
        chunks.append(df)
        log(f"    {sn}: {len(df):,} rows")

    del sheets_dict
    gc.collect()

    if not chunks:
        base_cols = SILP_COLS if is_silp else SOFTLAB_COLS
        return pd.DataFrame(columns=base_cols + ["source_file"])

    out = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()

    log(f"    TOTAL: {len(out):,} rows ({time.time()-t0:.0f}s)")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Harmonization
# ──────────────────────────────────────────────────────────────────────────────

def harmonize(df: pd.DataFrame) -> pd.DataFrame:
    """Apply source-specific lab mapping, parse numerics, produce final schema."""
    df["group_test_id"] = df["group_test_id"].astype(str).str.strip()
    df["test_id"] = df["test_id"].astype(str).str.strip()

    # Decide source by tag
    is_silp = df["source_file"].astype(str).str.contains("silp", case=False, na=False)

    map_soft = build_map_df(LAB_MAP_SOFTLAB)
    map_silp = build_map_df(LAB_MAP_SILP)

    df_soft = df.loc[~is_silp].merge(map_soft, on=["group_test_id", "test_id"], how="left")
    df_silp = df.loc[is_silp].merge(map_silp, on=["group_test_id", "test_id"], how="left")
    df = pd.concat([df_soft, df_silp], ignore_index=True)

    # Unmapped handling (QC-friendly)
    df["unmapped_flag"] = df["lab_category"].isna()
    df.loc[df["unmapped_flag"], "lab_category"] = "UNMAPPED"
    df.loc[df["unmapped_flag"], "lab_name"] = (
        df.loc[df["unmapped_flag"], "group_test_id"] + "::" + df.loc[df["unmapped_flag"], "test_id"]
    )

    # Results
    df["result_raw"] = df["result"].astype(str)
    df["result_numeric"] = parse_numeric_vec(df["result"])

    # Dates
    df["test_dt"] = parse_dt(df["test_dt"])
    df["verified_dt"] = parse_dt(df["verified_dt"])

    # IDs
    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    df["order_id"] = df["order_id"].astype(str).str.strip()

    # Light normalization
    df["units"] = df["units"].astype(str).str.strip()
    df["status"] = df["status"].astype(str).str.strip()
    df["state"] = df["state"].astype(str).str.strip()

    return df[FINAL_COLS].copy()


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log("=" * 60)
    log("Building master parquet from Softlab/SILP extractions")
    log("=" * 60)
    t_start = time.time()

    all_dfs = []

    # Softlab files
    softlab_files = [
        (SRC / "Softlab extraction 15-17.xlsx", "softlab_15-17"),
        (SRC / "Softlab extraction 18-20.xlsx", "softlab_18-20"),
        (SRC / "Softlab extraction 21-23.xlsx", "softlab_21-23"),
        (SRC / "Softlab extraction 24-25.xlsx", "softlab_24-25"),
    ]

    for fpath, tag in softlab_files:
        df = read_file_calamine(fpath, tag, is_silp=False)
        log(f"  Harmonizing {tag} ...")
        df = harmonize(df)
        all_dfs.append(df)
        log(f"  -> {tag}: {len(df):,} harmonized rows")
        del df
        gc.collect()

    # SILP file
    silp_path = SRC / "SILP Extraction Mai 2025 - Décembre 2025.xlsx"
    df = read_file_calamine(silp_path, "silp_2025", is_silp=True)
    log("  Harmonizing silp_2025 ...")
    df = harmonize(df)
    all_dfs.append(df)
    log(f"  -> silp_2025: {len(df):,} harmonized rows")
    del df
    gc.collect()

    # Concatenate
    log("\nConcatenating all files ...")
    master = pd.concat(all_dfs, ignore_index=True)
    del all_dfs
    gc.collect()

    log(f"\n{'=' * 60}")
    log(f"Master dataset: {len(master):,} rows")
    log(f"Date range: {master['test_dt'].min()} -> {master['test_dt'].max()}")
    log(f"Unique patients: {master['patient_id'].nunique():,}")

    log("\nLab categories (top 30):")
    log(master["lab_category"].value_counts().head(30).to_string())

    log("\nUnmapped codes (top 30):")
    log(master.loc[master["unmapped_flag"], "lab_name"].value_counts().head(30).to_string())

    log(
        f"\nNumeric result coverage: {master['result_numeric'].notna().sum():,} / {len(master):,} "
        f"({100*master['result_numeric'].notna().mean():.1f}%)"
    )

    # Save parquet
    out_path = OUT / "softlab_master.parquet"
    log(f"\nSaving to {out_path} ...")
    master.to_parquet(out_path, index=False, engine="pyarrow")
    log(f"Saved: {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")

    # Day-of-year coverage
    master["date"] = master["test_dt"].dt.date
    days_covered = master["date"].nunique()
    log(f"Unique dates with data: {days_covered:,}")

    elapsed = time.time() - t_start
    log(f"\nTotal elapsed: {elapsed/60:.1f} minutes")
    log("Done!")
