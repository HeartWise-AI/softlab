#!/usr/bin/env python3
"""
Build a harmonized master parquet from Softlab & SILP Excel extractions.
Lab results: creatinine, CBC, troponin, BNP, electrolytes (2015-2025).
Uses calamine engine for fast xlsx reading.
"""

import sys, warnings, gc, time
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd
import numpy as np
from pathlib import Path

# Force unbuffered output
def log(msg):
    print(msg, flush=True)

SRC = Path("/media//data1/datasets/softlab")
OUT = Path("/volume/softlab")
OUT.mkdir(exist_ok=True)

# ── Column schema ──────────────────────────────────────────────────────────
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
    "units", "status", "state", "lab_category", "lab_name", "source_file"
]

# ── Lab mapping: (group_test_id, test_id) -> (category, human_name) ───────
LAB_MAP = {
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
    ("NAKCL", "CL2"):   ("Electrolytes", "Chloride"),
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
    ("FSC", "HB1"):     ("CBC", "Hemoglobin"),
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


def parse_numeric_vec(series):
    """Vectorized French-number parsing."""
    s = series.astype(str).str.strip()
    # Mark non-numeric
    bad = s.isin([".", "-", "", "----", "N/A", "NA", "Normal", "See comment", "nan", "None", "none"])
    # Strip < > prefixes
    s = s.str.lstrip("<>").str.strip()
    # French decimal -> dot
    s = s.str.replace(",", ".", regex=False)
    result = pd.to_numeric(s, errors="coerce")
    result[bad] = np.nan
    return result


def read_file_calamine(filepath, source_tag, is_silp=False):
    """Read all data sheets from an xlsx using calamine engine."""
    log(f"  Reading {filepath.name} ...")
    t0 = time.time()
    sheets_dict = pd.read_excel(filepath, sheet_name=None, header=None, engine="calamine")
    t1 = time.time()
    log(f"    Excel parse: {t1-t0:.0f}s")

    chunks = []
    for sn, df in sheets_dict.items():
        if sn == "SQL":
            continue
        # Convert first column to string to filter
        df[0] = df[0].astype(str)
        # Drop header rows and empty/nan rows
        df = df[~df[0].isin(["LAST_NAME", "nan", "None", ""])]
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
        cols = SOFTLAB_COLS + ["source_file"]
        return pd.DataFrame(columns=cols)

    result = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()
    log(f"    TOTAL: {len(result):,} rows ({time.time()-t0:.0f}s)")
    return result


def harmonize(df):
    """Apply lab mapping, parse numerics, produce final schema."""
    # Ensure string types for group/test id
    df["group_test_id"] = df["group_test_id"].astype(str).str.strip()
    df["test_id"] = df["test_id"].astype(str).str.strip()

    # Vectorized lab mapping via merge
    map_df = pd.DataFrame([
        {"group_test_id": k[0], "test_id": k[1], "lab_category": v[0], "lab_name": v[1]}
        for k, v in LAB_MAP.items()
    ])
    df = df.merge(map_df, on=["group_test_id", "test_id"], how="left")

    # Tag unmapped tests with their group/test id
    unmapped = df["lab_category"].isna()
    df.loc[unmapped, "lab_category"] = df.loc[unmapped, "group_test_id"]
    df.loc[unmapped, "lab_name"] = df.loc[unmapped, "test_id"]

    # Parse result
    df["result_raw"] = df["result"].astype(str)
    df["result_numeric"] = parse_numeric_vec(df["result"])

    # Parse dates
    df["test_dt"] = pd.to_datetime(df["test_dt"], errors="coerce")
    df["verified_dt"] = pd.to_datetime(df["verified_dt"], errors="coerce")

    # Clean IDs
    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    df["order_id"] = df["order_id"].astype(str).str.strip()

    df = df[FINAL_COLS].copy()
    return df


# ── MAIN ───────────────────────────────────────────────────────────────────
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
    log(f"\nLab categories:")
    log(master["lab_category"].value_counts().to_string())
    log(f"\nNumeric result coverage: {master['result_numeric'].notna().sum():,} / {len(master):,} "
        f"({100*master['result_numeric'].notna().mean():.1f}%)")

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
