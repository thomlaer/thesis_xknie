import re
import pandas as pd
from config import llm_source, gold_mn, gold_final, llm_labels, final_dataset

niet_knie = re.compile(
    r'(?m)^\s*(?:DX|X)\s*[- ]?\s*'
    r'(?:Bekken|Heup(?:en)?|Femur|Thorax|Clavicula|Schouder|Elleboog|Pols|Hand|Voet|Enkel'
    r'|Rug|Wervelkolom|Cervicaal|Lumbaal|Tibia|Fibula|Humerus|Radius|Ulna|Ribben?'
    r'|Bovenarm|Onderarm|Bovenbeen|Onderbeen)',
    re.IGNORECASE,
)

knie_sectie = re.compile(r'(?m)^\s*(?:DX|X)\s*[- ]?\s*Knie', re.IGNORECASE)

gold_rename = {
    "definitief_handmatig_label":  "handmatig_label",
    "definitief_artrose_aanwezig": "handmatig_artrose_aanwezig",
    "definitief_artrose_graad":    "handmatig_artrose_graad",
    "definitief_nhg":              "handmatig_nhg",
    "herkomst":                    "gold_herkomst",
}

gold_columns = [
    "case_id", "handmatig_label", "handmatig_artrose_aanwezig",
    "handmatig_artrose_graad", "handmatig_nhg",
    "label_martijn", "label_heleen",
    "label_artrose_martijn", "label_artrose_heleen",
    "artrose_graad_martijn", "artrose_graad_heleen",
    "nhg_martijn", "nhg_heleen", "gold_herkomst",
]

front_cols = [
    "case_id", "pacs_tekst_origineel", "datum", "onderzoek", "zijde",
    "klinische_gegevens_huisarts", "vraagstelling_huisarts",
    "bevindingen", "bevindingen_origineel", "conclusie",
    "is_split", "tekst_origineel",
]

back_cols = [
    "label_meerdere_raw", "label_meerdere", "excl_addendum",
    "excl_lr_niet_gesplitst", "excl_lr_gesplitst", "excl_lr_totaal",
    "excl_protesis", "excl_meerdere", "excl_lege_klinische_tekst",
    "excl_totaal_mumc",
]

required_llm_columns = ["artrose_radiologie", "artrose_vraagstelling"]


def is_blank(v):
    return v is None or str(v).strip().lower() in ("", "nan")


def normalize_text(s):
    return re.sub(r'\s+', ' ', str(s)).strip().lower()


def extract_addendum_body(tekst):
    if not isinstance(tekst, str):
        return None
    m = re.match(r'LET\s+OP[!.]?\s+Dit\s+is\s+een\s+addendum[.!]?\s+[^\n]*\n+', tekst, re.IGNORECASE)
    if not m:
        m = re.match(r'addendum[.!]?\s+[^\n]*\n+', tekst, re.IGNORECASE)
    if m:
        body = tekst[m.end():].strip()
        return body if body else None
    return None


def flag_addendum_duplicates(df):
    excl = pd.Series(False, index=df.index)
    bodies = []
    for _, row in df.iterrows():
        body = extract_addendum_body(str(row.get("pacs_tekst_origineel", "") or ""))
        if body and len(body) > 100:
            bodies.append((normalize_text(body), row["case_id"]))
    if not bodies:
        return excl
    for idx, row in df.iterrows():
        raw = str(row.get("pacs_tekst_origineel", "") or "")
        if extract_addendum_body(raw) is not None:
            continue
        norm = normalize_text(raw)
        if len(norm) < 100:
            continue
        for norm_body, add_id in bodies:
            if norm in norm_body and row["case_id"] > add_id:
                excl.at[idx] = True
                break
    return excl


def heeft_meerdere_lichaamsdelen(tekst):
    if not isinstance(tekst, str):
        return False
    return bool(niet_knie.search(tekst)) and bool(knie_sectie.search(tekst))


def add_exclusion_columns(df):
    df["excl_lr_niet_gesplitst"]    = (df["zijde"] == "L+R") & (~df["is_split"].astype(bool))
    df["excl_lr_gesplitst"]         = df["is_split"].astype(bool)
    df["excl_lr_totaal"]            = df["excl_lr_niet_gesplitst"] | df["excl_lr_gesplitst"]
    df["excl_protesis"]             = df["label_protesis"] == "ja"
    df["label_meerdere_raw"]        = df["pacs_tekst_origineel"].apply(heeft_meerdere_lichaamsdelen)
    df["label_meerdere"]            = df["label_meerdere_raw"].map({True: "ja", False: "nee"})
    df["excl_meerdere"]             = df["label_meerdere_raw"]
    df["excl_lege_klinische_tekst"] = (
        df["klinische_gegevens_huisarts"].apply(is_blank) &
        df["vraagstelling_huisarts"].apply(is_blank)
    )
    df["excl_totaal_mumc"] = (
        df["excl_lr_totaal"] | df["excl_protesis"] | df["excl_meerdere"]
        | df["excl_addendum"] | df["excl_lege_klinische_tekst"]
    )
    return df


def build_llm_labels():
    df = pd.read_excel(llm_source)

    pacs = pd.read_excel(gold_mn, usecols=["nummer", "PACS rapport tekst"])
    pacs.rename(columns={"nummer": "case_id", "PACS rapport tekst": "pacs_tekst_origineel"}, inplace=True)
    df = df.merge(pacs, on="case_id", how="left")

    missing = [c for c in required_llm_columns if c not in df.columns]
    if missing:
        raise KeyError(f"missing expected columns in llm source: {missing}")

    df["excl_addendum"]        = flag_addendum_duplicates(df)
    df = add_exclusion_columns(df)

    midden = [c for c in df.columns if c not in front_cols + back_cols]
    df = df[front_cols + midden + back_cols]
    df.to_excel(llm_labels, index=False)
    return df


def build_final_dataset(df):
    gold = pd.read_excel(gold_final).rename(columns=gold_rename)
    if "label_protesis" in df.columns and "label_protesis" in gold.columns:
        gold = gold.drop(columns=["label_protesis"])
    gold_sel = gold[[c for c in gold_columns if c in gold.columns]].copy()
    gold_ids = set(gold_sel["case_id"].dropna())

    not_excluded = ~df["excl_totaal_mumc"].astype(bool)
    is_gold      = df["case_id"].isin(gold_ids)
    primary = df[not_excluded | is_gold].copy()

    result = primary.merge(gold_sel, on="case_id", how="left")

    if result["case_id"].duplicated().any():
        result = (result
                  .sort_values("excl_totaal_mumc")
                  .drop_duplicates(subset=["case_id"], keep="first")
                  .reset_index(drop=True))

    result.to_excel(final_dataset, index=False)
    return result


def main():
    print("building llm_labels.xlsx...")
    df = build_llm_labels()
    n_excl = int(df["excl_totaal_mumc"].sum())
    print(f"  total: {len(df)}, excluded: {n_excl}, primary: {len(df) - n_excl}")

    print("building x_knie_definitief.xlsx...")
    result = build_final_dataset(df)
    n_gold = result["handmatig_label"].notna().sum()
    print(f"  primary: {len(result)}, with gold labels: {n_gold}")


if __name__ == "__main__":
    main()
