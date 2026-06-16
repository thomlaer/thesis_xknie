import re
import pandas as pd
import openpyxl
from pathlib import Path

from config import raw_input, output as output_dir, clean_csv, clean_xlsx

input  = str(raw_input)
output = str(output_dir)

radio_start = re.compile(
    r'\b(?:'
    r'DX\s+Knie'
    r'|X\s*[- ]?\s*Knie'
    r'|R(?:\u00f6|\u00c3\u00b6|o)ntgen(?:opname)?'
    r'|Opname\s+van\s+de\s+knie'
    r'|Opnamen?\s+knie'
    r'|Knie(?:\u00ebn|en|\u00c3\u00abn|n)?\s+(?:links?|rechts?|beiderzijds)\s+in\s+\d+\s+richtingen'
    r'|AP\s+en\s+laterale\s+opname'
    r'|Normaal\s+trabeculair'
    r'|Normale\s+(?:stand|trabeculatie|kalkhoudendheid|patellofemorale)'
    r'|Intacte\s+corticali'
    r'|Intacte\s+corticale'
    r'|Geen\s+vergelijkend\s+beeldvorming'
    r'|Geen\s+duidelijke\s+gewricht'
    r'|Geen\s+hydrops'
    r'|Behouden\s+(?:hoogte\s+van\s+de\s+)?gewrichts'
    r'|Aanpunting\s+van'
    r'|Er\s+werd\s+vergeleken'
    r'|Neutrale\s+stand'
    r'|Normale\s+kalkhoudendheid'
    r')\b',
    re.IGNORECASE
)

bericht_sectie = re.compile(
    r'(?:^|\n)\s*(?::|(?:Samen|Verslag|N\s*slag|[0-9]+\s*slag|Knie(?:\u00ebn|en|\u00c3\u00abn|n)?|rechter\s*knie|linker\s*knie|rechterknie|linkerknie)\s*[.:\n])',
    re.IGNORECASE
)

dx_sectie = re.compile(
    r'^\s*(?:'
    r'DX\s+[A-Za-zÀ-ÿ][^\n]*'
    r'|X\s*[- ]?\s*[A-Za-zÀ-ÿ][^\n]*:'
    r'|X\s*/\s*CWK\s*:'
    r'|bekken\s*/\s*heupen\s*:'
    r'|bekken\s+en\s+(?:rechter|linker)?\s*femur\s*:'
    r'|(?:rechter|linker)?\s*(?:voet|enkel|bekken|heup|heupen|femur|hand|pols|wervelkolom|cwk|knie)\s*:'
    r'|(?:rechtervoet|linkervoet|rechterknie|linkerknie|rechterfemur|linkerfemur)\s*:'
    r')',
    re.IGNORECASE | re.MULTILINE
)

zijde_header = re.compile(
    r'^(?:\s*)('
    r'(?:DX\s+Knie|X\s*[- ]?\s*Knie|X-knie|Knie)\s+(?:rechts?|links?)\b[^\n]*'
    r'|(?:rechter\s*knie|linker\s*knie|rechterknie|linkerknie)\s*[:.]'
    r'|(?:rechts?|links?)\s*[:.]'
    r')',
    re.IGNORECASE | re.MULTILINE
)


def clean_text(tekst):
    if not isinstance(tekst, str):
        return None
    tekst = tekst.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    tekst = re.sub(r"[ \t]+", " ", tekst)
    tekst = re.sub(r"\n{3,}", "\n\n", tekst)
    tekst = re.sub(
        r'\n*Dit\s+verslag\s+is\s+gemaakt\s+met\s+spraakherkenning\b.*',
        '', tekst, flags=re.DOTALL | re.IGNORECASE
    )
    return tekst.strip() or None


def leeg_naar_none(waarde):
    if waarde is None:
        return None
    waarde = str(waarde).strip()
    if waarde == "" or waarde.lower() in ("nan", "none"):
        return None
    return waarde


def is_knie_header(header):
    return bool(re.search(r'\bknie(?:\u00ebn|en|\u00c3\u00abn|n)?\b', header, re.IGNORECASE))


def vorige_nietlege_regel(tekst, positie):
    voor = tekst[:positie].splitlines()
    for regel in reversed(voor):
        if regel.strip():
            return regel.strip()
    return ""


def beperk_tot_knie_onderzoek(tekst):
    matches = []
    for match in dx_sectie.finditer(tekst):
        vorige = vorige_nietlege_regel(tekst, match.start())
        if vorige.lower() in ("vraagstelling:", "klinische gegevens:", "medische gegevens:", "gevraagd onderzoek:"):
            continue
        matches.append(match)

    if not matches:
        return tekst

    knie_indices = [i for i, match in enumerate(matches) if is_knie_header(match.group(0))]
    if knie_indices:
        intro = tekst[:matches[0].start()].strip()
        knie_onderdelen = []
        for i in knie_indices:
            start = matches[i].start()
            einde = len(tekst)
            for j in range(i + 1, len(matches)):
                if not re.search(r'(?:rechter|linker)?knie', matches[j].group(0), re.IGNORECASE):
                    einde = matches[j].start()
                    break
            onderdeel = tekst[start:einde].strip()
            if onderdeel and '\n' in onderdeel:
                knie_onderdelen.append(onderdeel)
        if knie_onderdelen:
            onderdelen = ([intro] if intro else []) + knie_onderdelen
            return "\n\n".join(onderdelen).strip()
        return tekst

    return tekst[:matches[0].start()].strip() or tekst


def verwijder_lege_kliniek(tekst):
    if not tekst:
        return None
    if re.fullmatch(r"geen\s+(klinische|medische)\s+gegevens\s+verstrekt\.?", tekst.strip(), re.IGNORECASE):
        return None
    return tekst.strip() or None


def split_klin_vraag(tekst):
    if not tekst:
        return None, None
    tekst = tekst.strip()
    if not tekst:
        return None, None

    tekst = re.sub(r'^\s*(Medische gegevens|Klinische gegevens|Kliniek)\s*:\s*', '', tekst, flags=re.IGNORECASE).strip()
    tekst = re.sub(r'^\s*Gevraagd onderzoek\s*:.*$', '', tekst, flags=re.IGNORECASE | re.MULTILINE).strip()
    tekst = verwijder_lege_kliniek(tekst)
    if not tekst:
        return None, None

    parts = re.split(r'(?<=[.!?])\s+', tekst)
    parts = [p.strip() for p in parts if p.strip()]

    if not parts:
        return None, None
    if len(parts) == 1:
        if not tekst.endswith('?'):
            return tekst, None
        for sep in [",", ";", ":"]:
            pos = tekst.rfind(sep)
            if pos > 0:
                links = tekst[:pos].strip()
                rechts = tekst[pos + 1:].strip()
                if rechts.endswith("?"):
                    return links or None, rechts or None
        return None, tekst

    i = len(parts) - 1
    while i >= 0 and parts[i].endswith('?'):
        i -= 1

    if i < 0:
        return None, ' '.join(parts)
    if i == len(parts) - 1:
        return ' '.join(parts), None

    return ' '.join(parts[:i + 1]).strip() or None, ' '.join(parts[i + 1:]).strip() or None


def detect_zijde(tekst):
    if not tekst:
        return "onbekend"
    h = tekst.upper()
    if "L+R" in h or "BEIDERZIJDS" in h or "BEIDE" in h or "KNIEËN" in h or "KNIEEN" in h:
        return "L+R"
    heeft_links = bool(re.search(r'\bLINKS?\b|\bLINKERKNIE\b|LINKSZIJDIG', h))
    heeft_rechts = bool(re.search(r'\bRECHTS?\b|\bRECHTERKNIE\b|RECHTSZIJDIG|RECHTZIJDIG', h))
    if heeft_links and heeft_rechts:
        return "L+R"
    if heeft_links:
        return "links"
    if heeft_rechts:
        return "rechts"
    return "onbekend"


def zijde_uit_header(header):
    h = header.lower()
    if "rechts" in h or "rechter" in h:
        return "rechts"
    if "links" in h or "linker" in h:
        return "links"
    return None


def heeft_zijde_headers(tekst):
    if not tekst:
        return False
    zijdes = [zijde_uit_header(m.group(1)) for m in zijde_header.finditer(tekst)]
    return "rechts" in zijdes and "links" in zijdes


def extract_sides(bevindingen_text):
    if not bevindingen_text:
        return None, None

    matches = list(zijde_header.finditer(bevindingen_text))
    blokken = {}
    for i, match in enumerate(matches):
        zijde = zijde_uit_header(match.group(1))
        if zijde is None:
            continue
        einde = matches[i + 1].start() if i + 1 < len(matches) else len(bevindingen_text)
        blok = bevindingen_text[match.start():einde].strip()
        inhoud = bevindingen_text[match.end():einde].strip(" \n:.;")
        if len(inhoud) >= 15:
            blokken[zijde] = blok

    return blokken.get("rechts") or None, blokken.get("links") or None


def eerste_bericht_marker(tekst):
    matches = []

    for match in bericht_sectie.finditer(tekst):
        marker = match.group(0).lower()
        if re.search(r'rechter\s*knie|linker\s*knie|rechterknie|linkerknie', marker):
            matches.append((match.start(), match.start()))
        else:
            matches.append((match.start(), match.end()))
    for match in radio_start.finditer(tekst):
        regel_start = tekst.rfind("\n", 0, match.start()) + 1
        vorige = vorige_nietlege_regel(tekst, match.start()).lower()
        if vorige in ("vraagstelling:", "klinische gegevens:", "medische gegevens:", "gevraagd onderzoek:"):
            continue
        if tekst[regel_start:match.start()].strip() == "":
            matches.append((match.start(), match.start()))

    if not matches:
        return None
    return sorted(matches, key=lambda x: x[0])[0]


def parse_klinische_blokken(tekst):
    label_pattern = re.compile(
        r"(?im)^\s*(Klinische\s+gegevens|Vraagstelling|Gevraagd\s+onderzoek)\s*:\s*")
    label_matches = list(label_pattern.finditer(tekst))
    if label_matches:
        klinische_delen = []
        vraag_delen = []

        for i, match in enumerate(label_matches):
            label = match.group(1).lower()
            einde = label_matches[i + 1].start() if i + 1 < len(label_matches) else len(tekst)
            inhoud = tekst[match.end():einde].strip()
            if not inhoud:
                continue
            if label.startswith("klinische"):
                klinische_delen.append(inhoud)
            elif label.startswith("vraagstelling"):
                vraag_delen.append(re.sub(r'\s+', ' ', inhoud).strip())

        if not klinische_delen:
            voor_label = tekst[:label_matches[0].start()].strip()
            if voor_label:
                klinische_delen.append(voor_label)
        klinisch = "\n".join(klinische_delen).strip() or None
        vraagstelling = " ".join(vraag_delen).strip() or None
        return verwijder_lege_kliniek(klinisch), vraagstelling

    klin_match = re.search(
        r"Klinische\s+gegevens\s*:(.*?)(?=\bGevraagd\s+onderzoek\s*:|Vraagstelling\s*:|\Z)",
        tekst, re.DOTALL | re.IGNORECASE
    )
    vraag_match = re.search(
        r"Vraagstelling\s*:(.*?)(?=\n\n|\bConclusie\b|\bDX\s+[A-Za-z]|\Z)",
        tekst, re.DOTALL | re.IGNORECASE
    )

    if klin_match:
        klinisch = klin_match.group(1).strip() or None
        vraagstelling = re.sub(r'\s+', ' ', vraag_match.group(1)).strip() or None if vraag_match else None
        if vraagstelling is None and klinisch:
            klinisch, vraagstelling = split_klin_vraag(klinisch)
        return verwijder_lege_kliniek(klinisch), vraagstelling

    if vraag_match:
        klinisch_raw = re.split(
            r"\bGevraagd\s+onderzoek\s*:|Vraagstelling\s*:",
            tekst, maxsplit=1, flags=re.IGNORECASE
        )[0].strip()
        klinisch = klinisch_raw or None
        vraagstelling = re.sub(r'\s+', ' ', vraag_match.group(1)).strip() or None
        return verwijder_lege_kliniek(klinisch), vraagstelling

    return split_klin_vraag(tekst)


def filter_conclusie_knie(conclusie):
    if not conclusie:
        return conclusie
    segmenten = re.split(
        r'(?m)(?:^|\n)(?='
        r'(?:[IVX]{1,4}|[0-9]+)\.\s+'
        r'|\b(?:Hand(?:en)?|Bekken|Heup(?:en)?|Schouder|Voet|Enkel|Pols|Rug|Wervelkolom|Thorax|Clavicula|Femur|Humerus|Elleboog)\s*:'
        r')',
        conclusie
    )
    segmenten = [s.strip() for s in segmenten if s.strip()]
    if len(segmenten) <= 1:
        return conclusie
    knie_delen = [s for s in segmenten if re.search(r'knie', s, re.IGNORECASE)]
    if not knie_delen or len(knie_delen) == len(segmenten):
        return conclusie
    return '\n'.join(knie_delen).strip()


_NIET_KNIE_DEEL = (
    r'Bekken|Heup(?:en)?|Femur|Thorax|Clavicula|Schouder|Elleboog|Pols|Hand|Voet|Enkel'
    r'|Rug|Wervelkolom|Cervicaal|Lumbaal|Tibia|Fibula|Humerus|Radius|Ulna|Ribben?'
    r'|Bovenarm|Onderarm|Bovenbeen|Onderbeen'
)
niet_knie_subsectie = re.compile(
    r'(?m)^\s*(?:'
    r'(?:' + _NIET_KNIE_DEEL + r')[^\n]*'
    r'|(?:US|MRI|CT|Echo(?:grafie)?|PET)\s+(?:' + _NIET_KNIE_DEEL + r')[^\n]*'
    r'):\s*$',
    re.IGNORECASE
)

niet_knie_para_label = re.compile(
    r'^(?:X-?|DX\s+)?(?:' + _NIET_KNIE_DEEL + r')[^\n]*[:/]',
    re.IGNORECASE
)


def clean_bevindingen(tekst):
    if not tekst:
        return None
    tekst = re.sub(r'^\s*(?:Samen|Verslag|N\s*slag|[0-9]+\s*slag)\s*[.:]?\s*', '', tekst, flags=re.IGNORECASE).strip()
    tekst = re.sub(r'^\s*(?:Knie(?:\u00ebn|en|\u00c3\u00abn|n)?)\s*:\s*', '', tekst, flags=re.IGNORECASE).strip()
    tekst = re.sub(r'(?m)^\s*:\s*$', '', tekst).strip()
    m = niet_knie_subsectie.search(tekst)
    if m:
        tekst = tekst[:m.start()].strip()
    paragrafen = re.split(r'\n\n+', tekst)
    gefilterd = [p for p in paragrafen if not niet_knie_para_label.match(p.strip())]
    if len(gefilterd) < len(paragrafen):
        tekst = '\n\n'.join(gefilterd).strip()
    return tekst or None


def parse_kliniek_format(tekst, concl_pos):
    kliniek_m = re.search(r'^\s*Kliniek\s*:', tekst, re.IGNORECASE | re.MULTILINE)
    kliniek_start = kliniek_m.end()
    rest = tekst[kliniek_start:]

    bericht_m = eerste_bericht_marker(rest)
    if bericht_m:
        klin_raw = rest[:bericht_m[0]].strip()
        bev_raw  = rest[bericht_m[1]:]
    else:
        klin_raw = rest.strip()
        bev_raw  = ""

    klinisch, vraagstelling = split_klin_vraag(klin_raw)

    if bev_raw:
        concl_in_bev = re.search(r'\bConclusie\b', bev_raw, re.IGNORECASE)
        bevindingen_raw = bev_raw[:concl_in_bev.start()].strip() if concl_in_bev else bev_raw.strip()
    else:
        bevindingen_raw = None

    bevindingen_raw = clean_bevindingen(bevindingen_raw)

    if bevindingen_raw:
        eerste_regel = bevindingen_raw.split('\n')[0]
        zijde = detect_zijde(eerste_regel)
        if zijde == "onbekend":
            zijde = detect_zijde(bevindingen_raw[:300])
    else:
        zijde = "onbekend"

    return klinisch, vraagstelling, bevindingen_raw, zijde


def parse_row(tekst):
    leeg = {
        "klinische_gegevens_huisarts": None,
        "vraagstelling_huisarts":      None,
        "bevindingen":                 None,
        "bevindingen_origineel":       None,
        "conclusie":                   None,
        "zijde":                       "onbekend",
        "is_split":                    False,
        "tekst_origineel":             tekst if isinstance(tekst, str) else None,
    }

    if not isinstance(tekst, str) or not tekst.strip():
        return [leeg]

    tekst = clean_text(tekst)
    if not tekst:
        return [leeg]
    tekst_vol = tekst
    tekst = beperk_tot_knie_onderzoek(tekst)

    concl_pat = (
        r"\bConclusie\b\s*[.:]?\s*(.*?)"
        r"(?=Pati[ëe]nten\s+kunnen|Dit\s+verslag|Pati[ëe]nt\s+is\s+doorverwezen|N\.B\.|Medische\s+gegevens|\Z)"
    )
    concl_match = re.search(concl_pat, tekst, re.DOTALL | re.IGNORECASE)
    if concl_match:
        conclusie = concl_match.group(1).strip() or None
        tekst_voor_concl = tekst[:concl_match.start()].strip()
    else:
        concl_match_vol = re.search(concl_pat, tekst_vol, re.DOTALL | re.IGNORECASE)
        conclusie = concl_match_vol.group(1).strip() or None if concl_match_vol else None
        tekst_voor_concl = tekst.strip()
    conclusie = filter_conclusie_knie(conclusie)

    kliniek_m = re.search(r'^\s*Kliniek\s*:', tekst_voor_concl, re.IGNORECASE | re.MULTILINE)
    medisch_m = re.search(r'Medische\s+gegevens\s*:', tekst_voor_concl, re.IGNORECASE)
    klinische_m = re.search(r'^\s*Klinische\s+gegevens\s*:', tekst_voor_concl, re.IGNORECASE | re.MULTILINE)

    klinisch        = None
    vraagstelling   = None
    bevindingen_raw = None
    zijde           = "onbekend"

    if kliniek_m:
        klinisch, vraagstelling, bevindingen_raw, zijde = parse_kliniek_format(
            tekst_voor_concl, concl_match
        )

    elif medisch_m or klinische_m:
        start = medisch_m.end() if medisch_m else klinische_m.start()
        rest = tekst_voor_concl[start:].strip()

        bericht_m = eerste_bericht_marker(rest)
        if bericht_m:
            med_raw_stripped = rest[:bericht_m[0]].strip()
            bevindingen_raw = rest[bericht_m[1]:].strip() or None
        else:
            med_raw_stripped = rest

        klinisch, vraagstelling = parse_klinische_blokken(med_raw_stripped)
        bevindingen_raw = clean_bevindingen(bevindingen_raw)

        if bevindingen_raw:
            zijde = detect_zijde(bevindingen_raw[:300])

    else:
        bericht_m = eerste_bericht_marker(tekst_voor_concl)
        if bericht_m:
            klinisch_deel = tekst_voor_concl[:bericht_m[0]].strip()
            bevindingen_raw = tekst_voor_concl[bericht_m[1]:].strip() or None
            klinisch, vraagstelling = split_klin_vraag(klinisch_deel) if klinisch_deel else (None, None)
        else:
            klinisch, vraagstelling = split_klin_vraag(tekst_voor_concl)

        bevindingen_raw = clean_bevindingen(bevindingen_raw)
        if bevindingen_raw:
            zijde = detect_zijde(bevindingen_raw[:300])

    if vraagstelling is None and klinisch:
        klinisch, vraagstelling = split_klin_vraag(klinisch)

    base = {
        "klinische_gegevens_huisarts": klinisch,
        "vraagstelling_huisarts":      vraagstelling,
        "conclusie":                   conclusie,
        "tekst_origineel":             tekst,
    }

    is_bilateral = bool(bevindingen_raw) and heeft_zijde_headers(bevindingen_raw)

    if is_bilateral and bevindingen_raw:
        rechts_bev, links_bev = extract_sides(bevindingen_raw)
        if rechts_bev or links_bev:
            return [
                {**base, "bevindingen": rechts_bev, "bevindingen_origineel": bevindingen_raw,
                 "zijde": "rechts", "is_split": True},
                {**base, "bevindingen": links_bev,  "bevindingen_origineel": bevindingen_raw,
                 "zijde": "links",  "is_split": True},
            ]

    if zijde == "onbekend":
        zijde = detect_zijde(tekst[:300])

    return [{**base, "bevindingen": bevindingen_raw, "bevindingen_origineel": None,
             "zijde": zijde, "is_split": False}]


def parse_dataset():
    wb = openpyxl.load_workbook(input)
    ws = wb.active

    raw_rows = []
    for i, row in enumerate(ws.iter_rows(values_only=True)):
        if i == 0:
            continue
        if row[0] is None and (len(row) < 2 or row[1] is None):
            continue
        raw_rows.append({
            "case_id":   i,
            "datum":     row[0],
            "onderzoek": row[1],
            "tekst":     row[2] if len(row) > 2 else None,
        })

    output_rows = []
    for raw in raw_rows:
        parsed = parse_row(raw["tekst"])
        for p in parsed:
            
            
            zijde_from_onderzoek = detect_zijde(str(raw["onderzoek"]) if raw["onderzoek"] else "")
            
            
            if zijde_from_onderzoek != "onbekend":
                p["zijde"] = zijde_from_onderzoek
            output_rows.append({
                "case_id":   raw["case_id"],
                "datum":     raw["datum"],
                "onderzoek": raw["onderzoek"],
                **p,
            })

    df = pd.DataFrame(output_rows, columns=[
        "case_id", "datum", "onderzoek", "zijde",
        "klinische_gegevens_huisarts", "vraagstelling_huisarts",
        "bevindingen", "bevindingen_origineel", "conclusie",
        "is_split", "tekst_origineel",
    ])

    for col in ["klinische_gegevens_huisarts", "vraagstelling_huisarts",
                "bevindingen", "bevindingen_origineel", "conclusie", "tekst_origineel"]:
        df[col] = df[col].where(df[col].notna() & (df[col] != ''), other=None)

    Path(output).mkdir(parents=True, exist_ok=True)
    df.to_excel(clean_xlsx, index=False)
    df.to_csv(clean_csv, index=False, encoding="utf-8-sig")

    print(f"Klaar: {len(df)} rijen verwerkt.")
    n_split = int(df["is_split"].sum())
    print(f"  Waarvan {n_split} rijen afkomstig uit L+R splitsing ({n_split // 2} bilaterale gevallen).")
    print()
    for col in ["klinische_gegevens_huisarts", "vraagstelling_huisarts", "bevindingen", "conclusie"]:
        n_leeg = df[col].isna().sum()
        print(f"  {col}: {len(df) - n_leeg} gevuld, {n_leeg} leeg ({100 * n_leeg / len(df):.1f}%)")
    print()
    print("  Zijde verdeling:")
    for z, n in df["zijde"].value_counts().items():
        print(f"    {z}: {n}")

    verdacht = df[
        df["tekst_origineel"].notna()
        & df["bevindingen"].isna()
        & df["tekst_origineel"].str.contains(r"Verslag|DX\s+Knie|X[- ]?knie|Conclusie", case=False, na=False, regex=True)
    ]
    if len(verdacht) > 0:
        print(f"\n  LET OP: {len(verdacht)} rijen lijken radiologietekst te bevatten maar hebben lege bevindingen.")

    lange_vragen = df[df["vraagstelling_huisarts"].notna() & (df["vraagstelling_huisarts"].str.len() > 200)]
    if len(lange_vragen) > 0:
        print(f"\n  LET OP: {len(lange_vragen)} rijen met vraagstelling langer dan 200 tekens.")


parse_dataset()
