import argparse
import json
import re
import math
import pandas as pd
import requests
from pathlib import Path

from config import (
    clean_csv, output as output_dir, model, ollama_url,
    sample_size as configured_sample_size, checkpoint,
    motivatie as configured_motivatie, few_shots_path,
)

input  = str(clean_csv)
output = str(output_dir)
sample_size = configured_sample_size
motivatie = configured_motivatie




_FEW_SHOT_PAD = few_shots_path
with open(_FEW_SHOT_PAD, encoding="utf-8") as _f:
    _FS = json.load(_f)

few_shot_uitkomst_bevindingen  = _FS["uitkomst_bevindingen"]
few_shot_uitkomst_conclusie    = _FS["uitkomst_conclusie"]
few_shot_uitkomst_gecombineerd = _FS["uitkomst_gecombineerd"]
few_shot_artrose               = _FS["artrose"]
few_shot_nhg                   = _FS["nhg"]


consensus_uitkomst = (
    "Aanvullende criteria uit consensusbespreking:\n\n"
    "- Als artrose of een artrose-gerelateerde term wordt beschreven (ook als onderdeel, "
    "zoals osteofyten, gewrichtsspleetversmalling, subchondrale sclerose, aanpunting), "
    "is de uitkomst afwijkend.\n"
    "- Forse aanpunting van de eminentia intercondylaris duidt op een degeneratieve "
    "verandering en is afwijkend.\n"
    "- Aanpunting van de eminentia intercondylaris (ook zonder \"forse\") duidt op "
    "degeneratie en is afwijkend.\n"
    "- Een duidelijk zichtbare osteofyt is afwijkend.\n"
    "- Patellofemoraal aanpunting is een degeneratieve verandering en is afwijkend.\n"
    "- \"Forse\" in de context van degeneratieve veranderingen betekent ernstige pathologie, "
    "maar niet de zwaarste graad; het beeld is afwijkend.\n"
    "- \"Minimale suggestie van\" een bevinding zonder verdere pathologie kan als "
    "niet-afwijkend beschouwd worden als het verslag geen andere afwijkingen vermeldt.\n\n"
)

consensus_artrose = (
    "Aanvullende criteria uit consensusbespreking:\n\n"
    "- Forse aanpunting van de eminentia intercondylaris: graad 2.\n"
    "- Aanpunting van de eminentia intercondylaris zonder \"forse\": graad 1 tot 2.\n"
    "- Duidelijk zichtbare osteofyt: graad 2.\n"
    "- Patellofemoraal aanpunting: graad 1.\n"
    "- \"Minimale suggestie van\" artrose: graad 0.\n"
    "- \"Forse\" in de context van degeneratieve veranderingen: graad 3, niet de zwaarste graad.\n\n"
)

consensus_nhg = (
    "Aanvullende criteria uit consensusbespreking:\n\n"
    "- Aanhoudende pijnklachten aan één knie, ook bij lichte omschrijvingen zoals "
    "\"beetje\", \"minimaal\" of \"enige\": label ja, tenzij de klachten duidelijk lang "
    "geleden zijn begonnen of er sprake is van omvangende terminologie.\n"
    "- Fractuurvermoeden geldt als traumatische aanleiding en geeft label ja. "
    "Het vermoeden moet wel expliciet in de tekst beschreven zijn; anders is het label nee.\n"
    "- Niet-traumatische klachten zoals artrose, overbelasting of chronische pijn "
    "zonder alarmsymptomen: label nee.\n\n"
)




artrose_pat = re.compile(
    r'\b(?:gon)?artrose\b'
    r'|\b(?:rand)?osteofyt(?:en|vorming)?\b'
    r'|\bosteofytvorming\b'
    r'|\bgewrichtsspleetversmalling\b'
    r'|\baanpunting\b'
    r'|\bsubchondrale\s+(?:sclerose|cyste[ns]?)\b'
    r'|\bdegeneratiev\w+\b'
    r'|\bartrotisch\w*\b'
    r'|\bhaakvorming\b',
    re.IGNORECASE,
)


def bouw_few_shot_uitkomst(veld):
    intro = (
        "De onderstaande voorbeelden zijn afkomstig van andere verslagen en laten zien "
        "hoe de labels worden toegepast. Gebruik ze uitsluitend als referentie en "
        "classificeer daarna uitsluitend de tekst die verderop staat.\n\n"
    )
    regels = [intro]
    if veld == "bevindingen":
        for i, v in enumerate(few_shot_uitkomst_bevindingen, start=1):
            regels.append(f"Voorbeeld {i} ({v['label']}):")
            regels.append(f"Bevindingen: \"{v['tekst']}\"")
            regels.append(f"Label: {v['label']}\n")
    elif veld == "conclusie":
        for i, v in enumerate(few_shot_uitkomst_conclusie, start=1):
            regels.append(f"Voorbeeld {i} ({v['label']}):")
            regels.append(f"Conclusie: \"{v['tekst']}\"")
            regels.append(f"Label: {v['label']}\n")
    else:
        for i, v in enumerate(few_shot_uitkomst_gecombineerd, start=1):
            regels.append(f"Voorbeeld {i} ({v['label']}):")
            regels.append(f"Bevindingen: \"{v['bevindingen']}\"")
            regels.append(f"Conclusie: \"{v['conclusie']}\"")
            regels.append(f"Label: {v['label']}\n")
    return "\n".join(regels) + "\n"


def bouw_few_shot_artrose():
    intro = (
        "De onderstaande voorbeelden zijn afkomstig van andere verslagen en laten zien "
        "hoe de graden worden toegepast. Gebruik ze uitsluitend als referentie en "
        "bepaal daarna uitsluitend de graad van het verslag dat verderop staat.\n\n"
    )
    regels = [intro]
    for i, v in enumerate(few_shot_artrose, start=1):
        regels.append(f"Voorbeeld {i} (graad {v['graad']}):")
        regels.append(f"Bevindingen: \"{v['bevindingen']}\"")
        regels.append(f"Conclusie: \"{v['conclusie']}\"")
        regels.append(f"Graad: {v['graad']}\n")
    return "\n".join(regels) + "\n"


def bouw_few_shot_nhg():
    intro = (
        "De onderstaande voorbeelden zijn afkomstig van andere aanvragen en laten zien "
        "hoe de NHG-labels worden toegepast. Gebruik ze uitsluitend als referentie en "
        "beoordeel daarna uitsluitend de aanvraag die verderop staat.\n\n"
    )
    regels = [intro]
    for i, v in enumerate(few_shot_nhg, start=1):
        regels.append(f"Voorbeeld {i} ({v['label']}):")
        regels.append(f"Tekst: \"{v['tekst']}\"")
        regels.append(f"Label: {v['label']}\n")
    return "\n".join(regels) + "\n"


def vraag_llm_logprob(tekst, veld, klinische_gegevens="", vraagstelling="", zijde="", onderzoek="", extra_block=""):
    if not tekst or str(tekst).strip() == "nan":
        return {"conf_afwijkend": None, "conf_niet_afwijkend": None, "conf_onbekend": None}

    klinische_context = ""
    if klinische_gegevens and str(klinische_gegevens).strip() not in ("", "nan"):
        klinische_context += f"Klinische gegevens huisarts:\n{klinische_gegevens}\n\n"
    if vraagstelling and str(vraagstelling).strip() not in ("", "nan"):
        klinische_context += f"Vraagstelling huisarts:\n{vraagstelling}\n\n"
    zijde_context     = f"Let op: classificeer uitsluitend de {zijde} knie.\n\n" if zijde and str(zijde).strip() not in ("", "nan", "onbekend") else ""
    onderzoek_context = f"Beantwoord uitsluitend voor het onderzoek: {onderzoek}. Negeer bevindingen van andere onderzoeken.\n\n" if onderzoek and str(onderzoek).strip() not in ("", "nan") else ""

    prompt = f"""Je moet radiologische verslagen classificeren op basis van de beschreven radiologische bevindingen.
Classificeer de radiologische uitkomst in exact één van de volgende labels: "afwijkend", "niet-afwijkend" of "onbekend".

Afwijkend:
Kies dit label wanneer in het radiologieverslag minstens één van de volgende categorieën wordt beschreven: degeneratieve veranderingen, postoperatieve veranderingen of aanwezigheid van orthopedisch materiaal, fracturen, laesies, fragmentatie, botlucentie, standsafwijkingen, ossale afwijkingen, wekedelenafwijkingen en ontwikkelingsafwijkingen.

Niet-afwijkend:
Kies dit label wanneer geen van de bij 'afwijkend' genoemde categorieën aanwezig is. Dit geldt voor verslagen waarin geen radiologische afwijkingen worden beschreven of waarin het beeld als normaal wordt weergegeven.

Onbekend:
Kies dit label wanneer de tekst onvoldoende informatie bevat om de radiologische uitkomst betrouwbaar te classificeren, bijvoorbeeld wanneer de tekst leeg is, alleen een vraagstelling bevat zonder radiologische bevindingen, of te kort of te onduidelijk is om een oordeel te vormen, wees hier zeer kritisch op.

{extra_block}{klinische_context}{zijde_context}{onderzoek_context}Baseer de classificatie uitsluitend op de radiologische bevindingen en conclusie. De klinische context is bedoeld als achtergrondinformatie over de reden van verwijzing.

Tekst ({veld}):
{tekst}

Antwoord met EXACT ÉÉN woord: afwijkend, niet-afwijkend, of onbekend."""

    try:
        response = requests.post(
            ollama_url,
            json={
                "model":        model,
                "prompt":       prompt,
                "stream":       False,
                "logprobs":     True,
                "top_logprobs": 5,
                "options":      {"temperature": 0, "num_predict": 5}
            },
            timeout=120
        )
        data     = response.json()
        logprobs = data.get("logprobs", [])

        conf_afwijkend = conf_niet_afwijkend = conf_onbekend = 0.0

        if logprobs and len(logprobs) > 0:
            top = logprobs[0].get("top_logprobs", [])
            for kandidaat in top:
                token = kandidaat["token"].lower().strip()
                kans  = math.exp(kandidaat["logprob"])
                if token.startswith("a"):
                    conf_afwijkend += kans
                elif token.startswith("n"):
                    conf_niet_afwijkend += kans
                elif token.startswith("o"):
                    conf_onbekend += kans

            totaal = conf_afwijkend + conf_niet_afwijkend + conf_onbekend
            if totaal > 0:
                conf_afwijkend      = round(conf_afwijkend / totaal, 3)
                conf_niet_afwijkend = round(conf_niet_afwijkend / totaal, 3)
                conf_onbekend       = round(conf_onbekend / totaal, 3)

        return {
            "conf_afwijkend":      conf_afwijkend,
            "conf_niet_afwijkend": conf_niet_afwijkend,
            "conf_onbekend":       conf_onbekend,
        }

    except Exception:
        return {"conf_afwijkend": None, "conf_niet_afwijkend": None, "conf_onbekend": None}


def label_uit_logprob(conf):
    a = conf["conf_afwijkend"]
    n = conf["conf_niet_afwijkend"]
    o = conf["conf_onbekend"]

    if a is None and n is None and o is None:
        return "onbekend"

    confs = {
        "afwijkend":      a or 0,
        "niet-afwijkend": n or 0,
        "onbekend":       o or 0,
    }
    if all(v == 0 for v in confs.values()):
        return "onbekend"

    return max(confs, key=confs.get)


def vraag_motivatie(tekst, veld, klinische_gegevens="", vraagstelling="", zijde="", onderzoek=""):
    if not tekst or str(tekst).strip() == "nan":
        return None

    klinische_context = ""
    if klinische_gegevens and str(klinische_gegevens).strip() not in ("", "nan"):
        klinische_context += f"Klinische gegevens huisarts:\n{klinische_gegevens}\n\n"
    if vraagstelling and str(vraagstelling).strip() not in ("", "nan"):
        klinische_context += f"Vraagstelling huisarts:\n{vraagstelling}\n\n"
    zijde_context     = f"Let op: classificeer uitsluitend de {zijde} knie.\n\n" if zijde and str(zijde).strip() not in ("", "nan", "onbekend") else ""
    onderzoek_context = f"Beantwoord uitsluitend voor het onderzoek: {onderzoek}. Negeer bevindingen van andere onderzoeken.\n\n" if onderzoek and str(onderzoek).strip() not in ("", "nan") else ""

    prompt = f"""Je bent een helper die radiologische verslagen classificeert op basis van de beschreven radiologische bevindingen.
Classificeer de radiologische uitkomst en geef een korte motivatie.

Afwijkend:
Kies dit label wanneer in het radiologieverslag minstens één van de volgende abnormaliteitscategorieën worden beschreven: degeneratieve veranderingen, postoperatieve veranderingen of aanwezigheid van niet-arthroplastisch orthopedisch materiaal, fracturen, laesies, fragmentatie, botlucentie, standsafwijkingen (malalignment), ossale afwijkingen, wekedelenafwijkingen, ontwikkelingsafwijkingen of trauma.

Niet-afwijkend:
Kies dit label wanneer geen van de hierboven genoemde abnormaliteitscategorieën worden beschreven en er geen arthroplastiek aanwezig is.

Onbekend:
Kies dit label wanneer de tekst onvoldoende informatie bevat om de radiologische uitkomst betrouwbaar te classificeren.

{klinische_context}{zijde_context}{onderzoek_context}Tekst ({veld}):
{tekst}

Geef je antwoord in het volgende formaat:
Label: <afwijkend | niet-afwijkend | onbekend>
Motivatie: <één zin met de reden voor dit label>"""

    try:
        response = requests.post(
            ollama_url,
            json={
                "model":   model,
                "prompt":  prompt,
                "stream":  False,
                "options": {"temperature": 0, "num_predict": 80}
            },
            timeout=120
        )
        antwoord = response.json().get("response", "").strip()
        for regel in antwoord.splitlines():
            if regel.lower().startswith("motivatie:"):
                return regel.split(":", 1)[1].strip()
        return antwoord.splitlines()[0] if antwoord else None

    except Exception:
        return None


def check_artrose_vraagstelling(vraagstelling):
    tekst = str(vraagstelling).lower() if vraagstelling and str(vraagstelling) != "nan" else ""
    return "ja" if artrose_pat.search(tekst) else "nee"


def check_artrose_radiologie(bevindingen, conclusie):
    onderdelen = [
        str(bevindingen) if bevindingen and str(bevindingen) != "nan" else "",
        str(conclusie)   if conclusie   and str(conclusie)   != "nan" else "",
    ]
    tekst = " ".join(onderdelen).lower()
    return "ja" if artrose_pat.search(tekst) else "nee"


kl_schaal = """Graad 0: Geen artrose. Normaal gewricht, geen afwijkingen zichtbaar.

Graad 1: Minimale artrose. Zeer geringe osteofytvorming, geen of nauwelijks gewrichtsspleetversmalling.

Graad 2: Milde artrose. Duidelijkere osteofytvorming, lichte versmalling van de gewrichtsspleet.

Graad 3: Matige artrose. Duidelijke gewrichtsspleetversmalling, subchondrale sclerose en/of cysten.

Graad 4: Ernstige artrose. Sterk versmalde of opgeheven gewrichtsspleet, prominente osteofyten.

Onbekend: Onvoldoende informatie om de ernst te bepalen."""


def parse_graad(waarde):
    if not waarde:
        return "onbekend"
    w = str(waarde).lower()
    if "onbekend" in w:
        return "onbekend"
    m = re.search(r'[0-4]', w)
    return int(m.group()) if m else "onbekend"


def vraag_artrose_graad(tekst_conclusie, tekst_bevindingen, extra_block=""):
    onderdelen = []
    if tekst_bevindingen and str(tekst_bevindingen).strip() not in ("", "nan"):
        onderdelen.append(f"Bevindingen:\n{tekst_bevindingen}")
    if tekst_conclusie and str(tekst_conclusie).strip() not in ("", "nan"):
        onderdelen.append(f"Conclusie:\n{tekst_conclusie}")

    if not onderdelen:
        return None, None

    tekst_combined = "\n\n".join(onderdelen)

    prompt = f"""Je bent een radiologisch classificatieassistent. Bepaal op basis van de onderstaande bevindingen en conclusie de ernst van artrose volgens de Kellgren-Lawrence schaal en geef een korte motivatie.

{kl_schaal}

{extra_block}Radiologisch verslag:
{tekst_combined}

Geef je antwoord in het volgende formaat:
Graad: <0 | 1 | 2 | 3 | 4 | onbekend>
Motivatie: <één zin met de radiologische kenmerken die tot dit oordeel leiden>"""

    try:
        response = requests.post(
            ollama_url,
            json={
                "model":   model,
                "prompt":  prompt,
                "stream":  False,
                "options": {"temperature": 0, "num_predict": 80}
            },
            timeout=120
        )
        antwoord = response.json().get("response", "").strip()

        graad     = None
        motivatie = None

        for regel in antwoord.splitlines():
            regel_lower = regel.lower()
            if regel_lower.startswith("graad:"):
                graad = parse_graad(regel.split(":", 1)[1].strip())
            elif regel_lower.startswith("motivatie:"):
                motivatie = regel.split(":", 1)[1].strip()

        if graad is None:
            graad = parse_graad(antwoord.splitlines()[0] if antwoord else "onbekend")

        return graad, motivatie

    except Exception:
        return None, None


def vraag_trauma_label(klinische_gegevens, vraagstelling):
    kg = str(klinische_gegevens).strip() if klinische_gegevens else ""
    vr = str(vraagstelling).strip() if vraagstelling else ""

    if kg in ("", "nan") and vr in ("", "nan"):
        return {"label_trauma": "onbekend", "motivatie_trauma": "Geen klinische informatie beschikbaar."}

    tekst_combined = ""
    if kg not in ("", "nan"):
        tekst_combined += f"Klinische gegevens huisarts:\n{kg}\n\n"
    if vr not in ("", "nan"):
        tekst_combined += f"Vraagstelling huisarts:\n{vr}"

    prompt = f"""Bepaal op basis van de onderstaande klinische informatie van de huisarts of er sprake is van een traumatische aanleiding voor het knieonderzoek.

Gebruik exact één van de volgende labels:

trauma: Er is sprake van een recent trauma (minder dan 2 maanden geleden). Denk aan vallen, stoten, verdraaiing, sportletsel of andere acute verwonding.

oud_trauma: Er is sprake van een trauma dat 2 maanden of langer geleden heeft plaatsgevonden, of er wordt verwezen naar een eerder doorgemaakte blessure of letsel.

niet_trauma: Er is geen sprake van een trauma. De klachten zijn geleidelijk ontstaan, chronisch van aard, of hebben een niet-traumatische oorzaak zoals artrose, ontsteking of overbelasting.

onbekend: De klinische informatie is onvoldoende, te vaag of ontbreekt om te bepalen of er sprake is van trauma.

Klinische informatie:
{tekst_combined}

Geef je antwoord in het volgende formaat:
Label: <trauma | oud_trauma | niet_trauma | onbekend>
Motivatie: <één zin met de reden voor dit label>"""

    try:
        response = requests.post(
            ollama_url,
            json={
                "model":   model,
                "prompt":  prompt,
                "stream":  False,
                "options": {"temperature": 0, "num_predict": 60}
            },
            timeout=120
        )
        antwoord = response.json().get("response", "").strip()

        label     = "onbekend"
        motivatie = antwoord

        for regel in antwoord.splitlines():
            regel_lower = regel.lower()
            if regel_lower.startswith("label:"):
                waarde = regel.split(":", 1)[1].strip().lower()
                if waarde in ("trauma", "oud_trauma", "niet_trauma", "onbekend"):
                    label = waarde
            elif regel_lower.startswith("motivatie:"):
                motivatie = regel.split(":", 1)[1].strip()

        return {"label_trauma": label, "motivatie_trauma": motivatie}

    except Exception:
        return {"label_trauma": "onbekend", "motivatie_trauma": None}


def vraag_fractuur_label(klinische_gegevens, vraagstelling):
    kg = str(klinische_gegevens).strip() if klinische_gegevens else ""
    vr = str(vraagstelling).strip() if vraagstelling else ""

    if kg in ("", "nan") and vr in ("", "nan"):
        return {"label_fractuur": "onbekend", "motivatie_fractuur": "Geen klinische informatie beschikbaar."}

    tekst_combined = ""
    if kg not in ("", "nan"):
        tekst_combined += f"Klinische gegevens huisarts:\n{kg}\n\n"
    if vr not in ("", "nan"):
        tekst_combined += f"Vraagstelling huisarts:\n{vr}"

    prompt = f"""Bepaal op basis van de onderstaande klinische informatie of de huisarts een fractuur vermoedt of uitsluit als aanleiding voor het knieonderzoek.

Gebruik exact één van de volgende labels:

ja: Er is een expliciet of impliciet klinisch vermoeden van een fractuur. Denk aan termen als 'fractuur?', 'botbreuk', 'uit te sluiten fractuur', of beschrijvingen die sterk wijzen op een fractuur.

nee: Er is geen vermoeden van een fractuur. De klachten zijn duidelijk van andere aard, zoals artrose, bandletsel, zwelling of overbelasting.

onbekend: De informatie is te vaag of ontbreekt om te bepalen of een fractuur wordt vermoed.

Klinische informatie:
{tekst_combined}

Geef je antwoord in het volgende formaat:
Label: <ja | nee | onbekend>
Motivatie: <één zin met de reden voor dit label>"""

    try:
        response = requests.post(
            ollama_url,
            json={
                "model":   model,
                "prompt":  prompt,
                "stream":  False,
                "options": {"temperature": 0, "num_predict": 60}
            },
            timeout=120
        )
        antwoord = response.json().get("response", "").strip()

        label     = "onbekend"
        motivatie = antwoord

        for regel in antwoord.splitlines():
            regel_lower = regel.lower()
            if regel_lower.startswith("label:"):
                waarde = regel.split(":", 1)[1].strip().lower()
                if waarde in ("ja", "nee", "onbekend"):
                    label = waarde
            elif regel_lower.startswith("motivatie:"):
                motivatie = regel.split(":", 1)[1].strip()

        return {"label_fractuur": label, "motivatie_fractuur": motivatie}

    except Exception:
        return {"label_fractuur": "onbekend", "motivatie_fractuur": None}


def vraag_protesis_label(bevindingen, conclusie):
    onderdelen = []
    if bevindingen and str(bevindingen).strip() not in ("", "nan"):
        onderdelen.append(f"Bevindingen:\n{bevindingen}")
    if conclusie and str(conclusie).strip() not in ("", "nan"):
        onderdelen.append(f"Conclusie:\n{conclusie}")

    if not onderdelen:
        return {"label_protesis": "onbekend", "motivatie_protesis": "Geen radiologische tekst beschikbaar."}

    tekst_combined = "\n\n".join(onderdelen)

    prompt = f"""Bepaal op basis van het onderstaande radiologieverslag of er sprake is van een knieprothese (arthroplastiek).

Gebruik exact één van de volgende labels:

ja: Het verslag beschrijft expliciet de aanwezigheid van een knieprothese, totale kniearthroplastiek, unicompartimentele prothese of ander arthroplastisch implantaat.

nee: Het verslag beschrijft geen prothese. Er kan wel ander orthopedisch materiaal aanwezig zijn (zoals een schroef of plaat), maar geen arthroplastiek.

onbekend: De tekst is onvoldoende om te bepalen of een prothese aanwezig is.

Radiologisch verslag:
{tekst_combined}

Geef je antwoord in het volgende formaat:
Label: <ja | nee | onbekend>
Motivatie: <één zin met de reden voor dit label>"""

    try:
        response = requests.post(
            ollama_url,
            json={
                "model":   model,
                "prompt":  prompt,
                "stream":  False,
                "options": {"temperature": 0, "num_predict": 60}
            },
            timeout=120
        )
        antwoord = response.json().get("response", "").strip()

        label     = "onbekend"
        motivatie = antwoord

        for regel in antwoord.splitlines():
            regel_lower = regel.lower()
            if regel_lower.startswith("label:"):
                waarde = regel.split(":", 1)[1].strip().lower()
                if waarde in ("ja", "nee", "onbekend"):
                    label = waarde
            elif regel_lower.startswith("motivatie:"):
                motivatie = regel.split(":", 1)[1].strip()

        return {"label_protesis": label, "motivatie_protesis": motivatie}

    except Exception:
        return {"label_protesis": "onbekend", "motivatie_protesis": None}


def beoordeel_nhg_compliance(klinische_gegevens, vraagstelling, label_trauma="", extra_block=""):
    
    
    
    kg = str(klinische_gegevens).strip() if klinische_gegevens else ""
    vr = str(vraagstelling).strip() if vraagstelling else ""

    if kg in ("", "nan") and vr in ("", "nan"):
        return {"label_nhg": "onbekend", "motivatie_nhg": "Geen klinische informatie beschikbaar."}

    tekst_combined = ""
    if kg not in ("", "nan"):
        tekst_combined += f"Klinische gegevens:\n{kg}\n\n"
    if vr not in ("", "nan"):
        tekst_combined += f"Vraagstelling:\n{vr}"

    prompt = f"""Je moet NHG-richtlijnen labelen voor knieklachten (M66 traumatisch, M107 niet-traumatisch).

Beoordeel of de röntgenaanvraag (X-knie) gerechtvaardigd is op basis van de onderstaande klinische informatie.

NHG-CRITERIA:

traumatische knieklachten (M66):
- Röntgen ja gerechtvaardigd: uitsluitend bij klinisch vermoeden van een fractuur.
- Röntgen nee: bij contusie, distorsie, band- of meniscusletsel zonder fractuurverdenking.
- MRI wordt in de eerste lijn niet aanbevolen.

niet traumatische knieklachten (M107):
- Röntgen ja gerechtvaardigd: bij klinisch vermoeden van fractuur, epifysiolyse, tumor, osteomyelitis of osteochondritis dissecans, of persisterende unilaterale kniepijn zonder andere verklaring.
- Röntgen nee: bij knieartrose (gonartrose), patellofemoraal pijnsyndroom, bursitis prepatellaris, iliotibiale bandsyndroom, jumper's knee, ziekte van Osgood-Schlatter.
{extra_block}Klinische informatie:
{tekst_combined}

Gebruik exact één van de volgende labels:
ja: De röntgenaanvraag is gerechtvaardigd volgens de NHG-richtlijnen.
nee: De röntgenaanvraag is niet gerechtvaardigd volgens de NHG-richtlijnen.
onbekend: Er is onvoldoende informatie om een oordeel te geven.

Geef je antwoord in het volgende formaat:
Label: <ja | nee | onbekend>
Motivatie: <één zin met de reden voor dit oordeel>"""

    try:
        response = requests.post(
            ollama_url,
            json={
                "model":   model,
                "prompt":  prompt,
                "stream":  False,
                "options": {"temperature": 0, "num_predict": 80}
            },
            timeout=120
        )
        antwoord = response.json().get("response", "").strip()

        label     = "onbekend"
        motivatie = antwoord

        for regel in antwoord.splitlines():
            regel_lower = regel.lower()
            if regel_lower.startswith("label:"):
                waarde = regel.split(":", 1)[1].strip().lower()
                if waarde in ("ja", "nee", "onbekend"):
                    label = waarde
            elif regel_lower.startswith("motivatie:"):
                motivatie = regel.split(":", 1)[1].strip()

        return {"label_nhg": label, "motivatie_nhg": motivatie}

    except Exception:
        return {"label_nhg": "onbekend", "motivatie_nhg": None}


def run_variant_uitkomst(df, suffix, extra_block, total):
    for veld in ["bevindingen", "conclusie", "gecombineerd"]:
        df[f"label_{veld}_{suffix}"]               = None
        df[f"conf_afwijkend_{veld}_{suffix}"]      = None
        df[f"conf_niet_afwijkend_{veld}_{suffix}"] = None
        df[f"conf_onbekend_{veld}_{suffix}"]       = None

    for veld in ["bevindingen", "conclusie"]:
        blok = extra_block(veld) if callable(extra_block) else extra_block
        for teller, (idx, row) in enumerate(df.iterrows(), start=1):
            tekst = row.get(veld)
            if not tekst or str(tekst).strip() in ("", "nan"):
                continue
            conf = vraag_llm_logprob(
                tekst              = str(tekst),
                veld               = veld,
                klinische_gegevens = row.get("klinische_gegevens_huisarts", ""),
                vraagstelling      = row.get("vraagstelling_huisarts", ""),
                zijde              = row.get("zijde", ""),
                onderzoek          = row.get("onderzoek", ""),
                extra_block        = blok,
            )
            df.at[idx, f"label_{veld}_{suffix}"]               = label_uit_logprob(conf)
            df.at[idx, f"conf_afwijkend_{veld}_{suffix}"]      = conf["conf_afwijkend"]
            df.at[idx, f"conf_niet_afwijkend_{veld}_{suffix}"] = conf["conf_niet_afwijkend"]
            df.at[idx, f"conf_onbekend_{veld}_{suffix}"]       = conf["conf_onbekend"]
            print(f"{suffix} | {teller}/{total} {veld} klaar", flush=True)
            if teller % checkpoint == 0:
                df.to_csv(f"{output}/llm_labels_checkpoint.csv", index=False, encoding="utf-8-sig")

    blok = extra_block("gecombineerd") if callable(extra_block) else extra_block
    for teller, (idx, row) in enumerate(df.iterrows(), start=1):
        bev   = str(row.get("bevindingen", "") or "").strip()
        concl = str(row.get("conclusie",   "") or "").strip()
        onderdelen = []
        if bev   and bev   != "nan":
            onderdelen.append(f"Bevindingen:\n{bev}")
        if concl and concl != "nan":
            onderdelen.append(f"Conclusie:\n{concl}")
        tekst_gecombineerd = "\n\n".join(onderdelen) if onderdelen else None

        conf = vraag_llm_logprob(
            tekst              = tekst_gecombineerd,
            veld               = "bevindingen en conclusie",
            klinische_gegevens = row.get("klinische_gegevens_huisarts", ""),
            vraagstelling      = row.get("vraagstelling_huisarts", ""),
            zijde              = row.get("zijde", ""),
            onderzoek          = row.get("onderzoek", ""),
            extra_block        = blok,
        )
        df.at[idx, f"label_gecombineerd_{suffix}"]               = label_uit_logprob(conf)
        df.at[idx, f"conf_afwijkend_gecombineerd_{suffix}"]      = conf["conf_afwijkend"]
        df.at[idx, f"conf_niet_afwijkend_gecombineerd_{suffix}"] = conf["conf_niet_afwijkend"]
        df.at[idx, f"conf_onbekend_gecombineerd_{suffix}"]       = conf["conf_onbekend"]
        print(f"{suffix} | {teller}/{total} gecombineerd klaar", flush=True)
        if teller % checkpoint == 0:
            df.to_csv(f"{output}/llm_labels_checkpoint.csv", index=False, encoding="utf-8-sig")


def run_variant_artrose(df, suffix, extra_block):
    df[f"artrose_graad_{suffix}"] = None
    total = len(df)
    for teller, (idx, row) in enumerate(df.iterrows(), start=1):
        graad, _ = vraag_artrose_graad(
            tekst_conclusie   = row.get("conclusie"),
            tekst_bevindingen = row.get("bevindingen"),
            extra_block       = extra_block,
        )
        df.at[idx, f"artrose_graad_{suffix}"] = graad
        print(f"{suffix} | {teller}/{total} artrose_graad klaar", flush=True)
        if teller % checkpoint == 0:
            df.to_csv(f"{output}/llm_labels_checkpoint.csv", index=False, encoding="utf-8-sig")


def run_variant_nhg(df, suffix, extra_block):
    df[f"label_nhg_{suffix}"] = None
    total = len(df)
    for teller, (idx, row) in enumerate(df.iterrows(), start=1):
        resultaat = beoordeel_nhg_compliance(
            klinische_gegevens = row.get("klinische_gegevens_huisarts", ""),
            vraagstelling      = row.get("vraagstelling_huisarts", ""),
            label_trauma       = "",
            extra_block        = extra_block,
        )
        df.at[idx, f"label_nhg_{suffix}"] = resultaat["label_nhg"]
        print(f"{suffix} | {teller}/{total} nhg klaar", flush=True)
        if teller % checkpoint == 0:
            df.to_csv(f"{output}/llm_labels_checkpoint.csv", index=False, encoding="utf-8-sig")


def label_dataset(input_path=None, output_path=None, sample_size_override=None, motivatie_override=None):
    global input, output, sample_size, motivatie

    if input_path is not None:
        input = str(input_path)
    if output_path is not None:
        output = str(output_path)
    if sample_size_override is not None:
        sample_size = int(sample_size_override)
    if motivatie_override is not None:
        motivatie = bool(motivatie_override)

    df = pd.read_csv(input, encoding="utf-8-sig")
    df = df[df["onderzoek"].str.contains("Knie", case=False, na=False)].copy()
    if sample_size > 0:
        df = df.head(sample_size).copy()
    else:
        df = df.copy()

    df = df.reset_index(drop=True)
    total = len(df)

    df["label_protesis"]     = None
    df["motivatie_protesis"] = None

    for teller, (idx, row) in enumerate(df.iterrows(), start=1):
        resultaat = vraag_protesis_label(
            bevindingen = row.get("bevindingen"),
            conclusie   = row.get("conclusie")
        )
        df.at[idx, "label_protesis"]     = resultaat["label_protesis"]
        df.at[idx, "motivatie_protesis"] = resultaat["motivatie_protesis"]
        print(f"{teller}/{total} protesis klaar", flush=True)
        if teller % checkpoint == 0:
            df.to_csv(f"{output}/llm_labels_checkpoint.csv", index=False, encoding="utf-8-sig")

    df["label_fractuur"]     = None
    df["motivatie_fractuur"] = None

    for teller, (idx, row) in enumerate(df.iterrows(), start=1):
        resultaat = vraag_fractuur_label(
            klinische_gegevens = row.get("klinische_gegevens_huisarts", ""),
            vraagstelling      = row.get("vraagstelling_huisarts", "")
        )
        df.at[idx, "label_fractuur"]     = resultaat["label_fractuur"]
        df.at[idx, "motivatie_fractuur"] = resultaat["motivatie_fractuur"]
        print(f"{teller}/{total} fractuur klaar", flush=True)
        if teller % checkpoint == 0:
            df.to_csv(f"{output}/llm_labels_checkpoint.csv", index=False, encoding="utf-8-sig")

    for veld in ["bevindingen", "conclusie"]:
        df[f"label_{veld}"]               = None
        df[f"conf_afwijkend_{veld}"]      = None
        df[f"conf_niet_afwijkend_{veld}"] = None
        df[f"conf_onbekend_{veld}"]       = None
        if motivatie:
            df[f"motivatie_{veld}"] = None

        for teller, (idx, row) in enumerate(df.iterrows(), start=1):
            conf = vraag_llm_logprob(
                tekst              = row[veld],
                veld               = veld,
                klinische_gegevens = row.get("klinische_gegevens_huisarts", ""),
                vraagstelling      = row.get("vraagstelling_huisarts", ""),
                zijde              = row.get("zijde", ""),
                onderzoek          = row.get("onderzoek", "")
            )
            df.at[idx, f"label_{veld}"]               = label_uit_logprob(conf)
            df.at[idx, f"conf_afwijkend_{veld}"]      = conf["conf_afwijkend"]
            df.at[idx, f"conf_niet_afwijkend_{veld}"] = conf["conf_niet_afwijkend"]
            df.at[idx, f"conf_onbekend_{veld}"]       = conf["conf_onbekend"]

            if motivatie:
                df.at[idx, f"motivatie_{veld}"] = vraag_motivatie(
                    tekst              = row[veld],
                    veld               = veld,
                    klinische_gegevens = row.get("klinische_gegevens_huisarts", ""),
                    vraagstelling      = row.get("vraagstelling_huisarts", ""),
                    zijde              = row.get("zijde", ""),
                    onderzoek          = row.get("onderzoek", "")
                )

            print(f"{teller}/{total} {veld} klaar", flush=True)
            if teller % checkpoint == 0:
                df.to_csv(f"{output}/llm_labels_checkpoint.csv", index=False, encoding="utf-8-sig")

    df["label_gecombineerd"]               = None
    df["conf_afwijkend_gecombineerd"]      = None
    df["conf_niet_afwijkend_gecombineerd"] = None
    df["conf_onbekend_gecombineerd"]       = None
    if motivatie:
        df["motivatie_gecombineerd"] = None

    for teller, (idx, row) in enumerate(df.iterrows(), start=1):
        bev   = str(row.get("bevindingen", "") or "").strip()
        concl = str(row.get("conclusie", "") or "").strip()
        onderdelen = []
        if bev and bev != "nan":
            onderdelen.append(f"Bevindingen:\n{bev}")
        if concl and concl != "nan":
            onderdelen.append(f"Conclusie:\n{concl}")

        tekst_gecombineerd = "\n\n".join(onderdelen) if onderdelen else None

        conf = vraag_llm_logprob(
            tekst              = tekst_gecombineerd,
            veld               = "bevindingen en conclusie",
            klinische_gegevens = row.get("klinische_gegevens_huisarts", ""),
            vraagstelling      = row.get("vraagstelling_huisarts", ""),
            zijde              = row.get("zijde", ""),
            onderzoek          = row.get("onderzoek", "")
        )
        df.at[idx, "label_gecombineerd"]               = label_uit_logprob(conf)
        df.at[idx, "conf_afwijkend_gecombineerd"]      = conf["conf_afwijkend"]
        df.at[idx, "conf_niet_afwijkend_gecombineerd"] = conf["conf_niet_afwijkend"]
        df.at[idx, "conf_onbekend_gecombineerd"]       = conf["conf_onbekend"]

        if motivatie:
            df.at[idx, "motivatie_gecombineerd"] = vraag_motivatie(
                tekst              = tekst_gecombineerd,
                veld               = "bevindingen en conclusie",
                klinische_gegevens = row.get("klinische_gegevens_huisarts", ""),
                vraagstelling      = row.get("vraagstelling_huisarts", ""),
                zijde              = row.get("zijde", ""),
                onderzoek          = row.get("onderzoek", "")
            )

        print(f"{teller}/{total} gecombineerd klaar", flush=True)
        if teller % checkpoint == 0:
            df.to_csv(f"{output}/llm_labels_checkpoint.csv", index=False, encoding="utf-8-sig")

    run_variant_uitkomst(df, "fewshot",   bouw_few_shot_uitkomst, total)
    run_variant_uitkomst(df, "consensus", consensus_uitkomst,     total)

    df["artrose_vraagstelling"] = df["vraagstelling_huisarts"].apply(check_artrose_vraagstelling)
    df["artrose_radiologie"] = df.apply(
        lambda r: check_artrose_radiologie(r.get("bevindingen"), r.get("conclusie")),
        axis=1
    )

    df["artrose_graad"]           = None
    df["motivatie_artrose_graad"] = None

    for teller, (idx, row) in enumerate(df.iterrows(), start=1):
        graad, motivatie_artrose = vraag_artrose_graad(
            tekst_conclusie   = row.get("conclusie"),
            tekst_bevindingen = row.get("bevindingen")
        )
        df.at[idx, "artrose_graad"]           = graad
        df.at[idx, "motivatie_artrose_graad"] = motivatie_artrose
        print(f"{teller}/{total} artrose_graad klaar", flush=True)
        if teller % checkpoint == 0:
            df.to_csv(f"{output}/llm_labels_checkpoint.csv", index=False, encoding="utf-8-sig")

    run_variant_artrose(df, "fewshot",   bouw_few_shot_artrose())
    run_variant_artrose(df, "consensus", consensus_artrose)

    df["label_trauma"]     = None
    df["motivatie_trauma"] = None

    for teller, (idx, row) in enumerate(df.iterrows(), start=1):
        resultaat = vraag_trauma_label(
            klinische_gegevens = row.get("klinische_gegevens_huisarts", ""),
            vraagstelling      = row.get("vraagstelling_huisarts", "")
        )
        df.at[idx, "label_trauma"]     = resultaat["label_trauma"]
        df.at[idx, "motivatie_trauma"] = resultaat["motivatie_trauma"]
        print(f"{teller}/{total} trauma klaar", flush=True)
        if teller % checkpoint == 0:
            df.to_csv(f"{output}/llm_labels_checkpoint.csv", index=False, encoding="utf-8-sig")

    df["label_nhg"]     = None
    df["motivatie_nhg"] = None

    for teller, (idx, row) in enumerate(df.iterrows(), start=1):
        resultaat = beoordeel_nhg_compliance(
            klinische_gegevens = row.get("klinische_gegevens_huisarts", ""),
            vraagstelling      = row.get("vraagstelling_huisarts", ""),
            label_trauma       = "",
        )
        df.at[idx, "label_nhg"]     = resultaat["label_nhg"]
        df.at[idx, "motivatie_nhg"] = resultaat["motivatie_nhg"]
        print(f"{teller}/{total} nhg klaar", flush=True)
        if teller % checkpoint == 0:
            df.to_csv(f"{output}/llm_labels_checkpoint.csv", index=False, encoding="utf-8-sig")

    run_variant_nhg(df, "fewshot",   bouw_few_shot_nhg())
    run_variant_nhg(df, "consensus", consensus_nhg)

    basis_kolommen    = ["case_id", "datum", "onderzoek", "zijde"]
    klinisch_kolommen = ["klinische_gegevens_huisarts", "vraagstelling_huisarts"]
    radio_kolommen    = ["bevindingen", "bevindingen_origineel", "conclusie"]

    label_kolommen = []
    for veld in ["bevindingen", "conclusie", "gecombineerd"]:
        label_kolommen.append(f"label_{veld}")
        label_kolommen.extend([f"conf_afwijkend_{veld}", f"conf_niet_afwijkend_{veld}", f"conf_onbekend_{veld}"])
        if motivatie:
            label_kolommen.append(f"motivatie_{veld}")

    variant_uitkomst_kolommen = []
    for suffix in ["fewshot", "consensus"]:
        for veld in ["bevindingen", "conclusie", "gecombineerd"]:
            variant_uitkomst_kolommen.append(f"label_{veld}_{suffix}")
            variant_uitkomst_kolommen.extend([
                f"conf_afwijkend_{veld}_{suffix}",
                f"conf_niet_afwijkend_{veld}_{suffix}",
                f"conf_onbekend_{veld}_{suffix}",
            ])

    artrose_kolommen  = [
        "artrose_vraagstelling", "artrose_radiologie",
        "artrose_graad", "motivatie_artrose_graad",
        "artrose_graad_fewshot", "artrose_graad_consensus",
    ]
    trauma_kolommen   = ["label_trauma", "motivatie_trauma"]
    nhg_kolommen      = ["label_nhg", "motivatie_nhg", "label_nhg_fewshot", "label_nhg_consensus"]
    protesis_kolommen = ["label_protesis",  "motivatie_protesis"]
    fractuur_kolommen = ["label_fractuur",  "motivatie_fractuur"]
    eind_kolommen     = ["is_split", "tekst_origineel"]

    gewenste_volgorde = (
        basis_kolommen + klinisch_kolommen + radio_kolommen + label_kolommen +
        variant_uitkomst_kolommen + artrose_kolommen + trauma_kolommen +
        nhg_kolommen + protesis_kolommen + fractuur_kolommen + eind_kolommen
    )

    aanwezige_kolommen = [k for k in gewenste_volgorde if k in df.columns]
    overige_kolommen   = [k for k in df.columns if k not in aanwezige_kolommen]
    df = df[aanwezige_kolommen + overige_kolommen]

    Path(output).mkdir(parents=True, exist_ok=True)
    df.to_excel(f"{output}/llm_labels.xlsx", index=False)
    df.to_csv(f"{output}/llm_labels.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=input)
    parser.add_argument("--output", default=output)
    parser.add_argument("--sample-size", type=int, default=sample_size)
    parser.add_argument("--no-motivation", "--geen-motivatie", action="store_true")
    args = parser.parse_args()
    label_dataset(args.input, args.output, args.sample_size, not args.no_motivation)
