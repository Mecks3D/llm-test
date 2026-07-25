"""Estrattore e mappatore deterministico dal micro-mondo a entità, luoghi e possessi per il Graph-JEPA.

Converte le frasi UD (grafi) o gli eventi del simulatore in indici per i tensori
del modello JEPA.
"""
from __future__ import annotations

from typing import Any
from mondo import dati_mondo as dm

# Elenco fisso e deterministico di persone, oggetti e luoghi del micro-mondo
PERSONE_ID: tuple[str, ...] = tuple(p.id for p in dm.PERSONE)
OGGETTI_ID: tuple[str, ...] = tuple(o.lemma for o in dm.OGGETTI_UNICI) + tuple(
    v["lemma_unita"] for v in dm.RISORSE.values()
)
LUOGHI_ID: tuple[str, ...] = tuple(l.id for l in dm.LUOGHI)

# Entità totali (persone + oggetti)
ENTITA_ID: tuple[str, ...] = PERSONE_ID + OGGETTI_ID

# Target spaziali ed estesi (luoghi + persone come portatori + target nullo "nessuno")
TARGETS_ID: tuple[str, ...] = LUOGHI_ID + PERSONE_ID + ("nessuno",)

# Mappe da stringa a indice
ENTITA2ID: dict[str, int] = {ent: i for i, ent in enumerate(ENTITA_ID)}
ID2ENTITA: dict[int, str] = {i: ent for i, ent in enumerate(ENTITA_ID)}

LUOGO2ID: dict[str, int] = {luogo: i for i, luogo in enumerate(LUOGHI_ID)}
ID2LUOGO: dict[int, str] = {i: luogo for i, luogo in enumerate(LUOGHI_ID)}

PERSONA2ID: dict[str, int] = {p: i for i, p in enumerate(PERSONE_ID)}
ID2PERSONA: dict[int, str] = {i: p for i, p in enumerate(PERSONE_ID)}

TARGET2ID: dict[str, int] = {t: i for i, t in enumerate(TARGETS_ID)}
ID2TARGET: dict[int, str] = {i: t for i, t in enumerate(TARGETS_ID)}

N_PERSONE: int = len(PERSONE_ID)
N_OGGETTI: int = len(OGGETTI_ID)
N_ENTITA: int = len(ENTITA_ID)
N_LUOGHI: int = len(LUOGHI_ID)
N_TARGETS: int = len(TARGETS_ID)

# Azioni supportate dal trasformatore di stato
AZIONI_SUPPORTATE: tuple[str, ...] = (
    "andare",
    "prendere",
    "posare",
    "raccogliere",
    "estrarre",
    "mettere",
    "mettere_dentro",
    "tirare_fuori",
    "dare",
    "mangiare",
)
AZIONE2ID: dict[str, int] = {az: i for i, az in enumerate(AZIONI_SUPPORTATE)}


def normalizza_entita(nome: str | None) -> str | None:
    """Normalizza un nome (es. 'mela_1' -> 'mela', 'Sara' -> 'sara') e verifica se è un'entità valida."""
    if not nome:
        return None
    nome_l = nome.lower()
    if nome_l in ENTITA2ID:
        return nome_l
    base = nome_l.split("_")[0]
    if base in ENTITA2ID:
        return base
    return None


def normalizza_target(nome: str | None) -> str | None:
    """Normalizza un nome target (luogo, persona o 'nessuno') in un target valido."""
    if not nome:
        return None
    nome_l = nome.lower()
    if nome_l in TARGET2ID:
        return nome_l
    base = nome_l.split("_")[0]
    if base in TARGET2ID:
        return base
    return None


def estrai_evento_da_grafo(g: Any) -> dict[str, str | None]:
    """Estrae l'evento strutturato da un grafo UD emesso dal simulatore o lingua."""
    radice_lemma = g.nodi[0].lemma
    soggetto = None
    luogo_orig = None
    luogo_dest = None
    oggetto = None
    destinatario = None
    argomento = None

    for arco in g.archi:
        if arco.testa == 0:
            rel = arco.relazione
            lemma = g.nodi[arco.dipendente].lemma.lower()
            if rel == "nsubj":
                soggetto = lemma
            elif rel in ("obl:luogo", "obl:destinazione"):
                luogo_dest = lemma
            elif rel == "obl:origine":
                luogo_orig = lemma
            elif rel in ("obj", "obl:oggetto"):
                oggetto = lemma
            elif rel in ("iobj", "destinatario"):
                destinatario = lemma
            elif rel in ("obl:argomento", "argomento"):
                argomento = lemma

    return {
        "azione": radice_lemma,
        "soggetto": soggetto,
        "oggetto": oggetto,
        "origine": luogo_orig,
        "destinazione": luogo_dest,
        "destinatario": destinatario,
        "argomento": argomento,
    }


