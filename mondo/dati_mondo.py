"""Il contenuto del micro-mondo: SOLO dati, nessuna logica.

Per aggiungere un luogo, un personaggio o un oggetto si tocca solo questo
file. Il motore (motore.py) e le azioni (azioni.py) sono generici e non
conoscono nomi specifici come "cucina" o "Sara".
"""
from __future__ import annotations

from .tipi import Luogo, Persona, TipoOggetto

# ---------------------------------------------------------------------------
# Luoghi e collegamenti (grafo di adiacenza, non una griglia geometrica)
# ---------------------------------------------------------------------------

LUOGHI: tuple[Luogo, ...] = (
    Luogo(id="cucina", lemma="cucina"),
    Luogo(id="salotto", lemma="salotto"),
    Luogo(id="giardino", lemma="giardino"),
    Luogo(id="camera", lemma="camera"),
    Luogo(id="orto", lemma="orto"),
    Luogo(id="bosco", lemma="bosco"),
)

# Collegamenti bidirezionali: elencare una sola volta ogni coppia, il
# grafo simmetrico si costruisce in modo automatico in costruisci_collegamenti().
COLLEGAMENTI_BASE: tuple[tuple[str, str], ...] = (
    ("cucina", "salotto"),
    ("cucina", "giardino"),
    ("salotto", "camera"),
    ("giardino", "orto"),
    ("giardino", "bosco"),
)


def costruisci_collegamenti() -> dict[str, frozenset[str]]:
    vicini: dict[str, set[str]] = {luogo.id: set() for luogo in LUOGHI}
    for a, b in COLLEGAMENTI_BASE:
        vicini[a].add(b)
        vicini[b].add(a)
    return {k: frozenset(v) for k, v in vicini.items()}


# ---------------------------------------------------------------------------
# Personaggi: una famiglia, tre generazioni
# ---------------------------------------------------------------------------

PERSONE: tuple[Persona, ...] = (
    Persona(id="anna", lemma="Anna", genere="f", eta="anziano", luogo_preferito="cucina"),
    Persona(id="piero", lemma="Piero", genere="m", eta="anziano", luogo_preferito="orto"),
    Persona(id="maria", lemma="Maria", genere="f", eta="adulto", luogo_preferito=None),
    Persona(id="marco", lemma="Marco", genere="m", eta="adulto", luogo_preferito="bosco"),
    Persona(id="sara", lemma="Sara", genere="f", eta="bambino", luogo_preferito="giardino"),
    Persona(id="luca", lemma="Luca", genere="m", eta="bambino", luogo_preferito=None),
)

LUOGO_INIZIALE: dict[str, str] = {
    "anna": "cucina",
    "piero": "orto",
    "maria": "cucina",
    "marco": "salotto",
    "sara": "giardino",
    "luca": "giardino",
}

# Parentela: relazioni di base (non simmetriche); le relazioni derivate
# (figlio_di, fratello_di, nonno_di, moglie_di, ...) si calcolano da queste
# in modello_parentela.py, così questo file resta puro elenco di fatti.
#   genitore_di: (genitore, figlio)
#   coniuge_di: coppia non ordinata (marito, moglie) — un solo verso qui,
#               l'inverso si deriva.
GENITORE_DI: tuple[tuple[str, str], ...] = (
    ("anna", "maria"),
    ("piero", "maria"),
    ("maria", "sara"),
    ("marco", "sara"),
    ("maria", "luca"),
    ("marco", "luca"),
)

CONIUGE_DI: tuple[tuple[str, str], ...] = (
    ("piero", "anna"),
    ("marco", "maria"),
)


# ---------------------------------------------------------------------------
# Oggetti unici (esistono fin dall'inizio, un'unica istanza, id == lemma)
# ---------------------------------------------------------------------------

OGGETTI_UNICI: tuple[TipoOggetto, ...] = (
    TipoOggetto(lemma="pane", commestibile=True),
    TipoOggetto(lemma="palla"),
    TipoOggetto(lemma="cestino", contenitore=True, apribile=False),
    TipoOggetto(lemma="scatola", contenitore=True, apribile=True),
    TipoOggetto(lemma="secchio", contenitore=True, apribile=False),
    TipoOggetto(lemma="libro"),
    TipoOggetto(lemma="camino", contenitore=True, apribile=False, fisso=True),
)

# Dove si trova ciascun oggetto unico all'inizio della storia.
LUOGO_INIZIALE_OGGETTO: dict[str, str] = {
    "pane": "cucina",
    "palla": "giardino",
    "cestino": "cucina",
    "scatola": "salotto",
    "secchio": "cucina",
    "libro": "camera",
    "camino": "salotto",
}

# Possesso statico iniziale (di chi è, indipendentemente da chi lo porta ora).
PROPRIETARIO_INIZIALE: dict[str, str] = {
    "palla": "sara",
    "libro": "piero",
}

# Lo stato "aperto/chiuso" iniziale dei contenitori apribili.
APERTO_INIZIALE: dict[str, bool] = {
    "scatola": False,
}


# ---------------------------------------------------------------------------
# Risorse finite: fonte -> (lemma dell'unità, quantità iniziale, luogo)
# ---------------------------------------------------------------------------

RISORSE: dict[str, dict] = {
    "melo": {"lemma_unita": "mela", "quantita_iniziale": 8, "luogo": "orto", "commestibile": True},
    "pozzo": {"lemma_unita": "acqua", "quantita_iniziale": 20, "luogo": "orto", "commestibile": False},
    "bosco_legna": {"lemma_unita": "legna", "quantita_iniziale": 15, "luogo": "bosco", "commestibile": False},
}

# Attrezzo richiesto per raccogliere da una fonte (None = nessuno richiesto).
ATTREZZO_RICHIESTO: dict[str, str] = {
    "pozzo": "secchio",
}

# Soglie della fisiologia (scala 0..SOGLIA_MASSIMA).
SOGLIA_MASSIMA = 10
# Soglia di stanchezza/fame oltre la quale si può solo dormire/mangiare.
# Varia per età: i bambini e gli anziani si stancano prima degli adulti —
# serve anche a distribuire nel tempo i sonni dei personaggi, invece di
# un crollo simultaneo di tutta la famiglia allo stesso tick.
SOGLIA_ESAUSTO_PER_ETA = {"bambino": 8, "adulto": 10, "anziano": 9}
RISTORO_FAME_MANGIARE = 5
RISTORO_STANCHEZZA_DORMIRE = 10  # dormire azzera la stanchezza
