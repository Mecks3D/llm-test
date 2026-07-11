"""Caricamento e validazione del lessico (lingua/lessico.tsv).

Il lessico è l'unica fonte del vocabolario: mondo, lingua e cervello lo
leggono da qui (PROGETTO.md). Questo modulo si limita a leggerlo e a
verificarne la coerenza con `mondo/` (senza mai importare `mondo/` in altri
punti di `lingua/` diversi dalla validazione e dai test).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_PERCORSO_DEFAULT = Path(__file__).with_name("lessico.tsv")

N_PRIM = 65

# Ordine normativo dei 65 primitivi NSM (FASE1_PIANO.md §7.1). Non riordinare.
ORDINE_PRIM: tuple[str, ...] = (
    "io", "tu", "qualcuno", "qualcosa", "gente", "corpo",
    "tipo", "parte",
    "questo", "stesso", "altro",
    "uno", "due", "alcuni", "tutto", "molti", "pochi",
    "buono", "cattivo",
    "grande", "piccolo",
    "pensare", "sapere", "volere", "non-volere", "sentire", "vedere", "udire",
    "dire", "parola", "vero",
    "fare", "accadere", "muoversi",
    "trovarsi", "esserci", "essere", "mio",
    "vivere", "morire",
    "quando", "adesso", "prima", "dopo", "molto-tempo", "poco-tempo", "per-un-po", "momento",
    "dove", "qui", "sopra", "sotto", "lontano", "vicino", "lato", "dentro", "toccare",
    "non", "forse", "potere", "perche", "se",
    "molto", "piu",
    "come",
)
assert len(ORDINE_PRIM) == N_PRIM

# Le 19 relazioni di parentela normative (FASE1_PIANO.md §6).
RELAZIONI_PARENTELA: tuple[str, ...] = (
    "padre_di", "madre_di", "figlio_di", "figlia_di", "marito_di", "moglie_di",
    "fratello_di", "sorella_di", "nonno_di", "nonna_di", "nipote_di",
    "suocero_di", "suocera_di", "genero_di", "nuora_di", "zio_di", "zia_di",
    "cugino_di", "cugina_di",
)

# Lemmi richiesti dagli stampi di domande/risposte, oltre a quelli già
# richiesti come azioni (FASE1_PIANO.md §6).
LEMMI_DOMANDE: tuple[str, ...] = (
    "chi", "quanti", "quante", "che-parente", "dove", "perche",
    "nessuno", "non-lo-so", "avere", "portare", "raccogliere",
    "trovarsi", "essere", "esserci", "che-cosa", "fare",
)

# Tratti specifici che lingua/stampi.py legge direttamente all'avvio (import
# time), oltre ai lemmi già richiesti sopra. Non derivano da nessuna categoria
# esistente (azioni, persone, luoghi...): vanno controllati a parte, altrimenti
# una rinomina in lessico.tsv passerebbe valida() e si romperebbe solo al primo
# import di stampi.py con un KeyError.
TRATTI_RICHIESTI_DA_STAMPI: tuple[tuple[str, str], ...] = (
    ("essere", "ausiliare_f_sing"),
    ("essere", "ausiliare_f_plur"),
    ("nessuno", "apocope_m"),
    ("nessuno", "femminile"),
)


@dataclass(frozen=True)
class VoceLessico:
    lemma: str
    categoria: str
    tratti: dict[str, str]
    definizione: str


class Lessico:
    def __init__(self, voci: list[VoceLessico]) -> None:
        self._voci = voci
        self._per_lemma: dict[str, VoceLessico] = {}
        self._per_categoria: dict[str, list[VoceLessico]] = {}
        for v in voci:
            self._per_lemma[v.lemma] = v
            self._per_categoria.setdefault(v.categoria, []).append(v)

    def __getitem__(self, lemma: str) -> VoceLessico:
        return self._per_lemma[lemma]

    def __contains__(self, lemma: str) -> bool:
        return lemma in self._per_lemma

    def get(self, lemma: str) -> VoceLessico | None:
        return self._per_lemma.get(lemma)

    def per_categoria(self, categoria: str) -> list[VoceLessico]:
        return self._per_categoria.get(categoria, [])

    def voci(self) -> list[VoceLessico]:
        return self._voci

    def valida(self) -> None:
        """Verifica la coerenza del lessico. Solleva ValueError al primo
        problema trovato, con un messaggio che indica lemma e motivo."""
        if len(self._voci) < N_PRIM:
            raise ValueError(f"il lessico ha solo {len(self._voci)} righe, servono almeno {N_PRIM} PRIM")

        for i, lemma_atteso in enumerate(ORDINE_PRIM):
            v = self._voci[i]
            if v.categoria != "PRIM":
                raise ValueError(f"riga {i}: attesa categoria PRIM per {lemma_atteso!r}, trovata {v.categoria!r}")
            if v.lemma != lemma_atteso:
                raise ValueError(f"riga {i}: atteso PRIM {lemma_atteso!r}, trovato {v.lemma!r}")

        lemmi_visti: set[str] = set()
        for v in self._voci:
            if v.lemma in lemmi_visti:
                raise ValueError(f"lemma duplicato: {v.lemma!r}")
            lemmi_visti.add(v.lemma)

        self._valida_contro_mondo()

    def _valida_contro_mondo(self) -> None:
        from mondo import dati_mondo as dm
        from mondo.azioni import AZIONI

        for nome_azione in list(AZIONI.keys()) + ["bruciare"]:
            if nome_azione not in self:
                raise ValueError(f"lemma d'azione mancante nel lessico: {nome_azione!r}")

        for persona in dm.PERSONE:
            if persona.id not in self:
                raise ValueError(f"persona mancante nel lessico: {persona.id!r}")
            genere_lessico = self[persona.id].tratti.get("genere")
            if genere_lessico != persona.genere:
                raise ValueError(
                    f"genere incoerente per {persona.id!r}: mondo={persona.genere!r} lessico={genere_lessico!r}"
                )

        for luogo in dm.LUOGHI:
            if luogo.id not in self:
                raise ValueError(f"luogo mancante nel lessico: {luogo.id!r}")

        for tipo_oggetto in dm.OGGETTI_UNICI:
            if tipo_oggetto.lemma not in self:
                raise ValueError(f"oggetto mancante nel lessico: {tipo_oggetto.lemma!r}")

        for fonte, info in dm.RISORSE.items():
            if fonte not in self:
                raise ValueError(f"fonte mancante nel lessico: {fonte!r}")
            if info["lemma_unita"] not in self:
                raise ValueError(f"risorsa mancante nel lessico: {info['lemma_unita']!r}")

        for relazione in RELAZIONI_PARENTELA:
            if relazione not in self:
                raise ValueError(f"relazione di parentela mancante nel lessico: {relazione!r}")

        for lemma in LEMMI_DOMANDE:
            if lemma not in self:
                raise ValueError(f"lemma di domande/risposte mancante nel lessico: {lemma!r}")

        for lemma, tratto in TRATTI_RICHIESTI_DA_STAMPI:
            if lemma not in self:
                raise ValueError(f"lemma mancante nel lessico: {lemma!r} (richiesto per il tratto {tratto!r} usato da lingua/stampi.py)")
            if tratto not in self[lemma].tratti:
                raise ValueError(f"tratto mancante nel lessico: lemma {lemma!r} tratto {tratto!r} (richiesto da lingua/stampi.py)")


def _analizza_tratti(campo: str) -> dict[str, str]:
    if campo == "-":
        return {}
    tratti: dict[str, str] = {}
    for coppia in campo.split(","):
        chiave, _, valore = coppia.partition("=")
        if not _ or not chiave:
            raise ValueError(f"tratto mal formato: {coppia!r}")
        tratti[chiave] = valore
    return tratti


def carica_lessico(percorso: str | Path = _PERCORSO_DEFAULT) -> Lessico:
    voci: list[VoceLessico] = []
    with open(percorso, encoding="utf-8") as f:
        for numero_riga, riga in enumerate(f, start=1):
            riga = riga.rstrip("\n")
            if not riga or riga.startswith("#"):
                continue
            campi = riga.split("\t")
            if len(campi) != 4:
                raise ValueError(f"riga {numero_riga}: attese 4 colonne separate da TAB, trovate {len(campi)}: {riga!r}")
            lemma, categoria, tratti_raw, definizione = campi
            voci.append(VoceLessico(
                lemma=lemma, categoria=categoria,
                tratti=_analizza_tratti(tratti_raw), definizione=definizione,
            ))
    return Lessico(voci)
