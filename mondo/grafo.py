"""Rappresentazione a grafo (stile Universal Dependencies) di eventi e domande.

Regola non negoziabile del progetto: la valutazione è sempre grafo vs grafo,
mai stringa vs stringa. Questo modulo produce i grafi-verità; il testo
(Fase 1) è un front-end che non deve mai essere l'unico confronto possibile.

`Grafo` è immutabile e confrontabile con `==`: due grafi sono uguali se hanno
esattamente gli stessi nodi e gli stessi archi.
"""
from __future__ import annotations

from dataclasses import dataclass

from .tipi import Evento


@dataclass(frozen=True)
class NodoGrafo:
    id: int
    lemma: str
    pos: str  # VERB, NOUN, PROPN, ADJ, NUM, PRON, ADV, X


@dataclass(frozen=True)
class ArcoGrafo:
    testa: int
    dipendente: int
    relazione: str  # nsubj, obj, iobj, obl:luogo, obl:origine, obl:tempo, ...


@dataclass(frozen=True)
class Grafo:
    nodi: tuple[NodoGrafo, ...]
    archi: tuple[ArcoGrafo, ...]


# Token di prima classe per le domande senza risposta determinabile.
NON_LO_SO = Grafo(
    nodi=(NodoGrafo(id=0, lemma="non-lo-so", pos="X"),),
    archi=(),
)


class _CostruttoreGrafo:
    """Helper interno per assemblare un Grafo con id progressivi."""

    def __init__(self) -> None:
        self._nodi: list[NodoGrafo] = []
        self._archi: list[ArcoGrafo] = []

    def nodo(self, lemma: str, pos: str) -> int:
        id_nuovo = len(self._nodi)
        self._nodi.append(NodoGrafo(id=id_nuovo, lemma=lemma, pos=pos))
        return id_nuovo

    def arco(self, testa: int, dipendente: int, relazione: str) -> None:
        self._archi.append(ArcoGrafo(testa=testa, dipendente=dipendente, relazione=relazione))

    def costruisci(self) -> Grafo:
        return Grafo(nodi=tuple(self._nodi), archi=tuple(self._archi))


def evento_a_grafo(evento: Evento) -> Grafo:
    """Converte un evento strutturato nel grafo concettuale corrispondente.

    Questo è il formato che il cervello (Fase 2) vedrà davvero: il testo
    (Fase 1) è generato a valle di questo grafo, non il contrario.
    """
    c = _CostruttoreGrafo()
    radice = c.nodo(evento.azione, "VERB")
    agente = c.nodo(evento.agente, "PROPN")
    c.arco(radice, agente, "nsubj")

    if evento.oggetto is not None:
        obj = c.nodo(evento.oggetto, "NOUN")
        c.arco(radice, obj, "obj")

    if evento.destinatario is not None:
        dest = c.nodo(evento.destinatario, "PROPN")
        c.arco(radice, dest, "iobj")

    if evento.argomento is not None:
        arg = c.nodo(evento.argomento, "NOUN")
        c.arco(radice, arg, "obl:argomento")

    if evento.luogo_origine is not None:
        origine = c.nodo(evento.luogo_origine, "NOUN")
        c.arco(radice, origine, "obl:origine")

    if evento.luogo is not None:
        luogo = c.nodo(evento.luogo, "NOUN")
        c.arco(radice, luogo, "obl:luogo")

    tempo = c.nodo(str(evento.t), "NUM")
    c.arco(radice, tempo, "obl:tempo")

    return c.costruisci()


def _pos_per_relazione(relazione: str) -> str:
    if relazione == "obl:quantita":
        return "NUM"
    if relazione == "quesito":
        return "PRON"
    return "NOUN"


def grafo_fatto(radice_lemma: str, **archi_per_relazione: str) -> Grafo:
    """Costruisce un piccolo grafo "fatto" (usato per domande e risposte).

    Esempio: grafo_fatto("essere", nsubj="mela", **{"obl:luogo": "cucina"})
    rappresenta il fatto "la mela è in cucina"; il marcatore speciale
    `quesito="dove"` rappresenta l'elemento interrogativo di una domanda.
    """
    c = _CostruttoreGrafo()
    radice = c.nodo(radice_lemma, "VERB")
    for relazione, lemma in archi_per_relazione.items():
        nodo = c.nodo(lemma, _pos_per_relazione(relazione))
        c.arco(radice, nodo, relazione)
    return c.costruisci()


def grafo_a_dict(g: Grafo) -> dict:
    return {
        "nodi": [{"id": n.id, "lemma": n.lemma, "pos": n.pos} for n in g.nodi],
        "archi": [{"testa": a.testa, "dipendente": a.dipendente, "relazione": a.relazione} for a in g.archi],
    }
