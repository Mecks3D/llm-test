"""Il confine deterministico tra `mondo/` e il testo italiano (FASE1.md).

Espone l'API pubblica del modulo: caricamento del lessico, morfologia,
contesto di discorso, verbalizzatore, parser e filtro. Cresce una tappa alla
volta (vedi fasi/FASE1_PIANO.md §13); per ora solo il lessico è pronto.
"""
from __future__ import annotations

from . import morfologia
from .analizza import analizza_domanda, analizza_evento, analizza_risposta, analizza_storia
from .contesto import StatoDiscorso
from .lessico import Lessico, VoceLessico, carica_lessico
from .verbalizza import verbalizza_domanda, verbalizza_evento, verbalizza_risposta, verbalizza_storia

__all__ = [
    "Lessico", "VoceLessico", "carica_lessico", "morfologia",
    "StatoDiscorso", "verbalizza_evento", "verbalizza_storia",
    "verbalizza_domanda", "verbalizza_risposta",
    "analizza_evento", "analizza_storia",
    "analizza_domanda", "analizza_risposta",
]
