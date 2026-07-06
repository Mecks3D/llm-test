"""Il confine deterministico tra `mondo/` e il testo italiano (FASE1.md).

Espone l'API pubblica del modulo: caricamento del lessico, morfologia,
contesto di discorso, verbalizzatore, parser e filtro. Cresce una tappa alla
volta (vedi fasi/FASE1_PIANO.md §13); per ora solo il lessico è pronto.
"""
from __future__ import annotations

from . import morfologia
from .lessico import Lessico, VoceLessico, carica_lessico

__all__ = ["Lessico", "VoceLessico", "carica_lessico", "morfologia"]
