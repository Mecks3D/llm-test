"""Canale percettivo: il formato di osservazione e i suoi backend.

`tipi.py` è stdlib puro e non importa nulla del progetto: è la cucitura
sim-to-real. `mente/` vede solo quello, mai `mondo/` (FASE_MENTE.md §3).
"""
from .tipi import ConfigPercezione, Osservazione, Rilevazione

__all__ = ["ConfigPercezione", "Osservazione", "Rilevazione"]
