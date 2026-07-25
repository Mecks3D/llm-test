"""Riferimento simbolico: la stessa credenza di `mente/`, eseguita a regole.

Regola di metodo (FASE_MENTE.md §9): ogni numero riportato ha accanto il
numero di questo modulo sulla stessa sonda. Se la rete non lo batte, non c'è
risultato — sul `jepa/` non lo batteva e per sei settimane non si era visto.

Nessun torch qui dentro: stdlib puro, gira in secondi su CPU.
"""
from .tracker import IGNOTO, NON_LO_SO, Risposta, Slot, TrackerRegole

__all__ = ["IGNOTO", "NON_LO_SO", "Risposta", "Slot", "TrackerRegole"]
