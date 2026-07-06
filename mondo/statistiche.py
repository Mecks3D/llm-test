"""Statistiche di copertura sul dataset generato (FASE0.md, criteri di
accettazione): ogni azione e ogni tipo di domanda deve comparire almeno
l'1% delle volte; distribuzione dei lemmi usata per un controllo a vista.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .azioni import AZIONI

# Le ~15 azioni STRIPS di FASE0.md sono quelle su cui vale il criterio di
# accettazione ">= 1%". "bruciare" non è tra queste: è un evento di sistema
# (il camino consuma legna da solo, nessun personaggio lo "fa") emesso solo
# per rispettare l'invariante di conservazione con un evento tracciato
# invece di una sparizione silenziosa — non ci si aspetta che compaia con
# la stessa frequenza di un'azione scelta da un personaggio ogni tick.
AZIONI_CON_SOGLIA = frozenset(AZIONI.keys())


@dataclass
class Statistiche:
    n_storie: int = 0
    n_eventi: int = 0
    n_domande: int = 0
    azioni: Counter = field(default_factory=Counter)
    tipi_domanda: Counter = field(default_factory=Counter)
    non_lo_so_per_tipo: Counter = field(default_factory=Counter)
    lemmi: Counter = field(default_factory=Counter)

    def percentuale_azioni(self) -> dict[str, float]:
        return {a: 100 * c / self.n_eventi for a, c in self.azioni.items()} if self.n_eventi else {}

    def percentuale_tipi_domanda(self) -> dict[str, float]:
        return {t: 100 * c / self.n_domande for t, c in self.tipi_domanda.items()} if self.n_domande else {}

    def percentuale_non_lo_so(self, tipo: str) -> float:
        tot = self.tipi_domanda[tipo]
        return 100 * self.non_lo_so_per_tipo[tipo] / tot if tot else 0.0

    def azioni_sotto_soglia(self, soglia_percento: float = 1.0) -> list[str]:
        pct = self.percentuale_azioni()
        return [a for a, p in pct.items() if a in AZIONI_CON_SOGLIA and p < soglia_percento]

    def tipi_domanda_sotto_soglia(self, soglia_percento: float = 1.0) -> list[str]:
        pct = self.percentuale_tipi_domanda()
        return [t for t, p in pct.items() if p < soglia_percento]


def _lemmi_evento(evento_dict: dict) -> list[str]:
    lemmi = [evento_dict["azione"], evento_dict["agente"]]
    for campo in ("oggetto", "destinatario", "luogo", "luogo_origine", "argomento"):
        if campo in evento_dict:
            lemmi.append(evento_dict[campo])
    return lemmi


def calcola_statistiche(record_dicts: list[dict]) -> Statistiche:
    """`record_dicts` è la lista dei record già serializzati (come quelli
    scritti in JSONL da generatore.py): {"seed", "eventi", "domande"}."""
    stats = Statistiche()
    for record in record_dicts:
        stats.n_storie += 1
        for evento in record["eventi"]:
            stats.n_eventi += 1
            stats.azioni[evento["azione"]] += 1
            for lemma in _lemmi_evento(evento):
                stats.lemmi[lemma] += 1
        for domanda in record["domande"]:
            stats.n_domande += 1
            stats.tipi_domanda[domanda["tipo"]] += 1
            if domanda["grafo_risposta"]["nodi"][0]["lemma"] == "non-lo-so":
                stats.non_lo_so_per_tipo[domanda["tipo"]] += 1
    return stats


def stampa_statistiche(stats: Statistiche) -> None:
    print(f"storie: {stats.n_storie}   eventi: {stats.n_eventi}   domande: {stats.n_domande}")

    print("\n-- copertura azioni (le 15 STRIPS di FASE0.md; 'bruciare' è un evento di sistema, vedi sotto) --")
    for azione, pct in sorted(stats.percentuale_azioni().items(), key=lambda kv: -kv[1]):
        if azione not in AZIONI_CON_SOGLIA:
            continue
        avviso = "  <-- SOTTO 1%!" if pct < 1.0 else ""
        print(f"  {azione:16s} {stats.azioni[azione]:7d}  {pct:5.2f}%{avviso}")

    fuori_soglia = {a: p for a, p in stats.percentuale_azioni().items() if a not in AZIONI_CON_SOGLIA}
    if fuori_soglia:
        print("\n-- eventi di sistema (non soggetti alla soglia 1%) --")
        for azione, pct in sorted(fuori_soglia.items(), key=lambda kv: -kv[1]):
            print(f"  {azione:16s} {stats.azioni[azione]:7d}  {pct:5.2f}%")

    print("\n-- copertura tipi di domanda (e quota non-lo-so) --")
    for tipo, pct in sorted(stats.percentuale_tipi_domanda().items(), key=lambda kv: -kv[1]):
        avviso = "  <-- SOTTO 1%!" if pct < 1.0 else ""
        quota_nls = stats.percentuale_non_lo_so(tipo)
        print(f"  {tipo:12s} {stats.tipi_domanda[tipo]:6d}  {pct:5.2f}%   non-lo-so: {quota_nls:5.1f}%{avviso}")

    print(f"\n-- lemmi distinti usati negli eventi: {len(stats.lemmi)} --")
    for lemma, c in stats.lemmi.most_common(15):
        print(f"  {lemma:16s} {c:7d}")
