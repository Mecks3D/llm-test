"""Formato di osservazione: che cosa consegna il sistema di telecamere.

Stdlib puro, nessun import del progetto. È il contratto fra percezione e
`mente/`: la telecamera sintetica (`sintetica.py`) e quella reale
(`reale.py`, futura) producono queste stesse strutture, e il modello non
sa quale delle due sta guardando (FASE_MENTE.md §3, §4).

Le tre configurazioni della decisione 1 (Andrea, 2026-07-25) si ottengono
accendendo campi opzionali:

  solo classi          -> `riquadro=None`, `id_traccia=None`   (default, caso duro)
  + posizione          -> `riquadro` valorizzato
  + track id           -> `id_traccia` valorizzato
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class Rilevazione:
    """Una singola rilevazione: ciò che il detector dice di vedere.

    `classe` è un'etichetta di CLASSE, mai di istanza: "mela", non "mela_3".
    Ricostruire l'individuo dietro la classe è il problema centrale del
    progetto, non un dettaglio da aggirare.
    """

    classe: str
    confidenza: float = 1.0
    riquadro: tuple[float, float, float, float] | None = None
    id_traccia: str | None = None


@dataclass(frozen=True)
class Osservazione:
    """Ciò che una telecamera consegna per un istante e una vista.

    `completa=True` significa: ciò che non è elencato NON è nella vista.
    È evidenza negativa ed è informativissima ("la palla non è in cucina").
    Un sensore affidabile la fornisce, uno rumoroso no: renderla esplicita
    evita che il modello impari a fidarsi di assenze non informative.

    `verita` è materiale DIAGNOSTICO, allineato per indice a `rilevazioni`:
    l'id d'istanza reale dietro ogni rilevazione (`None` per un falso
    positivo, e l'intero campo è `None` quando la verità non esiste, come
    con una telecamera vera). Sta fuori da `Rilevazione` di proposito: non
    fa parte del payload del sensore e `senza_verita()` lo rimuove prima di
    darlo a un modello.
    """

    t: int
    vista: str
    rilevazioni: tuple[Rilevazione, ...] = ()
    completa: bool = True
    verita: tuple[str | None, ...] | None = None

    def __post_init__(self) -> None:
        if self.verita is not None and len(self.verita) != len(self.rilevazioni):
            raise ValueError(
                f"verita ({len(self.verita)}) non allineata a rilevazioni "
                f"({len(self.rilevazioni)}) nella vista {self.vista!r} al tick {self.t}"
            )

    def senza_verita(self) -> "Osservazione":
        """Copia priva del materiale diagnostico: ciò che il modello può vedere."""
        return replace(self, verita=None)

    def classi(self) -> tuple[str, ...]:
        return tuple(r.classe for r in self.rilevazioni)


@dataclass(frozen=True)
class ConfigPercezione:
    """Manopole di degradazione del sensore (FASE_MENTE.md §4.2).

    Il livello di rumore è un ASSE SPERIMENTALE, non una costante: le sonde
    riportano sempre la curva accuratezza vs rumore.

    `persone_identificate`: se True il sensore distingue le persone per nome
    (un sistema con riconoscimento facciale — è una capacità del sensore
    esterno, ammessa dalla decisione 3). Se False anche le persone sono
    classe generica "persona" e l'identità può arrivare solo dalla lingua:
    variante più dura, sensata da M2 in poi, non a M1 dove renderebbe il
    compito degenere.

    `riquadro`: STUB. Il micro-mondo non ha coordinate; finché non è scelto
    il detector reale (decisione 1 lasciata aperta) le posizioni sono
    sintetiche. Non usarle per concludere alcunché sul caso "+ posizione".
    """

    p_mancata: float = 0.0
    p_falso_positivo: float = 0.0
    p_confusione: float = 0.0
    completa: bool = True
    persone_identificate: bool = True
    riquadro: bool = False
    id_traccia: bool = False
    confidenza_informativa: bool = True
    viste: tuple[str, ...] | None = None  # None = tutti i luoghi
    solo_viste_abitate: bool = False

    def pulita(self) -> bool:
        return self.p_mancata == 0.0 and self.p_falso_positivo == 0.0 and self.p_confusione == 0.0


CLASSE_PERSONA_GENERICA = "persona"

# Confusioni plausibili per un detector: oggetti che si somigliano.
# Simmetrica per costruzione in `rumore.py`.
CONFUSIONI_PLAUSIBILI: tuple[frozenset[str], ...] = (
    frozenset({"cestino", "scatola", "secchio"}),  # contenitori
    frozenset({"mela", "pane"}),                   # cibo piccolo
    frozenset({"palla", "mela"}),                  # tondi
    frozenset({"acqua", "legna"}),                 # materiali sfusi
)
