"""Tracker a regole: credenza persistente su individui e relazioni.

Stessa interfaccia che avrà `mente/` (FASE_MENTE.md §9), così le sonde girano
identiche sui due e i numeri stanno sulla stessa scala:

    osserva(Osservazione)   evidenza percettiva
    ascolta(Grafo)          evidenza linguistica
    dove(slot) / chi_ha(slot) / predici(vista)

Lo stato è un insieme di **slot**, uno per individuo creduto esistente. Il
bersaglio di uno slot è un luogo oppure **un altro slot**: "tenuto da Anna" e
"dentro il cestino" sono la stessa relazione, e la catena si risolve per
risalita (FASE_MENTE.md §5.1-5.2). Il `jepa/` cablava invece `luoghi+persone`
e i contenitori sparivano in silenzio: 731 eventi su 6139 scartati.

Persistenza: senza evidenza la credenza NON cambia (§5.3). È ciò che dà la
permanenza degli oggetti.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from percezione.tipi import Osservazione

NON_LO_SO = "non lo so"
IGNOTO = "ignoto"
NESSUNO = "nessuno"

# Tipi di evidenza, in ordine di peso (§5.5): vedere batte il sentir dire.
PESO_EVIDENZA = {"visione": 3, "lingua": 2, "assenza": 1, "inferenza": 0}


@dataclass
class Slot:
    """Un individuo creduto esistente."""

    id: int
    classe: str
    rel_tipo: str = IGNOTO  # "luogo" | "slot" | "distrutto" | IGNOTO
    rel_valore: str | int = ""
    t_evidenza: int = -1  # ultimo tick con evidenza DIRETTA
    origine: str = "visione"  # come è nato lo slot
    conferme: int = 0
    assenze: int = 0  # viste complete consecutive in cui non è comparso

    def noto(self) -> bool:
        return self.rel_tipo in ("luogo", "slot")


@dataclass
class TrackerRegole:
    """Credenza a slot aggiornata da percezione e lingua.

    `assenze_per_dubbio`: quante viste complete consecutive senza vedere
    l'individuo servono prima di sospendere il giudizio. Con un sensore
    pulito basta 1; con `p_mancata > 0` alzarlo evita di buttare via una
    credenza giusta per una singola mancata rilevazione.
    """

    luoghi: tuple[str, ...]
    assenze_per_dubbio: int = 1
    politica_assenza: str = "dubita"
    classi_persona: frozenset[str] = frozenset()
    slot: list[Slot] = field(default_factory=list)
    _per_lingua: dict[str, int] = field(default_factory=dict)  # id d'istanza -> slot
    _nome_di_slot: dict[int, str] = field(default_factory=dict)  # slot -> id d'istanza
    _t: int = 0
    # DIAGNOSTICA: (slot, id d'istanza reale) per ogni rilevazione assorbita.
    # Serve solo ad `allineamento.py` per valutare il binding; nessuna
    # decisione del tracker la legge (garantito da test_regole.py).
    _assorbite: list[tuple[int, str]] = field(default_factory=list)

    # -- ciclo di vita degli slot -------------------------------------------

    def _nuovo(self, classe: str, origine: str) -> Slot:
        s = Slot(id=len(self.slot), classe=classe, origine=origine)
        self.slot.append(s)
        return s

    def _sposta(self, s: Slot, tipo: str, valore: str | int, evidenza: str) -> None:
        s.rel_tipo, s.rel_valore = tipo, valore
        s.assenze = 0
        if PESO_EVIDENZA[evidenza] >= PESO_EVIDENZA["lingua"]:
            s.t_evidenza = self._t
            s.conferme += 1

    # -- risoluzione della catena (§5.2) ------------------------------------

    def luogo_di(self, s: Slot, profondita: int = 0) -> str:
        if profondita > 8:  # difesa contro catene cicliche
            return NON_LO_SO
        if s.rel_tipo == "luogo":
            return str(s.rel_valore)
        if s.rel_tipo == "slot":
            return self.luogo_di(self.slot[int(s.rel_valore)], profondita + 1)
        return NON_LO_SO

    def portatore_di(self, s: Slot, profondita: int = 0) -> str:
        """Chi tiene l'individuo: risale finché trova una persona, oppure
        `nessuno` se la catena finisce in un luogo."""
        if profondita > 8:
            return NON_LO_SO
        if s.rel_tipo == "luogo":
            return NESSUNO
        if s.rel_tipo == "slot":
            contenitore = self.slot[int(s.rel_valore)]
            if contenitore.classe in self.classi_persona:
                return contenitore.classe
            return self.portatore_di(contenitore, profondita + 1)
        return NON_LO_SO

    # -- canale percettivo ---------------------------------------------------

    def osserva(self, oss: Osservazione) -> None:
        """Evidenza visiva su una vista. Vede classi, non individui: deve
        decidere da sé quale slot sta guardando (associazione dati, §5.4)."""
        self._t = max(self._t, oss.t)
        vista = oss.vista
        assegnati: set[int] = set()
        verita = oss.verita or (None,) * len(oss.rilevazioni)

        ordine = sorted(range(len(oss.rilevazioni)), key=lambda i: -oss.rilevazioni[i].confidenza)
        for i in ordine:
            ril = oss.rilevazioni[i]
            s = self._associa(ril.classe, vista, escludi=assegnati)
            if s is None:
                s = self._nuovo(ril.classe, origine="visione")
            assegnati.add(s.id)
            if self.luogo_di(s) == vista:
                # La credenza corrente è già COERENTE con ciò che vedo. La vista
                # non sa distinguere "la mela è in cucina" da "la mela è in mano
                # ad Anna che è in cucina": se sovrascrivesse, cancellerebbe la
                # relazione più ricca stabilita dalla lingua. Conferma e basta.
                # (Misurato: senza questa regola il possesso crolla da 0,77 a 0,50
                # quando si accende la vista accanto alla lingua.)
                s.t_evidenza, s.assenze = self._t, 0
                s.conferme += 1
            else:
                self._sposta(s, "luogo", vista, "visione")
            if verita[i] is not None:
                self._assorbite.append((s.id, verita[i]))

        if oss.completa and self.politica_assenza != "ignora":
            self._evidenza_negativa(vista, assegnati)

    def _associa(self, classe: str, vista: str, escludi: set[int]) -> Slot | None:
        """Quale slot sto guardando? Preferisce, nell'ordine:
        chi credo già qui, chi ho perso di vista da più tempo, chi non so dove sia.

        Il criterio "chi ho perso da più tempo" è la versione simbolica del
        buon senso che la rete dovrà imparare: la mela che credevo in cucina
        difficilmente è questa che vedo in giardino, se l'ho vista in cucina
        un attimo fa.
        """
        candidati = [
            s
            for s in self.slot
            if s.classe == classe
            and s.id not in escludi
            and s.rel_tipo != "distrutto"
            and not self._smentito_dallesclusivita(s, vista)
        ]
        if not candidati:
            return None
        # "credo già qui" va giudicato sulla catena risolta, non sulla relazione
        # diretta: un oggetto in mano a chi sta in cucina è in cucina.
        qui = [s for s in candidati if self.luogo_di(s) == vista]
        if qui:
            return max(qui, key=lambda s: s.t_evidenza)
        ignoti = [s for s in candidati if not s.noto()]
        if ignoti:
            return min(ignoti, key=lambda s: s.t_evidenza)
        return min(candidati, key=lambda s: s.t_evidenza)

    def _smentito_dallesclusivita(self, s: Slot, vista: str) -> bool:
        """Questo slot NON può essere ciò che sto guardando: l'ho già visto
        altrove in questo stesso istante, e una cosa sta in un posto solo.

        È l'esclusività (§5.1) portata dentro l'associazione dati. Senza,
        due mele viste nello stesso tick in due stanze diverse finiscono nello
        stesso slot e il tracker si convince che la mela teletrasporti.
        """
        return s.t_evidenza == self._t and self.luogo_di(s) not in (vista, NON_LO_SO)

    def _evidenza_negativa(self, vista: str, visti: set[int]) -> None:
        """Vista completa: ciò che non compare qui, qui non c'è.

        Vale però solo per chi credo appoggiato *direttamente* nel luogo. Se
        credo che la mela sia dentro il cestino, non vederla non è una
        contraddizione: i contenitori occludono. Applicare l'assenza anche a
        quei casi distrugge ciò che la lingua ha stabilito e che la telecamera
        non può smentire.

        `politica_assenza` decide che farne (ablazione nella sonda P5):
          "azzera"  non lo vedo -> non so più dov'è. Massima reattività,
                    ma butta via credenze giuste ogni volta che qualcosa
                    finisce dentro un contenitore che poi viene chiuso.
          "dubita"  tengo la credenza e abbasso la confidenza. Un oggetto che
                    sparisce dalla vista di solito è ancora lì, solo nascosto:
                    è il buon senso che serve, ed è il default perché misurato
                    migliore (P5).
          "ignora"  l'assenza non dice nulla (sensore inaffidabile).
        """
        for s in self.slot:
            if s.id in visti or s.rel_tipo != "luogo" or s.rel_valore != vista:
                continue
            s.assenze += 1
            if self.politica_assenza == "azzera" and s.assenze >= self.assenze_per_dubbio:
                s.rel_tipo, s.rel_valore = IGNOTO, ""

    # -- canale linguistico (§5.6) -------------------------------------------

    def _slot_per_nome(self, nome: str, luogo_atteso: str | None = None) -> Slot:
        """Risolve un riferimento linguistico a uno slot, **fondendo i canali**.

        La lingua nomina individui (`mela_3`), la vista vede solo classi: al
        primo incontro il nome va legato a uno degli slot che la vista ha già
        costruito, altrimenti i due canali mantengono credenze separate sullo
        stesso oggetto e si sabotano a vicenda (l'evidenza negativa della
        vista cancella ciò che la lingua ha appena stabilito — misurato).

        È la stessa associazione dati di `_associa`, con un indizio in più: il
        luogo di cui la frase sta parlando.
        """
        if nome in self._per_lingua:
            return self.slot[self._per_lingua[nome]]

        classe = nome.split("_")[0]
        candidati = [
            s
            for s in self.slot
            if s.classe == classe and s.id not in self._nome_di_slot and s.rel_tipo != "distrutto"
        ]
        scelto: Slot | None = None
        if candidati:
            qui = [s for s in candidati if s.rel_tipo == "luogo" and s.rel_valore == luogo_atteso]
            ignoti = [s for s in candidati if not s.noto()]
            if qui:
                scelto = max(qui, key=lambda s: s.t_evidenza)
            elif luogo_atteso is None and ignoti:
                scelto = min(ignoti, key=lambda s: s.t_evidenza)
            elif ignoti:
                scelto = min(ignoti, key=lambda s: s.t_evidenza)
        if scelto is None:
            scelto = self._nuovo(classe, origine="lingua")

        self._per_lingua[nome] = scelto.id
        self._nome_di_slot[scelto.id] = nome
        return scelto

    def ascolta_evento(
        self,
        azione: str,
        agente: str | None,
        oggetto: str | None = None,
        luogo: str | None = None,
        destinatario: str | None = None,
        argomento: str | None = None,
        testimoni: Iterable[str] = (),
        t: int | None = None,
    ) -> None:
        """Evidenza linguistica da un evento raccontato.

        Riceve i campi già estratti dal grafo UD (`lingua/` a valle del parser):
        `regole/` non fa parsing.
        """
        if t is not None:
            self._t = max(self._t, t)

        qui = luogo if luogo in self.luoghi else None

        # (a) qualunque frase che nomini un luogo localizza chi agisce e chi assiste.
        #     È l'informazione che l'estrattore del `jepa/` buttava via: 60,4%
        #     degli eventi, e da sola vale ~20 punti di accuratezza.
        if qui:
            if agente:
                self._sposta(self._slot_per_nome(agente, qui), "luogo", qui, "lingua")
            for testimone in testimoni:
                self._sposta(self._slot_per_nome(testimone, qui), "luogo", qui, "lingua")

        # (b) transizioni di contenimento
        if azione == "andare" and agente and qui:
            self._sposta(self._slot_per_nome(agente, qui), "luogo", qui, "lingua")
        elif azione in ("prendere", "raccogliere", "estrarre", "tirare_fuori") and oggetto and agente:
            self._sposta(
                self._slot_per_nome(oggetto, qui),
                "slot",
                self._slot_per_nome(agente, qui).id,
                "lingua",
            )
        elif azione == "posare" and oggetto and qui:
            self._sposta(self._slot_per_nome(oggetto, qui), "luogo", qui, "lingua")
        elif azione in ("mettere", "mettere_dentro") and oggetto and argomento:
            self._sposta(
                self._slot_per_nome(oggetto, qui),
                "slot",
                self._slot_per_nome(argomento, qui).id,
                "lingua",
            )
        elif azione == "dare" and oggetto and destinatario:
            self._sposta(
                self._slot_per_nome(oggetto, qui),
                "slot",
                self._slot_per_nome(destinatario, qui).id,
                "lingua",
            )
        elif azione in ("mangiare", "bruciare") and oggetto:
            s = self._slot_per_nome(oggetto, qui)
            s.rel_tipo, s.rel_valore = "distrutto", ""
            s.t_evidenza = self._t
        elif oggetto and qui:
            # l'oggetto nominato in una frase è dove agisce chi la compie
            self._sposta(self._slot_per_nome(oggetto, qui), "luogo", qui, "lingua")

    # -- interrogazione (§5.7) -----------------------------------------------

    def dove(self, s: Slot | None) -> Risposta:
        if s is None:
            return Risposta(NON_LO_SO, 0.0)
        return Risposta(self.luogo_di(s), self._confidenza(s))

    def chi_ha(self, s: Slot | None) -> Risposta:
        if s is None:
            return Risposta(NON_LO_SO, 0.0)
        return Risposta(self.portatore_di(s), self._confidenza(s))

    def _confidenza(self, s: Slot) -> float:
        """Decresce con l'età dell'evidenza e con le assenze non spiegate:
        è la base della curva di astensione (P4)."""
        if not s.noto():
            return 0.0
        eta = max(0, self._t - s.t_evidenza)
        return round(1.0 / (1.0 + 0.15 * eta + 0.30 * s.assenze), 3)

    def predici(self, vista: str) -> tuple[str, ...]:
        """Classi che mi aspetto di vedere se la telecamera guarda `vista`.

        È il bersaglio della loss predittiva di `mente/` (§6) e la sonda P7.
        """
        attese = [s.classe for s in self.slot if self.luogo_di(s) == vista]
        return tuple(sorted(attese))


@dataclass(frozen=True)
class Risposta:
    valore: str
    confidenza: float = 1.0

    def astenuta(self) -> bool:
        return self.valore == NON_LO_SO
