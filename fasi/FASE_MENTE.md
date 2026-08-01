# FASE MENTE — Credenza persistente su entità e relazioni

Stato: piano approvato nelle sue quattro decisioni portanti (2026-07-25), non
ancora implementato.

## 0. Perché questa fase esiste

Due esperimenti hanno fallito in modi opposti e complementari:

- **`v1`** (transformer autoregressivo su testo, esame 0,573) non cabla nulla:
  la rete trova la scorciatoia e fa retrieval associativo per frequenza e
  recency invece di binding entità→luogo (diagnosi in
  `fasi/FASE2_PIANO_DIAGNOSI.md`).
- **`jepa/`** (esame 0,646 su dev) cabla tutto: misurato il 2026-07-25, un
  esecutore simbolico di trenta righe che applica *la stessa*
  rappresentazione fa 0,655 — **la rete non contribuisce nulla**. In più
  legge gli `Evento` del simulatore, cioè la verità simbolica, saltando il
  problema percettivo; e non è ricorrente (tiene solo l'ultimo evento per
  entità e sovrascrive).

La lezione: serve una divisione esplicita fra ciò che è cablato e ciò che è
appreso, e un'ablazione simbolica obbligatoria accanto a ogni numero.

L'obiettivo finale cambia inoltre la natura del problema. Il sistema dovrà
ricevere: (a) un flusso di **percezione** da telecamere esterne, che non vede
eventi ma **stati parziali** — classi di oggetti presenti in una vista, senza
identità di istanza, con buchi ed errori; (b) un flusso **linguistico** sparso
e asincrono, che parla anche di ciò che non si vede. Deve mantenere una
credenza sul mondo e la si deve poter interrogare.

Quindi il problema centrale non è "dove sta la mela" ma **"questa mela è
*quella* mela?"**. Nel deployment reale il collasso `mela_3 → mela` non è un
bug: è il problema.

## 1. Decisioni portanti (Andrea, 2026-07-25)

1. **Percezione**: formato configurabile sui tre casi (solo classi / +
   posizione / + track id). Si allena e si valuta sul caso più difficile —
   **solo classi** — verificando che i casi più facili siano un miglioramento
   gratuito.
2. **Lingua**: solo **input**. L'uscita è lo stato interrogabile, verbalizzato
   dal verbalizzatore deterministico già esistente. **Nessuna generazione
   appresa**: il collo di bottiglia della decodifica (Fase B) esce dal
   progetto.
3. **Regola 1**: il rilevatore di oggetti è un **sensore esterno** e può essere
   pre-addestrato. Tutto ciò che sta a valle — identità, relazioni, lingua,
   ragionamento — nasce vuoto. Nessun embedding o encoder importato nella
   `mente/`.
4. **Orizzonte**: si resta a lungo nel micro-mondo simulato. Il passaggio al
   reale avviene solo quando le sonde di §6 sono verdi.

## 2. Le tre capacità da misurare

Sono la definizione operativa di "buon senso su entità e relazioni". Vanno
misurate **separatamente**, mai in un unico numero aggregato.

1. **Permanenza** — se non vedo più la palla esiste ancora, ed è dove l'ho
   lasciata finché non ho prova del contrario.
2. **Binding / associazione** — questa rilevazione, questo sintagma nominale,
   a quale individuo si riferiscono?
3. **Propagazione relazionale** — Anna tiene il cestino, il cestino contiene la
   mela, Anna va in cucina ⟹ la mela è in cucina, senza averla vista muoversi.

## 3. Moduli e dipendenze

```
mondo/  ──▶ percezione/sintetica.py ─┐
                                     ├─▶ percezione/tipi.py (Osservazione) ──▶ mente/
(detector reale) ──▶ percezione/reale.py ─┘
lingua/ ──▶ (grafo UD) ─────────────────────────────────────────────────────▶ mente/
mente/  ──▶ interroga ──▶ grafo risposta ──▶ lingua/verbalizzatore ──▶ testo
regole/ = esecutore simbolico di riferimento (stessa interfaccia di mente/)
```

Vincolo di disaccoppiamento, non negoziabile: **`mente/` non importa mai
`mondo/`**. Vede solo `percezione/tipi.py` e i grafi UD. È la cucitura
sim-to-real: quando arriva la telecamera vera cambia solo il backend di
`percezione/`.

`mondo/`, `lingua/`, `cervello/`, `jepa/` non si toccano. `jepa/` resta
archiviato come esperimento concluso.

## 4. Il formato di osservazione (`percezione/tipi.py`)

```python
@dataclass(frozen=True)
class Rilevazione:
    classe: str                       # lessico chiuso del micro-mondo
    confidenza: float
    riquadro: tuple[float, ...] | None = None   # caso "+ posizione"
    id_traccia: str | None = None               # caso "+ track id"

@dataclass(frozen=True)
class Osservazione:
    t: int
    vista: str                        # id telecamera (≈ una stanza)
    rilevazioni: tuple[Rilevazione, ...]
    completa: bool                    # vedi sotto
```

`completa=True` significa: **ciò che non è elencato non è nella vista**. È
evidenza *negativa* ed è informativissima ("la palla non è in cucina"). Un
sensore affidabile la fornisce, uno rumoroso no. Renderla esplicita evita che
il modello impari a fidarsi di assenze che non sono informative.

### 4.1 Telecamera sintetica (`percezione/sintetica.py`)

`mondo/` ha già `StatoMondo.testimoni_in(luogo)` e `luogo_effettivo`: la
visibilità c'è. A ogni tick, per ogni luogo, si emette l'insieme delle entità
lì presenti — **escluse quelle dentro contenitori chiusi** (occlusione reale,
non simulata).

### 4.2 Degradazione (`percezione/rumore.py`)

RNG con seed esplicito, deterministico. Manopole indipendenti:

| manopola | effetto |
|---|---|
| `p_mancata` | rilevazione persa |
| `p_falso_positivo` | rilevazione inventata |
| `confusione` | matrice di confusione fra classi vicine |
| `id_istanza` | se `False` (default) le istanze collassano in classi |
| `riquadro`, `id_traccia` | presenza/assenza dei campi opzionali |
| `completa` | evidenza negativa affidabile sì/no |

Il livello di rumore è un **asse sperimentale**, non una costante: si riporta
sempre la curva accuratezza vs rumore.

## 5. Il modello (`mente/`)

### 5.1 Stato — object file a slot

Pool fisso di `K` slot (default 32), allocati dinamicamente. Ogni slot porta:

- `attivo ∈ [0,1]` — occupazione morbida;
- `tipo` — distribuzione sulle classi del lessico;
- `rel` — **logit su `N_luoghi + K + 1` bersagli**: i luoghi, gli altri slot,
  e `ignoto`. Softmax ⟹ esclusività cablata;
- `identita ∈ ℝ^d` — i tratti che distinguono due mele fra loro (dove l'ho
  vista, chi l'ha toccata, quando);
- `eta_evidenza` — tick dall'ultima evidenza diretta.

I bersagli includono **gli slot stessi**: "tenuto da Anna" e "dentro il
cestino" diventano la stessa relazione, e le persone sono slot come gli altri.
La transitività si ottiene per costruzione anziché con casi speciali (il
`jepa/` aveva `luoghi + persone` cablati a mano, e infatti i contenitori
sparivano: 731 eventi su 6139 scartati in silenzio).

I luoghi restano un insieme fisso e separato: la mappa è conoscenza di sfondo
(`PROGETTO.md`), non un'entità da tracciare.

### 5.2 Risoluzione della catena (cablata)

Probabilità di trovarsi in un luogo fisico, per punto fisso:

```
L₀ = P_luogo
Lₙ₊₁ = P_luogo + P_slot · Lₙ        (3–4 iterazioni)
```

Aciclicità: penalità morbida sulla diagonale e sui 2-cicli di `P_slot`.

### 5.3 Persistenza (cablata) — il fix centrale

L'aggiornamento è un **delta con gate**, mai una sovrascrittura:

```
delta = MLP_delta(evidenza, stato_slot)
gate  = σ(MLP_gate(tipo_evidenza, qualità, stato_slot))
rel ← rel + gate · delta
```

Nessuna evidenza ⟹ nessun contributo ⟹ la credenza **persiste**. La permanenza
degli oggetti è architetturale, non sperata. (`jepa/modello_jepa.py:138` faceva
l'opposto: `= deltas`.)

### 5.4 Associazione dati (appresa) — il cuore

Date `M` rilevazioni e `K` slot, si costruisce una matrice di punteggio
`M × (K+1)` (il `+1` è "entità nuova") e si normalizza con Sinkhorn per
ottenere un assegnamento morbido e differenziabile (una rilevazione, uno slot).

**Attenzione al dustbin — errore facile.** Una Sinkhorn doppiamente stocastica
normalizza *anche* la colonna `K+1`, che quindi assorbe al più **una**
rilevazione. Se in un frame compaiono tre oggetti nuovi, due vengono forzati
dentro slot esistenti sbagliati e il binding si corrompe in silenzio. Serve la
formulazione di SuperGlue: **riga e colonna di scarto con marginale rilassato**
— la colonna `K+1` ha capacità `M` (non 1) e la riga `M+1` ha capacità `K`,
così ogni rilevazione può essere nuova e ogni slot può restare non osservato.
Le altre righe e colonne mantengono marginale 1.

Test di accettazione, da scrivere prima del modello: un frame con `n` classi
mai viste su una memoria vuota deve allocare **`n` slot distinti**, per
`n = 1, 2, 3`. È il caso che la formulazione ingenua sbaglia.

Tratti del punteggio: compatibilità di classe, compatibilità spaziale (credo
che questo slot sia in questa stanza?), recency, similarità dell'embedding
d'identità, corrispondenza di `id_traccia` quando c'è.

Qui vive il buon senso: *la mela che credevo in cucina probabilmente non è
questa che vedo in giardino, a meno che qualcuno non l'abbia portata.*

### 5.5 Il gate dell'evidenza (appreso)

Un modello che pesa uguale "vedo la mela" e "Marco dice che la mela è in
cucina" non ha buon senso. `tipo_evidenza ∈ {visione diretta, assenza da vista
completa, lingua, inferenza}` entra nel gate.

### 5.6 Canale linguistico (`mente/lingua.py`)

Grafo UD → evidenza sugli slot. Riusa l'idea di
`jepa/vocabolario_grafi.py::estrai_evento_da_grafo` (che nel `jepa/` era scritto
e mai usato fuori dai test), ma i referenti vanno **risolti a slot**: la
coreferenza diventa lo stesso problema dell'associazione dati di §5.4, con gli
stessi pesi.

### 5.7 Interrogazione (`mente/interroga.py`)

Domande: dov'è X, chi ha Y, cosa c'è in Z, quanti X. Risposta per
probabilità sullo stato, con **astensione** quando l'entropia supera una
soglia calibrata. Uscita: grafo risposta → verbalizzatore deterministico.
Valutazione **grafo vs grafo** (regola 4), mai stringa vs stringa.

### 5.8 Dimensioni

`K=32`, `d=64` ⟹ ordine 10⁵ parametri. Gira su CPU in locale, in secondi.
`torch.manual_seed` esplicito ovunque (regola 2): il `jepa/` non ce l'ha e
infatti dà 63,62% o 64,63% a seconda del giro.

## 6. Obiettivo di addestramento

**Principale — predittivo auto-supervisionato, senza etichette.** Dalla
credenza corrente si predice la **prossima osservazione**: quali classi
comparirebbero se la telecamera *V* guardasse al tick *t+1*. È l'unica
supervisione che il mondo dà davvero, ed è ciò che forza la permanenza: per
predire "quando Anna entra in cucina vedrai la palla" devi aver conservato
dov'è la palla.

Nota: il bersaglio è **discreto e ancorato** (rilevazioni vere), non un latente.
Quindi non servono EMA, stop-gradient né altri accorgimenti anti-collasso: il
collasso è impossibile per costruzione. È l'idea JEPA fatta nel modo che qui
funziona.

**Trappola da misurare.** La loss predittiva è banalmente soddisfacibile
prevedendo "non cambia nulla". Si riporta quindi sempre la loss ristretta al
sottoinsieme **sorprendente** — le rilevazioni che differiscono dal frame
precedente. Un modello che non batte "copia il frame precedente" su quel
sottoinsieme non ha imparato niente.

**Ausiliaria, solo nello stadio iniziale del curriculum** — supervisione sullo
stato vero del mondo, come un bambino a cui si dice come stanno le cose. Si
spegne appena le sonde reggono senza.

**Le domande non sono un obiettivo di addestramento**: solo valutazione.

## 7. Curriculum

| Stadio | Ingresso | Che cosa deve emergere |
|---|---|---|
| M0 | — | fondamenta e riferimento simbolico, nessuna rete |
| M1 | sola percezione | permanenza, associazione, propagazione |
| M2 | percezione + lingua sul visibile | grounding: la frase e la vista parlano della stessa cosa |
| M3 | + lingua sul non visibile | testimonianza, fiducia, evidenze in conflitto |
| M4 | esame ufficiale | seed ≥ 1.000.000, grafo vs grafo |

È anche il curriculum di un bambino: prima la permanenza (~8 mesi), poi il
grounding, poi il sentito dire.

## 8. Sonde di valutazione

Mai un numero solo. Ogni sonda accanto al riferimento simbolico di §9.

| # | sonda | misura |
|---|---|---|
| P1 | permanenza | accuratezza vs N tick dall'ultima vista |
| P2 | binding | due istanze identiche (`mela_1`, `mela_2`) nella stessa storia |
| P3 | interferenza | accuratezza vs cast 1→6 — **il criterio di accettazione del piano JEPA, mai misurato** |
| P4 | calibrazione | curva astensione/accuratezza, ECE sul non-lo-so |
| P5 | robustezza | accuratezza vs `p_mancata`, `p_falso_positivo`, confusione |
| P6 | propagazione | profondità della catena di contenimento 1, 2, 3 |
| P7 | sorpresa | loss predittiva sul sottoinsieme che cambia (§6) |

### 8.1 Modalità solo-lingua

Ablazione con il canale percettivo spento: ingresso solo testo, come `v1`.
È **l'unico confronto onesto con `v1` (0,573)** — stessa distribuzione, stesso
ingresso, stesso protocollo d'esame. E risponde finalmente alla domanda di
ricerca rimasta aperta dalla Fase B: *lo stato strutturato rompe la
scorciatoia di binding?*

## 9. Riferimento simbolico obbligatorio (`regole/`)

Esecutore a regole con la **stessa interfaccia** di `mente/`: consuma
`Osservazione` e grafi UD, mantiene lo stesso stato, risponde alle stesse
domande. Nessun torch.

Regola di metodo: **ogni numero riportato ha accanto il numero del riferimento
simbolico sulla stessa sonda.** Se la rete non lo batte, non c'è risultato.
Misurato il 2026-07-25: sul `jepa/` non lo batteva, e per sei settimane non si
era visto.

Il riferimento serve anche come tetto: sul dev del `jepa/`, un tracker
simbolico grezzo (istanze vere, contenitori, "ogni evento con `luogo`
localizza agente e testimoni") fa già 87% sulla posizione contro il 64% della
rete.

## 10. Rischi noti e mitigazioni

| rischio | mitigazione |
|---|---|
| l'associazione dati non converge | stadio M1a con associazione forzata dall'oracolo (teacher forcing), poi si rilassa |
| instabilità nell'allocazione degli slot | `K` fisso e generoso, penalità sull'occupazione, slot mai deallocati dentro una storia |
| loss predittiva banale | P7: si misura solo sul sottoinsieme sorprendente |
| gap sim-to-real | `mente/` non importa `mondo/`; il rumore è un asse, non una costante |
| dimenticare l'ablazione | il riferimento simbolico gira nello stesso comando di valutazione |

## 11. Ordine di lavoro

1. **M0.1** `percezione/tipi.py` + `sintetica.py` + `rumore.py`, con test.
2. **M0.2** `regole/` completo, e le sette sonde di §8 sopra di esso. Questo dà
   il tetto e l'impalcatura di misura prima di scrivere una riga di rete.
3. **M0.3** riportare le sonde per `v1` e per `jepa/` con lo stesso strumento,
   così la storia del progetto è su una scala unica.
4. **M1** slot, persistenza, catena, loss predittiva. Associazione forzata,
   poi appresa. Sonde P1, P2, P3, P5, P6, P7.
   **Piano esecutivo: `fasi/FASE_MENTE_M1_PIANO.md`.** Contiene i numeri da
   battere rimisurati su un campione utile, che correggono §12.2 (vedi la nota
   qui sotto), e il criterio di fallimento.
5. **M2–M3** canale linguistico.
6. **M4** esame ufficiale + modalità solo-lingua (§8.1).

M0 è interamente stdlib, gira in secondi su CPU, ed è utile anche se la parte
neurale venisse abbandonata.

## 12. Risultati di M0 (misurati il 2026-07-25)

`.venv/bin/python -m sonde.esegui --storie 40 [--con-jepa]`, 40 storie, cast
pieno, seed 2000-2039. 35 secondi su CPU (70 in più con `--con-jepa`).

### 12.1 Le due scale, e perché servono entrambe

| sistema | canale | narrativa | reale |
|---|---|---|---|
| `jepa/` | lingua | 0,651 | 0,719 |
| `regole/` | lingua | 0,804 | 0,751 |
| `regole/` | visione | 0,666 | 0,772 |
| `regole/` | visione + lingua | 0,779 | **0,872** |

Scoperta di metodo, emersa misurando: **l'oro epistemico di
`mondo/domande.py` è definito rispetto alla sola narrazione.** Dice "non lo
so" quando il fatto non è derivabile dagli eventi raccontati — ma la
telecamera l'ha visto, e il sistema viene contato in errore per aver saputo.
Su 49 regressioni apparenti aggiungendo la vista alla lingua, 26 erano di
questo tipo. Da qui le due scale di `banco.Esito`:

- **narrativa**, contro l'oro di `mondo/domande.py`: è la scala storica del
  progetto (`v1` 0,573, `jepa/` 0,646), onesta solo per il canale linguistico;
- **reale**, contro lo stato vero del mondo: la scala del deployment, l'unica
  su cui i canali si possono confrontare fra loro.

Sulla scala reale i due canali sono complementari e la fusione (0,872) batte
entrambi i canali singoli (0,751 e 0,772). È il numero che giustifica
l'architettura, ed è protetto da un test.

### 12.2 Le sonde

| sonda | esito |
|---|---|
| **P1 permanenza** | crolla: 0,80 a età 0, ~0,10 da 3 tick in poi, 0,00 sul mai visto. Con 2 camere su 6 e sola visione il riferimento non ha nessun modello di ciò che accade fuori campo: **è il vuoto che `mente/` deve riempire** (ma vedi la correzione qui sotto) |
| **P2 binding** | 0,718 sulle istanze ambigue contro 0,883 sulle uniche: 17 punti di divario. Purezza delle rilevazioni 0,975 |
| **P3 interferenza** | 0,947 (cast 1) → 0,872 (cast 6): degrado di 7 punti. `v1` sullo stesso asse crollava da 0,98 a 0,57. **Lo stato strutturato regge dove il transformer si sfalda** |
| **P4 calibrazione** | astenendosi sul 50% meno sicuro l'esattezza sale da 0,870 a 0,907: la confidenza significa qualcosa, ma poco |
| **P5 robustezza** | 0,872 pulito → 0,832 con rumore pieno. Falsi positivi e confusione di classe fanno più danno delle mancate rilevazioni, perché corrompono il binding (purezza 0,975 → 0,886) |
| **P6 propagazione** | 0,957 / 0,871 / 0,625 a profondità 1 / 2 / 3 |
| **P7 sorpresa** | 0,059 sulle classi che cambiano, contro 0,000 della baseline che copia il frame. **Il riferimento non ha modello del moto: è il bersaglio più chiaro per la rete** |

### 12.2.1 Correzione a P1 e P7 (2026-08-01)

Rimisurando su un campione utile per scrivere il piano di M1 sono emerse due
cose che rendono i numeri qui sopra inadatti a fare da bersaglio.

**P1 era misurata su un campione troppo piccolo.** A `n_storie=20` (il default
delle sonde) la fascia "età ≥ 1" contiene **30 casi**, da cui il "~0,10". A
`n_storie=200` sono 272 casi e il valore è **0,0588**; a 400 casi sono 517 e
resta 0,0580. Il numero da battere è 0,0588, e tutte le misure di M1 si fanno
ad almeno 200 storie (7,5 s in locale). Da notare anche la distribuzione della
massa: "età 0" è il 44% delle domande e "mai vista" il 47%, quindi la fascia
che misura davvero la permanenza è il 9% del totale.

**La baseline "copia frame" di P7 è uno zero per costruzione, non una misura.**
`banco._misura_predizione` definisce il sottoinsieme sorpresa come
`reale △ precedente`, e la baseline predice `precedente`: su una differenza
simmetrica `(c in prima) == (c in reale)` è falsa per ogni `c`, sempre. Quindi
§6 di questo documento è troppo generoso quando dice che "un modello che non
batte la copia del frame non ha imparato niente": batterla non significa
niente. L'unico confronto che conta su P7 è con `regole/`.

Per completezza, P7 a `n_storie=200`: sola visione **0,0083**, visione+lingua
0,0505 (il 0,059 della tabella era a 40 storie su entrambi i canali). M1 è lo
stadio a sola percezione, quindi il suo bersaglio è 0,0083.

### 12.3 Tre regole di buon senso trovate misurando

Non erano nel piano; sono emerse perché le sonde le hanno rese visibili, e
sono esattamente ciò che §5 chiede alla rete di imparare da sola.

1. **La vista conferma, non sovrascrive.** Vedere la mela in cucina non deve
   cancellare "è in mano ad Anna": la telecamera non sa distinguere i due
   casi. Senza questa regola il possesso crolla da 0,77 a 0,50 appena si
   accende la vista accanto alla lingua.
2. **L'assenza non azzera.** Ablazione P5b: `ignora` 0,872, `dubita` 0,872,
   `azzera` 0,792. Un oggetto che sparisce dalla vista di solito è solo
   nascosto — nel micro-mondo i contenitori si chiudono. Azzerare la credenza
   costa 8 punti. Default: `dubita` (tiene la credenza, abbassa la confidenza).
3. **L'esclusività vale anche in fase di associazione.** Due mele viste nello
   stesso istante in due stanze diverse non possono essere lo stesso
   individuo. Senza il vincolo il tracker le fonde e si convince che la mela
   si teletrasporti (trovato da un test, non a occhio).

### 12.4 Che cosa resta scoperto

- **`v1` non è ancora misurato — il lavoro è pronto, va lanciato su Colab.**
  L'adattatore c'è (`sonde/adattatori.py::esiti_v1`, `--con-v1` nella CLI) e
  funziona su campioni piccoli. La misura completa è però un job torch da
  decine di minuti su una macchina a 4 core, quindi va su GPU: serve una cella
  nel notebook e il push del commit. Un tentativo in locale il 2026-07-26 è
  stato ucciso dal timeout dopo 50 minuti senza produrre nulla.

  Due cose da sapere prima di rilanciarlo:

  - **Il confronto va fatto sulla distribuzione di `v1`, non su quella delle
    sonde.** Lo stadio 1 di `v1` vive su storie di 3-6 tick e sole domande di
    posizione (`configs/v1.yaml`); sulle storie di 8-22 tick sarebbe fuori
    distribuzione. Di qui `banco.lunghezza_stadio1` (byte-identica a
    `esami/genera.py::_n_tick`) e i seed d'esame >= 1.000.000 — il regime
    esatto in cui `v1` ha prodotto lo 0,573.
  - **I checkpoint sono fuori dal repo**, in `/home/andrea/Scaricati/`
    (`v1`, `v1_facile`, `v1_anti`, `v1_grad1..3`), e non si caricavano più: il
    vocabolario è cresciuto di due token dopo quei run (`che-cosa` per
    l'esperimento "tempo", `[STATO]` per la Fase B). Sono appesi in coda, gli
    id vecchi non si sono spostati, quindi `_carica_modello_epoca` costruisce
    il modello alla dimensione letta dal checkpoint. Un test in
    `tests/test_regole.py` fallisce rumorosamente se un domani il vocabolario
    smettesse di essere un'estensione.
- `riquadro` è uno stub (il micro-mondo non ha coordinate): il caso
  "+ posizione" della decisione 1 non è ancora valutabile davvero.
- P6 a profondità 3 ha pochi casi (N=8): serve un campione più grande, o
  storie costruite apposta.

## 13. Il documento di visione: che cosa se ne prende

Esiste un documento di visione a ruota libera, prodotto con un altro agente
(`visione-sviluppo-cervello-bambino.md`, tenuto fuori dal repo). È un buon
documento di **direzione a lungo termine** e un cattivo documento di **piano**:
gran parte riscrive §5 in prosa più ambiziosa, e va letto sapendo che questo
file resta la specifica. Triage fatto il 2026-08-01.

**Assorbito qui.** La correzione del dustbin di Sinkhorn (§5.4) nasce
dall'esame della sua §2, che ha l'errore descritto sopra.

**Accolto come lavoro futuro, fuori da M1–M4.**

- **M5 — teoria della mente (test di Sally-Anne).** È l'unica idea davvero
  nuova del documento con una realizzazione architetturale evidente: una
  seconda mappa di credenza indicizzata per agente, con lo stesso codice di §5,
  aggiornata solo dalle evidenze che *quell'agente* ha ricevuto. Dà un esame
  che nessun'altra parte del progetto sa dare. Prerequisito: M3 verde.
- **Memoria dormiente per cambio di contesto.** Snapshot degli slot quando il
  contesto cambia. Ragionevole quando i luoghi supereranno `K=32`; nel
  micro-mondo attuale `K=32` copre tutto, quindi non serve.
- **Log causale per slot** ("`git log` dell'entità"). Costa una lista in
  append e renderebbe le sonde molto più leggibili in diagnosi. Da valutare in
  M1 se il debug lo richiede, non prima.
- **Slot #0 "IO".** Riservare lo slot 0 costa zero e un domani serve per le
  relazioni egocentriche. Ma finché il sistema è un *osservatore* che non
  agisce, è una prenotazione e non una funzione: si riserva l'indice, non si
  costruisce niente sopra.

**Da chiarire prima di usarlo.** Il documento parla di "grafi di dipendenza
sintattica UD standard" come se esistessero nel progetto. Non esistono: non c'è
spaCy da nessuna parte e `lingua/analizza.py` è un parser scritto a mano su
lessico chiuso. O è solo un nome nuovo per ciò che c'è già — e allora si dice
"grafo di dipendenza", senza "UD" — oppure introduce una dipendenza che tocca
la Regola 1 di `CLAUDE.md` e va decisa esplicitamente. §5.6 di questo file
intende la prima lettura.

**Non accolto.** La sezione su traumi e "riconcettualizzazione notturna" non
descrive un meccanismo: "un evento isolato non riscrive una regola generale" è
un aggiornamento bayesiano con prior, cioè esattamente ciò che il gate di §5.5
fa se è addestrato bene. Chiamarlo AI safety fa sembrare fatta una cosa che non
è stata nemmeno definita. Resta come nota, fuori dal piano. Stessa sorte per il
"Global Workspace", che è un'etichetta sulla mappa a slot che già esiste.
