# M1 — Slot, persistenza, catena, loss predittiva

Piano esecutivo dello stadio M1 di `fasi/FASE_MENTE.md` §7 e §11.4.
Prerequisito: M0 (fatto, `3c68bf4`). Leggere prima `FASE_MENTE.md`, tutto.

Stato: **APPROVATO da Andrea il 2026-08-01.** Si esegue alla lettera. Le
decisioni di design sono chiuse: non si riaprono, non si "migliorano" per
iniziativa. I punti su cui fermarsi e chiedere sono elencati in §9 e sono gli
unici.

---

## 0. Il criterio di successo, prima di tutto il resto

M1 ha successo se **`mente/` batte `regole/` su P1 e P7**, misurate con lo
stesso comando, sugli stessi seed, sullo stesso canale. Nient'altro conta: non
la loss che scende, non un numero aggregato, non "sembra che abbia imparato".

I numeri da battere sono stati rimisurati il 2026-08-01 con `n_storie=200`
(verificati stabili a 400) e sono questi:

### P1 — permanenza, canale `("visione",)`, viste `("cucina", "salotto")`

| fascia d'età | n | `regole/` |
|---|---|---|
| 0 | 1304 | 0,7945 |
| 1-2 | 49 | 0,0204 |
| 3-5 | 78 | 0,0513 |
| 6-10 | 98 | 0,0714 |
| 11+ | 47 | 0,0851 |
| **età ≥ 1 (aggregato)** | **272** | **0,0588** |
| mai vista | 1403 | 0,0000 |

**Il bersaglio è l'aggregato "età ≥ 1": 0,0588.**

Due esclusioni, motivate:

- **"età 0" non è un bersaglio.** Con la telecamera che inquadra l'entità
  nell'istante stesso della domanda non si misura permanenza, si misura il
  sensore. `regole/` fa 0,79 e il margine che resta è rumore di binding.
- **"mai vista" non è un bersaglio.** È lo 0,0000 di `regole/` su 1403 casi,
  ed è **strutturale**: con due camere su sei, un'entità che nessuna telecamera
  ha mai inquadrato non ha uno slot, l'allineamento non le assegna niente e la
  risposta è per forza `non lo so`. Nessuna architettura di M1 può migliorarlo
  restando sul solo canale visivo. Va **riportato**, mai usato come bersaglio.

Nota di onestà sul campione: a `n_storie=20` (il default delle sonde) la fascia
"età ≥ 1" ha solo 30 casi e dà 0,100 — il numero che `FASE_MENTE.md` §12.2
riporta come "~0,10". È rumore di campione piccolo. **Il numero giusto è
0,0588 su 272 casi**, e tutte le misure di M1 si fanno a `n_storie=200`
(7,5 s in locale) o più. Aggiornare §12.2 quando M1 chiude.

### P7 — sorpresa, `n_storie=200`

| canale | sottoinsieme | n | `regole/` |
|---|---|---|---|
| `("visione",)` | solo sorpresa | 11480 | **0,0083** |
| `("visione","lingua")` | solo sorpresa | 11480 | 0,0505 |
| entrambi | copia frame | 11480 | 0,0000 |

**Il bersaglio di M1 è 0,0083**, cioè la riga sola visione: M1 è lo stadio
"sola percezione" (`FASE_MENTE.md` §7) e `mente/` non avrà ancora il canale
linguistico. Confrontarsi con 0,0505 sarebbe confrontare canali diversi, cioè
l'errore che §12.1 documenta.

**La baseline "copia frame" è uno zero per costruzione, non una misura.**
`banco._misura_predizione` definisce il sottoinsieme sorpresa come
`reale △ precedente`; la baseline predice `precedente`; su una differenza
simmetrica la condizione `(c in prima) == (c in reale)` è falsa per ogni `c`,
sempre. Quindi lo 0,0000 è una tautologia. `FASE_MENTE.md` §6 la presenta come
"il modello che non la batte non ha imparato niente": è vero al contrario —
**batterla non significa niente**. L'unico confronto che conta è con `regole/`.
Non modificare la sonda per aggiustare questo: annotarlo e basta (vedi §9.a).

### Soglie di accettazione

| esito | condizione |
|---|---|
| **fallimento** | P1(età≥1) ≤ 0,0588 **oppure** P7(sorpresa, visione) ≤ 0,0083 |
| **successo debole** | entrambi superati oltre l'intervallo di confidenza (§8.3) |
| **successo forte** | P1(età≥1) ≥ 0,20 **e** P7(sorpresa, visione) ≥ 0,10 |

In caso di fallimento: **fermarsi e riferire.** Non aggiungere capacità al
modello, non cambiare le sonde, non allargare il budget di training. Questo
progetto ha già pagato due volte il prezzo di epicicli aggiunti a un
esperimento che non funzionava. Il fallimento di M1 è un risultato: significa
che su questo micro-mondo la permanenza appresa non batte quella cablata, ed è
esattamente la cosa che non si sa.

### Perché il bersaglio è raggiungibile

`regole/` ha già la persistenza cablata (senza evidenza non cambia nulla) e
ciò nonostante fa 0,0588. Il motivo è che **non ha modello del moto**: quando
un oggetto esce dalle due stanze inquadrate, `regole/` continua a crederlo
nell'ultima stanza vista, mentre nel frattempo qualcuno lo ha portato altrove.
Per fare meglio serve sapere che le persone si spostano, che chi tiene un
oggetto se lo porta dietro, che un oggetto in un contenitore segue il
contenitore. È conoscenza statistica sul mondo, imparabile dalla sola loss
predittiva, e non è cablata da nessuna parte. È il vuoto che M1 deve riempire.

---

## 1. Che cosa esiste già, e va rispettato

`mente/` deve essere intercambiabile con `regole/` dentro il banco di misura.
L'interfaccia è definita da `regole/tracker.py`; questi sono i membri che il
banco e le sonde toccano davvero (verificato leggendo `sonde/banco.py`):

| membro | chi lo usa | contratto |
|---|---|---|
| `osserva(Osservazione)` | `banco.costruisci_tracker:191` | evidenza visiva su una vista |
| `ascolta_evento(...)` | `banco.costruisci_tracker:194` | evidenza linguistica — **M2, non M1** |
| `dove(slot) -> Risposta` | `banco.valuta_storia:269` | posizione |
| `chi_ha(slot) -> Risposta` | `banco.valuta_storia:269` | possesso |
| `predici(vista) -> tuple[str,...]` | `banco._misura_predizione:219` | classi attese, ordinate |
| `.slot[i]` | `banco.valuta_storia:268` | indicizzabile per id |
| `._assorbite` | `regole/allineamento.py:25` | `list[(slot_id, id_istanza_reale)]`, **solo diagnostica** |
| `._per_lingua` | `regole/allineamento.py:45` | `dict[id_istanza -> slot_id]`, vuoto in M1 |

`Risposta` (`regole/tracker.py:345`) è `(valore: str, confidenza: float)`, con
`NON_LO_SO = "non lo so"` e `NESSUNO = "nessuno"`. Riusare quelle costanti,
non ridefinirle.

Il lessico è **chiuso e deterministico**: 6 luoghi (`dm.LUOGHI`), 6 persone
(`dm.PERSONE`), 3 unità da risorsa (`mela`, `acqua`, `legna`), 7 oggetti unici
(`pane`, `palla`, `cestino`, `scatola`, `secchio`, `libro`, `camino`), più
`CLASSE_PERSONA_GENERICA = "persona"`. **17 classi in totale.** Attenzione:
un campione di 60 storie ne mostra solo 15 (`acqua` e `persona` non compaiono).
Costruire il vocabolario dai dati porterebbe a un crash quando la classe
mancante arriva: **si costruisce da `dm`, non dalle osservazioni.**

---

## 2. Vincoli specifici di M1

Oltre alle sei regole non negoziabili di `CLAUDE.md`:

1. **`mente/` non importa `mondo/`.** Mai. Nemmeno per il vocabolario: le
   costanti si ricopiano in `mente/vocabolario.py` con un test che verifica
   che coincidano con `dm` (il test sta in `tests/`, che può importare
   entrambi). È la cucitura sim-to-real di `FASE_MENTE.md` §3.
2. **`Osservazione.verita` non deve influenzare nessuna decisione in
   valutazione.** Serve al teacher forcing in training (§5) ed è registrata in
   `_assorbite` per l'allineamento post-hoc. Guardia obbligatoria: test
   analogo a `tests/test_regole.py::test_verita_non_influenza`.
3. **Determinismo**: `torch.manual_seed` esplicito, `torch.use_deterministic_algorithms(True)`,
   nessun `random` globale. Stesso seed → stessi pesi → stessi numeri.
4. **Separazione dei seed**:
   - training: `[10_000, 500_000)`
   - dev / sonde: `2000 … 2000+n_storie` (gli stessi di M0, per confrontabilità)
   - esame: `≥ 1_000_000` — **mai toccati in M1**
5. **Locale finché sta sotto i 5 minuti.** Tutte le misure di questo piano
   girano in secondi; se un training supera i 5 minuti, si va su Colab
   (commit + push + notebook), non si aspetta in locale.

---

## 3. Architettura di `mente/`

Sei file. Nessuno supera le ~200 righe.

```
mente/vocabolario.py   classi, luoghi, indici. Nessun torch.
mente/stato.py         StatoMente (tensori) + risoluzione della catena
mente/aggiorna.py      MLP_delta, MLP_gate — la persistenza
mente/associa.py       associazione dati: oracolo (M1a) e Sinkhorn (M1b)
mente/mente.py         classe Mente: l'interfaccia di §1
mente/addestra.py      loss predittiva, ciclo di training, checkpoint
```

### 3.1 `StatoMente` (`mente/stato.py`)

`K = 32`, `d = 64`, `C = 17` classi, `L = 6` luoghi.

| campo | forma | significato |
|---|---|---|
| `attivo` | `(K,)` | occupazione morbida, in `[0,1]` |
| `tipo` | `(K, C)` | logit sulla classe |
| `rel` | `(K, L + K + 1)` | logit sul bersaglio: luoghi, slot, `ignoto` |
| `identita` | `(K, d)` | tratti che distinguono due mele fra loro |
| `eta` | `(K,)` | tick dall'ultima evidenza diretta |

L'ultima colonna di `rel` è `ignoto`. La softmax su `rel` dà l'esclusività
cablata: un individuo sta in un posto solo (`FASE_MENTE.md` §5.1).

### 3.2 Risoluzione della catena (`mente/stato.py`)

Da `P = softmax(rel)` si estraggono `P_luogo (K,L)`, `P_slot (K,K)`,
`p_ignoto (K,)`. Poi il punto fisso di `FASE_MENTE.md` §5.2:

```
L_0     = P_luogo
L_{n+1} = P_luogo + P_slot @ L_n         (4 iterazioni)
```

`L[k, v]` = probabilità che lo slot `k` si trovi fisicamente nel luogo `v`.
Differenziabile, nessun ciclo Python sugli slot.

Per `chi_ha` serve la stessa risalita fermata sulle persone:

```
H_0     = P_slot * m_persona            (m_persona: maschera (K,) sugli slot di classe persona)
H_{n+1} = H_0 + (P_slot * (1 - m_persona)) @ H_n
```

`H[k, j]` = probabilità che lo slot `k` sia tenuto (anche indirettamente) dallo
slot-persona `j`. Se la massa su tutte le persone è sotto soglia, la risposta è
`NESSUNO`.

Aciclicità: penalità morbida `λ_ciclo * (tr(P_slot) + Σ_{i≠j} P_slot[i,j]·P_slot[j,i])`,
con `λ_ciclo = 0.1`. Non è un vincolo duro: serve solo a scoraggiare "A è dentro
B che è dentro A".

### 3.3 Persistenza (`mente/aggiorna.py`) — il punto centrale

```
delta = MLP_delta(evidenza, stato_slot)          # -> (L + K + 1)
gate  = sigmoid(MLP_gate(tipo_evidenza, confidenza, stato_slot))   # -> scalare
rel  <- rel + gate * delta
```

Entrambi gli MLP: due strati, larghezza 64, `tanh`. `tipo_evidenza` è un
one-hot su `{visione, assenza, lingua, inferenza}` (in M1 si usano solo i primi
due, ma il campo esiste già per non cambiare forma a M2).

**Invariante non negoziabile, protetta da test**: se in un tick uno slot non
riceve nessuna evidenza, `rel` di quello slot resta **byte-identico**. Nessun
decadimento, nessuna normalizzazione globale che lo tocchi di striscio. È
la permanenza degli oggetti resa architetturale; `jepa/modello_jepa.py:138`
faceva `rel = deltas` ed è il bug che ha ucciso l'esperimento precedente.

`eta` si incrementa di 1 per gli slot senza evidenza e si azzera per gli altri.
`eta` **non** entra in `rel`: entra solo nella confidenza riportata.

### 3.4 Associazione dati (`mente/associa.py`)

Il punteggio fra la rilevazione `m` e lo slot `k` è un MLP su questi tratti:

| tratto | forma |
|---|---|
| compatibilità di classe: `softmax(tipo[k])[classe(m)]` | 1 |
| compatibilità spaziale: `L[k, vista]` | 1 |
| recency: `1/(1+eta[k])` | 1 |
| occupazione: `attivo[k]` | 1 |
| similarità d'identità: `cos(identita[k], emb(classe(m)))` | 1 |
| confidenza della rilevazione | 1 |

Più una colonna di scarto (entità nuova) con punteggio da un parametro appreso.

**M1a — associazione forzata.** In training l'assegnamento viene dall'oracolo:
`Osservazione.verita[i]` dà l'id d'istanza reale, che si mappa allo slot già
assegnato a quell'istanza (o a uno nuovo). L'assegnamento dell'oracolo è
**anche il bersaglio** di una cross-entropy sulla matrice di punteggio, così lo
scorer impara mentre il resto del modello riceve evidenza pulita.

In **valutazione** l'oracolo non esiste: si usa `argmax` greedy sul punteggio,
in ordine di confidenza decrescente, con il vincolo di esclusività
(uno slot per rilevazione dentro lo stesso tick, e nessuno slot che risulti già
visto altrove in questo stesso istante — è `regole/tracker.py:176` riscritto in
tensori). Sopra la soglia di scarto si alloca uno slot nuovo.

**M1b — Sinkhorn.** Sostituisce l'argmax greedy in valutazione. Vale la
correzione del dustbin di `FASE_MENTE.md` §5.4: **riga e colonna di scarto con
marginale rilassato** (colonna di capacità `M`, riga di capacità `K`), non una
Sinkhorn doppiamente stocastica ingenua. 4 iterazioni.

M1b si affronta **solo se M1a ha superato le soglie di §0.** Se M1a fallisce,
Sinkhorn non lo salva: il problema sarebbe altrove.

### 3.5 Interrogazione (`mente/mente.py`)

- `dove(slot)`: `argmax_v L[k, v]`; confidenza = quel massimo.
- `chi_ha(slot)`: `argmax_j H[k, j]` fra gli slot-persona; `NESSUNO` se la
  massa totale sulle persone è < 0,5; confidenza = quel massimo.
- `predici(vista)`: le classi `c` per cui
  `P(c presente in vista) = 1 - Π_k (1 - L[k,vista] · attivo[k] · softmax(tipo[k])[c])`
  supera 0,5. Ritorna una tupla **ordinata**, come `regole/tracker.py:342`.

**Astensione: in M1 la soglia è 0, cioè non ci si astiene mai.** Motivo: la
scala `esatto_reale` conta l'astensione come errore, quindi astenersi
peggiorerebbe P1 e renderebbe il confronto con `regole/` illeggibile. La curva
di astensione è la sonda P4 e si guarda dopo, separatamente, senza toccare
questi numeri.

---

## 4. T0 — Generalizzare il banco (si fa per primo)

Oggi `sonde/banco.py:160` istanzia `TrackerRegole` a mano: le sonde non possono
girare su `mente/`. Senza questo passo l'agente che esegue finirà per
duplicare il banco, e i due sistemi non staranno più sulla stessa scala — cioè
l'unica cosa che questo progetto ha imparato a non fare.

1. Aggiungere `fabbrica: Callable[[], Credenza] | None = None` a
   `costruisci_tracker`, `valuta_storia`, `valuta_campione`, e passarlo
   attraverso. `None` ⟹ `TrackerRegole` con i parametri di oggi.
2. Definire `Credenza` come `typing.Protocol` in `sonde/credenza.py`, con la
   tabella di §1. Nessun import di torch in `sonde/`.
3. Propagare `fabbrica` alle sette sonde di `sonde/sonde.py`.
4. `sonde/esegui.py`: aggiungere `--sistema {regole,mente}` (default `regole`)
   e `--checkpoint PATH` (obbligatorio con `--sistema mente`).
5. **`sonde/esegui.py:100` stampa P7 solo su visione+lingua**, perché chiama
   `p7_sorpresa(seed_base, n_storie)` con il default `CANALI_TUTTI`. La riga
   che serve a M1 — sola visione — oggi non compare nell'output del comando.
   Stamparle **entrambe**, etichettate per canale, e non sostituire quella
   esistente: 0,0505 è il numero storico di M0 e va conservato.

**Test di accettazione di T0**: con `fabbrica=None` i numeri di M0 restano
identici cifra per cifra. Conviene congelarli in un test parametrico prima di
toccare il file.

---

## 5. Addestramento (`mente/addestra.py`)

**Obiettivo principale — predittivo, auto-supervisionato** (`FASE_MENTE.md` §6):
dalla credenza al tick `t`, predire quali classi comparirebbero nella vista `V`
al tick `t+1`. Bernoulli per classe, BCE contro l'insieme davvero osservato.

L'ordine dentro un tick è quello che il banco usa già
(`banco.costruisci_tracker:177`): **prima si predicono tutte le viste del tick,
poi si guarda.** Le telecamere sono simultanee; se si alternasse, la predizione
della cucina userebbe ciò che si è appena visto in salotto.

**Loss totale:**

```
L = BCE_predizione
  + λ_assoc · CE_associazione        (solo M1a, λ_assoc = 1.0)
  + λ_ciclo · penalità_aciclicità    (λ_ciclo = 0.1)
  + λ_occ   · |Σ attivo - n_atteso|  (λ_occ = 0.01)
```

**Ausiliaria di stato vero: NON si usa in M1.** `FASE_MENTE.md` §6 la
ammette "solo nello stadio iniziale del curriculum". Tenerla spenta rende il
risultato interpretabile: se P1 migliora, è la loss predittiva che ha
funzionato. Se M1a fallisce **entrambe** le soglie, accenderla è la prima cosa
da provare — ma è un esperimento nuovo da riferire, non una modifica silenziosa.

**Configurazione di partenza** (`configs/m1a.yaml`):

| iperparametro | valore |
|---|---|
| storie di training | 2000, seed `[10_000, 12_000)` |
| lunghezza | `banco.lunghezza` (8-22 tick) |
| percezione | `ConfigPercezione()` pulita, tutte le viste |
| epoche | 20 |
| ottimizzatore | Adam, lr `3e-4` |
| batch | 8 storie |
| `torch.manual_seed` | 1 |

Le storie di training usano **tutte** le viste, non le due di P1: il modello
deve imparare la dinamica del mondo, e P1 misura poi se quella dinamica regge
sotto copertura parziale. È la generalizzazione che si vuole verificare, non
un adattamento alla condizione di test.

**Budget**: se un'epoca supera i 15 secondi in locale, fermarsi e riferire il
tempo prima di lanciare le 20 (§9.c).

---

## 6. Ordine dei task

Ogni task chiude con i suoi test verdi e un commit. Suite a 605 test verde
all'inizio: non deve mai scendere.

| # | task | test di accettazione |
|---|---|---|
| **T0** | banco parametrico (§4) | numeri di M0 identici con `fabbrica=None` |
| **T1** | `mente/vocabolario.py` | 17 classi, 6 luoghi; test che coincidano con `dm` |
| **T2** | `StatoMente` + catena (§3.1-3.2) | catena a profondità 1/2/3 su stati costruiti a mano; `dove` e `chi_ha` corretti su un caso noto |
| **T3** | `aggiorna.py` (§3.3) | **slot senza evidenza ⟹ `rel` byte-identico**; `eta` cresce solo dove deve |
| **T4** | `associa.py` — oracolo + argmax greedy | esclusività: due rilevazioni della stessa classe in due viste allo stesso tick non finiscono nello stesso slot |
| **T5** | `mente/mente.py` | implementa il Protocol `Credenza`; `verita` non influenza le decisioni; `ascolta_evento` alza `NotImplementedError` |
| **T6** | `addestra.py` + `configs/m1a.yaml` | 2 epoche su 20 storie girano; stesso seed ⟹ stessa loss cifra per cifra |
| **T7** | **misura M1a** (§8) | tabella di confronto contro `regole/` |
| **T8** | M1b: Sinkhorn con dustbin corretto | `n` classi mai viste su memoria vuota ⟹ `n` slot distinti, per n = 1, 2, 3 |
| **T9** | **misura M1b**, stesso protocollo di T7 | tabella aggiornata |

T8-T9 **solo se T7 supera le soglie di §0.**

---

## 7. Che cosa M1 NON fa

Elencato perché non ricompaia per iniziativa di chi esegue:

- **Niente canale linguistico.** È M2. `ascolta_evento` alza
  `NotImplementedError` e i confronti si fanno su `canali=("visione",)`.
- **Niente generazione.** Decisione portante 2: la lingua è solo ingresso.
- **Niente esame ufficiale.** I seed `≥ 1_000_000` non si toccano: è M4.
- **Niente `riquadro` né `id_traccia`.** Si resta sul caso duro, solo classi
  (decisione portante 1). `riquadro` è per di più uno stub.
- **Niente memoria dormiente, niente slot #0, niente teoria della mente.**
  Sono in `FASE_MENTE.md` §13 come lavoro futuro, con il motivo.
- **Niente modifiche a `mondo/`, `lingua/`, `cervello/`, `esami/`, `jepa/`.**
  Se sembrano necessarie, è §9.

---

## 8. Protocollo di misura (T7 e T9)

### 8.1 I comandi, esattamente questi

```bash
.venv/bin/python -m sonde.esegui --sistema regole --storie 200
.venv/bin/python -m sonde.esegui --sistema mente --storie 200 \
    --checkpoint dati/m1a/modello.pt
```

Non esiste (né va aggiunto) un `--canali` globale: il canale non è una scelta
dell'utente ma parte della definizione di ciascuna sonda. `p1_permanenza`
(`sonde/sonde.py:73`) fissa da sé `canali=("visione",)`, perché la lingua
rinfrescherebbe le credenze scavalcando la prova. P7 è l'unica sonda che
prende il canale come parametro, e dopo il punto 5 di T0 le stampa entrambe.

`--storie 200`, non il default 20: a 20 la fascia "età ≥ 1" di P1 ha 30 casi e
il numero non significa niente (§0). Seed base 2000, invariato.

### 8.2 La tabella da produrre

Una riga per sonda, `regole/` e `mente/` affiancati, con `n`. Mai un numero di
`mente/` senza il suo accanto — è la regola 5 di `CLAUDE.md`.

Obbligatorie: P1 (per fascia **e** aggregato età≥1), P7 (sola visione).
Riportare anche P2, P3, P5, P6 per verificare che non siano regredite: un
guadagno su P1 pagato con un crollo del binding non è un guadagno.

### 8.3 Quando un margine è reale

Con `n = 272` e `p ≈ 0,06` l'errore standard è ~0,014: su P1 serve almeno
**0,09** per parlare di differenza, e sotto 0,12 resta debole. Con
`n = 11480` e `p ≈ 0,008` su P7 l'errore standard è ~0,0008: lì basta
**0,012**, ma un margine così stretto è ugualmente poco interessante — vedere
la soglia di successo forte in §0.

Riportare sempre `n` accanto a ogni numero. Un valore senza `n` non si valuta.

---

## 9. Punti su cui fermarsi e chiedere ad Andrea

Solo questi. Su tutto il resto il piano è la decisione.

- **a.** Se emerge che una sonda è definita male (come la baseline "copia
  frame" di §0, che è uno zero per costruzione): **annotarlo e riferire, non
  modificarla.** Cambiare lo strumento di misura a metà esperimento rende
  incomparabili tutti i numeri precedenti.
- **b.** Se M1a fallisce le soglie di §0. Fermarsi e riferire con i numeri.
  Non accendere la supervisione ausiliaria, non allargare il budget, non
  aggiungere capacità di propria iniziativa.
- **c.** Se un'epoca di training supera i 15 secondi, o un training supera i
  5 minuti: riferire il tempo prima di procedere, perché cambia dove si esegue
  (`CLAUDE.md`, "Dove si esegue").
- **d.** Se serve toccare `mondo/`, `lingua/`, o le firme esistenti di
  `regole/`. `mente/` deve adattarsi all'interfaccia che c'è, non il contrario.
- **e.** Se `mente/` non riesce a soddisfare il Protocol `Credenza` senza
  cambiare `sonde/banco.py` oltre a quanto previsto da T0.

---

## 10. Quando M1 è chiuso

1. Aggiornare `FASE_MENTE.md` §12 con la tabella di §8.2 e correggere il
   "~0,10" di §12.2 in 0,0588 su 272 casi, spiegando il perché.
2. Scrivere in §12.4 che cosa resta scoperto **anche se M1 è andato bene**.
3. Se M1 ha successo, il passo dopo è M2 (canale linguistico), non
   l'ottimizzazione di M1.
