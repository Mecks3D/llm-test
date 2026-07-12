# PIANO ESECUTIVO — Fase B: supervisione densa in-sequenza dello stato del mondo (run `v1_stato`)

Piano esecutivo nello stile di `FASE2_PIANO_ANTISCORCIATOIA.md`: decisioni
vincolanti, tappe T1–T7 con cancelli, pensato per un esecutore in
conversazione pulita. Prerequisiti di lettura: `PROGETTO.md`, `CLAUDE.md`,
`fasi/FASE2.md`, `fasi/FASE2_PIANO.md` (§1, §4, §7),
`fasi/FASE2_PIANO_DIAGNOSI.md` (§3 definisce la Fase B e §6-§7 l'albero
decisionale) e `fasi/FASE2_PIANO_TEMPO.md` §8 (la diagnosi che orienta il
formato dello stato: vedi §0 qui sotto).

---

## 0. Contesto e obiettivo

La roadmap di diagnosi (`FASE2_PIANO_DIAGNOSI.md` §6-§7) è arrivata a questa
lettura, consolidata dall'esperimento "tempo":

- Il modello **non** ha un problema di capacità/step: con un solo personaggio
  e budget normale traccia lo stato nel tempo ben oltre le baseline
  (`tracking_tempo` 0,646 con euristiche 0,0 per costruzione). Quindi **A2
  (ablation) perde priorità**.
- Il muro del multi-personaggio è **interferenza/binding tra entità**, non
  assenza della capacità di base.
- Il collo fine, isolato dall'anatomia degli errori dei tipi tempo, è la
  **precisione dell'indicizzazione temporale**: il modello localizza il
  quartiere giusto della storia ma non il tick esatto (puntatore sfocato
  ±1-2 tick). Non è la propagazione dello stato in sé, né la complessità
  della risposta.

La Fase B attacca entrambe le cose con l'unica leva-dati non ancora provata:
**dentro la stessa sequenza**, a ogni fine tick il modello emette il grafo
dello stato corrente (dove si trova ciascuna persona), poi la storia
prosegue; la domanda resta DOPO la storia, come sempre. Costringe a
*mantenere* lo stato di tutte le entità lungo tutto il contesto invece di
ricalcolarlo una volta sola alla fine (interferenza), e l'etichetta di tick
esplicita in ogni blocco dà un indice temporale discreto e leggibile
(puntatore sfocato). È la realizzazione letterale del "grafo-pensiero" di
`PROGETTO.md`: il modello simula il mondo mentre legge, e rispondere diventa
leggere il proprio stato.

**Attenzione alla sovrapposizione col passato** (`FASE2_PIANO_DIAGNOSI.md`
§3): le storie troncate di `v1_anti` erano già "domande a istanti intermedi",
ma come **record separati** — e non sono bastate (0,513). La differenza qui è
che lo stato è **in-sequenza**, prima della domanda: cambia cosa il modello è
costretto a computare durante la lettura, non solo cosa gli si chiede dopo.

**Regola aurea** (informativo in entrambi gli esiti):

- **Se funziona** (esame ≥ 0,80 e tracking-split su, blocchi-stato accurati)
  → la supervisione densa rompe l'interferenza; si consolida e si torna al
  curriculum portando la ricetta. Prossimo dubbio eventuale: C2 (slot entità).
- **Se non basta** (esame invariato ~0,57, o stato accurato ma risposta no)
  → la leva-dati da sola non basta e serve una leva architetturale →
  `FASE2_PIANO_DIAGNOSI.md` §4: C1 (ricorrenza), poi C2 (slot entità).

**Confronto storico di riferimento** (stessa distribuzione ufficiale stadio 1,
cast pieno): `v1` 0,573; `v1_anti` 0,513; `v1_facile` 0,745; `v1_grad2` 0,904
(ma cast/ storie facili). Il termine di paragone diretto per `v1_stato` è
**`v1`: 0,573**, perché condivide la stessa distribuzione ufficiale e cambia
un solo fattore (la supervisione densa dello stato).

---

## 1. Decisioni vincolanti (Andrea, 2026-07-12 — non riaprirle)

Prese in questa sessione (i tre bivi di design elencati in
`FASE2_PIANO_DIAGNOSI.md` §3):

1. **All'esame i blocchi `[STATO]` li GENERA il modello** (autoregressivi),
   non sono dati in input. È il punto dell'esperimento: il modello "pensa ad
   alta voce" in grafi, ogni blocco è verificabile tick-per-tick contro il
   simulatore. Costo accettato: decodifica più lunga, errori che si possono
   propagare. Darli in input sarebbe leak della risposta → vietato.
2. **Contenuto del blocco stato = solo le posizioni delle persone** a fine
   tick (`trovarsi` per ciascuna persona del cast). Niente oggetti
   trasportati per lo stadio 1: bastano le posizioni per tutte le domande
   attuali e i blocchi restano corti (budget di contesto sotto controllo).
   Gli oggetti si aggiungeranno se/quando servirà lo stadio 3.
3. **Ogni blocco `[STATO]` porta un'etichetta di tick esplicita** (es.
   `( tick nove )` come primo elemento del blocco). Dà al modello un indice
   temporale discreto: è l'intervento diretto sul "puntatore temporale
   sfocato" isolato dalla diagnosi tempo (`FASE2_PIANO_TEMPO.md` §8).

Ereditate (vincoli permanenti della roadmap, non rimetterle in discussione):

4. **Esperimento a parte**: run `v1_stato`, config e `dati/stato/` propri. Il
   curriculum ufficiale, i dev/esame ufficiali e i config esistenti **non
   cambiano** (vincolo permanente 2 di `FASE2_PIANO_DIAGNOSI.md`). Il formato
   normativo di `FASE2.md` (`componi_esempio` senza stato) resta il default
   byte-identico; lo stato è una variante attivata dal config.
5. **Da zero**: pesi casuali, nessun `--pesi-iniziali` (attribuzione pulita;
   i checkpoint esistenti portano dentro la scorciatoia frequenza/recency).
6. **Stessa distribuzione ufficiale stadio 1, cast pieno**: solo il tipo
   `posizione`, dev/esame nella distribuzione ufficiale INVARIATA (senza
   blocchi stato — vedi §5), così il confronto con `v1` (0,573) è diretto.
   L'unica variabile nuova è la supervisione densa nel train.
7. **A2 (ablation capacità/budget) resta fuori**: non si lancia in parallelo.
8. **Modello IDENTICO** a `configs/v1.yaml` (n_layer 8, n_head 8, d_model
   256, d_ff 1024): la Fase B è una leva-dati/formato, non architetturale.
   Le leve architetturali sono la Fase C, che viene solo dopo (§9).

Decisione di progetto dichiarata qui, vincolante per l'esecutore:

9. **Semantica di "stato a fine tick"**: identica al piano tempo
   (`FASE2_PIANO_TEMPO.md` §1.7) — lo stato al termine del tick N è la
   posizione di ciascuna persona DOPO il suo evento del tick N. È già
   verificato empiricamente lì (0 mismatch con `stato_finale.luogo_effettivo`
   della storia troncata) e va **riusato**, non re-derivato.

---

## 2. Formato della sequenza — `cervello/sequenza.py`

Nuovo token speciale `STATO` in `cervello/vocabolario.py::TOKEN_SPECIALI`
(in coda, così gli id esistenti non si spostano — verificare che
`vocabolario.json` si rigeneri e i test di round-trip esistenti restino
verdi). La sequenza con stato interlacciato:

```
[STORIA]
  ( andare ( nsubj sara ) ( obl:origine cucina ) ( obl:luogo giardino ) ( obl:tempo nove ) )
  ( ...eventuali altri eventi del tick nove... )
  [STATO] ( tick nove )
          ( trovarsi ( nsubj sara ) ( obl:luogo giardino ) )
          ( trovarsi ( nsubj anna ) ( obl:luogo camera ) )
          ...una `trovarsi` per ogni persona del cast, ordine deterministico...
  ( ...eventi del tick dieci... )
  [STATO] ( tick dieci ) ...
[DOMANDA] ( trovarsi ( nsubj mela primo ) ( quesito dove ) )
[RISPOSTA] ( essere ( nsubj mela primo ) ( obl:luogo cucina ) ) [FINE]
```

Regole normative:

- **Raggruppamento per tick**: gli eventi si emettono nell'ordine della
  storia; un blocco `[STATO]` si inserisce **dopo l'ultimo evento di ogni
  tick** in cui è successo qualcosa. Il confine di tick e l'ordinale si
  ricavano dal simulatore (`mondo/`), non si indovinano dall'ordine — T1
  verifica come esporre l'ordinale (è già disponibile: le domande tempo lo
  usano). Un tick senza eventi non genera blocco (non c'è nulla da fine-tick).
- **Etichetta di tick**: primo elemento del blocco, `( tick <ordinale> )`,
  con l'ordinale come lemma-numero già nel vocabolario (gli stessi token dei
  `obl:tempo`). NON un token nuovo: riusa i numeri esistenti.
- **`trovarsi` per persona**: una `trovarsi(nsubj=persona, obl:luogo=luogo)`
  per ciascuna persona del cast, in **ordine deterministico** (l'ordine del
  cast del config, non l'ordine di menzione — così l'oro è unico e il
  round-trip stabile). Il verbo/relazioni riusano quelli già prodotti dalle
  domande di posizione (`trovarsi`/`obl:luogo`): niente lessico nuovo,
  niente stampi nuovi (a differenza del piano tempo, qui `lingua/` NON si
  tocca — verificare in T1 che tutti i token dello stato siano già nel
  vocabolario).
- **Nuova funzione** `componi_esempio_stato(...)` (o parametro `blocchi_stato`
  di `componi_esempio`): non tocca `componi_esempio` esistente (cancello
  byte-identico dei default). Round-trip: sequenza-con-stato → lista di grafi
  (eventi, blocchi stato con tick, domanda, risposta) → sequenza esatta.

---

## 3. Maschera di loss — `cervello/dati.py`

Oggi `_maschera_piena` (in `cervello/dati.py`) è vera solo dalle posizioni
dopo `[RISPOSTA]` fino a `[FINE]`: storia e domanda sono date, non si
imparano. Per la Fase B i **blocchi `[STATO]` vanno imparati** (è il punto:
il modello li deve generare), gli eventi della storia no (restano dati).

- Nuova maschera `_maschera_stato`: vera sulle posizioni **dentro ogni blocco
  `[STATO]`** (dal token successivo a `[STATO]` fino alla fine del blocco) E
  sulle posizioni della risposta (come oggi). Falsa sugli eventi della storia,
  sulla domanda, sui token `[STORIA]`/`[STATO]`/`[DOMANDA]`/`[RISPOSTA]`
  stessi (si predicono, non si imparano a emettere — coerente con la
  convenzione attuale che impara i contenuti, non i delimitatori: verificare
  la convenzione esatta di `_maschera_piena` in T1 e ricalcarla).
- Attivata solo quando l'esempio contiene blocchi stato (config `stato:
  true`); senza, `impacchetta_batch` usa la maschera piena di sempre
  (byte-identico). Test: un batch senza stato produce la stessa maschera di
  prima.
- La loss resta la cross-entropy mascherata di `addestra.py::_calcola_loss`
  invariata: cambia solo *dove* la maschera è vera.

---

## 4. Generazione dello stato-oro — `esami/genera.py` (+ `mondo/`)

Nel **train** i blocchi `[STATO]` sono verità del simulatore. Il generatore,
per ogni storia del train, ricava a ogni fine tick la posizione di ciascuna
persona e la linearizza col formato §2.

- **Fonte dello stato**: il simulatore conosce già la posizione di ogni
  persona a ogni tick (è ciò che `FASE2_PIANO_TEMPO.md` §1.7 ha verificato:
  la posizione a fine tick N = `stato_finale.luogo_effettivo` della storia
  troncata a N). Riusare quell'accesso; T1 individua il punto esatto in
  `mondo/` (probabile: rieseguire/ispezionare la traccia della storia). NON
  reimplementare la fisica: leggere lo stato dal motore.
- **Solo split train**: dev ed esame restano nella distribuzione ufficiale
  **senza** blocchi stato (vincolo 6). All'esame lo stato lo genera il modello
  (§5), non il generatore.
- **`esami/genera.py` si estende senza toccare l'esistente**: gate su un flag
  di config (`dataset.stato: true`); `genera_esame`/i generatori del tipo
  `posizione` restano byte-identici quando il flag è assente.

---

## 5. Esame: decodifica con stato generato dal modello — `esami/esamina.py`

Decisione 1: all'esame i blocchi `[STATO]` li genera il modello. L'esame NON
è più un singolo decode dopo `[RISPOSTA]`: è un **decode interlacciato** sulla
storia data.

Procedura normativa per un esempio d'esame:

1. Si parte da `[STORIA]` + eventi del **tick 1** (dati, teacher-forced).
2. Al confine di tick il modello **genera** in autoregressione il blocco
   `[STATO]` (da `[STATO]` fino alla `[CHIUSA]` che chiude l'ultima
   `trovarsi`, cioè finché non emette l'inizio del tick successivo o un token
   di controllo). Il blocco generato si **appende** al contesto (free-run).
3. Si appendono gli eventi del **tick successivo** (dati, teacher-forced).
4. Ripetere 2-3 fino a fine storia; poi `[DOMANDA]` (dato) e si genera la
   `[RISPOSTA]` come oggi (`esami.esamina.valuta_esempio`).

- **Metrica primaria (stadio 1)**: esattezza della risposta finale, grafo vs
  grafo, INVARIATA — direttamente confrontabile con `v1` 0,573.
- **Metrica ausiliaria (nuova, verificabile)**: accuratezza dei blocchi
  `[STATO]` generati contro la verità del simulatore, **per tick e per
  distanza dalla coda**. È la lente che dice *se* il modello sta davvero
  mantenendo lo stato e *dove* si sfoca (collega direttamente al collo
  "puntatore ±1-2 tick"). Va in `esami/diagnosi.py` (§6).
- **Rischio dichiarato** (accettato in decisione 1): un blocco stato sbagliato
  a un tick può inquinare i tick successivi. La metrica ausiliaria lo misura;
  se domina, è un input per la lettura §8 (leva-dati non basta → C1).
- Questo è il pezzo implementativo più delicato del piano: la decodifica
  interlacciata (teacher-force eventi + free-run stato) non esiste ancora.
  Isolarla in una funzione testabile (`valuta_esempio_stato`) e coprirla con
  un test deterministico su un modello giocattolo/seed fisso.

---

## 6. Nuovi artefatti

### 6.1 `configs/v1_stato.yaml`

Come `configs/v1.yaml` (modello e training IDENTICI) più il blocco `stato`:

```yaml
# Fase B — supervisione densa in-sequenza dello stato (vedi
# fasi/FASE2_PIANO_STATO.md). Cast pieno, da zero, dev/esame ufficiali
# INVARIATI (confronto diretto con v1: 0,573). Solo il train ha i blocchi
# [STATO]; all'esame li genera il modello.

nome_run: v1_stato
device: auto
seed_torch: 1337

percorsi:
  lessico: lingua/lessico.tsv
  vocabolario: cervello/vocabolario.json
  dati_dir: dati/stato
  risultati_dir: dati/risultati

dataset:
  ctx: 3072              # VERIFICARE le lunghezze reali con lo stato (T5)
  train_storie: 6000
  dev_storie: 300
  esame_storie: 300
  n_per_tipo: 8
  stato: true            # SOLO split train: blocchi [STATO] per-tick

stadi:
  1:
    tipi: [posizione]
    soglia: 0.95
    storie_corte: true

modello:                 # IDENTICO a configs/v1.yaml (~7,18M parametri)
  n_layer: 8
  n_head: 8
  d_model: 256
  d_ff: 1024
  dropout: 0.1

training:
  batch: 4
  accumulo: 8
  lr: 3.0e-4
  beta1: 0.9
  beta2: 0.95
  weight_decay: 0.1
  warmup_step: 200
  grad_clip: 1.0
  max_step: 8000
  intervallo_valutazione: 500
  dev_campione: 200
```

Più `configs/v1_stato_fumo.yaml` per il fumo su Colab: come sopra ma
`train_storie: 300`, `max_step: 600`, `intervallo_valutazione: 200`.

### 6.2 Diagnosi estesa — `esami/diagnosi.py`

Aggiungere una sezione "stato" (come già fatto per "tempo"): oltre alle
metriche esistenti sul tipo `posizione`, riportare l'**accuratezza dei
blocchi stato generati** per tick e per fascia di `distanza_coda` (0, 1-2,
3-5, ≥6) e l'anatomia degli errori di stato (persona giusta/luogo di un tick
vicino ±1-2 vs luogo di un'altra persona vs altro). È la verifica empirica
diretta dell'ipotesi §0 (puntatore sfocato + interferenza).

### 6.3 Notebook Colab

Nuova sezione nel notebook di training (come per gli altri run): fumo
`v1_stato_fumo`, poi run vero `v1_stato`, con la stampa della composizione del
train (proporzione eventi vs blocchi stato, lunghezze reali delle sequenze —
serve al cancello ctx di T5).

---

## 7. Test (in `tests/test_cervello.py` e `tests/test_esami.py`)

1. **Round-trip con stato** (`sequenza.py`): sequenza-con-stato → grafi →
   sequenza esatta, con ≥2 tick e cast ≥2.
2. **Byte-identità default** (`sequenza.py`, `dati.py`): senza `stato`, la
   sequenza e la maschera sono identiche a oggi.
3. **Maschera stato** (`dati.py`): vera sui contenuti dei blocchi stato + sui
   contenuti della risposta, falsa sugli eventi/domanda/delimitatori.
4. **Nessun token nuovo** (`vocabolario.py`): tutti i token di un blocco stato
   d'esempio sono già nel vocabolario; `STATO` è l'unico speciale aggiunto e
   in coda (id esistenti invariati).
5. **Generazione stato-oro** (`genera.py`): su una storia seed-fissa, i blocchi
   stato coincidono con lo stato del simulatore a ogni tick (riusa la verifica
   §1.9).
6. **Decodifica interlacciata** (`esamina.py`): `valuta_esempio_stato` su
   modello giocattolo/seed fisso — teacher-force eventi, free-run stato,
   forma della sequenza risultante corretta e deterministica.
7. Il gruppo di test esistente (round-trip senza stato, maschera piena,
   esame ufficiale) resta verde.

---

## 8. Tappe di lavoro (con cancelli)

- **T1 — formato sequenza + vocabolario** (`sequenza.py`, `vocabolario.py`):
  §2 + test 1, 2, 4. Individuare l'accesso allo stato per-tick in `mondo/` e
  come esporre l'ordinale di tick. Cancello: test verdi, round-trip esistente
  intatto, `vocabolario.json` rigenerato con id vecchi invariati.
- **T2 — maschera di loss** (`dati.py`): §3 + test 3. Cancello: test verdi,
  maschera default byte-identica.
- **T3 — generazione stato-oro** (`genera.py` + `mondo/`): §4 + test 5.
  Cancello: test verdi; stato-oro == simulatore su 300 storie campione.
- **T4 — decodifica interlacciata d'esame** (`esamina.py`): §5 + test 6.
  Cancello: test verdi; una corsa locale su CPU con un checkpoint casuale
  produce sequenze ben formate (non si valuta la qualità, solo la forma).
- **T5 — config + notebook + misura ctx + fumo**: §6. Cancello: fumo
  `v1_stato_fumo` su Colab OK (pipeline intera, composizione train sensata,
  **lunghezze reali < ctx con margine** — se sforano 3072, decidere con
  Andrea: alzare ctx o accorciare le storie), esattezza dev > 0 a fine fumo.
  Lo lancia Andrea.
- **T6 — run vero** (~1-1,5h su Colab, lo lancia Andrea): dataset pieno,
  training, esame ufficiale con decodifica interlacciata, `diagnosi.py` esteso
  su `stadio1.pt` E `stadio1_best.pt`. Consegna: numeri (esame + accuratezza
  stato per tick/distanza) + aggiornamento dello "Stato di avanzamento" di
  `FASE2_PIANO_DIAGNOSI.md` §7 e di questo piano.
- **T7 — lettura e verdetto**: §9, con Andrea.
- Un commit per tappa. Test/inferenza pesanti su Colab, mai in locale (memoria:
  commit+push → Colab, vale anche per inferenza/diagnosi >5 min). Prima di
  T5/T6, self-review del codice delle tappe precedenti (memoria: revisione
  dopo sessioni lunghe).

---

## 9. Criteri di lettura del risultato (T7)

Confronto diretto: **`v1` esame 0,573** (stessa distribuzione ufficiale). Sul
best tra `stadio1.pt` e `stadio1_best.pt`:

- **Successo pieno**: esame ≥ 0,95 → stadio 1 superato. Si prosegue col
  curriculum portando la ricetta dello stato denso (decidere con Andrea se
  gli stadi 2-3 la ereditano e se lo stato deve includere gli oggetti).
- **Progresso forte**: esame ≥ 0,80 E accuratezza dei blocchi stato alta e
  **piatta rispetto alla distanza dalla coda** (il modello mantiene davvero
  lo stato, il puntatore non si sfoca) → la leva funziona; proporre ad Andrea
  di consolidare o scalare. Se lo stato è accurato ma la risposta resta sotto,
  il collo è la *lettura* dello stato al momento della domanda, non il
  mantenimento → nota per C2 (indicizzazione per entità).
- **Progresso debole o nullo**: esame ~0,57 (invariato vs `v1`), oppure stato
  accurato ai primi tick ma che degrada con la distanza dalla coda (puntatore
  ancora sfocato nonostante l'indice esplicito) → la leva-dati/formato da sola
  non basta. Fermarsi e proporre ad Andrea la **Fase C**
  (`FASE2_PIANO_DIAGNOSI.md` §4): C1 (ibrido ricorrente, stato esplicito che
  si aggiorna tick-dopo-tick) prima, C2 (slot per entità) dopo.

---

## 10. Rimandato esplicitamente (non farlo ora)

- **Oggetti trasportati nel blocco stato**: solo se lo stadio 1 passa e serve
  per lo stadio 3 (decisione 2).
- **Stato denso su dev/esame ufficiali** o su altri tipi di domanda: no, il
  confronto con `v1` deve restare pulito (vincolo 6).
- **Leve architetturali (Fase C: C1 ricorrenza, C2 slot entità)**: solo dopo
  l'esito di questo esperimento, con decisione esplicita di Andrea (§9).
- **A2 (ablation capacità/budget)**: fuori (vincolo 7).
- **"[DOMANDA] prima della storia"** (riordino sequenza): resta la leva-formato
  rimandata da `FASE2_PIANO_ANTISCORCIATOIA.md` §9; non si combina con la Fase
  B nello stesso run per non confondere l'attribuzione.
