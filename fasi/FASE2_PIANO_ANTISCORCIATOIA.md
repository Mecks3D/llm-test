# PIANO ESECUTIVO — Esperimento anti-scorciatoia (stadio 1, cast pieno, run `v1_anti`)

Questo piano è pensato per essere eseguito **da un modello meno capace, in una
conversazione pulita**, come FASE1_PIANO.md e FASE2_PIANO.md. È normativo: le
decisioni segnate come vincolanti non si rimettono in discussione. Prerequisiti
di lettura: `PROGETTO.md`, `fasi/FASE2.md`, `fasi/FASE2_PIANO.md` (in
particolare §1 "Decisioni vincolanti", che restano TUTTE valide, e lo "Stato
di avanzamento"). In caso di conflitto tra questo piano e il codice, vince il
piano; se il piano sembra sbagliato, fermarsi e chiedere ad Andrea.

---

## 0. Contesto: perché questo esperimento

Lo stadio 1 del curriculum ha fallito l'esame due volte: **0,573** a cast
pieno (run `v1`, 20.000 step) e **0,7448** a cast ridotto di 3 personaggi
(run `v1_facile`, soglia 0,95). L'analisi qualitativa approfondita del
2026-07-08 (checkpoint `v1_facile` valutato in locale su tutti i 1504 esempi
d'esame rigenerati per seed, più storie-trabocchetto costruite a mano) ha
stabilito la diagnosi, che è la motivazione di ogni scelta qui sotto:

- Il modello **non fa binding entità→luogo**: fa un recupero associativo
  pesato su **frequenza e recency dei luoghi**. La regola simbolica "rispondi
  col luogo dell'ultimo evento che menziona il bersaglio" risolve il **93,8%**
  dell'esame (sottoinsieme con oro noto, n=1105); il modello lì fa il 65,7%.
- Esattezza condizionata: **79,3%** quando l'oro coincide col luogo più
  frequente della storia, **52,7%** quando no. **87,2%** se l'ultima menzione
  del bersaglio è a fine storia, **48,8%** se è nella prima metà.
- Effetto ordine nel cast: maria (i suoi eventi chiudono ogni tick) 91,8%;
  anna 59,7%; piero 58,6%. Bastano 1-2 eventi di interferenza dopo l'ultima
  menzione del bersaglio per degradare la risposta. Questo spiega anche il
  salto 0,573→0,745 col cast ridotto: meno entità = meno interferenza.
- **Esattezza sul train = 0,80** (storie viste ~16 epoche): il modello non
  fitta nemmeno il training set → NON è un gap di generalizzazione, è un
  **plateau di ottimizzazione** (shortcut learning: la scorciatoia paga su
  ~2/3 degli esempi di train e SGD non ne esce). La loss è già mascherata
  sulla sola risposta: non c'è diluizione del segnale.
- Cosa il modello HA imparato: grammatica dei grafi (0 malformate su 1504) e
  calibrazione non-lo-so quasi perfetta (396/399 astensioni corrette, 3
  invenzioni). Non vanno rotte.

**Idea dell'esperimento** (approvata da Andrea, 2026-07-08): rendere la
scorciatoia non redditizia nel training set, senza toccare né il simulatore
né il modello. Due leve insieme:

1. **Selezione anti-scorciatoia delle domande di train**: sovracampionare le
   domande dove la scorciatoia sbaglia (misurato: a cast pieno una storia di
   stadio 1 ha in mediana 7 domande di posizione "difficili" su ~14 candidate
   — basta selezionare, non serve generare nulla di nuovo).
2. **Storie troncate**: per ogni seed, record aggiuntivi con la stessa storia
   fermata a tick intermedi. VERIFICATO empiricamente (180/180 su cast pieno
   e ridotto): `genera_storia(seed, n_tick=k)` produce esattamente il
   prefisso della storia piena, con `stato_finale` coerente al tick k. La
   stessa entità ha così risposte diverse a istanti diversi della stessa
   storia: l'associazione statica storia→luogo smette di funzionare.

---

## 1. Decisioni vincolanti (già prese con Andrea — non rimetterle in discussione)

1. **Run nuova `v1_anti`, cast PIENO (6 personaggi), stadio 1, da pesi
   casuali (da zero).** Niente `--pesi-iniziali`: l'attribuzione deve essere
   pulita (se migliora, è merito dei dati) e i checkpoint esistenti hanno già
   consolidato la scorciatoia. Il confronto diretto è con lo 0,573 di `v1`.
2. **Budget alleggerito** (richiesta esplicita di Andrea: non ~3h):
   6.000 storie base, `max_step` 8.000 → **stima ~1-1,2h su Colab** (T7 di
   `v1`: 20.000 step ≈ 2,5-3h con eval inclusi, stesso formato di storie).
   La decisione 10 di FASE2_PIANO resta valida: prima del run vero si fa un
   fumo e si riporta la stima; il lancio lo fa comunque Andrea su Colab.
3. **Dev ed esame restano la distribuzione UFFICIALE, invariata**: selezione
   anti-scorciatoia e troncamenti si applicano **solo allo split `train`**.
   L'esame di `v1_anti` usa le stesse finestre di seed di `v1` (esame stadio
   1 = seed 1.000.000–1.000.299, 300 storie): il numero finale è direttamente
   confrontabile con 0,573.
4. **Comportamento di default invariato byte per byte**: le nuove chiavi di
   config sono opzionali; senza di esse `esami/genera.py` produce record
   IDENTICI a oggi (test obbligatorio, vedi §6). Le run `v1` e `v1_facile`
   devono restare riproducibili.
5. **`mondo/` e `lingua/` NON si toccano** (decisione 8 di FASE2_PIANO, qui
   senza alcuna eccezione: tutto il lavoro vive in `esami/`, `cervello/`,
   `configs/`, `tests/`, notebook). Se sembra servire una modifica lì,
   fermarsi e chiedere.
6. **Niente modifiche al modello o al formato sequenza**: la leva
   "[DOMANDA] prima della storia" è esplicitamente RIMANDATA (§9); si prova
   solo la leva dati. `cervello/sequenza.py`, `cervello/modello.py`,
   `cervello/dati.py` non si toccano.
7. **Best-dev checkpoint**: `cervello/addestra.py` salva anche il checkpoint
   col miglior `esattezza_dev` del run (già segnalato ad Andrea dopo T7: il
   picco 0,605 a step 18.000 era andato perso). Unica modifica ammessa ad
   `addestra.py` in questo piano.
8. **Determinismo**: come sempre, ogni RNG con seed esplicito derivato dal
   seed della storia; stesso config → stesso dataset byte per byte.

---

## 2. Selezione anti-scorciatoia — `esami/genera.py`

### 2.1 Proprietà misurabili di una domanda di posizione

Per una domanda di tipo `posizione` con oro noto (non `non-lo-so`), su una
storia con eventi `E = [e_0 … e_{n-1}]` (inclusi eventi di sistema come
`bruciare`):

- `luoghi(E)` = multinsieme dei `luogo` degli eventi (i `None` esclusi);
  `piu_frequente(E)` = il luogo col conteggio massimo; a parità, il primo
  incontrato scorrendo gli eventi in ordine (== `Counter.most_common`, che
  preserva l'ordine di inserimento).
- `um(bersaglio)` = indice dell'ULTIMO evento che "menziona" il bersaglio:
  - bersaglio persona → eventi con `agente == bersaglio` o
    `destinatario == bersaglio`;
  - bersaglio oggetto → eventi con `oggetto == bersaglio` o
    `argomento == bersaglio`, **escludendo** `azione == "cercare"` (cercare
    X dice che X NON è lì — stessa esclusione di
    `mondo/domande.py::_oggetti_con_posizione_nota`).
- `distanza_coda` = `n - 1 - um(bersaglio)` (eventi dopo l'ultima menzione).

Una domanda è **DIFFICILE** se vale almeno una di:

- **D1**: `oro != piu_frequente(E)` — sconfigge la scorciatoia di frequenza
  (il discriminatore più forte misurato: 79,3% vs 52,7%);
- **D2**: `distanza_coda >= 3` — sconfigge la scorciatoia di recency
  (misurato il degrado con l'interferenza; 3 = più di mezzo tick di cast
  pieno dopo l'ultima menzione);
- **D3**: `oro != E[um].luogo` — tracking puro: la risposta non è MAI
  co-menzionata col bersaglio (oggetto trasportato da chi lo tiene; oggi il
  modello fa il 45,1% su questi).

Se il bersaglio non è mai menzionato (`um` inesistente) la domanda ha oro
`non-lo-so` e non è né facile né difficile: è nel gruppo NLS.

### 2.2 Algoritmo di selezione (normativo)

In `genera_record`, quando il config ha `dataset.anti_scorciatoia` (solo
split train — vedi §4 per il plumbing):

1. Genera le candidate con `genera_domande(storia, rng_domande,
   n_per_tipo=candidate_per_tipo)` dove `candidate_per_tipo` viene dal
   config (default normativo: 999 = "tutte le entità"); filtra i tipi
   ammessi dello stadio come oggi.
2. Partiziona le candidate `posizione` in tre gruppi: `NLS` (oro non-lo-so),
   `DIFF` (difficili, §2.1), `FACILI` (le altre). Le proprietà si calcolano
   con gli eventi della storia del record (quella eventualmente troncata).
3. Componi `n_per_tipo` domande (default 8, come oggi):
   - `n_nls = min(round(0.2 * n_per_tipo), len(NLS))` — mantiene la quota di
     astensione ~15-20% di PROGETTO.md (la calibrazione non-lo-so oggi
     funziona e non va rotta);
   - `n_diff = min(round(quota_difficili * (n_per_tipo - n_nls)), len(DIFF))`
     con `quota_difficili` dal config;
   - riempi i posti restanti con `FACILI`, poi (se mancano) con altre `DIFF`,
     poi con altre `NLS`.
   Campionamenti e shuffle finale con un RNG dedicato
   `random.Random(f"anti-{seed}-{n_tick}")` (il `n_tick` distingue i record
   troncati dello stesso seed). L'ordine dei gruppi dentro il record va
   mescolato (mai NLS/DIFF/FACILI in blocchi riconoscibili).
4. Tipi diversi da `posizione` (stadi futuri): la selezione NON si applica,
   passthrough del comportamento attuale. Questo esperimento è stadio 1.

Nel record JSONL ogni esempio selezionato porta un campo in più
`"difficolta": "difficile" | "facile" | "non-lo-so"` (diagnostica; i
consumatori attuali — `cervello/dati.py`, `esami/esamina.py` — leggono solo
`storia`/`esempi`/`domanda`/`risposta` e lo ignorano, verificato).

## 3. Storie troncate — `esami/genera.py`

Quando il config ha `dataset.troncamenti: true` (solo split train): per ogni
seed, oltre al record della storia piena (`n_tick = n`), si emettono record
aggiuntivi per ogni `k` in `range(3, n)` (quindi n=3 → nessuno; n=6 → k=3,4,5;
a stadio 1 le storie sono corte, 3-6 tick, media +1,5 record per storia).

Regole normative:

- La storia troncata si ottiene con `genera_storia(seed=seed, n_tick=k,
  persone=...)` — MAI tagliando a mano la lista eventi: serve lo
  `stato_finale` coerente al tick k per l'oro delle domande (equivalenza
  prefisso verificata empiricamente 180/180, vedi §0).
- RNG delle domande del record troncato: `random.Random(f"domande-{seed}-t{k}")`
  (decorrelato dalla storia piena, che resta `f"domande-{seed}"`).
- Il record troncato porta il campo `"troncamento": k` (assente nel record
  pieno) e passa per la stessa selezione anti-scorciatoia di §2.
- Il controllo `len(composto) > ctx` resta identico (i troncati sono più
  corti, mai un problema).
- Finestre di seed invariate: i record troncati usano lo stesso seed del
  record pieno (stessa riga di finestra, più righe nel JSONL). Il campo
  `seed` resta quello vero.

## 4. Plumbing config e split

`carica_config` non cambia. In `genera_dataset(stadio, split, config)`:
le opzioni `anti_scorciatoia` e `troncamenti` si leggono da
`config["dataset"]` ma si applicano SOLO se `split == "train"`; per dev ed
esame il percorso di codice deve restare quello attuale (stessa
`rng_domande`, stesso `n_per_tipo`, nessun record extra). Implementazione
suggerita: `genera_record(stadio, seed, config, *, split)` con default che
preserva la firma attuale per i chiamanti esistenti, oppure un parametro
booleano — a scelta dell'esecutore, purché i test di §6 passino.

## 5. Nuovi artefatti

### 5.1 `configs/v1_anti.yaml`

```yaml
# Esperimento anti-scorciatoia (vedi fasi/FASE2_PIANO_ANTISCORCIATOIA.md §0-§1):
# cast pieno, da zero, train con selezione anti-scorciatoia + storie troncate.
# Dev/esame = distribuzione ufficiale invariata (confronto diretto con v1: 0,573).

nome_run: v1_anti
device: auto
seed_torch: 1337

percorsi:
  lessico: lingua/lessico.tsv
  vocabolario: cervello/vocabolario.json
  dati_dir: dati/anti
  risultati_dir: dati/risultati

dataset:
  ctx: 3072
  train_storie: 6000
  dev_storie: 300
  esame_storie: 300
  n_per_tipo: 8
  anti_scorciatoia:        # SOLO split train (vedi piano §2/§4)
    quota_difficili: 0.6
    candidate_per_tipo: 999
  troncamenti: true        # SOLO split train (vedi piano §3)

stadi:
  1:
    tipi: [posizione]
    soglia: 0.95
    storie_corte: true

modello:                   # IDENTICO a configs/v1.yaml (~7,18M parametri)
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

Più un `configs/v1_anti_fumo.yaml` per il fumo su Colab: come sopra ma
`train_storie: 300`, `max_step: 600`, `intervallo_valutazione: 200`.

### 5.2 Best-dev checkpoint — `cervello/addestra.py`

A ogni valutazione, se `esattezza_dev` supera il massimo visto finora nel
run dello stadio, salvare `stadio<N>_best.pt` (stesso formato del checkpoint
di fine stadio: basta il modello + `step` + `esattezza_dev`; NON serve
l'ottimizzatore). Scrittura atomica (tmp+rename) e replica via
`--copia-sicurezza` come per gli altri file. Il checkpoint di fine stadio
`stadio<N>.pt` resta l'ultimo (comportamento invariato); il best è un file
in più. Il massimo va tenuto anche dentro `stadio<N>_parziale.pt` così la
ripresa dopo interruzione non "dimentica" il best già visto.

### 5.3 Strumento di diagnosi — `esami/diagnosi.py` (nuovo modulo, torch)

Rende permanente l'analisi del 2026-07-08 (oggi vive in script di scratchpad
non committati). CLI:

```
python -m esami.diagnosi --config configs/v1_anti.yaml --stadio 1 \
    --checkpoint dati/risultati/v1_anti/stadio1.pt [--split esame] [--max-esempi N]
```

Per ogni esempio dello split: decodifica greedy (riusare
`esami.esamina.valuta_esempio`), poi rigenera la `Storia` dal seed del record
(`genera_storia` + `_n_tick` + cast del config — deterministico) e calcola le
proprietà di §2.1. Output: JSON in
`dati/risultati/<run>/diagnosi_stadio<N>.json` + stampa. Metriche normative:

1. esattezza e conteggi per categoria (come `esamina.py`);
2. **baseline euristiche** sul sottoinsieme con oro noto: "ultima menzione
   del bersaglio", "luogo più frequente", "luogo dell'ultimo evento", e
   l'esattezza del modello sullo stesso sottoinsieme;
3. **esattezza condizionata**: oro==più-frequente sì/no; fasce di
   `distanza_coda` (0, 1-2, 3-5, ≥6); D3 (tracking puro) sì/no;
4. **anatomia degli errori**: generato == più-frequente / ultimo-evento /
   luogo-finale-di-altra-persona / ultima-menzione-stantia / altro;
5. esattezza per entità (bersaglio).

I numeri di riferimento del checkpoint `v1_facile` (per confronto) sono in
§0 e nella memoria di progetto.

### 5.4 Notebook Colab — `colab_training.ipynb`, sezione 10

Nuova sezione "10. Esperimento anti-scorciatoia (v1_anti)" sul modello esatto
della sezione 7 (stadio 1 di produzione): genera dataset con
`configs/v1_anti.yaml`, stampa la composizione del train (conteggio
difficile/facile/non-lo-so e record troncati/pieni — verifica visiva della
quota), poi `addestra.py --config configs/v1_anti.yaml --stadio 1
--copia-sicurezza <Drive>`, poi `esamina.py` e `diagnosi.py` sia su
`stadio1.pt` sia su `stadio1_best.pt`, con copia dei risultati su Drive.
Prima cella della sezione: fumo con `v1_anti_fumo.yaml` (~10 min).

## 6. Test (in `tests/test_esami.py` e `tests/test_cervello.py`)

Obbligatori, con la stessa disciplina dei gruppi esistenti:

1. **Default invariato byte per byte**: per una manciata di seed,
   `genera_record` con `configs/v1.yaml` e `configs/v1_facile.yaml` (nessuna
   chiave nuova) produce ESATTAMENTE il dict di oggi (fissare il valore
   atteso prima di toccare il codice, o confrontare con il codice a monte
   via git). È il cancello più importante del piano.
2. **Proprietà D1/D2/D3**: su storie costruite negli test (liste di Evento
   sintetiche) con esiti noti — un oggetto trasportato (D3), un bersaglio
   con coda lunga (D2), un oro fuori dal luogo dominante (D1) — la
   classificazione difficile/facile è quella attesa. ATTENZIONE (trappola
   scoperta il 2026-07-08): se si costruiscono eventi a mano, l'ordine
   dentro un tick è SEMPRE quello del cast (anna→piero→maria…, 222/222 nei
   dati reali) — qui serve solo per realismo dei test, ma non violarlo.
3. **Quota rispettata**: con `anti_scorciatoia` attivo su ~50 storie reali,
   la frazione di esempi `difficolta == "difficile"` tra quelli con oro noto
   è ≥ `quota_difficili` (tolleranza per storie povere di candidate), e la
   frazione non-lo-so resta nel range 15-25%.
4. **Troncamenti**: (a) gli eventi del record troncato a k sono il prefisso
   esatto (confronto con la storia piena filtrata per `t <= k`); (b) campo
   `troncamento` presente/assente correttamente; (c) con `troncamenti: true`
   una storia da 6 tick produce 4 record (pieno + k=3,4,5); (d) l'oro di una
   domanda sulla stessa entità può cambiare tra troncamenti (testare su un
   seed dove succede, cercandolo programmaticamente).
5. **Split protetti**: `genera_dataset(1, "dev"|"esame", config_anti)` produce
   record IDENTICI a quelli senza chiavi anti (byte per byte).
6. **Rifiuto seed d'esame**: invariato (i test esistenti devono restare verdi).
7. **Best-dev**: con un modello iniettato e una sequenza di esattezze dev
   pilotata, `stadio<N>_best.pt` contiene i pesi del momento migliore, non
   dell'ultimo; sopravvive a interruzione+ripresa (estendere
   `TestCheckpointIntraStadio`).
8. **Diagnosi**: con un modello iniettato su un dataset giocattolo, le
   metriche 2-4 di §5.3 hanno i valori attesi calcolati a mano.

## 7. Tappe di lavoro (con cancelli)

- **T1 — selezione anti-scorciatoia** (`esami/genera.py`): §2 + test 1-3, 5-6.
  Cancello: tutti i test verdi, incluso il byte-identico di default.
- **T2 — troncamenti** (`esami/genera.py`): §3 + test 4. Cancello: test verdi.
- **T3 — best-dev** (`cervello/addestra.py`): §5.2 + test 7. Cancello: test
  verdi (il gruppo checkpoint esistente NON deve rompersi).
- **T4 — diagnosi** (`esami/diagnosi.py`): §5.3 + test 8. Cancello: test
  verdi; una corsa locale su CPU con `--max-esempi 40` sul checkpoint
  `v1_facile` (in `/home/andrea/Scaricati/v1_facile/stadio1.pt`, se ancora
  presente) riproduce l'ordine di grandezza dei numeri di §0.
- **T5 — config + notebook + fumo**: §5.1 + §5.4. Cancello: fumo `v1_anti_fumo`
  su Colab OK (pipeline intera, composizione del train stampata e sensata,
  esattezza dev > 0 a fine fumo) — lo lancia Andrea.
- **T6 — run vero** (~1-1,2h su Colab, lo lancia Andrea): dataset pieno,
  training, esame ufficiale, `diagnosi.py` su `stadio1.pt` E
  `stadio1_best.pt`. Consegna: numeri + aggiornamento dello "Stato di
  avanzamento" di FASE2_PIANO.md e di questo piano.
- Un commit per tappa, come sempre; test pesanti su Colab, mai in locale
  (vedi memoria: workflow commit+push → Colab).

## 8. Criteri di lettura del risultato (T6)

Confronti: `v1` esame 0,573; baseline "ultima menzione" 93,8%; condizionata
"oro != più frequente" oggi 52,7% (`v1_facile`, che è il checkpoint MIGLIORE
finora). Sul best tra `stadio1.pt` e `stadio1_best.pt`:

- **Successo pieno**: esame ≥ 0,95 → stadio 1 superato; si prosegue col
  curriculum portandosi dietro la ricetta anti-scorciatoia (decidere con
  Andrea se rigenerare anche gli stadi 2-3 con le stesse chiavi).
- **Progresso forte**: esame ≥ 0,80 E condizionata "oro != più frequente"
  ≥ 0,70 E baseline-gap in chiusura (modello vicino alla baseline "ultima
  menzione" sul suo sottoinsieme) → la leva funziona ma non basta il budget:
  proporre ad Andrea di scalare (più storie, più step, quota più alta).
- **Progresso debole o nullo**: esame < 0,80 o esattezza train ancora ~0,8
  (misurarla con `diagnosi.py --split train --max-esempi 200`) → la leva
  dati da sola non rompe il plateau: fermarsi e proporre ad Andrea la leva
  di formato "[DOMANDA] prima della storia" (modifica alla spec FASE2,
  decisione sua), eventualmente + LR costante.

## 9. Rimandato esplicitamente (non farlo ora)

- **"[DOMANDA] prima della storia"** (riordino della sequenza): leva
  potenzialmente più forte ma cambia il formato normativo di FASE2
  (`componi_esempio`, dati, esame). Solo dopo l'esito di questo esperimento,
  con decisione esplicita di Andrea.
- Incatenamento di cast (3→4-5→6 via `--pesi-iniziali`): superato dalla
  diagnosi (trasferirebbe la scorciatoia, non la regola); i meccanismi
  restano disponibili se si vorrà riprovarci.
- LR costante / cicli di LR, KV cache nella decodifica, attention analysis:
  fuori scope.
- Stadi 2-3 con dati anti-scorciatoia: dopo, se lo stadio 1 passa.
