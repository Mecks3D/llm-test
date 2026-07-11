# PIANO DI RICERCA — Diagnosi e leve contro lo shortcut-learning (Fase 2a, stadio 1)

Questo documento è diverso dai piani esecutivi (FASE1_PIANO, FASE2_PIANO,
FASE2_PIANO_ANTISCORCIATOIA): non è un piano per un esecutore in conversazione
pulita, ma la **roadmap di ricerca** concordata con Andrea il 2026-07-10.
Ogni esperimento qui elencato, quando arriva il suo turno, riceve il suo
mini-piano esecutivo (decisioni vincolanti, tappe, cancelli) nello stile dei
piani precedenti. Le decisioni già prese da Andrea e segnate come **vincolo
permanente** (§1) non si rimettono in discussione; tutto il resto è ordinato
per costo crescente e si decide strada facendo, in base agli esiti.

Prerequisiti di lettura: `PROGETTO.md`, `fasi/FASE2.md`, `fasi/FASE2_PIANO.md`
(§1 decisioni vincolanti, tutte ancora valide), `fasi/FASE2_PIANO_ANTISCORCIATOIA.md`
(§0 per la diagnosi, §8 per l'esito).

---

## 0. Contesto: dove siamo e perché questo piano

Lo stadio 1 (domande di posizione) non passa l'esame (soglia 0,95). Storia
completa dei run, tutti riproducibili per seed:

| Run | Setup | Esame | Note |
|---|---|---|---|
| `v1` | cast pieno (6), 20.000 step | 0,573 | T7 originale |
| `v1_facile` | cast 3, da checkpoint `v1` | 0,7448 | salto secco di cast |
| `v1_anti` | cast pieno, da zero, dati anti-scorciatoia + troncamenti | 0,5129 | leva dati = "progresso debole" |
| `v1_grad1` | cast 1, storie 1-3 tick | 0,9826 | |
| `v1_grad2` | cast 2, da best di grad1 | 0,9040 | **miglior checkpoint attuale**, usato in `interfaccia/` |
| `v1_grad3` | cast 3, da best di grad2 | 0,7836 | vs 0,7448 del salto secco: gradualità = piccolo vantaggio, stessa firma di errore |

**Diagnosi consolidata** (analisi 2026-07-08 su `v1_facile`, confermata su
`v1_anti` e `v1_grad3` con `esami/diagnosi.py`):

- Il modello non fa binding entità→luogo: fa retrieval associativo pesato su
  **frequenza e recency**. La regola simbolica "luogo dell'ultima menzione del
  bersaglio" fa 0,94-0,97; il modello 0,65-0,66 sullo stesso sottoinsieme.
- **Esattezza sul train ~0,80** dopo ~16 epoche: plateau di ottimizzazione
  (SGD trova la scorciatoia e non ne esce), NON gap di generalizzazione.
- La leva dati (selezione anti-scorciatoia + storie troncate, run `v1_anti`)
  non rompe la scorciatoia: margine modello-sopra-baseline identico tra train
  ed esame. Il curriculum a cast crescente nemmeno: sposta il numero, non la
  causa.
- Cosa il modello HA imparato e NON va rotto: grammatica dei grafi
  (0 malformate) e calibrazione non-lo-so quasi perfetta (396/399).

**Conclusione condivisa**: le leve "più dati curati" e "curriculum" sono
esaurite. Restano tre famiglie: (A) diagnosi economiche mai fatte che dicono
DOVE è il collo di bottiglia; (B) supervisione densa dello stato del mondo
(unica leva-dati non provata che attacca la causa); (C) leve architetturali.
Questo piano le ordina. Regola aurea: **ogni esperimento deve essere
informativo qualunque sia l'esito** — se un run non cambia la decisione
successiva in nessuno dei due esiti, non si lancia.

---

## 1. Vincoli permanenti (decisioni di Andrea, non riaprirle)

1. **Niente `[DOMANDA]` prima della storia come formato del progetto.** Il
   modello deve leggere e formarsi una rappresentazione propria del mondo,
   poi ricevere la domanda ("leggi prima, poi rispondi"). Il query-first è
   ammesso SOLO come run diagnostico una-tantum, dichiarato non normativo
   (esperimento A4), se Andrea decide di lanciarlo.
2. **Dev ed esame ufficiali invariati** (stesse finestre di seed di
   FASE2_PIANO): ogni nuovo numero deve restare confrontabile con la tabella
   in §0. Split nuovi (es. tracking-puro, A3) si AGGIUNGONO, non sostituiscono.
3. **Mai addestrare su seed d'esame**; determinismo con RNG espliciti;
   `mondo/` e `lingua/` non si toccano senza fermarsi e chiedere (decisione 8
   di FASE2_PIANO — l'unica eccezione storica, il parametro `persone_cast`,
   fu approvata esplicitamente).
4. **Metrica primaria di lettura d'ora in poi**: margine del modello sopra le
   baseline euristiche di `esami/diagnosi.py` (+ split tracking-puro quando
   esiste), non l'esattezza grezza da sola. La calibrazione non-lo-so si
   monitora a ogni run: se degrada, va segnalato come regressione.
5. **Run > 2h su Colab solo con ok esplicito di Andrea** (decisione 10 di
   FASE2_PIANO, sempre valida). Workflow test/training: commit+push, poi
   Colab — mai training in locale (CPU impraticabile, misurato).
6. **Dipendenze**: restano stdlib + torch + pyyaml. Niente pacchetti CUDA
   esterni (es. `mamba-ssm`): un eventuale strato ricorrente si implementa
   in-repo, in PyTorch puro.

---

## 2. FASE A — Diagnosi economiche (prima di qualunque leva costosa)

Obiettivo: distinguere tre ipotesi che oggi sono indistinguibili —
(i) sotto-training (il meccanismo non ha ancora avuto tempo/step di formarsi),
(ii) capacità insufficiente, (iii) limite del setup (formato/architettura).
Costo totale: 2-3 giorni di calendario, 2 run Colab da ~1,5-3,5h, il resto in
locale gratis.

### A1. Ispezione degli head di attenzione (locale, gratis) — PRIMO PASSO

Il modello è piccolo apposta per essere ispezionabile (PROGETTO.md, principio
di piccolezza) e finora si sono guardati solo gli output, mai i meccanismi
interni. Il meccanismo canonico per "trova l'ultima occorrenza di questo
lemma e copia il contesto" è l'**induction head** (Olsson et al. 2022), che
notoriamente compare con una transizione di fase durante il training. Con
8 layer × 8 head si controlla direttamente.

- **Cosa**: nuovo modulo `esami/ispeziona.py` (stile `diagnosi.py`: riusa il
  caricamento checkpoint di `esamina.py`, CPU, con test iniettati). Per ogni
  head di ogni layer, su un campione di esempi reali: punteggio di attenzione
  ai pattern "stessa-entità precedente" (dal token del bersaglio nella domanda
  verso le menzioni del bersaglio nella storia, in particolare l'ultima) e
  punteggio di induction classico (attenzione al token successivo
  all'occorrenza precedente del token corrente). Output JSON + stampa
  leggibile (mappa layer×head).
- **Su cosa**: i checkpoint già in locale (`~/Scaricati/v1/`, `v1_facile/`,
  `v1_anti/`) + `v1_grad2/stadio1_best.pt` (da scaricare da Drive, serve
  comunque per l'interfaccia).
- **Lettura**:
  - Nessun head con firma di induction/binding → indizio forte di
    **sotto-training**: la priorità passa ad A2-braccio-lungo (più step).
  - Head presenti e puntati giusto, ma risposta sbagliata → il problema è
    **a valle** (come la rete usa l'informazione): la priorità passa a B e C.
  - Confronto grad2 (0,904) vs v1_anti (0,513): se grad2 ha head che v1_anti
    non ha, si vede *cosa* la gradualità ha comprato.

### A2. Ablation di capacità e di budget (2 run Colab)

Tutti gli esperimenti finora hanno tenuto l'architettura fissa (8 layer,
8 head, d=256, ~7,18M) e variato solo i dati. Il train fermo a ~0,80 grida
"ottimizzazione o capacità": va isolata la variabile prima di concludere che
serve un'architettura diversa.

- **Base di confronto**: la ricetta `v1_anti` (cast pieno, da zero, dati
  anti-scorciatoia) — è l'unico run da-zero recente, baseline pulita 0,5129.
  Dati IDENTICI (stessi seed, stesso config dataset), cambia solo il modello
  o il budget.
- **Braccio (a) — capacità**: d_model 384 (o 12 layer), ~15-16M parametri
  (dentro lo spirito ~10-20M di PROGETTO.md), stesso max_step 8.000.
  Stima ~1,5-2h.
- **Braccio (b) — budget**: architettura invariata, max_step 24.000 (3×) con
  schedule cosine fresco. Motivato anche dalle curve dev mai in plateau netto
  (grad3 a 4.000 step saliva ancora, rumorosamente) e dalla transizione di
  fase degli induction head, che può arrivare tardi. Stima ~3-3,5h → serve
  l'ok esplicito di Andrea (vincolo 5).
- **Lettura** (per ciascun braccio, con `diagnosi.py` completo):
  - Train sale ben sopra 0,80 e il margine sopra-baseline si allarga → il
    collo era capacità/ottimizzazione: strada più economica del previsto,
    si scala quella PRIMA di toccare formato o architettura.
  - Nulla cambia → evidenza forte che la leva è strutturale (B, poi C), e
    d'ora in poi nessuno può obiettare "bastava un modello più grande".

### A3. Split d'esame "tracking puro" (locale, permanente)

Rendere ufficiale la metrica che oggi si ricava a mano da `diagnosi.py`.

- **Cosa**: in `esami/genera.py`, riusando la classificazione D1/D2/D3 già
  esistente (T1 anti-scorciatoia), un file d'esame AGGIUNTIVO per run —
  `esame_tracking.json` — con sole domande dove TUTTE le euristiche sbagliano:
  D1 ∧ D2 ∧ D3 (oro ≠ luogo più frequente della storia; ≥3 eventi dopo
  l'ultima menzione del bersaglio; oro ≠ luogo dell'evento dell'ultima
  menzione — definizioni esatte in `esami/genera.py::_classifica_domanda_posizione`,
  dove oggi basta UNA delle tre per "difficile": qui serve la congiunzione).
  Seed dalla finestra d'esame, MAI di train. `diagnosi.py` lo valuta e lo
  riporta accanto all'esame ufficiale.
- **Lettura**: è il "punteggio di tracking vero" di ogni run presente e
  futuro. Su questo split le scorciatoie valgono ~0 per costruzione: ogni
  punto sopra il caso è binding reale. Da qui in avanti ogni esperimento
  riporta: esame ufficiale, esame tracking, margini sopra baseline,
  calibrazione non-lo-so.

### A4. (Opzionale, decide Andrea) Run diagnostico query-first

UNA run con `[DOMANDA]` prima della storia, dichiarata **non normativa**
(vincolo 1): non cambia FASE2, non produce checkpoint "di linea". Misura il
**tetto**: quanta parte del gap è dovuta al "memorizzare senza sapere cosa
serve".

- **Lettura**: esame ~0,95 → conferma che l'informazione è tutta estraibile
  e il collo è la memorizzazione non diretta → rafforza la scommessa su B
  (che dà al modello un motivo per mantenere TUTTO lo stato). Esame ancora
  basso → problema più profondo (rafforza A2/C). Stima ~1-1,2h.
- Se A1+A2 danno già un quadro chiaro, questo run si può saltare.

---

## 3. FASE B — Supervisione densa in-sequenza dello stato del mondo

L'unica leva-dati non ancora provata che attacca la causa. ATTENZIONE alla
sovrapposizione con il passato: le storie troncate di `v1_anti` erano già
"domande a istanti intermedi", ma come **record separati** — e non sono
bastate. La versione nuova è **dentro la stessa sequenza**: una sola storia
in cui, a ogni fine tick, il modello emette il grafo dello stato corrente
(dov'è ciascuna persona; per lo stadio 1 bastano le posizioni), e la domanda
resta DOPO la storia come sempre. Costringe a *mantenere* lo stato lungo
tutto il contesto invece di ricalcolarlo una volta sola alla fine.

È la realizzazione letterale del vincolo "leggi prima, poi rispondi": il
modello impara a simulare il mondo mentre legge (il "grafo-pensiero" di
PROGETTO.md reso esplicito e verificabile tick per tick contro la verità del
simulatore), e rispondere diventa leggere il proprio stato.

**Decisioni di design da prendere con Andrea PRIMA del mini-piano** (bivi
veri, non dettagli):

1. **All'esame, i blocchi `[STATO]` chi li scrive?** Se fossero dati in input
   conterrebbero la risposta (leak). Le opzioni: (i) il modello li GENERA
   autoregressivamente a ogni fine tick anche all'esame (raccomandata: è il
   punto dell'esperimento, il modello "pensa ad alta voce" in grafi, ogni
   blocco è verificabile; costo: decodifica più lunga, errori che si
   propagano); (ii) presenti solo in training, esame invariato (rischio:
   mismatch di formato train/esame).
2. **Contenuto del blocco stato** per lo stadio 1: solo posizioni delle
   persone, o anche oggetti trasportati? (Le posizioni bastano per le domande
   di stadio 1; gli oggetti preparano lo stadio 3.)
3. **Budget di contesto**: i blocchi stato allungano le sequenze; misurare le
   lunghezze reali (ctx attuale 3072) prima di fissare i config — con le
   storie corte dei gradini il margine c'è, a cast pieno va verificato.

Il mini-piano esecutivo (formato ANTISCORCIATOIA: tappe, cancelli, config
`v1_stato*`, byte-identità dei default) si scrive dopo la Fase A, incorporando
quello che A1/A2 hanno insegnato.

---

## 4. FASE C — Leve architetturali (solo dopo A e B)

Da aprire se A dice "non è capacità" e B non basta (o non basta da sola).
Due gradini, dal meno al più invasivo:

### C1. Ibrido ricorrente / state-space

Sostituire parte dei layer di attenzione con uno strato ricorrente a stato
esplicito — è la "questione aperta" già in PROGETTO.md. Uno stato che si
aggiorna tick dopo tick è la forma naturale di "dov'è ognuno adesso" (memoria
di lavoro). A parità di parametri, confronto diretto col transformer puro
sugli stessi dati (il migliore emerso da B). Implementazione in PyTorch puro
in-repo (vincolo 6): va bene una ricorrenza gated minimale (famiglia
minGRU/LRU/Mamba semplificato), non serve il kernel CUDA ufficiale a questa
scala.

### C2. Memoria indicizzata per entità (stile Recurrent Entity Networks)

Slot di memoria dedicati alle entità: il prior architetturale esplicito per
il binding. È struttura, non conoscenza importata (non viola la regola 1 di
CLAUDE.md), e le Entity Networks nacquero proprio per bAbI. È però la mossa
più invasiva e cabla nel modello la nozione di "entità" che forse vogliamo
fargli scoprire: ultima della lista, solo se C1 non basta.

---

## 5. Orizzonte (non in programma ora)

Riformulare il cervello come **funzione di transizione sugli stati-mondo**
(predire nello spazio latente dei grafi, direzione JEPA/energy-based già
annotata in PROGETTO.md) e la generazione a diffusione su grafo (v2, Fase 3).
È la destinazione coerente di tutto questo lavoro — "un modello che pensa in
stati del mondo, la lingua è solo la pelle" — ma è un progetto di mesi: si
affronta quando lo stadio 1 è vinto, non per scavalcarlo.

---

## 6. Albero di decisione (sintesi operativa)

```
A1 (head, locale) ──┬─ nessun head di binding → priorità A2b (più step)
                    └─ head presenti ma inutilizzati → priorità B

A2 (capacità/budget) ─┬─ train >> 0,80, margini su → scalare QUELLA leva, poi rivalutare
                      └─ invariato → B con fiducia (capacità esclusa)

A3 (split tracking) → metrica permanente per tutto il resto

B (stato in-sequenza) ─┬─ margini e tracking-split su → consolidare, tornare al curriculum
                       └─ insufficiente → C1 (ricorrenza) → C2 (slot entità)
```

Ordine dei primi passi (2026-07-11): **A1 e A3 si possono iniziare subito in
locale** (nessun run Colab, nessuna decisione pendente); A2 richiede solo la
scelta del braccio e l'ok di Andrea per il lancio; A4 resta a discrezione di
Andrea.

---

## 7. Stato di avanzamento

(da tenere aggiornato qui, come negli altri piani)

- 2026-07-10: piano scritto e concordato con Andrea. Nessun esperimento
  ancora iniziato. Prossimo passo: A1 (`esami/ispeziona.py`) + A3 (split
  tracking-puro), entrambi in locale; scaricare `v1_grad2/stadio1_best.pt`
  da Drive (serve sia per A1 sia per l'interfaccia).

- **2026-07-11: A1 e A3 fatti, entrambi in locale (v1_grad2/stadio1_best.pt
  era già in locale, non è servito scaricarlo di nuovo).**

  **A1** (`esami/ispeziona.py`, commit 85f90c4, 17 test): l'attenzione si
  ricalcola a mano (il kernel fuso di `modello.py` non restituisce i pesi),
  verificato bit a bit contro il forward diretto. Due punteggi per head/layer
  su 50 esempi "posizione" per checkpoint: (a) attenzione dal bersaglio nella
  domanda verso le sue menzioni nella storia (tutte e sola ultima); (b)
  induction classico (Olsson et al.: attenzione al token successivo alla
  precedente occorrenza del token corrente). Girato su tutti i checkpoint già
  in locale:

  | Run (esame ufficiale) | T medio | max "ultima menzione" | vs uniforme | max induction |
  |---|---|---|---|---|
  | v1 (0,573) | 440 | 0,013 | ~5,7× | 0,021 |
  | v1_facile (0,745) | 225 | 0,054 | ~12,1× | 0,032 |
  | v1_anti (0,513) | 440 | 0,013 | ~5,7× | 0,009 |
  | v1_grad2 (0,904) | 89 | 0,039 | ~3,5× | 0,033 |

  **Nessun checkpoint ha sviluppato una head con firma forte di "stessa
  entità"/induction** — nemmeno `v1_grad2`, il migliore comportamentalmente:
  l'attenzione massima resta bassa in assoluto (1-5%) e solo modestamente
  sopra il caso (mai vicina alla saturazione tipica di una vera induction
  head). Punta verso l'ipotesi "sotto-training/capacità insufficiente" (§6:
  "nessun head di binding → priorità A2b, più step").

  **A3** (`esami/genera.py::genera_esame_tracking`/`_tracking_puro` +
  `esami/diagnosi.py --split tracking`, commit f886579, 7 test nuovi):
  nuovo file `tracking.jsonl` per run, sotto-insieme VERIFICATO di
  `esame.jsonl` (stessi seed/domande candidate) con solo le domande dove
  D1 ∧ D2 ∧ D3 sono TUTTE vere. Risultati (`diagnosi_stadio1_tracking.json`):

  | Run | Esame ufficiale | Tracking puro (n) | baseline "ultimo evento storia" |
  |---|---|---|---|
  | v1 | 0,573 | 0,378 (n=82) | 0,305 |
  | v1_facile | 0,745 | 0,440 (n=50) | 0,380 |
  | v1_anti | 0,513 | 0,122 (n=82) | 0,305 |
  | v1_grad2 | 0,904 | 0,467 (n=15) | 0,267 |

  Su tracking puro l'esattezza crolla ovunque rispetto all'esame ufficiale
  (v1_grad2: 0,904 → 0,467); `v1_anti` scende SOTTO la baseline banale
  (0,122 < 0,305). Conferma quantitativa che l'esame ufficiale è in gran
  parte scorciatoia: solo `v1_facile`/`v1_grad2` restano un poco sopra la
  baseline "ultimo evento", un segnale di binding vero ma debole. Nota per
  chi rilegge: `n` di `v1_grad2` è piccolo (15) perché con cast 2 e storie
  1-3 tick ci sono strutturalmente pochi casi "difficili" — è anche questo
  parte della spiegazione del suo 0,904 sull'esame ufficiale.

  **Lettura combinata A1+A3, coerente con l'albero §6**: nessun meccanismo di
  binding dedicato + crollo su tracking puro → priorità ad **A2** (ablation
  capacità/budget su Colab). Prossimo passo: decidere con Andrea il braccio
  (capacità d=384/12-layer, o budget 3× step) e ottenere il suo ok esplicito
  prima del run (>2h per il braccio budget, vincolo 5).
