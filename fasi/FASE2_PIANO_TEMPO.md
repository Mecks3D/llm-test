# PIANO ESECUTIVO — Esperimento "tempo": un personaggio, molte ore, domande condizionate

Piano esecutivo nello stile di FASE2_PIANO_ANTISCORCIATOIA.md: decisioni
vincolanti, tappe T1–T6 con cancelli, pensato per un esecutore in
conversazione pulita. Prerequisiti di lettura: `PROGETTO.md`, `CLAUDE.md`,
`fasi/FASE2_PIANO.md` (§1), `fasi/FASE2_PIANO_DIAGNOSI.md` (contesto della
diagnosi e vincoli permanenti — questo esperimento è una **linea parallela**
alla roadmap A2/B/C di quel piano, concordata con Andrea il 2026-07-11).

---

## 0. Contesto e obiettivo

La diagnosi (FASE2_PIANO_DIAGNOSI.md §0 e §7) dice che il modello non fa
binding entità→luogo: usa scorciatoie frequenza/recency, nessuna head di
attenzione dedicata (A1), crollo sullo split tracking-puro (A3). Tutti gli
esperimenti finora chiedevano però solo lo **stato finale**.

Questo esperimento toglie del tutto la confusione tra entità (UN solo
personaggio per storia) e allunga le storie (12–24 tick), per misurare una
capacità più a monte: **seguire come cambia lo stato di UNA persona nel
tempo**. Le domande sono nuove, condizionate nel tempo o nel luogo:

- «Dove si trova Anna alle due?» (posizione a un tick passato)
- «Che cosa fa Anna alle quattro?» (azione a un tick passato)
- «Che cosa fa Anna in giardino?» (azione in un luogo)

Perché è informativo in entrambi gli esiti (regola aurea del piano di
diagnosi):

- **Se il modello riesce** → il tracking temporale di una singola entità si
  forma; il collo di bottiglia del multi-personaggio è l'interferenza tra
  entità → rafforza la Fase B (stato denso) e C2 (slot per entità).
- **Se fallisce anche qui** → il tracking temporale in sé non si forma a
  questa scala/budget → rafforza A2 (capacità/budget) e C1 (ricorrenza).
- Lettura fine possibile: `azione_tempo` è quasi un lookup (trova il tick N e
  copia l'evento), `posizione_tempo` richiede di **propagare** lo stato tra i
  tick (la posizione alle due può essere stata stabilita all'una). Se la
  prima riesce e la seconda no, il modello sa cercare ma non mantiene stato.

Attenzione al confronto storico: `v1_grad1` (cast 1, storie 1–3 tick, solo
stato finale) fece 0,9826 — perché questo esperimento dica qualcosa di nuovo
le storie DEVONO essere lunghe e le domande davvero condizionate.

---

## 1. Decisioni vincolanti (Andrea, 2026-07-11 — non riaprirle)

1. **Ambiguità**: una domanda si genera SOLO se la risposta d'oro è unica
   (precedente già nel codice: «perché X dorme?» in `mondo/domande.py`,
   `_genera_causa`). I casi ambigui («che cosa fa X in giardino?» quando in
   giardino ha fatto cose diverse) **non si chiedono**. Vietato usare
   `non-lo-so` per l'ambiguità: nel progetto significa «non derivabile dalla
   storia», non «più risposte possibili» — corromperebbe la calibrazione
   epistemica, il risultato migliore ottenuto finora.
2. **Da zero**: pesi casuali, nessun `--pesi-iniziali` (attribuzione pulita;
   i checkpoint esistenti portano dentro la scorciatoia frequenza/recency).
3. **Esperimento a parte** (run `v1_tempo`, config e `dati/tempo/` propri):
   il curriculum ufficiale, i dev/esame ufficiali e i config esistenti non
   cambiano (vincolo permanente 2 di FASE2_PIANO_DIAGNOSI.md). Nel train i
   tipi nuovi **affiancano** il tipo `posizione` esistente (lo stato finale
   è il caso degenere e fa da àncora facile; è anche la fonte delle domande
   non-lo-so del mix, vedi §2.4).
4. **Cast rotante**: una sola persona per storia, a rotazione deterministica
   tra le 6 (`seed % 6`), non sempre anna — il soggetto della domanda resta
   informativo, il risultato generalizza sull'identità.
5. **Tocco a `lingua/` approvato** (esplicitamente, come richiede CLAUDE.md):
   nuova riga `che-cosa` nel lessico **in coda al file** + nuovi stampi;
   cancello duro: round-trip 100% (vedi T2).
6. A2 (ablation capacità/budget) resta in coda: **non** si lancia in
   parallelo per ora.

Decisioni di progetto prese in questo piano (dichiarate qui, vincolanti per
l'esecutore):

7. **Semantica temporale**: «alle N» = stato **al termine del tick N** (dopo
   il suo evento), coerente con la narrazione («Alle due Anna va in cucina»
   → alle due Anna è in cucina). Verificato empiricamente: la posizione
   ricavata dagli eventi coincide con `stato_finale.luogo_effettivo` della
   storia troncata al tick N (0 mismatch su 300 storie × 4 tick campione).
8. **Tempo verbale**: presente narrativo, coerente con le storie («Dove si
   trova Anna alle due?»). NIENTE imperfetto: zero morfologia nuova.
9. **Niente `troncamenti`** nel config `v1_tempo`: le domande condizionate
   nel tempo danno già supervisione densa (è il loro scopo) e con storie da
   24 tick i troncamenti esploderebbero il dataset.
10. **`mondo/domande.py` si estende senza toccare l'esistente**: nuova
    funzione di ingresso `genera_domande_tempo(...)`; `genera_domande`,
    `_GENERATORI`, `QUOTA_NON_LO_SO_PER_TIPO` restano byte-identici.

---

## 2. T1 — `mondo/domande.py`: i tre tipi nuovi

Modulo protetto: questa modifica è l'oggetto dell'approvazione di Andrea del
2026-07-11 (§1). Non toccare nient'altro in `mondo/`.

### 2.1 Helper di stato (nuovi, privati)

- `_posizione_al_tick(storia, pid, t) -> str | None`: luogo dell'ULTIMO
  evento con `e.agente == pid` e `e.t <= t` (gli eventi hanno sempre
  `luogo` valorizzato — verificato su 300 storie; essere difensivi e saltare
  eventuali `luogo None`). `None` se nessun evento localizzante ≤ t (può
  succedere solo a inizio storia, es. persona che dorme dal tick 1: posizione
  davvero non derivabile → non-lo-so).
- `_evento_al_tick(storia, pid, t) -> Evento | None`: l'evento con
  `e.agente == pid` e `e.t == t` (con cast 1 è al più uno).
- `_grafo_evento_senza_tempo(evento) -> Grafo`: stesso grafo di
  `evento_a_grafo` ma SENZA il nodo/arco `obl:tempo` (non modificare
  `evento_a_grafo`: costruire a parte, stessi ordini di nodi/archi).

### 2.2 `_genera_posizione_tempo(storia, rng, n, n_tick)`

- Candidati: ogni `t` in `1..n_tick` per il protagonista `pid` (l'unica
  persona in `storia.stato_finale.persone` — asserire che sia una sola).
- Domanda: `grafo_fatto("trovarsi", nsubj=pid, **{"obl:tempo":
  lemma_numero(t)}, quesito="dove")` (ordine kwargs esattamente questo).
- Risposta: se `_posizione_al_tick` dà `L` →
  `grafo_fatto("essere", nsubj=pid, **{"obl:luogo": L, "obl:tempo":
  lemma_numero(t)})`; altrimenti `NON_LO_SO`.
- Selezione: `rng.sample` di `min(n, len(candidati))` sull'elenco ordinato
  per `t` crescente (determinismo).

### 2.3 `_genera_azione_tempo(storia, rng, n, n_tick)`

- Domanda: `grafo_fatto("fare", nsubj=pid, **{"obl:tempo":
  lemma_numero(t)}, quesito="che-cosa")`.
- Risposta, per ogni `t` in `1..n_tick`:
  - se `_evento_al_tick` trova l'evento `e` → `evento_a_grafo(e)` (contiene
    già `obl:tempo` = t; la risposta è l'evento stesso);
  - se il tick non ha evento e l'ultimo evento del pid con `t' < t` è
    `dormire` → il pid sta dormendo (per costruzione del motore: un tick da
    sveglio ha sempre un evento) → risposta
    `grafo_fatto("dormire", nsubj=pid, **{"obl:tempo": lemma_numero(t)})`;
  - se il tick non ha evento e l'ultimo evento del pid NON è `dormire` →
    `ValueError` rumoroso (assunzione del motore violata, non nascondere);
  - se il tick non ha evento e NON esiste alcun evento del pid con `t' < t`
    → `NON_LO_SO` (inizio storia, stato ignoto).
- Misurato: ~14% dei tick sono continuazione di sonno → il caso «dorme» è
  frequente e prezioso (la risposta giusta non è mai copia di un evento).

### 2.4 `_genera_azione_luogo(storia, rng, n)`

- Candidati: i luoghi `L` in cui il pid ha ≥1 evento e in cui TUTTI i suoi
  eventi, una volta privati del tempo (`_grafo_evento_senza_tempo`), sono
  **grafi identici** (stessa azione, oggetto, destinatario, argomento,
  origine). Copre sia i luoghi visitati una volta sola sia le ripetizioni
  identiche («in bosco cerca sempre legna»).
- Domanda: `grafo_fatto("fare", nsubj=pid, **{"obl:luogo": L},
  quesito="che-cosa")`.
- Risposta: quel grafo evento senza tempo. Mai `NON_LO_SO` qui:
  - casi ambigui → non si chiedono (decisione 1);
  - luoghi mai visitati → non si chiedono: con cast 1 il lettore SA che il
    pid non ha fatto nulla lì (ogni suo tick è narrato), quindi non-lo-so
    sarebbe epistemicamente falso, e la risposta vera («niente») non esiste
    nel formato. Documentarlo nel docstring.
- Nota epistemica da docstring (stile nota "parentela"): i tipi nuovi hanno
  quota non-lo-so strutturalmente ~0 (`posizione_tempo`/`azione_tempo` solo
  il raro caso inizio-storia-nel-sonno); le domande non-lo-so del mix
  arrivano dal tipo `posizione` esistente (oggetti mai localizzati).

### 2.5 Punto d'ingresso e firma

```python
def genera_domande_tempo(storia, rng, n_per_tipo, n_tick) -> list[Domanda]
```

`n_tick` è passato dal chiamante (la `Storia` non lo porta e i tick di coda
possono essere senza eventi). I tipi si chiamano esattamente
`"posizione_tempo"`, `"azione_tempo"`, `"azione_luogo"`.

### 2.6 Test (in `tests/test_mondo.py` o file nuovo `tests/test_domande_tempo.py`)

1. **Verità via prefisso** (il test più importante): per ~50 seed × vari t,
   l'oro di `posizione_tempo` al tick t == `genera_storia(seed, n_tick=t,
   persone=cast).stato_finale.luogo_effettivo(pid)` (proprietà-prefisso già
   verificata nel progetto).
2. Risposta «dorme» sui tick di sonno; `ValueError` su storia artificiale
   che viola l'assunzione (costruire un `Storia` finto con un buco).
3. `azione_luogo`: luogo con eventi diversi → escluso; con eventi identici
   ripetuti → incluso, risposta senza nodo tempo.
4. Determinismo: stesso seed → stesse domande, byte per byte.
5. **Byte-identità dell'esistente**: `genera_domande` su alcuni seed produce
   esattamente gli stessi grafi di prima della modifica (confrontare con
   output registrato prima di toccare il file).

---

## 3. T2 — `lingua/`: lessico e stampi (approvato, cancello round-trip)

### 3.1 Lessico

- Aggiungere IN CODA a `lingua/lessico.tsv` (dopo l'ultima riga dati, con
  una riga di commento che spiega il perché):
  `che-cosa	INTERR	superficie=che cosa	-`
- **Motivo della coda, vincolante**: gli id di `vocabolario.json` seguono
  l'ordine di riga del lessico; una riga in coda lascia invariati gli id
  0..280 esistenti e i checkpoint vecchi restano leggibili. NON inserire
  nella sezione «Interrogativi» a metà file.
- Rigenerare `cervello/vocabolario.json` (lo sha del lessico cambia).
  **Test-cancello**: gli id dei token già esistenti sono identici a prima
  (caricare il vecchio JSON da git e confrontare), il nuovo token è l'ultimo.

### 3.2 Stampi (`lingua/stampi.py`) e inversi

Nuovi stampi di domanda (superficie al presente, decisione 8; per l'orario
riusare le regole di superficie esistenti di `prefisso_tempo`: «alle due»,
«all'una»):

- `trovarsi + quesito=dove + obl:tempo` → «Dove si trova Anna alle due?»
- `fare + quesito=che-cosa + obl:tempo` → «Che cosa fa Anna alle due?»
- `fare + quesito=che-cosa + obl:luogo` → «Che cosa fa Anna in cucina?»
  (usare il tratto `loc_in` del luogo, come già fanno gli stampi esistenti)

Nuovi stampi di risposta:

- `essere + nsubj + obl:luogo + obl:tempo` → «Alle due Anna è in cucina.»
  (prefisso tempo + stampo posizione esistente);
- risposta di `azione_tempo` = grafo evento → riusare la resa eventi
  esistente con un contesto di discorso fresco («Alle due Anna raccoglie una
  mela.»);
- risposta di `azione_luogo` = grafo evento SENZA tempo → variante della
  resa eventi senza prefisso orario («Anna raccoglie una mela nell'orto.»).

Ogni stampo nuovo ha il suo inverso nel parser (`analizza`/inversi in
stampi.py), ricostruendo i grafi con gli stessi ordini kwargs di §2 (grafi
byte-identici, convenzione di FASE1_PIANO). Aggiornare `formario`/`accordo`
dove serve.

### 3.3 Cancelli T2 (duri)

1. Round-trip **dei tipi nuovi**: ≥300 seed, per ogni domanda/risposta di
   `genera_domande_tempo`: grafo → testo → grafo identico.
2. Round-trip **storico invariato**: la verifica di sempre
   (`python -m lingua verifica` su migliaia di seed) resta 100,0000%.
3. Tutti i test esistenti verdi.

---

## 4. T3 — `esami/genera.py`: cast rotante, integrazione, split tracking-tempo

1. **`dataset.cast_rotante: true`** (errore se presente insieme a `cast`):
   il cast del seed `s` è `(dm.PERSONE[s % len(dm.PERSONE)],)`. Rifattorare
   i punti che oggi chiamano `_cast_persone(config)` in una
   `_cast_per_seed(config, seed)`; con la chiave assente il comportamento è
   quello di sempre (cancello byte-identità: rigenerare un campione di
   record `v1_grad1`/`v1_anti` e confrontare col contenuto attuale).
2. **Integrazione**: in `genera_record` (e nella generazione candidate di
   `genera_esame_tracking`-analogo, vedi punto 3), se uno dei tre tipi nuovi
   è in `tipi_ammessi`:
   `candidate += genera_domande_tempo(storia, random.Random(f"domande-tempo-{seed}"), n_per_tipo=n_candidate, n_tick=n_tick)`.
   RNG **separato** da `f"domande-{seed}"`: le estrazioni dei tipi esistenti
   non cambiano. I tipi nuovi passano dal percorso normale (filtro
   `tipi_ammessi`), NON dalla selezione anti-scorciatoia.
3. **Split diagnostico `tracking_tempo.jsonl`** (analogo A3, solo esame):
   domande `posizione_tempo` dove valgono TUTTE: oro ≠ luogo più frequente
   della storia; oro ≠ posizione finale del pid; `n_tick − t ≥ 3` (la
   risposta non è né la scorciatoia di frequenza, né lo stato finale, né
   roba fresca). Funzione + CLI `--split tracking-tempo`, stesso formato
   record; `esami/diagnosi.py` o `esamina` lo valutano come già fanno con
   `tracking.jsonl` (estendere il minimo indispensabile).
4. Test: byte-identità default; rotazione deterministica; sotto-insieme
   verificato per `tracking_tempo.jsonl` (stesse domande presenti in
   `esame.jsonl`, stesso pattern dei test A3).

---

## 5. T4 — Config, misura lunghezze, notebook

- `configs/v1_tempo.yaml`: come `v1_grad1.yaml` tranne — `nome_run:
  v1_tempo`; `dati_dir: dati/tempo`; `cast_rotante: true` (niente `cast`);
  `train_storie: 3000, dev_storie: 200, esame_storie: 200, n_per_tipo: 8`;
  stadio 1 `tipi: [posizione, posizione_tempo, azione_tempo, azione_luogo]`,
  `storie_corte: {min: 12, max: 24}`, `soglia: 0.95`; `training` come
  `v1_grad1` ma `max_step: 8000`, `intervallo_valutazione: 500`,
  `early_stop: false` (vogliamo la curva intera). Blocco `modello:`
  IDENTICO (stesse shape di sempre).
- `configs/v1_tempo_fumo.yaml`: 300 storie, `max_step: 300`, per il cancello
  di pipeline su Colab.
- **Misura prima del run** (in locale, stdlib): distribuzione delle
  lunghezze delle sequenze composte sul train — cancello: max < ctx 3072
  (atteso: ~12–24 eventi × ~15–20 token + domanda + risposta, ampio margine;
  `genera.py` comunque fallisce rumorosamente se sfora). Riportare anche la
  composizione per tipo e la quota non-lo-so del mix.
- `colab_training.ipynb`: **sezione 12** sul modello della sezione 10/11
  (fumo → run vero con `--copia-sicurezza` su Drive → esame +
  `tracking_tempo` + diagnosi, JSON copiati su Drive).

---

## 6. T5 — Run (Colab, lancia Andrea)

Workflow di sempre (commit+push, poi Colab; mai training in locale). Prima
il fumo (`v1_tempo_fumo`): cancello = pipeline verde, composizione per tipo
sensata, `esattezza_dev` sopra zero. Poi il run vero: stima ~1–1,5h (le
sequenze sono ~4–5× più corte di v1_anti a parità di step); se la stima
misurata dal fumo supera le 2h serve l'ok esplicito di Andrea (vincolo 5 del
piano di diagnosi).

---

## 7. T6 — Lettura dei risultati (criteri fissati PRIMA del run)

Riportare: esattezza per tipo (`esattezza_per_tipo` di `esamina`), split
`tracking_tempo` con margini sopra le due baseline (luogo più frequente,
posizione finale), calibrazione non-lo-so (sul tipo `posizione` del mix —
se degrada è una regressione da segnalare).

| Esito | Lettura | Conseguenza |
|---|---|---|
| `posizione_tempo` ≥ ~0,9 e `tracking_tempo` ben sopra le baseline | il tracking mono-entità si forma; il muro multi-entità è interferenza/binding | rafforza Fase B (stato denso) e C2 (slot entità); progettare B con questo dato |
| `azione_tempo` alta ma `posizione_tempo` bassa | sa fare lookup del tick, non propaga lo stato | segnale pro-ricorrenza (C1) / pro-B; A2 meno urgente |
| tutto basso anche a 1 entità | il tracking temporale non si forma a questa scala/budget | priorità A2 (capacità/budget) con fiducia; poi C1 |

In ogni caso: aggiornare §7 di `FASE2_PIANO_DIAGNOSI.md` (questo esperimento
vi compare come linea parallela) e lo «Stato di avanzamento» qui sotto.

---

## 8. Stato di avanzamento

- 2026-07-11: piano scritto e approvato da Andrea (decisioni §1 confermate
  via domande esplicite: ambiguità → non si chiede; da zero; esperimento a
  parte con mix; cast rotante; tocco lingua/ approvato; niente A2 in
  parallelo). Nessuna tappa ancora eseguita. Prossimo passo: T1.
