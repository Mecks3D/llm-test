# PIANO ESECUTIVO — FASE 2a: `cervello/` + `esami/` (v1 float, autoregressivo sul grafo)

Documento esecutivo per l'agente che implementa la Fase 2a. Prima di scrivere
codice, leggere nell'ordine: `CLAUDE.md`, `PROGETTO.md`, `fasi/FASE2.md`.
Questo piano **specializza** quei documenti: dove questo piano è più
dettagliato, si segue questo piano; se sembra **contraddirli**, fermarsi e
chiedere ad Andrea, non improvvisare. Le scelte di design qui dentro sono già
state discusse e approvate da Andrea (2026-07-07): non riaprirle e non
"migliorarle" in corsa.

La Fase 2b (ternarizzazione) è FUORI da questo piano: avrà un piano suo dopo
che la v1 float supera gli esami degli stadi 1–3.

---

## 0. Contesto in due paragrafi

Il progetto separa bordi deterministici (Fase 0: `mondo/` genera storie come
`Evento` e domande/risposte come grafi UD; Fase 1: `lingua/` fa
grafo ↔ frase italiana con round-trip 100%) da un centro appreso. La Fase 2a
costruisce il centro: un transformer decoder-only ~7M parametri (ricetta
nanoGPT, nessuna variazione creativa) che legge la storia come grafi
linearizzati e genera il grafo-risposta, addestrato con un curriculum a
stadi e promosso solo superando esami su seed mai visti.

Il criterio che governa tutto: **la valutazione è sempre grafo vs grafo,
mai stringa vs stringa** (regola non negoziabile #4). La risposta generata
dal modello è una sequenza di token che si riconverte in `Grafo` e si
confronta con `==` al grafo-verità. E la regola gemella (#3): **mai
addestrare su seed d'esame** — i seed ≥ 1.000.000 sono riservati agli esami.

## 1. Decisioni vincolanti (già prese con Andrea — non rimetterle in discussione)

1. **ctx = 3072** (non 512: misurato sui dati reali, le sequenze complete
   vanno da ~1.100 a ~2.700 token; FASE2.md è già stato corretto).
2. **Curriculum della 2a a 3 stadi** (mappa pragmatica del curriculum di
   PROGETTO.md sui tipi di domanda esistenti in `mondo/domande.py`):
   - **stadio 1** — storie CORTE (3–6 tick), solo domande `posizione`;
   - **stadio 2** — storie piene (8–22 tick, `_lunghezza_storia`), solo `posizione`;
   - **stadio 3** — storie piene, tipi `possesso`, `conteggio`, `transfer`,
     `parentela`.
   Soglie d'esame: stadi 1–2 ≥ 95%, stadio 3 ≥ 90% di risposte esatte.
   Gli stadi 4 (`causa` + raccolta) e 5 (`deduzione`) NON servono per
   l'accettazione della 2a: la pipeline deve essere generica per stadio
   (guidata dal config), ma i run degli stadi 4–5 sono lavoro successivo.
   Le definizioni NSM per lo stadio 1 sono RIMANDATE (sezione 12).
3. **Istanze di risorsa nei grafi (`mela_2`, `legna_7`…): si scompongono in
   lemma + ordinale**, sempre e incondizionatamente (anche `mela_1` →
   `mela primo`), con l'ordinale maschile base del lessico (categoria `ORD`).
   Motivo: il vocabolario è generato dal lessico e gli id-istanza non ne
   fanno parte; è lo stesso principio del fix dei nodi NUM (niente cifre
   grezze) e rispecchia ciò che fa già il verbalizzatore ("la seconda mela").
   A differenza della Fase 1, qui NON c'è contesto di discorso: la
   scomposizione è context-free e il round-trip è locale al singolo grafo.
4. **La risposta è un grafo completo linearizzato**, mai un token secco:
   `[RISPOSTA] ( essere ( nsubj mela primo ) ( obl:luogo cucina ) ) [FINE]`.
   "Non lo so" è `( non-lo-so )`. Così la valutazione resta grafo vs grafo
   senza casi speciali.
5. **Loss mascherata sulla sola risposta**: cross-entropy next-token solo
   sulle posizioni dal primo token dopo `[RISPOSTA]` fino a `[FINE]`
   incluso. Storia e domanda sono date, non vanno imparate a pappagallo.
6. **Decodifica d'esame: greedy (argmax), deterministica.** Niente
   temperatura, niente sampling.
7. **Niente trucchi architetturali**: nanoGPT liscio (pre-LN, GELU,
   embedding posizionali appresi, weight tying). Niente RoPE, Mamba, MoE.
   La virtù della v1 è essere noiosa (FASE2.md).
8. **`mondo/` e `lingua/` non si toccano.** Il rifiuto dei seed d'esame
   vive in `esami/genera.py`, che è l'UNICO punto d'ingresso ammesso per
   scrivere dataset (non si chiama `mondo.generatore.scrivi_dataset`
   direttamente). Se qualcosa in `mondo/` o `lingua/` sembra sbagliato,
   FERMARSI e chiedere ad Andrea.
9. **Parametri attesi ~7,3M**, non 8–10M: il delta rispetto a FASE2.md
   viene dal vocabolario piccolo (~350 token). NON gonfiare il modello per
   arrivare a 10M.
10. **Prima di ogni run lungo si misura e si chiede**: dopo il run di fumo
    si riportano i token/sec reali e la stima di durata del curriculum
    completo; run > 2 ore non si lanciano senza l'ok di Andrea.

## 2. Struttura dei moduli

```
cervello/
  __init__.py        # esporta l'API pubblica
  vocabolario.py     # genera/carica vocabolario.json DAL lessico
  vocabolario.json   # artefatto generato, COMMITTATO, con test di idempotenza
  sequenza.py        # grafo <-> lista di token (stdlib puro, niente torch)
  modello.py         # il transformer (torch)
  dati.py            # JSONL -> batch di tensori con maschera loss (torch)
  addestra.py        # training loop + orchestrazione curriculum (torch)
esami/
  __init__.py
  genera.py          # scrive i dataset per stadio/split (stdlib puro)
  esamina.py         # esame: decodifica greedy + confronto grafo vs grafo (torch)
configs/
  v1.yaml            # iperparametri + curriculum + percorsi
dati/                # generati, MAI committati
  stadio1/{train,dev,esame}.jsonl   (idem stadio2, stadio3)
  risultati/<nome_run>/             (log, checkpoint, esiti esami, copia config)
tests/
  test_cervello.py
  test_esami.py
```

Dipendenze e regole di import: `cervello/` e `esami/` possono importare
`mondo/` e `lingua/` (mai il contrario). PyTorch SOLO in
`cervello/modello.py`, `cervello/dati.py`, `cervello/addestra.py`,
`esami/esamina.py`: `vocabolario.py`, `sequenza.py` e `esami/genera.py`
restano stdlib puro, così i loro test girano senza torch. Nuove dipendenze
ammesse nella `.venv`: `torch` e `pyyaml` (per i config), nient'altro.
Identificatori e docstring in italiano. Nessun `random` globale; ogni RNG
con seed esplicito. Determinismo: `torch.manual_seed` dal config +
`torch.use_deterministic_algorithms(True)` dove possibile (su CPU il run è
riproducibile byte per byte; su GPU documentare nel log eventuali fonti di
non-determinismo residue).

## 3. Vocabolario — `cervello/vocabolario.py` + `vocabolario.json`

Generato dal lessico (`lingua.lessico.carica_lessico()`), mai a mano.
**Ordine normativo degli id**:

1. **id 0–64**: i 65 lemmi `PRIM` nell'ordine delle righe del lessico
   (è il contratto di PROGETTO.md: i primitivi occupano i token id 0–64).
2. **id 65–71**: token speciali, in quest'ordine:
   `[PAD]`, `[STORIA]`, `[DOMANDA]`, `[RISPOSTA]`, `[FINE]`, `(`, `)`.
3. **id 72–86**: le 15 relazioni UD usate dai grafi, in ordine alfabetico:
   `advcl:causa`, `iobj`, `nmod:agente`, `nmod:destinatario`,
   `nmod:oggetto`, `nmod:parentela`, `nmod:relativo`, `nsubj`, `obj`,
   `obl:argomento`, `obl:luogo`, `obl:origine`, `obl:quantita`,
   `obl:tempo`, `quesito`.
4. **id 87–…**: tutti gli altri lemmi del lessico (OGNI riga non-PRIM,
   comprese FUNZ e ORD), nell'ordine delle righe del file.

`vocabolario.json`: `{"token": [...], "versione_lessico": "<sha256 di
lessico.tsv>"}`. API: `genera_vocabolario() -> Vocabolario`,
`carica_vocabolario(percorso=…) -> Vocabolario` (con controllo che lo
sha256 del lessico corrente coincida: se no, `ValueError` che dice di
rigenerare), `Vocabolario.id(token) -> int`, `.token(id) -> str`,
`.dimensione`. Il file si committa; un test verifica che rigenerarlo
produca byte identici.

## 4. Linearizzazione — `cervello/sequenza.py`

Grammatica normativa della sequenza (un grafo alla volta):

```
grafo    := ( lemma_radice ramo* )
ramo     := ( relazione lemma [ordinale] )
```

- L'ordine dei rami = l'ordine di `grafo.archi` (che è l'ordine di
  costruzione: per gli eventi quello di `evento_a_grafo`, per i fatti
  l'ordine dei kwargs di `grafo_fatto`).
- Nodo con lemma-istanza `lemma_N` → due token: `lemma` + ordinale
  maschile di valore N dal lessico (`mela_2` → `mela secondo`). Sempre,
  anche per N=1. N > 30 (oltre gli ORD del lessico) → `ValueError`
  (empiricamente non accade mai su 10.000 seed; se accade è un bug a monte).
- Ogni altro nodo → un token: il lemma.
- `NON_LO_SO` → `( non-lo-so )`.
- Golden normativo (evento `t=9, andare, agente=sara, origine=cucina,
  destinazione=giardino`):
  `( andare ( nsubj sara ) ( obl:origine cucina ) ( obl:luogo giardino ) ( obl:tempo nove ) )`
  e domanda `grafo_fatto("trovarsi", nsubj="mela_1", quesito="dove")`:
  `( trovarsi ( nsubj mela primo ) ( quesito dove ) )`.

API (firme esatte):

```python
def grafo_a_token(grafo: Grafo) -> list[str]: ...
def token_a_grafo(token: Sequence[str], famiglia: str) -> Grafo: ...
    # famiglia: "evento" | "fatto"; ValueError con messaggio chiaro se malformata
def componi_esempio(storia: Sequence[Sequence[str]], domanda: Sequence[str],
                    risposta: Sequence[str]) -> list[str]: ...
    # [STORIA] <grafi eventi concatenati> [DOMANDA] <grafo> [RISPOSTA] <grafo> [FINE]
```

Come in Fase 1, **mai costruire nodi/archi a mano** nella ricostruzione:

- `famiglia="evento"`: dai rami si ricava un
  `Evento(t=valore del lemma NUM (tratto `valore` nel lessico), azione=radice,
  agente=nsubj, oggetto=obj, destinatario=iobj, argomento=obl:argomento,
  luogo_origine=obl:origine, luogo=obl:luogo)` — lemma+ordinale si ricompone
  in `f"{lemma}_{N}"` — e si ritorna `evento_a_grafo(evento)`.
- `famiglia="fatto"`: si chiama `grafo_fatto(radice, **kwargs)` con i kwargs
  nell'ORDINE di apparizione dei rami (l'ordine determina gli id dei nodi:
  così il round-trip è byte-identico); la ricomposizione lemma+ordinale →
  `lemma_N` vale anche qui (es. transfer con `obj="mela_2"`).
  `( non-lo-so )` → costante `NON_LO_SO`.
- Round-trip di riferimento nei test:
  `token_a_grafo(grafo_a_token(g), famiglia) == g` per OGNI grafo del mondo
  (eventi, domande, risposte).

## 5. Dataset — `esami/genera.py` e formato JSONL

**Finestre di seed normative** (disgiunte per costruzione; `s` = stadio):

| split | seed | vincolo |
|---|---|---|
| train  | `100_000·(s−1) + i`, `i` in `[0, train_storie)` | < 1.000.000, rifiuto altrimenti |
| dev    | `800_000 + 10_000·(s−1) + i`, `i` in `[0, dev_storie)` | < 1.000.000 |
| esame  | `1_000_000 + 10_000·(s−1) + i`, `i` in `[0, esame_storie)` | ≥ 1.000.000, rifiuto altrimenti |

`genera.py` **rifiuta** (eccezione con messaggio chiaro, niente file
scritto) un train/dev con qualunque seed ≥ 1.000.000 e un esame con seed
< 1.000.000: è il requisito "il generatore rifiuta" di FASE2.md. Il dev
serve alle valutazioni durante il training; l'esame si usa SOLO al cancello
di fine stadio.

Generazione per storia: stadio 1 → `n_tick = random.Random(f"stadio1-{seed}")
.randint(3, 6)`; stadi ≥ 2 → `n_tick = mondo.generatore._lunghezza_storia(seed)`.
Poi `genera_storia(seed, n_tick)` e `genera_domande(storia,
random.Random(f"domande-{seed}"), n_per_tipo=8)`, tenendo SOLO i tipi dello
stadio (sezione 1.2). Le quote "non lo so" restano quelle del generatore:
non filtrarle né bilanciarle.

Formato JSONL (un record per storia, token come stringhe — leggibile e
compatto; la conversione in id avviene al caricamento):

```json
{"stadio": 1, "seed": 12, "storia": ["(", "andare", ...],
 "esempi": [{"tipo": "posizione", "domanda": ["(", ...], "risposta": ["(", ...]}]}
```

Vincoli verificati alla scrittura: ogni sequenza composta
(`componi_esempio`) ≤ ctx del config (fallire rumorosamente, non troncare);
ogni token presente nel vocabolario; determinismo (stesso config → file
byte-identici, testato).

CLI: `python -m esami.genera --config configs/v1.yaml --stadio 1 --split train`
(e `dev`/`esame`; senza `--split` li scrive tutti e tre) →
`dati/stadio1/train.jsonl` ecc.

## 6. Modello — `cervello/modello.py`

Ricetta nanoGPT, senza variazioni:

- `n_layer=8, n_head=8, d_model=256, d_ff=1024, ctx=3072`, vocab dal
  vocabolario. Pre-LayerNorm, GELU, dropout 0.1 (attn e residui),
  embedding posizionali APPRESI, embedding di input legato alla testa di
  output (weight tying). Attenzione causale con
  `F.scaled_dot_product_attention(is_causal=True)`.
- Init nanoGPT: normale (0, 0.02), proiezioni residue scalate
  `0.02/sqrt(2·n_layer)`, bias a zero.
- ≈ 7,3M parametri (contati in un test; non gonfiare, vedi decisione 9).
- `forward(idx) -> logits`; il modello non sa nulla di maschere di loss
  (vivono in `dati.py`/`addestra.py`). Deve girare su CPU.
- Il padding a destra con `[PAD]` non richiede maschera d'attenzione: con
  la maschera causale i pad non influenzano le posizioni precedenti e la
  loss sui pad è già esclusa dalla maschera risposta.

## 7. Dati e training — `cervello/dati.py`, `cervello/addestra.py`, `configs/v1.yaml`

`dati.py`: carica i JSONL, compone con `componi_esempio`, converte in id,
mescola con RNG seedato per epoca, impacchetta batch paddati alla sequenza
più lunga del batch, produce `(input, bersaglio, maschera)` dove la maschera
è vera solo dal primo token dopo `[RISPOSTA]` a `[FINE]` inclusi.

`addestra.py` — training di uno stadio:

- AdamW: lr 3e-4, betas (0.9, 0.95), weight decay 0.1 solo sui pesi 2D;
  warmup lineare 200 step poi cosine decay a lr/10; grad clip 1.0;
  accumulo di gradiente dal config (batch piccoli se la memoria è poca).
- Ogni `intervallo_valutazione` step (default 500): loss di train, loss dev,
  **esattezza-risposta su dev** = decodifica greedy su `dev_campione`
  esempi (default 200) e confronto grafo vs grafo (riusare la logica di
  `esamina.py`, non duplicarla). Ferma lo stadio quando l'esattezza dev
  ≥ soglia dello stadio + 1 punto per 2 valutazioni consecutive, o a
  `max_step` (default 20.000).
- Log su file `dati/risultati/<nome_run>/log.jsonl` (una riga per
  valutazione: step, stadio, loss train/dev, esattezza dev, token/sec) +
  stdout; checkpoint `stadio<N>.pt`; copia del config nel run.

**Orchestrazione curriculum** (il "comando solo" di FASE2.md):
`python -m cervello.addestra --config configs/v1.yaml` esegue in sequenza
stadio 1 → esame 1 → stadio 2 → esame 2 → stadio 3 → esame 3, fermandosi al
primo esame fallito (exit code ≠ 0). Il training dello stadio N parte dal
checkpoint dello stadio N−1 e usa come train la CONCATENAZIONE dei train
degli stadi ≤ N (ripasso: si aggiunge, non si sostituisce). Con
`--stadio N` esegue solo quello stadio (per debug/ripresa). Il seed di
torch e quello del mescolamento vengono dal config.

`configs/v1.yaml` — tutte le scelte numeriche vivono qui, con questi
default: `train_storie: 5000`, `dev_storie: 300`, `esame_storie: 300`,
`n_per_tipo: 8`, `batch: 4`, `accumulo: 8`, `max_step: 20000`,
`intervallo_valutazione: 500`, `dev_campione: 200`, `seed_torch: 1337`,
`nome_run: v1`, `device: auto`, più gli iperparametri della sezione 6 e le
soglie/tipi per stadio della sezione 1.2.

## 8. Esame — `esami/esamina.py`

`python -m esami.esamina --config configs/v1.yaml --stadio N --checkpoint
dati/risultati/v1/stadioN.pt`:

1. Per ogni esempio dell'esame: si dà al modello
   `[STORIA]…[DOMANDA]…[RISPOSTA]` e si decodifica greedy fino a `[FINE]`
   (tetto: `ctx`).
2. I token generati si riconvertono con `token_a_grafo(…, "fatto")`;
   una sequenza malformata NON fa crashare: conta come risposta errata
   di categoria `malformata`.
3. Esatto ⇔ grafo generato `==` grafo-verità (ricostruito con la stessa
   `token_a_grafo` dai token del dataset, così il confronto è omogeneo).
4. Esito in `dati/risultati/<nome_run>/esame_stadio<N>.json`: esattezza
   totale e per tipo, numero di esempi, e le metriche di calibrazione
   dell'onestà epistemica (PROGETTO.md): `invenzioni` (oro = non-lo-so ma
   il modello risponde altro), `astensioni_errate` (oro determinabile ma il
   modello dice non-lo-so), `malformate`. Le soglie di passaggio restano
   sull'esattezza; le metriche di calibrazione si riportano sempre.
5. Exit code 0 solo se l'esattezza ≥ soglia dello stadio.

## 9. Test — `tests/test_cervello.py`, `tests/test_esami.py`

I test che richiedono torch vanno marcati (`@pytest.mark.torch`) e saltati
con un messaggio chiaro se torch non è installato. Gruppi richiesti:

1. **Vocabolario**: PRIM = id 0–64 nell'ordine del lessico; speciali e
   relazioni agli id normativi (sezione 3); rigenerazione byte-identica al
   `vocabolario.json` committato; sha256 del lessico verificato (lessico
   alterato → `ValueError`).
2. **Sequenza**: i due golden normativi della sezione 4 confrontati
   token per token; round-trip `token_a_grafo(grafo_a_token(g)) == g` su
   eventi + domande + risposte dei seed 0–299 (tutti i tipi); istanze
   (`mela_2` → `mela secondo` e ritorno); `NON_LO_SO`; ogni token prodotto
   esiste nel vocabolario; sequenze malformate (parentesi sbilanciate,
   relazione ignota, ordinale orfano) → `ValueError` chiaro.
3. **Genera**: rifiuto seed (train con seed ≥ 1.000.000 e esame con seed
   < 1.000.000 → eccezione, nessun file); determinismo byte per byte;
   stadio 1 con storie 3–6 tick e sole domande posizione; ogni sequenza
   composta ≤ ctx; finestre di seed conformi alla tabella.
4. **Dati** (torch): maschera di loss su un esempio golden (vere SOLO le
   posizioni della risposta fino a `[FINE]`); padding corretto.
5. **Modello** (torch): forward con shape attese; conteggio parametri
   ~7,3M (±5%); determinismo (stesso seed → stessi logits due volte).
6. **Canary di apprendimento** (torch, marcato `slow`): 32 esempi di
   stadio 1, ≤ 1.000 step su CPU → esattezza-risposta 100% in overfit.
   Se non ci arriva, qualcosa è scollegato: NON proseguire alle tappe dopo.
7. **Esamina** (torch): con risposte iniettate (senza modello) il conteggio
   di esatte/invenzioni/astensioni_errate/malformate è giusto; confronto
   sempre grafo vs grafo (mai stringhe).

## 10. Tappe di lavoro (con cancelli)

Procedere in quest'ordine; non passare alla tappa successiva finché il
cancello non è verde. Al termine di ogni tappa proporre un commit
(`git add` mirato + messaggio "Fase 2a: …").

- **T1** `vocabolario.py` + `vocabolario.json` + test gruppo 1.
- **T2** `sequenza.py` + test gruppo 2 (round-trip 0–299 al 100%).
- **T3** `esami/genera.py` + sezione dataset di `configs/v1.yaml` + test
  gruppo 3; generare `dati/stadio1/` e riportare dimensioni file e
  distribuzione lunghezze sequenze.
- **T4** installare torch nella `.venv` (CPU o CUDA a seconda della
  macchina; riportare quale); `modello.py` + `dati.py` + test gruppi 4–5.
- **T5** canary di apprendimento (gruppo 6). Cancello duro: 100% in
  overfit prima di andare avanti.
- **T6** `addestra.py` completo + `esami/esamina.py` + test gruppo 7 +
  run di fumo (stadio 1 con `train_storie: 200`, `max_step: 300`): la
  loss deve scendere e il log/checkpoint devono comparire in
  `dati/risultati/`. Riportare i token/sec e la STIMA di durata del
  curriculum completo con i default: se supera ~2 ore, fermarsi e
  chiedere ad Andrea prima di lanciare (decisione 10).
- **T7** run vero: `python -m cervello.addestra --config configs/v1.yaml`
  (stadi 1→3 con esami ai cancelli). Consegna finale per Andrea: tabella
  esattezza per stadio e per tipo + invenzioni/astensioni/malformate +
  curva di loss (percorsi dei file in `dati/risultati/v1/`).

Criteri di accettazione finali = quelli di FASE2.md: esami stadi 1–3
superati (≥95/95/90%) su seed mai visti; run riproducibile da un comando
solo; curve e risultati salvati in `dati/risultati/`.

## 11. Trappole note (da FASE2.md, più quelle di questo progetto)

- Se un esame non passa, prima si guardano i DATI (bug in genera.py? nel
  curriculum troppo brusco?), poi il modello: con etichette perfette un
  fallimento è quasi sempre a monte.
- Mai valutare su testo: il confronto è `Grafo == Grafo`, sempre.
- Mai "provare" o addestrare su seed ≥ 1.000.000, nemmeno per debug.
- `dati/` non si committa (né dataset né checkpoint); `vocabolario.json`
  invece SÌ (è un artefatto normativo con test di idempotenza).
- Non ottimizzare prematuramente il training (niente compile, niente AMP
  nella v1): prima corretto, poi veloce; il collo di bottiglia si misura
  con i token/sec nel log.

## 12. Rimandato esplicitamente (non farlo ora)

- **Definizioni NSM nello stadio 1**: aspettano che Andrea completi le
  definizioni TODO del lessico e che si definisca una forma-grafo per le
  definizioni (decisione di Andrea 2026-07-07).
- **Stadi 4–5** (`causa`+raccolta, `deduzione`): la pipeline li supporta
  via config, ma i run sono lavoro successivo all'accettazione della 2a.
- **Fase 2b** (BitLinear, ternarizzazione): piano separato, solo dopo gli
  esami 1–3.
- Stadio 6 (teoria della mente), domande "quando/prima/dopo" (non esiste
  il generatore in `mondo/domande.py`), ibrido Mamba, diffusione (Fase 3).
