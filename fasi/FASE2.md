# FASE 2 — Il cervello (v1: float, autoregressivo sul grafo)

**Obiettivo**: una rete ~10M parametri che legge storie come grafi
linearizzati e risponde alle domande del curriculum. Prima in float
(2a), poi ternaria (2b). Dipendenze: PyTorch. Modulo: `cervello/`.

## Linearizzazione dei grafi — `cervello/sequenza.py`

Il grafo diventa una sequenza di token con visita in profondità:

```
[STORIA] ( mangiare passato ( sogg bambino plur ) ( ogg mela plur ) ) ...
[DOMANDA] ( trovarsi presente ( sogg mela ) ( luogo [MASK] ) )
[RISPOSTA] cucina [FINE]
```

- Vocabolario = 65 primitivi (id 0–64) + lemmi del lessico + relazioni UD
  usate + tratti + token speciali. Ordine stabile, salvato in
  `cervello/vocabolario.json`, generato DAL lessico (mai a mano).
- Totale atteso ≈ 2.500–3.500 token. Niente BPE, niente sottoparole:
  un lemma = un token. Il tokenizer del progetto è il parser della Fase 1.
- Round-trip testato: grafo → sequenza → grafo esatto.

## Modello v1 — `cervello/modello.py`

Decoder-only transformer, ricetta nanoGPT senza variazioni creative:

- n_layer=8, n_head=8, d_model=256, ctx=512, pre-LayerNorm, GELU,
  dropout 0.1, embedding legati all'output. ≈ 8–10M parametri.
- Training: cross-entropy next-token su `[STORIA][DOMANDA][RISPOSTA]`,
  con la loss **mascherata sulla sola parte risposta** (la storia è data,
  non va imparata a pappagallo).
- AdamW, lr 3e-4 con cosine decay, batch a gradiente accumulato se la
  GPU è piccola. Deve girare anche su CPU per i run di debug.
- Log: loss, esattezza sulla risposta, token/sec — su file, non solo stdout.

## Curriculum e esami — `esami/`

- Un dataset per stadio (vedi PROGETTO.md), generato con seed disgiunti
  train/esame. Si passa allo stadio successivo solo con esame superato:
  stadi 1–2: ≥ 95% di risposte esatte; stadi 3–4: ≥ 90%.
- **Esattezza = confronto di grafi**, non di stringhe: la risposta generata
  si riconverte in grafo e si confronta con il grafo-verità.
- Le definizioni dei primitivi ("che cosa significa X?") entrano nel
  training dello stadio 1 come dati normali.
- Regola assoluta: mai addestrare su seed d'esame. Il generatore rifiuta
  di produrre training set con seed nel range riservato agli esami.

## Fase 2b — ternarizzazione

Solo DOPO che la v1 float passa gli stadi 1–3:

- Sostituire `nn.Linear` con `BitLinear`: pesi ternarizzati con soglia
  absmean (W → {−1,0,+1} · scala), attivazioni quantizzate a 8 bit absmax,
  gradiente con straight-through estimator. Pesi ombra in float durante il
  training; RMSNorm prima di ogni BitLinear.
- Ri-addestrare da zero (non convertire il modello float: la ricetta
  BitNet funziona in quantization-aware training, non in post-quantizzazione).
- Consegna: tabella di confronto float vs ternario sugli stessi esami +
  percentuale di pesi a zero + stima di consumo (moltiplicazioni eliminate).

## Criteri di accettazione (fase 2a)

- Esami stadi 1–3 superati con le soglie sopra, su seed mai visti.
- Un run completo di training riproducibile da un comando solo
  (`python -m cervello.addestra --config configs/v1.yaml`).
- Curva di loss e risultati d'esame salvati in `dati/risultati/`.

## Trappole note

- Non aggiungere trucchi architetturali alla v1 (niente Mamba, MoE, RoPE
  esotici): la v1 è la linea di base, la sua virtù è essere noiosa.
- Se un esame non passa, prima si guardano i DATI (bug nel simulatore o nel
  parser?), poi il modello. Con etichette perfette, un fallimento è quasi
  sempre un bug a monte o un curriculum troppo brusco.
- Non valutare mai su testo con confronto di stringhe: sempre grafo vs grafo.
