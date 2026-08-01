# Cervello-Bambino

Modello linguistico minimale ispirato allo sviluppo di un bambino.
**Leggere prima `PROGETTO.md`** (visione e principi), poi la specifica della
fase su cui si lavora. In caso di conflitto tra codice e documenti, i documenti
vincono; se una specifica sembra sbagliata, chiedere, non improvvisare.

**Fase corrente: `fasi/FASE_MENTE.md`** — credenza persistente su entità e
relazioni, alimentata da percezione (telecamere) e lingua. Le fasi precedenti
(`FASE0`, `FASE1`, `FASE2*`, `FASE_JEPA_PIANO`) sono **storia**: si leggono per
capire perché siamo qui, non si eseguono. Tag `fase-b-finale` e `jepa-chiuso`.

## Regole non negoziabili

1. Nessun modello o embedding pre-addestrato (niente SONAR, niente LLM per
   generare dati): il bambino nasce vuoto. Unica eccezione ammessa: spaCy
   come parser in Fase 1, se si sceglie quella strada.
2. Determinismo: ogni generazione casuale riceve un RNG con seed esplicito;
   stesso seed → stesso output byte per byte. Mai `random` globale.
3. Mai addestrare su seed riservati agli esami.
4. La valutazione è sempre grafo vs grafo, mai stringa vs stringa.
5. **Ogni numero riportato ha accanto il numero del riferimento simbolico
   `regole/` sulla stessa sonda.** Se la rete non lo batte, non c'è risultato.
   Vale anche per le misure intermedie (`FASE_MENTE.md` §9).
6. Le due scale di valutazione — **narrativa** (oro di `mondo/domande.py`) e
   **reale** (stato vero del mondo) — non si mescolano mai in un numero solo.
   Confonderle fa sembrare che aggiungere la telecamera peggiori le cose
   (`FASE_MENTE.md` §12.1).

## Convenzioni

- Python ≥ 3.11, dipendenze minime (stdlib per `mondo/`, `lingua/`,
  `percezione/`, `regole/`, `sonde/`; PyTorch solo in `cervello/` e `mente/`).
- Identificatori e docstring in italiano (il dominio è italiano: `verbalizza`,
  `evento_a_grafo`), termini tecnici ML in inglese dove è l'uso comune.
- Test con pytest in `tests/`; ogni modulo nuovo arriva con i suoi test.
- Determinismo: anche `torch.manual_seed` esplicito, non solo gli RNG stdlib.

### Moduli

| modulo | stato | ruolo |
|---|---|---|
| `mondo/` | vivo | simulatore del micro-mondo |
| `lingua/` | vivo | verbalizzatore + parser + filtro |
| `percezione/` | vivo | formato osservazione, telecamera sintetica, rumore |
| `regole/` | vivo | riferimento simbolico — il tetto da battere |
| `sonde/` | vivo | banco di misura, sette sonde |
| `mente/` | da scrivere | il modello della fase corrente |
| `cervello/`, `esami/`, `configs/` | congelati | Fase B; restano solo perché `sonde/adattatori.py::esiti_v1` li importa |
| `jepa/` | congelato | esperimento chiuso; usato solo da `--con-jepa` |
| `archivio/` | morto | notebook e materiale di fasi concluse |
| `dati/` | generato | mai committato |

Dipendenze consentite: `mondo/` non importa `lingua/`; `lingua/` non importa
`cervello/`; **`mente/` non importa mai `mondo/`** — vede solo
`percezione/tipi.py` e i grafi della lingua. È la cucitura sim-to-real: quando
arriva la telecamera vera cambia solo il backend di `percezione/`.

## Dove si esegue

- `mondo/`, `lingua/`, `percezione/`, `regole/`, `sonde/`: stdlib, locale, secondi.
- `mente/` (M1, ~10⁵ parametri): **locale finché una run sta sotto i 5 minuti**.
- Oltre i 5 minuti, o qualsiasi cosa tocchi i checkpoint di `cervello/`:
  commit + push + Colab, mai in locale.
