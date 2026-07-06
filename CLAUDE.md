# Cervello-Bambino

Modello linguistico minimale ispirato allo sviluppo di un bambino.
**Leggere prima `PROGETTO.md`** (visione e principi), poi la specifica della
fase su cui si lavora in `fasi/FASE<N>.md`. In caso di conflitto tra codice e
documenti, i documenti vincono; se una specifica sembra sbagliata, chiedere,
non improvvisare.

## Regole non negoziabili

1. Nessun modello o embedding pre-addestrato (niente SONAR, niente LLM per
   generare dati): il bambino nasce vuoto. Unica eccezione ammessa: spaCy
   come parser in Fase 1, se si sceglie quella strada.
2. Determinismo: ogni generazione casuale riceve un RNG con seed esplicito;
   stesso seed → stesso output byte per byte. Mai `random` globale.
3. Mai addestrare su seed riservati agli esami.
4. La valutazione è sempre grafo vs grafo, mai stringa vs stringa.

## Convenzioni

- Python ≥ 3.11, dipendenze minime (stdlib per `mondo/` e `lingua/`,
  PyTorch solo in `cervello/`).
- Identificatori e docstring in italiano (il dominio è italiano: `verbalizza`,
  `evento_a_grafo`), termini tecnici ML in inglese dove è l'uso comune.
- Test con pytest in `tests/`; ogni modulo nuovo arriva con i suoi test.
- Moduli: `mondo/` (simulatore), `lingua/` (verbalizzatore+parser+filtro),
  `cervello/` (modello), `esami/` (valutazione), `dati/` (generati, mai
  committati), `fasi/` (specifiche).
- `mondo/` non importa `lingua/`; `lingua/` non importa `cervello/`.
