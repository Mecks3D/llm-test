# FASE 1 — La lingua (verbalizzatore + parser + filtro)

**Obiettivo**: il confine deterministico tra mondo e testo. Grafo → frase
italiana corretta, frase → grafo, senza perdita, sulla grammatica controllata.

**Modulo**: `lingua/`  — dipende da `mondo/` (per i tipi), niente ML.

## Il lessico — `lingua/lessico.tsv`

Una riga per lemma: `lemma  categoria  tratti  definizione_in_primitivi`

- Si parte con i ~300 lemmi che il micro-mondo usa davvero; obiettivo di
  crescita ~1.500–2.000. Il lessico è l'UNICA fonte del vocabolario: mondo,
  lingua e cervello lo leggono da qui.
- I 65 primitivi NSM sono le prime 65 righe, marcate `PRIM`. Ogni altro
  lemma ha una definizione composta di primitivi (può essere approssimativa;
  quelle mancanti si segnano `TODO`, non si inventano male).

## Verbalizzatore — `verbalizza(grafo) -> str`

- A regole: un template per predicato/azione + un motore morfologico per
  gli accordi. Copre SOLO ciò che il micro-mondo produce:
  - tempi: presente, passato prossimo, imperfetto;
  - articoli determinativi/indeterminativi con elisione (lo/l'/il/la…);
  - plurali regolari + i pochi irregolari del lessico (segnati nel lessico);
  - pronomi soggetto e clitici semplici solo se servono ai template.
- Vietato: gestire casi che il mondo non genera "per completezza". Ogni
  regola deve avere almeno una frase del corpus che la usa.

## Parser — `analizza(frase) -> grafo`

- Due implementazioni ammesse (scegliere UNA per la v1):
  a) inverso delle regole del verbalizzatore (consigliata: sulla grammatica
     chiusa è più semplice e fa 100% per costruzione);
  b) spaCy `it_core_news_lg` + mappatura verso il nostro schema di grafo.
- In entrambi i casi la validazione è identica: confronto con i grafi-verità
  del simulatore. Il parser non si giudica mai a occhio.

## Filtro-regole — `filtro.py`

- Lista di pattern di grafo vietati in un file di configurazione
  (`lingua/regole_filtro.txt`), applicata a input e output.
- Per ora la lista è minima (2–3 pattern segnaposto): conta la sede
  architetturale, non il contenuto. Non espanderla in questa fase.

## Criteri di accettazione

- **Giro completo**: per 10.000 frasi generate, `analizza(verbalizza(g)) == g`
  esattamente, ≥ 99,9% (target 100%; ogni fallimento va capito, non tollerato).
- Ogni frase prodotta passa un controllo di accordo automatico (genere/numero
  soggetto-verbo-articolo) — test dedicato.
- Lettura umana: 100 frasi campione stampate per revisione manuale di Andrea
  (le frasi devono suonare da italiano semplice, non da traduzione automatica).
- Nessuna frase generata contiene lemmi fuori dal lessico.

## Trappole note

- Non usare un LLM come verbalizzatore o parser "temporaneo": il confine
  deve restare deterministico e ispezionabile (principio 1).
- Non fondere verbalizzatore e simulatore: `mondo/` non importa `lingua/`.
- La morfologia italiana completa è un pozzo senza fondo: fermarsi alla
  grammatica controllata è una scelta di progetto, non una scorciatoia.
