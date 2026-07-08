# PIANO ESECUTIVO — Interfaccia grafica dimostrativa (visualizza il sistema end-to-end)

Documento scritto per essere eseguito in una conversazione pulita da un
modello meno capace, come `FASE1_PIANO.md`/`FASE2_PIANO.md`. Non è
normativo per il curriculum (non tocca esami/soglie/dataset): è un
**nuovo strumento**, indipendente dal resto, per osservare interattivamente
cosa fa il sistema — utile sia come demo sia come strumento di debug
qualitativo (finora fatto con script di scratchpad usa-e-getta).

---

## 0. Obiettivo

Un'applicazione desktop con finestra grafica che permette di:

1. generare una storia (scegliendo seed, cast, lunghezza),
2. vederla in italiano leggibile (non in token grezzi),
3. scegliere una domanda (tra quelle derivabili dalla storia),
4. far rispondere il modello (checkpoint già addestrato, caricato da file),
5. vedere la risposta del modello in italiano, la risposta corretta, e se
   coincidono — con la possibilità di vedere anche i token grezzi per chi
   vuole guardare "sotto il cofano".

Non genera nulla di nuovo: **riusa** i moduli già scritti (mondo/, lingua/,
cervello/, esami/), non duplica logica.

## 1. Perché ora e perché ha senso indipendentemente dall'esito del curriculum

Serve sia per mostrare il sistema (demo) sia per esplorare a occhio gli
errori del modello — quello che finora si è fatto con script Python
scritti al volo in scratchpad (`analisi_qualitativa.py`, `posthoc.py`,
`trabocchetti.py`, mai committati). Ha valore anche se lo stadio 1 non ha
ancora superato l'esame: anzi, è più utile ORA, per costruire storie
"trabocchetto" a mano e vedere subito dove il modello sbaglia, senza
riscrivere ogni volta un mini-script.

## 2. Scelta tecnica: tkinter, non un'app web

**Raccomandazione**: `tkinter` (nella stdlib di Python, zero dipendenze
nuove — coerente con "dipendenze minime" di CLAUDE.md). Un'unica finestra
desktop, nessun server, nessun browser. L'alternativa (Flask/FastAPI +
pagina HTML) è più bella esteticamente ma aggiunge un processo server,
una dipendenza nuova e più parti in movimento per un beneficio che qui non
serve (non è un prodotto multi-utente, è uno strumento locale per Andrea).

Se in fase di implementazione tkinter risultasse troppo scarno per quello
che si vuole mostrare, la spec sotto si adatta facilmente: le funzioni di
riuso (§4) sono le stesse, cambia solo il layer di presentazione.

## 3. Dove vive: nuovo modulo top-level `interfaccia/`

Non dentro `esami/` o `cervello/` (sono concern diversi: valutazione batch
vs. esplorazione interattiva). Nuovo pacchetto sullo stesso piano di
`mondo/`, `lingua/`, `cervello/`, `esami/`:

```
interfaccia/
  __init__.py
  app.py          # finestra tkinter, punto d'ingresso (python -m interfaccia.app)
  ponte.py        # "collante" senza stato fisso: storia -> testo,
                   # domande candidate, esecuzione del modello, confronto
```

`interfaccia/` può importare da `mondo/`, `lingua/`, `cervello/`, `esami/`
(non è uno dei moduli con vincoli di importazione di CLAUDE.md — quei
vincoli valgono solo per `mondo/` che non deve importare `lingua/`).

## 4. Cosa riusare, modulo per modulo (niente di nuovo da inventare)

- **Genera storia**: `mondo.simulatore.genera_storia(seed, n_tick, persone)`.
  Cast opzionale (`mondo.dati_mondo.PERSONE`, sottoinsieme come già fa
  `esami/genera.py::_cast_persone`); lunghezza opzionale (`n_tick` a mano,
  oppure `mondo.generatore._lunghezza_storia(seed)` per una storia piena,
  oppure un valore fisso 3-6 per "stadio 1 corto").
- **Verbalizza la storia in italiano**: `lingua.verbalizza.verbalizza_evento`
  + `lingua.contesto.StatoDiscorso` (contesto condiviso, si aggiorna evento
  per evento) — ESATTAMENTE il ciclo già scritto in
  `lingua/__main__.py::_comando_campione_storia` (raggruppare per tick,
  unire le frasi dello stesso tick). Non riscriverlo, calcare quella
  funzione.
- **Domande candidate**: `mondo.domande.genera_domande(storia, rng, n_per_tipo)`
  per ottenere una lista di `Domanda` (tipo, grafo_domanda, grafo_risposta).
  Filtrare per i tipi che il checkpoint caricato sa gestire (leggere
  `config["stadi"][stadio]["tipi"]` dal file di config passato).
- **Verbalizza domanda/risposta**: `lingua.verbalizza.verbalizza_domanda` /
  `verbalizza_risposta` (stesso `contesto` accumulato dagli eventi, così le
  istanze tipo "la seconda mela" si risolvono correttamente).
- **Tag di difficoltà (opzionale ma consigliato)**: riusare
  `esami.genera._classifica_domanda_posizione(storia, domanda)` per
  mostrare "[facile]"/"[difficile]"/"[non-lo-so]" accanto a ogni domanda
  nella lista — utile per scegliere apposta casi interessanti da testare.
- **Esegui il modello**: caricare checkpoint+config con la stessa logica di
  `esami.esamina._carica_modello` (vocabolario, `ConfigModello`, pesi);
  comporre il prefisso con `cervello.sequenza.componi_esempio` (o a mano,
  come fa `esami.esamina.valuta_esempio`); decodificare con
  `esami.esamina.decodifica_greedy`; interpretare il risultato con
  `cervello.sequenza.token_a_grafo(..., "fatto")` (gestire `ValueError` per
  le sequenze malformate, non farle crashare l'interfaccia).
- **Confronto/categoria**: `esami.esamina._categoria(oro, generato)` per
  sapere se è esatto/errore/invenzione/astensione_errata/malformata, e
  colorare la risposta di conseguenza (verde/rosso).

## 5. Layout della finestra (indicativo, l'implementatore può adattare)

```
┌─────────────────────────────────────────────────────────────┐
│ Checkpoint: [percorso .pt]  Config: [percorso .yaml]  [Carica]│
├─────────────────────────────────────────────────────────────┤
│ Seed: [____]  [Seed casuale]   Cast: [x]anna [x]piero ...     │
│ Tick: [____] (o "storia corta 3-6" / "storia piena")          │
│                                    [Genera storia]            │
├─────────────────────────────────────────────────────────────┤
│  (testo della storia in italiano, per tick, scrollabile)      │
│  Alle nove Anna va in cucina. ...                             │
├─────────────────────────────────────────────────────────────┤
│ Domande candidate:                                            │
│  [ ] Dove si trova Piero?           [difficile]               │
│  [ ] Dove si trova la palla?        [facile]                  │
│  ...                                    [Chiedi al modello]    │
├─────────────────────────────────────────────────────────────┤
│ Risposta del modello:  "Piero è in giardino."      (ERRORE)   │
│ Risposta corretta:     "Piero è in cucina."                    │
│ [ ] mostra token grezzi (domanda / risposta modello / oro)     │
└─────────────────────────────────────────────────────────────┘
```

## 6. Determinismo (regola non negoziabile #2 di CLAUDE.md)

"Seed casuale" nell'interfaccia NON deve chiamare `random` per generare la
storia direttamente: usa `random` (globale, va bene qui perché è UI, non
generazione dati per training/esame) solo per **scegliere un numero di
seed** da mostrare nel campo, poi la storia si genera SEMPRE con
`genera_storia(seed=quel_numero, ...)`, deterministica. Stesso seed
digitato due volte -> stessa storia, byte per byte.

## 7. Punti da decidere con Andrea prima/durante l'implementazione

1. **Seed d'esame (>= 1.000.000) nell'interfaccia**: permetterli o no?
   Motivo per NON permetterli: se Andrea esplora molto a mano su seed
   d'esame, rischia di "contaminare" la propria intuizione su storie che
   dovrebbero restare cieche per la valutazione ufficiale. Suggerimento:
   di default rifiutarli (stesso messaggio di errore di `esami/genera.py`),
   con un flag esplicito `--permetti-seed-esame` per chi lo vuole
   consapevolmente.
2. **tkinter vs web app**: confermare tkinter (raccomandato, §2) o
   preferire un'interfaccia più curata esteticamente accettando la
   dipendenza in più.
3. **Domande "a mano" oltre a quelle candidate**: oltre a scegliere tra le
   domande generate da `genera_domande`, permettere di scegliere
   liberamente un'entità della storia e formare "Dove si trova X?" anche
   se non era tra le candidate originali (utile per le storie-trabocchetto
   costruite ad hoc). Consigliato: sì, ma è un secondo giro, non blocca la
   prima versione funzionante.
4. **Stadi diversi da 1**: la spec sopra assume domande di tipo
   "posizione" (stadio 1, l'unico con un modello da testare oggi). Quando
   stadio 3 sarà pronto (possesso/conteggio/transfer/parentela), l'unica
   modifica prevista è quali tipi di domanda filtrare — nessun cambiamento
   architetturale.

## 8. Cosa NON fare

- Non tocca `mondo/`, `lingua/`, `cervello/`, `esami/` (solo li importa).
- Non serve nessun training qui: l'interfaccia carica SOLO checkpoint già
  addestrati (locali, scaricati da Drive come si fa oggi per `esamina`/
  `diagnosi`).
- Non serve GPU: la decodifica di un singolo esempio è già stata misurata
  in locale su CPU (~0,5s/esempio in `esami.diagnosi`) — a velocità
  interattiva senza problemi.
