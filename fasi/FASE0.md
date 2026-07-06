# FASE 0 — Il micro-mondo

**Obiettivo**: un simulatore simbolico deterministico che genera storie
(sequenze di eventi) e domande con risposta esatta. Nessun machine learning
in questa fase. Solo Python standard, zero dipendenze esterne.

**Modulo**: `mondo/`

## Modello dei dati

- **Entità**: persone, oggetti, luoghi. Ogni entità ha id, lemma, tipo,
  attributi statici (colore, dimensione…).
- **Luoghi**: un grafo di luoghi nominati collegati da passaggi
  (cucina—giardino—…). NON una griglia geometrica: solo adiacenza.
- **Relazioni statiche**: parentela (madre_di, fratello_di…). Il possesso
  statico ("di chi è X") è rimandato: con lo stato iniziale ignoto (vedi
  sotto) la proprietà non è mai rivelata dagli eventi, quindi nessuna
  domanda su di essa sarebbe derivabile; tornerà quando esisterà un
  meccanismo di rivelazione (es. "dire" con contenuto).
- **Stato dinamico**: posizione di ogni persona/oggetto, contenimento
  (la mela è nel cestino), oggetti aperti/chiusi.
- **Stato iniziale ignoto (epistemica del mondo)**: si distingue tra
  *struttura* e *fatti contingenti*. La struttura — mappa dei luoghi,
  famiglia, arredi fissi (il camino è in salotto), regole del mondo
  ("le mani iniziano vuote", le azioni e i loro effetti) — è identica in
  ogni storia ed è conoscenza di sfondo lecita. I fatti contingenti —
  posizioni iniziali di persone e oggetti, contenimento iniziale,
  aperto/chiuso, quantità iniziali delle risorse — sono estratti a caso
  per seed e NON sono mai rivelati al lettore: un fatto contingente è
  conoscibile solo se stabilito dagli eventi della storia. Motivo: se lo
  stato iniziale fosse fisso, il modello potrebbe memorizzarlo dal corpus
  e le risposte d'oro "non lo so" diventerebbero incoerenti con ciò che è
  di fatto apprendibile.
- **Tempo**: discreto, a tick. Ogni evento ha un timestamp.
- **Energia e risorse**: ogni persona ha fame/stanchezza che crescono col
  tempo e si ripristinano con mangiare/dormire (un personaggio esausto può
  solo dormire). Il sonno dura più tick (la stanchezza si recupera
  gradualmente), chi dorme non agisce e non ci si addormenta se riposati:
  ogni "dormire" registra nell'evento la sua causa quando è determinata
  (esaustione), niente quando è un pisolino volontario. Gli oggetti
  consumabili hanno quantità finite: la mela mangiata viene rimossa dal
  mondo, la legna è contata. Invariante di conservazione: nulla si crea o
  sparisce se non per effetto di un'azione.
- **Testimoni**: ogni evento registra chi era presente E SVEGLIO nel luogo
  e quindi lo ha visto (chi dorme è presente ma non vede; chi si addormenta
  vede sé stesso farlo). (Serve nello stadio 5 del curriculum — teoria
  della mente. Predisporlo ORA costa una riga, aggiungerlo dopo costa una
  riscrittura.)

## Azioni (~15, stile STRIPS: precondizioni + effetti)

andare, prendere, posare, mettere_dentro, tirare_fuori, dare, mangiare,
aprire, chiudere, guardare, dire, dormire, svegliarsi, giocare, cercare.

Ogni azione è dichiarata come dati (precondizioni, effetti), non come codice
sparso: deve essere possibile aggiungere un'azione toccando un solo file.

## Generazione delle storie

- Lo stato iniziale (fatti contingenti, vedi sopra) è estratto dallo stesso
  RNG seedato della storia.
- Ogni personaggio ha una politica semplice: sceglie a caso (RNG con seed)
  tra le azioni valide nel suo stato, con qualche bias configurabile.
- Una storia = N tick di simulazione → lista di eventi strutturati.
- Determinismo assoluto: stesso seed → stessa storia, byte per byte.
- Split train/eval **per seed**: gli esami usano seed mai visti in training.

## Generazione delle domande

Template per tipo di task, risposta calcolata dallo stato del mondo (mai
scritta a mano, mai ambigua):

1. posizione: "Dove si trova X?" / "Dov'era X prima di andare in Y?"
2. possesso: "Chi ha X adesso?" (il possesso statico "Di chi è X?" è
   rimandato, vedi Modello dei dati).
3. conteggio: "Quanti oggetti porta X?" (sempre derivabile: le mani
   iniziano vuote per regola e ogni passaggio di mano è un evento) /
   "Quanti oggetti ci sono in Y?" (luogo o contenitore: derivabile solo
   se OGNI oggetto è stato localizzato dagli eventi — un oggetto mai
   menzionato potrebbe trovarsi proprio lì).
4. transfer: "Chi ha dato X a Y?"
5. parentela (stile CLUTRR): catene di 2–4 passi. Deviazione accettata:
   la famiglia attuale (6 persone, struttura fissa = conoscenza di sfondo)
   copre tutte le coppie in ≤2 passi, quindi questo tipo non produce
   domande "non lo so"; per catene più lunghe e quote non-lo-so servirà
   una famiglia più larga.
6. deduzione multi-hop: combinazioni dei precedenti.
7. causa/energia: "Perché X dorme?" (derivabile solo se ogni sonno di X
   nella storia ha causa registrata "stanchezza"; i pisolini volontari non
   hanno causa determinabile → "non lo so"; cause miste → domanda esclusa
   perché ambigua) / "Quante mele sono state raccolte?" (conteggio di
   eventi; "quante restano" NON è lecita: la quantità iniziale è un fatto
   contingente mai rivelato).
8. **non determinabile**: domande la cui risposta NON è ricavabile dagli
   eventi della storia (informazione mai menzionata o senza testimoni).
   Risposta d'oro: il token `non-lo-so`. Devono essere ~15–20% delle domande
   di OGNI tipo e di ogni stadio (eccezione documentata: parentela, punto 5),
   generate mescolando domande vere su fatti fuori dalla storia. Il
   generatore verifica formalmente la non-derivabilità (non basta
   "probabilmente non si sa").
9. (stadio 6) credenze: "Dove pensa X che sia Y?" — usa i testimoni.

## Formato di uscita

JSONL, un record per storia:

```json
{"seed": 42, "eventi": [{"t": 0, "azione": "prendere", "agente": "sara",
  "oggetto": "mela", "luogo": "cucina", "testimoni": ["sara", "luca"]}, ...],
 "domande": [{"tipo": "posizione", "grafo_domanda": ..., "grafo_risposta": ...}]}
```

Gli eventi sono strutture, NON testo: il testo lo produce la Fase 1.
Ogni evento deve essere convertibile nel grafo UD corrispondente
(funzione `evento_a_grafo`), perché quello è il formato che il cervello vede.

## Criteri di accettazione

- `pytest tests/test_mondo.py` verde, con test sugli **invarianti**:
  nessun oggetto in due posti, precondizioni sempre rispettate prima di
  ogni effetto, testimoni sempre coerenti con le posizioni.
- Generazione di 10.000 storie in < 1 minuto su CPU.
- Statistiche di copertura stampabili: ogni azione e ogni tipo di domanda
  appare almeno l'1% delle volte; distribuzione dei lemmi.
- Stesso seed → output identico (test di riproducibilità byte per byte).

## Trappole note (da NON fare)

- Non usare un LLM per generare o "arricchire" le storie: viola il
  principio 2 del progetto (PROGETTO.md).
- Non mettere testo italiano dentro `mondo/`: la lingua vive in `lingua/`.
- Non usare `random` globale: ogni generatore riceve il suo RNG seedato.
