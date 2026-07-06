# Progetto Cervello-Bambino

Un modello linguistico minimale ispirato allo sviluppo cognitivo di un bambino:
piccolo, ispezionabile, a basso consumo. Impara competenza linguistica, logica e
relazioni — **non** nozioni enciclopediche (quelle verranno dopo, come memoria
esterna consultabile).

## Principi

1. **Determinismo ai bordi, apprendimento al centro.** Tutto ciò che è regolare
   (morfologia, sintassi, verbalizzazione, regole non negoziabili) vive in
   componenti simbolici ispezionabili. La rete impara solo ciò che non si può
   scrivere a regole: significato, relazioni, inferenza.
2. **Nessuna scorciatoia pre-addestrata.** Niente embedding o encoder presi da
   modelli grandi (es. SONAR): importerebbero conoscenze e bias che vogliamo
   tenere fuori. Il bambino nasce vuoto.
3. **Struttura prima, nozioni poi.** Il modello si misura solo su
   grammaticalità e ragionamento, mai su fatti del mondo.
4. **Piccolezza come strategia.** Il risparmio computazionale viene dalla scala
   (~10M parametri) e dal parser, non dall'eliminazione dell'attenzione.

## Architettura

```
testo → [PARSER deterministico] → grafo UD → [CERVELLO appreso] → grafo UD → [VERBALIZZATORE deterministico] → testo
                                       ↑ filtro-regole (etica dura) su input e output ↑
```

### Front-end deterministico
- Punto chiave che rende tutto misurabile: **il simulatore emette sia il testo
  sia il grafo-verità di ogni frase**. Parser e verbalizzatore non si valutano
  "a occhio" ma contro i grafi-verità, con accuratezza esatta.
- Verbalizzatore: grafo → testo flesso, a regole. Deve coprire SOLO la
  grammatica controllata del micro-mondo, non tutto l'italiano.
- Parser: testo → grafo (spaCy/Stanza con mappatura, o inverso delle regole
  del verbalizzatore — sulla grammatica chiusa le due strade sono equivalenti).
- Filtro simbolico per le regole non negoziabili (fuori dalla rete).
- **Test di accettazione fase 1**: giro completo frase → grafo → frase
  senza perdita, sul vocabolario controllato.

### Spazio concettuale (l'interlingua)
- Rappresentazione interna: grafi di lemmi + relazioni Universal Dependencies.
  UD è identico per 100+ lingue → la lingua è una "pelle" sostituibile,
  il cervello non impara "l'italiano" ma il ragionamento su grafi.
- **Vocabolario-seme**: i ~65 primitivi semantici NSM (Wierzbicka) — IO, TU,
  VOLERE, SAPERE, BUONO, PRIMA, PERCHÉ… — come primi vettori dello spazio
  latente; il resto del vocabolario si definisce per composizione.
  Meccanismo concreto: i primitivi occupano i token id 0–64 del vocabolario;
  ogni lemma del lessico ha una definizione in primitivi, e le definizioni
  entrano nel corpus come dati ("che cosa significa X?" → grafo-definizione).
- Residuo dichiarato: i lemmi sono italiani; la mappatura lemma → concetto
  interlinguale è un passo successivo.

### Il cervello
- Rete piccola (~5–20M parametri) sui grafi concettuali linearizzati.
- Backbone v1: **transformer decoder semplice** (stile nanoGPT, ~8 strati,
  d=256). Motivo: a questa scala l'attenzione costa nulla e la ricetta è
  collaudatissima — un esecutore la implementa senza sorprese. L'ibrido con
  strati ricorrenti (tipo Mamba) è un esperimento successivo, non un
  requisito (vedi questioni aperte).
- **Pesi ternari {−1, 0, +1}** (stile BitNet b1.58): +1 eccitatorio,
  −1 inibitorio, 0 sinapsi assente. Solo somme e sottrazioni in inferenza.
- Attivazioni a 8 bit (la "frequenza di firing" vive lì, non comprimerle a 1 bit).
- Training: quantization-aware con pesi ombra in float (straight-through).
- Percorso: prima versione in float per separare i bug di architettura da
  quelli di quantizzazione, poi ri-addestramento ternario.
- Crescita successiva: espansione tipo Net2Net quando la base è solida,
  non zeri sparsi in una rete grande (non fanno risparmiare nulla in float).

### Uscita (generazione)
- Interfaccia fissa: il cervello produce un **grafo-pensiero**, il
  verbalizzatore lo rende testo. La generazione riguarda il grafo, mai i token.
- **v1 — autoregressivo sul grafo**: il grafo si scrive un elemento alla
  volta. Semplice, collaudato, facile da debuggare.
- **v2 — diffusione discreta sul grafo**: si parte da un grafo-rumore e lo si
  raffina in K passi (stile LLaDA / DiGress, ma su ~20 slot concettuali, non su
  migliaia di token → è il regime dove la diffusione costa poco). Vantaggi
  attesi: coerenza globale del pensiero (niente miopia sinistra→destra),
  auto-correzione iterativa (affidabilità), parallelismo.
- Stessa interfaccia per v1 e v2 → si confrontano sui benchmark di logica a
  parità di FLOP, senza rifare nulla intorno.

## Corpus: il micro-mondo

Le storie NON vengono generate da un LLM (importerebbe i suoi bias, violando
il principio 2). Vengono generate da un **simulatore simbolico**: un piccolo
mondo a regole (personaggi, oggetti, luoghi, azioni, possesso, parentele,
tempo) i cui eventi vengono verbalizzati deterministicamente.

- Ogni storia ha una semantica verificabile → le domande di logica hanno
  risposta esatta per costruzione (etichette perfette, zero rumore).
- Le parole sono *radicate*: MANGIARE è un evento del mondo, non una
  co-occorrenza statistica.
- Vocabolario controllato: ~1.500–2.000 lemmi italiani ("parlare da mamma").
- **Fisica intuitiva come regole del mondo** (non come nozioni): energia dei
  personaggi (fame, stanchezza, dormire/mangiare per ripristinarle), risorse
  finite e consumabili (la mela mangiata non esiste più, la legna è contata),
  conservazione (nulla si duplica o sparisce senza causa), causa-effetto
  (il fuoco consuma legna e scalda). Il "mondo reale" entra come struttura,
  mai come dati esterni.
- Task di valutazione: stile bAbI (posizione, possesso, conteggio, deduzione)
  e CLUTRR (parentele), più ragionamento causale e sulle quantità,
  generati dallo stesso simulatore ma su semi separati.
- **Domande senza risposta determinabile**: il simulatore sa che cosa è
  conoscibile dalla storia; genera anche domande la cui risposta d'oro è
  "non lo so". Presenti in OGNI stadio del curriculum.

## Curriculum (stadi di sviluppo)

1. Grammatica e frasi semplici (il mondo descrive sé stesso).
2. Spazio e tempo (dov'è, quando, prima/dopo).
3. Possesso, quantità, parentele.
4. Fisica intuitiva: energia, risorse, conservazione, causa-effetto.
5. Catene di deduzione a più passi.
6. Credenze e stati mentali (X crede che…, teoria della mente).

Trasversale a tutti gli stadi: le domande "non lo so" (vedi Etica e valori).

Ogni stadio ha un "esame di passaggio" su dati mai visti. Mai addestrare
sui set di valutazione.

## Etica e valori

- Regole dure: nel filtro simbolico ai bordi (deterministico, ispezionabile).
- Disposizioni morbide: nel curriculum — si scelgono le storie su cui il
  bambino cresce, come fa un genitore.
- **Onestà epistemica come primo valore** (bias scelto e dichiarato):
  - "non lo so" è un token di prima classe e una risposta corretta a tutti
    gli effetti, allenata fin dallo stadio 1;
  - negli esami, la risposta inventata costa PIÙ dell'astensione: l'esame
    misura sia l'esattezza sia la calibrazione (astenersi quando la storia
    non contiene la risposta, rispondere quando la contiene);
  - dire la verità è misurabile per costruzione: il simulatore conosce sia
    i fatti sia ciò che è conoscibile, quindi ogni invenzione è rilevabile.
- Onestà nostra: non esiste un modo noto per "cablare" valori garantiti nei
  pesi; questi sono valori insegnati e misurati, non dimostrati.

## Roadmap

Le specifiche esecutive di ogni fase sono in `fasi/FASE<N>.md`: contratto dei
moduli, formati dati, criteri di accettazione. Chi implementa parte da lì.

- **Fase 0** — simulatore del micro-mondo + generatore di storie e task.
- **Fase 1** — front-end: parser ↔ verbalizzatore, giro completo senza perdita.
- **Fase 2** — cervello v1 (float, autoregressivo sul grafo), esami del curriculum.
- **Fase 2b** — ri-addestramento ternario, confronto qualità/consumo.
- **Fase 3** — uscita a diffusione (v2), confronto con v1 a parità di FLOP.
- **Fase 4** — crescita: memoria esterna per le nozioni, seconda "pelle" linguistica.

## Questioni aperte

- Ibrido con strati ricorrenti (Mamba) al posto di parte dell'attenzione:
  esperimento da fare dopo la v1, confrontando a parità di parametri.
- Mappatura lemmi → concetti interlinguali (oltre i 65 primitivi).
- Obiettivo di training per i vettori-concetto: cosa impedisce di perdere
  dettagli che servono molte frasi dopo?
- Quanti passi di diffusione servono perché la v2 batta la v1 a parità di
  energia?
- Direzione futura da tenere d'occhio: predire nello spazio latente senza
  ricostruire la superficie (famiglia JEPA / energy-based).
