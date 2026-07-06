# PIANO ESECUTIVO — FASE 1: `lingua/` (verbalizzatore + parser + filtro)

Documento esecutivo per l'agente che implementa la Fase 1. Prima di scrivere
codice, leggere nell'ordine: `CLAUDE.md`, `PROGETTO.md`, `fasi/FASE1.md`.
Questo piano **specializza** quei documenti: dove questo piano è più
dettagliato, si segue questo piano; se sembra **contraddirli**, fermarsi e
chiedere ad Andrea, non improvvisare. Le scelte di design qui dentro sono già
state discusse e approvate da Andrea (2026-07-07): non riaprirle e non
"migliorarle" in corsa.

---

## 0. Contesto in due paragrafi

Il progetto Cervello-Bambino separa un centro appreso (una piccola rete, Fase
2) da bordi deterministici. La Fase 0 (già fatta, in `mondo/`) genera storie
come sequenze di `Evento` strutturati e domande/risposte come grafi UD
(`mondo/grafo.py`). La Fase 1 costruisce il confine col testo: `verbalizza`
(grafo → frase italiana) e `analizza` (frase → grafo), più un filtro simbolico
sui grafi. Niente ML, niente LLM, solo regole: stdlib Python ≥ 3.11.

Il criterio che governa tutto: **round-trip esatto**. Per ogni grafo `g`
prodotto dal mondo, `analizza(verbalizza(g)) == g` (uguaglianza esatta di
`Grafo`, che è un dataclass frozen confrontabile con `==`). La valutazione è
sempre grafo vs grafo, mai stringa vs stringa.

## 1. Decisioni vincolanti (già prese con Andrea — non rimetterle in discussione)

1. **Parser = inverso delle regole del verbalizzatore** (opzione (a) di
   FASE1.md). Niente spaCy, niente dipendenze esterne.
2. **Stampi dichiarativi, unica fonte**: ogni costrutto della lingua è
   definito UNA volta come dato (tabella/stampo); il verbalizzatore lo rende,
   il parser lo riconosce. Mai due grammatiche scritte separatamente a mano.
3. **Istanze delle risorse (`mela_3`, `acqua_1`…): ordinali solo se
   ambiguo.** Prima menzione indefinita ("una mela"), menzioni successive
   definite ("la mela"); se nella storia sono già comparse ≥ 2 istanze dello
   stesso lemma, TUTTE le menzioni successive di quel lemma usano l'ordinale
   ("la prima mela", "la seconda mela"). L'ordinale è l'indice numerico
   nell'id (`mela_3` → "terza"): coincide con l'ordine di introduzione perché
   `StatoMondo.nuovo_id` numera in ordine di creazione e la creazione è
   sempre il primo evento che menziona l'istanza.
4. **Tempo: orario a inizio tick.** La prima frase di ogni tick apre con
   l'ora ("Alle nove …", tick 1 → "All'una …"), le frasi successive dello
   stesso tick aprono con "Intanto …". I tick senza eventi non producono
   frasi: l'ora successiva risincronizza da sola. Ogni frase-evento inizia
   quindi SEMPRE con "Alle …"/"All'una …" oppure "Intanto …".
5. **Copertura: tutte e tre le famiglie di grafi** — eventi
   (`evento_a_grafo`), domande e risposte (`grafo_fatto`, incluso
   `NON_LO_SO`). Il filtro si applica a tutte.
6. **Contesto di discorso esplicito.** Le firme diventano
   `verbalizza*(grafo, contesto)` / `analizza*(frase, contesto)`: è una
   deviazione dichiarata rispetto alle firme senza contesto di FASE1.md,
   necessaria per le decisioni 3 e 4. Il round-trip esatto vale a livello di
   storia (e di singola frase a parità di contesto).
7. **Tempi verbali della v1**: presente narrativo per gli eventi; passato
   prossimo SOLO nei quattro stampi che lo richiedono (domanda/risposta
   transfer e raccolta). L'imperfetto NON si implementa: nessuna frase del
   corpus lo usa, e FASE1.md vieta regole senza frasi che le usino.
8. **`mondo/` non si tocca.** Niente refactor di `mondo/` per fargli leggere
   il lessico (arriverà in una fase successiva): la coerenza
   lessico ↔ `dati_mondo.py` si garantisce con un test, non con un refactor.
9. I `testimoni` degli eventi non compaiono né nei grafi (già oggi
   `evento_a_grafo` li scarta) né nel testo: non sono compito della Fase 1.

## 2. Struttura del modulo

```
lingua/
  __init__.py        # esporta l'API pubblica (elenco sotto)
  lessico.py         # caricamento e validazione di lessico.tsv
  lessico.tsv        # IL lessico (sezione 7): unica fonte del vocabolario
  morfologia.py      # articoli, preposizioni articolate, plurali, numeri,
                     # ordinali, ore, accordo aggettivi
  contesto.py        # StatoDiscorso
  stampi.py          # gli stampi dichiarativi (eventi + domande + risposte)
  verbalizza.py      # rendering: grafo -> frase
  analizza.py        # riconoscimento: frase -> grafo
  filtro.py          # filtro-regole sui grafi
  regole_filtro.txt  # pattern vietati (segnaposto)
  __main__.py        # CLI: campione e verifica
tests/
  test_lingua.py     # tutti i test della fase (pytest)
```

Dipendenze: `lingua/` importa da `mondo/` (`tipi.Evento`, `grafo.Grafo`,
`grafo.evento_a_grafo`, `grafo.grafo_fatto`, `grafo.NON_LO_SO`,
`dati_mondo`, `azioni.AZIONI`, `simulatore`, `domande`, `generatore` — questi
ultimi solo nei test/CLI). `mondo/` NON deve mai importare `lingua/`.
Identificatori e docstring in italiano. Nessun `random` globale: l'unico RNG
ammesso è nel CLI `campione` (seed esplicito). `verbalizza`/`analizza` sono
funzioni pure deterministiche: **mai** iterare su `set`/`dict` senza ordine
fissato quando l'ordine influenza l'output.

### API pubblica (firme esatte)

```python
# contesto.py
@dataclass
class StatoDiscorso:
    tick_corrente: int | None = None
    max_indice: dict[str, int] = field(default_factory=dict)   # lemma -> max indice istanza visto
    posizione_persone: dict[str, str] = field(default_factory=dict)  # persona_id -> luogo_id noto al lettore
    def registra_evento(self, evento: Evento) -> None: ...

# verbalizza.py   (tutte MUTANO contesto solo per gli eventi; domande/risposte lo leggono soltanto)
def verbalizza_evento(grafo: Grafo, contesto: StatoDiscorso) -> str: ...
def verbalizza_domanda(grafo: Grafo, contesto: StatoDiscorso) -> str: ...
def verbalizza_risposta(grafo: Grafo, contesto: StatoDiscorso) -> str: ...
def verbalizza_storia(grafi: Sequence[Grafo], contesto: StatoDiscorso | None = None) -> list[str]: ...

# analizza.py   (speculari; contesti INDIPENDENTI da quelli usati per verbalizzare)
def analizza_evento(frase: str, contesto: StatoDiscorso) -> Grafo: ...
def analizza_domanda(frase: str, contesto: StatoDiscorso) -> Grafo: ...
def analizza_risposta(frase: str, contesto: StatoDiscorso) -> Grafo: ...
def analizza_storia(frasi: Sequence[str], contesto: StatoDiscorso | None = None) -> list[Grafo]: ...

# filtro.py
@dataclass(frozen=True)
class RisultatoFiltro:
    ammesso: bool
    regola_violata: str | None = None
def filtra(grafo: Grafo) -> RisultatoFiltro: ...
```

Il round-trip di riferimento (così va scritto nei test):

```python
grafi = [evento_a_grafo(e) for e in storia.eventi]
cv = StatoDiscorso();  frasi = verbalizza_storia(grafi, cv)
cp = StatoDiscorso();  assert analizza_storia(frasi, cp) == grafi
# domande e risposte usano i contesti di FINE storia (cv per verbalizzare, cp per analizzare)
for d in genera_domande(storia, rng_domande, n_per_tipo=8):
    assert analizza_domanda(verbalizza_domanda(d.grafo_domanda, cv), cp) == d.grafo_domanda
    assert analizza_risposta(verbalizza_risposta(d.grafo_risposta, cv), cp) == d.grafo_risposta
```

### Come il parser ricostruisce i grafi (byte-identici)

Mai costruire nodi/archi a mano nel parser. Si ricostruisce la struttura di
partenza e si riusa il costruttore del mondo, così la numerazione dei nodi
coincide per costruzione:

- **Eventi**: il parser ricava un `Evento(t, azione, agente, oggetto,
  destinatario, luogo, luogo_origine, argomento)` (testimoni: lasciare il
  default `()`) e ritorna `evento_a_grafo(evento)`.
- **Domande/risposte**: il parser chiama `grafo_fatto` con ESATTAMENTE gli
  argomenti e l'ORDINE elencati nella sezione 6 (l'ordine dei kwargs
  determina gli id dei nodi — copiato da `mondo/domande.py`, non cambiarlo).
- **"Non lo so."** → ritornare la costante `NON_LO_SO` importata.

In `verbalizza.py` serve l'inverso `grafo_a_evento(grafo) -> Evento`
(lettura degli archi per relazione: `nsubj`→agente, `obj`→oggetto,
`iobj`→destinatario, `obl:argomento`→argomento, `obl:origine`→luogo_origine,
`obl:luogo`→luogo, `obl:tempo`→t; radice→azione).

## 3. Il contesto di discorso

`StatoDiscorso` contiene SOLO tre cose (non aggiungerne altre):

1. `tick_corrente` — per decidere/leggere "Alle X" vs "Intanto".
2. `max_indice[lemma]` — il massimo indice di istanza visto finora per un
   lemma di risorsa ("mela", "acqua", "legna"). Non si decrementa MAI
   (nemmeno se l'istanza viene mangiata o bruciata).
3. `posizione_persone[persona_id]` — l'ultimo luogo in cui il lettore ha
   collocato ogni persona.

`registra_evento(evento)` (chiamato da verbalizzatore E parser, dopo ogni
frase-evento, con l'evento COMPLETO):

- `tick_corrente = evento.t`
- se `evento.oggetto` è un'istanza (id della forma `lemma_N`):
  `max_indice[lemma] = max(max_indice.get(lemma, 0), N)`
- `posizione_persone[agente] = evento.luogo` (se `evento.luogo` non è None;
  per `bruciare` l'agente è "camino": NON registrarlo tra le persone)
- se l'azione è `dare` o `dire`: anche
  `posizione_persone[destinatario] = evento.luogo`
- se l'azione è `guardare` e `evento.oggetto` è un id di persona: anche
  `posizione_persone[oggetto] = evento.luogo`

Domande e risposte NON chiamano `registra_evento`.

### Regola di esplicitazione del luogo (unica, vale per tutti gli stampi-evento)

Il complemento di luogo in coda (" in cucina", " nel bosco", …) si scrive
**solo se serve**, cioè:

- **mai**, se il luogo è derivabile strutturalmente: azione `prendere` con
  fonte (il luogo è quello della fonte: melo→orto, pozzo→orto,
  bosco_legna→bosco), `mettere_dentro` con contenitore `camino` (→ salotto),
  azione `bruciare` (→ salotto);
- **mai**, se `posizione_persone[agente] == evento.luogo` (il lettore sa già
  dov'è l'agente);
- **sì**, in tutti gli altri casi (prima apparizione di un personaggio, o
  grafo costruito a mano con luogo incoerente col contesto).

Il parser applica la stessa regola al contrario: se il luogo non è nella
frase, lo deduce (fonte/camino/bruciare, altrimenti
`posizione_persone[agente]`). Per `andare` vale la regola gemella
sull'**origine**: "va dalla cucina in giardino" se la posizione dell'agente
è ignota al lettore, "va in giardino" se è nota (e l'origine si deduce dal
contesto). La destinazione si scrive sempre.

## 4. La grammatica del testo

### 4.1 Forma delle frasi

- Eventi: `[Prefisso tempo] [soggetto] [verbo] [complementi fissi dello
  stampo] [luogo?] [causa?].` — punto finale, un solo punto per frase.
- Prefisso tempo: tick 1 → `All'una ` ; tick ≥ 2 → `Alle {numero in
  lettere} ` ; stesso tick della frase precedente → `Intanto `.
  La prima frase in assoluto di una storia ha sempre l'ora, mai "Intanto".
- Domande: nessun prefisso tempo, finiscono con `?`.
- Risposte: nessun prefisso, iniziale maiuscola, punto finale.
- Codifica UTF-8; gli accenti si scrivono (è, dà, perché, c'è, ventitré).
- `verbalizza_storia` ritorna una lista di frasi (una per grafo, stesso
  ordine); l'eventuale unione in paragrafo è solo presentazione del CLI.

### 4.2 Sintagma nominale — `sn(entita_id, contesto)` e il suo inverso

| tipo di id | esempio | resa |
|---|---|---|
| persona | `sara` | superficie del nome proprio: "Sara" (capitalizza il lemma) |
| `nessuno` / `qualcuno` | — | "nessuno" / "qualcuno" |
| oggetto unico (id == lemma) | `palla` | articolo determinativo + lemma: "la palla", "il pane", "il cestino" |
| istanza risorsa, **prima menzione** (solo nello stampo prendere-con-fonte) | `mela_1` | indefinito: "una mela"; per i nomi massa (`massa=si` nel lessico): "dell'acqua", "della legna" |
| istanza risorsa, menzioni successive, `max_indice[lemma] == 1` | `mela_1` | definito semplice: "la mela", "l'acqua", "la legna" |
| istanza risorsa, menzioni successive, `max_indice[lemma] >= 2` | `mela_2` | definito + ordinale dall'indice dell'id: "la seconda mela", "la prima acqua" |

Parsing inverso: "una mela"/"dell'acqua" in uno stampo prendere-con-fonte
introduce l'istanza con indice `max_indice[lemma] + 1`; "la mela" →
`lemma_1` (lecito solo se `max_indice == 1`); "la seconda mela" → `mela_2`.
Un ordinale oltre `max_indice` o un definito semplice con `max_indice >= 2`
sono errori di parsing (sollevare `ValueError` con messaggio chiaro).

### 4.3 Luoghi (forme fisse, dal lessico — non derivarle con regole)

| luogo | stato/moto a: `loc_in` | origine: `loc_da` |
|---|---|---|
| cucina | in cucina | dalla cucina |
| salotto | in salotto | dal salotto |
| giardino | in giardino | dal giardino |
| camera | in camera | dalla camera |
| orto | nell'orto | dall'orto |
| bosco | nel bosco | dal bosco |

I **contenitori** usano invece le preposizioni articolate regolari di
`morfologia.py`: "nel cestino", "nella scatola", "nel secchio", "nel
camino"; "dal cestino", "dalla scatola", "dal secchio", "dal camino".

## 5. Stampi degli eventi (tabella normativa)

Colonna "frase" = corpo dopo il prefisso tempo. `{luogo?}` segue la regola
della sezione 3. `{Ag}` = sn(agente), `{O}` = sn(oggetto), `{D}` =
sn(destinatario). Ordine di riconoscimento nel parser = ordine di questa
tabella (dall'alto: il primo stampo che combacia sull'INTERA frase vince).

| # | azione (radice del grafo) | frase |
|---|---|---|
| 1 | andare (origine da esplicitare) | `{Ag} va {loc_da(origine)} {loc_in(dest)}.` |
| 2 | andare (origine nota) | `{Ag} va {loc_in(dest)}.` |
| 3 | prendere con `argomento`=melo | `{Ag} raccoglie una mela dal melo.` |
| 4 | prendere con `argomento`=pozzo | `{Ag} prende dell'acqua dal pozzo.` |
| 5 | prendere con `argomento`=bosco_legna | `{Ag} raccoglie della legna nel bosco.` |
| 6 | tirare_fuori | `{Ag} tira fuori {O} {da(contenitore)}{luogo?}.` |
| 7 | prendere senza fonte | `{Ag} prende {O}{luogo?}.` |
| 8 | posare | `{Ag} posa {O}{luogo?}.` |
| 9 | mettere_dentro | `{Ag} mette {O} {in(contenitore)}{luogo?}.` |
| 10 | dare | `{Ag} dà {O} a {D}{luogo?}.` |
| 11 | mangiare | `{Ag} mangia {O}{luogo?}.` |
| 12 | aprire | `{Ag} apre {O}{luogo?}.` |
| 13 | chiudere | `{Ag} chiude {O}{luogo?}.` |
| 14 | guardare | `{Ag} guarda {O}{luogo?}.` (O può essere una persona) |
| 15 | dire | `{Ag} dice qualcosa a {D}{luogo?}.` |
| 16 | dormire, `argomento`="stanchezza" | `{Ag} si addormenta{luogo?} perché è {stanco/stanca}.` |
| 17 | dormire, `argomento`=None | `{Ag} si addormenta{luogo?}.` |
| 18 | svegliarsi | `{Ag} si sveglia{luogo?}.` |
| 19 | giocare con oggetto | `{Ag} gioca con {O}{luogo?}.` |
| 20 | giocare senza oggetto | `{Ag} gioca{luogo?}.` |
| 21 | cercare | `{Ag} cerca {O}{luogo?}.` (luogo = dove si trova chi cerca) |
| 22 | bruciare (agente="camino") | `il camino brucia {O}.` |

Note vincolanti:

- Negli stampi 3–5 l'`Evento` ricostruito ha `oggetto` = nuova istanza,
  `argomento` = fonte (dedotta dal lemma: mela→melo, acqua→pozzo,
  legna→bosco_legna), `luogo` = luogo della fonte. Nella frase il luogo non
  compare mai.
- Stampo 16/17: "stanco"/"stanca" secondo il genere dell'agente (lessico).
  La causa mappa su `argomento="stanchezza"`; senza causa, `argomento=None`.
- Stampo 22: `agente="camino"`, `luogo="salotto"` sempre; il prefisso tempo
  si applica normalmente ("Alle dieci il camino brucia la legna." /
  "Intanto il camino brucia la seconda legna.").
- `{luogo?}` esplicito si rende come ` {loc_in(luogo)}` in coda; nello
  stampo 16 la causa "perché è …" viene DOPO il luogo:
  `Luca si addormenta in camera perché è stanco.`

## 6. Stampi di domande e risposte (tabella normativa)

Nessun prefisso tempo. Contesto: quello di fine storia (sola lettura).
"Non lo so." ↔ `NON_LO_SO` vale come risposta per OGNI tipo. Nella colonna
ricostruzione, l'ordine dei kwargs è NORMATIVO (deve replicare
`mondo/domande.py`). `{n}` = numero in lettere; `{B}` = luogo (`in B` dalla
tabella 4.3) oppure contenitore (`nel/nella B`).

| tipo | domanda: frase e ricostruzione | risposta: frase e ricostruzione |
|---|---|---|
| posizione | `Dove si trova {E}?` → `grafo_fatto("trovarsi", nsubj=E, quesito="dove")` | `{E} è {loc_in(L)}.` → `grafo_fatto("essere", nsubj=E, **{"obl:luogo": L})` |
| possesso | `Chi ha {O}?` → `grafo_fatto("avere", obj=O, quesito="chi")` | `{P} ha {O}.` (P può essere "nessuno" → "Nessuno ha …") → `grafo_fatto("avere", nsubj=P, obj=O)` |
| conteggio-persona | `Quanti oggetti porta {P}?` → `grafo_fatto("portare", nsubj=P, quesito="quanti")` | n=0: `{P} non porta nessun oggetto.` · n=1: `{P} porta un oggetto.` · n≥2: `{P} porta {n} oggetti.` → `grafo_fatto("portare", nsubj=P, **{"obl:quantita": str(n)})` |
| conteggio-posto | `Quanti oggetti ci sono {B}?` → `grafo_fatto("esserci", **{"obl:luogo": B, "quesito": "quanti"})` | n=0: `{B} non c'è nessun oggetto.` · n=1: `{B} c'è un oggetto.` · n≥2: `{B} ci sono {n} oggetti.` (B con iniziale maiuscola: "In cucina …", "Nel cestino …") → `grafo_fatto("esserci", **{"obl:luogo": B, "obl:quantita": str(n)})` |
| transfer | `Chi ha dato {O} a {D}?` → `grafo_fatto("dare", obj=O, iobj=D, quesito="chi")` | `{A} ha dato {O} a {D}.` → `grafo_fatto("dare", nsubj=A, obj=O, iobj=D)` |
| parentela | `Che parente è {A} di {B}?` → `grafo_fatto("essere", nsubj=A, **{"nmod:relativo": B, "quesito": "che-parente"})` | `{A} è {SN relazione} di {B}.` → `grafo_fatto("essere", nsubj=A, **{"nmod:parentela": rel, "nmod:relativo": B})` |
| deduzione | `Dove si trova {O} che {A} ha dato a {D}?` (A può essere "qualcuno") → `grafo_fatto("trovarsi", **{"nmod:agente": A, "nmod:oggetto": O, "nmod:destinatario": D, "quesito": "dove"})` | come "posizione": `{O} è {loc_in(L)}.` → `grafo_fatto("essere", nsubj=O, **{"obl:luogo": L})` |
| causa | `Perché {P} dorme?` → `grafo_fatto("dormire", nsubj=P, quesito="perche")` | `{P} dorme perché è {stanco/stanca}.` → `grafo_fatto("dormire", nsubj=P, **{"advcl:causa": "stanchezza"})` |
| raccolta (tipo "causa", 2ª parte) | mela: `Quante mele sono state raccolte?` · massa: `Quante volte è stata raccolta l'acqua?` / `… la legna?` → `grafo_fatto("raccogliere", obj=U, quesito="quante")` | mela: n=0 `Non è stata raccolta nessuna mela.` · n=1 `È stata raccolta una mela.` · n≥2 `Sono state raccolte {n} mele.` — massa: n=0 `L'acqua non è mai stata raccolta.` · n=1 `L'acqua è stata raccolta una volta.` · n≥2 `L'acqua è stata raccolta {n} volte.` (idem "La legna …") → `grafo_fatto("raccogliere", obj=U, **{"obl:quantita": str(n)})` |

Mappa parentela (relazione → sintagma; inverso univoco, l'articolo di
"nipote" segue il genere di A ma il parsing ignora l'articolo):

padre_di→"il padre", madre_di→"la madre", figlio_di→"il figlio",
figlia_di→"la figlia", marito_di→"il marito", moglie_di→"la moglie",
fratello_di→"il fratello", sorella_di→"la sorella", nonno_di→"il nonno",
nonna_di→"la nonna", nipote_di→"il nipote"/"la nipote",
suocero_di→"il suocero", suocera_di→"la suocera", genero_di→"il genero",
nuora_di→"la nuora", zio_di→"lo zio", zia_di→"la zia",
cugino_di→"il cugino", cugina_di→"la cugina".

Ordine di riconoscimento delle domande: deduzione PRIMA di posizione (è più
lunga e la contiene); per il resto l'ordine è libero perché gli incipit sono
distinti. Le risposte si distinguono per verbo e struttura; "Non lo so." si
controlla per prima, con confronto esatto della stringa.

## 7. Il lessico — `lingua/lessico.tsv`

Formato: 4 colonne separate da TAB — `lemma`, `categoria`, `tratti`,
`definizione`. Tratti: coppie `chiave=valore` separate da virgola, `-` se
vuoti. Definizione: testo in primitivi NSM, `TODO` se mancante, `-` per i
primitivi stessi. Righe che iniziano con `#` = commento. **L'ORDINE DELLE
RIGHE È PARTE DEL CONTRATTO**: le prime 65 righe sono i primitivi NSM nella
sequenza sotto (in Fase 2 diventeranno i token id 0–64); non riordinarle mai.

### 7.1 Le 65 righe PRIM (ordine normativo)

Scriverle con categoria `PRIM`, tratti `pos=…`, definizione `-`. Elenco
lineare normativo `indice lemma (pos)`, raggruppato per famiglia NSM:

- Sostantivi: 0 io (PRON) · 1 tu (PRON) · 2 qualcuno (PRON) ·
  3 qualcosa (PRON) · 4 gente (NOUN) · 5 corpo (NOUN)
- Relazionali: 6 tipo (NOUN) · 7 parte (NOUN)
- Determinanti: 8 questo (PRON) · 9 stesso (ADJ) · 10 altro (ADJ)
- Quantificatori: 11 uno (NUM) · 12 due (NUM) · 13 alcuni (ADJ) ·
  14 tutto (ADJ) · 15 molti (ADJ) · 16 pochi (ADJ)
- Valutatori: 17 buono (ADJ) · 18 cattivo (ADJ)
- Descrittori: 19 grande (ADJ) · 20 piccolo (ADJ)
- Predicati mentali: 21 pensare (VERB) · 22 sapere (VERB) ·
  23 volere (VERB) · 24 non-volere (VERB) · 25 sentire (VERB) ·
  26 vedere (VERB) · 27 udire (VERB)
- Parola: 28 dire (VERB) · 29 parola (NOUN) · 30 vero (ADJ)
- Azioni ed eventi: 31 fare (VERB) · 32 accadere (VERB) ·
  33 muoversi (VERB)
- Esistenza e possesso: 34 trovarsi (VERB) · 35 esserci (VERB) ·
  36 essere (VERB) · 37 mio (ADJ)
- Vita e morte: 38 vivere (VERB) · 39 morire (VERB)
- Tempo: 40 quando (ADV) · 41 adesso (ADV) · 42 prima (ADV) ·
  43 dopo (ADV) · 44 molto-tempo (ADV) · 45 poco-tempo (ADV) ·
  46 per-un-po (ADV) · 47 momento (NOUN)
- Spazio: 48 dove (ADV) · 49 qui (ADV) · 50 sopra (ADV) · 51 sotto (ADV) ·
  52 lontano (ADV) · 53 vicino (ADV) · 54 lato (NOUN) · 55 dentro (ADV) ·
  56 toccare (VERB)
- Concetti logici: 57 non (ADV) · 58 forse (ADV) · 59 potere (VERB) ·
  60 perche (ADV) · 61 se (ADV)
- Intensificatore e aumentativo: 62 molto (ADV) · 63 piu (ADV)
- Similarità: 64 come (ADV)

Note: "perche" e "piu" senza accento nel lemma (i lemmi dei grafi non hanno
accenti: `domande.py` usa `quesito="perche"`); la SUPERFICIE scritta è
"perché"/"più". "molti" = MOLTO~MANY, "molto" = intensificatore VERY:
distinti apposta. `dire`, `trovarsi`, `esserci`, `essere`, `dove`, `perche`,
`uno`, `due`, `qualcuno`, `qualcosa`, `non`, `prima`, `dopo` sono primitivi
E parole usate dagli stampi: una sola riga, quella PRIM.

### 7.2 Il resto del lessico (dopo la riga 64, in quest'ordine di sezioni)

**Persone** (`PROPRIO`, tratti `genere=…`; superficie = lemma capitalizzato):
`anna` f, `piero` m, `maria` f, `marco` m, `sara` f, `luca` m.

**Luoghi** (`NOME`, tratti `genere=…,loc_in=…,loc_da=…` — valori nella
tabella 4.3): cucina f, salotto m, giardino m, camera f, orto m, bosco m.

**Oggetti unici** (`NOME`, tratti `genere=…,plurale=…`): pane m/pani,
palla f/palle, cestino m/cestini, scatola f/scatole, secchio m/secchi,
libro m/libri, camino m/camini.

**Risorse** (`NOME`): mela f, plurale=mele; acqua f, plurale=acque,
massa=si; legna f, plurale=legne, massa=si.

**Fonti** (`NOME`): melo m; pozzo m; bosco_legna m con tratti
`superficie=nel bosco` (è un identificatore del mondo: la resa in frase è
fissata dallo stampo 5, la riga serve perché è un lemma dei grafi).

**Verbi degli stampi** (`VERBO`, tratti con le forme usate; le azioni sono
lemmi dei grafi anche quando la superficie è di un altro verbo):
andare pres3s=va · prendere pres3s=prende · posare pres3s=posa ·
mettere_dentro superficie_pres3s=mette · tirare_fuori
superficie_pres3s=tira fuori · dare pres3s=dà, part=dato ·
mangiare pres3s=mangia · aprire pres3s=apre · chiudere pres3s=chiude ·
guardare pres3s=guarda · dormire pres3s=dorme · svegliarsi pres3s=si
sveglia · giocare pres3s=gioca · cercare pres3s=cerca · bruciare
pres3s=brucia · addormentarsi pres3s=si addormenta (solo superficie dello
stampo dormire) · raccogliere pres3s=raccoglie, part=raccolto · avere
pres3s=ha · portare pres3s=porta. (`trovarsi` pres3s=si trova, `essere`
pres3s=è, `esserci` pres3s=c'è, pres3p=ci sono, `dire` pres3s=dice: questi
quattro sono righe PRIM — le forme verbali si aggiungono nei loro tratti.)

**Parentela** (`REL`, tratti `superficie=…` dalla mappa in sezione 6):
padre_di, madre_di, figlio_di, figlia_di, marito_di, moglie_di,
fratello_di, sorella_di, nonno_di, nonna_di, nipote_di, suocero_di,
suocera_di, genero_di, nuora_di, zio_di, zia_di, cugino_di, cugina_di.

**Altri nomi** (`NOME`): oggetto m, plurale=oggetti; volta f,
plurale=volte; stanchezza f.

**Aggettivi** (`AGG`): stanco, tratti `femminile=stanca`.

**Interrogativi e speciali**: chi `INTERR` · quanti `INTERR` · quante
`INTERR` · che-parente `INTERR`, superficie=che parente · non-lo-so `SPEC`,
superficie=Non lo so. · nessuno `PRON`.

**Numeri** (`NUM`, tratti `valore=N`): zero(0), tre(3), quattro(4),
cinque(5), sei(6), sette(7), otto(8), nove(9), dieci(10), undici(11),
dodici(12), tredici(13), quattordici(14), quindici(15), sedici(16),
diciassette(17), diciotto(18), diciannove(19), venti(20), ventuno(21),
ventidue(22), ventitre(23), ventiquattro(24), venticinque(25),
ventisei(26), ventisette(27), ventotto(28), ventinove(29), trenta(30), poi
composizione regolare fino a settanta(70): trentuno, trentadue,
trentatre, … quaranta, … cinquanta, … sessanta, … settanta. ("uno" e "due"
sono già PRIM 11–12 e NON si ripetono; nei lemmi niente accento:
ventitre → superficie "ventitré", idem trentatre/quarantatre/….)

**Ordinali** (`ORD`, tratti `valore=N`, femminile regolare -o→-a):
primo(1), secondo(2), terzo(3), quarto(4), quinto(5), sesto(6),
settimo(7), ottavo(8), nono(9), decimo(10), undicesimo(11),
dodicesimo(12), tredicesimo(13), quattordicesimo(14), quindicesimo(15),
sedicesimo(16), diciassettesimo(17), diciottesimo(18),
diciannovesimo(19), ventesimo(20), ventunesimo(21), ventiduesimo(22),
ventitreesimo(23), ventiquattresimo(24), venticinquesimo(25),
ventiseiesimo(26), ventisettesimo(27), ventottesimo(28),
ventinovesimo(29), trentesimo(30).

**Parole funzione** (`FUNZ`): il, lo, la, i, gli, le, un, una, di, a, da,
in, con, su, per, e, che, ci, si, mai, intanto, alle.

**Definizioni**: per la v1 quasi tutte `TODO` (NON inventarle: le completerà
Andrea). Scrivere solo queste cinque, come esempio del formato:
- mela: `qualcosa di piccolo; la gente può mangiare questa cosa; è buono`
- mangiare: `qualcuno fa qualcosa con qualcosa; dopo, questa cosa è dentro il corpo di questo qualcuno`
- dare: `qualcuno ha qualcosa; questo qualcuno fa qualcosa; dopo, un altro qualcuno ha questa cosa`
- dormire: `qualcuno non fa niente per molto tempo; il corpo di questo qualcuno lo vuole`
- cucina: `parte della casa; qui la gente fa qualcosa; dopo, la gente può mangiare`

`lessico.py` espone `carica_lessico(percorso=…) -> Lessico` con accesso per
lemma e per categoria, e `Lessico.valida()` che verifica: 65 righe PRIM in
testa nell'ordine dato; lemmi unici; tratti ben formati; presenza di tutti i
lemmi richiesti dagli stampi (azioni di `mondo.azioni.AZIONI` + "bruciare",
persone/luoghi/oggetti/risorse/fonti di `dati_mondo.py` con genere coerente
per le persone, le 19 relazioni di parentela, i lemmi delle domande).

## 8. Morfologia — `morfologia.py`

Tutto guidato dal lessico, funzioni pure:

- `articolo_det(lemma, plurale=False) -> str` — regole: m sing "lo" se il
  lemma inizia per z/s+consonante/gn/ps, "l'" se vocale, altrimenti "il";
  m plur "gli" se vocale/z/s+consonante, altrimenti "i"; f sing "l'" se
  vocale altrimenti "la"; f plur "le". L'articolo elide attaccato:
  "l'acqua" (nessuno spazio dopo l'apostrofo).
- `articolo_indet(lemma) -> str` — m "uno" se z/s+consonante altrimenti
  "un"; f "un'" se vocale altrimenti "una".
- `partitivo(lemma) -> str` — per i nomi massa: "dell'acqua", "della legna"
  (di + articolo determinativo).
- `prep_articolata(prep, lemma) -> str` — combinazioni servite: a/da/di/in +
  il→al/dal/del/nel, +la→alla/dalla/della/nella, +l'→all'/dall'/dell'/nell'.
- `numero_in_lettere(n) / numero_da_lettere(s)` — 0–70, dalle righe NUM del
  lessico (più uno/due dai PRIM), superficie con accento per 23/33/43/….
- `ordinale(n, genere) / ordinale_inverso(s)` — 1–30, femminile in -a.
- `ora_in_lettere(t) -> str` — t=1→"all'una", t≥2→"alle "+numero. Inverso
  `ora_da_lettere`.
- `plurale(lemma) -> str` — dal tratto `plurale`.
- `forma_verbale(lemma, forma) -> str` — legge i tratti (pres3s, pres3p,
  part, superficie_pres3s).
- `aggettivo(lemma, genere) -> str` — "stanco"/"stanca".

## 9. Filtro — `filtro.py` e `regole_filtro.txt`

Formato di `regole_filtro.txt` (una regola per riga, `#` commenta):

```
# Segnaposto architetturali (FASE1.md): conta la sede, non il contenuto.
# Sintassi: radice=LEMMA rel=LEMMA [rel=LEMMA ...]
radice=mangiare obj=palla
radice=mangiare obj=libro
radice=bruciare obj=pane
```

Una regola scatta se: il lemma del nodo radice (il nodo id 0) è `radice` E
per ogni coppia `rel=LEMMA` esiste un arco dalla radice con quella relazione
verso un nodo con quel lemma (le istanze contano per lemma: `mela_2` ha
lemma "mela" ai fini del filtro — estrarre il lemma con la stessa regola
`lemma_N` della sezione 4.2). `filtra(grafo)` ritorna il primo esito;
`NON_LO_SO` è sempre ammesso. Non espandere la lista in questa fase.

Collocazione architetturale (per ora solo nei punti di verifica): il
round-trip e il CLI `verifica` chiamano `filtra` su ogni grafo in ingresso
(dopo `analizza`) e in uscita (prima di `verbalizza`) e falliscono se una
regola scatta — sul corpus del mondo non deve scattare mai.

## 10. CLI — `python -m lingua …`

- `campione-storia --seed 42` — genera la storia (via
  `mondo.generatore.genera_record`-equivalente: stessa lunghezza per seed,
  stesse domande), stampa il testo (frasi unite da spazio, a capo per tick),
  poi le domande e risposte verbalizzate.
- `campione-frasi --n 100 --seed 7` — estrae N frasi da storie con seed
  campionati da un RNG seedato in [0, 500) e le stampa (per la revisione
  umana di Andrea).
- `verifica --da 0 --a 10000` — per ogni seed: round-trip completo su
  eventi, domande e risposte + filtro + controllo accordo + controllo
  lemmi-fuori-lessico. Stampa OGNI fallimento (seed, frase, grafo atteso vs
  ottenuto) e un riepilogo (frasi totali, % esatte). Exit code 0 solo se
  ≥ 99,9% (l'obiettivo resta 100%: ogni fallimento va capito e corretto,
  non tollerato).

Usare SOLO seed < 1.000.000 (`mondo.generatore.SEED_ESAME_MINIMO`): i seed
d'esame non si usano nemmeno qui, per igiene.

## 11. Controlli automatici ausiliari

- **Accordo** (criterio FASE1): un checker indipendente dagli stampi che
  tokenizza ogni frase generata e, per ogni coppia
  articolo/preposizione-articolata + (ordinale) + nome, verifica genere,
  numero ed elisione contro il lessico. Deve passare su tutte le frasi del
  `verifica`.
- **Lemmi fuori lessico** (criterio FASE1): costruire una volta il
  "formario" = insieme di tutte le superfici lecite (parole FUNZ, forme
  flesse dei verbi del lessico, singolari/plurali dei nomi, ordinali,
  numeri, ore, nomi propri capitalizzati, superfici composte spezzate in
  parole, "perché", "più", "ventitré"…); tokenizzare ogni frase (split su
  spazi; staccare la punteggiatura `.?!,`; le forme elise come "l'", "un'",
  "dell'" si separano all'apostrofo tenendo l'apostrofo nel prefisso) e
  verificare che ogni token (case-insensitive tranne i nomi propri)
  appartenga al formario.

## 12. Test — `tests/test_lingua.py`

Con pytest, nella `.venv` locale (vedi sezione 14). Gruppi richiesti:

1. **Lessico**: `valida()` passa; 65 PRIM in testa nell'ordine normativo;
   copertura dei lemmi di `dati_mondo`/azioni/parentela/domande; generi
   persone coerenti con `dati_mondo.PERSONE`.
2. **Morfologia**: casi puntuali di articoli/elisione ("l'acqua", "lo zio",
   "gli oggetti", "un'…", "nell'orto"); numeri e ordinali round-trip su
   tutto il dominio (0–70, 1–30); ore round-trip 1–24.
3. **Golden eventi** (sequenza A, contesto condiviso — le frasi attese sono
   NORMATIVE, il test le confronta con `==`):

   | Evento (t, azione, campi) | frase attesa |
   |---|---|
   | 9, andare, sara, orig=cucina, dest=giardino | Alle nove Sara va dalla cucina in giardino. |
   | 9, prendere, piero, ogg=mela_1, arg=melo, luogo=orto | Intanto Piero raccoglie una mela dal melo. |
   | 10, giocare, sara, ogg=palla, luogo=giardino | Alle dieci Sara gioca con la palla. |
   | 10, prendere, anna, ogg=acqua_1, arg=pozzo, luogo=orto | Intanto Anna prende dell'acqua dal pozzo. |
   | 11, prendere, piero, ogg=mela_2, arg=melo, luogo=orto | Alle undici Piero raccoglie una mela dal melo. |
   | 11, mettere_dentro, piero, ogg=mela_1, arg=cestino, luogo=orto | Intanto Piero mette la prima mela nel cestino. |
   | 12, andare, piero, orig=orto, dest=giardino | Alle dodici Piero va in giardino. |
   | 12, dare, piero, ogg=mela_2, dest=sara, luogo=giardino | Intanto Piero dà la seconda mela a Sara. |
   | 13, mangiare, sara, ogg=mela_2, luogo=giardino | Alle tredici Sara mangia la seconda mela. |
   | 14, dormire, luca, luogo=camera, arg=stanchezza | Alle quattordici Luca si addormenta in camera perché è stanco. |
   | 16, svegliarsi, luca, luogo=camera | Alle sedici Luca si sveglia. |
   | 16, cercare, maria, ogg=libro, luogo=salotto | Intanto Maria cerca il libro in salotto. |
   | 17, aprire, maria, ogg=scatola, luogo=salotto | Alle diciassette Maria apre la scatola. |
   | 17, dire, maria, dest=marco, luogo=salotto | Intanto Maria dice qualcosa a Marco. |
   | 18, tirare_fuori, marco, ogg=pane, arg=scatola, luogo=salotto | Alle diciotto Marco tira fuori il pane dalla scatola. |
   | 18, guardare, sara, ogg=palla, luogo=giardino | Intanto Sara guarda la palla. |
   | 19, chiudere, maria, ogg=scatola, luogo=salotto | Alle diciannove Maria chiude la scatola. |
   | 19, posare, marco, ogg=pane, luogo=salotto | Intanto Marco posa il pane. |

   (Nota il tick 15 vuoto tra "si addormenta" e "si sveglia".)

   **Sequenza B** (contesto nuovo — catena causale del camino):

   | Evento | frase attesa |
   |---|---|
   | 6, prendere, marco, ogg=legna_1, arg=bosco_legna, luogo=bosco | Alle sei Marco raccoglie della legna nel bosco. |
   | 7, andare, marco, orig=bosco, dest=salotto | Alle sette Marco va in salotto. |
   | 8, mettere_dentro, marco, ogg=legna_1, arg=camino, luogo=salotto | Alle otto Marco mette la legna nel camino. |
   | 8, bruciare, camino, ogg=legna_1, luogo=salotto | Intanto il camino brucia la legna. |

4. **Golden domande/risposte** (contesto: fine sequenza A):
   "Dove si trova la prima mela?" / "La prima mela è nell'orto." ·
   "Chi ha il secchio?" / "Nessuno ha il secchio." ·
   "Quanti oggetti porta Sara?" / "Sara porta un oggetto." e il caso
   "Sara non porta nessun oggetto." · "Quanti oggetti ci sono nel
   cestino?" / "Nel cestino c'è un oggetto." · "Quanti oggetti ci sono in
   cucina?" / "In cucina ci sono tre oggetti." · "Chi ha dato la seconda
   mela a Sara?" / "Piero ha dato la seconda mela a Sara." · "Che parente è
   Anna di Maria?" / "Anna è la madre di Maria." (più un caso "lo zio" per
   l'elisione e uno "nipote" per l'articolo dal genere) · "Dove si trova la
   seconda mela che qualcuno ha dato a Sara?" / "Non lo so." · "Perché Luca
   dorme?" / "Luca dorme perché è stanco." · "Quante mele sono state
   raccolte?" / "Sono state raccolte due mele." · "Quante volte è stata
   raccolta l'acqua?" / "L'acqua è stata raccolta una volta."
   Per ognuno: verbalizza → confronto stringa, poi analizza → confronto
   grafo con la ricostruzione normativa della sezione 6.
5. **Round-trip di massa (test veloce)**: seed 0–299, eventi + domande +
   risposte, 100% esatto, filtro mai violato. (Il run completo 0–9999 è del
   CLI `verifica`, non di pytest.)
6. **Accordo e formario** sui seed 0–99.
7. **Filtro**: `filtra(grafo_fatto("mangiare", nsubj="sara", obj="palla"))`
   scatta (e con `obj="palla"` sostituito da un'istanza `mela_2` di lemma
   "mela" NON scatta la regola della palla); `NON_LO_SO` ammesso; grafi del
   corpus tutti ammessi.
8. **Determinismo**: due esecuzioni indipendenti su seed 42 producono liste
   di frasi identiche byte per byte.
9. **Errori chiari**: frase malformata → `ValueError` con messaggio utile
   (mai un match silenziosamente sbagliato).

## 13. Tappe di lavoro (con cancelli)

Procedere in quest'ordine; non passare alla tappa successiva finché il
cancello non è verde. Al termine di ogni tappa proporre un commit
(`git add` mirato + messaggio "Fase 1: …").

- **T1** `lessico.tsv` + `lessico.py` + test gruppo 1.
- **T2** `morfologia.py` + test gruppo 2.
- **T3** `contesto.py` + `stampi.py` + `verbalizza.py` per gli EVENTI +
  golden 3; stampare a schermo 20 frasi di esempio dal seed 42 per un primo
  sguardo di Andrea.
- **T4** `analizza.py` per gli eventi + round-trip eventi sui seed 0–299 al
  100%.
- **T5** stampi domande/risposte (verbalizza + analizza) + golden 4 +
  round-trip completo (gruppo 5).
- **T6** `filtro.py` + `regole_filtro.txt` + test gruppo 7; integrazione del
  filtro nel round-trip.
- **T7** checker di accordo e formario (gruppi 6, 8, 9) + `__main__.py`;
  eseguire `verifica --da 0 --a 10000` e riportare il risultato; eseguire
  `campione-frasi --n 100 --seed 7` e INCOLLARE l'output nella risposta
  finale per la revisione umana di Andrea.

Criteri di accettazione finali = quelli di FASE1.md, misurati così:
round-trip ≥ 99,9% (target 100%) su 10.000 storie via CLI `verifica`;
accordo automatico e formario senza violazioni; 100 frasi stampate per
Andrea; nessun lemma fuori lessico.

## 14. Ambiente e regole di condotta

- Sulla macchina di Andrea non c'è pip/venv di sistema: usare la venv locale
  del progetto. Se `.venv/` non esiste: `python3 -m venv .venv &&
  .venv/bin/pip install pytest`. Eseguire i test con
  `.venv/bin/python -m pytest tests/ -q`.
- Vietato: LLM o modelli pre-addestrati in qualunque ruolo; `random`
  globale; addestrare o "provare" su seed ≥ 1.000.000; testo italiano
  dentro `mondo/`; modifiche a `mondo/` (se qualcosa lì sembra sbagliato o
  manca, FERMARSI e chiedere ad Andrea).
- Non aggiungere costrutti "per completezza": ogni regola di lingua deve
  essere usata da almeno una frase del corpus o degli stampi (FASE1.md).
- Non inventare definizioni NSM oltre le cinque date: scrivere `TODO`.
- La morfologia italiana completa è fuori scopo: coprire SOLO ciò che gli
  stampi producono.
- Se un caso reale del simulatore non rientra negli stampi di questo piano
  (capita se il piano ha un buco): non improvvisare una regola nuova —
  riportare il caso ad Andrea con seed e frase.

## 15. Rimandato esplicitamente (non farlo ora)

- Refactor di `mondo/` perché legga il lessico (oggi i lemmi vivono in
  `dati_mondo.py`; il test di coerenza basta).
- Completamento delle definizioni NSM (Andrea).
- Imperfetto, pronomi clitici, seconda pelle linguistica, contenuto vero
  del filtro.
