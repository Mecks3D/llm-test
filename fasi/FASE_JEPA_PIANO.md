# PIANO ESECUTIVO — Esperimento Graph-JEPA World Model

## 0. Motivazione e Obiettivo

L'esperimento nasce dalla presa di coscienza che i Transformer Autoregressivi ($v1$) soffrono di problemi strutturali nel tracciamento di uno stato discreto:
1. **Explosione del contesto** e decodifica sequenziale lenta.
2. **Interferenza e perdita di binding** tra più personaggi al crescere del cast ($0.573$ di accuratezza su cast pieno).
3. **Mancanza di vincoli fisici immutabili** (il modello tenta di apprendere la fisica del mondo solo per via statistica dai token).

Il **Graph-JEPA World Model** risponde sostituendo il Transformer con un'architettura Neuro-Simbolica:
- **Struttura**: Il mondo è un grafo di entità e relazioni.
- **Fisica cablata**: Vincoli matematici nell'architettura (es. un'entità non può essere in due posti contemporaneamente).
- **JEPA / Energy Engine**: Invece di generare token, il modello aggiorna uno stato latente e misura l'energia di compatibilità delle risposte.

---

## 1. Architettura del Modello (`jepa/modello_jepa.py`)

### 1.1 Rappresentazione dello Stato ($Z$)
Per un vocabolario di $N_e$ entità (personaggi e oggetti) e $N_l$ luoghi:
- Matrice di posizioni $\mathbf{P} \in \mathbb{R}^{N_e \times N_l}$.
- Vincolo di esclusività: $\mathbf{P}_{i, :} = \text{softmax}(\mathbf{W}_i)$, garantendo che $\sum_l \mathbf{P}_{i,l} = 1$.

### 1.2 Transizione (GNN Predictor)
Per ogni evento $e = (\text{verbo}, \text{soggetto}, \text{destinazione})$:
- Calcolo dell'impulso $\Delta$.
- Aggiornamento dello stato $\mathbf{P}_{t+1} = (1 - \alpha) \mathbf{P}_t + \alpha \Delta$.

### 1.3 Energy Head (Risposta e Epistemic Honesty)
Data una domanda $Q = (\text{soggetto}, \text{quesito}=\text{dove})$ e una risposta candidata $R = \text{luogo}$:
- Energia $E(Q, R) = 1.0 - \mathbf{P}_{\text{soggetto}, \text{luogo}}$.
- Se $\max_l \mathbf{P}_{\text{soggetto}, l} < \text{soglia}$ (stato incerto/non menzionato), l'energia minima viene assegnata alla risposta `NON_LO_SO`.

---

## 2. Organizzazione File

- `jepa/`: Cartella isolata per l'esperimento.
  - `jepa/vocabolario_grafi.py`: Parser deterministico da frasi UD a entità/luoghi.
  - `jepa/modello_jepa.py`: Modello PyTorch del Graph-JEPA (~200k parametri).
  - `jepa/addestra_jepa.py`: Script di addestramento e valutazione locale su CPU.
- `tests/test_jepa.py`: Test unitari di correttezza matematica.

---

## 3. Criteri di Accettazione e Valutazione

- **Esecuzione Locale**: Deve girare interamente su CPU in < 1 minuto per 300 storie.
- **Accuratezza Posizione (Stadio 1)**: Target $\ge 0.90$ sul benchmark ufficiale d'esame.
- **Zero Interferenza**: Nessun degrado di accuratezza passando da 1 personaggio a 6 personaggi nel cast.
