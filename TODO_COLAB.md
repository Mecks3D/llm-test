# TODO: training su Google Colab

Andrea vuole spostare le fasi di training intensive (PyTorch, `cervello/`)
su Google Colab, perché il suo PC ha solo CPU. Vuole che sia Claude a
occuparsi di tutta la configurazione, non farla manualmente lui.

## Stato attuale

- Il repo **non è ancora su GitHub** (solo locale, git init già fatto).
- Nessun notebook Colab esiste ancora.

## Da fare nella prossima conversazione

1. Creare un repository GitHub (chiedere ad Andrea se privato o pubblico)
   e pushare il codice esistente. Verificare che `dati/` non venga
   committato (per convenzione di progetto, contiene solo dati generati).
2. Scrivere un notebook `.ipynb` (o script pensato per Colab) che:
   - clona il repo da GitHub;
   - seleziona/verifica il runtime GPU;
   - installa le dipendenze necessarie (PyTorch + eventuali altre di
     `cervello/`);
   - monta Google Drive per salvare i checkpoint (la sessione Colab
     perde tutto alla scadenza);
   - lancia il training con gli stessi comandi/seed usati in locale
     (il progetto richiede determinismo: seed espliciti, mai `random`
     globale — nessuna modifica al codice serve per girare su Colab).
3. Verificare con Andrea quale fase/comando di training va effettivamente
   lanciato (dipende da dove sarà arrivato lo sviluppo di `cervello/`).
