"""Adattatori: sistemi preesistenti misurati con lo stesso strumento.

FASE_MENTE.md §11, M0.3. Serve a mettere la storia del progetto su una scala
unica: finora ogni esperimento ha riportato il proprio numero, calcolato a
modo suo, e il confronto fra `v1` (0,573) e `jepa/` (0,646) si è rivelato
privo di senso.

Richiede torch solo per `jepa/`; il resto di `sonde/` resta stdlib puro.
"""
from __future__ import annotations

import random

from mondo.domande import genera_domande
from mondo.simulatore import genera_storia

from regole.tracker import NON_LO_SO as NON_LO_SO_REGOLE

from .banco import Esito, _entita_della_domanda, _oro, _oro_vero, _profondita, lunghezza


def esiti_jepa(seed_base: int = 2000, n_storie: int = 40, modello=None) -> list[Esito]:
    """Fa rispondere il modello `jepa/` alle stesse domande delle sonde.

    Il `jepa/` conosce classi, non istanze (`mela_3` -> riga `mela`): le
    domande su istanze diverse della stessa classe finiscono tutte sulla
    stessa credenza. Non è un adattamento sfavorevole, è il modello com'è —
    ed è esattamente ciò che la sonda P2 deve rendere visibile.

    Canale: solo lingua. Il `jepa/` legge gli eventi del simulatore, cioè la
    narrazione già disambiguata; non ha alcun canale percettivo.
    """
    import torch

    from jepa.addestra_jepa import estrai_evento_strutturato
    from jepa.vocabolario_grafi import ENTITA2ID, normalizza_entita

    if modello is None:
        raise ValueError("serve un GraphWorldJEPA già addestrato")

    esiti: list[Esito] = []
    with torch.no_grad():
        for i in range(n_storie):
            seed = seed_base + i
            n_tick = lunghezza(seed)
            storia = genera_storia(seed=seed, n_tick=n_tick)
            stato = storia.stato_finale

            eventi = [
                t for t in (estrai_evento_strutturato(ev) for ev in storia.eventi) if t is not None
            ]
            ev_tensor = (
                torch.tensor(eventi, dtype=torch.long)
                if eventi
                else torch.zeros((0, 3), dtype=torch.long)
            )
            logits = modello.inizializza_stato(batch_size=1)
            logits = modello.aggiorna_stato_sequenza(logits, ev_tensor)

            conta: dict[str, int] = {}
            for ogg in stato.oggetti.values():
                conta[ogg.lemma] = conta.get(ogg.lemma, 0) + 1

            for d in genera_domande(storia, random.Random(f"domande-{seed}"), n_per_tipo=8):
                if d.tipo not in ("posizione", "possesso"):
                    continue
                eid = _entita_della_domanda(d, stato)
                if eid is None:
                    continue
                normalizzato = normalizza_entita(eid)
                if normalizzato is None or normalizzato not in ENTITA2ID:
                    continue
                _, predetto = modello.calcola_energia_risposte(
                    logits, ENTITA2ID[normalizzato], tipo_domanda=d.tipo
                )
                classe = eid if eid in stato.persone else stato.oggetti[eid].lemma
                esiti.append(
                    Esito(
                        seed=seed,
                        tipo=d.tipo,
                        entita=eid,
                        oro=_oro(d, d.tipo),
                        oro_vero=_oro_vero(stato, eid, d.tipo),
                        predetto=predetto,
                        confidenza=1.0,  # il jepa/ non espone una confidenza calibrata
                        eta_evidenza=-1,  # nessun canale percettivo: P1 non si applica
                        ambigua=conta.get(classe, 0) > 1,
                        profondita=_profondita(stato, eid),
                        cast=len(stato.persone),
                    )
                )
    return esiti


def addestra_jepa_locale(seed_torch: int = 0):
    """Addestra il `jepa/` con seed esplicito (il suo script non ne ha uno,
    e infatti il numero riportato balla di un punto fra i giri)."""
    import torch
    import torch.nn.functional as F
    import torch.optim as optim

    from jepa.addestra_jepa import prepara_dataset_jepa
    from jepa.modello_jepa import GraphWorldJEPA
    from jepa.vocabolario_grafi import LUOGO2ID, N_LUOGHI, PERSONA2ID

    torch.manual_seed(seed_torch)
    train = prepara_dataset_jepa(n_storie=200, seed_base=100)
    modello = GraphWorldJEPA(d_embed=64, soglia_non_lo_so=0.45)
    ottimizzatore = optim.AdamW(modello.parameters(), lr=1e-2, weight_decay=1e-4)
    accumulo = 16

    for _ in range(16):
        modello.train()
        ottimizzatore.zero_grad()
        for idx, esempio in enumerate(train):
            logits = modello.inizializza_stato(batch_size=1)
            logits = modello.aggiorna_stato_sequenza(logits, esempio["eventi_tensor"])
            oro, s_idx, tipo = esempio["risposta_oro"], esempio["soggetto_id"], esempio["tipo_domanda"]
            bersaglio = None
            if tipo == "posizione" and oro in LUOGO2ID:
                bersaglio = LUOGO2ID[oro]
            elif tipo == "possesso" and oro in PERSONA2ID:
                bersaglio = N_LUOGHI + PERSONA2ID[oro]
            elif tipo == "possesso" and oro == "nessuno":
                prob = modello.ottieni_probabilita_effettive_luogo(logits)[0, s_idx, :]
                bersaglio = int(torch.argmax(prob).item())
            if bersaglio is not None:
                perdita = F.cross_entropy(
                    logits[0, s_idx, :].unsqueeze(0), torch.tensor([bersaglio])
                ) / accumulo
                perdita.backward()
            if (idx + 1) % accumulo == 0:
                ottimizzatore.step()
                ottimizzatore.zero_grad()
        ottimizzatore.step()
        ottimizzatore.zero_grad()

    modello.eval()
    return modello


# ---------------------------------------------------------------------------
# `v1`: il transformer autoregressivo della Fase 2a
# ---------------------------------------------------------------------------

# Checkpoint scaricati da Colab. `v1` è la baseline storica (0,573 d'esame);
# `v1_grad2` è il miglior checkpoint del progetto (0,904), quello che gira in
# `interfaccia/`.
CHECKPOINT_V1 = {
    "v1": "/home/andrea/Scaricati/v1/stadio1.pt",
    "v1_facile": "/home/andrea/Scaricati/v1_facile/stadio1.pt",
    "v1_anti": "/home/andrea/Scaricati/v1_anti/stadio1_best.pt",
    "v1_grad1": "/home/andrea/Scaricati/v1_grad1/stadio1_best.pt",
    "v1_grad2": "/home/andrea/Scaricati/v1_grad2/stadio1_best.pt",
    "v1_grad3": "/home/andrea/Scaricati/v1_grad3/stadio1_best.pt",
}


def _carica_modello_epoca(config: dict, checkpoint: str, device: str, vocab):
    """Carica un checkpoint di `v1` con la dimensione di vocabolario che aveva
    quando è stato addestrato.

    Il vocabolario è cresciuto dopo quei run (`che-cosa` per l'esperimento
    "tempo", `[STATO]` per la Fase B), ma i token nuovi sono stati APPESI in
    coda: gli id 0..280 non si sono spostati, quindi il checkpoint vecchio e il
    vocabolario attuale sono compatibili sul prefisso comune. La guardia qui
    sotto verifica l'ipotesi invece di darla per buona, e la decodifica non può
    comunque emettere i token nuovi: la testa del modello ha meno logit.
    """
    import torch

    from cervello.modello import ConfigModello, Modello

    stato = torch.load(checkpoint, map_location=device)
    dimensione = stato["modello"]["tok_emb.weight"].shape[0]
    if dimensione > vocab.dimensione:
        raise ValueError(
            f"checkpoint con vocabolario piu grande dell'attuale "
            f"({dimensione} > {vocab.dimensione}): non e un'estensione, non caricabile"
        )
    cfg = ConfigModello(vocab_size=dimensione, ctx=config["dataset"]["ctx"], **config["modello"])
    modello = Modello(cfg).to(device)
    modello.load_state_dict(stato["modello"])
    modello.eval()
    return modello, dimensione


def esiti_v1(
    checkpoint: str,
    seed_base: int = 1_000_000,
    n_storie: int = 60,
    config_percorso: str | None = None,
) -> list[Esito]:
    """Fa rispondere `v1` alle stesse domande delle sonde, sulle SUE storie.

    Lo stadio 1 di `v1` è addestrato su storie di 3-6 tick e sole domande di
    posizione (`configs/v1.yaml`): valutarlo su storie di 8-22 tick sarebbe
    fuori distribuzione e il confronto direbbe poco. Qui si usano
    `lunghezza_stadio1` e la finestra di seed d'esame (>= 1.000.000), cioè
    esattamente il regime in cui `v1` ha prodotto lo 0,573.

    Solo domande di posizione: `v1` allo stadio 1 non ne conosce altre.
    """
    import torch

    from cervello.sequenza import DOMANDA, FINE, RISPOSTA, STORIA, grafo_a_token, token_a_grafo
    from cervello.vocabolario import carica_vocabolario
    from esami.esamina import decodifica_greedy, dispositivo
    from esami.genera import carica_config
    from mondo.grafo import evento_a_grafo

    from .banco import LUOGHI, lunghezza_stadio1

    config = carica_config(config_percorso) if config_percorso else carica_config()
    device = dispositivo(config)
    ctx = config["dataset"]["ctx"]
    vocab = carica_vocabolario()
    modello, dimensione_epoca = _carica_modello_epoca(config, checkpoint, device, vocab)

    esiti: list[Esito] = []
    with torch.no_grad():
        for i in range(n_storie):
            seed = seed_base + i
            storia = genera_storia(seed=seed, n_tick=lunghezza_stadio1(seed))
            stato = storia.stato_finale
            token_storia: list[str] = []
            for ev in storia.eventi:
                token_storia.extend(grafo_a_token(evento_a_grafo(ev)))

            conta: dict[str, int] = {}
            for ogg in stato.oggetti.values():
                conta[ogg.lemma] = conta.get(ogg.lemma, 0) + 1

            for d in genera_domande(storia, random.Random(f"domande-{seed}"), n_per_tipo=8):
                if d.tipo != "posizione":
                    continue
                eid = _entita_della_domanda(d, stato)
                if eid is None:
                    continue
                prefisso = [
                    STORIA, *token_storia, DOMANDA, *grafo_a_token(d.grafo_domanda), RISPOSTA
                ]
                ids = [vocab.id(t) for t in prefisso]
                if max(ids) >= dimensione_epoca:
                    # la domanda usa un token che il checkpoint non ha mai visto
                    continue
                generati = [
                    vocab.token(i) for i in decodifica_greedy(modello, vocab, ids, ctx, device)
                ]
                if generati and generati[-1] == FINE:
                    generati = generati[:-1]
                try:
                    grafo = token_a_grafo(generati, "fatto")
                    predetto = next(
                        (n.lemma for n in grafo.nodi if n.lemma in LUOGHI), NON_LO_SO_REGOLE
                    )
                    if any(n.lemma == "non-lo-so" for n in grafo.nodi):
                        predetto = NON_LO_SO_REGOLE
                except ValueError:
                    predetto = "«malformata»"  # conta come errore su entrambe le scale

                classe = eid if eid in stato.persone else stato.oggetti[eid].lemma
                esiti.append(
                    Esito(
                        seed=seed,
                        tipo="posizione",
                        entita=eid,
                        oro=_oro(d, "posizione"),
                        oro_vero=_oro_vero(stato, eid, "posizione"),
                        predetto=predetto,
                        confidenza=1.0,  # decodifica greedy: nessuna confidenza esposta
                        eta_evidenza=-1,  # nessun canale percettivo
                        ambigua=conta.get(classe, 0) > 1,
                        profondita=_profondita(stato, eid),
                        cast=len(stato.persone),
                    )
                )
    return esiti
