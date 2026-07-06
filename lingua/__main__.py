"""CLI di lingua/ (FASE1_PIANO.md §10): campione-storia, campione-frasi,
verifica. Unico punto ammesso per un RNG con seed esplicito (le storie e le
domande restano generate da `mondo/`, che non conosce `lingua/`)."""
from __future__ import annotations

import argparse
import random
import sys

from mondo.domande import genera_domande
from mondo.generatore import N_PER_TIPO_DEFAULT, SEED_ESAME_MINIMO, _lunghezza_storia
from mondo.grafo import evento_a_grafo
from mondo.simulatore import genera_storia

from .accordo import controlla_accordo
from .analizza import analizza_domanda, analizza_evento, analizza_risposta
from .contesto import StatoDiscorso
from .filtro import filtra
from .formario import parole_fuori_formario
from .verbalizza import verbalizza_domanda, verbalizza_evento, verbalizza_risposta

SOGLIA_ACCETTAZIONE = 0.999


def _controlla_seed(seed: int) -> None:
    if seed >= SEED_ESAME_MINIMO:
        raise SystemExit(f"seed {seed} riservato agli esami (>= {SEED_ESAME_MINIMO}): rifiutato")


def _genera_storia_e_domande(seed: int):
    _controlla_seed(seed)
    n_tick = _lunghezza_storia(seed)
    storia = genera_storia(seed=seed, n_tick=n_tick)
    rng_domande = random.Random(f"domande-{seed}")
    domande = genera_domande(storia, rng_domande, n_per_tipo=N_PER_TIPO_DEFAULT)
    return storia, domande


def _comando_campione_storia(args: argparse.Namespace) -> int:
    storia, domande = _genera_storia_e_domande(args.seed)
    contesto = StatoDiscorso()
    tick_corrente = None
    riga: list[str] = []
    for evento in storia.eventi:
        frase = verbalizza_evento(evento_a_grafo(evento), contesto)
        if tick_corrente is not None and evento.t != tick_corrente:
            print(" ".join(riga))
            riga = []
        riga.append(frase)
        tick_corrente = evento.t
    if riga:
        print(" ".join(riga))

    print()
    for d in domande:
        print(verbalizza_domanda(d.grafo_domanda, contesto))
        print(verbalizza_risposta(d.grafo_risposta, contesto))
    return 0


def _comando_campione_frasi(args: argparse.Namespace) -> int:
    rng = random.Random(args.seed)
    frasi: list[str] = []
    while len(frasi) < args.n:
        seed_storia = rng.randrange(500)
        storia, _domande = _genera_storia_e_domande(seed_storia)
        contesto = StatoDiscorso()
        for evento in storia.eventi:
            frasi.append(verbalizza_evento(evento_a_grafo(evento), contesto))
    for frase in frasi[:args.n]:
        print(frase)
    return 0


def _verifica_evento(evento, cv: StatoDiscorso, cp: StatoDiscorso, seed: int, fallimenti: list[str]) -> bool:
    grafo_atteso = evento_a_grafo(evento)
    filtro_uscita = filtra(grafo_atteso)
    frase = verbalizza_evento(grafo_atteso, cv)
    errori_accordo = controlla_accordo(frase)
    fuori_lessico = parole_fuori_formario(frase)
    try:
        grafo_ottenuto = analizza_evento(frase, cp)
    except ValueError as e:
        fallimenti.append(f"seed {seed}: evento {evento.t} {evento.azione!r}: '{frase}' -> errore di parsing: {e}")
        return False
    filtro_ingresso = filtra(grafo_ottenuto)
    ok = (grafo_ottenuto == grafo_atteso and filtro_uscita.ammesso and filtro_ingresso.ammesso
          and not errori_accordo and not fuori_lessico)
    if not ok:
        fallimenti.append(
            f"seed {seed}: evento {evento.t} {evento.azione!r}: '{frase}' "
            f"grafo_atteso={grafo_atteso} grafo_ottenuto={grafo_ottenuto} "
            f"filtro_uscita={filtro_uscita} filtro_ingresso={filtro_ingresso} "
            f"accordo={errori_accordo} fuori_lessico={fuori_lessico}"
        )
    return ok


def _verifica_domanda_o_risposta(grafo, rendi, analizza, cv, cp, seed, etichetta, fallimenti) -> bool:
    filtro_uscita = filtra(grafo)
    frase = rendi(grafo, cv)
    errori_accordo = controlla_accordo(frase)
    fuori_lessico = parole_fuori_formario(frase)
    try:
        grafo_ottenuto = analizza(frase, cp)
    except ValueError as e:
        fallimenti.append(f"seed {seed}: {etichetta}: '{frase}' -> errore di parsing: {e}")
        return False
    filtro_ingresso = filtra(grafo_ottenuto)
    ok = (grafo_ottenuto == grafo and filtro_uscita.ammesso and filtro_ingresso.ammesso
          and not errori_accordo and not fuori_lessico)
    if not ok:
        fallimenti.append(
            f"seed {seed}: {etichetta}: '{frase}' grafo_atteso={grafo} grafo_ottenuto={grafo_ottenuto} "
            f"filtro_uscita={filtro_uscita} filtro_ingresso={filtro_ingresso} "
            f"accordo={errori_accordo} fuori_lessico={fuori_lessico}"
        )
    return ok


def _comando_verifica(args: argparse.Namespace) -> int:
    totale = 0
    esatte = 0
    fallimenti: list[str] = []
    for seed in range(args.da, args.a):
        storia, domande = _genera_storia_e_domande(seed)
        cv = StatoDiscorso()
        cp = StatoDiscorso()
        for evento in storia.eventi:
            totale += 1
            if _verifica_evento(evento, cv, cp, seed, fallimenti):
                esatte += 1
        for d in domande:
            totale += 1
            if _verifica_domanda_o_risposta(d.grafo_domanda, verbalizza_domanda, analizza_domanda,
                                             cv, cp, seed, f"domanda {d.tipo}", fallimenti):
                esatte += 1
            totale += 1
            if _verifica_domanda_o_risposta(d.grafo_risposta, verbalizza_risposta, analizza_risposta,
                                             cv, cp, seed, f"risposta {d.tipo}", fallimenti):
                esatte += 1

    for fallimento in fallimenti:
        print(fallimento)

    percentuale = (esatte / totale * 100) if totale else 100.0
    print(f"\nfrasi totali: {totale}  esatte: {esatte}  ({percentuale:.4f}%)")
    return 0 if totale and esatte / totale >= SOGLIA_ACCETTAZIONE else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m lingua")
    sotto = parser.add_subparsers(dest="comando", required=True)

    p_storia = sotto.add_parser("campione-storia")
    p_storia.add_argument("--seed", type=int, required=True)
    p_storia.set_defaults(funzione=_comando_campione_storia)

    p_frasi = sotto.add_parser("campione-frasi")
    p_frasi.add_argument("--n", type=int, default=100)
    p_frasi.add_argument("--seed", type=int, required=True)
    p_frasi.set_defaults(funzione=_comando_campione_frasi)

    p_verifica = sotto.add_parser("verifica")
    p_verifica.add_argument("--da", type=int, default=0)
    p_verifica.add_argument("--a", type=int, default=10_000)
    p_verifica.set_defaults(funzione=_comando_verifica)

    args = parser.parse_args(argv)
    return args.funzione(args)


if __name__ == "__main__":
    sys.exit(main())
