"""Finestra tkinter, punto d'ingresso (`python -m interfaccia.app`).

fasi/INTERFACCIA_PIANO.md §5: carica un checkpoint già addestrato, genera
una storia da un seed, mostra il testo in italiano, fa rispondere il
modello a una domanda candidata e confronta con la risposta corretta.
Nessuna logica qui: tutto passa da `interfaccia.ponte`.
"""
from __future__ import annotations

import argparse
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from mondo import dati_mondo as dm

from . import ponte

COLORE_ESATTO = "#1a7f37"
COLORE_ERRORE = "#c0392b"
COLORE_CATEGORIE = {
    "esatto": COLORE_ESATTO,
    "invenzione": COLORE_ERRORE,
    "astensione_errata": COLORE_ERRORE,
    "malformata": COLORE_ERRORE,
    "errore": COLORE_ERRORE,
}

PERCORSO_CONFIG_DEFAULT = str(Path(__file__).resolve().parent.parent / "configs" / "v1.yaml")


class App(tk.Tk):
    def __init__(self, permetti_seed_esame: bool) -> None:
        super().__init__()
        self.title("Cervello-Bambino — interfaccia dimostrativa")
        self.geometry("900x700")

        self._permetti_seed_esame = permetti_seed_esame
        self._config: dict | None = None
        self._device: str | None = None
        self._modello = None
        self._vocab = None
        self._storia_gen: ponte.StoriaGenerata | None = None
        self._domande: list[ponte.DomandaMostrata] = []
        self._var_cast: dict[str, tk.BooleanVar] = {}

        self._costruisci_sezione_caricamento()
        self._costruisci_sezione_storia()
        self._costruisci_area_storia()
        self._costruisci_sezione_domande()
        self._costruisci_sezione_risposta()

    # -- sezione 1: checkpoint/config -----------------------------------

    def _costruisci_sezione_caricamento(self) -> None:
        f = ttk.LabelFrame(self, text="Checkpoint")
        f.pack(fill="x", padx=8, pady=3)

        ttk.Label(f, text="Checkpoint (.pt):").grid(row=0, column=0, sticky="w")
        self._entry_checkpoint = ttk.Entry(f, width=50)
        self._entry_checkpoint.grid(row=0, column=1, sticky="we", padx=4)
        ttk.Button(f, text="Sfoglia...", command=self._sfoglia_checkpoint).grid(row=0, column=2)

        ttk.Label(f, text="Config (.yaml):").grid(row=1, column=0, sticky="w")
        self._entry_config = ttk.Entry(f, width=50)
        self._entry_config.insert(0, PERCORSO_CONFIG_DEFAULT)
        self._entry_config.grid(row=1, column=1, sticky="we", padx=4)
        ttk.Button(f, text="Sfoglia...", command=self._sfoglia_config).grid(row=1, column=2)

        ttk.Label(f, text="Stadio:").grid(row=2, column=0, sticky="w")
        self._entry_stadio = ttk.Entry(f, width=5)
        self._entry_stadio.insert(0, "1")
        self._entry_stadio.grid(row=2, column=1, sticky="w", padx=4)

        ttk.Button(f, text="Carica", command=self._carica).grid(row=2, column=2)

        self._label_stato = ttk.Label(f, text="(nessun checkpoint caricato)")
        self._label_stato.grid(row=3, column=0, columnspan=3, sticky="w", pady=(4, 0))

        f.columnconfigure(1, weight=1)

    def _sfoglia_checkpoint(self) -> None:
        percorso = filedialog.askopenfilename(filetypes=[("Checkpoint PyTorch", "*.pt"), ("Tutti i file", "*")])
        if percorso:
            self._entry_checkpoint.delete(0, tk.END)
            self._entry_checkpoint.insert(0, percorso)

    def _sfoglia_config(self) -> None:
        percorso = filedialog.askopenfilename(filetypes=[("Config YAML", "*.yaml"), ("Tutti i file", "*")])
        if percorso:
            self._entry_config.delete(0, tk.END)
            self._entry_config.insert(0, percorso)

    def _carica(self) -> None:
        percorso_checkpoint = self._entry_checkpoint.get().strip()
        percorso_config = self._entry_config.get().strip()
        if not percorso_checkpoint:
            messagebox.showerror("Errore", "Indica il percorso del checkpoint (.pt).")
            return
        try:
            stadio = int(self._entry_stadio.get().strip())
        except ValueError:
            messagebox.showerror("Errore", "Lo stadio deve essere un numero intero.")
            return

        try:
            config = ponte.carica_config(percorso_config)
            if stadio not in config["stadi"]:
                raise ValueError(f"stadio {stadio} non definito nel config (disponibili: {sorted(config['stadi'])})")
            device = ponte.dispositivo(config)
            modello, vocab = ponte.carica_modello(config, percorso_checkpoint, device)
        except Exception as exc:  # noqa: BLE001 - mostrato all'utente, non un crash silenzioso
            messagebox.showerror("Caricamento fallito", str(exc))
            return

        self._config = config
        self._stadio = stadio
        self._device = device
        self._modello = modello
        self._vocab = vocab
        self._label_stato.configure(
            text=f"Caricato: {Path(percorso_checkpoint).name}  (stadio {stadio}, device {device})"
        )
        self._aggiorna_cast_da_config()
        self._abilita_sezione_storia(True)

    # -- sezione 2: seed / cast / tick -----------------------------------

    def _costruisci_sezione_storia(self) -> None:
        f = ttk.LabelFrame(self, text="Storia")
        f.pack(fill="x", padx=8, pady=3)
        self._frame_storia_controlli = f

        ttk.Label(f, text="Seed:").grid(row=0, column=0, sticky="w")
        self._entry_seed = ttk.Entry(f, width=12)
        self._entry_seed.insert(0, "7")
        self._entry_seed.grid(row=0, column=1, sticky="w", padx=4)
        ttk.Button(f, text="Seed casuale", command=self._seed_casuale).grid(row=0, column=2, padx=4)

        ttk.Label(f, text="Cast:").grid(row=1, column=0, sticky="nw")
        frame_cast = ttk.Frame(f)
        frame_cast.grid(row=1, column=1, columnspan=2, sticky="w")
        for p in dm.PERSONE:
            var = tk.BooleanVar(value=True)
            self._var_cast[p.id] = var
            ttk.Checkbutton(frame_cast, text=p.lemma, variable=var).pack(side="left")

        self._var_modo_tick = tk.StringVar(value="auto")
        ttk.Label(f, text="Tick:").grid(row=2, column=0, sticky="w")
        frame_tick = ttk.Frame(f)
        frame_tick.grid(row=2, column=1, columnspan=2, sticky="w")
        ttk.Radiobutton(frame_tick, text="Auto (secondo lo stadio)", value="auto",
                         variable=self._var_modo_tick).pack(side="left")
        ttk.Radiobutton(frame_tick, text="Manuale:", value="manuale",
                         variable=self._var_modo_tick).pack(side="left")
        self._entry_tick = ttk.Entry(frame_tick, width=5)
        self._entry_tick.insert(0, "6")
        self._entry_tick.pack(side="left")

        self._bottone_genera_storia = ttk.Button(f, text="Genera storia", command=self._genera_storia)
        self._bottone_genera_storia.grid(row=3, column=0, columnspan=3, pady=(6, 0))

        self._abilita_sezione_storia(False)

    def _abilita_sezione_storia(self, abilitato: bool) -> None:
        stato = "normal" if abilitato else "disabled"
        for figlio in self._frame_storia_controlli.winfo_children():
            self._imposta_stato_ricorsivo(figlio, stato)

    def _imposta_stato_ricorsivo(self, widget, stato: str) -> None:
        try:
            widget.configure(state=stato)
        except tk.TclError:
            pass
        for figlio in widget.winfo_children():
            self._imposta_stato_ricorsivo(figlio, stato)

    def _aggiorna_cast_da_config(self) -> None:
        cast_config = self._config["dataset"].get("cast") if self._config else None
        for pid, var in self._var_cast.items():
            var.set(cast_config is None or pid in cast_config)

    def _seed_casuale(self) -> None:
        import random
        limite = 999_999 if not self._permetti_seed_esame else 9_999_999
        seed = random.randrange(limite)
        self._entry_seed.delete(0, tk.END)
        self._entry_seed.insert(0, str(seed))

    def _genera_storia(self) -> None:
        try:
            seed = int(self._entry_seed.get().strip())
        except ValueError:
            messagebox.showerror("Errore", "Il seed deve essere un numero intero.")
            return
        try:
            ponte.verifica_seed(seed, self._permetti_seed_esame)
        except ValueError as exc:
            messagebox.showerror("Seed rifiutato", str(exc))
            return

        id_scelti = [pid for pid, var in self._var_cast.items() if var.get()]
        if not id_scelti:
            messagebox.showerror("Errore", "Scegli almeno una persona nel cast.")
            return
        cast = ponte.cast_da_id(id_scelti) if len(id_scelti) < len(dm.PERSONE) else None

        if self._var_modo_tick.get() == "auto":
            n_tick = ponte.n_tick_auto(self._stadio, seed, self._config)
        else:
            try:
                n_tick = int(self._entry_tick.get().strip())
            except ValueError:
                messagebox.showerror("Errore", "Il numero di tick manuale deve essere un intero.")
                return

        self._storia_gen = ponte.genera_storia_e_testo(seed=seed, n_tick=n_tick, cast=cast)
        self._mostra_storia()

        tipi_ammessi = set(self._config["stadi"][self._stadio]["tipi"])
        self._domande = ponte.domande_candidate(self._storia_gen, seed, tipi_ammessi)
        self._mostra_domande()
        self._pulisci_risposta()

    # -- area testo storia -------------------------------------------------

    def _costruisci_area_storia(self) -> None:
        f = ttk.LabelFrame(self, text="Storia (italiano)")
        f.pack(fill="both", expand=True, padx=8, pady=3)
        self._testo_storia = tk.Text(f, height=6, wrap="word")
        self._testo_storia.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(f, command=self._testo_storia.yview)
        scroll.pack(side="right", fill="y")
        self._testo_storia.configure(yscrollcommand=scroll.set, state="disabled")

    def _mostra_storia(self) -> None:
        self._testo_storia.configure(state="normal")
        self._testo_storia.delete("1.0", tk.END)
        self._testo_storia.insert(tk.END, "\n".join(self._storia_gen.righe_per_tick))
        self._testo_storia.configure(state="disabled")

    # -- sezione domande ----------------------------------------------------

    def _costruisci_sezione_domande(self) -> None:
        f = ttk.LabelFrame(self, text="Domande candidate")
        f.pack(fill="both", expand=True, padx=8, pady=3)

        self._lista_domande = tk.Listbox(f, height=5)
        self._lista_domande.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(f, command=self._lista_domande.yview)
        scroll.pack(side="left", fill="y")
        self._lista_domande.configure(yscrollcommand=scroll.set)

        ttk.Button(f, text="Chiedi al modello", command=self._chiedi_al_modello).pack(side="left", padx=6)

    def _mostra_domande(self) -> None:
        self._lista_domande.delete(0, tk.END)
        for d in self._domande:
            self._lista_domande.insert(tk.END, f"[{d.difficolta}] {d.testo_domanda}")

    # -- sezione risposta ----------------------------------------------------

    def _costruisci_sezione_risposta(self) -> None:
        f = ttk.LabelFrame(self, text="Risposta")
        f.pack(fill="x", padx=8, pady=3)

        ttk.Label(f, text="Risposta del modello:").grid(row=0, column=0, sticky="w")
        self._label_risposta_modello = ttk.Label(f, text="—")
        self._label_risposta_modello.grid(row=0, column=1, sticky="w")

        ttk.Label(f, text="Risposta corretta:").grid(row=1, column=0, sticky="w")
        self._label_risposta_oro = ttk.Label(f, text="—")
        self._label_risposta_oro.grid(row=1, column=1, sticky="w")

        self._var_token_grezzi = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            f, text="mostra token grezzi", variable=self._var_token_grezzi,
            command=self._mostra_token_grezzi,
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(4, 0))

        self._label_token = ttk.Label(f, text="", foreground="#555555", wraplength=820, justify="left")
        self._label_token.grid(row=3, column=0, columnspan=2, sticky="w")

        f.columnconfigure(1, weight=1)

    def _pulisci_risposta(self) -> None:
        self._label_risposta_modello.configure(text="—", foreground="black")
        self._label_risposta_oro.configure(text="—")
        self._label_token.configure(text="")
        self._ultimo_esito = None

    def _chiedi_al_modello(self) -> None:
        selezione = self._lista_domande.curselection()
        if not selezione:
            messagebox.showerror("Errore", "Scegli una domanda dalla lista.")
            return
        domanda = self._domande[selezione[0]]
        ctx = self._config["dataset"]["ctx"]
        esito = ponte.chiedi_al_modello(
            self._modello, self._vocab, self._storia_gen, domanda, ctx, self._device,
        )
        self._ultimo_esito = esito
        colore = COLORE_CATEGORIE[esito.categoria]
        self._label_risposta_modello.configure(
            text=f"{esito.testo_risposta_modello}   ({esito.categoria})", foreground=colore,
        )
        self._label_risposta_oro.configure(text=domanda.testo_risposta_oro)
        self._mostra_token_grezzi()

    def _mostra_token_grezzi(self) -> None:
        if not self._var_token_grezzi.get() or self._ultimo_esito is None:
            self._label_token.configure(text="")
            return
        e = self._ultimo_esito
        testo = (
            f"domanda:  {' '.join(e.token_domanda)}\n"
            f"modello:  {' '.join(e.token_risposta_modello)}\n"
            f"oro:      {' '.join(e.token_risposta_oro)}"
        )
        self._label_token.configure(text=testo)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--permetti-seed-esame", action="store_true",
        help="permette di generare storie con seed d'esame (>= 1.000.000): "
             "usarlo consapevolmente, contamina l'intuizione su storie che devono restare cieche",
    )
    args = ap.parse_args(argv)

    app = App(permetti_seed_esame=args.permetti_seed_esame)
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
