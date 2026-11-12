# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "altair==6.2.2",
#     "marimo",
#     "numpy==2.5.4",
#     "polars==1.42.1",
#     "py3dmol==2.5.5",
#     "pyarrow==25.0.0",
#     "torch==2.11.0",
#     "tmtools==0.3.0",
#     "transformers==5.14.1",
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import altair as alt
    import polars as pl

    return alt, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Outline for this notebook

    ### What have we learned so far?
    - Protein language models write sequences (Unit 8, Unit 9)
    - ESMFold turns a sequence into a structure, and pLDDT and TM-score tell us how far to trust it (Unit 10)

    ### Today: inverse folding
    - The idea: keep the shape, change the letters
    - A simple use case
    - Where inverse folding sits in the timeline of protein generative models
    - How ProteinMPNN works (with an interactive look at its neighbor graph)
    - Run ProteinMPNN on an antibody chain
    - Check the designs with ESMFold

    ### What's next?
    - Exercise: ProteinMPNN versus EvoDiff
    - Project notebook: ProteinMPNN on a target backbone, ESMFold, and interface metrics
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Inverse folding

    **Structure prediction** goes from sequence to structure. ESMFold does this.

    **Inverse folding** goes the other way: you give the model a structure, and it gives back sequences that should fold into that structure.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### A simple use case

    You have a protein that works, but it is hard to make in the lab. Maybe it clumps, falls apart when heated, or yields very little when you grow it in cells.

    1. Take its structure (from the PDB or a prediction).
    2. Ask an inverse folding model for new sequences with the same shape.
    3. Order a handful of the new sequences and test them in the lab.

    The shape is what dictates the protein function, new sequences that fold into the same structure should keep the function.

    Below we try this on the antibody chain from the Unit 10 exercise, and we check that the designs fold back into the same shape.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A timeline of protein generative models

    Inverse folding models and protein language models appeared around the same time. ProteinMPNN and ProGen2 were released in the same month, but they work from different starting points:

    - **Sequence models** learn from sequences alone.
    - **Inverse folding models** see a structure and write a sequence for it.
    - **Structure and backbone models** predict or make the shape itself.

    Dates are the first public (preprint) release, to the nearest month.
    """)
    return


@app.cell(hide_code=True)
def _(alt, pl):
    import datetime as dt

    timeline = pl.DataFrame(
        {
            "model": [
                "Structured Transformer",
                "ESM-IF1",
                "ProteinMPNN",
                "ProGen",
                "ProGen2",
                "EvoDiff",
                "AlphaFold2",
                "RFdiffusion",
                "RFdiffusion2",
            ],
            "date": [
                dt.date(2019, 12, 1),
                dt.date(2022, 4, 1),
                dt.date(2022, 6, 1),
                dt.date(2020, 3, 1),
                dt.date(2022, 6, 1),
                dt.date(2023, 9, 1),
                dt.date(2021, 7, 1),
                dt.date(2022, 12, 1),
                dt.date(2025, 4, 1),
            ],
            "kind": [
                "Inverse folding",
                "Inverse folding",
                "Inverse folding",
                "Sequence model",
                "Sequence model",
                "Sequence model",
                "Structure / backbone",
                "Structure / backbone",
                "Structure / backbone",
            ],
            # ESM-IF1 and ProteinMPNN are two months apart, so label them on opposite sides
            "label_above": [
                True,
                True,
                False,
                True,
                True,
                True,
                True,
                True,
                True,
            ],
        }
    )

    _base = alt.Chart(timeline).encode(
        x=alt.X(
            "date:T",
            title=None,
            scale=alt.Scale(
                domain=[dt.date(2019, 1, 1), dt.date(2025, 10, 1)]
            ),
        ),
        y=alt.Y(
            "kind:N",
            title=None,
            sort=["Sequence model", "Inverse folding", "Structure / backbone"],
        ),
    )
    _points = _base.mark_circle(size=140).encode(
        color=alt.Color("kind:N", legend=None),
        tooltip=["model", "date:T", "kind"],
    )
    _labels_above = (
        _base.transform_filter(alt.datum.label_above)
        .mark_text(dy=-16, fontSize=12)
        .encode(text="model")
    )
    _labels_below = (
        _base.transform_filter(~alt.datum.label_above)
        .mark_text(dy=18, fontSize=12)
        .encode(text="model")
    )
    (_points + _labels_above + _labels_below).properties(
        width=760,
        height=260,
        title="Protein generative models, by first preprint",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(
            "RFdiffusion makes a backbone, ProteinMPNN writes a sequence for it, "
            "and a structure predictor checks the result. That is the binder design "
            "pipeline in this unit's readings."
        ),
        kind="info",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How ProteinMPNN works

    Think of a puzzle. The backbone is the empty grid. The sequence is the letters you fill in. The shape tells you which letters fit where.

    ### 1. Turn the structure into a graph

    - **Input.** The 3D coordinates of the backbone atoms (N, Cα, C, O). No sequence.
    - **Nodes.** Each residue is a node.
    - **Edges.** Each residue is linked to its **k nearest neighbors in space** (k = 48 for the model we use), measured between Cα atoms. Neighbors in space matter more than neighbors along the chain, because a residue's choice of amino acid depends on what touches it.
    - **Edge features.** For each linked pair, the distances between their backbone atoms, plus how far apart they are along the chain.

    ### 2. Read the graph once (encoder)

    Residues pass messages along the edges. After a few rounds, each residue has a summary of its surroundings. This step runs **one time**.

    ### 3. Write the sequence one residue at a time (decoder)

    Repeat until every position is filled:

    1. Pick the next position from a **random order**.
    2. The network combines that residue's summary with the amino acids already chosen.
    3. It gives a probability for each of the 20 amino acids.
    4. **Sample** one amino acid from those probabilities.

    Only this loop repeats. The graph stays fixed.

    Early picks have only the shape to go on. Later picks also see the earlier choices, which keeps the whole sequence consistent.

    ### Why every run gives a different design

    The order is random, and each amino acid is sampled. Many sequences fit the same shape, and each run finds a different one.

    ### Terms to know

    - **Temperature.** Low temperature picks the most likely amino acid each time. High temperature adds variety.
    - **Score.** The average negative log probability of the chosen residues. Lower is better. It tells you how well the sequence fits the shape, according to the model.
    - **Sequence recovery.** If the structure came from a real protein, the share of positions where the designed residue equals the native one.

      | | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
      | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
      | Native | A | C | D | E | F | G | H | I | K | L |
      | Design | A | C | D | **S** | F | G | **Q** | I | **R** | L |

      The design matches at 7 of 10 positions, so recovery is 0.70.

    Try the neighbor graph below.
    """)
    return


@app.cell
def _():
    import numpy as np
    import py3Dmol
    from tmtools import tm_align

    def ca_coords(pdb_text, chain="A"):
        """CA coordinates of one chain, as an (n, 3) array."""
        return np.array(
            [
                [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                for line in pdb_text.splitlines()
                if line.startswith("ATOM")
                and line[12:16].strip() == "CA"
                and line[21] == chain
            ]
        )

    def move_pdb(pdb_text, rotation, translation):
        """Apply a rotation and shift to every atom of a PDB string."""
        lines = []
        for line in pdb_text.splitlines():
            if line.startswith(("ATOM", "HETATM")):
                xyz = np.array(
                    [
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54]),
                    ]
                )
                x, y, z = rotation @ xyz + translation
                line = f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}"
            lines.append(line)
        return "\n".join(lines)

    return ca_coords, move_pdb, np, py3Dmol, tm_align


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The target backbone

    This is chain B of the antibody 32F3 (heavy chain, 220 residues) from the Unit 10 complex. We give ProteinMPNN only the backbone coordinates.
    """)
    return


@app.cell
def _(mo):
    ab_path = mo.notebook_dir() / "antibody_chain_b.pdb"
    ab_pdb = ab_path.read_text()
    # Residue numbers as they appear in the PDB file, in order
    ab_resi = [
        int(line[22:26])
        for line in ab_pdb.splitlines()
        if line.startswith("ATOM") and line[12:16].strip() == "CA"
    ]
    return ab_path, ab_pdb, ab_resi


@app.cell(hide_code=True)
def _(mo):
    residue = mo.ui.slider(
        1, 220, value=110, step=1, label="Residue", show_value=True
    )
    k = mo.ui.slider(
        4, 64, value=48, step=4, label="Neighbors (k)", show_value=True
    )
    mo.hstack([residue, k], justify="start")
    return k, residue


@app.cell(hide_code=True)
def _(ab_pdb, ab_resi, ca_coords, k, mo, np, py3Dmol, residue):
    _ca = ca_coords(ab_pdb, "B")
    _i = residue.value - 1
    _dist = np.linalg.norm(_ca - _ca[_i], axis=1)
    _nbrs = [int(j) for j in np.argsort(_dist) if j != _i][: k.value]
    _along_chain = sorted(abs(j - _i) for j in _nbrs)

    _view = py3Dmol.view(width=500, height=400)
    _view.addModel(ab_pdb, "pdb")
    _view.setStyle({"cartoon": {"color": "#c9ced4"}})
    _view.addStyle(
        {"resi": [ab_resi[j] for j in _nbrs]},
        {"cartoon": {"color": "#2f6fdb"}, "stick": {"color": "#2f6fdb"}},
    )
    _view.addStyle(
        {"resi": [ab_resi[_i]]},
        {"cartoon": {"color": "#d62728"}, "stick": {"color": "#d62728"}},
    )
    _view.zoomTo({"resi": [ab_resi[j] for j in [_i] + _nbrs]})
    _far = sum(d > 10 for d in _along_chain)
    mo.vstack(
        [
            mo.md(
                f"**Residue {residue.value} (red) and its {k.value} nearest neighbors (blue).** "
                f"The farthest neighbor is **{_dist[_nbrs[-1]]:.1f} Å** away. "
                f"**{_far} of {k.value}** neighbors are more than 10 residues away along the chain. "
                "They touch in space, but they are far apart in the sequence."
            ),
            mo.iframe(_view._make_html(), height=420),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Move the residue slider to a loop on the surface, then to a residue in the middle of the folded domain. A buried residue has many neighbors that are far away in the sequence. That is what the model uses to decide which amino acid fits.

    Lower k to see how little context the model would have. Raise it to see more.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup: ProteinMPNN and ESMFold

    The cells below get the ProteinMPNN code, and define one function to run it and one to fold a sequence with ESMFold.
    """)
    return


@app.cell
def _():
    import subprocess
    from pathlib import Path

    mpnn_dir = Path.home() / ".cache" / "ProteinMPNN"
    if not mpnn_dir.exists():
        subprocess.run(
            [
                "git",
                "clone",
                "-q",
                "https://github.com/dauparas/ProteinMPNN.git",
                str(mpnn_dir),
            ],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "checkout",
                "-q",
                "8907e6671bfbfc92303b5f79c4b5e6ce47cdef57",
            ],
            check=True,
            cwd=mpnn_dir,
        )
    return Path, mpnn_dir, subprocess


@app.cell
def _(Path, mpnn_dir, pl, subprocess):
    def run_proteinmpnn(pdb_path, n=24, temperature=0.2, seed=37):
        """Design n sequences for the backbone in pdb_path. Row 0 is the native sequence."""
        pdb_path = Path(pdb_path).resolve()
        out_dir = Path.home() / ".cache" / "mpnn_out"
        subprocess.run(
            [
                "python",
                "protein_mpnn_run.py",
                "--pdb_path",
                str(pdb_path),
                "--out_folder",
                str(out_dir),
                "--num_seq_per_target",
                str(n),
                "--sampling_temp",
                str(temperature),
                "--seed",
                str(seed),
                "--batch_size",
                "8",
            ],
            check=True,
            capture_output=True,
            text=True,
            cwd=mpnn_dir,
        )
        header, rows = None, []
        fasta = (out_dir / "seqs" / f"{pdb_path.stem}.fa").read_text()
        for line in fasta.splitlines():
            if line.startswith(">"):
                header = line[1:]
            elif line:
                rows.append((header, line))
        return pl.DataFrame(
            {
                "name": ["native"]
                + [f"design_{i:02d}" for i in range(1, len(rows))],
                "sequence": [s for _, s in rows],
                "mpnn_score": [
                    float(h.split("score=")[1].split(",")[0]) for h, _ in rows
                ],
                "seq_recovery": [1.0]
                + [float(h.split("seq_recovery=")[1]) for h, _ in rows[1:]],
            }
        )

    return (run_proteinmpnn,)


@app.cell
def _():
    from functools import cache

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    @cache
    def load_esmfold():
        from transformers import AutoTokenizer, EsmForProteinFolding

        tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        model = (
            EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
            .eval()
            .to(device)
        )
        return tokenizer, model

    def fold(sequence):
        """Fold one sequence with ESMFold. Returns (pdb_text, per-residue pLDDT on a 0-100 scale)."""
        tokenizer, model = load_esmfold()
        inputs = tokenizer(
            [sequence], return_tensors="pt", add_special_tokens=False
        ).to(device)
        with torch.no_grad():
            output = model(**inputs)
        return model.output_to_pdb(output)[0], output.plddt[
            0, :, 1
        ].cpu().numpy() * 100

    device
    return (fold,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Run ProteinMPNN

    Pick a sampling temperature and press the button. At low temperature the designs stay close to each other. At high temperature they spread out.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    temperature = mo.ui.dropdown(
        options={"0.1": 0.1, "0.2": 0.2, "0.5": 0.5, "1.0": 1.0},
        value="0.2",
        label="Sampling temperature",
    )
    run = mo.ui.run_button(label="Design 24 sequences")
    mo.hstack([temperature, run], justify="start")
    return run, temperature


@app.cell
def _(ab_path, mo, run, run_proteinmpnn, temperature):
    mo.stop(not run.value, mo.md("Press the button to design sequences."))
    designs = run_proteinmpnn(ab_path, n=24, temperature=temperature.value)
    designs.drop("sequence")
    return (designs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The first row is the native sequence. The rest are designs. Each design differs from the native chain, which is what we wanted: new letters on the same backbone.
    """)
    return


@app.cell(hide_code=True)
def _(alt, designs, mo):
    _d = designs.filter(designs["name"] != "native")
    _chart = (
        alt.Chart(_d)
        .mark_circle(size=80)
        .encode(
            x=alt.X(
                "seq_recovery:Q",
                title="Sequence recovery",
                scale=alt.Scale(zero=False),
            ),
            y=alt.Y(
                "mpnn_score:Q",
                title="ProteinMPNN score (lower is better)",
                scale=alt.Scale(zero=False),
            ),
            tooltip=["name", "seq_recovery", "mpnn_score"],
        )
        .properties(
            width=420,
            height=280,
            title="Designs: how close to native, and how well they fit",
        )
    )
    mo.vstack(
        [
            _chart,
            mo.md(
                f"Mean sequence recovery: **{_d['seq_recovery'].mean():.0%}**. "
                f"On average, **{1 - _d['seq_recovery'].mean():.0%}** of residues changed."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Which positions does ProteinMPNN keep?

    For each position in the chain, the share of designs that kept the native residue.
    """)
    return


@app.cell(hide_code=True)
def _(alt, designs, np, pl):
    _native = designs["sequence"][0]
    _others = designs["sequence"][1:].to_list()
    kept = pl.DataFrame(
        {
            "position": list(range(1, len(_native) + 1)),
            "kept": [
                float(np.mean([s[i] == _native[i] for s in _others]))
                for i in range(len(_native))
            ],
        }
    )
    alt.Chart(kept).mark_bar().encode(
        x=alt.X("position:Q", title="Residue position"),
        y=alt.Y("kept:Q", title="Share of designs that kept it"),
    ).properties(width=640, height=200)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check the designs with ESMFold

    A design is only useful if it folds into the shape we asked for. This is called **self-consistency**:

    1. Fold each designed sequence with ESMFold.
    2. Compare the predicted structure to the original backbone with the TM-score.
    3. Check pLDDT.

    Folding 25 sequences takes a while. To keep the session fast, the results for the default settings are saved in `fold_cache.csv`. Any sequence not in the cache is folded live.
    """)
    return


@app.cell
def _(ab_pdb, ca_coords, designs, fold, mo, pl, tm_align):
    _cache = pl.read_csv(mo.notebook_dir() / "fold_cache.csv").unique(
        "sequence"
    )
    _lookup = {r["sequence"]: r for r in _cache.iter_rows(named=True)}
    _ref_ca = ca_coords(ab_pdb, "B")
    _ref_seq = designs["sequence"][0]

    _rows = []
    for _seq in designs["sequence"]:
        if _seq in _lookup:
            _rows.append(
                {
                    "sequence": _seq,
                    "plddt": _lookup[_seq]["plddt"],
                    "tm_score": _lookup[_seq]["tm_score"],
                }
            )
        else:
            _pdb, _plddt = fold(_seq)
            _tm = tm_align(ca_coords(_pdb, "A"), _ref_ca, _seq, _ref_seq)
            _rows.append(
                {
                    "sequence": _seq,
                    "plddt": float(_plddt.mean()),
                    "tm_score": float(_tm.tm_norm_chain2),
                }
            )

    checked = designs.join(pl.DataFrame(_rows), on="sequence", how="left")
    checked.drop("sequence")
    return (checked,)


@app.cell(hide_code=True)
def _(alt, checked, mo):
    _d = checked.filter(checked["name"] != "native")
    _native = checked.filter(checked["name"] == "native")
    _points = (
        alt.Chart(_d)
        .mark_circle(size=80)
        .encode(
            x=alt.X(
                "plddt:Q", title="Mean pLDDT", scale=alt.Scale(zero=False)
            ),
            y=alt.Y(
                "tm_score:Q",
                title="TM-score to original backbone",
                scale=alt.Scale(zero=False),
            ),
            color=alt.Color("seq_recovery:Q", title="Sequence recovery"),
            tooltip=["name", "plddt", "tm_score", "seq_recovery"],
        )
    )
    _native_pt = (
        alt.Chart(_native)
        .mark_point(shape="diamond", size=200, color="black", filled=True)
        .encode(
            x="plddt:Q", y="tm_score:Q", tooltip=["name", "plddt", "tm_score"]
        )
    )
    mo.vstack(
        [
            (_points + _native_pt).properties(
                width=440,
                height=300,
                title="Self-consistency (black diamond = native sequence)",
            ),
            mo.md(
                f"Designs: mean TM-score **{_d['tm_score'].mean():.2f}**, mean pLDDT **{_d['plddt'].mean():.1f}**. "
                f"The native sequence: TM-score **{_native['tm_score'][0]:.2f}**, pLDDT **{_native['plddt'][0]:.1f}**."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Notice that the designs score **higher** than the native sequence. ESMFold finds the shape from the designed sequences more easily than from the real one.

    That is partly good news: the new sequences keep the shape. It is also a warning. ProteinMPNN and ESMFold both learned from natural proteins, and designed sequences are often "cleaner" than natural ones. A confident prediction tells you the sequence is plausible. It does not tell you the protein works.

    The lab is still the final test, which is the same lesson as the Bennett et al. paper from this unit's readings.
    """)
    return


@app.cell(hide_code=True)
def _(checked, mo):
    pick = mo.ui.dropdown(
        options=checked["name"].to_list()[1:],
        value="design_01",
        label="Design to fold and overlay",
    )
    pick
    return (pick,)


@app.cell(hide_code=True)
def _(ab_pdb, ca_coords, checked, fold, mo, move_pdb, pick, py3Dmol, tm_align):
    _seq = checked.filter(checked["name"] == pick.value)["sequence"][0]
    _native_seq = checked["sequence"][0]
    _pdb, _ = fold(_seq)
    _tm = tm_align(
        ca_coords(_pdb, "A"), ca_coords(ab_pdb, "B"), _seq, _native_seq
    )

    _view = py3Dmol.view(width=500, height=400)
    _view.addModel(ab_pdb, "pdb")
    _view.setStyle({"model": 0}, {"cartoon": {"color": "#9aa3ad"}})
    _view.addModel(move_pdb(_pdb, _tm.u, _tm.t), "pdb")
    _view.setStyle({"model": 1}, {"cartoon": {"color": "#2f6fdb"}})
    _view.zoomTo()
    mo.vstack(
        [
            mo.md(
                f"**{pick.value} folded by ESMFold, on the original backbone.** "
                f"TM-score: **{_tm.tm_norm_chain2:.2f}**. RMSD: **{_tm.rmsd:.2f} Å**. "
                "Gray = original backbone, blue = predicted from the new sequence."
            ),
            mo.iframe(_view._make_html(), height=420),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    - Inverse folding: structure in, sequences out. Keep the shape, change the letters.
    - ProteinMPNN reads the backbone as a graph of nearest neighbors in space, and writes a sequence one residue at a time.
    - Temperature sets how much the designs vary.
    - Check designs by folding them again (self-consistency) and comparing to the backbone with TM-score and pLDDT.
    - A good computational score is a filter, not proof of function.

    ## Exercise

    How different are ProteinMPNN sequences from natural proteins, and from sequences made by a model that never sees a structure (EvoDiff)? You will embed the sequences with ProtT5 and compare them on a UMAP, as in Figure 3 of the EvoDiff paper.
    """)
    return


if __name__ == "__main__":
    app.run()
