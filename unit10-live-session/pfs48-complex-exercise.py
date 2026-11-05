# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "altair==6.2.2",
#     "marimo",
#     "numpy==2.5.4",
#     "polars==1.42.1",
#     "py3dmol==2.5.5",
#     "pyarrow==25.0.0",
#     "tmtools==0.3.0",
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
    import numpy as np
    import polars as pl
    import py3Dmol
    from tmtools import tm_align

    return np, pl, py3Dmol


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Vaccines teach the body to make antibodies against a germ's protein, called an **antigen**. If we can predict where a known, effective antibody binds the antigen, we can choose which part of the antigen to put in a vaccine.

    Here, ColabFold predicted how the antibody 32F3 binds a malaria protein, Pfs48/45. A lab structure of the same pair exists. Is the prediction right?

    Pfs48/45 has three folded sections, called domains. We use the last one, D3, because that is where the antibody binds.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup

    The functions below read in protein structure data and return coordinates for all the residues (`ca_coords`) and adjust the residue positions for one protein (`move_pdb`) so that it can be compared to a counterpart in 3D space.
    """)
    return


@app.cell
def _(np):
    import urllib.request

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

    return move_pdb, urllib


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In the following cell we retrieve the lab measured protein structure for an antibody that binds Pfs48/45 and load a predicted structure.
    """)
    return


@app.cell
def _(urllib):
    # Pfs48/45 domain 3 (the part 32F3 binds), residues 291-428
    d3_sequence = "EKKVIHGCNFSSNVSSKHTFTDSLDISLVDDSAHISCNVHLSEPKYNHLVGLNCPGDIIPDCFFQVYQPESEELEPSNIVYLDSQINIGDIEYYEDAEGDDKIKLFGIVGSIPKTTSFTCICKKDKKSAYMTVTIDSA"

    # lab structure 7ZWI: chain A = Pfs48/45 D3, B = 32F3 heavy chain, C = 32F3 light chain
    lab_pdb = (
        urllib.request.urlopen(
            "https://www.ebi.ac.uk/pdbe/entry-files/download/pdb7zwi.ent",
            timeout=60,
        )
        .read()
        .decode()
    )

    # ColabFold (AlphaFold2-multimer) prediction of D3 + 32F3, templates off
    colabfold_pdb = (
        urllib.request.urlopen(
            "https://raw.githubusercontent.com/UNC-DATA-791/live-session-notebooks/main/unit10-live-session/pfs48_d3_32f3_colabfold_model_3.pdb",
            timeout=60,
        )
        .read()
        .decode()
    )
    len(d3_sequence)
    return colabfold_pdb, lab_pdb


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Score the antigen

    How close is the predicted Pfs48/45 D3 (chain `A` in `colabfold_pdb`) to the lab D3 (chain `A` in `lab_pdb`)? Get the coords for each with `ca_coords` using the PDB text from the previous cell. Pass the coords to the `tm_align` function. We can pass in dummy sequences for the third and fourth inputs to `tm_align`.

    /// details | Hint

    ```python
    pred_d3 = ca_coords(colabfold_pdb, "A")
    lab_d3 = ca_coords(lab_pdb, "A")
    d3_fit = tm_align(pred_d3, lab_d3, "A" * len(pred_d3), "A" * len(lab_d3))
    ```

    `d3_fit.tm_norm_chain2` is the TM-score scaled by the lab chain. `d3_fit.rmsd` is the RMSD in Å.

    ///
    """)
    return


@app.cell
def _():
    # TODO: get the CA coordinates of chain "A" from both structures.
    #   Then align them with `tm_align(coords_1, coords_2, seq_1, seq_2)`.
    #   The sequences can be strings of "A" with the same length as the coordinates.
    pred_d3 = ...
    lab_d3 = ...
    d3_fit = ...
    return (d3_fit,)


@app.cell
def _(d3_fit, mo):
    mo.stop(d3_fit is ..., mo.md("Fill in the TODO above."))
    mo.md(
        f"TM-score: **{d3_fit.tm_norm_chain2:.2f}**. RMSD: **{d3_fit.rmsd:.1f} Å**."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Score the antibody

    Do the same for the antibody. Compare only the **variable** part, where it binds: the first 120 residues of the heavy chain `B` and the first 108 residues of the light chain `C`.

    /// details | Hint

    ```python
    pred_heavy = ca_coords(colabfold_pdb, "B")[:120]
    lab_heavy = ca_coords(lab_pdb, "B")[:120]
    heavy_fit = tm_align(pred_heavy, lab_heavy, "A" * len(pred_heavy), "A" * len(lab_heavy))
    ```

    Use `"C"` and `[:108]` for the light chain. Name the variables `pred_light`, `lab_light` and `light_fit`.

    ///
    """)
    return


@app.cell
def _():
    # TODO: align the variable part of the heavy chain "B" (first 120 residues)
    #   and of the light chain "C" (first 108 residues), like you did for the antigen.
    pred_heavy = ...
    lab_heavy = ...
    heavy_fit = ...

    pred_light = ...
    lab_light = ...
    light_fit = ...
    return heavy_fit, light_fit


@app.cell
def _(d3_fit, heavy_fit, light_fit, mo, pl):
    mo.stop(
        d3_fit is ... or heavy_fit is ... or light_fit is ...,
        mo.md("Fill in the TODOs above."),
    )
    pl.DataFrame(
        {
            "part": [
                "Pfs48/45 D3",
                "32F3 heavy, variable part",
                "32F3 light, variable part",
            ],
            "TM-score": [
                round(f.tm_norm_chain2, 2)
                for f in (d3_fit, heavy_fit, light_fit)
            ],
            "RMSD (Å)": [
                round(f.rmsd, 1) for f in (d3_fit, heavy_fit, light_fit)
            ],
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Where is the antibody?

    Each part looks right on its own. Does the antibody sit in the right place on the antigen?

    Move the predicted antibody with the antigen alignment (`d3_fit.u` and `d3_fit.t`). Then measure how far it is from the lab antibody.

    /// details | Hint

    Reuse `pred_heavy`, `lab_heavy`, `pred_light` and `lab_light` from Task 2.

    ```python
    moved = pred_heavy @ d3_fit.u.T + d3_fit.t
    heavy_shift = np.sqrt(((moved - lab_heavy) ** 2).sum(axis=1).mean())
    ```

    Do the same for the light chain.

    ///
    """)
    return


@app.cell
def _():
    # TODO: reuse the coordinates from Task 2 (`pred_heavy`, `lab_heavy`, `pred_light`, `lab_light`).
    #   1. move the predicted coordinates: coords @ d3_fit.u.T + d3_fit.t
    #   2. RMSD between moved and lab: np.sqrt(((moved - lab) ** 2).sum(axis=1).mean())
    heavy_shift = ...
    light_shift = ...
    return heavy_shift, light_shift


@app.cell
def _(d3_fit, heavy_shift, light_shift, mo, pl):
    mo.stop(
        d3_fit is ... or heavy_shift is ... or light_shift is ...,
        mo.md("Fill in the TODOs above."),
    )
    pl.DataFrame(
        {
            "part": ["32F3 heavy, variable part", "32F3 light, variable part"],
            "distance from lab position (Å)": [
                round(float(heavy_shift), 1),
                round(float(light_shift), 1),
            ],
        }
    )
    return


@app.cell
def _(colabfold_pdb, d3_fit, lab_pdb, mo, move_pdb, py3Dmol):
    mo.stop(d3_fit is ..., mo.md("Fill in the TODOs above."))
    _view = py3Dmol.view(width=600, height=450)
    _view.addModel(lab_pdb, "pdb")
    _view.setStyle(
        {"model": 0}, {}
    )  # hide everything first, then show chains A-C
    _view.setStyle(
        {"model": 0, "chain": "A"}, {"cartoon": {"color": "#6b7280"}}
    )
    _view.setStyle(
        {"model": 0, "chain": ["B", "C"]}, {"cartoon": {"color": "#c9ced6"}}
    )
    _view.addModel(move_pdb(colabfold_pdb, d3_fit.u, d3_fit.t), "pdb")
    _view.setStyle(
        {"model": 1, "chain": "A"}, {"cartoon": {"color": "#2f6fdb"}}
    )
    _view.setStyle(
        {"model": 1, "chain": ["B", "C"]}, {"cartoon": {"color": "#d9552b"}}
    )
    _view.zoomTo()
    mo.vstack(
        [
            mo.md(
                "Lab structure: dark gray (antigen), light gray (antibody). Prediction: blue (antigen), orange (antibody). Lined up on the antigen."
            ),
            mo.iframe(_view._make_html(), height=470),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Is the complex right?

    Every part scored high. The antibody sits far from where it belongs.

    - Which ColabFold score would have warned you? Look at `i_ptm` in `pfs48_d3_32f3_colabfold_log.txt`.
    - Why did pLDDT stay high?
    - Would you use this model to design a vaccine? Why or why not?
    """)
    return


if __name__ == "__main__":
    app.run()
